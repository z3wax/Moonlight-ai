"""``LLM`` facade (SPEC §1, §8, §13.6): config -> model + tokenizer -> train / chat / benchmark / save.

```python
import moonlight as ml
llm = ml.LLM(ml.config("ml-20m-chat"))   # random weights on cuda (if available) else cpu
llm.train()                              # Trainer.fit(): tokenizer, data, all phases, export
llm.chat("Salom!")                       # -> str
llm.chat("2+2?", tools=True)             # built-in tools (calculator, current_time, unit_convert)
llm.plan("REST API loyihasini rejalashtir")          # PLAN FORMAT answer
llm.build_project("string utils module + tests", "out/proj")   # -> ProjectResult
llm.benchmark("cc-002")                  # -> Report
llm.save("out/ml-20m-chat.safetensors"); llm2 = ml.LLM.from_pretrained("out/ml-20m-chat.safetensors")
```

Tokenizer resolution: the ``tokenizer`` argument (``MoonlightTokenizer`` or path), else
``config.tokenizer.path`` when that file exists, else None until :meth:`LLM.train` (the trainer
loads / trains one in ``Trainer.prepare_data``). Chatting without a tokenizer raises
``RuntimeError("train first or load a checkpoint ...")`` - except for offline configs
(``data.offline: true``, used by tests / debug) whose model has ``vocab_size >= 288``: those fall back
to the untrained byte-level tokenizer so a random model still produces (garbage) text. The fallback
is never used for training.

This module imports torch lazily (inside methods), so ``moonlight.LLM`` can be referenced without
torch installed.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from moonlight.chat import (
    PLAN_SYSTEM_PROMPT,
    ChatSession,
    ProjectResult,
    _TextStreamer,
    build_project,
    resolve_generation,
)
from moonlight.config import Config, TrainConfig
from moonlight.tokenizer import BOS_ID, END_ID, EOS_ID, MIN_VOCAB_SIZE, NUM_SPECIAL_TOKENS, PAD_ID
from moonlight.utils import get_device, get_logger

__all__ = ["LLM", "TrainResult", "train_config", "resolve_dtype"]

log = get_logger("moonlight.llm")

PathLike = Union[str, "os.PathLike[str]"]

_DTYPE_NAMES = {
    "fp32": "float32", "float32": "float32", "float": "float32", "32": "float32",
    "bf16": "bfloat16", "bfloat16": "bfloat16",
    "fp16": "float16", "float16": "float16", "half": "float16", "16": "float16",
}


class TrainResult(dict):
    """Result of :meth:`LLM.train` (dict-like; keys also readable as attributes).

    Keys (from ``Trainer.fit``): ``name``, ``phases`` (per-phase dicts), ``steps``, ``tokens``,
    ``elapsed`` (s), ``final_val_loss``, ``output_dir``, ``checkpoint``, ``model_path``,
    ``tokenizer_path``.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __repr__(self) -> str:
        return f"TrainResult({dict.__repr__(self)})"


def resolve_dtype(dtype: Any) -> Any:
    """``None``/``"auto"`` -> None (keep float32 weights); ``"bf16"``/``"fp16"``/``"fp32"`` (and
    aliases) or a ``torch.dtype`` -> ``torch.dtype``."""
    if dtype is None or (isinstance(dtype, str) and dtype.strip().lower() in ("", "auto")):
        return None
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    name = _DTYPE_NAMES.get(str(dtype).strip().lower().replace("torch.", ""))
    if name is None:
        raise ValueError(f"unsupported dtype {dtype!r}; use fp32, bf16 or fp16")
    return getattr(torch, name)


def _as_config(config: Any, overrides: Any = None) -> Config:
    """Config / registry name / YAML path / mapping -> validated ``Config`` (a copy)."""
    from moonlight.registry import load

    if isinstance(config, Mapping):
        cfg = Config.from_dict(config)
        if overrides:
            cfg.apply_overrides(overrides)
        return cfg.validate()
    if isinstance(config, (Config, str, os.PathLike)):
        return load(config, overrides=overrides)
    raise TypeError(f"config must be a Config, a registry name / YAML path or a mapping, got {type(config).__name__}")


_TRAIN_FIELDS = frozenset(f for f in TrainConfig.__dataclass_fields__)
_TOP_FIELDS = frozenset(f for f in Config.__dataclass_fields__)


def _override_key(key: str) -> str:
    """Keyword-argument override name -> dotted config key.

    ``train__batch_size`` -> ``train.batch_size``; bare ``TrainConfig`` fields (``batch_size``,
    ``seq_len``, ``output_dir`` ...) -> ``train.<key>``; ``offline`` -> ``data.offline``.
    """
    if "." in key:
        return key
    if "__" in key:
        return key.replace("__", ".")
    if key == "offline":
        return "data.offline"
    if key in _TRAIN_FIELDS and key not in _TOP_FIELDS:
        return f"train.{key}"
    return key


def _cast_model(model: Any, dtype: Any) -> Any:
    """``model.to(dtype)`` keeping float32 MoE ``router_bias`` buffers in float32."""
    import torch

    model.to(dtype=dtype)
    for module in model.modules():
        buf = getattr(module, "router_bias", None)
        if isinstance(buf, torch.Tensor) and buf.dtype != torch.float32:
            module.router_bias = buf.float()
    return model


class LLM:
    """Model + tokenizer + config facade (SPEC §8).

    Args:
        config: ``Config``, registry name (``"ml-20m-chat"``), YAML path or mapping.
        tokenizer: ``MoonlightTokenizer``, path to a ``tokenizer.json``, or None (see module
            docstring for the resolution order).
        device: ``"cuda"``, ``"cpu"``, ``"cuda:1"`` ...; default: CUDA if available (else MPS / CPU,
            env ``MOONLIGHT_DEVICE`` overrides) via :func:`moonlight.utils.get_device`.
        dtype: weight dtype (``"bf16"``, ``"fp16"``, ``"fp32"``, torch dtype); default float32
            weights (CUDA inference then runs under bf16/fp16 autocast).
        build: build the (randomly initialised) model now; ``False`` defers it to the first use.

    Attributes: ``config``, ``model`` (``MoonlightForCausalLM`` or None), ``tokenizer``,
    ``device`` (str), ``train_result`` (last :class:`TrainResult` or None).
    """

    def __init__(self, config: Any, tokenizer: Any = None, device: Optional[str] = None, dtype: Any = None,
                 build: bool = True) -> None:
        self.config: Config = _as_config(config)
        self.device: str = get_device(device)
        self.dtype = dtype
        self.tokenizer = self._load_tokenizer(tokenizer)
        self.model: Any = None
        self.train_result: Optional[TrainResult] = None
        self._fallback_tokenizer: Any = None
        self._vision: Any = None
        if build:
            self.build()

    # ------------------------------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------------------------------
    def _load_tokenizer(self, tokenizer: Any) -> Any:
        from moonlight.tokenizer import MoonlightTokenizer

        if tokenizer is None:
            path = self.config.tokenizer.path
            if path:
                p = Path(os.path.expandvars(str(path))).expanduser()
                if p.is_file():
                    tokenizer = MoonlightTokenizer.from_file(p)
            if tokenizer is None:
                return None
        elif isinstance(tokenizer, (str, os.PathLike)):
            tokenizer = MoonlightTokenizer.from_file(tokenizer)
        vocab = int(self.config.model.vocab_size)
        if int(tokenizer.vocab_size) > vocab:
            raise ValueError(f"tokenizer vocab ({tokenizer.vocab_size}) exceeds model.vocab_size ({vocab})")
        return tokenizer

    def build(self) -> Any:
        """Build ``MoonlightForCausalLM(config.model, config.multimodal)`` on ``self.device`` (eval
        mode, random init) unless already built; returns the model."""
        if self.model is not None:
            return self.model
        import torch

        from moonlight.model.model import MoonlightForCausalLM

        device = torch.device(self.device)
        try:
            with device:  # allocate directly on the target device (torch >= 2.0)
                model = MoonlightForCausalLM(self.config.model, self.config.multimodal)
        except (TypeError, AttributeError, RuntimeError):
            model = MoonlightForCausalLM(self.config.model, self.config.multimodal).to(device)
        dt = resolve_dtype(self.dtype)
        if dt is not None:
            _cast_model(model, dt)
        model.eval()
        self.model = model
        return model

    def require_model(self) -> Any:
        """The model (built on demand)."""
        return self.model if self.model is not None else self.build()

    def require_tokenizer(self) -> Any:
        """The tokenizer for chat / generation.

        Raises ``RuntimeError("train first or load a checkpoint ...")`` when there is none, except
        for offline configs with ``model.vocab_size >= 288`` which use an untrained byte-level
        tokenizer (garbage text from a random model, but no crash).
        """
        if self.tokenizer is not None:
            return self.tokenizer
        if self._fallback_tokenizer is not None:
            return self._fallback_tokenizer
        vocab = int(self.config.model.vocab_size)
        if bool(self.config.data.offline):
            if vocab >= MIN_VOCAB_SIZE:
                from moonlight.tokenizer import MoonlightTokenizer

                log.warning("%s: no trained tokenizer yet; using the byte-level tokenizer (offline/debug mode)",
                            self.config.name)
                self._fallback_tokenizer = MoonlightTokenizer.byte_level()
                return self._fallback_tokenizer
            raise RuntimeError(f"train first or load a checkpoint: {self.config.name!r} has no tokenizer and its "
                               f"vocab_size ({vocab}) is too small for the byte-level fallback "
                               f"(needs >= {MIN_VOCAB_SIZE})")
        raise RuntimeError(f"train first or load a checkpoint: {self.config.name!r} has no tokenizer "
                           f"(tokenizer.path is unset or missing). Run llm.train() or use "
                           f"LLM.from_pretrained(path).")

    # ------------------------------------------------------------------------------------------
    # info
    # ------------------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.config.name

    def num_params(self, trainable_only: bool = False) -> int:
        """Parameter count of the built model (else the exact estimate from the config)."""
        if self.model is not None:
            return int(self.model.num_params(trainable_only=trainable_only))
        from moonlight.params import estimate_params

        return int(estimate_params(self.config).total)

    def __repr__(self) -> str:
        from moonlight.utils import human_int

        tok = "none" if self.tokenizer is None else f"vocab {self.tokenizer.vocab_size}"
        built = "built" if self.model is not None else "not built"
        return (f"LLM({self.config.name!r}, params={human_int(self.num_params())}, device={self.device!r}, "
                f"model {built}, tokenizer {tok})")

    def autocast(self) -> Any:
        """Inference autocast context: bf16 (fp16 if unsupported) on CUDA for float32 weights,
        otherwise a no-op."""
        model = self.require_model()
        import torch

        device = model.device
        if device.type == "cuda" and model.dtype == torch.float32:
            dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast(device_type="cuda", dtype=dt)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------------------------------
    def generate_ids(self, prompt_ids: Any, max_new_tokens: Optional[int] = None,
                     temperature: Optional[float] = None, top_k: Optional[int] = None,
                     top_p: Optional[float] = None, repetition_penalty: Optional[float] = None,
                     no_repeat_ngram: Optional[int] = None, n: Optional[int] = None, seed: Optional[int] = None,
                     stop_ids: Iterable[int] = (END_ID, EOS_ID), banned_ids: Iterable[int] = (),
                     image_features: Any = None, stream: Any = None) -> Any:
        """Low-level sampling around :func:`moonlight.model.generation.generate`.

        ``prompt_ids``: list[int] or (T,) / (1, T) tensor. Unset sampling arguments default to
        ``config.chat`` (``n`` defaults to ``best_of``; forced to 1 when greedy or streaming).
        ``max_new_tokens`` is clipped so prompt + reply fit ``model.max_seq_len``.
        ``image_features``: (B_img, n_image_tokens, vision_dim) for the prompt's ``<|image|>``
        tokens. ``stream``: token-id callback. Runs under :meth:`autocast`. Returns a
        ``GenerationResult`` (``tokens`` = best sample without the stop token).
        """
        from moonlight.model.generation import generate

        model = self.require_model()
        c = self.config.chat
        prompt = prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else list(prompt_ids)
        if prompt and isinstance(prompt[0], list):  # (1, T)
            prompt = prompt[0]
        max_len = int(self.config.model.max_seq_len)
        max_new = int(c.max_new_tokens if max_new_tokens is None else max_new_tokens)
        max_new = max(1, min(max_new, max_len - len(prompt)))
        temp = float(c.temperature if temperature is None else temperature)
        n_samples = int(c.best_of if n is None else n)
        if temp <= 0 or stream is not None:
            n_samples = 1
        with self.autocast():
            return generate(
                model, prompt, max_new_tokens=max_new, temperature=temp,
                top_k=int(c.top_k if top_k is None else top_k),
                top_p=float(c.top_p if top_p is None else top_p),
                repetition_penalty=float(c.repetition_penalty if repetition_penalty is None else repetition_penalty),
                no_repeat_ngram=int(c.no_repeat_ngram if no_repeat_ngram is None else no_repeat_ngram),
                stop_ids=tuple(int(s) for s in stop_ids), n=max(1, n_samples), image_features=image_features,
                banned_ids=tuple(int(b) for b in banned_ids), stream=stream, seed=seed)

    def generate(self, prompt: str, **gen: Any) -> str:
        """Plain text continuation of ``prompt`` (no chat template; ``<|bos|>`` prepended).

        ``**gen``: :data:`moonlight.chat.GEN_KEYS` (``stream`` = callable receiving text deltas).
        Stops at ``<|eos|>`` / ``<|end|>``. Returns the generated text only.
        """
        tok = self.require_tokenizer()
        g = resolve_generation(self.config.chat, gen)
        ids = tok.encode(prompt, add_bos=True)
        max_len = int(self.config.model.max_seq_len)
        budget = max(8, max_len - min(int(g["max_new_tokens"]), max(1, max_len // 2)))
        if len(ids) > budget:
            ids = [BOS_ID] + ids[-(budget - 1):]
        model_vocab = int(self.config.model.vocab_size)
        banned = {PAD_ID, *range(15, NUM_SPECIAL_TOKENS), *range(max(tok.vocab_size, NUM_SPECIAL_TOKENS), model_vocab)}
        banned |= {int(b) for b in (g.get("banned_ids") or ())}
        streamer = _TextStreamer(tok, g["stream"], show_thinking=True) if g.get("stream") else None
        res = self.generate_ids(ids, max_new_tokens=g["max_new_tokens"], temperature=g["temperature"],
                                top_k=g["top_k"], top_p=g["top_p"], repetition_penalty=g["repetition_penalty"],
                                no_repeat_ngram=g["no_repeat_ngram"], n=g["n"], seed=g["seed"],
                                stop_ids=(EOS_ID, END_ID), banned_ids=sorted(b for b in banned if b < model_vocab),
                                stream=streamer)
        return tok.decode(res.tokens, skip_special=True)

    # ------------------------------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------------------------------
    def session(self, system: Optional[str] = None, tools: Any = None, **kw: Any) -> ChatSession:
        """A new :class:`~moonlight.chat.ChatSession` bound to this model."""
        return ChatSession(self, system=system, tools=tools, **kw)

    def chat(self, message: str, history: Optional[Sequence[Mapping[str, Any]]] = None,
             system: Optional[str] = None, tools: Any = None, images: Any = None, **gen: Any) -> str:
        """One chat reply (SPEC §8).

        Args:
            message: user text.
            history: previous SPEC §5 messages (not modified; use :class:`ChatSession` for state).
            system: system prompt (default ``config.chat.system_prompt``).
            tools: None, True (built-ins), a ``ToolRegistry`` or a list of tools / functions /
                schema dicts / built-in names; enables the tool loop.
            images: image(s) for this message (multimodal models).
            **gen: generation overrides (``max_new_tokens``, ``temperature``, ``top_k``, ``top_p``,
                ``repetition_penalty``, ``no_repeat_ngram``, ``best_of``, ``seed``, ``stream``,
                ``show_thinking``, ``max_tool_rounds``, ``allow_code_exec``).
        """
        session_kw = {k: gen.pop(k) for k in ("max_tool_rounds", "allow_code_exec") if k in gen}
        sess = ChatSession(self, system=system, tools=tools, history=history, **session_kw, **gen)
        return sess.send(message, images=images)

    def plan(self, task: str, **gen: Any) -> str:
        """Answer ``task`` in the Moonlight PLAN FORMAT (``PLAN_SYSTEM_PROMPT``): short analysis,
        numbered Plan, Verification checklist."""
        return self.chat(task, system=PLAN_SYSTEM_PROMPT, **gen)

    def build_project(self, task: str, workdir: PathLike, max_steps: int = 24, **gen: Any) -> ProjectResult:
        """Agentic project building in ``workdir`` (see :func:`moonlight.chat.build_project`).

        Extra keywords ``allow_exec``, ``timeout``, ``max_nudges`` are forwarded; the rest are
        generation settings. Returns ``ProjectResult(workdir, files, tests_passed, steps,
        transcript, ...)``.
        """
        return build_project(self, task, workdir, max_steps=max_steps, **gen)

    # ------------------------------------------------------------------------------------------
    # images
    # ------------------------------------------------------------------------------------------
    def encode_images(self, imgs: Any) -> Any:
        """Images (PIL / path / URL, one or a list) -> features (B, n_image_tokens, vision_dim)
        float32 via the frozen :class:`~moonlight.model.vision.VisionEncoder` (loaded lazily on
        ``self.device``; needs ``transformers``)."""
        mm = self.config.multimodal
        if not bool(mm.enabled):
            raise ValueError(f"model {self.config.name!r} is not multimodal (multimodal.enabled is false)")
        if self._vision is None:
            from moonlight.model.vision import VisionEncoder

            self._vision = VisionEncoder(mm.vision_encoder, n_tokens=int(mm.n_image_tokens), device=self.device)
        items = list(imgs) if isinstance(imgs, (list, tuple)) else [imgs]
        feats = self._vision.encode(items)
        if int(feats.shape[-1]) != int(mm.vision_dim):
            raise ValueError(f"vision encoder {self._vision.loaded_name or mm.vision_encoder!r} produces "
                             f"{feats.shape[-1]}-d features but multimodal.vision_dim is {mm.vision_dim}")
        return feats.float()

    # ------------------------------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------------------------------
    def train(self, overrides: Any = None, *, data: Optional[Mapping[str, Any]] = None,
              callbacks: Optional[Sequence[Any]] = None, **kw: Any) -> TrainResult:
        """Run ``config.train.phases`` with :class:`moonlight.train.trainer.Trainer`.

        Overrides (applied to ``self.config`` before training): a mapping / ``"k=v"`` list in
        ``overrides`` and/or keywords (``batch_size=16`` -> ``train.batch_size``,
        ``train__seq_len=256`` -> ``train.seq_len``, ``offline=True`` -> ``data.offline``).
        Architecture overrides (``model.*`` / ``multimodal.*``) are rejected once the model exists.
        ``data`` / ``callbacks`` are passed to the Trainer. Afterwards ``self.tokenizer`` holds the
        trained tokenizer, the model is in eval mode, and the final model was exported to
        ``{output_dir}/model.safetensors``. Returns a :class:`TrainResult`.
        """
        items = {_override_key(k): v for k, v in kw.items()}
        if overrides or items:
            new_cfg = self.config.copy()
            if overrides:
                new_cfg.apply_overrides(overrides)
            if items:
                new_cfg.apply_overrides(items)
            new_cfg.validate()
            arch_changed = (new_cfg.model.to_dict() != self.config.model.to_dict()
                            or new_cfg.multimodal.to_dict() != self.config.multimodal.to_dict())
            if arch_changed and self.model is not None:
                raise ValueError("model.* / multimodal.* overrides change the architecture of an already built "
                                 "model; pass them when creating the LLM: "
                                 "LLM(ml.config(name, overrides={...}))")
            self.config = new_cfg
        from moonlight.train.trainer import Trainer

        model = self.require_model()
        trainer = Trainer(model, self.config, tokenizer=self.tokenizer, data=data, callbacks=callbacks)
        result = trainer.fit()
        self.tokenizer = trainer.tokenizer
        self._fallback_tokenizer = None
        self.model = trainer.raw_model
        self.model.eval()
        self.device = str(self.model.device)
        self.train_result = TrainResult(result)
        return self.train_result

    # ------------------------------------------------------------------------------------------
    # benchmarks
    # ------------------------------------------------------------------------------------------
    def benchmark(self, name: Optional[str] = "default", **kw: Any) -> Any:
        """Run a benchmark suite on this model: ``"default"`` (``config.benchmark``) or a suite
        name (``"cc-001"`` ... ``"cc-010"``). ``**kw`` -> ``Suite.run`` (``limit``, ``out``,
        ``offline``). Returns the ``Report``."""
        from moonlight.benchmarks.suites import type as suite_type

        suite_name = self.config.benchmark if name in (None, "", "default") else name
        return suite_type(suite_name).run(self, **kw)

    # ------------------------------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------------------------------
    def save(self, path: PathLike, dtype: Optional[str] = None) -> Path:
        """Export weights + config + tokenizer into one ``.safetensors`` file (SPEC §7
        ``export_model``). A path without the ``.safetensors`` suffix is a directory receiving
        ``model.safetensors`` (+ ``tokenizer.json`` and ``config.yaml``). ``dtype``: ``"fp32"``
        (default) or ``"bf16"``. Returns the model file path."""
        from moonlight.train.checkpoint import export_model
        from moonlight.utils import is_main_process

        model = self.require_model()
        p = Path(path).expanduser()
        if p.suffix.lower() == ".safetensors":
            target = p
        else:
            target = p / "model.safetensors"
            if is_main_process():
                p.mkdir(parents=True, exist_ok=True)
                self.config.to_yaml(p / "config.yaml")
                if self.tokenizer is not None:
                    self.tokenizer.save(p / "tokenizer.json")
        export_model(target, model, self.config, self.tokenizer, dtype=dtype)
        return target

    @staticmethod
    def _resolve_pretrained(path: PathLike) -> Path:
        """File / directory / registry name -> the ``.safetensors`` file to load."""
        p = Path(os.fspath(path)).expanduser()
        if p.is_file():
            return p
        if p.is_dir():
            for name in ("model.safetensors", "checkpoint.safetensors"):
                if (p / name).is_file():
                    return p / name
            found = sorted(p.glob("*.safetensors"))
            if len(found) == 1:
                return found[0]
            raise FileNotFoundError(f"{p}: no model.safetensors / checkpoint.safetensors"
                                    + (f" (several candidates: {[f.name for f in found]})" if found else ""))
        if p.suffix.lower() != ".safetensors":  # a config name: look for its trained export
            from moonlight.registry import load

            try:
                cfg = load(os.fspath(path))
            except (FileNotFoundError, KeyError, ValueError):
                cfg = None
            if cfg is not None:
                cand = Path(cfg.train.output_dir).expanduser() / "model.safetensors"
                if cand.is_file():
                    return cand
                raise FileNotFoundError(f"no trained model for config {cfg.name!r} at {cand}; train it first "
                                        f"(`moonlight train {cfg.name}`)")
        raise FileNotFoundError(f"model file not found: {p}")

    @classmethod
    def from_pretrained(cls, path: PathLike, device: Optional[str] = None, dtype: Any = None,
                        tokenizer: Any = None) -> "LLM":
        """Load an exported model (``format: moonlight-v1``) or a training checkpoint.

        ``path``: a ``.safetensors`` file, a directory holding ``model.safetensors`` /
        ``checkpoint.safetensors``, or a config name whose ``{output_dir}/model.safetensors``
        exists. The config comes from the file metadata (else a sibling ``config.yaml``), the
        tokenizer from the metadata (else a sibling ``tokenizer.json``) unless given.
        """
        from moonlight.config import load_yaml
        from moonlight.tokenizer import MoonlightTokenizer
        from moonlight.train.checkpoint import CKPT_FORMAT, read_metadata

        file = cls._resolve_pretrained(path)
        meta = read_metadata(file)
        fmt = meta.get("format", "")
        cfg_text = meta.get("config") or ""
        if not cfg_text.strip() and (file.parent / "config.yaml").is_file():
            cfg_text = (file.parent / "config.yaml").read_text(encoding="utf-8")
        if not cfg_text.strip():
            raise ValueError(f"{file}: no config in the metadata and no config.yaml next to it")
        raw = load_yaml(cfg_text if "\n" in cfg_text else cfg_text + "\n")
        cfg = Config.from_dict(raw).validate()

        if tokenizer is None:
            tok_text = meta.get("tokenizer") or ""
            if tok_text.strip():
                tokenizer = MoonlightTokenizer.from_str(tok_text)
            elif (file.parent / "tokenizer.json").is_file():
                tokenizer = MoonlightTokenizer.from_file(file.parent / "tokenizer.json")

        llm = cls(cfg, tokenizer=tokenizer, device=device, dtype=dtype, build=False)
        from safetensors.torch import load_file

        tensors = load_file(str(file), device="cpu")
        if fmt == CKPT_FORMAT or any(k.startswith("model/") for k in tensors):
            state = {k[len("model/"):]: v for k, v in tensors.items() if k.startswith("model/")}
        else:
            state = tensors
        model = llm.build()
        expected = set(model.state_dict().keys())
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        if missing or unexpected:
            raise RuntimeError(f"{file} does not match its config {cfg.name!r}: missing={missing[:8]} "
                               f"unexpected={unexpected[:8]}")
        model.load_state_dict(state, strict=True)
        model.eval()
        return llm


def train_config(name_or_cfg: Any, overrides: Any = None, output: Optional[PathLike] = None, **train_kw: Any) -> LLM:
    """One-liner ``ml.train(...)``: config -> :class:`LLM` -> ``train()`` -> save.

    Args:
        name_or_cfg: registry name, YAML path, mapping or ``Config``.
        overrides: dotted overrides applied before building (``{"train.batch_size": 16}`` or
            ``["data.offline=true"]``); architecture overrides are allowed here.
        output: optional extra export path (``.safetensors`` file or directory); the trainer
            always exports ``{output_dir}/model.safetensors``.
        **train_kw: forwarded to :meth:`LLM.train` (keyword overrides, ``data``, ``callbacks``),
            plus ``device`` / ``dtype`` for the LLM.

    Returns the trained :class:`LLM` (its :class:`TrainResult` is ``llm.train_result``).
    """
    device = train_kw.pop("device", None)
    dtype = train_kw.pop("dtype", None)
    llm = LLM(_as_config(name_or_cfg, overrides), device=device, dtype=dtype)
    llm.train(**train_kw)
    if output is not None:
        path = llm.save(output)
        if llm.train_result is not None:
            llm.train_result["saved_to"] = str(path)
    return llm
