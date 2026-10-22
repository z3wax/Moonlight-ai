"""The Moonlight trainer (SPEC §7 + §13.5): phases, AMP, compile, accumulation, EMA, eval, budgets.

```python
trainer = Trainer(model, config, tokenizer=None)   # tokenizer None -> trained / loaded in prepare_data
result = trainer.fit()                             # runs config.train.phases, resumes automatically
```

Phase dicts (``config.train.phases``, free-form) are dispatched by ``type`` (default: ``name``;
``"sft-code"`` / ``"sft2"`` resolve to ``sft``). Supported types:

=========  =================================================================================
pretrain   LM batches from the token bins; sequence-length curriculum; Muon momentum warmup
           0.85 -> 0.95 over 300 steps; ``sft_mix`` (SFT micro-batch fraction, always) and
           ``anneal_sft_mix`` (fraction during the LR decay window) when SFT data exists.
anneal     like pretrain with the LR decaying linearly over the WHOLE phase and
           ``anneal_sft_mix`` applied throughout.
long       like pretrain at ``seq_len`` (phase key, default ``model.max_seq_len``) with the
           micro-batch shrunk to keep tokens/step; ``lr_scale`` default 0.1.
align      image-caption batches; only ``vision_proj`` trainable (default ``trainable``);
           requires ``multimodal.enabled`` and image data (else skipped).
sft        packed conversation batches + ``replay`` fraction of LM batches + ``image_mix``
           fraction of image batches.
dpo        preference pairs (``sources``, ``synthetic_pairs``), ``beta`` 0.1, ``loss`` dpo|ipo,
           frozen reference copy; ``lr_scale`` default 0.1.
grpo       verifier-rewarded RL (``tasks``, ``group_size`` 8, ``prompts_per_step`` 4,
           ``max_new_tokens`` 256, ``temperature`` 1.0, ``kl_coef`` 0.02, ``clip`` 0.2);
           ``lr_scale`` default 0.05.
=========  =================================================================================

Common phase keys: budgets ``tokens`` / ``minutes`` / ``steps`` / ``epochs`` (any subset; progress
= max of the ratios; default 1 epoch), ``lr_scale``, ``trainable`` (name prefixes; others
frozen), ``seq_len``, ``batch_size`` (micro-batch override), ``warmup_steps`` / ``decay_frac`` /
``min_lr_ratio`` / ``schedule`` (schedule overrides), ``ema`` (false disables EMA).

Precision: ``auto`` -> bf16 autocast when the GPU supports it, else fp16 autocast +
``GradScaler`` (one ``unscale_`` per optimizer, clipping over all parameters, ``step`` per
optimizer, one ``update``); CPU / MPS -> fp32 without autocast. Gradient accumulation
``accum = batch_tokens // (batch_size * seq_len * world)`` with DDP ``no_sync`` / FSDP2
``set_requires_gradient_sync(False)`` on non-final micro-steps. ``torch.compile`` (CUDA, torch
>= 2.1, ``train.compile``) wraps the model per phase; all curriculum / mixed-batch shapes are
warmed up BEFORE the phase timer starts; any compile failure falls back to eager. DPO / GRPO
phases (variable shapes) always run eager.

EMA (``train.ema`` > 0, not for dpo/grpo): fp32 shadow of the trainable parameters updated with
``torch._foreach_lerp_`` after every optimizer step (decay warmup ``min(ema, (1+t)/(10+t))``). At
phase end the EMA and raw weights are evaluated and the better one is kept.

Checkpoints: ``{output_dir}/checkpoint.safetensors`` every ``checkpoint_every_min`` minutes and at
phase end; ``fit()`` resumes from it (finished phases are skipped, an unfinished phase continues
with its step / tokens / elapsed time and optimizer + EMA state). Metrics are appended to
``{output_dir}/metrics.jsonl`` (rank 0); wandb is used when installed and ``WANDB_PROJECT`` is set.
The final model is exported to ``{output_dir}/model.safetensors`` (weights + config + tokenizer).
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from moonlight.train.checkpoint import (
    checkpoint_path,
    export_model,
    load_checkpoint,
    read_checkpoint_state,
    save_checkpoint,
)
from moonlight.train.distributed import (
    DistInfo,
    all_reduce_max,
    all_reduce_mean,
    barrier,
    broadcast_object,
    fsdp_version,
    grad_sync,
    init_distributed,
    unwrap_model,
    wrap_model,
)
from moonlight.train.optim import Muon, build_optimizers, clean_param_name, set_trainable
from moonlight.train.schedule import (
    curriculum_lengths,
    curriculum_seq_len,
    in_decay,
    lr_multiplier,
    muon_momentum,
    phase_budgets,
    phase_progress,
)
from moonlight.utils import get_logger, human_int, human_time, seed_everything

__all__ = [
    "Trainer",
    "ModelEMA",
    "PHASE_TYPES",
    "phase_type",
    "resolve_precision",
    "compile_supported",
    "SYNTHETIC_GENERATORS",
]

log = get_logger("moonlight.train")

PHASE_TYPES = ("pretrain", "anneal", "long", "align", "sft", "dpo", "grpo")
_LM_TYPES = ("pretrain", "anneal", "long")
_DEFAULT_LR_SCALE = {"long": 0.1, "dpo": 0.1, "grpo": 0.05}

#: ``data.synthetic`` key -> (generator function name in moonlight.data.synthetic, count when
#: the value is ``true``). ``identity`` additionally takes the persona name (``chat.name``).
SYNTHETIC_GENERATORS: Dict[str, Tuple[str, int]] = {
    "identity": ("identity_conversations", 300),
    "greetings": ("greeting_conversations", 300),
    "knowledge": ("knowledge_conversations", 400),
    "arithmetic": ("arithmetic_conversations", 1000),
    "tool_calls": ("tool_call_conversations", 500),
    "code": ("code_conversations", 300),
    "think": ("think_conversations", 300),
    "plan": ("plan_conversations", 200),
    "constraints": ("constraint_conversations", 300),
    "detail": ("detail_conversations", 200),
    "projects": ("project_conversations", 50),
}

_SAMPLE_PROMPTS = ("Salom! Sen kimsan?", "What is 12 + 7?")
_IGNORE = -100


# ================================================================================================
# small helpers
# ================================================================================================


def phase_type(phase: Mapping[str, Any]) -> str:
    """Resolve a phase dict to one of :data:`PHASE_TYPES` (``type`` key, else ``name``).

    Exact matches win; otherwise the leading word without trailing digits is tried
    (``"sft-code"`` -> ``sft``, ``"pretrain2"`` -> ``pretrain``). Unknown -> ``ValueError``.
    """
    raw = str(phase.get("type") or phase.get("name") or "").strip().lower()
    if raw in PHASE_TYPES:
        return raw
    head = re.sub(r"\d+$", "", re.split(r"[-_\s.:/]", raw)[0]) if raw else ""
    if head in PHASE_TYPES:
        return head
    raise ValueError(f"unknown training phase {phase.get('name')!r} (type {raw!r}); "
                     f"expected one of {', '.join(PHASE_TYPES)}")


def resolve_precision(precision: str, device: torch.device) -> str:
    """``auto``/``bf16``/``fp16``/``fp32`` -> the precision actually used on ``device``.

    CPU / MPS always run fp32 (no autocast). On CUDA ``auto`` picks bf16 when supported, else
    fp16; an explicit ``bf16`` on a GPU without bf16 falls back to fp16.
    """
    p = (precision or "auto").lower()
    if device.type != "cuda":
        return "fp32"
    bf16_ok = torch.cuda.is_bf16_supported()
    if p == "auto":
        return "bf16" if bf16_ok else "fp16"
    if p == "bf16" and not bf16_ok:
        log.warning("bf16 is not supported on this GPU; using fp16 + GradScaler")
        return "fp16"
    if p not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"unknown precision {precision!r}")
    return p


def _torch_version() -> Tuple[int, int]:
    m = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def compile_supported(enabled: bool, device: torch.device) -> Tuple[bool, str]:
    """Whether ``torch.compile`` should be used: ``(ok, reason)``.

    Disabled when ``train.compile`` is false, env ``MOONLIGHT_COMPILE=0``, not on CUDA, torch < 2.1,
    or on Windows.
    """
    if not enabled:
        return False, "train.compile is false"
    if os.environ.get("MOONLIGHT_COMPILE", "").strip().lower() in ("0", "false", "no", "off"):
        return False, "MOONLIGHT_COMPILE=0"
    if device.type != "cuda":
        return False, f"device {device.type} (compile only on CUDA)"
    if _torch_version() < (2, 1) or not hasattr(torch, "compile"):
        return False, f"torch {torch.__version__} < 2.1"
    if sys.platform.startswith("win"):
        return False, "Windows"
    return True, "enabled"


def _get(d: Any, key: str, default: Any = None) -> Any:
    if isinstance(d, Mapping):
        v = d.get(key, default)
    else:
        v = getattr(d, key, default)
    return default if v is None else v


def _as_long(t: Any, device: torch.device) -> torch.Tensor:
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(t))
    t = torch.as_tensor(t)
    if t.dtype != torch.long:
        t = t.long()
    return t.to(device, non_blocking=True)


def _as_feats(t: Any, device: torch.device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(t))
    return torch.as_tensor(t).to(device, non_blocking=True)


def _assets_dir() -> Path:
    import moonlight.assets as assets

    return Path(assets.__file__).resolve().parent


def _conversation_text(conv: Sequence[Mapping[str, Any]]) -> str:
    """Plain text of a conversation (message contents + tool-call JSON) for tokenizer training."""
    parts = []
    for m in conv:
        content = m.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
        for call in m.get("tool_calls") or []:
            try:
                parts.append(json.dumps(call, ensure_ascii=False))
            except (TypeError, ValueError):
                pass
    return "\n".join(parts)


def _split_val(result: Any) -> Tuple[Any, Any, Any, Any]:
    """Normalise ``moonlight.data.sft.split_val`` output to ``(tr_ids, tr_mask, va_ids, va_mask)``."""
    if isinstance(result, Mapping):
        return (result["train_ids"], result["train_mask"], result["val_ids"], result["val_mask"])
    result = tuple(result)
    if len(result) == 4:
        return result  # type: ignore[return-value]
    if len(result) == 2 and all(isinstance(r, (tuple, list)) and len(r) == 2 for r in result):
        (a, b), (c, d) = result
        return a, b, c, d
    raise TypeError("unexpected split_val() result; expected (train_ids, train_mask, val_ids, val_mask)")


def _pair(item: Any) -> Optional[Tuple[list, list]]:
    """Normalise a preference item to ``(chosen_msgs, rejected_msgs)``."""
    if item is None:
        return None
    if isinstance(item, Mapping):
        c, r = item.get("chosen"), item.get("rejected")
    elif isinstance(item, (tuple, list)) and len(item) == 2:
        c, r = item
    else:
        return None
    if not c or not r:
        return None
    return list(c), list(r)


# ================================================================================================
# EMA
# ================================================================================================


class ModelEMA:
    """fp32 exponential moving average of (trainable) parameters.

    ``update(model, step)``: ``shadow = lerp(shadow, param, 1 - d)`` with
    ``d = min(decay, (1 + step) / (10 + step))`` (fast warmup so short phases are not dominated by
    the initial weights), batched with ``torch._foreach_lerp_`` when possible. Names are clean
    FQNs of ``unwrap_model(model)``. Works with FSDP2 ``DTensor`` parameters (per-tensor ops).
    """

    def __init__(self, model: torch.nn.Module, decay: float, only_trainable: bool = True) -> None:
        self.decay = float(decay)
        self.names: List[str] = []
        self.shadow: Dict[str, torch.Tensor] = {}
        for n, p in unwrap_model(model).named_parameters():
            if only_trainable and not p.requires_grad:
                continue
            n = clean_param_name(n)
            self.names.append(n)
            self.shadow[n] = p.detach().float().clone()
        self._dtensor = any(type(t).__name__ == "DTensor" for t in self.shadow.values())

    def _params(self, model: torch.nn.Module) -> List[torch.Tensor]:
        named = {clean_param_name(n): p for n, p in unwrap_model(model).named_parameters()}
        return [named[n] for n in self.names]

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        params = [p.detach().float() for p in self._params(model)]
        shadows = [self.shadow[n] for n in self.names]
        if not shadows:
            return
        if not self._dtensor and hasattr(torch, "_foreach_lerp_"):
            torch._foreach_lerp_(shadows, params, 1.0 - d)
        else:
            for s, p in zip(shadows, params):
                s.lerp_(p, 1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        """Overwrite the model parameters with the EMA weights."""
        for p, n in zip(self._params(model), self.names):
            p.copy_(self.shadow[n].to(p.dtype))

    @contextlib.contextmanager
    def swap(self, model: torch.nn.Module) -> Iterator[None]:
        """Temporarily load the EMA weights into ``model`` (restored on exit)."""
        params = self._params(model)
        backup = [p.detach().clone() for p in params]
        try:
            self.copy_to(model)
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(params, backup):
                    p.copy_(b)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return dict(self.shadow)

    @torch.no_grad()
    def load_state_dict(self, sd: Mapping[str, torch.Tensor]) -> None:
        for n, v in sd.items():
            if n not in self.shadow:
                continue
            tgt = self.shadow[n]
            if type(tgt).__name__ == "DTensor" and type(v).__name__ != "DTensor":
                try:
                    from torch.distributed.tensor import distribute_tensor  # torch >= 2.4
                except ImportError:  # pragma: no cover
                    from torch.distributed._tensor import distribute_tensor
                # full checkpoint tensor -> this rank's shard (same mesh / placements as the param)
                v = distribute_tensor(v.to(dtype=tgt.dtype), tgt.device_mesh, tgt.placements)
                tgt.copy_(v)
            else:
                tgt.copy_(v.to(device=tgt.device, dtype=tgt.dtype))


# ================================================================================================
# Trainer
# ================================================================================================


class _PhaseRun:
    """Mutable per-phase state shared by the training loop helpers."""

    def __init__(self, phase: Mapping[str, Any], ptype: str, index: int) -> None:
        self.phase = phase
        self.ptype = ptype
        self.index = index
        self.name = str(phase.get("name") or ptype)
        self.step = 0
        self.tokens = 0.0
        self.elapsed = 0.0
        self.best_val: Optional[float] = None
        self.budgets: Dict[str, float] = {}
        self.steps_per_epoch: Optional[float] = None
        self.accum = 1
        self.sched: Dict[str, Any] = {}
        self.rng = random.Random(0)
        self.momentum_warmup = False
        self.last_metrics: Dict[str, float] = {}


class Trainer:
    """Runs ``config.train.phases`` on ``model`` (see the module docstring).

    Args:
        model: ``MoonlightForCausalLM`` (moved to the process device here).
        config: full ``moonlight.config.Config``.
        tokenizer: ``MoonlightTokenizer`` or None (then loaded from ``tokenizer.path`` /
            ``{output_dir}/tokenizer.json`` or trained in :meth:`prepare_data`).
        data: optional pre-built data overriding :meth:`prepare_data` products, keys:
            ``train_bin``, ``val_bin`` (paths), ``sft`` = ``(ids, mask)``, ``sft_val`` =
            ``(ids, mask)``, ``image_loader``, ``pref_{i}`` (encoded pairs for phase ``i``),
            ``grpo_{i}`` (prompt records for phase ``i``).
        dist: ``DistInfo`` (default :func:`init_distributed`).
        callbacks: callables ``cb(event, payload)`` or objects with ``on_<event>(payload)``;
            events: ``log``, ``eval``, ``checkpoint``, ``phase_start``, ``phase_end``.
    """

    def __init__(self, model: torch.nn.Module, config: Any, tokenizer: Any = None, data: Optional[Mapping[str, Any]] = None,
                 dist: Optional[DistInfo] = None, callbacks: Optional[Sequence[Any]] = None) -> None:
        self.config = config
        self.tc = config.train
        self.dist = dist or init_distributed()
        self.device = torch.device(self.dist.device)
        self.raw_model: torch.nn.Module = unwrap_model(model).to(self.device)
        self.tokenizer = tokenizer
        self.data: Dict[str, Any] = dict(data or {})
        self.callbacks = list(callbacks or [])
        self.output_dir = Path(self.tc.output_dir).expanduser()
        self.data_dir = self.output_dir / "data"
        self.precision = resolve_precision(self.tc.precision, self.device)
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.precision)
        strategy = str(self.tc.distributed.strategy or "none").lower()
        self.strategy = strategy if self.dist.world > 1 else "none"
        self.compile_ok, self.compile_reason = compile_supported(bool(self.tc.compile), self.device)
        # FSDP shards once (in place for FSDP2); DDP is (re-)wrapped per phase after freezing.
        if self.strategy == "fsdp":
            self.model: torch.nn.Module = wrap_model(self.raw_model, "fsdp", self.dist, precision=self.precision,
                                                     fsdp_shard=self.tc.distributed.fsdp_shard)
        else:
            self.model = self.raw_model
        self.fwd_model: torch.nn.Module = self.model
        self.ema: Optional[ModelEMA] = None
        self.scaler: Any = None
        self.optimizers: List[torch.optim.Optimizer] = []
        self.results: List[Dict[str, Any]] = []
        self.global_step = 0
        self.total_tokens = 0.0
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.ckpt_path = checkpoint_path(self.output_dir)
        self._wandb: Any = None
        self._wandb_tried = False
        self._run: Optional[_PhaseRun] = None
        self._image_loader: Any = None
        self._ref_model: Optional[torch.nn.Module] = None
        self._compiled = False
        self._prepared = False
        self._fit_start = time.perf_counter()

    # --------------------------------------------------------------------------------------------
    # properties / small utilities
    # --------------------------------------------------------------------------------------------
    @property
    def is_main(self) -> bool:
        return self.dist.is_main

    @property
    def seq_len(self) -> int:
        return int(min(self.tc.seq_len, self.config.model.max_seq_len))

    def _autocast(self) -> Any:
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, f"on_{event}", None)
            if fn is not None:
                fn(payload)
            elif callable(cb):
                cb(event, payload)

    def _base_accum(self, micro_batch: Optional[int] = None, seq_len: Optional[int] = None) -> int:
        mb = int(micro_batch or self.tc.batch_size)
        sl = int(seq_len or self.seq_len)
        return max(1, int(self.tc.batch_tokens) // max(1, mb * sl * self.dist.world))

    def _sync_flags(self, *values: float) -> List[float]:
        """Element-wise max over ranks (agreement on progress / checkpoint decisions)."""
        if self.dist.world <= 1:
            return list(values)
        t = torch.tensor(list(values), dtype=torch.float64, device=self.device)
        all_reduce_max(t)
        return [float(v) for v in t.tolist()]

    # --------------------------------------------------------------------------------------------
    # logging
    # --------------------------------------------------------------------------------------------
    def _wandb_run(self) -> Any:
        if self._wandb_tried:
            return self._wandb
        self._wandb_tried = True
        if not self.is_main or not os.environ.get("WANDB_PROJECT"):
            return None
        try:
            import wandb

            self._wandb = wandb.init(project=os.environ["WANDB_PROJECT"], name=self.config.name,
                                     config=self.config.to_dict(), dir=str(self.output_dir))
        except Exception as e:  # pragma: no cover - optional dependency
            log.warning("wandb disabled: %s", e)
            self._wandb = None
        return self._wandb

    def _write_metrics(self, record: Dict[str, Any]) -> None:
        if not self.is_main:
            return
        record = dict(record)
        record.setdefault("time", round(time.time(), 3))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=lambda o: str(o)) + "\n")
        run = self._wandb_run()
        if run is not None:
            try:
                nums = {f"{record.get('phase', 'train')}/{k}": v for k, v in record.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)}
                run.log(nums, step=self.global_step)
            except Exception:  # pragma: no cover
                pass

    def _mem_gb(self) -> float:
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / 2**30
        return 0.0

    def _log_line(self, msg: str) -> None:
        if self.is_main:
            print(msg, flush=True)

    # --------------------------------------------------------------------------------------------
    # data preparation
    # --------------------------------------------------------------------------------------------
    def _data_signature(self) -> str:
        c = self.config
        payload = {"data": c.data.to_dict(), "tokenizer": c.tokenizer.to_dict(), "phases": c.train.phases,
                   "seq_len": c.train.seq_len, "seed": c.train.seed, "chat": c.chat.name,
                   "mm": c.multimodal.to_dict(),
                   "tokenizer_hash": hashlib.sha256(self.tokenizer.to_str().encode("utf-8")).hexdigest()
                   if self.tokenizer is not None else None}
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _needs(self) -> Dict[str, Any]:
        """Which data products the configured phases need."""
        phases = list(self.tc.phases)
        types = [phase_type(p) for p in phases]
        mm = bool(self.config.multimodal.enabled)
        need_lm = any(t in _LM_TYPES for t in types) or any(
            t == "sft" and float(_get(p, "replay", 0.0)) > 0 for p, t in zip(phases, types))
        need_sft = any(t in ("sft", "dpo", "grpo") for t in types) or any(
            t in _LM_TYPES and (float(_get(p, "sft_mix", 0.0)) > 0 or float(_get(p, "anneal_sft_mix", 0.0)) > 0)
            for p, t in zip(phases, types))
        need_img = mm and bool(self.config.data.vision) and any(
            t == "align" or (t == "sft" and float(_get(p, "image_mix", 0.0)) > 0) for p, t in zip(phases, types))
        return {"lm": need_lm, "sft": need_sft, "image": need_img, "types": types}

    def _iter_source_texts(self, name: str, max_chars: float) -> Iterator[str]:
        """Stream cleaned-enough documents of a pretrain source (downloads are cached)."""
        from moonlight.data.adapters import to_text
        from moonlight.data.download import download, iter_rows
        from moonlight.data.sources import get_source

        spec = get_source(name)
        used = 0
        dest = self.config.data.cache_path / "raw" / name
        for url in spec.urls:
            path = download(url, dest)
            for row in iter_rows(path, spec.fmt):
                text = to_text(name, row)
                if not text:
                    continue
                yield text
                used += len(text)
                if used >= max_chars:
                    return

    def _offline_texts(self) -> List[str]:
        out = []
        d = _assets_dir()
        for fname in ("sample_uz.txt", "sample_en.txt", "sample_code.py"):
            p = d / fname
            if p.is_file():
                out.append(p.read_text(encoding="utf-8"))
        return out

    def _synthetic_conversations(self, scale: float = 1.0) -> List[list]:
        """Conversations from ``data.synthetic`` (see :data:`SYNTHETIC_GENERATORS`)."""
        from moonlight.data import synthetic

        spec = dict(self.config.data.synthetic or {})
        if spec.get("identity") and "greetings" not in spec:
            spec["greetings"] = True  # chat sanity: greetings accompany identity data
        convs: List[list] = []
        for key, value in spec.items():
            if key not in SYNTHETIC_GENERATORS:
                log.warning("data.synthetic.%s: unknown generator (ignored)", key)
                continue
            fn_name, default_n = SYNTHETIC_GENERATORS[key]
            if value is False or value is None:
                continue
            n = default_n if value is True else int(value)
            n = int(max(0, round(n * scale)))
            if n <= 0:
                continue
            fn = getattr(synthetic, fn_name, None)
            if fn is None:
                log.warning("moonlight.data.synthetic.%s not available (data.synthetic.%s ignored)", fn_name, key)
                continue
            got = fn(self.config.chat.name, n) if key == "identity" else fn(n)
            convs.extend(list(got))
        return convs

    def _tokenizer_corpus(self) -> Iterator[str]:
        """Texts for tokenizer training: pretrain sources (weighted, online) or bundled assets
        (offline / fallback), plus synthetic conversations; capped by ``tokenizer.train_chars``."""
        budget = float(self.config.tokenizer.train_chars)
        used = 0.0
        texts: List[str] = list(self._offline_texts())
        try:
            texts.extend(_conversation_text(c) for c in self._synthetic_conversations(scale=0.25))
        except Exception as e:
            log.warning("synthetic data unavailable for tokenizer training: %s", e)
        for t in texts:
            if used >= budget:
                return
            used += len(t)
            yield t
        if self.config.data.offline:
            return
        entries = [e for e in self.config.data.pretrain if _get(e, "source")]
        total_w = sum(float(_get(e, "weight", 1.0)) for e in entries) or 1.0
        remaining = budget - used
        streams = []
        for e in entries:
            share = remaining * float(_get(e, "weight", 1.0)) / total_w
            share = min(share, float(_get(e, "max_chars", share)))
            streams.append(iter(self._iter_source_texts(str(e["source"]), share)))
        # round-robin so every source is represented even if a large one is slow
        while streams and used < budget:
            alive = []
            for s in streams:
                try:
                    for _ in range(16):
                        t = next(s)
                        used += len(t)
                        yield t
                    alive.append(s)
                except StopIteration:
                    pass
                except Exception as e:  # network / format problems: drop that source
                    log.warning("tokenizer corpus: source failed (%s); continuing without it", e)
            streams = alive

    def _ensure_tokenizer(self) -> None:
        from moonlight.tokenizer import MoonlightTokenizer, train_tokenizer

        out_path = self.output_dir / "tokenizer.json"
        if self.tokenizer is None:
            cfg_path = self.config.tokenizer.path
            cfg_file = Path(cfg_path).expanduser() if cfg_path else None
            source = None
            if self.is_main:  # decide once (rank 0) so every rank takes the same branch / barrier
                source = "config" if cfg_file is not None and cfg_file.is_file() else (
                    "output" if out_path.is_file() else "train")
            source = broadcast_object(source)
            if source == "config":
                self.tokenizer = MoonlightTokenizer.from_file(cfg_file)
            elif source == "output":
                self.tokenizer = MoonlightTokenizer.from_file(out_path)
            else:
                if self.is_main:
                    t0 = time.perf_counter()
                    vocab = int(self.config.tokenizer.vocab_size)
                    tok = train_tokenizer(self._tokenizer_corpus(), vocab, path=out_path,
                                          split_digits=bool(self.config.tokenizer.split_digits),
                                          max_chars=int(self.config.tokenizer.train_chars))
                    if cfg_file is not None:
                        tok.save(cfg_file)
                    self._log_line(f"[data] tokenizer trained: vocab {tok.vocab_size} "
                                   f"({human_time(time.perf_counter() - t0)}) -> {out_path}")
                barrier()
                self.tokenizer = MoonlightTokenizer.from_file(out_path)
        if self.is_main:
            on_disk = out_path.read_text(encoding="utf-8") if out_path.is_file() else None
            if on_disk != self.tokenizer.to_str():
                self.tokenizer.save(out_path)
        if self.tokenizer.vocab_size > int(self.config.model.vocab_size):
            raise ValueError(f"tokenizer vocab ({self.tokenizer.vocab_size}) exceeds model.vocab_size "
                             f"({self.config.model.vocab_size})")

    def _build_sft(self) -> Optional[Dict[str, Any]]:
        from moonlight.data.sft import encode_conversations, pack, split_val

        convs: List[list] = []
        if not self.config.data.offline:
            from moonlight.data.adapters import to_conversation
            from moonlight.data.download import download, iter_rows
            from moonlight.data.sources import get_source

            for e in self.config.data.sft:
                name = str(_get(e, "source", ""))
                if not name:
                    continue
                limit = int(_get(e, "max_examples", 10**12))
                got = 0
                try:
                    spec = get_source(name)
                    for url in spec.urls:
                        path = download(url, self.config.data.cache_path / "raw" / name)
                        for row in iter_rows(path, spec.fmt):
                            conv = to_conversation(name, row)
                            if conv:
                                convs.append(conv)
                                got += 1
                                if got >= limit:
                                    break
                        if got >= limit:
                            break
                except Exception as ex:
                    log.warning("SFT source %s failed after %d examples: %s", name, got, ex)
                if self.is_main:
                    log.info("SFT source %s: %d conversations", name, got)
        try:
            convs.extend(self._synthetic_conversations())
        except Exception as ex:
            log.warning("synthetic SFT data failed: %s", ex)
        if not convs:
            return None
        random.Random(int(self.tc.seed)).shuffle(convs)
        examples = encode_conversations(convs, self.tokenizer, n_image_tokens=0)
        ids, mask = pack(examples, self.seq_len)
        if len(ids) == 0:
            return None
        tr_ids, tr_mask, va_ids, va_mask = _split_val(split_val(ids, mask, frac=0.02))
        return {"n_conversations": len(convs), "train": (np.asarray(tr_ids), np.asarray(tr_mask)),
                "val": (np.asarray(va_ids), np.asarray(va_mask))}

    def _build_pref(self, phase: Mapping[str, Any]) -> List[Dict[str, Any]]:
        from moonlight.data.sft import encode_preference

        pairs: List[Tuple[list, list]] = []
        if not self.config.data.offline:
            from moonlight.data import adapters
            from moonlight.data.download import download, iter_rows
            from moonlight.data.sources import get_source

            for e in phase.get("sources") or []:
                name = str(_get(e, "source", e if isinstance(e, str) else ""))
                limit = int(_get(e, "max_examples", 10**12)) if isinstance(e, Mapping) else 10**12
                got = 0
                try:
                    spec = get_source(name)
                    for url in spec.urls:
                        path = download(url, self.config.data.cache_path / "raw" / name)
                        for row in iter_rows(path, spec.fmt):
                            try:
                                item = adapters.to_preference(row)
                            except TypeError:  # (source_name, row) signature
                                item = adapters.to_preference(name, row)
                            pr = _pair(item)
                            if pr is not None:
                                pairs.append(pr)
                                got += 1
                                if got >= limit:
                                    break
                        if got >= limit:
                            break
                except Exception as ex:
                    log.warning("preference source %s failed after %d pairs: %s", name, got, ex)
        n_syn = int(_get(phase, "synthetic_pairs", 0))
        if n_syn > 0:
            from moonlight.data import synthetic

            for item in synthetic.preference_pairs_synthetic(n_syn):
                pr = _pair(item)
                if pr is not None:
                    pairs.append(pr)
        seq_len = int(_get(phase, "seq_len", self.seq_len))
        seq_len = min(seq_len, int(self.config.model.max_seq_len))
        encoded = []
        for c, r in pairs:
            try:
                enc = encode_preference(c, r, self.tokenizer, seq_len)
            except Exception:
                continue
            if enc:
                encoded.append(enc)
        random.Random(int(self.tc.seed) + 7).shuffle(encoded)
        return encoded

    def _build_grpo(self, phase: Mapping[str, Any], index: int) -> List[Dict[str, Any]]:
        from moonlight.data import synthetic

        kinds = phase.get("tasks") or ["arithmetic"]
        if isinstance(kinds, str):
            kinds = [k.strip() for k in kinds.split(",") if k.strip()]
        kinds = list(kinds)
        if self.config.data.offline:
            kinds = [k for k in kinds if k != "gsm8k"] or ["arithmetic"]
        n = int(_get(phase, "n_prompts", 1024))
        seed = int(self.tc.seed) + 101 * (index + 1)
        try:
            prompts = synthetic.grpo_prompt_set(kinds, n, seed)
        except Exception as ex:
            if "gsm8k" in kinds:
                log.warning("grpo_prompt_set with gsm8k failed (%s); retrying without it", ex)
                kinds = [k for k in kinds if k != "gsm8k"] or ["arithmetic"]
                prompts = synthetic.grpo_prompt_set(kinds, n, seed)
            else:
                raise
        return [dict(p) for p in prompts]

    def _build_images(self) -> Any:
        from moonlight.data.vision import build_image_dataset
        from moonlight.model.vision import VisionEncoder

        mm = self.config.multimodal
        enc = VisionEncoder(mm.vision_encoder, int(mm.n_image_tokens), device=self.device)
        return build_image_dataset(self.config, self.tokenizer, enc, self.data_dir / "vision")

    def _image_loader_from(self, built: Any) -> Any:
        """Find / construct an ``ImageSFTLoader`` from ``build_image_dataset``'s result."""
        if built is None:
            return None
        if hasattr(built, "next"):
            return built
        if isinstance(built, Mapping):
            for key in ("loader", "train_loader", "train"):
                if hasattr(built.get(key), "next"):
                    return built[key]
        if isinstance(built, (tuple, list)):
            for item in built:
                if hasattr(item, "next"):
                    return item
        from moonlight.data.vision import ImageSFTLoader

        return ImageSFTLoader(built, batch_size=int(self.tc.batch_size), device=str(self.device),
                              seed=int(self.tc.seed) + self.dist.rank)

    def prepare_data(self) -> Dict[str, Any]:
        """Build (rank 0) / load (all ranks) everything the phases need; idempotent.

        1. tokenizer: given -> used; else ``tokenizer.path`` or ``{output_dir}/tokenizer.json``
           if present; else trained on pretrain sources (online) / bundled assets (offline) plus
           synthetic conversations, up to ``tokenizer.train_chars`` characters.
        2. pretrain token bins via ``moonlight.data.pretrain.build_pretrain_tokens``.
        3. SFT: ``data.sft`` sources (online) + ``data.synthetic`` generators -> encode -> pack
           (``train.seq_len``) -> 2% validation split (``.npy`` under ``{output_dir}/data``).
        4. per ``dpo`` phase: preference pairs (``sources`` + ``synthetic_pairs``), encoded;
           per ``grpo`` phase: ``grpo_prompt_set(tasks, n_prompts, seed)``.
        5. images (multimodal + ``data.vision``): ``build_image_dataset`` with a ``VisionEncoder``.

        Products are cached in ``{output_dir}/data/manifest.json`` keyed by a signature of the
        data-relevant config, so resumed runs skip rebuilding. Failures of optional products
        are logged and make the corresponding phases skip.
        """
        if self._prepared:
            return self.data
        if self.is_main:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        barrier()
        self._ensure_tokenizer()
        needs = self._needs()
        sig = self._data_signature()
        manifest_path = self.data_dir / "manifest.json"
        manifest: Dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
            if manifest.get("signature") != sig:
                manifest = {}

        if self.is_main:
            t0 = time.perf_counter()
            new = dict(manifest)
            new["signature"] = sig
            if needs["lm"] and "train_bin" not in self.data and not (
                    manifest.get("train_bin") and Path(manifest["train_bin"]).is_file()):
                from moonlight.data.pretrain import build_pretrain_tokens

                train_bin, val_bin, stats = build_pretrain_tokens(self.config, self.tokenizer, self.data_dir / "pretrain")
                new["train_bin"], new["val_bin"] = str(train_bin), str(val_bin)
                new["pretrain_stats"] = stats if isinstance(stats, Mapping) else {}
                self._log_line(f"[data] pretrain bins ready: {train_bin}")
            if needs["sft"] and "sft" not in self.data and not manifest.get("sft_done"):
                built = self._build_sft()
                if built is not None:
                    np.save(self.data_dir / "sft_ids.npy", built["train"][0])
                    np.save(self.data_dir / "sft_mask.npy", built["train"][1])
                    np.save(self.data_dir / "sft_val_ids.npy", built["val"][0])
                    np.save(self.data_dir / "sft_val_mask.npy", built["val"][1])
                    new["sft"] = True
                    self._log_line(f"[data] SFT: {built['n_conversations']} conversations -> "
                                   f"{len(built['train'][0])} packed train / {len(built['val'][0])} val sequences")
                else:
                    new["sft"] = False
                new["sft_done"] = True
            for i, (p, t) in enumerate(zip(self.tc.phases, needs["types"])):
                if t == "dpo" and f"pref_{i}" not in self.data and not manifest.get(f"pref_{i}"):
                    try:
                        pairs = self._build_pref(p)
                    except Exception as ex:
                        log.warning("DPO data for phase %s failed: %s", p.get("name"), ex)
                        pairs = []
                    with open(self.data_dir / f"pref_{i}.pkl", "wb") as f:
                        pickle.dump(pairs, f)
                    new[f"pref_{i}"] = len(pairs)
                    self._log_line(f"[data] DPO phase {p.get('name')}: {len(pairs)} preference pairs")
                if t == "grpo" and f"grpo_{i}" not in self.data and not manifest.get(f"grpo_{i}"):
                    try:
                        prompts = self._build_grpo(p, i)
                    except Exception as ex:
                        log.warning("GRPO prompts for phase %s failed: %s", p.get("name"), ex)
                        prompts = []
                    with open(self.data_dir / f"grpo_{i}.pkl", "wb") as f:
                        pickle.dump(prompts, f)
                    new[f"grpo_{i}"] = len(prompts)
                    self._log_line(f"[data] GRPO phase {p.get('name')}: {len(prompts)} prompts")
            if needs["image"] and "image_loader" not in self.data:
                try:
                    self._image_loader = self._image_loader_from(self._build_images())
                    new["image"] = self._image_loader is not None
                except Exception as ex:
                    log.warning("image dataset unavailable (%s); align/image batches will be skipped", ex)
                    new["image"] = False
            manifest_path.write_text(json.dumps(new, indent=1, default=str), encoding="utf-8")
            self._log_line(f"[data] prepared in {human_time(time.perf_counter() - t0)}")
        barrier()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # ---- load products on every rank ---------------------------------------------------------
        if "train_bin" not in self.data and manifest.get("train_bin"):
            self.data["train_bin"] = manifest["train_bin"]
            self.data["val_bin"] = manifest.get("val_bin")
            self.data["pretrain_stats"] = manifest.get("pretrain_stats", {})
        if "sft" not in self.data and manifest.get("sft"):
            self.data["sft"] = (np.load(self.data_dir / "sft_ids.npy"), np.load(self.data_dir / "sft_mask.npy"))
            self.data["sft_val"] = (np.load(self.data_dir / "sft_val_ids.npy"),
                                    np.load(self.data_dir / "sft_val_mask.npy"))
        for i, t in enumerate(needs["types"]):
            for kind in ("pref", "grpo"):
                key = f"{kind}_{i}"
                if t == ("dpo" if kind == "pref" else "grpo") and key not in self.data and manifest.get(key):
                    with open(self.data_dir / f"{key}.pkl", "rb") as f:
                        self.data[key] = pickle.load(f)
        if "image_loader" in self.data:
            self._image_loader = self.data["image_loader"]
        elif needs["image"] and manifest.get("image") and self._image_loader is None:
            try:  # non-main ranks: rebuild from the (cached) features written by rank 0
                self._image_loader = self._image_loader_from(self._build_images())
            except Exception as ex:
                log.warning("image loader unavailable on rank %d: %s", self.dist.rank, ex)
        self._prepared = True
        return self.data

    # --------------------------------------------------------------------------------------------
    # loaders
    # --------------------------------------------------------------------------------------------
    def _lm_loader(self, seq_len: int, batch_size: int, seed: int, path: Optional[str] = None) -> Any:
        from moonlight.data.pretrain import PretrainLoader

        return PretrainLoader(path or self.data["train_bin"], seq_len, batch_size, str(self.device), seed,
                              rank=self.dist.rank, world=self.dist.world)

    def _sft_loader(self, batch_size: int, seed: int, val: bool = False) -> Any:
        from moonlight.data.sft import SFTLoader

        ids, mask = self.data["sft_val" if val else "sft"]
        if val and len(ids) == 0:
            ids, mask = self.data["sft"]
        return SFTLoader(ids, mask, batch_size, str(self.device), seed)

    def _train_tokens(self) -> Optional[float]:
        stats = self.data.get("pretrain_stats") or {}
        for key in ("train_tokens", "n_train_tokens", "tokens_train"):
            if key in stats:
                return float(stats[key])
        path = self.data.get("train_bin")
        if not path or not Path(path).is_file():
            return None
        itemsize = 2
        meta = Path(path).with_suffix(".json")
        try:
            if meta.is_file():
                m = json.loads(meta.read_text(encoding="utf-8"))
                itemsize = int(np.dtype(m.get("dtype", "uint16")).itemsize)
            elif int(self.config.model.vocab_size) > 65535:
                itemsize = 4
        except (OSError, ValueError, TypeError):
            pass
        return Path(path).stat().st_size / itemsize

    # --------------------------------------------------------------------------------------------
    # model setup per phase
    # --------------------------------------------------------------------------------------------
    def _release_wrappers(self) -> None:
        if self.strategy != "fsdp":
            self.model = self.raw_model
        self.fwd_model = self.model
        self.optimizers = []
        gc.collect()

    def _setup_model(self, trainable: Any, exclude: Sequence[str], allow_compile: bool) -> Tuple[int, int]:
        """Freeze/unfreeze, (re-)wrap for DDP, compile. Returns (trainable, total) param counts."""
        self._release_wrappers()
        n_train, n_total = set_trainable(self.raw_model, trainable, exclude)
        self.raw_model.train()
        if self.strategy == "ddp":
            self.model = wrap_model(self.raw_model, "ddp", self.dist)
        self.fwd_model = self.model
        self._compiled = False
        if allow_compile and self.compile_ok:
            try:
                import torch._dynamo as dynamo

                dynamo.config.cache_size_limit = max(int(getattr(dynamo.config, "cache_size_limit", 8)), 32)
                self.fwd_model = torch.compile(self.model, dynamic=False)
                self._compiled = True
            except Exception as e:
                log.warning("torch.compile unavailable (%s); running eager", e)
                self.fwd_model = self.model
        return n_train, n_total

    def _make_scaler(self) -> Any:
        if self.precision != "fp16":
            return None
        if self.strategy == "fsdp" and fsdp_version(self.model) == 1:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            return ShardedGradScaler()
        try:
            return torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler()

    def _backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _zero_grad(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
        for p in self.raw_model.parameters():  # warm-up / skipped steps may leave grads on frozen params
            p.grad = None

    def _clip_and_step(self) -> Tuple[float, bool]:
        """Unscale (fp16), clip over ALL trainable parameters, step every optimizer, update scaler.

        Returns ``(grad_norm, stepped)``; a non-finite norm without a GradScaler skips the step.
        """
        if self.scaler is not None:
            for opt in self.optimizers:
                self.scaler.unscale_(opt)
        clip = float(self.tc.optimizer.grad_clip or 0.0)
        max_norm = clip if clip > 0 else float("inf")
        if self.strategy == "fsdp" and fsdp_version(self.model) == 1:
            norm = self.model.clip_grad_norm_(max_norm)
        else:
            params = [p for p in self.raw_model.parameters() if p.requires_grad and p.grad is not None]
            norm = torch.nn.utils.clip_grad_norm_(params, max_norm) if params else torch.zeros(())
        if hasattr(norm, "full_tensor"):
            norm = norm.full_tensor()
        norm_f = float(norm.item()) if torch.is_tensor(norm) else float(norm)
        stepped = True
        if self.scaler is not None:
            for opt in self.optimizers:
                self.scaler.step(opt)
            self.scaler.update()
        elif math.isfinite(norm_f):
            for opt in self.optimizers:
                opt.step()
        else:
            stepped = False
            log.warning("non-finite gradient norm (%s); skipping optimizer step", norm_f)
        self._zero_grad()
        return norm_f, stepped

    def _set_lr(self, mult: float, momentum: Optional[float]) -> None:
        for opt in self.optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * mult
                if momentum is not None and isinstance(opt, Muon):
                    g["momentum"] = momentum

    # --------------------------------------------------------------------------------------------
    # batches & losses
    # --------------------------------------------------------------------------------------------
    def _next_lm(self, loader: Any, seq_len: int, mb: int) -> Tuple[torch.Tensor, ...]:
        x, y = loader.next(seq_len=seq_len, batch_size=mb)
        return _as_long(x, self.device), _as_long(y, self.device), None

    def _next_sft(self, loader: Any) -> Tuple[torch.Tensor, ...]:
        x, y = loader.next()[:2]
        return _as_long(x, self.device), _as_long(y, self.device), None

    def _next_image(self) -> Tuple[torch.Tensor, ...]:
        x, y, feats = self._image_loader.next()[:3]
        return _as_long(x, self.device), _as_long(y, self.device), _as_feats(feats, self.device)

    def _lm_loss(self, model: Any, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x, y, feats = batch
        out = model(x, targets=y, image_features=feats)
        extra = {}
        if getattr(out, "lm_loss", None) is not None:
            extra["lm_loss"] = out.lm_loss.detach()
        return out.loss, extra

    def _accumulate(self, run: _PhaseRun, next_batch: Callable[[int], Tuple[torch.Tensor, ...]],
                    loss_fn: Callable[[Any, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, Dict[str, Any]]]
                    ) -> Tuple[torch.Tensor, float, Dict[str, float]]:
        """``run.accum`` micro-steps of forward/backward. Returns (mean loss, local tokens, extras)."""
        loss_sum = torch.zeros((), device=self.device)
        tokens = 0.0
        extras: Dict[str, Any] = {}
        for micro in range(run.accum):
            batch = next_batch(micro)
            tokens += float(batch[0].numel())
            with grad_sync(self.model, micro == run.accum - 1):
                with self._autocast():
                    loss, extra = loss_fn(self.fwd_model, batch)
                self._backward(loss / run.accum)
            loss_sum += loss.detach().float()
            for k, v in extra.items():
                extras[k] = extras.get(k, 0.0) + (v.float() if torch.is_tensor(v) else float(v)) / run.accum
        return loss_sum / run.accum, tokens, extras

    # --------------------------------------------------------------------------------------------
    # evaluation & samples
    # --------------------------------------------------------------------------------------------
    def _phase_by_name(self, name: str) -> Optional[Mapping[str, Any]]:
        for p in self.tc.phases:
            if str(p.get("name")) == str(name):
                return p
        return None

    @torch.no_grad()
    def evaluate(self, phase_name: str) -> float:
        """Mean validation loss (``eval_batches`` batches, averaged over ranks) for a phase.

        pretrain/anneal/long: held-out token bin (at the phase's sequence length); sft/dpo/grpo:
        SFT validation split (falls back to the LM bin); align: image batches. Returns ``nan``
        when no evaluation data exists. Call on every rank (collective under DDP/FSDP).
        """
        if self._run is not None and self._run.name == str(phase_name):
            ptype, phase = self._run.ptype, self._run.phase
        else:
            phase = self._phase_by_name(phase_name) or {"name": phase_name}
            try:
                ptype = phase_type(phase)
            except ValueError:
                ptype = "pretrain"
        n_batches = max(1, int(self.tc.eval_batches or 1))
        bs = int(_get(phase, "batch_size", self.tc.batch_size))
        seed = 987_654 + 17 * self.dist.rank
        model = self.model  # uncompiled (eval shapes differ; avoids recompiles)
        was_training = self.raw_model.training
        self.raw_model.eval()
        losses: List[float] = []
        try:
            batches: Optional[Callable[[], Tuple[torch.Tensor, ...]]] = None
            if ptype == "align":
                if self._image_loader is not None:
                    batches = self._next_image
            elif ptype in _LM_TYPES or "sft" not in self.data:
                if self.data.get("train_bin"):
                    seq = self._phase_seq_len(phase, ptype)
                    mb = max(1, bs * self.seq_len // seq) if ptype == "long" else bs
                    try:
                        loader = self._lm_loader(seq, mb, seed, path=self.data.get("val_bin") or None)
                        loader.next()  # probe: a tiny val bin may not fit one window
                    except Exception:
                        loader = self._lm_loader(seq, mb, seed)
                    batches = lambda: self._next_lm(loader, seq, mb)  # noqa: E731
            else:
                sl = self._sft_loader(bs, seed, val=True)
                batches = lambda: self._next_sft(sl)  # noqa: E731
            if batches is None:
                return float("nan")
            for _ in range(n_batches):
                x, y, feats = batches()
                with self._autocast():
                    out = model(x, targets=y, image_features=feats)
                lm = out.lm_loss if getattr(out, "lm_loss", None) is not None else out.loss
                losses.append(float(lm.float().item()))
        finally:
            self.raw_model.train(was_training)
        value = sum(losses) / max(1, len(losses))
        return float(all_reduce_mean(value)) if self.dist.world > 1 else value

    @torch.no_grad()
    def sample_chat(self, prompts: Sequence[str] = _SAMPLE_PROMPTS, max_new_tokens: int = 48) -> List[Tuple[str, str]]:
        """Greedy-ish chat samples for monitoring (``generate`` + ``encode_chat``).

        Rank 0 only for single-process / DDP (unwrapped module); every rank under FSDP (fixed
        length generation keeps the collectives aligned). Returns ``[(prompt, reply)]``.
        """
        if self.tokenizer is None:
            return []
        from moonlight.model.generation import generate
        from moonlight.tokenizer import END_ID, EOS_ID

        fsdp = self.strategy == "fsdp"
        if not fsdp and not self.is_main:
            return []
        model = self.model if fsdp else self.raw_model
        chat = self.config.chat
        out = []
        for i, prompt in enumerate(prompts):
            ids, _ = self.tokenizer.encode_chat([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                                system=chat.system_prompt)
            with self._autocast():
                res = generate(model, ids, max_new_tokens=int(max_new_tokens), temperature=0.0, top_k=0, top_p=1.0,
                               repetition_penalty=float(chat.repetition_penalty),
                               no_repeat_ngram=int(chat.no_repeat_ngram),
                               stop_ids=() if fsdp else (END_ID, EOS_ID), n=1, seed=1234 + i)
            toks = res.tokens
            if fsdp:
                cut = [k for k, t in enumerate(toks) if t in (END_ID, EOS_ID)]
                toks = toks[: cut[0]] if cut else toks
            out.append((prompt, self.tokenizer.decode(toks)))
        return out

    # --------------------------------------------------------------------------------------------
    # checkpoints
    # --------------------------------------------------------------------------------------------
    def _state(self, run: _PhaseRun, done: bool) -> Dict[str, Any]:
        return {
            "phase_index": run.index,
            "phase_name": run.name,
            "phase_type": run.ptype,
            "phase_done": bool(done),
            "step": run.step,
            "tokens_seen": run.tokens,
            "elapsed": run.elapsed,
            "best_val": run.best_val,
            "global_step": self.global_step,
            "total_tokens": self.total_tokens,
            "results": self.results,
            "config_name": self.config.name,
            "saved_at": time.time(),
        }

    def save(self, run: _PhaseRun, done: bool = False) -> Path:
        """Write ``{output_dir}/checkpoint.safetensors`` (all ranks must call)."""
        t0 = time.perf_counter()
        path = save_checkpoint(self.ckpt_path, self.model, self.optimizers, self._state(run, done),
                               ema=self.ema, config=self.config)
        if self.is_main:
            log.info("checkpoint saved: %s (%.1fs)", path, time.perf_counter() - t0)
        self._emit("checkpoint", {"path": str(path), "phase": run.name, "step": run.step, "done": done})
        return path

    # --------------------------------------------------------------------------------------------
    # phases
    # --------------------------------------------------------------------------------------------
    def _phase_seq_len(self, phase: Mapping[str, Any], ptype: str) -> int:
        max_len = int(self.config.model.max_seq_len)
        if ptype == "long":
            want = int(_get(phase, "seq_len", max_len))
        else:
            want = int(_get(phase, "seq_len", self.seq_len))
        if want > max_len:
            log.warning("phase %s seq_len %d > model.max_seq_len %d; clamped", phase.get("name"), want, max_len)
            want = max_len
        return max(1, want)

    def _availability(self, phase: Mapping[str, Any], ptype: str, index: int) -> Optional[str]:
        """None if the phase can run, else the reason it is skipped."""
        if ptype in _LM_TYPES:
            return None if self.data.get("train_bin") else "no pretraining data"
        if ptype == "sft":
            return None if "sft" in self.data and len(self.data["sft"][0]) > 0 else "no SFT data"
        if ptype == "align":
            if not self.config.multimodal.enabled or getattr(self.raw_model, "vision_proj", None) is None:
                return "multimodal disabled"
            return None if self._image_loader is not None else "no image data"
        if ptype == "dpo":
            return None if self.data.get(f"pref_{index}") else "no preference pairs"
        if ptype == "grpo":
            return None if self.data.get(f"grpo_{index}") else "no GRPO prompts"
        return f"unsupported phase type {ptype}"

    def _schedule_for(self, phase: Mapping[str, Any], ptype: str) -> Dict[str, Any]:
        sched = dict(self.tc.schedule.to_dict())
        sub = phase.get("schedule")
        if isinstance(sub, Mapping):
            sched.update(sub)
        elif isinstance(sub, str):
            sched["name"] = sub
        for key in ("warmup_steps", "decay_frac", "min_lr_ratio"):
            if phase.get(key) is not None:
                sched[key] = phase[key]
        if ptype == "anneal":
            sched.update(name="wsd", decay_frac=1.0)
            sched.setdefault("warmup_steps", 0)
            if phase.get("warmup_steps") is None:
                sched["warmup_steps"] = 0
        return sched

    def run_phase(self, phase: Mapping[str, Any], *, index: Optional[int] = None,
                  resume: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Run one phase (dispatch by :func:`phase_type`); returns a result dict.

        ``index`` is the phase position in ``train.phases`` (used for data keys / checkpoints);
        ``resume`` is a checkpoint state of this unfinished phase (``fit`` passes it).
        """
        if not self._prepared:
            self.prepare_data()
        ptype = phase_type(phase)
        if index is None:
            index = next((i for i, p in enumerate(self.tc.phases) if p is phase or p == phase), len(self.results))
        name = str(phase.get("name") or ptype)
        reason = self._availability(phase, ptype, index)
        if reason is not None:
            if resume:  # keep the weights of the interrupted run even though this phase is skipped now
                load_checkpoint(self.ckpt_path, self.model, map_location="cpu")
            self._log_line(f"[{name}] skipped: {reason}")
            res = {"name": name, "type": ptype, "skipped": True, "reason": reason}
            self._write_metrics({"phase": name, "event": "skipped", "reason": reason})
            return res
        run = _PhaseRun(phase, ptype, index)
        self._run = run
        try:
            return self._train_phase(run, resume)
        finally:
            self._run = None
            self._ref_model = None
            self._release_wrappers()
            self.ema = None
            self.scaler = None

    def _train_phase(self, run: _PhaseRun, resume: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        phase, ptype = run.phase, run.ptype
        tc = self.tc
        lr_scale = float(_get(phase, "lr_scale", _DEFAULT_LR_SCALE.get(ptype, 1.0)))
        trainable = phase.get("trainable")
        if ptype == "align" and trainable is None:
            trainable = "vision_proj"
        # dpo/grpo forwards have no targets: MTP heads and the vision projector receive no
        # gradient, which DDP would reject -> freeze them for these phases
        exclude = ("mtp", "vision_proj") if ptype in ("dpo", "grpo") else ()
        allow_compile = ptype not in ("dpo", "grpo")
        n_train, n_total = self._setup_model(trainable, exclude, allow_compile)
        self.optimizers = build_optimizers(self.raw_model, tc, lr_scale=lr_scale, trainable=trainable)
        self.scaler = self._make_scaler()
        use_ema = float(tc.ema or 0.0) > 0 and ptype not in ("dpo", "grpo") and phase.get("ema", True) is not False
        self.ema = ModelEMA(self.raw_model, float(tc.ema)) if use_ema else None
        seed_base = int(tc.seed) + 1000 * (run.index + 1)
        if resume:
            load_checkpoint(self.ckpt_path, self.model, self.optimizers, self.ema, map_location="cpu")
            run.step = int(resume.get("step", 0))
            run.tokens = float(resume.get("tokens_seen", 0.0))
            run.elapsed = float(resume.get("elapsed", 0.0))
            run.best_val = resume.get("best_val")
            self._log_line(f"[{run.name}] resuming at step {run.step} ({human_int(run.tokens)} tokens, "
                           f"{human_time(run.elapsed)})")
        run.rng = random.Random(seed_base + 7919 * run.step + self.dist.rank)
        loader_seed = seed_base + 104_729 * run.step + 31 * self.dist.rank
        run.sched = self._schedule_for(phase, ptype)
        run.budgets = phase_budgets(phase) or {"epochs": 1.0}
        run.momentum_warmup = ptype == "pretrain"
        bs = int(_get(phase, "batch_size", tc.batch_size))
        world = self.dist.world

        # ---- step function per phase type -------------------------------------------------------
        warm_batches: List[Tuple[torch.Tensor, ...]] = []
        step_fn: Callable[[float], Tuple[torch.Tensor, float, Dict[str, float]]]
        if ptype in _LM_TYPES or ptype in ("sft", "align"):
            run.accum = self._base_accum(bs, self.seq_len)
            final_seq = self._phase_seq_len(phase, ptype)
            lm_loader = self._lm_loader(final_seq, bs, loader_seed) if self.data.get("train_bin") else None
            sft_loader = self._sft_loader(bs, loader_seed + 1) if "sft" in self.data else None
            cur = tc.curriculum
            use_curr = ptype == "pretrain" and bool(cur.enabled)
            sft_mix = float(_get(phase, "sft_mix", 0.0))
            anneal_mix = float(_get(phase, "anneal_sft_mix", 0.0))
            replay = float(_get(phase, "replay", 0.0)) if lm_loader is not None else 0.0
            image_mix = float(_get(phase, "image_mix", 0.0)) if self._image_loader is not None else 0.0
            long_mb = max(1, bs * self.seq_len // final_seq) if ptype == "long" else bs

            def lm_shape(progress: float) -> Tuple[int, int]:
                if use_curr:
                    s = curriculum_seq_len(progress, int(cur.start_seq_len), final_seq, float(cur.until_frac))
                    return s, max(1, bs * final_seq // s)
                return final_seq, long_mb

            def next_batch(progress: float) -> Tuple[torch.Tensor, ...]:
                r = run.rng.random()
                if ptype == "align":
                    return self._next_image()
                if ptype == "sft":
                    if r < image_mix:
                        return self._next_image()
                    if r < image_mix + replay:
                        return self._next_lm(lm_loader, self.seq_len, bs)
                    return self._next_sft(sft_loader)
                mix = sft_mix
                if anneal_mix > 0 and (ptype == "anneal" or in_decay(progress, run.sched)):
                    mix = max(mix, anneal_mix)
                if sft_loader is not None and mix > 0 and r < mix:
                    return self._next_sft(sft_loader)
                s, mb = lm_shape(progress)
                return self._next_lm(lm_loader, s, mb)

            def step_fn(progress: float) -> Tuple[torch.Tensor, float, Dict[str, float]]:
                return self._accumulate(run, lambda micro: next_batch(progress), self._lm_loss)

            # steps per epoch & compile warm-up shapes
            if ptype in _LM_TYPES:
                n_tok = self._train_tokens()
                if n_tok:
                    run.steps_per_epoch = n_tok / max(1.0, float(final_seq * long_mb * run.accum * world))
                if self._compiled:
                    lens = curriculum_lengths(int(cur.start_seq_len), final_seq) if use_curr else [final_seq]
                    for s in lens:
                        warm_batches.append(self._next_lm(lm_loader, s, max(1, bs * final_seq // s)
                                                          if use_curr else long_mb))
                    if sft_loader is not None and (sft_mix > 0 or anneal_mix > 0):
                        warm_batches.append(self._next_sft(sft_loader))
            elif ptype == "sft":
                n_seq = len(self.data["sft"][0])
                run.steps_per_epoch = n_seq / max(1.0, float(bs * run.accum * world))
                if self._compiled:
                    warm_batches.append(self._next_sft(sft_loader))
                    if replay > 0:
                        warm_batches.append(self._next_lm(lm_loader, self.seq_len, bs))
                    if image_mix > 0:
                        warm_batches.append(self._next_image())
            else:  # align
                n_img = getattr(self._image_loader, "n", None) or getattr(self._image_loader, "num_examples", None)
                if n_img is None and hasattr(self._image_loader, "__len__"):
                    try:
                        n_img = len(self._image_loader)
                    except TypeError:
                        n_img = None
                if n_img:
                    run.steps_per_epoch = float(n_img) / max(1.0, float(bs * run.accum * world))
                if self._compiled:
                    warm_batches.append(self._next_image())
        elif ptype == "dpo":
            step_fn = self._dpo_setup(run, bs, loader_seed)
        else:  # grpo
            step_fn = self._grpo_setup(run)

        if run.budgets.keys() == {"epochs"} and not run.steps_per_epoch:
            log.warning("[%s] epoch budget without a known dataset size: using 1000 steps per epoch", run.name)
            run.steps_per_epoch = 1000.0

        # ---- compile warm-up (before the timer) -------------------------------------------------
        if self._compiled and warm_batches:
            self._warmup(run, warm_batches)

        self._emit("phase_start", {"phase": run.name, "type": ptype, "trainable_params": n_train,
                                   "total_params": n_total, "accum": run.accum, "budgets": run.budgets})
        self._log_line(
            f"[{run.name}] start: type={ptype} trainable={human_int(n_train)}/{human_int(n_total)} "
            f"micro_batch={bs} accum={run.accum} world={world} precision={self.precision} "
            f"compile={'on' if self._compiled else 'off'} budgets={run.budgets}")
        return self._loop(run, step_fn)

    def _warmup(self, run: _PhaseRun, batches: List[Tuple[torch.Tensor, ...]]) -> None:
        """Compile every shape once (forward + backward), then discard the gradients."""
        t0 = time.perf_counter()
        try:
            for b in batches:
                with self._autocast():
                    loss, _ = self._lm_loss(self.fwd_model, b)
                self._backward(loss * 0.0)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._log_line(f"[{run.name}] compiled {len(batches)} shapes in {human_time(time.perf_counter() - t0)}")
        except Exception as e:
            log.warning("torch.compile failed during warm-up (%s: %s); falling back to eager",
                        type(e).__name__, str(e).splitlines()[0] if str(e) else "")
            self.fwd_model = self.model
            self._compiled = False
        self._zero_grad()
        if self.scaler is not None:  # forget any inf bookkeeping from the warm-up
            self.scaler = self._make_scaler()
        if hasattr(self.raw_model, "update_router_biases"):
            self.raw_model.update_router_biases(rate=0.0)  # discard MoE loads from warm-up forwards

    # ---- DPO --------------------------------------------------------------------------------------
    def _dpo_setup(self, run: _PhaseRun, bs: int, seed: int) -> Callable[[float], Tuple[torch.Tensor, float, Dict[str, float]]]:
        from moonlight.data.sft import PreferenceLoader
        from moonlight.tokenizer import PAD_ID

        from moonlight.train.align import concat_pairs, dpo_loss, make_reference_model, sequence_logprobs

        phase = run.phase
        pairs = self.data[f"pref_{run.index}"]
        pair_bs = int(_get(phase, "batch_size", max(1, bs // 2)))
        run.accum = self._base_accum(2 * pair_bs, self.seq_len)
        run.steps_per_epoch = len(pairs) / max(1.0, float(pair_bs * run.accum * self.dist.world))
        loader = PreferenceLoader(pairs, pair_bs, str(self.device), seed)
        ref = make_reference_model(self.raw_model if self.strategy != "fsdp" else self.model)
        beta = float(_get(phase, "beta", 0.1))
        kind = str(_get(phase, "loss", "dpo"))

        def next_batch(micro: int) -> Tuple[torch.Tensor, ...]:
            c_x, c_y, r_x, r_y = loader.next()[:4]
            x, y = concat_pairs(_as_long(c_x, self.device), _as_long(c_y, self.device),
                                _as_long(r_x, self.device), _as_long(r_y, self.device), pad_id=PAD_ID)
            return x, y, None

        def loss_fn(model: Any, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Dict[str, Any]]:
            x, y, _ = batch
            n = x.shape[0] // 2
            pol = sequence_logprobs(model, x, y)
            with torch.no_grad():
                rf = sequence_logprobs(ref, x, y)
            loss, m = dpo_loss(pol[:n], pol[n:], rf[:n], rf[n:], beta=beta, loss_type=kind)
            return loss, m

        def step_fn(progress: float) -> Tuple[torch.Tensor, float, Dict[str, float]]:
            return self._accumulate(run, next_batch, loss_fn)

        self._ref_model = ref
        return step_fn

    # ---- GRPO -------------------------------------------------------------------------------------
    def _grpo_setup(self, run: _PhaseRun) -> Callable[[float], Tuple[torch.Tensor, float, Dict[str, float]]]:
        from moonlight.train.align import grpo_step, make_reference_model

        phase = run.phase
        prompts = self.data[f"grpo_{run.index}"]
        pps_global = max(1, int(_get(phase, "prompts_per_step", 4)))
        pps = max(1, math.ceil(pps_global / self.dist.world))
        run.accum = 1
        run.steps_per_epoch = len(prompts) / float(pps * self.dist.world)
        kl_coef = float(_get(phase, "kl_coef", 0.02))
        ref = make_reference_model(self.raw_model if self.strategy != "fsdp" else self.model) if kl_coef > 0 else None
        fsdp = self.strategy == "fsdp"
        sampler = self.model if fsdp else self.raw_model
        G = int(_get(phase, "group_size", 8))
        max_new = int(_get(phase, "max_new_tokens", 256))
        temperature = float(_get(phase, "temperature", 1.0))
        clip = float(_get(phase, "clip", 0.2))
        world, rank = self.dist.world, self.dist.rank

        def step_fn(progress: float) -> Tuple[torch.Tensor, float, Dict[str, float]]:
            start = (run.step * pps * world + rank * pps)
            batch = [prompts[(start + k) % len(prompts)] for k in range(pps)]
            m = grpo_step(self.fwd_model, sampler, ref, self.tokenizer, batch, group_size=G, max_new_tokens=max_new,
                          temperature=temperature, kl_coef=kl_coef, clip=clip,
                          seed=int(self.tc.seed) + run.step * world + rank,
                          max_seq_len=int(self.config.model.max_seq_len), fixed_length=fsdp,
                          sync_context=lambda last: grad_sync(self.model, last), backward=self._backward,
                          autocast=self._autocast)
            loss = torch.tensor(m.pop("loss"), device=self.device)
            tokens = m.pop("tokens")
            return loss, tokens, m

        self._ref_model = ref
        return step_fn

    # ---- the loop ---------------------------------------------------------------------------------
    def _loop(self, run: _PhaseRun, step_fn: Callable[[float], Tuple[torch.Tensor, float, Dict[str, float]]]) -> Dict[str, Any]:
        tc = self.tc
        world = self.dist.world
        log_every = max(1, int(tc.log_every or 1))
        eval_every = int(tc.eval_every or 0)
        ckpt_every_s = float(tc.checkpoint_every_min or 0.0) * 60.0
        base_momentum = float(tc.optimizer.momentum)
        t_start = time.perf_counter() - run.elapsed
        last_ckpt = time.perf_counter()
        last_log_t = time.perf_counter()
        last_log_tokens = run.tokens
        loss_acc = torch.zeros((), device=self.device)
        extra_acc: Dict[str, float] = {}
        n_acc = 0
        skipped_steps = 0
        last_loss = float("nan")
        first_loss: Optional[float] = None
        mult = 1.0
        interrupted = False
        try:
            while True:
                run.elapsed = time.perf_counter() - t_start
                progress = phase_progress(run.budgets, run.step, run.tokens, run.elapsed, run.steps_per_epoch)
                want_ckpt = 1.0 if (ckpt_every_s > 0 and time.perf_counter() - last_ckpt >= ckpt_every_s) else 0.0
                progress, want_ckpt = self._sync_flags(progress, want_ckpt)
                if progress >= 1.0:
                    break
                if want_ckpt:
                    self.save(run, done=False)
                    last_ckpt = time.perf_counter()
                mult = lr_multiplier(run.step, progress, run.sched)
                mom = None
                if any(isinstance(o, Muon) for o in self.optimizers):
                    mom = muon_momentum(run.step, 300, 0.85, base_momentum) if run.momentum_warmup else base_momentum
                self._set_lr(mult, mom)

                loss, tokens_local, extras = step_fn(progress)
                _, stepped = self._clip_and_step()
                if not stepped:
                    skipped_steps += 1
                if hasattr(self.raw_model, "update_router_biases"):
                    self.raw_model.update_router_biases()
                if self.ema is not None and stepped:
                    self.ema.update(self.raw_model, run.step)
                run.step += 1
                self.global_step += 1
                step_tokens = tokens_local * world
                run.tokens += step_tokens
                self.total_tokens += step_tokens
                loss_acc += loss.detach().float()
                last_loss = float(loss.detach().float().item())  # grad-norm .item() already syncs
                if first_loss is None:
                    first_loss = last_loss
                for k, v in extras.items():
                    extra_acc[k] = extra_acc.get(k, 0.0) + (float(v.item()) if torch.is_tensor(v) else float(v))
                n_acc += 1

                if run.step % log_every == 0 or run.step == 1:
                    mean_loss = loss_acc / max(1, n_acc)
                    if world > 1:
                        all_reduce_mean(mean_loss)
                    window_loss = float(mean_loss.item())
                    now = time.perf_counter()
                    tok_s = (run.tokens - last_log_tokens) / max(1e-9, now - last_log_t)
                    run.elapsed = now - t_start
                    rec: Dict[str, Any] = {"phase": run.name, "step": run.step, "global_step": self.global_step,
                                           "loss": round(window_loss, 5), "lr_mult": round(mult, 5),
                                           "tok_s": round(tok_s, 1), "tokens": run.tokens,
                                           "mem_gb": round(self._mem_gb(), 3), "elapsed": round(run.elapsed, 2),
                                           "progress": round(progress, 5)}
                    for k, v in extra_acc.items():
                        rec[k] = round(v / max(1, n_acc), 5)
                    self._write_metrics(rec)
                    extra_txt = "".join(f" | {k} {rec[k]:.3f}" for k in ("reward", "solve_rate", "reward_acc",
                                                                         "margin") if k in rec)
                    self._log_line(f"[{run.name}] step {run.step:>6} | loss {window_loss:.4f} | lr x{mult:.3f} | "
                                   f"{tok_s / 1e3:.1f}k tok/s | mem {rec['mem_gb']:.1f}G | "
                                   f"{human_time(run.elapsed)} | {progress * 100:.1f}%{extra_txt}")
                    self._emit("log", rec)
                    loss_acc.zero_()
                    extra_acc = {}
                    n_acc = 0
                    last_log_t, last_log_tokens = now, run.tokens
                if eval_every > 0 and run.step % eval_every == 0:
                    self._eval_and_sample(run)
        except KeyboardInterrupt:
            interrupted = True
            if world == 1:
                self._log_line(f"[{run.name}] interrupted at step {run.step}; saving checkpoint")
                run.elapsed = time.perf_counter() - t_start
                self.save(run, done=False)
            raise
        finally:
            if not interrupted:
                run.elapsed = time.perf_counter() - t_start
        return self._finish_phase(run, last_loss, skipped_steps, first_loss)

    def _eval_and_sample(self, run: _PhaseRun) -> float:
        val = self.evaluate(run.name)
        if math.isfinite(val):
            run.best_val = val if run.best_val is None else min(run.best_val, val)
        samples = self.sample_chat()
        rec = {"phase": run.name, "step": run.step, "event": "eval", "val_loss": val,
               "samples": [{"prompt": p, "reply": r} for p, r in samples]}
        self._write_metrics(rec)
        self._log_line(f"[{run.name}] eval step {run.step}: val_loss {val:.4f}")
        for p, r in samples:
            self._log_line(f"    > {p}\n    < {r[:200]!r}")
        self._emit("eval", rec)
        return val

    def _finish_phase(self, run: _PhaseRun, last_loss: float, skipped_steps: int,
                      first_loss: Optional[float] = None) -> Dict[str, Any]:
        val_raw = self.evaluate(run.name)
        val = val_raw
        used_ema = False
        if self.ema is not None and run.step > 0:
            with self.ema.swap(self.raw_model):
                val_ema = self.evaluate(run.name)
            if math.isfinite(val_ema) and (not math.isfinite(val_raw) or val_ema < val_raw):
                self.ema.copy_to(self.raw_model)
                val, used_ema = val_ema, True
            self._log_line(f"[{run.name}] end: val raw {val_raw:.4f} | ema {val_ema:.4f} -> "
                           f"keeping {'EMA' if used_ema else 'raw'} weights")
        else:
            self._log_line(f"[{run.name}] end: val {val_raw:.4f}")
        if math.isfinite(val):
            run.best_val = val if run.best_val is None else min(run.best_val, val)
        result = {"name": run.name, "type": run.ptype, "skipped": False, "steps": run.step, "tokens": run.tokens,
                  "elapsed": run.elapsed, "val_loss": val, "val_loss_raw": val_raw, "used_ema": used_ema,
                  "best_val": run.best_val, "first_train_loss": first_loss, "final_train_loss": last_loss,
                  "skipped_steps": skipped_steps}
        self.results.append(result)  # visible in the phase-end checkpoint state
        try:
            self.save(run, done=True)
        finally:
            self.results.pop()
        self._write_metrics(dict(result, phase=run.name, event="phase_end"))
        self._emit("phase_end", result)
        return result

    # --------------------------------------------------------------------------------------------
    # fit
    # --------------------------------------------------------------------------------------------
    def fit(self) -> Dict[str, Any]:
        """Run all phases (resuming from ``{output_dir}/checkpoint.safetensors`` when present).

        Returns a dict: ``phases`` (per-phase results), ``steps``, ``tokens``, ``elapsed``,
        ``final_val_loss``, ``output_dir``, ``checkpoint``, ``model_path`` (exported final
        model), ``tokenizer_path``. The trained tokenizer is available as ``trainer.tokenizer``.
        """
        t0 = time.perf_counter()
        phases = list(self.tc.phases)
        for p in phases:  # fail fast on typos
            phase_type(p)
        seed_everything(int(self.tc.seed) + self.dist.rank)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            try:
                self.config.to_yaml(self.output_dir / "config.yaml")
            except Exception as e:  # pragma: no cover
                log.warning("could not write config.yaml: %s", e)
        barrier()
        self.prepare_data()

        start, resume = 0, None
        if self.ckpt_path.is_file():
            state = read_checkpoint_state(self.ckpt_path)
            idx = int(state.get("phase_index", -1))
            self.results = list(state.get("results", []))
            self.global_step = int(state.get("global_step", 0))
            self.total_tokens = float(state.get("total_tokens", 0.0))
            if state.get("phase_done"):
                load_checkpoint(self.ckpt_path, self.model, map_location="cpu")
                start = idx + 1
                self._log_line(f"[resume] {self.ckpt_path}: phases 0..{idx} done; continuing at phase {start}")
            else:  # unfinished phase: run_phase restores weights, optimizer and EMA state
                start, resume = idx, state
        for i, phase in enumerate(phases):
            if i < start:
                continue
            res = self.run_phase(phase, index=i, resume=resume if i == start else None)
            self.results.append(res)
            resume = None
        set_trainable(self.raw_model, None)
        self.raw_model.eval()

        model_path = self.output_dir / "model.safetensors"
        export_model(model_path, self.model, self.config, self.tokenizer)
        finals = [r.get("val_loss") for r in self.results if not r.get("skipped") and r.get("val_loss") is not None]
        result = {
            "name": self.config.name,
            "phases": self.results,
            "steps": self.global_step,
            "tokens": self.total_tokens,
            "elapsed": time.perf_counter() - t0,
            "final_val_loss": finals[-1] if finals else None,
            "output_dir": str(self.output_dir),
            "checkpoint": str(self.ckpt_path),
            "model_path": str(model_path),
            "tokenizer_path": str(self.output_dir / "tokenizer.json"),
        }
        if self.is_main:
            (self.output_dir / "train_result.json").write_text(json.dumps(result, indent=1, default=str),
                                                               encoding="utf-8")
            self._log_line(f"[done] {len(self.results)} phases, {self.global_step} steps, "
                           f"{human_int(self.total_tokens)} tokens in {human_time(result['elapsed'])} -> {model_path}")
        if self._wandb is not None:  # pragma: no cover
            try:
                self._wandb.finish()
            except Exception:
                pass
        return result
