"""Safetensors checkpoints, resume and final export (SPEC §7).

Training checkpoint (``save_checkpoint``) - ONE ``.safetensors`` file:

* tensors ``model/{name}`` (full model ``state_dict``, persistent buffers included),
  ``ema/{name}`` (EMA shadow weights, optional),
  ``opt{i}/{param_name}/{key}`` (tensor state of optimizer ``i``, e.g. ``momentum_buffer``,
  ``exp_avg``, ``exp_avg_sq``, ``step``), keyed by the clean parameter FQN;
* metadata (all values are ``str``, as safetensors requires): ``format``
  (``moonlight-ckpt-v1``), ``state`` (JSON trainer state: phase index, step, tokens_seen, elapsed,
  best_val, ...), ``config`` (YAML of the full config), ``optim`` (JSON: per optimizer the param
  group hyper-parameters with their parameter names + non-tensor per-parameter state).

The write is atomic (temp file + ``os.replace``). Under FSDP the full (unsharded) state is
gathered with ``torch.distributed.checkpoint.state_dict`` (a collective: call ``save_checkpoint``
/ ``export_model`` on EVERY rank; only rank 0 writes). ``load_checkpoint`` is called on every rank
(the file must be on a shared filesystem) and restores sharded models / optimizers via
``set_model_state_dict`` / ``set_optimizer_state_dict``.

Final export (``export_model``): plain parameter names (no ``model/`` prefix), fp32 or bf16
weights, metadata ``format=moonlight-v1``, ``config`` (YAML), ``tokenizer`` (tokenizer JSON).

The metadata readers (:func:`read_metadata`, :func:`read_checkpoint_state`) parse the safetensors
header directly and need neither torch nor safetensors.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "export_model",
    "load_exported",
    "read_metadata",
    "read_checkpoint_state",
    "checkpoint_path",
    "CHECKPOINT_NAME",
    "CKPT_FORMAT",
    "EXPORT_FORMAT",
]

PathLike = Union[str, os.PathLike]

CHECKPOINT_NAME = "checkpoint.safetensors"
CKPT_FORMAT = "moonlight-ckpt-v1"
EXPORT_FORMAT = "moonlight-v1"


def checkpoint_path(output_dir: PathLike) -> Path:
    """``{output_dir}/checkpoint.safetensors`` (the file ``Trainer.fit`` resumes from)."""
    return Path(output_dir).expanduser() / CHECKPOINT_NAME


# ------------------------------------------------------------------------------------------------
# torch-free header readers
# ------------------------------------------------------------------------------------------------


def _read_header(path: PathLike) -> Dict[str, Any]:
    with open(Path(path).expanduser(), "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file (truncated header)")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 100 * 2**20:
            raise ValueError(f"{path}: not a safetensors file (header size {n})")
        return json.loads(f.read(n).decode("utf-8"))


def read_metadata(path: PathLike) -> Dict[str, str]:
    """The ``__metadata__`` mapping of a safetensors file (``{}`` if absent). Torch-free."""
    return dict(_read_header(path).get("__metadata__") or {})


def read_checkpoint_state(path: PathLike) -> Dict[str, Any]:
    """Decoded trainer ``state`` JSON of a training checkpoint (``{}`` if absent). Torch-free."""
    text = read_metadata(path).get("state")
    return json.loads(text) if text else {}


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------


def _safetensors():
    try:
        import safetensors.torch as st
    except ImportError as e:  # pragma: no cover - hard dependency in pyproject
        raise ImportError("checkpointing needs the 'safetensors' package: pip install safetensors") from e
    return st


def _clean(name: str) -> str:
    from moonlight.train.optim import clean_param_name

    return clean_param_name(name)


def _is_dtensor(t: Any) -> bool:
    return type(t).__name__ == "DTensor" and hasattr(t, "full_tensor")


def _to_cpu(t: Any) -> Any:
    """Detached, contiguous CPU copy (DTensor -> full tensor; collective under FSDP2)."""
    if _is_dtensor(t):
        t = t.full_tensor()
    # clone: safetensors refuses tensors that share storage (e.g. views / tied weights)
    return t.detach().to("cpu").contiguous().clone()


def _is_main() -> bool:
    from moonlight.utils import get_rank

    return get_rank() == 0


def _fsdp(model: Any) -> bool:
    from moonlight.train.distributed import is_fsdp

    return is_fsdp(model)


def _state_dict_options(**kw: Any) -> Any:
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    return StateDictOptions(**kw)


def _model_state(model: Any) -> Dict[str, Any]:
    """Full model state dict with clean names on CPU (all ranks must call under FSDP)."""
    if _fsdp(model):
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        sd = get_model_state_dict(model, options=_state_dict_options(full_state_dict=True, cpu_offload=True))
    else:
        from moonlight.train.distributed import unwrap_model

        sd = unwrap_model(model).state_dict()
    return {_clean(k): _to_cpu(v) for k, v in sd.items() if hasattr(v, "detach")}


def _named_param_ids(model: Any) -> Dict[int, str]:
    from moonlight.train.distributed import unwrap_model

    return {id(p): _clean(n) for n, p in unwrap_model(model).named_parameters()}


def _hyper(group: Mapping[str, Any]) -> Dict[str, Any]:
    """JSON-safe hyper-parameters of a param group (``params`` excluded)."""
    out = {}
    for k, v in group.items():
        if k == "params":
            continue
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (bool, int, float, str)) for x in v):
            out[k] = list(v)
    return out


# Implementation switches depend on the device the run resumes on (e.g. fused AdamW is CUDA-only),
# so they are never restored from a checkpoint; only hyper-parameters are.
_NON_RESTORED = frozenset(("params", "param_names", "fused", "foreach", "capturable", "differentiable"))


def _restorable(group: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in group.items() if k not in _NON_RESTORED}


def _optimizer_states(model: Any, optimizers: Sequence[Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """(tensors ``opt{i}/{name}/{key}``, JSON meta per optimizer). Collective under FSDP."""
    tensors: Dict[str, Any] = {}
    metas: List[Dict[str, Any]] = []
    fsdp = _fsdp(model)
    id2name = None if fsdp else _named_param_ids(model)
    for i, opt in enumerate(optimizers):
        if fsdp:
            from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

            osd = get_optimizer_state_dict(model, opt, options=_state_dict_options(full_state_dict=True,
                                                                                   cpu_offload=True))
            state_by_name = {_clean(str(k)): v for k, v in osd.get("state", {}).items()}
            groups = [dict(_hyper(g), param_names=[_clean(str(n)) for n in g["params"]])
                      for g in osd.get("param_groups", [])]
        else:
            state_by_name = {}
            groups = []
            for g in opt.param_groups:
                names = [id2name.get(id(p), f"__unnamed_{id(p)}") for p in g["params"]]
                groups.append(dict(_hyper(g), param_names=names))
                for p, n in zip(g["params"], names):
                    if p in opt.state:
                        state_by_name[n] = opt.state[p]
        scalars: Dict[str, Dict[str, Any]] = {}
        for name, st in state_by_name.items():
            for key, val in st.items():
                if hasattr(val, "detach"):
                    tensors[f"opt{i}/{name}/{key}"] = _to_cpu(val)
                elif isinstance(val, (bool, int, float, str)) or val is None:
                    scalars.setdefault(name, {})[key] = val
        metas.append({"param_groups": groups, "scalars": scalars, "type": type(opt).__name__})
    return tensors, metas


def _ema_state(ema: Any) -> Dict[str, Any]:
    if ema is None:
        return {}
    sd = ema.state_dict() if hasattr(ema, "state_dict") else dict(ema)
    return {_clean(k): _to_cpu(v) for k, v in sd.items()}


def _config_yaml(config: Any) -> str:
    if config is None:
        return ""
    if isinstance(config, str):
        return config
    if hasattr(config, "to_yaml"):
        return config.to_yaml()
    import yaml

    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def _atomic_save(tensors: Dict[str, Any], path: Path, metadata: Dict[str, str]) -> None:
    st = _safetensors()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        st.save_file(tensors, str(tmp), metadata={k: str(v) for k, v in metadata.items()})
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _barrier() -> None:
    from moonlight.train.distributed import barrier

    barrier()


# ------------------------------------------------------------------------------------------------
# public API
# ------------------------------------------------------------------------------------------------


def save_checkpoint(path: PathLike, model: Any, optimizers: Optional[Sequence[Any]], state: Mapping[str, Any],
                    ema: Any = None, config: Any = None) -> Path:
    """Write a resumable training checkpoint (see module docstring for the layout).

    Args:
        path: target ``.safetensors`` file (parent dirs are created).
        model: the (possibly DDP/FSDP/compiled) model.
        optimizers: list of optimizers (or None) - state stored per optimizer index.
        state: JSON-serialisable trainer state (phase index, step, tokens_seen, elapsed, best_val..).
        ema: optional EMA object with ``state_dict() -> {name: tensor}`` (or a plain dict).
        config: full ``Config`` (or YAML string) stored as metadata ``config``.

    Returns the path. Call on every rank under FSDP (collective gather); only rank 0 writes; ends
    with a barrier.
    """
    path = Path(path).expanduser()
    tensors: Dict[str, Any] = {f"model/{k}": v for k, v in _model_state(model).items()}
    tensors.update({f"ema/{k}": v for k, v in _ema_state(ema).items()})
    opt_tensors, metas = _optimizer_states(model, list(optimizers or []))
    tensors.update(opt_tensors)
    if _is_main():
        meta = {
            "format": CKPT_FORMAT,
            "state": json.dumps(dict(state), default=_json_default),
            "config": _config_yaml(config),
            "optim": json.dumps(metas, default=_json_default),
        }
        _atomic_save(tensors, path, meta)
    _barrier()
    return path


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _load_model_tensors(model: Any, sd: Dict[str, Any], strict: bool = True) -> None:
    if _fsdp(model):
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        set_model_state_dict(model, sd, options=_state_dict_options(full_state_dict=True, strict=strict))
        return
    from moonlight.train.distributed import unwrap_model

    target = unwrap_model(model)
    expected = set(target.state_dict().keys())
    missing = sorted(expected - set(sd))
    unexpected = sorted(set(sd) - expected)
    if strict and (missing or unexpected):
        raise RuntimeError(f"checkpoint does not match the model: missing={missing[:8]} "
                           f"unexpected={unexpected[:8]} (shape/config mismatch?)")
    target.load_state_dict(sd, strict=strict)


def _load_optimizer(model: Any, opt: Any, i: int, tensors: Dict[str, Any], meta: Dict[str, Any]) -> None:
    prefix = f"opt{i}/"
    per_name: Dict[str, Dict[str, Any]] = {}
    for k, v in tensors.items():
        if k.startswith(prefix):
            name, key = k[len(prefix):].rsplit("/", 1)
            per_name.setdefault(name, {})[key] = v
    for name, sc in (meta.get("scalars") or {}).items():
        per_name.setdefault(name, {}).update(sc)
    saved_groups = meta.get("param_groups") or []

    if _fsdp(model):
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict

        cur = get_optimizer_state_dict(model, opt, options=_state_dict_options(full_state_dict=True))
        groups = []
        for gi, g in enumerate(cur.get("param_groups", [])):
            ng = dict(g)
            if gi < len(saved_groups):
                ng.update(_restorable(saved_groups[gi]))
            groups.append(ng)
        state = {fqn: per_name[_clean(str(fqn))] for g in groups for fqn in g["params"]
                 if _clean(str(fqn)) in per_name}
        set_optimizer_state_dict(model, opt, {"state": state, "param_groups": groups},
                                 options=_state_dict_options(full_state_dict=True))
        return

    id2name = _named_param_ids(model)
    cur = opt.state_dict()
    new_state: Dict[int, Dict[str, Any]] = {}
    idx = 0
    for g in opt.param_groups:
        for p in g["params"]:
            name = id2name.get(id(p))
            if name is not None and name in per_name:
                new_state[idx] = dict(per_name[name])
            idx += 1
    groups = []
    for gi, g in enumerate(cur["param_groups"]):
        ng = dict(g)
        if gi < len(saved_groups):
            ng.update(_restorable(saved_groups[gi]))
        groups.append(ng)
    opt.load_state_dict({"state": new_state, "param_groups": groups})


def load_checkpoint(path: PathLike, model: Any = None, optimizers: Optional[Sequence[Any]] = None,
                    ema: Any = None, map_location: Any = "cpu", strict: bool = True) -> Dict[str, Any]:
    """Restore a checkpoint written by :func:`save_checkpoint`; returns the trainer ``state`` dict.

    Args:
        path: checkpoint file.
        model: model to load ``model/*`` into (skipped when None).
        optimizers: optimizers to restore (index ``i`` <- ``opt{i}/*``); they must have been
            built for the same trainable parameters (group hyper-parameters are restored too).
        ema: EMA object with ``load_state_dict`` (or a dict of tensors updated in place).
        map_location: device the tensors are read to (state is later moved by
            ``Optimizer.load_state_dict`` / ``copy_`` to the parameters' devices).
        strict: require exact model key match.
    """
    path = Path(path).expanduser()
    st = _safetensors()
    meta = read_metadata(path)
    fmt = meta.get("format", "")
    if fmt and fmt != CKPT_FORMAT:
        raise ValueError(f"{path}: unexpected checkpoint format {fmt!r} (expected {CKPT_FORMAT})")
    tensors = st.load_file(str(path), device=str(map_location) if map_location is not None else "cpu")
    if model is not None:
        sd = {k[len("model/"):]: v for k, v in tensors.items() if k.startswith("model/")}
        _load_model_tensors(model, sd, strict=strict)
    if ema is not None:
        esd = {k[len("ema/"):]: v for k, v in tensors.items() if k.startswith("ema/")}
        if esd:
            if hasattr(ema, "load_state_dict"):
                ema.load_state_dict(esd)
            else:
                for k, v in esd.items():
                    if k in ema:
                        ema[k].copy_(v)
    if optimizers:
        metas = json.loads(meta.get("optim") or "[]")
        for i, opt in enumerate(optimizers):
            if any(k.startswith(f"opt{i}/") for k in tensors) or (i < len(metas) and metas[i].get("scalars")):
                _load_optimizer(model, opt, i, tensors, metas[i] if i < len(metas) else {})
    text = meta.get("state")
    return json.loads(text) if text else {}


def export_model(path: PathLike, model: Any, config: Any, tokenizer: Any = None,
                 dtype: Optional[str] = None) -> Path:
    """Write the final single-file model (SPEC §7): weights + ``config`` + ``tokenizer`` metadata.

    Args:
        path: output ``.safetensors``.
        model: trained model (wrappers are fine; collective under FSDP).
        config: full ``Config`` (stored as YAML) or YAML string.
        tokenizer: ``MoonlightTokenizer`` (stored with ``to_str()``) / JSON string / None.
        dtype: ``"fp32"`` (default) or ``"bf16"``; floating parameters are cast, persistent
            float32 buffers (``router_bias``) are kept in float32.

    Tensor names are the plain ``state_dict`` names (``embed.weight``, ``layers.0...``).
    """
    import torch

    path = Path(path).expanduser()
    sd = _model_state(model)
    want = (dtype or "fp32").lower()
    if want in ("bf16", "bfloat16"):
        cast = torch.bfloat16
    elif want in ("fp32", "float32"):
        cast = torch.float32
    elif want in ("fp16", "float16"):
        cast = torch.float16
    else:
        raise ValueError(f"unsupported export dtype {dtype!r}; use fp32 or bf16")
    out = {}
    for k, v in sd.items():
        if v.is_floating_point() and not k.endswith("router_bias"):
            v = v.to(cast)
        out[k] = v
    if _is_main():
        if tokenizer is None:
            tok = ""
        elif isinstance(tokenizer, str):
            tok = tokenizer
        else:
            tok = tokenizer.to_str()
        meta = {
            "format": EXPORT_FORMAT,
            "config": _config_yaml(config),
            "tokenizer": tok,
            "dtype": "bf16" if cast is torch.bfloat16 else ("fp16" if cast is torch.float16 else "fp32"),
            "num_params": str(sum(int(v.numel()) for k, v in out.items() if not k.endswith("router_bias"))),
        }
        _atomic_save(out, path, meta)
    _barrier()
    return path


def load_exported(path: PathLike, map_location: Any = "cpu") -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Read an :func:`export_model` file -> ``(state_dict, metadata)``."""
    st = _safetensors()
    meta = read_metadata(path)
    sd = st.load_file(str(Path(path).expanduser()), device=str(map_location))
    return sd, meta
