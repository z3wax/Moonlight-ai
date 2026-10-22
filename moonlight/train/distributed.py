"""Multi-GPU helpers (SPEC §7): process-group setup from torchrun env, DDP / FSDP wrapping, collectives.

Everything degrades to a no-op in a single process (no ``RANK``/``WORLD_SIZE`` env vars), so the
trainer can call these helpers unconditionally. ``torch`` is imported lazily inside functions, so
``DistInfo`` and env parsing work without torch.

Launch example::

    torchrun --nproc_per_node 8 -m moonlight train ml-1b-chat

Strategies (``cfg.train.distributed.strategy``):

* ``none`` - plain module (single process; with world > 1 this is rejected).
* ``ddp``  - ``DistributedDataParallel`` (``broadcast_buffers=False``: the only persistent buffers,
  MoE router biases, are updated identically on every rank by ``update_router_biases``).
* ``fsdp`` - FSDP2 ``fully_shard`` per ``Block`` (+ MTP blocks) and the root when available
  (torch >= 2.4), otherwise FSDP1 with a transformer auto-wrap policy on ``Block``. Mixed
  precision: params/compute in bf16 (fp16 when bf16 is unsupported), gradient reduction in fp32.
  ``fsdp_shard``: ``full`` (reshard after forward, ZeRO-3), ``grad_op`` (keep params gathered
  between forward and backward, ZeRO-2-like), ``hybrid`` (HSDP: shard within a node, replicate
  across nodes). ``tp_size`` / ``ep_size`` are documented placeholders - tensor and expert
  parallelism are NOT implemented (SPEC §12).
"""

from __future__ import annotations

import contextlib
import datetime
import os
from dataclasses import dataclass
from typing import Any, Iterator, Optional

__all__ = [
    "DistInfo",
    "init_distributed",
    "wrap_model",
    "unwrap_model",
    "is_distributed",
    "all_reduce_mean",
    "all_reduce_sum",
    "all_reduce_max",
    "broadcast_object",
    "barrier",
    "grad_sync",
    "is_fsdp",
    "fsdp_version",
    "cleanup",
]


@dataclass
class DistInfo:
    """Process placement. ``device`` is a string usable with ``torch.device`` (``"cuda:3"``)."""

    rank: int = 0
    local_rank: int = 0
    world: int = 1
    device: str = "cpu"
    is_main: bool = True
    backend: Optional[str] = None

    @property
    def distributed(self) -> bool:
        return self.world > 1


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, ""))
    except ValueError:
        return default


def _dist() -> Any:
    """``torch.distributed`` if available AND initialised, else None."""
    try:
        import torch.distributed as dist
    except ImportError:
        return None
    if dist.is_available() and dist.is_initialized():
        return dist
    return None


def is_distributed() -> bool:
    """True when a process group with more than one rank is initialised."""
    dist = _dist()
    return dist is not None and dist.get_world_size() > 1


def init_distributed(backend: Optional[str] = None, device: Optional[str] = None,
                     timeout_minutes: float = 30.0) -> DistInfo:
    """Initialise ``torch.distributed`` from torchrun env vars (no-op for a single process).

    Reads ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE`` (``MASTER_ADDR``/``MASTER_PORT`` are consumed
    by the ``env://`` init method). With CUDA each process is pinned to ``cuda:{LOCAL_RANK}`` and
    uses NCCL; otherwise gloo on CPU. An already initialised process group is reused.

    Args:
        backend: override the backend (``nccl`` / ``gloo``).
        device: device for the single-process case (default: ``moonlight.utils.get_device()``).
        timeout_minutes: collective timeout.
    """
    world = max(1, _env_int("WORLD_SIZE", 1))
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    dist = _dist()
    if world <= 1 and dist is None:
        from moonlight.utils import get_device

        return DistInfo(rank=0, local_rank=0, world=1, device=get_device(device), is_main=True)

    import torch
    import torch.distributed as tdist

    use_cuda = torch.cuda.is_available() and (device is None or str(device).startswith("cuda"))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        dev = f"cuda:{local_rank}"
    else:
        dev = "cpu"
    if dist is None:
        be = backend or ("nccl" if use_cuda else "gloo")
        kwargs: dict = dict(backend=be, timeout=datetime.timedelta(minutes=float(timeout_minutes)))
        if use_cuda:
            try:  # torch >= 2.3: bind the NCCL communicator eagerly to this device
                tdist.init_process_group(device_id=torch.device(dev), **kwargs)
            except TypeError:
                tdist.init_process_group(**kwargs)
        else:
            tdist.init_process_group(**kwargs)
    else:
        be = tdist.get_backend()
    rank = tdist.get_rank()
    world = tdist.get_world_size()
    return DistInfo(rank=rank, local_rank=local_rank, world=world, device=dev, is_main=rank == 0,
                    backend=str(be))


def cleanup() -> None:
    """Destroy the process group (if any)."""
    dist = _dist()
    if dist is not None:
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronise all ranks (no-op single-process)."""
    dist = _dist()
    if dist is not None and dist.get_world_size() > 1:
        dist.barrier()


def _reduce(value: Any, op_name: str, average: bool = False) -> Any:
    dist = _dist()
    if dist is None or dist.get_world_size() <= 1:
        return value
    import torch

    is_tensor = isinstance(value, torch.Tensor)
    if is_tensor:
        t = value
    else:
        dev = "cuda" if (torch.cuda.is_available() and dist.get_backend() == "nccl") else "cpu"
        if dev == "cuda":
            dev = f"cuda:{torch.cuda.current_device()}"
        t = torch.tensor(float(value), dtype=torch.float64, device=dev)
    op = getattr(dist.ReduceOp, op_name)
    if average and t.dtype in (torch.float16, torch.bfloat16):
        acc = t.float()
        dist.all_reduce(acc, op=op)
        acc /= dist.get_world_size()
        t.copy_(acc)
    else:
        dist.all_reduce(t, op=op)
        if average:
            t /= dist.get_world_size()
    return t if is_tensor else float(t.item())


def all_reduce_mean(t: Any) -> Any:
    """Average over ranks. Tensors are reduced IN PLACE and returned; python numbers -> float.
    (Integer tensors are not supported for averaging - convert to float first.)"""
    return _reduce(t, "SUM", average=True)


def all_reduce_sum(t: Any) -> Any:
    """Sum over ranks (tensor in place, or python number -> float)."""
    return _reduce(t, "SUM")


def all_reduce_max(t: Any) -> Any:
    """Max over ranks (tensor in place, or python number -> float). Used to agree on budgets."""
    return _reduce(t, "MAX")


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a picklable object from ``src`` to every rank (identity single-process)."""
    dist = _dist()
    if dist is None or dist.get_world_size() <= 1:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]


# ------------------------------------------------------------------------------------------------
# Wrapping
# ------------------------------------------------------------------------------------------------


def unwrap_model(model: Any) -> Any:
    """Return the underlying ``nn.Module`` (strips torch.compile ``_orig_mod``, DDP/FSDP1 ``module``).

    FSDP2 (``fully_shard``) modifies the module in place, so it is returned as is.
    """
    seen = 0
    while seen < 8:
        seen += 1
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        cls = type(model).__name__
        if cls in ("DistributedDataParallel", "FullyShardedDataParallel") and hasattr(model, "module"):
            model = model.module
            continue
        break
    return model


def fsdp_version(model: Any) -> int:
    """2 for an FSDP2 (``fully_shard``) module, 1 for an FSDP1 wrapper, 0 otherwise."""
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    if type(m).__name__ == "FullyShardedDataParallel":
        return 1
    if hasattr(m, "set_requires_gradient_sync") and hasattr(m, "reshard"):  # FSDPModule API
        return 2
    return 0


def is_fsdp(model: Any) -> bool:
    return fsdp_version(model) > 0


@contextlib.contextmanager
def grad_sync(model: Any, enabled: bool) -> Iterator[None]:
    """Context for one gradient-accumulation micro-step.

    ``enabled=False`` (non-final micro-steps) skips the gradient all-reduce / reduce-scatter:
    DDP and FSDP1 ``no_sync()``; FSDP2 ``set_requires_gradient_sync(False)`` (restored to True on
    exit). ``enabled=True`` or an unwrapped model -> plain context.
    """
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    if enabled:
        yield
        return
    if hasattr(m, "set_requires_gradient_sync"):  # FSDP2
        m.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            m.set_requires_gradient_sync(True)
        return
    if hasattr(m, "no_sync") and type(m).__name__ in ("DistributedDataParallel", "FullyShardedDataParallel"):
        with m.no_sync():
            yield
        return
    yield


def _mp_dtype(precision: Optional[str]) -> Any:
    import torch

    p = (precision or "bf16").lower()
    if p in ("fp32", "float32", "none"):
        return None
    if p in ("fp16", "float16"):
        return torch.float16
    if p in ("bf16", "bfloat16", "auto"):
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.bfloat16
    raise ValueError(f"unknown precision {precision!r}")


def _block_modules(model: Any) -> tuple:
    """``(blocks, Block class)``: modules sharded individually = every ``Block`` (decoder layers
    and MTP blocks)."""
    block_cls = getattr(model, "block_class", None)
    if block_cls is None:
        from moonlight.model.model import Block as block_cls  # noqa: N813
    return [m for m in model.modules() if isinstance(m, block_cls)], block_cls


def _wrap_fsdp2(model: Any, mp_dtype: Any, shard: str) -> Any:
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard  # torch >= 2.6
    except ImportError:
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard  # 2.4/2.5
    import torch

    kwargs: dict = {}
    if mp_dtype is not None:
        kwargs["mp_policy"] = MixedPrecisionPolicy(param_dtype=mp_dtype, reduce_dtype=torch.float32)
    shard = (shard or "full").lower()
    if shard in ("grad_op", "shard_grad_op", "zero2"):
        kwargs["reshard_after_forward"] = False
    elif shard in ("hybrid", "hsdp"):
        from torch.distributed.device_mesh import init_device_mesh

        import torch.distributed as dist

        world = dist.get_world_size()
        local = max(1, _env_int("LOCAL_WORLD_SIZE", world))
        if world % local != 0:
            raise ValueError(f"hybrid sharding needs WORLD_SIZE ({world}) divisible by LOCAL_WORLD_SIZE ({local})")
        dev_type = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs["mesh"] = init_device_mesh(dev_type, (world // local, local), mesh_dim_names=("replicate", "shard"))
    blocks, _ = _block_modules(model)
    for blk in blocks:
        fully_shard(blk, **kwargs)
    fully_shard(model, **kwargs)
    return model


def _wrap_fsdp1(model: Any, mp_dtype: Any, shard: str, device: str) -> Any:
    import functools

    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    _, block_cls = _block_modules(model)
    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={block_cls})
    strategies = {"full": ShardingStrategy.FULL_SHARD, "grad_op": ShardingStrategy.SHARD_GRAD_OP,
                  "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP, "zero2": ShardingStrategy.SHARD_GRAD_OP,
                  "hybrid": ShardingStrategy.HYBRID_SHARD, "hsdp": ShardingStrategy.HYBRID_SHARD,
                  "no_shard": ShardingStrategy.NO_SHARD}
    strategy = strategies.get((shard or "full").lower(), ShardingStrategy.FULL_SHARD)
    mp = None
    if mp_dtype is not None:
        mp = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    dev = torch.device(device)
    return FSDP(model, auto_wrap_policy=policy, mixed_precision=mp, sharding_strategy=strategy,
                device_id=dev if dev.type == "cuda" else None, use_orig_params=True,
                limit_all_gathers=True)


def wrap_model(model: Any, strategy: str = "none", dist: Optional[DistInfo] = None,
               precision: Optional[str] = "bf16", fsdp_shard: str = "full",
               find_unused_parameters: bool = False) -> Any:
    """Wrap ``model`` for data-parallel training.

    Args:
        model: ``MoonlightForCausalLM`` already moved to ``dist.device``.
        strategy: ``none`` | ``ddp`` | ``fsdp`` (``none`` with world > 1 -> ``ValueError``).
        dist: placement (default: :func:`init_distributed`).
        precision: FSDP mixed-precision param dtype (``bf16``/``fp16``/``fp32``/``auto``).
        fsdp_shard: ``full`` | ``grad_op`` | ``hybrid``.
        find_unused_parameters: forwarded to DDP.

    Returns:
        The wrapped model: DDP / FSDP1 wrapper objects, or ``model`` itself for FSDP2 (sharded in
        place) and for single-process runs (where ``ddp``/``fsdp`` are no-ops).
    """
    strategy = (strategy or "none").lower()
    dist = dist or init_distributed()
    if dist.world <= 1:
        return model
    if strategy == "none":
        raise ValueError("train.distributed.strategy is 'none' but WORLD_SIZE > 1; use ddp or fsdp")
    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        cuda = str(dist.device).startswith("cuda")
        return DDP(model, device_ids=[dist.local_rank] if cuda else None, broadcast_buffers=False,
                   find_unused_parameters=find_unused_parameters, gradient_as_bucket_view=True)
    if strategy == "fsdp":
        mp_dtype = _mp_dtype(precision)
        try:
            return _wrap_fsdp2(model, mp_dtype, fsdp_shard)
        except ImportError:
            return _wrap_fsdp1(model, mp_dtype, fsdp_shard, dist.device)
    raise ValueError(f"unknown distributed strategy {strategy!r}; expected none, ddp or fsdp")
