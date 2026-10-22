"""Optimizers (SPEC §7): Muon for weight matrices + AdamW for everything else.

``build_optimizers(model, train_cfg, lr_scale=1.0, trainable=None) -> list[Optimizer]``

Parameter assignment (names are "clean" FQNs, i.e. without ``module.`` / ``_orig_mod.`` /
``_fsdp_wrapped_module.`` wrapper prefixes):

* **Muon** - every parameter with ``ndim >= 2`` except the embedding (``embed.*``), the untied
  output head (``lm_head.*``), MoE routers (``*.router.*``) and ``vision_proj.norm.*``. Leading
  dims are treated as a batch of matrices: 3-D MoE expert weights ``(E, rows, cols)`` are
  orthogonalised per expert (as are the depthwise token-shift conv weights ``(d, 1, k)``).
  The update of every matrix is scaled by ``max(1, rows / cols) ** 0.5`` (last two dims).
* **AdamW** - the rest: embedding / lm_head at ``embed_lr``, everything else at ``adam_lr``.
  Weight decay is applied to matrices only (never to norms, biases, scalars, decays).

With ``train_cfg.optimizer.name == "adamw"`` every parameter uses AdamW (matrices at ``adam_lr``).
Every param group stores ``base_lr`` (= configured lr * ``lr_scale``) and a ``kind`` tag
(``muon`` | ``adam_embed`` | ``adam_matrix`` | ``adam_other``) so the trainer can apply the LR
schedule (``group["lr"] = group["base_lr"] * mult``) and the Muon momentum warmup.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.optim import Optimizer

__all__ = [
    "newton_schulz",
    "Muon",
    "build_optimizers",
    "clean_param_name",
    "is_muon_param",
    "parse_trainable",
    "set_trainable",
    "NS_COEFFS",
]

#: Quintic Newton-Schulz coefficients (Keller Jordan's Muon): fast convergence of singular values
#: to the neighbourhood of 1 (exact orthogonality is not needed for the update).
NS_COEFFS = (3.4445, -4.7750, 2.0315)

_WRAPPER_PARTS = frozenset(("_orig_mod", "module", "_fsdp_wrapped_module", "_checkpoint_wrapped_module"))


def clean_param_name(name: str) -> str:
    """Strip DDP / FSDP1 / torch.compile / activation-checkpoint wrapper components from a FQN
    (``"_orig_mod.module.layers.0.mixer.qkv.weight"`` -> ``"layers.0.mixer.qkv.weight"``)."""
    return ".".join(p for p in name.split(".") if p not in _WRAPPER_PARTS)


# ------------------------------------------------------------------------------------------------
# Newton-Schulz orthogonalisation
# ------------------------------------------------------------------------------------------------


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7,
                  dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Approximately orthogonalise the last two dims of ``G`` (batched over leading dims).

    Args:
        G: ``(..., rows, cols)`` tensor with ``ndim >= 2``.
        steps: quintic iterations (5 is enough for Muon).
        eps: added to the Frobenius norm before normalisation (all-zero matrices stay zero).
        dtype: compute dtype; default bfloat16 on CUDA, float32 elsewhere.

    Returns:
        Tensor of the same shape and dtype as ``G`` whose matrices have singular values ~ 1
        (roughly in [0.7, 1.2]), i.e. ``U V^T`` of the SVD ``G = U S V^T``.
    """
    if G.ndim < 2:
        raise ValueError(f"newton_schulz needs a tensor with ndim >= 2, got shape {tuple(G.shape)}")
    if dtype is None:
        dtype = torch.bfloat16 if G.is_cuda else torch.float32
    a, b, c = NS_COEFFS
    X = G.to(dtype)
    transposed = X.size(-2) > X.size(-1)
    if transposed:  # iterate on the wide orientation: X X^T is the smaller Gram matrix
        X = X.mT
    norm = X.float().norm(dim=(-2, -1), keepdim=True).to(dtype)
    X = X / (norm + eps)
    for _ in range(int(steps)):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


def _is_dtensor(t: Any) -> bool:
    return type(t).__name__ == "DTensor" and hasattr(t, "full_tensor") and hasattr(t, "placements")


def _orthogonalize(update: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz on a (possibly FSDP2-sharded) update; returns a tensor like ``update``.

    For a ``DTensor`` (FSDP2 / ``fully_shard``) the matrix is gathered with ``full_tensor()``
    (an all-gather: every rank needs the full matrix since NS mixes all rows and columns),
    orthogonalised redundantly on every rank, then re-sharded with ``distribute_tensor`` using the
    same mesh and placements (a local slice, no further communication). This trades extra compute
    for simplicity; it is exact.
    """
    if _is_dtensor(update):
        try:
            from torch.distributed.tensor import distribute_tensor  # torch >= 2.4
        except ImportError:  # pragma: no cover - older torch
            from torch.distributed._tensor import distribute_tensor
        full = update.full_tensor()
        ortho = newton_schulz(full, steps)
        return distribute_tensor(ortho, update.device_mesh, update.placements)
    return newton_schulz(update, steps)


def _muon_scale(shape: Sequence[int]) -> float:
    """``max(1, rows / cols) ** 0.5`` using the LAST two dims."""
    rows, cols = int(shape[-2]), int(shape[-1])
    return max(1.0, rows / max(cols, 1)) ** 0.5


class Muon(Optimizer):
    """MomentUm Orthogonalized by Newton-schulz (Muon) for 2-D (and batched N-D) weights.

    Per parameter ``p`` with gradient ``g`` (all shapes ``(..., rows, cols)``)::

        buf = momentum * buf + (1 - momentum) * g          # EMA of gradients
        u   = g + momentum * (buf - g)  if nesterov else buf  # = lerp(g, buf, momentum)
        u   = newton_schulz(u) * max(1, rows / cols) ** 0.5
        p   = p * (1 - lr * weight_decay) - lr * u

    Parameters of identical shape (and dtype/device) are orthogonalised together in ONE batched
    Newton-Schulz call (stacked along a new leading dim). FSDP2 ``DTensor`` parameters are handled
    by gathering the full matrix, orthogonalising and re-sharding (see :func:`_orthogonalize`).

    Args:
        params: iterable of parameters or param-group dicts.
        lr: learning rate (default 0.02).
        momentum: momentum coefficient (default 0.95; the trainer warms it up from 0.85).
        nesterov: use Nesterov momentum (default True).
        weight_decay: decoupled weight decay (default 0).
        ns_steps: Newton-Schulz iterations (default 5).
    """

    def __init__(self, params: Iterable[Any], lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, weight_decay: float = 0.0, ns_steps: int = 5) -> None:
        if lr < 0:
            raise ValueError(f"invalid lr {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid momentum {momentum}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight_decay {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(f"Muon only handles parameters with ndim >= 2 (got shape "
                                     f"{tuple(p.shape)}); use AdamW for vectors/scalars")

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            mom = float(group["momentum"])
            wd = float(group["weight_decay"])
            nesterov = bool(group["nesterov"])
            steps = int(group["ns_steps"])
            # 1) momentum update, collect the (pre-orthogonalisation) updates
            plain: Dict[Tuple[Any, ...], List[Tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
            sharded: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                buf.lerp_(g, 1.0 - mom)
                u = g.lerp(buf, mom) if nesterov else buf
                if _is_dtensor(u):
                    sharded.append((p, u))
                else:
                    plain[(tuple(u.shape), u.dtype, u.device)].append((p, u))
            # 2) batched Newton-Schulz per shape, then apply
            for (shape, _dt, _dev), items in plain.items():
                stacked = torch.stack([u for _, u in items])  # (n, ..., rows, cols)
                ortho = newton_schulz(stacked, steps)
                scale = _muon_scale(shape)
                for (p, _), o in zip(items, ortho.unbind(0)):
                    self._apply(p, o, lr, wd, scale)
            for p, u in sharded:
                self._apply(p, _orthogonalize(u, steps), lr, wd, _muon_scale(u.shape))
        return loss

    @staticmethod
    def _apply(p: torch.Tensor, update: torch.Tensor, lr: float, wd: float, scale: float) -> None:
        if wd != 0.0:
            p.mul_(1.0 - lr * wd)
        p.add_(update.to(p.dtype), alpha=-lr * scale)


# ------------------------------------------------------------------------------------------------
# Parameter grouping
# ------------------------------------------------------------------------------------------------


def _is_embedding_name(name: str) -> bool:
    return name.startswith("embed.") or name.startswith("lm_head.")


def is_muon_param(name: str, p: torch.Tensor) -> bool:
    """SPEC §7 rule: ndim >= 2 and not embedding / lm_head / router / vision_proj.norm."""
    name = clean_param_name(name)
    if p.ndim < 2:
        return False
    if _is_embedding_name(name) or name.startswith("vision_proj.norm"):
        return False
    if ".router." in f".{name}.":
        return False
    return True


def parse_trainable(trainable: Union[None, str, Sequence[str]]) -> Optional[List[str]]:
    """Normalise a phase ``trainable`` value to a list of name prefixes (``None`` = everything).

    Accepts ``"vision_proj"``, ``"vision_proj,mtp"``, ``["vision_proj", "norm"]``; ``"all"`` /
    ``"*"`` / empty -> ``None``.
    """
    if trainable is None:
        return None
    if isinstance(trainable, str):
        items = [t.strip() for t in trainable.replace(";", ",").split(",")]
    else:
        items = [str(t).strip() for t in trainable]
    items = [t for t in items if t]
    if not items or any(t in ("all", "*") for t in items):
        return None
    return items


def _matches(name: str, prefixes: Optional[Sequence[str]]) -> bool:
    if prefixes is None:
        return True
    name = clean_param_name(name)
    return any(name == p or name.startswith(p.rstrip(".") + ".") for p in prefixes)


def set_trainable(model: torch.nn.Module, trainable: Union[None, str, Sequence[str]] = None,
                  exclude: Sequence[str] = ()) -> Tuple[int, int]:
    """Set ``requires_grad`` = name matches ``trainable`` prefixes and not ``exclude`` prefixes.

    Returns ``(n_trainable_params, n_total_params)`` (element counts). Raises ``ValueError`` when
    a non-empty ``trainable`` selection matches no parameter (typo protection).
    """
    prefixes = parse_trainable(trainable)
    excl = [e for e in exclude if e]
    n_train = n_total = 0
    for name, p in model.named_parameters():
        on = _matches(name, prefixes) and not (excl and _matches(name, excl))
        p.requires_grad_(on)
        n_total += p.numel()
        n_train += p.numel() if on else 0
    if prefixes is not None and n_train == 0:
        raise ValueError(f"trainable={trainable!r} matches no parameter of the model")
    return n_train, n_total


def _opt_cfg(train_cfg: Any) -> Any:
    return getattr(train_cfg, "optimizer", train_cfg)


def _adamw(groups: List[Dict[str, Any]], betas: Tuple[float, float], device_type: str) -> Optimizer:
    kwargs: Dict[str, Any] = dict(betas=betas, eps=1e-8)
    has_dtensor = any(_is_dtensor(p) for g in groups for p in g["params"])
    if device_type == "cuda" and not has_dtensor:
        # fused kernels where supported (torch >= 2.0); foreach otherwise
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except (TypeError, RuntimeError, ValueError):
            pass
    try:
        return torch.optim.AdamW(groups, foreach=True, **kwargs)
    except (TypeError, RuntimeError):  # pragma: no cover
        return torch.optim.AdamW(groups, **kwargs)


def build_optimizers(model: torch.nn.Module, train_cfg: Any, lr_scale: float = 1.0,
                     trainable: Union[None, str, Sequence[str]] = None) -> List[Optimizer]:
    """Create ``[Muon, AdamW]`` (or ``[AdamW]``) for the trainable parameters of ``model``.

    Args:
        model: the (possibly DDP/FSDP/compiled-wrapped) model; names are cleaned of wrapper
            prefixes. Tied parameters are visited once (``named_parameters`` de-duplicates).
        train_cfg: ``TrainConfig`` (uses ``.optimizer``) or an ``OptimizerConfig``.
        lr_scale: multiplies every learning rate (phase ``lr_scale``).
        trainable: optional name prefixes (str / list) restricting the optimized parameters;
            parameters with ``requires_grad=False`` are always skipped.

    Returns:
        Non-empty list of optimizers; each param group has ``base_lr`` and ``kind``.
        Raises ``ValueError`` if no parameter is trainable.
    """
    oc = _opt_cfg(train_cfg)
    name = str(getattr(oc, "name", "muon")).lower()
    lr_scale = float(lr_scale)
    muon_lr = float(getattr(oc, "lr", 0.02)) * lr_scale
    adam_lr = float(getattr(oc, "adam_lr", 0.003)) * lr_scale
    embed_lr = float(getattr(oc, "embed_lr", adam_lr)) * lr_scale
    momentum = float(getattr(oc, "momentum", 0.95))
    betas_cfg = list(getattr(oc, "betas", [0.9, 0.95]))
    betas = (float(betas_cfg[0]), float(betas_cfg[1]))
    wd = float(getattr(oc, "weight_decay", 0.0))
    prefixes = parse_trainable(trainable)

    muon_params: List[torch.Tensor] = []
    embed_params: List[torch.Tensor] = []
    matrix_params: List[torch.Tensor] = []  # adamw mode: matrices get weight decay
    other_params: List[torch.Tensor] = []
    device_type = "cpu"
    for pname, p in model.named_parameters():
        if not p.requires_grad or not _matches(pname, prefixes):
            continue
        device_type = p.device.type
        clean = clean_param_name(pname)
        if _is_embedding_name(clean):
            embed_params.append(p)
        elif name == "muon" and is_muon_param(clean, p):
            muon_params.append(p)
        elif p.ndim >= 2:
            matrix_params.append(p)
        else:
            other_params.append(p)
    if not (muon_params or embed_params or matrix_params or other_params):
        raise ValueError("build_optimizers: no trainable parameters "
                         f"(trainable={trainable!r}; check requires_grad / phase 'trainable')")

    optimizers: List[Optimizer] = []
    if muon_params:
        optimizers.append(Muon([dict(params=muon_params, lr=muon_lr, base_lr=muon_lr, kind="muon")],
                               lr=muon_lr, momentum=momentum, nesterov=True, weight_decay=wd))
    groups: List[Dict[str, Any]] = []
    if embed_params:
        groups.append(dict(params=embed_params, lr=embed_lr, base_lr=embed_lr, weight_decay=wd,
                           kind="adam_embed"))
    if matrix_params:
        groups.append(dict(params=matrix_params, lr=adam_lr, base_lr=adam_lr, weight_decay=wd,
                           kind="adam_matrix"))
    if other_params:
        groups.append(dict(params=other_params, lr=adam_lr, base_lr=adam_lr, weight_decay=0.0,
                           kind="adam_other"))
    if groups:
        optimizers.append(_adamw(groups, betas, device_type))
    return optimizers
