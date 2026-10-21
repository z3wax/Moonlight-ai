"""Basic building blocks shared by every Moonlight model.

Contents
--------
- :func:`rms`              parameter-free RMS normalisation (used for q/k "qk_norm").
- :class:`RMSNorm`         RMS normalisation with a learned ``weight`` (no bias).
- :class:`RotaryEmbedding` (alias :data:`RoPE`) rotary position embedding, rotate-half convention.
- :func:`apply_rope`       apply cos/sin tables to a (B, H, T, hd) tensor.
- :class:`SwiGLU`          dense gated MLP with parameters ``w_gate_up`` (d -> 2h) and ``w_down`` (h -> d).
- :class:`TokenShift`      depthwise causal ``nn.Conv1d`` with a streaming (cached) mode.
- :func:`copy_into_`       in-place initialisation helper that is a no-op on meta tensors and
                           understands FSDP2 ``DTensor`` parameters.

Conventions: activations are (batch, time, channels) unless stated otherwise; every Linear layer
has ``bias=False``; normalisations are computed in float32 and cast back to the input dtype, so
they are safe under ``torch.autocast`` and with bf16/fp16 weights.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "RMS_EPS",
    "rms",
    "RMSNorm",
    "RotaryEmbedding",
    "RoPE",
    "rotate_half",
    "apply_rope",
    "SwiGLU",
    "TokenShift",
    "copy_into_",
    "is_meta",
]

RMS_EPS = 1e-6


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def is_meta(t: torch.Tensor) -> bool:
    """True if ``t`` lives on the meta device (shape-only tensor, no storage)."""
    return bool(getattr(t, "is_meta", False))


def _is_dtensor(t: torch.Tensor) -> bool:
    return hasattr(t, "device_mesh") and hasattr(t, "placements")


@torch.no_grad()
def copy_into_(param: torch.Tensor, value: torch.Tensor) -> None:
    """Copy ``value`` (a regular, full-size tensor) into ``param`` in place.

    * meta tensors are skipped (nothing to initialise),
    * FSDP2 ``DTensor`` parameters receive a distributed copy with the same placements,
    * otherwise ``value`` is moved to the parameter's device/dtype and copied.
    """
    if is_meta(param):
        return
    if _is_dtensor(param):
        try:  # torch >= 2.4
            from torch.distributed.tensor import distribute_tensor
        except ImportError:  # pragma: no cover - older torch
            from torch.distributed._tensor import distribute_tensor
        mesh = param.device_mesh
        value = value.to(device=torch.device(mesh.device_type), dtype=param.dtype)
        param.copy_(distribute_tensor(value, mesh, param.placements))
        return
    param.copy_(value.to(device=param.device, dtype=param.dtype))


# --------------------------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------------------------
def rms(x: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    """Parameter-free RMS normalisation over the last dimension.

    ``x``: any shape (..., D). Computed in float32; returns a tensor of ``x.shape`` and ``x.dtype``.
    """
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)


class RMSNorm(nn.Module):
    """RMS normalisation over the last dimension with a learned scale ``weight`` (dim,).

    ``forward(x)``: (..., dim) -> (..., dim) in ``x.dtype`` (statistics in float32).
    """

    def __init__(self, dim: int, eps: float = RMS_EPS) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim))

    def reset_parameters(self) -> None:
        if not is_meta(self.weight):
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}"


# --------------------------------------------------------------------------------------------
# rotary position embedding
# --------------------------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """Rotary position embedding (rotate-half convention, ``theta`` base).

    The module has no parameters and no buffers: the cos/sin tables are computed on the fly in
    float32 from the absolute positions. This is cheap (T x hd/2 trig ops per forward, shared by
    all layers) and robust to meta-device construction, ``to_empty()`` materialisation, checkpoint
    loading and ``model.to(bfloat16)`` (cached float tables would silently lose precision or stay
    uninitialised in those flows). It therefore trivially satisfies "rope buffers are
    non-persistent / not counted".

    ``forward(positions)``: positions (T,) integer tensor -> (cos, sin), each (T, head_dim) float32.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        self.head_dim = int(head_dim)
        self.theta = float(theta)

    def inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """(head_dim/2,) float32 inverse frequencies ``theta^(-2i/head_dim)``."""
        half = self.head_dim // 2
        exponent = torch.arange(half, device=device, dtype=torch.float32) * (2.0 / self.head_dim)
        return 1.0 / (self.theta ** exponent)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq(positions.device)
        freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]  # (T, hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, hd)
        return emb.cos(), emb.sin()

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, theta={self.theta}"


RoPE = RotaryEmbedding


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(..., hd) -> (..., hd): [x1, x2] -> [-x2, x1] with x1/x2 the two halves of the last dim."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` (B, H, T, hd) with tables ``cos``/``sin`` (T, hd); math in float32.

    Returns a tensor with the shape and dtype of ``x``.
    """
    xf = x.float()
    return (xf * cos + rotate_half(xf) * sin).to(x.dtype)


# --------------------------------------------------------------------------------------------
# feed-forward
# --------------------------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """Gated MLP: ``w_down(silu(gate) * up)`` with ``[gate | up] = w_gate_up(x)``.

    Parameters: ``w_gate_up`` Linear(dim, 2*hidden), ``w_down`` Linear(hidden, dim) -> 3*dim*hidden.
    ``forward(x)``: (..., dim) -> (..., dim).
    """

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.w_gate_up = nn.Linear(self.dim, 2 * self.hidden, bias=False)
        self.w_down = nn.Linear(self.hidden, self.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


# --------------------------------------------------------------------------------------------
# token shift
# --------------------------------------------------------------------------------------------
class TokenShift(nn.Conv1d):
    """Depthwise causal convolution over time ("token shift").

    This *is* ``nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)`` (single parameter
    ``weight`` of shape (dim, 1, kernel_size)), applied causally: output t only depends on inputs
    t-k+1 .. t. It is identity-initialised (``weight[:, 0, -1] = 1``, all other taps 0), so a fresh
    layer passes its input through unchanged.

    ``forward(x, conv_buf=None, return_buffer=False)``
        x:        (B, T, dim)
        conv_buf: (B, dim, k-1) the k-1 inputs preceding ``x`` (streaming decode); ``None`` means
                  zero left-padding (start of a sequence).
        returns   y (B, T, dim), or (y, new_buf) with new_buf (B, dim, k-1) = the last k-1 inputs
                  (in ``x.dtype``) when ``return_buffer`` is True.
    """

    def __init__(self, dim: int, kernel_size: int) -> None:
        if int(kernel_size) < 1:
            raise ValueError(f"TokenShift kernel_size must be >= 1, got {kernel_size}")
        super().__init__(int(dim), int(dim), int(kernel_size), groups=int(dim), bias=False)

    def reset_parameters(self) -> None:  # called by nn.Conv1d.__init__ as well
        w = self.weight
        if is_meta(w):
            return
        ident = torch.zeros(tuple(w.shape), dtype=torch.float32, device="cpu")
        ident[:, 0, -1] = 1.0
        copy_into_(w, ident)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        conv_buf: Optional[torch.Tensor] = None,
        return_buffer: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        k = self.kernel_size[0]
        xt = x.transpose(1, 2)  # (B, d, T)
        if conv_buf is not None:
            xp = torch.cat([conv_buf.to(device=xt.device, dtype=xt.dtype), xt], dim=-1)
        elif k > 1:
            xp = F.pad(xt, (k - 1, 0))
        else:
            xp = xt
        y = F.conv1d(xp, self.weight, None, 1, 0, 1, self.groups).transpose(1, 2)  # (B, T, d)
        if not return_buffer:
            return y
        # last k-1 inputs (xp always has >= k-1 columns); clone so the big tensor can be freed
        new_buf = xp[:, :, xp.shape[-1] - (k - 1):].clone()
        return y, new_buf
