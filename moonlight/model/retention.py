"""Retention: linear attention with a fixed per-head exponential decay (RetNet-style).

Per head with decay ``gamma`` in (0, 1):

    o_t = sum_{j <= t} gamma^(t - j) (q_t . k_j) v_j                       (1)
    S_t = gamma * S_{t-1} + k_t^T v_t,      o_t = q_t S_t                  (2)  (recurrent form)

The query scale (``head_dim ** -0.5``) is applied by the caller, these functions implement (1)
exactly. Four implementations that agree up to floating point rounding:

- :func:`retention_parallel`        O(T^2) masked decay matrix ``D[i, j] = gamma^(i-j) (i >= j)``.
- :func:`retention_chunkwise`       O(T * C): exact parallel form inside chunks of ``C`` tokens plus a
                                    recurrent state carried across chunks. For a chunk with local
                                    positions i, j (0-based) and incoming state ``S_prev``::

                                        intra = (Q K^T * D) V
                                        cross = (Q * gamma^(i+1)) @ S_prev
                                        S_new = gamma^L S_prev + sum_j gamma^(L-1-j) k_j^T v_j

                                    where ``L`` is the chunk length (the last chunk may be partial and
                                    then uses its own length).
- :func:`retention_recurrent_step`  one decoding step of (2).
- :func:`retention_final_state`     the state after a block of tokens (cache prefill),
                                    ``S_T = gamma^T S_0 + sum_j gamma^(T-1-j) k_j^T v_j``.

:func:`retention` dispatches between them (chunkwise for ``T > 256``, as required by the spec).

Shapes: ``q, k``: (B, H, T, dk); ``v``: (B, H, T, dv) (any leading dims ending in H work for the
parallel/chunkwise/final-state forms); ``log_gamma``: (H,) = log(gamma) < 0; ``state``:
(B, H, dk, dv). All decay factors and every state-related product are computed in float32 with
autocast disabled; the O(T*C) intra-chunk products follow the ambient autocast dtype. Outputs have
``v.dtype``; states are float32.
"""
from __future__ import annotations

import contextlib
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

__all__ = [
    "CHUNK_SIZE",
    "CHUNKWISE_MIN_LEN",
    "decay_init_logits",
    "log_gamma_from_logits",
    "decay_matrix",
    "retention",
    "retention_parallel",
    "retention_chunkwise",
    "retention_recurrent_step",
    "retention_final_state",
]

CHUNK_SIZE = 64
CHUNKWISE_MIN_LEN = 256  # chunkwise form is used when T > CHUNKWISE_MIN_LEN

LogGamma = Union[torch.Tensor, float, Sequence[float]]


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def decay_init_logits(n_heads: int, layer_idx: int) -> List[float]:
    """Initial ``decay_logit`` values for a layer (spec section 4.5).

    ``gamma_h = 1 - 2^(-x_h)`` with ``x = linspace(lo, hi, n_heads)``, ``lo = 2 + 0.3 (i mod 8)``,
    ``hi = 5 + 0.4 (i mod 8)`` and ``gamma = sigmoid(logit)``, i.e. ``logit = log(2^x - 1)``.
    """
    n = int(n_heads)
    if n <= 0:
        return []
    m = int(layer_idx) % 8
    lo = 2.0 + 0.3 * m
    hi = 5.0 + 0.4 * m
    if n == 1:
        xs = [lo]
    else:
        step = (hi - lo) / (n - 1)
        xs = [lo + step * h for h in range(n)]
    return [math.log(2.0 ** x - 1.0) for x in xs]


def log_gamma_from_logits(decay_logit: torch.Tensor) -> torch.Tensor:
    """``log(sigmoid(decay_logit))`` in float32 (numerically stable)."""
    return F.logsigmoid(decay_logit.float())


def _log_gamma(log_gamma: LogGamma, n_heads: int, device: torch.device) -> torch.Tensor:
    """Normalise ``log_gamma`` to a float32 (H,) tensor on ``device`` (scalars broadcast)."""
    if isinstance(log_gamma, torch.Tensor):
        lg = log_gamma.to(device=device, dtype=torch.float32).reshape(-1)
    else:
        lg = torch.tensor(log_gamma, dtype=torch.float32, device=device).reshape(-1)
    if lg.numel() == 1 and n_heads != 1:
        lg = lg.expand(n_heads)
    if lg.numel() != n_heads:
        raise ValueError(f"log_gamma has {lg.numel()} entries but the input has {n_heads} heads")
    return lg


def _autocast_off(device: torch.device):
    """Context manager disabling autocast for ``device`` (no-op for devices without autocast)."""
    if device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


def decay_matrix(log_gamma: torch.Tensor, length: int) -> torch.Tensor:
    """(H, L, L) float32 matrix ``D[h, i, j] = gamma_h^(i - j)`` for ``i >= j`` else 0.

    ``log_gamma``: (H,) float32.
    """
    idx = torch.arange(length, device=log_gamma.device, dtype=torch.float32)
    diff = idx[:, None] - idx[None, :]  # (L, L) = i - j
    decay = torch.exp(log_gamma.float()[:, None, None] * diff.clamp(min=0.0))
    return decay.masked_fill(diff < 0, 0.0)


# --------------------------------------------------------------------------------------------
# implementations
# --------------------------------------------------------------------------------------------
def retention_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_gamma: LogGamma,
    state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Parallel (quadratic) retention.

    q, k: (B, H, T, dk); v: (B, H, T, dv); log_gamma: (H,); optional ``state`` (B, H, dk, dv) is the
    recurrent state *before* the first token (adds ``gamma^(t+1) q_t S``). Returns (B, H, T, dv)
    in ``v.dtype``.
    """
    H, T = q.shape[-3], q.shape[-2]
    lg = _log_gamma(log_gamma, H, q.device)
    D = decay_matrix(lg, T)  # (H, T, T)
    scores = torch.matmul(q, k.transpose(-1, -2)).float() * D
    out = torch.matmul(scores.to(v.dtype), v)
    if state is not None:
        with _autocast_off(q.device):
            pos = torch.arange(1, T + 1, device=q.device, dtype=torch.float32)
            qdec = torch.exp(lg[:, None] * pos)[..., None]  # (H, T, 1)
            out = out.float() + torch.matmul(q.float() * qdec, state.float())
    return out.to(v.dtype)


def retention_final_state(
    k: torch.Tensor,
    v: torch.Tensor,
    log_gamma: LogGamma,
    state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """State after consuming the block (k, v): ``S_T = gamma^T S_0 + sum_j gamma^(T-1-j) k_j^T v_j``.

    k: (B, H, T, dk); v: (B, H, T, dv); state (optional S_0): (B, H, dk, dv).
    Returns float32 (B, H, dk, dv).
    """
    H, T = k.shape[-3], k.shape[-2]
    lg = _log_gamma(log_gamma, H, k.device)
    with _autocast_off(k.device):
        j = torch.arange(T, device=k.device, dtype=torch.float32)
        kdec = torch.exp(lg[:, None] * (T - 1 - j))[..., None]  # (H, T, 1)
        S = torch.matmul((k.float() * kdec).transpose(-1, -2), v.float())
        if state is not None:
            S = S + state.float() * torch.exp(lg * T)[:, None, None]
    return S


def retention_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_gamma: LogGamma,
    chunk: int = CHUNK_SIZE,
    state: Optional[torch.Tensor] = None,
    return_state: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Chunkwise retention: exact, O(T * chunk) memory/compute.

    q, k: (B, H, T, dk); v: (B, H, T, dv); log_gamma: (H,); ``state`` (B, H, dk, dv) optional
    initial state. Returns out (B, H, T, dv) in ``v.dtype``, plus the float32 final state
    (B, H, dk, dv) when ``return_state``.

    All chunks are processed in parallel for the intra-chunk part; the (cheap, element-wise) scan
    over chunk states is sequential. ``T`` is zero-padded up to a multiple of ``chunk`` -- padding
    only affects discarded outputs because the form is causal; the final state is computed from
    the true (possibly partial) last chunk.
    """
    lead = tuple(q.shape[:-2])
    T, dk = q.shape[-2], q.shape[-1]
    dv = v.shape[-1]
    H = q.shape[-3]
    C = int(chunk)
    if C < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    lg = _log_gamma(log_gamma, H, q.device)
    if T == 0:
        out = v.new_zeros(lead + (0, dv))
        if not return_state:
            return out
        S0 = state.float() if state is not None else torch.zeros(
            lead + (dk, dv), dtype=torch.float32, device=q.device)
        return out, S0

    N = (T + C - 1) // C
    pad = N * C - T
    if pad:
        qp, kp, vp = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    else:
        qp, kp, vp = q, k, v
    qc = qp.reshape(lead + (N, C, dk))
    kc = kp.reshape(lead + (N, C, dk))
    vc = vp.reshape(lead + (N, C, dv))

    # intra-chunk (parallel form inside each chunk, follows autocast)
    D = decay_matrix(lg, C)[:, None]  # (H, 1, C, C) broadcasts over (B, H, N, C, C)
    scores = torch.matmul(qc, kc.transpose(-1, -2)).float() * D
    intra = torch.matmul(scores.to(v.dtype), vc)  # (B, H, N, C, dv)

    with _autocast_off(q.device):
        idx = torch.arange(C, device=q.device, dtype=torch.float32)
        # contribution of each (full-length) chunk to the state at its end
        kdec = torch.exp(lg[:, None] * (C - 1 - idx))[:, None, :, None]  # (H, 1, C, 1)
        contrib = torch.matmul((kc.float() * kdec).transpose(-1, -2), vc.float())  # (B,H,N,dk,dv)
        g_chunk = torch.exp(lg * C)[:, None, None]  # (H, 1, 1)
        S = state.float() if state is not None else torch.zeros(
            lead + (dk, dv), dtype=torch.float32, device=q.device)
        prev: List[torch.Tensor] = []
        for c in range(N):
            prev.append(S)  # state before chunk c
            if c + 1 < N:
                S = S * g_chunk + contrib[..., c, :, :]
        S_prev = torch.stack(prev, dim=-3)  # (B, H, N, dk, dv)
        qdec = torch.exp(lg[:, None] * (idx + 1.0))[:, None, :, None]  # (H, 1, C, 1)
        cross = torch.matmul(qc.float() * qdec, S_prev)  # (B, H, N, C, dv)
        out = (intra.float() + cross).reshape(lead + (N * C, dv))[..., :T, :].to(v.dtype)
        if not return_state:
            return out
        # final state from the true last chunk (length L_last <= C)
        start = (N - 1) * C
        L_last = T - start
        j = torch.arange(L_last, device=q.device, dtype=torch.float32)
        kdec_last = torch.exp(lg[:, None] * (L_last - 1 - j))[..., None]  # (H, L_last, 1)
        k_last = k[..., start:, :].float()
        v_last = v[..., start:, :].float()
        S_final = prev[-1] * torch.exp(lg * L_last)[:, None, None] + torch.matmul(
            (k_last * kdec_last).transpose(-1, -2), v_last)
    return out, S_final


def retention_recurrent_step(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    state: Optional[torch.Tensor],
    log_gamma: LogGamma,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One recurrent step: ``S = gamma * S + k_t^T v_t``; ``o = q_t S``.

    q_t, k_t: (B, H, dk) or (B, H, 1, dk); v_t: (B, H, dv) or (B, H, 1, dv);
    state: (B, H, dk, dv) or ``None`` (zeros). Returns ``(o_t, new_state)`` with o_t of the same
    rank as ``q_t`` (dtype ``v_t.dtype``) and new_state float32 (B, H, dk, dv).
    """
    has_time = q_t.dim() == 4
    if has_time:
        q_t, k_t, v_t = q_t[..., 0, :], k_t[..., 0, :], v_t[..., 0, :]
    H = q_t.shape[-2]
    lg = _log_gamma(log_gamma, H, q_t.device)
    with _autocast_off(q_t.device):
        kv = k_t.float().unsqueeze(-1) * v_t.float().unsqueeze(-2)  # (B, H, dk, dv)
        if state is None:
            S = kv
        else:
            S = state.float() * torch.exp(lg)[:, None, None] + kv
        o = torch.matmul(q_t.float().unsqueeze(-2), S).squeeze(-2)  # (B, H, dv)
    o = o.to(v_t.dtype)
    if has_time:
        o = o.unsqueeze(-2)
    return o, S


def retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_gamma: LogGamma,
    state: Optional[torch.Tensor] = None,
    return_state: bool = False,
    chunk: int = CHUNK_SIZE,
    chunkwise_min_len: int = CHUNKWISE_MIN_LEN,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Retention with automatic algorithm choice.

    q, k: (B, H, T, dk); v: (B, H, T, dv). ``T == 1`` with a state request/initial state ->
    recurrent step; ``T > chunkwise_min_len`` -> chunkwise; otherwise parallel (+ final state).
    Returns out (B, H, T, dv) [, float32 state (B, H, dk, dv) if ``return_state``].
    """
    T = q.shape[-2]
    if T == 1 and q.dim() == 4 and (return_state or state is not None):
        o, S = retention_recurrent_step(q, k, v, state, log_gamma)
        return (o, S) if return_state else o
    if T > chunkwise_min_len:
        return retention_chunkwise(q, k, v, log_gamma, chunk=chunk, state=state,
                                   return_state=return_state)
    out = retention_parallel(q, k, v, log_gamma, state=state)
    if return_state:
        return out, retention_final_state(k, v, log_gamma, state=state)
    return out
