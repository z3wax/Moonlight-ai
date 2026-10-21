"""Grouped-query softmax attention (SDPA) with optional sliding window and KV cache.

The projections live in :class:`moonlight.model.model.HybridMixer` (single fused ``qkv``); this
module implements the attention computation itself:

- :func:`repeat_kv`            expand KV heads for grouped-query attention (``repeat_interleave``).
- :func:`sliding_window_mask`  boolean "may attend" mask for queries that follow ``n_cached`` keys.
- :func:`gqa_attention`        causal attention over (cached +) new keys, updating the cache.

Masking rules (query at absolute position p attends keys at positions in ``(p - window, p]``, or
``[0, p]`` for global layers):

- no cached keys and (global layer or T <= window): SDPA with ``is_causal=True`` (no mask tensor);
- a single query token (decoding): SDPA without a mask over all cached + new keys (for local
  layers only the last ``window`` of them, which are exactly the visible ones);
- anything else (sliding window with T > window, multi-token chunks after a cache offset): an
  explicit boolean mask of shape (T_q, T_k).

Cache contents for a layer: ``k, v`` of shape (B, Hkv, T_c, hd) *after* RoPE and *before* GQA
expansion. Global layers keep every key; local (sliding-window) layers keep only the last
``window`` keys (trimmed to the window).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = ["repeat_kv", "sliding_window_mask", "gqa_attention", "update_kv_cache"]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, Hkv, T, hd) -> (B, Hkv * n_rep, T, hd); query head h uses KV head h // n_rep."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def sliding_window_mask(
    q_len: int,
    k_len: int,
    window: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Boolean (q_len, k_len) mask, True = may attend.

    The queries are the last ``q_len`` positions of the ``k_len`` keys (i.e. ``k_len - q_len``
    keys precede the first query). Query i attends key j iff ``j <= i + off`` and, with a
    window, ``j > i + off - window`` where ``off = k_len - q_len``.
    """
    off = k_len - q_len
    qpos = torch.arange(q_len, device=device)[:, None] + off
    kpos = torch.arange(k_len, device=device)[None, :]
    mask = kpos <= qpos
    if window is not None:
        mask = mask & (kpos > qpos - window)
    return mask


def update_kv_cache(cache, k: torch.Tensor, v: torch.Tensor, window: Optional[int]) -> None:
    """Store the full key/value sequence (B, Hkv, T, hd) into ``cache`` (a LayerCache).

    Local layers keep only the last ``window`` positions (cloned, so a long prefill tensor can be
    released); global layers keep everything.
    """
    if window is None:
        cache.k, cache.v = k, v
        return
    keep = int(window)
    total = k.shape[2]
    if total <= keep:
        cache.k, cache.v = k, v
    else:
        cache.k = k[:, :, total - keep:].clone()
        cache.v = v[:, :, total - keep:].clone()


def gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: Optional[int] = None,
    cache=None,
) -> torch.Tensor:
    """Causal grouped-query attention.

    q: (B, Ha, Tq, hd) with RoPE applied; k, v: (B, Hkv, Tq, hd) for the *new* tokens (RoPE
    applied to k); ``Ha % Hkv == 0``. ``window``: sliding-window size for local layers (None =
    global). ``cache``: optional LayerCache whose ``k``/``v`` are prepended and then updated in
    place. Returns (B, Ha, Tq, hd). Scale is the SDPA default ``hd ** -0.5``.
    """
    Tq = q.shape[2]
    n_cached = 0
    if cache is not None:
        if cache.k is not None and cache.k.shape[2] > 0:
            k = torch.cat([cache.k.to(device=k.device, dtype=k.dtype), k], dim=2)
            v = torch.cat([cache.v.to(device=v.device, dtype=v.dtype), v], dim=2)
            n_cached = k.shape[2] - Tq
        update_kv_cache(cache, k, v, window)

    if Tq == 1 and window is not None and k.shape[2] > window:
        # a single query at position p sees exactly the keys (p - window, p]: the last `window`
        k, v = k[:, :, -window:], v[:, :, -window:]
        n_cached = window - 1

    n_rep = q.shape[1] // k.shape[1]
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    if n_cached == 0 and (window is None or Tq <= window):
        return F.scaled_dot_product_attention(q, k, v, is_causal=Tq > 1)
    if Tq == 1:
        return F.scaled_dot_product_attention(q, k, v)
    mask = sliding_window_mask(Tq, k.shape[2], window, device=q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
