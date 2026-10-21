"""Inference cache for Moonlight models (attention KV, retention state, token-shift buffer).

``InferenceCache(n_layers)`` holds one :class:`LayerCache` per transformer block:

- ``k``, ``v``:    attention keys/values (B, Hkv, T_c, hd) after RoPE; local (sliding-window)
                   layers keep only the last ``window`` positions, global layers keep all.
- ``state``:       retention state (B, Hr, hd, hd) float32, ``S_t = sum_j gamma^(t-j) k_j^T v_j``.
- ``conv_buf``:    token-shift buffer (B, d, k-1): the last k-1 (normalised) mixer inputs.

plus ``seen_tokens`` (int): number of positions already processed (the RoPE offset of the next
token). All rows of a batch share the same position (no ragged batches). ``reorder(index)`` and
``repeat(n)`` act on the batch dimension (used for best-of-N sampling); both modify the cache in
place and return it, so ``cache = cache.repeat(n)`` and ``cache.repeat(n)`` are equivalent.

This module is torch-agnostic at import time (only tensor methods are used).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional

__all__ = ["LayerCache", "InferenceCache"]

_FIELDS = ("k", "v", "state", "conv_buf")


@dataclass
class LayerCache:
    """Per-layer cache entries (any of them may be ``None`` if the layer has no such part)."""

    k: Optional[Any] = None
    v: Optional[Any] = None
    state: Optional[Any] = None
    conv_buf: Optional[Any] = None

    def tensors(self) -> Iterator[Any]:
        for name in _FIELDS:
            t = getattr(self, name)
            if t is not None:
                yield t

    def map_(self, fn: Callable[[Any], Any]) -> "LayerCache":
        for name in _FIELDS:
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, fn(t))
        return self


class InferenceCache:
    """Cache for incremental decoding; see module docstring for layouts."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = int(n_layers)
        self.layers: List[LayerCache] = [LayerCache() for _ in range(self.n_layers)]
        self.seen_tokens: int = 0

    # -- container protocol -------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_layers

    def __getitem__(self, idx: int) -> LayerCache:
        return self.layers[idx]

    def __iter__(self) -> Iterator[LayerCache]:
        return iter(self.layers)

    def __repr__(self) -> str:
        return (f"InferenceCache(n_layers={self.n_layers}, seen_tokens={self.seen_tokens}, "
                f"batch_size={self.batch_size}, nbytes={self.nbytes()})")

    # -- info ---------------------------------------------------------------------------------
    @property
    def batch_size(self) -> Optional[int]:
        """Batch size of the cached tensors (None while the cache is empty)."""
        for layer in self.layers:
            for t in layer.tensors():
                return int(t.shape[0])
        return None

    def is_empty(self) -> bool:
        return self.seen_tokens == 0 and self.batch_size is None

    def nbytes(self) -> int:
        """Total bytes held by cached tensors."""
        return int(sum(t.numel() * t.element_size() for layer in self.layers for t in layer.tensors()))

    # -- batch manipulation ---------------------------------------------------------------------
    def _map(self, fn: Callable[[Any], Any]) -> "InferenceCache":
        for layer in self.layers:
            layer.map_(fn)
        return self

    def reorder(self, index) -> "InferenceCache":
        """Select/reorder batch rows in place: every tensor ``t`` becomes ``t[index]``.

        ``index``: 1-D LongTensor (or list of ints) of row indices. Returns ``self``.
        """
        def sel(t):
            idx = index
            if not hasattr(idx, "to"):
                import torch  # local import keeps this module importable without torch

                idx = torch.as_tensor(list(idx), dtype=torch.long)
            return t.index_select(0, idx.to(t.device))

        return self._map(sel)

    def repeat(self, n: int) -> "InferenceCache":
        """Repeat every batch row ``n`` times (row b -> rows b*n .. b*n+n-1), in place."""
        n = int(n)
        if n < 1:
            raise ValueError(f"repeat count must be >= 1, got {n}")
        if n == 1:
            return self
        return self._map(lambda t: t.repeat_interleave(n, dim=0))

    def to(self, device=None, dtype=None) -> "InferenceCache":
        """Move tensors to ``device``; ``dtype`` applies to k/v/conv_buf (states stay float32)."""
        for layer in self.layers:
            for name in _FIELDS:
                t = getattr(layer, name)
                if t is None:
                    continue
                if dtype is not None and name != "state":
                    t = t.to(device=device, dtype=dtype)
                elif device is not None:
                    t = t.to(device=device)
                setattr(layer, name, t)
        return self

    def clone(self) -> "InferenceCache":
        """Deep copy (tensors cloned)."""
        new = InferenceCache(self.n_layers)
        new.seen_tokens = self.seen_tokens
        for src, dst in zip(self.layers, new.layers):
            for name in _FIELDS:
                t = getattr(src, name)
                setattr(dst, name, None if t is None else t.clone())
        return new

    copy = clone

    def reset(self) -> "InferenceCache":
        """Drop every cached tensor and reset ``seen_tokens`` to 0."""
        self.layers = [LayerCache() for _ in range(self.n_layers)]
        self.seen_tokens = 0
        return self
