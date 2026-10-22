"""Supervised fine-tuning and preference data: encoding, packing and loaders (SPEC §6, §13.4).

Module level is numpy-only (torch is imported lazily inside the loader methods), so encoding and
packing work - and are tested - without torch.

Shapes / dtypes
---------------
* :func:`encode_conversations` -> list of ``(ids int32 [T], mask uint8 [T])`` (one per conversation;
  ``mask[t] == 1`` iff token ``t`` is a training target, from ``tokenizer.encode_chat``).
* :func:`pack` -> ``ids int32 [N, seq_len + 1]``, ``mask uint8 [N, seq_len + 1]``; rows are filled
  greedily with whole conversations and right-padded with ``PAD_ID`` (mask 0).
* :class:`SFTLoader` ``.next()`` -> ``x int64 [B, seq_len]`` = ``ids[:, :-1]`` and
  ``y int64 [B, seq_len]`` = ``ids[:, 1:]`` with ``-100`` where ``mask[:, 1:] == 0``.
* :func:`encode_preference` -> dict of ``int32``/``uint8`` arrays of length <= ``seq_len + 1``;
  :class:`PreferenceLoader` ``.next()`` -> ``(c_x, c_y, r_x, r_y)`` int64 ``[B, T]`` with a common
  ``T`` (right-padded with ``PAD_ID`` / ``-100``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..tokenizer import BOS_ID, PAD_ID
from .adapters import split_tools

__all__ = [
    "IGNORE_INDEX", "encode_conversations", "pack", "split_val", "SFTLoader", "encode_preference",
    "PreferenceLoader",
]

IGNORE_INDEX = -100

Example = Tuple[np.ndarray, np.ndarray]


def _logger() -> Any:
    from ..utils import get_logger
    return get_logger("moonlight.data")


# =============================================================================================== encode

def encode_conversations(convs: Iterable[Sequence[Mapping[str, Any]]], tokenizer: Any, n_image_tokens: int = 0,
                         max_len: Optional[int] = None) -> List[Example]:
    """Tokenize conversations with ``tokenizer.encode_chat`` (SPEC §5 template).

    Tool schemas stored in a leading system message's ``"tools"`` key are passed as ``tools=``.
    Conversations that fail to encode (malformed messages) or have no trainable token are skipped
    (counted in a warning). ``max_len`` optionally truncates each example (keeps the beginning).

    Returns a list of ``(ids int32 [T], mask uint8 [T])``.
    """
    out: List[Example] = []
    bad = empty = 0
    for conv in convs:
        try:
            msgs, tools = split_tools(conv)
            ids, mask = tokenizer.encode_chat(msgs, tools=tools, n_image_tokens=n_image_tokens)
        except (TypeError, ValueError, KeyError, AttributeError):
            bad += 1
            continue
        if max_len is not None and len(ids) > max_len:
            ids, mask = ids[:max_len], mask[:max_len]
        if not any(mask):
            empty += 1
            continue
        out.append((np.asarray(ids, dtype=np.int32), np.asarray(mask, dtype=np.uint8)))
    if bad or empty:
        _logger().warning("encode_conversations: skipped %d malformed and %d untrainable conversation(s)", bad, empty)
    return out


# =============================================================================================== pack

def pack(examples: Sequence[Tuple[Any, Any]], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy packing of examples into rows of ``seq_len + 1`` tokens.

    Examples are taken in order; an example is appended to the current row if it fits, otherwise
    the row is closed (right-padded with ``PAD_ID``, mask 0) and a new row starts. An example
    longer than ``seq_len + 1`` gets a row of its own and is truncated to its first ``seq_len + 1``
    tokens (the prompt and the beginning of the answer are kept). Every row therefore starts at a
    conversation boundary (``<|bos|>``).

    Returns ``(ids int32 [N, seq_len + 1], mask uint8 [N, seq_len + 1])`` (``N == 0`` if no example).
    """
    seq_len = int(seq_len)
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    row_len = seq_len + 1
    rows: List[List[Tuple[np.ndarray, np.ndarray]]] = []
    cur: List[Tuple[np.ndarray, np.ndarray]] = []
    used = 0
    for ids, mask in examples:
        ids = np.asarray(ids, dtype=np.int32)
        mask = np.asarray(mask, dtype=np.uint8)
        if ids.shape != mask.shape or ids.ndim != 1:
            raise ValueError(f"example ids/mask must be 1-D of equal length, got {ids.shape} / {mask.shape}")
        n = len(ids)
        if n == 0:
            continue
        if n > row_len:
            ids, mask, n = ids[:row_len], mask[:row_len], row_len
        if used + n > row_len and cur:
            rows.append(cur)
            cur, used = [], 0
        cur.append((ids, mask))
        used += n
    if cur:
        rows.append(cur)
    out_ids = np.full((len(rows), row_len), PAD_ID, dtype=np.int32)
    out_mask = np.zeros((len(rows), row_len), dtype=np.uint8)
    for r, parts in enumerate(rows):
        pos = 0
        for ids, mask in parts:
            out_ids[r, pos:pos + len(ids)] = ids
            out_mask[r, pos:pos + len(ids)] = mask
            pos += len(ids)
    return out_ids, out_mask


def split_val(ids: np.ndarray, mask: np.ndarray, frac: float = 0.02, seed: int = 0
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random row split ``(train_ids, train_mask, val_ids, val_mask)``.

    ``max(1, round(frac * N))`` rows go to validation when ``N >= 2`` (never all rows); with a single
    row the validation set is a copy of it so evaluation still works. Deterministic for ``seed``.
    """
    ids = np.asarray(ids)
    mask = np.asarray(mask)
    n = len(ids)
    if n == 0:
        return ids, mask, ids[:0], mask[:0]
    if n == 1:
        return ids, mask, ids.copy(), mask.copy()
    n_val = min(n - 1, max(1, int(round(float(frac) * n))))
    perm = np.random.default_rng(seed).permutation(n)
    val_idx = np.sort(perm[:n_val])
    tr_idx = np.sort(perm[n_val:])
    return ids[tr_idx], mask[tr_idx], ids[val_idx], mask[val_idx]


def _targets(ids: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rows ``[B, L+1]`` -> ``x [B, L]`` int64, ``y [B, L]`` int64 (``-100`` where not trained)."""
    x = ids[:, :-1].astype(np.int64)
    y = ids[:, 1:].astype(np.int64)
    y[mask[:, 1:] == 0] = IGNORE_INDEX
    return x, y


def _to_device(arrays: Sequence[np.ndarray], device: Any) -> Tuple[Any, ...]:
    import torch  # lazy

    dev = torch.device(device) if device is not None else torch.device("cpu")
    out = []
    for a in arrays:
        t = torch.from_numpy(np.ascontiguousarray(a))
        if dev.type == "cuda":
            t = t.pin_memory().to(dev, non_blocking=True)
        elif dev.type != "cpu":
            t = t.to(dev)
        out.append(t)
    return tuple(out)


class _EpochSampler:
    """Shuffled row indices, reshuffled every epoch; ``rank``/``world`` take disjoint strided shards."""

    def __init__(self, n: int, seed: int, rank: int = 0, world: int = 1) -> None:
        if n <= 0:
            raise ValueError("no rows to sample from")
        self.n = n
        self.rank, self.world = int(rank), max(1, int(world))
        self.rng = np.random.default_rng([int(seed), 7919])
        self.epoch = 0
        self.pos = 0
        self.order = self._new_order()

    def _new_order(self) -> np.ndarray:
        perm = self.rng.permutation(self.n)  # same permutation on every rank (same seed) ...
        shard = perm[self.rank::self.world] if self.n >= self.world else perm  # ... disjoint shards
        return shard

    def take(self, k: int) -> np.ndarray:
        idx = []
        while len(idx) < k:
            if self.pos >= len(self.order):
                self.epoch += 1
                self.pos = 0
                self.order = self._new_order()
            take = min(k - len(idx), len(self.order) - self.pos)
            idx.extend(self.order[self.pos:self.pos + take].tolist())
            self.pos += take
        return np.asarray(idx, dtype=np.int64)

    @property
    def progress(self) -> float:
        """Epochs consumed (fractional) by this rank."""
        return self.epoch + self.pos / max(1, len(self.order))

    def state_dict(self) -> Dict[str, Any]:
        return {"epoch": self.epoch, "pos": self.pos, "order": self.order.tolist(), "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.epoch, self.pos = int(state["epoch"]), int(state["pos"])
        self.order = np.asarray(state["order"], dtype=np.int64)
        self.rng.bit_generator.state = state["rng"]


class SFTLoader:
    """Random batches of packed SFT rows.

    Args:
        ids: ``int32 [N, L+1]`` packed rows (from :func:`pack`); mask: ``uint8 [N, L+1]``.
        batch_size: rows per batch ``B``.
        device: torch device for the returned tensors (e.g. ``"cuda"``; pinned + non-blocking copy).
        seed: shuffling seed. ``rank``/``world``: optional data-parallel sharding of rows.

    ``next()`` returns ``(x, y)`` LongTensors ``[B, L]``: ``x = ids[:, :-1]``, ``y = ids[:, 1:]`` with
    ``-100`` where ``mask[:, 1:] == 0``. Rows are visited in a fresh random order every epoch;
    ``epoch`` is the (fractional) number of epochs consumed.
    """

    def __init__(self, ids: np.ndarray, mask: np.ndarray, batch_size: int, device: Any = "cpu", seed: int = 0,
                 rank: int = 0, world: int = 1) -> None:
        self.ids = np.asarray(ids)
        self.mask = np.asarray(mask)
        if self.ids.ndim != 2 or self.ids.shape != self.mask.shape:
            raise ValueError(f"ids/mask must be 2-D with equal shapes, got {self.ids.shape} / {self.mask.shape}")
        if self.ids.shape[1] < 2:
            raise ValueError("rows must have at least 2 tokens (seq_len + 1)")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.device = device
        self.sampler = _EpochSampler(len(self.ids), seed, rank, world)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def seq_len(self) -> int:
        return int(self.ids.shape[1] - 1)

    @property
    def epoch(self) -> float:
        return self.sampler.progress

    def next_numpy(self, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Like :meth:`next` but returns int64 numpy arrays (no torch needed)."""
        idx = self.sampler.take(int(batch_size or self.batch_size))
        return _targets(self.ids[idx], self.mask[idx])

    def next(self, batch_size: Optional[int] = None) -> Tuple[Any, Any]:
        """``(x, y)`` int64 tensors ``[B, L]`` on ``device``."""
        return _to_device(self.next_numpy(batch_size), self.device)  # type: ignore[return-value]

    def state_dict(self) -> Dict[str, Any]:
        return self.sampler.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.sampler.load_state_dict(state)


# =============================================================================================== preference

def _encode_split(msgs: Sequence[Mapping[str, Any]], tokenizer: Any) -> Tuple[List[int], List[int], int]:
    """Encode a conversation ending with an assistant reply; returns ``(ids, mask, response_start)``
    where ``mask`` is 1 only on the FINAL assistant reply (earlier turns are context)."""
    msgs, tools = split_tools(msgs)
    if not msgs or msgs[-1].get("role") != "assistant":
        raise ValueError("preference conversations must end with an assistant message")
    ids, mask = tokenizer.encode_chat(msgs, tools=tools)
    prompt_ids, _ = tokenizer.encode_chat(msgs[:-1], tools=tools, add_generation_prompt=True)
    if list(ids[:len(prompt_ids)]) == list(prompt_ids):
        start = len(prompt_ids)
    else:  # defensive fallback: last contiguous run of trainable tokens
        start = len(mask)
        while start > 0 and mask[start - 1]:
            start -= 1
    resp_mask = [0] * start + list(mask[start:])
    return list(ids), resp_mask, start


def _fit(ids: List[int], mask: List[int], start: int, max_len: int) -> Tuple[List[int], List[int]]:
    """Left-truncate the prompt (keeping ``<|bos|>``) so the sequence fits ``max_len``; the response is
    only cut (from the right) when it alone does not fit."""
    if len(ids) <= max_len:
        return ids, mask
    prompt, p_mask = ids[:start], mask[:start]
    resp, r_mask = ids[start:], mask[start:]
    min_prompt = min(len(prompt), max(2, max_len // 4))
    if len(resp) > max_len - min_prompt:
        keep = max_len - min_prompt
        resp, r_mask = resp[:keep], r_mask[:keep]
    room = max_len - len(resp)
    if len(prompt) > room:
        if room >= 2 and prompt[:1] == [BOS_ID]:
            prompt = [BOS_ID] + prompt[len(prompt) - (room - 1):]
            p_mask = [0] + p_mask[len(p_mask) - (room - 1):]
        else:
            prompt, p_mask = prompt[len(prompt) - room:], p_mask[len(p_mask) - room:]
    return prompt + resp, p_mask + r_mask


def encode_preference(chosen_msgs: Sequence[Mapping[str, Any]], rejected_msgs: Sequence[Mapping[str, Any]],
                      tokenizer: Any, seq_len: int) -> Optional[Dict[str, np.ndarray]]:
    """Encode one preference pair for DPO.

    Both conversations end with an assistant answer (the shared prompt precedes it). Each is encoded
    with the chat template; its mask is 1 only on the final answer (tokens + closing ``<|end|>``),
    so summed log-probs cover the response only. Sequences longer than ``seq_len + 1`` tokens are
    truncated from the LEFT of the prompt (``<|bos|>`` kept); the response is kept whole unless it
    alone exceeds the budget. Returns None if either side has no response token left.

    Returns ``{"chosen_ids": int32 [Tc], "chosen_mask": uint8 [Tc], "rejected_ids": int32 [Tr],
    "rejected_mask": uint8 [Tr]}`` with ``Tc, Tr <= seq_len + 1``.
    """
    max_len = int(seq_len) + 1
    if max_len < 4:
        raise ValueError(f"seq_len too small for preference data: {seq_len}")
    out: Dict[str, np.ndarray] = {}
    for key, msgs in (("chosen", chosen_msgs), ("rejected", rejected_msgs)):
        ids, mask, start = _encode_split(msgs, tokenizer)
        ids, mask = _fit(ids, mask, start, max_len)
        if not any(mask[1:]):
            return None
        out[f"{key}_ids"] = np.asarray(ids, dtype=np.int32)
        out[f"{key}_mask"] = np.asarray(mask, dtype=np.uint8)
    return out


class PreferenceLoader:
    """Random batches of encoded preference pairs (from :func:`encode_preference`).

    ``next()`` returns ``(c_x, c_y, r_x, r_y)`` int64 tensors ``[B, T]`` where ``T`` = longest
    (sequence - 1) in the batch over BOTH chosen and rejected; ``x`` is right-padded with ``PAD_ID``
    and ``y`` (next tokens) is ``-100`` on padding and wherever the mask of the target token is 0.
    """

    def __init__(self, pairs: Sequence[Mapping[str, Any]], batch_size: int, device: Any = "cpu", seed: int = 0,
                 rank: int = 0, world: int = 1) -> None:
        self.pairs = [p for p in pairs if p]
        if not self.pairs:
            raise ValueError("PreferenceLoader: no preference pairs")
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.sampler = _EpochSampler(len(self.pairs), seed, rank, world)

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def epoch(self) -> float:
        return self.sampler.progress

    def next_numpy(self, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.sampler.take(int(batch_size or self.batch_size))
        batch = [self.pairs[i] for i in idx]
        T = max(max(len(p["chosen_ids"]), len(p["rejected_ids"])) for p in batch) - 1
        B = len(batch)
        out = []
        for side in ("chosen", "rejected"):
            x = np.full((B, T), PAD_ID, dtype=np.int64)
            y = np.full((B, T), IGNORE_INDEX, dtype=np.int64)
            for b, p in enumerate(batch):
                ids = np.asarray(p[f"{side}_ids"], dtype=np.int64)
                mask = np.asarray(p[f"{side}_mask"], dtype=np.uint8)
                n = len(ids) - 1
                x[b, :n] = ids[:-1]
                tgt = ids[1:].copy()
                tgt[mask[1:] == 0] = IGNORE_INDEX
                y[b, :n] = tgt
            out += [x, y]
        return out[0], out[1], out[2], out[3]

    def next(self, batch_size: Optional[int] = None) -> Tuple[Any, Any, Any, Any]:
        """``(c_x, c_y, r_x, r_y)`` int64 tensors on ``device``."""
        return _to_device(self.next_numpy(batch_size), self.device)  # type: ignore[return-value]

    def state_dict(self) -> Dict[str, Any]:
        return self.sampler.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.sampler.load_state_dict(state)
