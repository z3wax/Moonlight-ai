"""Pretraining token bins and the random-window loader (SPEC §6).

:func:`build_pretrain_tokens` streams the ``cfg.data.pretrain`` sources (``{source, weight,
max_chars}``) with a weighted round-robin (deficit scheduling: the source with the lowest
``chars_used / weight`` goes next) inside per-source character budgets, cleans and filters every
document (``clean_code`` for code sources, ``clean_doc`` + ``good_doc`` for prose, exact dedup),
tokenizes in chunks of 512 documents with ``tokenizer.encode_batch``, appends ``<|eos|>`` after each
document and writes flat token files::

    out_dir/train.bin  out_dir/train.json   # uint16 if vocab_size <= 65535 else uint32
    out_dir/val.bin    out_dir/val.json     # every 200th document (0.5%) goes to validation

The ``.json`` metadata holds ``dtype``, ``n_tokens``, ``vocab_size``, ``eos_id`` and build stats; it
is what ``Path(bin).with_suffix(".json")`` resolves to. Files are written to ``*.part`` and renamed
atomically; an existing build with the same signature (sources, budgets, tokenizer) is reused.

Offline mode (``cfg.data.offline``) - or when every online source fails - uses the bundled assets
(``moonlight/assets/sample_uz.txt``, ``sample_en.txt``, ``sample_code.py``) repeated with shuffled
order, plus synthetic conversations rendered as plain text.

Budgets: ``max_chars`` per source, multiplied by ``cfg.data.data_scale`` when such an attribute
exists (it is not part of the SPEC config, so normally the factor is 1).

:class:`PretrainLoader` samples random windows from the memory-mapped bin (moved to the device as an
int32 tensor when the file is < 2 GB). torch is imported lazily (module import is torch-free).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

from ..tokenizer import EOS_ID
from .cleaning import clean_code, clean_doc, dedup_exact, good_doc

__all__ = ["build_pretrain_tokens", "PretrainLoader", "offline_documents", "bin_dtype", "read_meta",
           "ENCODE_CHUNK", "VAL_EVERY", "SMALL_FILE_BYTES"]

ENCODE_CHUNK = 512             # documents per encode_batch call
VAL_EVERY = 200                # every 200th document -> validation (0.5%)
SMALL_FILE_BYTES = 2 * 1024 ** 3
OFFLINE_ASSET_CHARS = 2_000_000   # asset characters (repeated) in offline mode
OFFLINE_SYNTHETIC_CONVS = 3000    # synthetic conversations rendered as text in offline mode
_MIN_DOC_CHARS = {"uz": 200, "en": 200, "multi": 200}


def _logger() -> Any:
    from ..utils import get_logger
    return get_logger("moonlight.data")


def bin_dtype(vocab_size: int) -> np.dtype:
    """``uint16`` when every id fits (vocab_size <= 65535, SPEC §6), else ``uint32``."""
    return np.dtype(np.uint16) if int(vocab_size) <= 65535 else np.dtype(np.uint32)


def read_meta(bin_path: Any) -> Dict[str, Any]:
    """Metadata JSON next to a ``.bin`` file (``{}`` if missing/unreadable)."""
    meta = Path(bin_path).with_suffix(".json")
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _assets_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "assets"


def _get(entry: Any, key: str, default: Any = None) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(key, default)
    return getattr(entry, key, default)


# ============================================================================================ documents

def _split_prose(text: str, target: int = 1500) -> List[str]:
    """Split a long text into ~``target``-char documents at blank lines."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    docs, cur, n = [], [], 0
    for p in paras:
        cur.append(p)
        n += len(p)
        if n >= target:
            docs.append("\n\n".join(cur))
            cur, n = [], 0
    if cur:
        docs.append("\n\n".join(cur))
    return docs


def offline_documents(seed: int = 0, asset_chars: int = OFFLINE_ASSET_CHARS,
                      synthetic_convs: int = OFFLINE_SYNTHETIC_CONVS, name: str = "Moonlight") -> List[Tuple[str, str]]:
    """``(source_label, text)`` documents for offline pretraining: bundled assets repeated (order
    shuffled each repeat) up to ``asset_chars`` characters, plus synthetic conversations as text."""
    from .synthetic import synthetic_texts

    d = _assets_dir()
    base: List[Tuple[str, str]] = []
    for fname, label in (("sample_uz.txt", "asset-uz"), ("sample_en.txt", "asset-en"), ("sample_code.py", "asset-code")):
        p = d / fname
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        if label == "asset-code":
            base.append((label, text))
        else:
            base += [(label, t) for t in _split_prose(text)]
    rng = random.Random(f"offline:{seed}")
    docs: List[Tuple[str, str]] = []
    total = sum(len(t) for _, t in base)
    if total:
        repeats = max(1, int(math.ceil(asset_chars / total)))
        for _ in range(repeats):
            order = list(base)
            rng.shuffle(order)
            docs += order
    if synthetic_convs > 0:
        docs += [("synthetic", t) for t in synthetic_texts(synthetic_convs, seed, name)]
    rng.shuffle(docs)
    return docs


def _clean(text: str, lang: str) -> Optional[str]:
    if lang == "code":
        return clean_code(text)
    doc = clean_doc(text, lang)
    return doc if good_doc(doc, _MIN_DOC_CHARS.get(lang, 200), lang if lang in ("uz", "en") else None) else None


def _source_stream(name: str, cache_dir: Path, budget: float) -> Iterator[str]:
    """Cleaned, deduplicated documents of one source (lazy shard downloads), until ``budget`` chars."""
    from .adapters import to_text
    from .download import iter_source_rows
    from .sources import get_source

    spec = get_source(name)

    def raw() -> Iterator[str]:
        for row in iter_source_rows(spec, cache_dir / "raw"):
            text = to_text(name, row)
            if text:
                cleaned = _clean(text, spec.language)
                if cleaned:
                    yield cleaned

    used = 0
    for doc in dedup_exact(raw()):
        if used >= budget:
            return
        if used + len(doc) > budget and used > 0:
            doc = doc[: int(budget - used)]
        used += len(doc)
        yield doc


def _weighted_round_robin(streams: Dict[str, Tuple[Iterator[str], float]]) -> Iterator[Tuple[str, str]]:
    """Interleave ``{name: (iterator, weight)}`` so chars consumed stay proportional to weights."""
    used = {k: 0.0 for k in streams}
    active = dict(streams)
    while active:
        name = min(active, key=lambda k: (used[k] / max(active[k][1], 1e-12), k))
        it = active[name][0]
        try:
            doc = next(it)
        except StopIteration:
            del active[name]
            continue
        except Exception as e:  # network / format failure of one source: drop it, keep the others
            _logger().warning("pretrain source %s failed (%s: %s); continuing without it", name, type(e).__name__, e)
            del active[name]
            continue
        used[name] += len(doc)
        yield name, doc


def _signature(cfg: Any, tokenizer: Any, offline: bool, scale: float) -> str:
    entries = [{"source": str(_get(e, "source")), "weight": float(_get(e, "weight", 1.0) or 0.0),
                "max_chars": float(_get(e, "max_chars", 0) or 0)} for e in (cfg.data.pretrain or [])]
    payload = {"entries": entries, "offline": offline, "scale": scale, "languages": list(cfg.data.languages),
               "tok": hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()[:16], "v": 2}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


class _BinWriter:
    """Append token arrays to ``<path>.part``; ``close()`` renames atomically and writes metadata."""

    def __init__(self, path: Path, dtype: np.dtype) -> None:
        self.path = path
        self.part = path.with_name(path.name + ".part")
        self.dtype = dtype
        self.fh = open(self.part, "wb")
        self.n_tokens = 0

    def write(self, ids: np.ndarray) -> None:
        arr = np.asarray(ids, dtype=self.dtype)
        self.fh.write(arr.tobytes())
        self.n_tokens += int(arr.size)

    def close(self, meta: Dict[str, Any]) -> None:
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.fh.close()
        os.replace(self.part, self.path)
        meta_path = self.path.with_suffix(".json")
        tmp = meta_path.with_name(meta_path.name + ".part")
        tmp.write_text(json.dumps({**meta, "n_tokens": self.n_tokens}, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, meta_path)

    def abort(self) -> None:
        try:
            self.fh.close()
            self.part.unlink()
        except OSError:
            pass


def build_pretrain_tokens(cfg: Any, tokenizer: Any, out_dir: Any) -> Tuple[Path, Path, Dict[str, Any]]:
    """Build ``train.bin`` / ``val.bin`` (+ ``.json`` metadata) under ``out_dir``.

    Args:
        cfg: :class:`moonlight.config.Config` (uses ``data.*`` and ``train.seed``).
        tokenizer: :class:`moonlight.tokenizer.MoonlightTokenizer`.
        out_dir: output directory (created).

    Returns ``(train_bin, val_bin, stats)`` where ``stats`` has ``train_tokens``, ``val_tokens``,
    ``docs``, ``chars``, ``per_source`` ({name: {docs, chars, tokens}}), ``dtype``, ``offline``.
    """
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    train_bin, val_bin = out / "train.bin", out / "val.bin"
    offline = bool(cfg.data.offline)
    scale = float(getattr(cfg.data, "data_scale", 1.0) or 1.0)
    sig = _signature(cfg, tokenizer, offline, scale)
    meta = read_meta(train_bin)
    if train_bin.is_file() and val_bin.is_file() and meta.get("signature") == sig:
        _logger().info("pretrain bins up to date: %s (%s tokens)", train_bin, meta.get("n_tokens"))
        return train_bin, val_bin, dict(meta.get("stats", {}), train_tokens=meta.get("n_tokens"),
                                        val_tokens=read_meta(val_bin).get("n_tokens"))

    seed = int(getattr(cfg.train, "seed", 0))
    dtype = bin_dtype(tokenizer.vocab_size)
    docs_iter: Iterator[Tuple[str, str]]
    if offline:
        docs_iter = iter(offline_documents(seed, name=str(getattr(cfg.chat, "name", "Moonlight"))))
    else:
        docs_iter = _online_documents(cfg, scale)

    t0 = time.perf_counter()
    train_w, val_w = _BinWriter(train_bin, dtype), _BinWriter(val_bin, dtype)
    per_source: Dict[str, Dict[str, int]] = {}
    n_docs = n_chars = 0
    buf: List[Tuple[str, str]] = []
    last_log = time.perf_counter()

    def flush() -> None:
        nonlocal n_docs
        if not buf:
            return
        # Raw corpus text may contain literal special-token strings (e.g. "<|end|>" in web pages
        # quoting other chat templates); they must stay plain text, not control ids. Only our own
        # synthetic renderings may use real control tokens (e.g. <|think|> ... <|/think|>).
        encoded: List[Any] = [None] * len(buf)
        for synthetic in (False, True):
            idx = [i for i, (src, _) in enumerate(buf) if (src == "synthetic") == synthetic]
            if idx:
                got = tokenizer.encode_batch([buf[i][1] for i in idx], allow_special=synthetic)
                for i, ids in zip(idx, got):
                    encoded[i] = ids
        for (src, text), ids in zip(buf, encoded):
            row = np.empty(len(ids) + 1, dtype=np.int64)
            row[:-1] = ids
            row[-1] = EOS_ID
            n_docs += 1
            (val_w if n_docs % VAL_EVERY == 0 else train_w).write(row)
            st = per_source.setdefault(src, {"docs": 0, "chars": 0, "tokens": 0})
            st["docs"] += 1
            st["chars"] += len(text)
            st["tokens"] += len(row)
        buf.clear()

    try:
        for src, text in docs_iter:
            buf.append((src, text))
            n_chars += len(text)
            if len(buf) >= ENCODE_CHUNK:
                flush()
                if time.perf_counter() - last_log > 30:
                    last_log = time.perf_counter()
                    _logger().info("pretrain tokens: %d docs, %.1fM chars, %.1fM train tokens", n_docs,
                                   n_chars / 1e6, train_w.n_tokens / 1e6)
        flush()
        if n_docs == 0 and not offline:
            _logger().warning("no online pretraining data could be read; falling back to bundled assets + "
                              "synthetic text (offline corpus)")
            for src, text in offline_documents(seed):
                buf.append((src, text))
                n_chars += len(text)
                if len(buf) >= ENCODE_CHUNK:
                    flush()
            flush()
            offline = True
        if val_w.n_tokens == 0 and train_w.n_tokens > 0:
            _logger().warning("fewer than %d documents: validation bin is empty", VAL_EVERY)
        stats: Dict[str, Any] = {"docs": n_docs, "chars": n_chars, "per_source": per_source, "dtype": dtype.name,
                                 "vocab_size": int(tokenizer.vocab_size), "offline": offline,
                                 "seconds": round(time.perf_counter() - t0, 2),
                                 "train_tokens": train_w.n_tokens, "val_tokens": val_w.n_tokens}
        common = {"dtype": dtype.name, "vocab_size": int(tokenizer.vocab_size), "eos_id": EOS_ID, "signature": sig}
        train_w.close({**common, "split": "train", "stats": stats})
        val_w.close({**common, "split": "val"})
    except BaseException:
        train_w.abort()
        val_w.abort()
        raise
    _logger().info("pretrain bins: %d docs, %d train / %d val tokens (%s) in %.1fs", n_docs, stats["train_tokens"],
                   stats["val_tokens"], dtype.name, stats["seconds"])
    return train_bin, val_bin, stats


def _online_documents(cfg: Any, scale: float) -> Iterator[Tuple[str, str]]:
    from .sources import get_source

    langs = set(cfg.data.languages or [])
    streams: Dict[str, Tuple[Iterator[str], float]] = {}
    for e in cfg.data.pretrain or []:
        name = str(_get(e, "source", "") or "")
        weight = float(_get(e, "weight", 1.0) or 0.0)
        max_chars = float(_get(e, "max_chars", 0) or 0) * scale
        if not name or weight <= 0 or max_chars <= 0:
            continue
        try:
            spec = get_source(name)
        except KeyError as ex:
            _logger().warning("%s", ex)
            continue
        if spec.language in ("uz", "en") and langs and spec.language not in langs:
            _logger().info("pretrain source %s skipped (language %s not in data.languages)", name, spec.language)
            continue
        streams[name] = (_source_stream(name, cfg.data.cache_path, max_chars), weight)
    return _weighted_round_robin(streams)


# ============================================================================================ loader

class PretrainLoader:
    """Random contiguous windows from a token bin.

    Args:
        path: ``.bin`` file (dtype from the ``.json`` metadata next to it, default uint16).
        seq_len: default window length ``L``; batch_size: default ``B``.
        device: torch device for returned tensors.
        seed: RNG seed; ``rank``/``world``: each rank draws from its own stream (seed offset by rank).

    ``next(seq_len=None, batch_size=None)`` -> ``(x, y)`` int64 tensors ``[B, L]`` with
    ``y[:, t] == x[:, t + 1]`` (next-token targets; documents are separated by ``<|eos|>``). Files
    smaller than 2 GB are copied to ``device`` once as an int32 tensor; larger files are read
    through ``np.memmap`` per batch. Raises ``ValueError`` if the bin has fewer than ``L + 1`` tokens.
    """

    def __init__(self, path: Any, seq_len: int, batch_size: int, device: Any = "cpu", seed: int = 0,
                 rank: int = 0, world: int = 1) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"token bin not found: {self.path}")
        meta = read_meta(self.path)
        self.dtype = np.dtype(meta.get("dtype", "uint16"))
        self.data = np.memmap(self.path, dtype=self.dtype, mode="r")
        self.n_tokens = int(self.data.shape[0])
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.device = device
        self.rank, self.world = int(rank), max(1, int(world))
        self.rng = np.random.default_rng([int(seed), self.rank, 104729])
        self._tensor: Any = None
        self._on_device = self.path.stat().st_size < SMALL_FILE_BYTES
        self._check(self.seq_len)

    def __len__(self) -> int:
        return self.n_tokens

    def _check(self, seq_len: int) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if self.n_tokens < seq_len + 1:
            raise ValueError(f"{self.path} has {self.n_tokens} tokens; need at least seq_len + 1 = {seq_len + 1}")

    def _device_tensor(self) -> Any:
        if self._tensor is None:
            import torch

            arr = np.asarray(self.data, dtype=np.int32)  # uint16/uint32 -> int32 (ids < 2^31)
            self._tensor = torch.from_numpy(arr).to(torch.device(self.device))
        return self._tensor

    def next_numpy(self, seq_len: Optional[int] = None, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Same windows as :meth:`next` as int64 numpy arrays (no torch)."""
        L = int(seq_len or self.seq_len)
        B = int(batch_size or self.batch_size)
        self._check(L)
        starts = self.rng.integers(0, self.n_tokens - L, size=B)
        win = np.stack([np.asarray(self.data[s:s + L + 1], dtype=np.int64) for s in starts])
        return np.ascontiguousarray(win[:, :-1]), np.ascontiguousarray(win[:, 1:])

    def next(self, seq_len: Optional[int] = None, batch_size: Optional[int] = None) -> Tuple[Any, Any]:
        """``(x, y)`` int64 tensors ``[B, L]`` on ``device``."""
        import torch

        L = int(seq_len or self.seq_len)
        B = int(batch_size or self.batch_size)
        self._check(L)
        starts = self.rng.integers(0, self.n_tokens - L, size=B)
        if self._on_device:
            data = self._device_tensor()
            idx = torch.as_tensor(starts, dtype=torch.long, device=data.device)[:, None] + torch.arange(
                L + 1, device=data.device)
            win = data[idx].long()
            return win[:, :-1].contiguous(), win[:, 1:].contiguous()
        win = np.stack([np.asarray(self.data[s:s + L + 1], dtype=np.int64) for s in starts])
        t = torch.from_numpy(win)
        dev = torch.device(self.device)
        if dev.type == "cuda":
            t = t.pin_memory().to(dev, non_blocking=True)
        else:
            t = t.to(dev)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def state_dict(self) -> Dict[str, Any]:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
