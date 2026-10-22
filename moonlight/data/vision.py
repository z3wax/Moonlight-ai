"""Image-caption data: cached encoder features + image conversations + loader (SPEC §6).

:func:`build_image_dataset` turns the ``cfg.data.vision`` sources (``{source, max_images}``, e.g.
``flickr8k`` / ``flickr30k``) into

* image features from a frozen :class:`moonlight.model.vision.VisionEncoder`, cached as float16
  ``.npy`` shards ``features_00000.npy`` ``[k, n_image_tokens, vision_dim]`` (<= 1024 images each);
* captioning conversations ``[{"role": "user", "images": 1, "content": prompt}, {"role":
  "assistant", "content": caption}]`` - the image placeholders sit at the START of the user turn
  (the chat template puts ``n_image_tokens`` ``<|image|>`` tokens right after ``<|user|>``);
  captions are English; when ``"uz"`` is in ``cfg.data.languages`` the first caption of up to
  ``translate_max`` images per source (entry key, default 2000; ``translate: false`` disables it) is
  translated en->uz with ``facebook/nllb-200-distilled-600M`` (``transformers``, imported lazily;
  failures only log a warning);
* packed token rows ``ids int32 [N, seq_len + 1]`` / ``mask uint8`` with a CSR image index
  (``row_ptr int64 [N + 1]``, ``row_images int64 [M]``): row ``r`` contains the images
  ``row_images[row_ptr[r]:row_ptr[r + 1]]`` in order of appearance.

Everything is cached in ``out_dir`` (``meta.json`` carries a signature of sources, encoder, token
count, tokenizer and seq_len), so a second call - e.g. on other ranks - only loads files and needs no
encoder. Offline (``cfg.data.offline``) or when no source can be read, a small deterministic set of
synthetic images (coloured shapes, captioned in Uzbek and English) is used.

:class:`ImageSFTLoader` ``.next()`` -> ``(x int64 [B, L], y int64 [B, L], image_features float32
[B_img, n_image_tokens, vision_dim])`` with ``B_img`` = number of images in the batch rows
(row-major order, matching the model's ``masked_scatter`` of ``<|image|>`` positions).

Module import is torch-free; torch is imported inside methods.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..tokenizer import IMAGE_ID, PAD_ID
from .sft import IGNORE_INDEX, _EpochSampler, _to_device

__all__ = ["ImageDataset", "ImageSFTLoader", "build_image_dataset", "synthetic_images", "translate_en_uz",
           "image_conversations", "SHARD_SIZE", "NLLB_MODEL"]

SHARD_SIZE = 1024
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
ENCODE_BATCH = 32

_PROMPTS_EN = ("Describe this image.", "What is shown in the picture?", "Write a short caption for this image.",
               "What do you see in this photo?", "Briefly describe the image.")
_PROMPTS_UZ = ("Rasmni tasvirlab bering.", "Rasmda nima tasvirlangan?", "Ushbu rasm uchun qisqa izoh yozing.",
               "Bu suratda nimani ko'ryapsiz?", "Rasmni qisqacha tasvirlab ber.")


def _logger() -> Any:
    from ..utils import get_logger
    return get_logger("moonlight.data")


def _get(entry: Any, key: str, default: Any = None) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(key, default)
    return getattr(entry, key, default)


# ============================================================================================ dataset

@dataclass
class ImageDataset:
    """Cached image-caption training data (see module docstring for shapes)."""

    out_dir: Path
    feature_files: List[Path]
    n_images: int
    n_image_tokens: int
    vision_dim: int
    ids: np.ndarray            # int32 [N, L + 1]
    mask: np.ndarray           # uint8 [N, L + 1]
    row_ptr: np.ndarray        # int64 [N + 1]
    row_images: np.ndarray     # int64 [M]
    val_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    stats: Dict[str, Any] = field(default_factory=dict)
    _shards: Optional[List[np.ndarray]] = field(default=None, repr=False)

    @property
    def n_rows(self) -> int:
        return int(len(self.ids))

    @property
    def train_rows(self) -> np.ndarray:
        all_rows = np.arange(self.n_rows, dtype=np.int64)
        if len(self.val_rows) == 0 or len(self.val_rows) >= self.n_rows:
            return all_rows
        keep = np.ones(self.n_rows, dtype=bool)
        keep[self.val_rows] = False
        return all_rows[keep]

    def shards(self) -> List[np.ndarray]:
        """Memory-mapped float16 feature shards."""
        if self._shards is None:
            self._shards = [np.load(p, mmap_mode="r") for p in self.feature_files]
        return self._shards

    def features(self, image_idx: Sequence[int]) -> np.ndarray:
        """float32 ``[len(image_idx), n_image_tokens, vision_dim]`` for global image indices."""
        shards = self.shards()
        out = np.empty((len(image_idx), self.n_image_tokens, self.vision_dim), dtype=np.float32)
        for k, i in enumerate(image_idx):
            s, o = divmod(int(i), SHARD_SIZE)
            out[k] = shards[s][o]
        return out

    def row_image_list(self, rows: Sequence[int]) -> List[int]:
        out: List[int] = []
        for r in rows:
            out.extend(self.row_images[self.row_ptr[r]:self.row_ptr[r + 1]].tolist())
        return out

    def loader(self, batch_size: int, device: Any = "cpu", seed: int = 0, split: str = "train",
               rank: int = 0, world: int = 1) -> "ImageSFTLoader":
        return ImageSFTLoader(self, batch_size, device, seed, split=split, rank=rank, world=world)


class ImageSFTLoader:
    """Random batches of packed image-caption rows.

    ``ImageSFTLoader(dataset, batch_size, device, seed, split="train", rank=0, world=1)``;
    ``next()`` -> ``(x, y, image_features)``: ``x``/``y`` int64 ``[B, L]`` (``y`` = next tokens,
    ``-100`` where not trained), ``image_features`` float32 ``[B_img, n_image_tokens, vision_dim]``
    for every ``<|image|>`` block in ``x`` in row-major order (``B_img`` may differ from ``B``).
    """

    def __init__(self, dataset: ImageDataset, batch_size: int, device: Any = "cpu", seed: int = 0,
                 split: str = "train", rank: int = 0, world: int = 1) -> None:
        if not isinstance(dataset, ImageDataset):
            raise TypeError(f"ImageSFTLoader expects an ImageDataset, got {type(dataset).__name__}")
        self.ds = dataset
        rows = dataset.val_rows if split == "val" and len(dataset.val_rows) else dataset.train_rows
        if len(rows) == 0:
            raise ValueError("image dataset has no rows")
        self.rows = np.asarray(rows, dtype=np.int64)
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.sampler = _EpochSampler(len(self.rows), seed, rank, world)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def epoch(self) -> float:
        return self.sampler.progress

    def next_numpy(self, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self.rows[self.sampler.take(int(batch_size or self.batch_size))]
        ids = self.ds.ids[rows]
        mask = self.ds.mask[rows]
        x = ids[:, :-1].astype(np.int64)
        y = ids[:, 1:].astype(np.int64)
        y[mask[:, 1:] == 0] = IGNORE_INDEX
        feats = self.ds.features(self.ds.row_image_list(rows))
        return x, y, feats

    def next(self, batch_size: Optional[int] = None) -> Tuple[Any, Any, Any]:
        return _to_device(self.next_numpy(batch_size), self.device)  # type: ignore[return-value]

    def state_dict(self) -> Dict[str, Any]:
        return self.sampler.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.sampler.load_state_dict(state)


# ============================================================================================ sources

_COLORS = [("red", "qizil", (220, 30, 30)), ("green", "yashil", (30, 160, 60)), ("blue", "ko'k", (30, 70, 210)),
           ("yellow", "sariq", (240, 210, 30)), ("black", "qora", (15, 15, 15)), ("white", "oq", (245, 245, 245)),
           ("orange", "to'q sariq", (245, 140, 20)), ("purple", "binafsha", (130, 40, 170))]
_SHAPES = [("circle", "doira"), ("square", "kvadrat"), ("triangle", "uchburchak")]


def synthetic_images(n: int = 256, seed: int = 0, size: int = 224) -> List[Tuple[np.ndarray, List[str], List[str]]]:
    """Deterministic coloured-shape images: ``(uint8 [size, size, 3], [en caption], [uz caption])``."""
    rng = random.Random(f"synthetic_images:{seed}")
    yy, xx = np.mgrid[0:size, 0:size]
    out = []
    for _ in range(int(n)):
        bg, fg = rng.sample(_COLORS, 2)
        shape_en, shape_uz = _pick(rng, _SHAPES)
        img = np.empty((size, size, 3), dtype=np.uint8)
        img[:] = bg[2]
        r = rng.randint(size // 6, size // 3)
        cx, cy = rng.randint(r, size - r), rng.randint(r, size - r)
        if shape_en == "circle":
            sel = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        elif shape_en == "square":
            sel = (abs(xx - cx) <= r) & (abs(yy - cy) <= r)
        else:  # upward triangle with apex at (cx, cy - r) and base at y = cy + r
            sel = (yy <= cy + r) & (yy >= cy - r) & (abs(xx - cx) <= (yy - (cy - r)) / 2.0)
        img[sel] = fg[2]
        en = f"A {fg[0]} {shape_en} on a {bg[0]} background."
        uz = f"{bg[1].capitalize()} fonda {fg[1]} {shape_uz}."
        out.append((img, [en], [uz]))
    return out


def _pick(rng: random.Random, seq: Sequence[Any]) -> Any:
    return seq[rng.randrange(len(seq))]


def _iter_source_images(name: str, cache_dir: Path, limit: int) -> Iterator[Tuple[Any, List[str]]]:
    """``(image, captions)`` from a vision source; the image is the raw HF value (``{"bytes", "path"}``)."""
    from .adapters import image_captions
    from .download import iter_source_rows

    n = 0
    for row in iter_source_rows(name, cache_dir / "raw"):
        img = row.get("image")
        caps = image_captions(name, row)
        if img is None or not caps:
            continue
        yield img, caps
        n += 1
        if n >= limit:
            return


def translate_en_uz(texts: Sequence[str], batch_size: int = 16, device: Any = None,
                    model_name: str = NLLB_MODEL, max_new_tokens: int = 128) -> List[str]:
    """Translate English -> Uzbek (Latin) with NLLB (``transformers``, imported lazily).

    Raises ImportError / OSError when the model is unavailable (callers treat that as optional).
    """
    import torch
    import transformers

    from .cleaning import norm_apostrophes

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = transformers.AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32).to(dev).eval()
    target = tok.convert_tokens_to_ids("uzn_Latn")
    out: List[str] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
            gen = model.generate(**enc, forced_bos_token_id=target, max_new_tokens=max_new_tokens, num_beams=2)
            out += [norm_apostrophes(t.strip()) for t in tok.batch_decode(gen, skip_special_tokens=True)]
    return out


def image_conversations(image_idx: int, captions_en: Sequence[str], captions_uz: Sequence[str],
                        rng: random.Random, per_image: int = 2) -> List[Tuple[int, List[Dict[str, Any]]]]:
    """Captioning conversations for one image: up to ``per_image`` English captions and every Uzbek one.
    Returns ``[(image_idx, conversation)]``."""
    out = []
    for cap in list(captions_en)[:per_image]:
        out.append((image_idx, [{"role": "user", "images": 1, "content": _pick(rng, _PROMPTS_EN)},
                                {"role": "assistant", "content": cap}]))
    for cap in captions_uz:
        out.append((image_idx, [{"role": "user", "images": 1, "content": _pick(rng, _PROMPTS_UZ)},
                                {"role": "assistant", "content": cap}]))
    return out


# ============================================================================================ build

def _signature(cfg: Any, tokenizer: Any, encoder_name: str, n_tok: int, seq_len: int) -> str:
    entries = [{k: _get(e, k) for k in ("source", "max_images", "translate", "translate_max")}
               for e in (cfg.data.vision or [])]
    payload = {"entries": entries, "offline": bool(cfg.data.offline), "langs": list(cfg.data.languages),
               "encoder": encoder_name, "n_tok": n_tok, "seq_len": seq_len,
               "tok": hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()[:16], "v": 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]


def _load_cached(out: Path, sig: str) -> Optional[ImageDataset]:
    meta_path = out / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta.get("signature") != sig:
        return None
    try:
        files = [out / f for f in meta["feature_files"]]
        if not all(f.is_file() for f in files):
            return None
        return ImageDataset(out, files, int(meta["n_images"]), int(meta["n_image_tokens"]), int(meta["vision_dim"]),
                            np.load(out / "ids.npy"), np.load(out / "mask.npy"), np.load(out / "row_ptr.npy"),
                            np.load(out / "row_images.npy"), np.load(out / "val_rows.npy"), meta.get("stats", {}))
    except (OSError, KeyError, ValueError):
        return None


class _FeatureWriter:
    """Buffers float16 features and writes ``features_XXXXX.npy`` shards of SHARD_SIZE images."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.buf: List[np.ndarray] = []
        self.files: List[Path] = []
        self.n = 0
        self.dim: Optional[int] = None

    def add(self, feats: np.ndarray) -> None:
        for f in feats:
            self.buf.append(np.asarray(f, dtype=np.float16))
            self.n += 1
            self.dim = int(f.shape[-1])
            if len(self.buf) == SHARD_SIZE:
                self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        path = self.out / f"features_{len(self.files):05d}.npy"
        tmp = path.with_name(path.stem + ".part.npy")
        np.save(tmp, np.stack(self.buf))
        os.replace(tmp, path)
        self.files.append(path)
        self.buf = []


def _encode_images(encoder: Any, images: List[Any]) -> Tuple[np.ndarray, List[int]]:
    """Encode a batch; returns (float32 [k, n_tok, D], indices of successfully encoded images)."""
    def to_np(t: Any) -> np.ndarray:
        if hasattr(t, "detach"):
            t = t.detach().float().cpu().numpy()
        return np.asarray(t, dtype=np.float32)

    try:
        return to_np(encoder.encode(images, batch_size=len(images))), list(range(len(images)))
    except Exception as e:  # noqa: BLE001 - a corrupt image fails the whole batch: retry one by one
        _logger().debug("batch encode failed (%s); retrying images individually", e)
    feats, ok = [], []
    for i, img in enumerate(images):
        try:
            feats.append(to_np(encoder.encode([img], batch_size=1))[0])
            ok.append(i)
        except Exception as e:  # noqa: BLE001
            _logger().warning("skipping undecodable image (%s: %s)", type(e).__name__, e)
    if not feats:
        return np.zeros((0, 0, 0), dtype=np.float32), []
    return np.stack(feats), ok


def build_image_dataset(cfg: Any, tokenizer: Any, encoder: Any, out_dir: Any,
                        seq_len: Optional[int] = None) -> Optional[ImageDataset]:
    """Build (or load from cache) the image-caption dataset; see the module docstring.

    Args:
        cfg: :class:`moonlight.config.Config` (``data.vision``, ``data.languages``, ``data.offline``,
            ``multimodal.n_image_tokens`` / ``vision_dim`` / ``vision_encoder``, ``train.seq_len``).
        tokenizer: :class:`moonlight.tokenizer.MoonlightTokenizer`.
        encoder: object with ``encode(images, batch_size) -> (B, n_tok, D)`` (``VisionEncoder``); may be
            None when the cache is complete.
        out_dir: cache directory.
        seq_len: row length minus one (default ``cfg.train.seq_len``).

    Returns an :class:`ImageDataset`, or None if no image could be encoded (or no encoder and no cache).
    Raises ``ValueError`` if the encoder's feature size differs from ``multimodal.vision_dim``.
    """
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    n_tok = int(cfg.multimodal.n_image_tokens)
    L = int(seq_len or cfg.train.seq_len)
    enc_name = str(cfg.multimodal.vision_encoder)  # the configured encoder defines the cache key
    sig = _signature(cfg, tokenizer, enc_name, n_tok, L)
    cached = _load_cached(out, sig)
    if cached is not None:
        return cached
    if encoder is None:
        _logger().warning("image dataset: no cache in %s and no encoder given", out)
        return None
    seed = int(getattr(cfg.train, "seed", 0))
    rng = random.Random(f"image_dataset:{seed}")
    want_uz = "uz" in [str(x) for x in (cfg.data.languages or [])]

    writer = _FeatureWriter(out)
    caps_en: List[List[str]] = []
    caps_uz: List[List[str]] = []
    per_source: Dict[str, int] = {}
    to_translate: List[Tuple[int, str]] = []

    def consume(label: str, items: Iterator[Tuple[Any, List[str], List[str]]], translate_max: int) -> None:
        batch: List[Tuple[Any, List[str], List[str]]] = []

        def flush() -> None:
            if not batch:
                return
            feats, ok = _encode_images(encoder, [b[0] for b in batch])
            if ok:
                if feats.shape[1] != n_tok:
                    raise ValueError(f"encoder returned {feats.shape[1]} tokens per image, expected {n_tok}")
                writer.add(feats)
            for i in ok:
                idx = len(caps_en)
                caps_en.append(batch[i][1])
                caps_uz.append(list(batch[i][2]))
                if want_uz and not batch[i][2] and per_translate[label] < translate_max:
                    to_translate.append((idx, batch[i][1][0]))
                    per_translate[label] += 1
                per_source[label] = per_source.get(label, 0) + 1
            batch.clear()

        for item in items:
            batch.append(item)
            if len(batch) >= ENCODE_BATCH:
                flush()
        flush()

    per_translate: Dict[str, int] = {}
    sources = [e for e in (cfg.data.vision or []) if _get(e, "source")]
    if not cfg.data.offline:
        for e in sources:
            name = str(_get(e, "source"))
            limit = int(_get(e, "max_images", 10 ** 9) or 10 ** 9)
            translate = _get(e, "translate", True) is not False
            per_translate[name] = 0
            try:
                consume(name, ((img, caps, []) for img, caps in _iter_source_images(name, cfg.data.cache_path, limit)),
                        int(_get(e, "translate_max", 2000)) if translate else 0)
            except ValueError:
                raise
            except Exception as ex:  # noqa: BLE001 - download / decode failure: keep other sources
                _logger().warning("vision source %s failed after %d images: %s", name, per_source.get(name, 0), ex)
    if writer.n == 0:
        n_syn = sum(int(_get(e, "max_images", 256) or 256) for e in sources) if sources else 256
        n_syn = max(32, min(n_syn, 512))
        _logger().info("image dataset: using %d synthetic shape images (%s)", n_syn,
                       "offline" if cfg.data.offline else "no source available")
        per_translate["synthetic"] = 0
        consume("synthetic", ((img, en, uz if want_uz else []) for img, en, uz in synthetic_images(n_syn, seed)), 0)
    writer.flush()
    if writer.n == 0:
        _logger().warning("image dataset: no image could be encoded")
        return None
    vision_dim = int(writer.dim or 0)
    want_dim = int(cfg.multimodal.vision_dim)
    if vision_dim != want_dim:
        raise ValueError(f"vision encoder features have dim {vision_dim} but multimodal.vision_dim = {want_dim}")

    # ---- optional en -> uz translation (cached)
    if to_translate:
        tr_path = out / "translations.json"
        cache: Dict[str, str] = {}
        if tr_path.is_file():
            try:
                cache = json.loads(tr_path.read_text(encoding="utf-8"))
            except ValueError:
                cache = {}
        todo = sorted({t for _, t in to_translate if t not in cache})
        if todo:
            try:
                for src, dst in zip(todo, translate_en_uz(todo)):
                    cache[src] = dst
                tr_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            except Exception as ex:  # noqa: BLE001 - translation is optional
                _logger().warning("caption translation unavailable (%s); Uzbek captions skipped", ex)
        for idx, text in to_translate:
            if cache.get(text):
                caps_uz[idx].append(cache[text])

    # ---- conversations -> token rows (packed, CSR image index)
    convs: List[Tuple[int, List[Dict[str, Any]]]] = []
    for i in range(writer.n):
        convs += image_conversations(i, caps_en[i], caps_uz[i], rng)
    rng.shuffle(convs)
    examples: List[Tuple[int, List[int], List[int]]] = []
    dropped = 0
    for idx, conv in convs:
        ids, mask = tokenizer.encode_chat(conv, n_image_tokens=n_tok)
        if len(ids) > L + 1:
            ids, mask = ids[:L + 1], mask[:L + 1]
        img_pos = [p for p, t in enumerate(ids) if t == IMAGE_ID]
        if len(img_pos) != n_tok or img_pos[-1] >= L or not any(mask[1:]):
            dropped += 1  # image block must lie fully inside x = ids[:L]
            continue
        examples.append((idx, ids, mask))
    if dropped:
        _logger().warning("image dataset: dropped %d conversations that do not fit seq_len=%d", dropped, L)
    if not examples:
        _logger().warning("image dataset: seq_len=%d too small for %d image tokens", L, n_tok)
        return None
    rows_ids: List[np.ndarray] = []
    rows_mask: List[np.ndarray] = []
    row_ptr = [0]
    row_images: List[int] = []
    cur_ids: List[int] = []
    cur_mask: List[int] = []
    cur_imgs: List[int] = []

    def close_row() -> None:
        r_ids = np.full(L + 1, PAD_ID, dtype=np.int32)
        r_mask = np.zeros(L + 1, dtype=np.uint8)
        r_ids[:len(cur_ids)] = cur_ids
        r_mask[:len(cur_mask)] = cur_mask
        rows_ids.append(r_ids)
        rows_mask.append(r_mask)
        row_images.extend(cur_imgs)
        row_ptr.append(len(row_images))

    for idx, ids, mask in examples:
        # an example may join the row only if its image block stays inside x (positions < L)
        last_img = max(p for p, t in enumerate(ids) if t == IMAGE_ID)
        fits = len(cur_ids) + len(ids) <= L + 1 and len(cur_ids) + last_img < L
        if cur_ids and not fits:
            close_row()
            cur_ids, cur_mask, cur_imgs = [], [], []
        cur_ids += list(ids)
        cur_mask += list(mask)
        cur_imgs.append(idx)
    if cur_ids:
        close_row()
    ids_arr = np.stack(rows_ids)
    mask_arr = np.stack(rows_mask)
    n_rows = len(ids_arr)
    n_val = min(n_rows - 1, max(1, n_rows // 50)) if n_rows >= 2 else 0
    val_rows = np.sort(np.random.default_rng(seed).permutation(n_rows)[:n_val]).astype(np.int64)
    stats = {"images": writer.n, "per_source": per_source, "conversations": len(examples), "rows": n_rows,
             "uz_captions": int(sum(len(c) for c in caps_uz)), "dropped": dropped}
    ds = ImageDataset(out, list(writer.files), writer.n, n_tok, vision_dim, ids_arr, mask_arr,
                      np.asarray(row_ptr, dtype=np.int64), np.asarray(row_images, dtype=np.int64), val_rows, stats)
    for name, arr in (("ids", ds.ids), ("mask", ds.mask), ("row_ptr", ds.row_ptr), ("row_images", ds.row_images),
                      ("val_rows", ds.val_rows)):
        tmp = out / f"{name}.part.npy"
        np.save(tmp, arr)
        os.replace(tmp, out / f"{name}.npy")
    meta = {"signature": sig, "feature_files": [p.name for p in ds.feature_files], "n_images": ds.n_images,
            "n_image_tokens": n_tok, "vision_dim": vision_dim, "encoder": enc_name,
            "encoder_loaded": str(getattr(encoder, "loaded_name", None) or getattr(encoder, "name", enc_name)),
            "stats": stats}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(out / "captions.jsonl", "w", encoding="utf-8") as fh:
        for i in range(writer.n):
            fh.write(json.dumps({"image": i, "en": caps_en[i], "uz": caps_uz[i]}, ensure_ascii=False) + "\n")
    _logger().info("image dataset: %d images, %d conversations -> %d rows (%s)", writer.n, len(examples), n_rows, out)
    return ds
