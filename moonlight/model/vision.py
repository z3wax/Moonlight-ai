"""Vision components: the trainable projector and a frozen, lazily-loaded image encoder.

- :class:`VisionProjector` (part of the language model, ``model.vision_proj``):
  ``fc2(gelu(fc1(norm(x))))`` mapping encoder features (..., vision_dim) to (..., d_model).
  Parameters: ``norm`` RMSNorm(vision_dim), ``fc1`` Linear(vision_dim, d), ``fc2`` Linear(d, d).
- :class:`VisionEncoder` (NOT an ``nn.Module`` and never part of the model's parameters): wraps a
  frozen HuggingFace ``SiglipVisionModel`` (fallback ``CLIPVisionModel``
  "openai/clip-vit-base-patch32"); ``transformers`` is imported lazily on first use. It accepts
  file paths, URLs, raw bytes, HF-datasets image dicts, numpy arrays or PIL images and returns
  (B, n_tokens, vision_dim) float32 features obtained by adaptive average pooling of the patch grid
  to ``sqrt(n_tokens) x sqrt(n_tokens)``.
"""
from __future__ import annotations

import io
import logging
import math
import os
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, is_meta

__all__ = ["VisionProjector", "VisionEncoder", "load_image", "DEFAULT_VISION_ENCODER",
           "FALLBACK_VISION_ENCODER"]

logger = logging.getLogger(__name__)

DEFAULT_VISION_ENCODER = "google/siglip-base-patch16-224"
FALLBACK_VISION_ENCODER = "openai/clip-vit-base-patch32"

# normalisation constants used when no HF image processor is available
_SIGLIP_MEAN_STD = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
_CLIP_MEAN_STD = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


class VisionProjector(nn.Module):
    """Maps vision-encoder features into the language model embedding space.

    ``forward(x)``: (..., vision_dim) -> (..., d_model). GELU between ``fc1`` and ``fc2``.
    Init: norm weight 1, fc1/fc2 ~ N(0, init_std) (fc2 is *not* zero-initialised).
    """

    def __init__(self, vision_dim: int, d_model: int, init_std: float = 0.02) -> None:
        super().__init__()
        self.vision_dim = int(vision_dim)
        self.d_model = int(d_model)
        self.init_std = float(init_std)
        self.norm = RMSNorm(self.vision_dim)
        self.fc1 = nn.Linear(self.vision_dim, self.d_model, bias=False)
        self.fc2 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        for fc in (self.fc1, self.fc2):
            if not is_meta(fc.weight):
                nn.init.normal_(fc.weight, mean=0.0, std=self.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


# --------------------------------------------------------------------------------------------
# image loading
# --------------------------------------------------------------------------------------------
def _pil():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("Pillow is required for image input: pip install pillow") from exc
    return Image


def load_image(src: Any, timeout: float = 30.0):
    """Load one image as an RGB ``PIL.Image``.

    Accepts a PIL image, a local path (str / ``os.PathLike``), an http(s) URL, raw ``bytes``, a
    HuggingFace-datasets image dict (``{"bytes": ..., "path": ...}``), a numpy array (H, W[, C])
    (uint8, or float in [0, 1]) or a torch tensor (C, H, W) / (H, W, C).
    """
    Image = _pil()
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, dict):
        if src.get("bytes"):
            return load_image(bytes(src["bytes"]), timeout)
        if src.get("path"):
            return load_image(src["path"], timeout)
        raise ValueError("image dict needs a 'bytes' or 'path' entry")
    if isinstance(src, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(bytes(src)))
        img.load()
        return img.convert("RGB")
    if isinstance(src, (str, os.PathLike)):
        s = os.fspath(src)
        if s.startswith(("http://", "https://")):
            import urllib.request

            req = urllib.request.Request(s, headers={"User-Agent": "moonlight/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - user URL
                return load_image(resp.read(), timeout)
        with Image.open(os.path.expanduser(s)) as img:
            img.load()
            return img.convert("RGB")
    if isinstance(src, torch.Tensor):
        arr = src.detach().cpu()
        if arr.dim() == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = arr.permute(1, 2, 0)
        src = arr.float().numpy() if arr.is_floating_point() else arr.numpy()
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None and isinstance(src, np.ndarray):
        arr = src
        if arr.dtype != np.uint8:
            arr = np.asarray(arr, dtype=np.float64)
            if arr.size and arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"unsupported image input of type {type(src).__name__}")


# --------------------------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------------------------
class VisionEncoder:
    """Frozen image encoder producing (B, n_tokens, vision_dim) features.

    ``VisionEncoder(name, n_tokens=49, device=None, dtype=None)``; ``device`` defaults to CUDA when
    available, ``dtype`` to float16 on CUDA and float32 otherwise. The HF model is loaded lazily
    (first call to :meth:`load` / :meth:`encode`): ``SiglipVisionModel`` for SigLIP names, falling
    back to ``CLIPVisionModel`` ("openai/clip-vit-base-patch32") if loading fails.
    """

    def __init__(
        self,
        name: str = DEFAULT_VISION_ENCODER,
        n_tokens: int = 49,
        device: Optional[Any] = None,
        dtype: Optional[torch.dtype] = None,
        image_size: int = 224,
    ) -> None:
        self.name = name or DEFAULT_VISION_ENCODER
        self.n_tokens = int(n_tokens)
        if self.n_tokens < 1:
            raise ValueError("n_tokens must be >= 1")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype
        self.image_size = int(image_size)
        self._model = None
        self._processor = None
        self._is_clip = False
        self.loaded_name: Optional[str] = None

    # -- loading --------------------------------------------------------------------------------
    def load(self) -> "VisionEncoder":
        """Load the HF vision tower (idempotent)."""
        if self._model is not None:
            return self
        import importlib.util

        if importlib.util.find_spec("transformers") is None:
            raise ImportError("VisionEncoder needs `transformers`: pip install transformers")
        errors: List[str] = []
        candidates: List[Tuple[str, bool]] = []
        if "clip" in self.name.lower() and "siglip" not in self.name.lower():
            candidates.append((self.name, True))
        else:
            candidates.append((self.name, False))
        if self.name != FALLBACK_VISION_ENCODER:
            candidates.append((FALLBACK_VISION_ENCODER, True))
        for cand, is_clip in candidates:
            try:
                model, processor = self._load_hf(cand, is_clip)
            except Exception as exc:  # network, auth, missing class, ... -> try the fallback
                errors.append(f"{cand}: {type(exc).__name__}: {exc}")
                logger.warning("could not load vision encoder %s (%s)", cand, exc)
                continue
            self._model, self._processor, self._is_clip, self.loaded_name = model, processor, is_clip, cand
            return self
        raise RuntimeError("no vision encoder could be loaded:\n" + "\n".join(errors))

    def _load_hf(self, name: str, is_clip: bool):
        import transformers

        cls = transformers.CLIPVisionModel if is_clip else transformers.SiglipVisionModel
        model = cls.from_pretrained(name, torch_dtype=self.dtype)
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        processor = None
        try:
            processor = transformers.AutoImageProcessor.from_pretrained(name)
        except Exception as exc:  # processor is optional: manual preprocessing is used instead
            logger.info("no image processor for %s (%s); using manual preprocessing", name, exc)
        return model, processor

    @property
    def vision_dim(self) -> int:
        """Hidden size of the encoder (loads the model)."""
        self.load()
        return int(self._model.config.hidden_size)

    def to(self, device=None, dtype=None) -> "VisionEncoder":
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        if self._model is not None:
            self._model = self._model.to(device=self.device, dtype=self.dtype)
        return self

    # -- preprocessing ----------------------------------------------------------------------------
    def _mean_std(self) -> Tuple[Sequence[float], Sequence[float]]:
        proc = self._processor
        if proc is not None and getattr(proc, "image_mean", None) is not None:
            return tuple(proc.image_mean), tuple(proc.image_std)
        return _CLIP_MEAN_STD if self._is_clip else _SIGLIP_MEAN_STD

    def preprocess(self, images: Sequence[Any]) -> torch.Tensor:
        """Images -> pixel_values (B, 3, S, S) float32 (resize to S=image_size, mean/std normalise).

        Uses the HF image processor when available, else a manual bicubic resize with the
        processor's (or the model family's default) mean/std.
        """
        pil = [load_image(im) for im in images]
        if self._processor is not None:
            try:
                out = self._processor(images=pil, return_tensors="pt")
                return out["pixel_values"].float()
            except Exception as exc:  # fall back to manual preprocessing
                logger.info("image processor failed (%s); using manual preprocessing", exc)
        import numpy as np

        Image = _pil()
        size = self.image_size
        mean, std = self._mean_std()
        mean_a = np.asarray(mean, dtype=np.float32)[:, None, None]
        std_a = np.asarray(std, dtype=np.float32)[:, None, None]
        resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
        arrs = []
        for im in pil:
            im = im.resize((size, size), resample=resample)
            a = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0  # (3, S, S)
            arrs.append((a - mean_a) / std_a)
        return torch.from_numpy(np.stack(arrs, axis=0).astype(np.float32))

    # -- encoding --------------------------------------------------------------------------------
    def pool(self, hidden: torch.Tensor) -> torch.Tensor:
        """Patch features (B, P[+1 CLS], D) -> (B, n_tokens, D) by adaptive average pooling.

        A leading CLS token is dropped when ``P - 1`` is a perfect square and ``P`` is not. If
        ``n_tokens`` is a perfect square the 2-D grid is pooled to ``sqrt(n) x sqrt(n)``; otherwise
        the flattened patch sequence is pooled 1-D to ``n_tokens``.
        """
        B, P, D = hidden.shape
        g = int(math.isqrt(P))
        if g * g != P and int(math.isqrt(P - 1)) ** 2 == P - 1:
            hidden = hidden[:, 1:]
            P -= 1
            g = int(math.isqrt(P))
        s = int(math.isqrt(self.n_tokens))
        x = hidden.float()
        if g * g == P and s * s == self.n_tokens:
            grid = x.transpose(1, 2).reshape(B, D, g, g)
            pooled = F.adaptive_avg_pool2d(grid, (s, s))  # (B, D, s, s)
            return pooled.flatten(2).transpose(1, 2).contiguous()
        pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), self.n_tokens)  # (B, D, n)
        return pooled.transpose(1, 2).contiguous()

    @torch.no_grad()
    def encode(self, images: Any, batch_size: int = 32) -> torch.Tensor:
        """Encode one image or a sequence of images -> (B, n_tokens, vision_dim) float32 on
        ``self.device``."""
        if not isinstance(images, (list, tuple)):
            images = [images]
        if len(images) == 0:
            raise ValueError("no images given")
        self.load()
        bs = max(1, int(batch_size))
        outs = []
        for i in range(0, len(images), bs):
            pv = self.preprocess(images[i:i + bs]).to(device=self.device, dtype=self.dtype)
            hidden = self._model(pixel_values=pv).last_hidden_state  # (b, P, D)
            outs.append(self.pool(hidden))
        return torch.cat(outs, dim=0)

    __call__ = encode
