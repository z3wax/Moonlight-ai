"""Moonlight data pipelines (SPEC §6, §13).

Submodules:

* :mod:`~moonlight.data.sources`   - ``DATASETS`` registry of Hugging Face sources (``get_source``)
* :mod:`~moonlight.data.download`  - cached downloads (``HF_TOKEN``), ``iter_rows``, ``resolve_hf_shards``
* :mod:`~moonlight.data.cleaning`  - Uzbek / English / code cleaning and quality filters
* :mod:`~moonlight.data.adapters`  - raw rows -> text / conversations / preference pairs
* :mod:`~moonlight.data.synthetic` - deterministic synthetic conversations (identity, knowledge,
  arithmetic, tools, code, reasoning, plans, constraints, details, agentic projects, GRPO prompts,
  preference pairs)
* :mod:`~moonlight.data.verifiers` - deterministic response checkers and rewards
* :mod:`~moonlight.data.pretrain`  - token bins + ``PretrainLoader``
* :mod:`~moonlight.data.sft`       - SFT / preference encoding, packing, loaders (numpy)
* :mod:`~moonlight.data.vision`    - image-caption features + ``ImageSFTLoader``

Importing this package is cheap and torch-free: submodules and the re-exported names below are
loaded lazily on first attribute access (PEP 562).
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

_SUBMODULES = ("sources", "download", "cleaning", "adapters", "synthetic", "verifiers", "pretrain", "sft", "vision")

#: public name -> submodule that defines it
_EXPORTS: Dict[str, str] = {
    "DatasetSpec": "sources", "DATASETS": "sources", "get_source": "sources", "list_sources": "sources",
    "iter_rows": "download", "resolve_hf_shards": "download",
    "iter_source_rows": "download", "DownloadError": "download",
    "norm_apostrophes": "cleaning", "norm_text": "cleaning", "clean_doc": "cleaning", "good_doc": "cleaning",
    "clean_code": "cleaning", "dedup_exact": "cleaning",
    "to_text": "adapters", "to_conversation": "adapters", "to_preference": "adapters", "to_rl_prompt": "adapters",
    "build_synthetic": "synthetic", "grpo_prompt_set": "synthetic", "preference_pairs_synthetic": "synthetic",
    "check": "verifiers", "check_all": "verifiers", "compute_reward": "verifiers",
    "build_pretrain_tokens": "pretrain", "PretrainLoader": "pretrain",
    "encode_conversations": "sft", "pack": "sft", "split_val": "sft", "SFTLoader": "sft",
    "encode_preference": "sft", "PreferenceLoader": "sft",
    "build_image_dataset": "vision", "ImageSFTLoader": "vision", "ImageDataset": "vision",
}

__all__: List[str] = list(_SUBMODULES) + sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    mod = _EXPORTS.get(name)
    if mod is not None:
        value = getattr(importlib.import_module(f"{__name__}.{mod}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))
