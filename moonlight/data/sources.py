"""Registry of the external datasets Moonlight can stream (SPEC §6, §13.1).

Every entry is a :class:`DatasetSpec` with fully resolved Hugging Face ``resolve/main`` URLs, a
file format understood by :func:`moonlight.data.download.iter_rows` and the name of the adapter
(see :mod:`moonlight.data.adapters`) that turns a raw row into text / a conversation / a
preference pair.

Shard names are fixed (no ``find``/glob at runtime). If a file ever 404s because the dataset owner
re-sharded it, :func:`moonlight.data.download.resolve_hf_shards` lists the directory through the HF
API (``https://huggingface.co/api/datasets/<repo>/tree/main/<dir>``) and the loader retries with
the listed shard names (see :func:`moonlight.data.download.download_source`).

This module is pure python (no torch, no network at import time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "HF_BASE", "KINDS", "FORMATS", "DatasetSpec", "DATASETS", "hf_url", "get_source", "list_sources",
    "repo_and_path",
]

HF_BASE = "https://huggingface.co"

#: dataset kinds (``pref`` = preference pairs for DPO, ``rl`` = prompts with verifiable answers).
KINDS: Tuple[str, ...] = ("pretrain", "sft", "vision", "bench", "pref", "rl")
#: on-disk formats understood by ``iter_rows``.
FORMATS: Tuple[str, ...] = ("parquet", "jsonl", "json", "json.gz")


def hf_url(repo: str, path: str) -> str:
    """``https://huggingface.co/datasets/<repo>/resolve/main/<path>``."""
    return f"{HF_BASE}/datasets/{repo}/resolve/main/{path.lstrip('/')}"


def repo_and_path(url: str) -> Optional[Tuple[str, str]]:
    """Inverse of :func:`hf_url`: ``(repo, path)`` or None for non-HF-dataset URLs."""
    prefix = f"{HF_BASE}/datasets/"
    if not url.startswith(prefix) or "/resolve/main/" not in url:
        return None
    repo, path = url[len(prefix):].split("/resolve/main/", 1)
    return repo, path


@dataclass
class DatasetSpec:
    """One dataset source.

    Attributes:
        name: registry key (e.g. ``"uz-news"``).
        kind: one of :data:`KINDS`.
        urls: download URLs (one per shard), read in order.
        fmt: one of :data:`FORMATS`.
        adapter: adapter name in :mod:`moonlight.data.adapters` (``ADAPTERS`` table).
        language: ``"uz"``, ``"en"``, ``"code"`` or ``"multi"`` (drives cleaning / filtering).
        license: license identifier as stated on the dataset card (verify before redistribution).
        gated: True if the dataset requires accepting terms on the Hub (needs ``HF_TOKEN``).
        notes: free-form remarks (columns, caveats).
        columns: parquet columns to read (None -> all); saves memory on wide tables.
    """

    name: str
    kind: str
    urls: List[str]
    fmt: str
    adapter: str
    language: str
    license: str
    gated: bool = False
    notes: str = ""
    columns: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"dataset {self.name!r}: kind must be one of {KINDS}, got {self.kind!r}")
        if self.fmt not in FORMATS:
            raise ValueError(f"dataset {self.name!r}: fmt must be one of {FORMATS}, got {self.fmt!r}")
        if not self.urls:
            raise ValueError(f"dataset {self.name!r}: needs at least one url")

    @property
    def repo(self) -> Optional[str]:
        """Hub repo id of the first url (None for non-HF urls)."""
        rp = repo_and_path(self.urls[0])
        return rp[0] if rp else None


def _shards(repo: str, pattern: str, n: int, width: int = 5) -> List[str]:
    """URLs for ``pattern.format(i=<zero padded i>, n=<zero padded n>)`` for i in 0..n-1."""
    return [hf_url(repo, pattern.format(i=str(i).zfill(width), n=str(n).zfill(width))) for i in range(n)]


def _one(repo: str, *paths: str) -> List[str]:
    return [hf_url(repo, p) for p in paths]


_SMOLTALK = "HuggingFaceTB/smoltalk"
_SMOLLM = "HuggingFaceTB/smollm-corpus"
_UZCRAWL = "tahrirchi/uz-crawl"

_SPECS: Sequence[DatasetSpec] = [
    # ------------------------------------------------------------------ pretrain
    DatasetSpec("uz-news", "pretrain", _shards(_UZCRAWL, "data/news-{i}-of-{n}.parquet", 7), "parquet",
                "text", "uz", "apache-2.0", notes="Uzbek news crawl (Latin + some Cyrillic); column text.",
                columns=["text"]),
    DatasetSpec("uz-telegram", "pretrain", _one(_UZCRAWL, "data/telegram_blogs-00000-of-00001.parquet"),
                "parquet", "text", "uz", "apache-2.0", notes="Uzbek Telegram blogs; column text.",
                columns=["text"]),
    DatasetSpec("uz-wiki", "pretrain", _one("wikimedia/wikipedia", "20231101.uz/train-00000-of-00001.parquet"),
                "parquet", "text", "uz", "cc-by-sa-3.0",
                notes="Uzbek Wikipedia 2023-11-01 dump; columns id,url,title,text.", columns=["title", "text"]),
    DatasetSpec("uz-books", "pretrain", _shards("tahrirchi/uz-books", "data/lat-{i}-of-{n}.parquet", 28),
                "parquet", "text", "uz", "apache-2.0", notes="Uzbek books, Latin script; column text.",
                columns=["text"]),
    DatasetSpec("fineweb-edu", "pretrain",
                _one(_SMOLLM, "fineweb-edu-dedup/train-00000-of-00234.parquet"), "parquet", "text", "en",
                "odc-by", notes="First shard of 234 (enough for <= 1B-token budgets); column text.",
                columns=["text"]),
    DatasetSpec("cosmopedia", "pretrain", _one(_SMOLLM, "cosmopedia-v2/train-00000-of-00104.parquet"),
                "parquet", "text", "en", "odc-by", notes="Synthetic textbooks; first shard of 104.",
                columns=["text"]),
    DatasetSpec("stack-python", "pretrain", _one("bigcode/the-stack-smol", "data/python/data.json"),
                "json", "content", "code", "other (the-stack terms of use)", gated=True,
                notes="Field content. Gated: accept the terms on the Hub and set HF_TOKEN."),
    DatasetSpec("codeparrot", "pretrain", _one("codeparrot/codeparrot-clean", "file-000000000001.json.gz"),
                "json.gz", "content", "code", "other (see dataset card)",
                notes="JSON lines inside gzip; field content."),
    # ------------------------------------------------------------------ sft
    DatasetSpec("alpaca-uz", "sft", _one("behbudiy/alpaca-cleaned-uz", "data/train-00000-of-00001.parquet"),
                "parquet", "alpaca", "uz", "cc-by-4.0", notes="instruction, input, output."),
    DatasetSpec("uzbek-instruct", "sft",
                _one("UAzimov/uzbek-instruct-llm", "uzbek-instruct-dataset-openai.jsonl"), "jsonl",
                "messages", "uz", "see dataset card", notes="OpenAI-style messages."),
    DatasetSpec("sft-uz-9k", "sft", _one("MLDataScientist/SFT-UZ-9k", "output_uz_sft.json"), "json",
                "sft_uz_9k", "uz", "see dataset card",
                notes="prompt, answer (outer quotes / escaped quotes stripped), system, history."),
    DatasetSpec("oasst2-uz", "sft",
                _one("MLDataScientist/oasst2_uzbek_threads", "data/train-00000-of-00001.parquet"), "parquet",
                "llama3_text", "uz", "apache-2.0", notes="column text in Llama-3 header format."),
    DatasetSpec("slimorca-uz", "sft",
                _shards("MLDataScientist/SlimOrca-Dedup-Uzbek-cleaned", "data/train-{i}-of-{n}.parquet", 2),
                "parquet", "sharegpt", "uz", "mit", notes="conversations [{from: system|human|gpt, value}]."),
    DatasetSpec("smoltalk", "sft", _shards(_SMOLTALK, "data/all/train-{i}-of-{n}.parquet", 9), "parquet",
                "messages", "en", "apache-2.0", notes="messages; the full SmolTalk mix.",
                columns=["messages"]),
    DatasetSpec("smoltalk-everyday", "sft",
                _one(_SMOLTALK, "data/everyday-conversations/train-00000-of-00001.parquet"), "parquet",
                "messages", "en", "apache-2.0", columns=["messages"]),
    DatasetSpec("smoltalk-code", "sft", _one(_SMOLTALK, "data/self-oss-instruct/train-00000-of-00001.parquet"),
                "parquet", "messages", "code", "apache-2.0", columns=["messages"]),
    DatasetSpec("smoltalk-tools", "sft", _one(_SMOLTALK, "data/apigen-80k/train-00000-of-00001.parquet"),
                "parquet", "apigen", "en", "apache-2.0",
                notes="APIGen function calling; tool calls in assistant content are parsed when possible.",
                columns=["messages"]),
    DatasetSpec("smoltalk-constraints", "sft",
                _one(_SMOLTALK, "data/smol-constraints/train-00000-of-00001.parquet"), "parquet", "messages",
                "en", "apache-2.0", columns=["messages"]),
    DatasetSpec("smoltalk-math", "sft", _one(_SMOLTALK, "data/numina-cot-100k/train-00000-of-00001.parquet"),
                "parquet", "messages", "en", "apache-2.0", columns=["messages"]),
    DatasetSpec("smoltalk-magpie", "sft",
                _shards(_SMOLTALK, "data/smol-magpie-ultra/train-{i}-of-{n}.parquet", 6), "parquet",
                "messages", "en", "apache-2.0", columns=["messages"]),
    DatasetSpec("openthoughts", "sft",
                _shards("open-thoughts/OpenThoughts-114k", "data/train-{i}-of-{n}.parquet", 6), "parquet",
                "openthoughts", "en", "apache-2.0",
                notes="system, conversations [{from: user|assistant, value}]; thought markers -> <|think|>.",
                columns=["system", "conversations"]),
    DatasetSpec("python-18k", "sft",
                _one("iamtarun/python_code_instructions_18k_alpaca",
                     "data/train-00000-of-00001-8b6e212f3e1ece96.parquet"), "parquet", "alpaca", "code",
                "see dataset card", notes="instruction, input, output (+ prompt)."),
    DatasetSpec("codefeedback", "sft",
                _one("m-a-p/CodeFeedback-Filtered-Instruction", "CodeFeedback-Filtered-Instruction.jsonl"),
                "jsonl", "query_answer", "code", "apache-2.0", notes="query, answer."),
    DatasetSpec("glaive-tools", "sft",
                _one("glaiveai/glaive-function-calling-v2", "glaive-function-calling-v2.json"), "json",
                "glaive", "en", "apache-2.0",
                notes="system ('SYSTEM: ... functions ... -' + JSON schemas), chat (USER:/ASSISTANT:/"
                      "FUNCTION RESPONSE:, <functioncall> {...}); converted to Moonlight tool format."),
    # ------------------------------------------------------------------ preference / rl
    DatasetSpec("ultrafeedback", "pref",
                _one("HuggingFaceH4/ultrafeedback_binarized", "data/train_prefs-00000-of-00001.parquet"),
                "parquet", "preference", "en", "mit",
                notes="prompt, chosen [messages], rejected [messages], score_chosen, score_rejected.",
                columns=["prompt", "chosen", "rejected", "score_chosen", "score_rejected"]),
    DatasetSpec("gsm8k-train", "rl", _one("openai/gsm8k", "main/train-00000-of-00001.parquet"), "parquet",
                "gsm8k", "en", "mit", notes="question, answer ('... #### N')."),
    # ------------------------------------------------------------------ vision
    DatasetSpec("flickr8k", "vision",
                _one("jxie/flickr8k", *(f"data/{s}.parquet" for s in (
                    "train-00000-of-00002-2f8f6bfa852eac4b", "train-00001-of-00002-2173151d8cd6c7fb",
                    "validation-00000-of-00001-7025a2b596f14b7b", "test-00000-of-00001-42a2661d12c73e48"))),
                "parquet", "flickr8k", "en", "see dataset card (research use)",
                notes="image {bytes}, caption_0..caption_4."),
    DatasetSpec("flickr30k", "vision", _shards("lmms-lab/flickr30k", "data/test-{i}-of-{n}.parquet", 9),
                "parquet", "flickr30k", "en", "see dataset card (research use)",
                notes="image {bytes}, caption (list of 5)."),
    # ------------------------------------------------------------------ bench
    DatasetSpec("humaneval", "bench",
                _one("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet"), "parquet",
                "identity", "code", "mit"),
    DatasetSpec("mbpp", "bench", _one("google-research-datasets/mbpp", "sanitized/test-00000-of-00001.parquet"),
                "parquet", "identity", "code", "cc-by-4.0"),
    DatasetSpec("mmlu", "bench", _one("cais/mmlu", "all/test-00000-of-00001.parquet"), "parquet", "identity",
                "en", "mit"),
    DatasetSpec("arc-easy", "bench", _one("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet"),
                "parquet", "identity", "en", "cc-by-sa-4.0"),
    DatasetSpec("arc-challenge", "bench", _one("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet"),
                "parquet", "identity", "en", "cc-by-sa-4.0"),
    DatasetSpec("hellaswag", "bench", _one("Rowan/hellaswag", "data/validation-00000-of-00001.parquet"),
                "parquet", "identity", "en", "mit"),
    DatasetSpec("gsm8k", "bench", _one("openai/gsm8k", "main/test-00000-of-00001.parquet"), "parquet",
                "identity", "en", "mit"),
    DatasetSpec("belebele-uz", "bench", _one("facebook/belebele", "data/uzn_Latn.jsonl"), "jsonl", "identity",
                "uz", "cc-by-sa-4.0"),
    DatasetSpec("ifeval", "bench", _one("google/IFEval", "ifeval_input_data.jsonl"), "jsonl", "identity", "en",
                "apache-2.0", notes="key, prompt, instruction_id_list, kwargs."),
]

#: name -> spec
DATASETS: Dict[str, DatasetSpec] = {s.name: s for s in _SPECS}
if len(DATASETS) != len(_SPECS):  # pragma: no cover - guards against copy/paste duplicates
    raise RuntimeError("duplicate dataset names in moonlight.data.sources")


def get_source(name: str) -> DatasetSpec:
    """Look up a dataset by registry name; raises ``KeyError`` listing the known names."""
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown data source {name!r}; known: {', '.join(sorted(DATASETS))}") from None


def list_sources(kind: Optional[str] = None) -> List[str]:
    """Sorted dataset names, optionally filtered by ``kind``."""
    return sorted(n for n, s in DATASETS.items() if kind is None or s.kind == kind)
