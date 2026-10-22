"""Shared pytest fixtures and helpers for the Moonlight test-suite (SPEC §11).

Design rules for every test module:

* Torch-dependent tests call ``torch = pytest.importorskip("torch")`` (module level or inside the
  test), so the torch-free part of the suite also runs on machines / CI jobs without PyTorch.
* Everything must be FAST on a 2-core CPU runner: models are built from tiny configs via
  :func:`tiny_config_dict` / the ``tiny_config`` fixture (``Config.from_dict``); registry configs are
  only used through overrides that shrink the training budgets.
* No network: tiny configs set ``data.offline: true`` and the environment variable
  ``MOONLIGHT_OFFLINE=1`` is set for the session (honoured by ``moonlight.data.download``).
"""

from __future__ import annotations

import copy
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow running the tests without `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT))

# Offline + quiet defaults for the whole session (set before moonlight modules are imported).
os.environ.setdefault("MOONLIGHT_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.pop("WANDB_PROJECT", None)

# ------------------------------------------------------------------------------------------------
# SPEC §10: the exact 100 config names, per tier (single source for all tests).
# ------------------------------------------------------------------------------------------------
TEST_TIER = """ml-1m-debug ml-1m-base ml-1m-chat ml-2m-base ml-2m-chat ml-2m-code ml-5m-base ml-5m-chat
ml-5m-code ml-10m-base ml-10m-chat ml-10m-code ml-20m-base ml-20m-chat ml-20m-code ml-20m-vl
ml-20m-colab-chat ml-20m-tools ml-50m-base ml-50m-chat ml-50m-code ml-50m-vl ml-50m-kaggle-chat
ml-50m-tools ml-50m-think""".split()
MINI_TIER = """ml-100m-base ml-100m-chat ml-100m-code ml-150m-chat ml-250m-base ml-250m-chat ml-250m-code
ml-250m-vl ml-250m-tools ml-250m-kaggle-chat ml-300m-base ml-300m-chat ml-300m-code ml-300m-vl
ml-300m-long ml-300m-tools ml-300m-think ml-300m-uz""".split()
SI_TIER = """ml-1b-base ml-1b-chat ml-1b-code ml-1b-agent ml-1b-think ml-2b-chat ml-2b-code ml-3b-base
ml-3b-chat ml-3b-code ml-3b-agent ml-7b-base ml-7b-chat ml-7b-code ml-7b-agent ml-7b-long ml-7b-think
ml-13b-base ml-13b-chat ml-13b-code ml-13b-agent ml-30b-base ml-30b-chat ml-30b-code ml-30b-vl
ml-30b-vl-chat ml-70b-base ml-70b-chat ml-70b-code ml-70b-vl ml-70b-agent ml-70b-think ml-175b-base
ml-175b-chat ml-175b-vl ml-405b-base ml-405b-chat ml-405b-vl ml-1t-base ml-1t-chat ml-1t-vl
ml-16b-a2b-chat ml-16b-a2b-code ml-30b-a3b-base ml-30b-a3b-chat ml-30b-a3b-vl ml-100b-a10b-chat
ml-100b-a10b-vl ml-235b-a22b-chat ml-235b-a22b-code ml-235b-a22b-vl ml-671b-a37b-chat
ml-671b-a37b-vl ml-1t-a32b-base ml-1t-a32b-chat ml-1t-a32b-code ml-1t-a32b-vl""".split()
CONFIG_TIERS: Dict[str, list] = {"test": TEST_TIER, "mini": MINI_TIER, "si": SI_TIER}
ALL_CONFIG_NAMES = TEST_TIER + MINI_TIER + SI_TIER

#: SPEC §9 benchmark tiers by NOMINAL total params: (suite, low, high) inclusive.
SUITE_RANGES = (
    ("cc-001", 1e6, 2e6),
    ("cc-002", 5e6, 20e6),
    ("cc-003", 50e6, 100e6),
    ("cc-004", 150e6, 300e6),
    ("cc-005", 1e9, 3e9),
    ("cc-006", 7e9, 13e9),
    ("cc-007", 16e9, 70e9),
    ("cc-008", 100e9, 235e9),
    ("cc-009", 405e9, 671e9),
    ("cc-010", 1e12, 1e12),
)
SUITE_NAMES = tuple(s for s, _, _ in SUITE_RANGES)


def expected_suite(nominal: float) -> str:
    """SPEC §9 suite for a nominal parameter count (exact table ranges)."""
    for name, lo, hi in SUITE_RANGES:
        if lo <= nominal <= hi:
            return name
    raise AssertionError(f"nominal size {nominal:g} is not covered by the SPEC §9 table")


# ------------------------------------------------------------------------------------------------
# tiny configs
# ------------------------------------------------------------------------------------------------
_TINY_MODEL: Dict[str, Any] = {
    "vocab_size": 512,       # >= 288 so the byte-level tokenizer fits (offline chat fallback)
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,            # 2 attention + 2 retention heads
    "head_dim": 16,
    "retention_ratio": 0.5,
    "n_kv_heads": 1,
    "attn_window": None,
    "global_every": 1,
    "ffn_hidden": 128,
    "token_shift": 4,
    "value_residual": True,
    "qk_norm": True,
    "output_gate": True,
    "rope_theta": 10000.0,
    "max_seq_len": 256,
    "tie_embeddings": True,
    "softcap": 30.0,
    "mtp_depth": 1,
    "mtp_block": False,
    "mtp_weight": 0.3,
    "z_loss": 0.0,
    "init_std": 0.02,
    "moe": {"enabled": False},
}


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def tiny_config_dict(model: Optional[Dict[str, Any]] = None, **sections: Any) -> Dict[str, Any]:
    """A complete, valid config mapping for a ~0.1M-parameter offline model.

    ``model`` entries are deep-merged into the tiny model section (``moe`` too); other keyword
    arguments deep-merge into the top-level sections (``train={...}``, ``multimodal={...}``...).
    ``tokenizer.vocab_size`` follows ``model.vocab_size`` unless given explicitly.
    """
    m = _deep_update(copy.deepcopy(_TINY_MODEL), dict(model or {}))
    d: Dict[str, Any] = {
        "name": "tiny-test",
        "description": "tiny unit-test model",
        "tier": "test",
        "kind": "chat",
        "benchmark": "cc-001",
        "nominal_params": "1M",
        "model": m,
        "tokenizer": {"vocab_size": m["vocab_size"]},
        "data": {"offline": True, "synthetic": {"identity": True, "knowledge": True, "arithmetic": 20,
                                                "tool_calls": 20, "code": 10}},
        "train": {"seq_len": 64, "batch_size": 4, "batch_tokens": 256, "compile": False, "precision": "fp32",
                  "ema": 0.0, "curriculum": {"enabled": False}},
        "chat": {"best_of": 1, "max_new_tokens": 8, "temperature": 0.7},
    }
    _deep_update(d, {k: v for k, v in sections.items() if k != "model"})
    if "tokenizer" not in sections or "vocab_size" not in (sections.get("tokenizer") or {}):
        d["tokenizer"]["vocab_size"] = d["model"]["vocab_size"]
    return d


def make_tiny_config(model: Optional[Dict[str, Any]] = None, validate: bool = True, **sections: Any):
    """``Config`` built from :func:`tiny_config_dict`; ``nominal_params`` is set to the exact estimate
    so ``validate()`` emits no size warning."""
    from moonlight.config import Config, format_size
    from moonlight.params import estimate_params

    cfg = Config.from_dict(tiny_config_dict(model, **sections))
    cfg.nominal_params = format_size(estimate_params(cfg).total, precision=3)
    if validate:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cfg.validate()
    return cfg


@pytest.fixture
def tiny_config() -> Callable[..., Any]:
    """Factory fixture: ``tiny_config(model={...}, train={...})`` -> validated ``Config``."""
    return make_tiny_config


@pytest.fixture(scope="session")
def byte_tokenizer():
    """The untrained byte-level tokenizer (32 specials + 256 bytes)."""
    from moonlight.tokenizer import MoonlightTokenizer

    return MoonlightTokenizer.byte_level()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def registry_configs():
    """``{name: Config}`` for every registry config (loaded once per session, size warnings muted)."""
    from moonlight.registry import list_configs, load

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for name in list_configs():
            out[name] = load(name, validate=False)
    return out


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "slow: slower end-to-end test (still CPU-friendly)")
