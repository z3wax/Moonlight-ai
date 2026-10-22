#!/usr/bin/env python
"""Generate the 100 Moonlight training recipes (SPEC §10 + §13.8) deterministically.

Writes ``configs/{tier}/{name}.yaml`` (exactly 100 files) and ``docs/CONFIGS.md`` (summary table).

Usage::

    python scripts/generate_configs.py              # (re)write configs/ and docs/CONFIGS.md
    python scripts/generate_configs.py --check      # exit 1 if the committed files differ / extra / missing
    python scripts/generate_configs.py --out DIR    # write DIR/configs/... and DIR/docs/CONFIGS.md instead
    python scripts/generate_configs.py --summary    # also print the size/tolerance table

Torch-free: only ``moonlight.config`` and ``moonlight.params`` are used (pure python + PyYAML).

How the shapes are chosen
-------------------------
* Dense sizes start from the SPEC §10 base-shape table. ``n_kv_heads`` is lowered to the largest
  divisor of the attention-head count when the table value does not divide it (2B, 3B, 13B).
  When the exact estimator (:func:`moonlight.params.estimate_params`) puts the SPEC shape within
  ``±5 %`` of the nominal size the shape is kept as is; otherwise ``ffn_hidden`` is re-solved in
  closed form (the total is linear in ``ffn_hidden``) and rounded to a multiple of 64 (``d < 2048``)
  or 128, which lands within ~1 % of nominal. Shapes are fitted once per size (without the vision
  projector) and shared by every variant of that size.
* MoE sizes start from the SPEC §10 MoE table. If the total is not within ``±5 %`` of the nominal
  TOTAL and the active count within ``±10 %`` of the ``-aNb`` label, a deterministic grid search over
  ``n_layers`` (``L0/2 .. L0``), ``expert_hidden`` (multiples of 128 in ``[he0/2, 3*he0]``) and
  ``n_shared`` (``n_shared0 .. 0``) keeps candidates with total within ``±5 %`` and active within
  ``±15 %`` and picks the lowest score ``0.5 * max(err_total / 0.05, err_active / 0.10) +
  2 |ln L/L0| + |ln he/he0| + 0.5 * removed_shared_experts`` (ties: more layers, smaller ``he``). ``d_model``, heads,
  ``head_dim``, ``n_experts``, ``top_k`` and ``first_dense_layers`` are never changed. (With the
  hybrid mixer's per-head retention K/V projections the SPEC MoE shapes are attention-heavy, so most
  of them start 20-40 % above their active label.)
  MoE models without dense layers (``first_dense_layers: 0``) carry ``ffn_hidden = 4 * d_model``
  only as a valid placeholder; it is not used by any layer.
* ``tie_embeddings`` is on for dense models below 7B and for every MoE model (an untied
  131k-token head would add 0.3-1 B parameters to every token's ACTIVE budget), off otherwise.

Every generated config is validated with ``Config.validate(strict=True)`` and checked against the
generator tolerances (dense / MoE total ``±10 %``, MoE active ``±20 %``) before anything is written.
"""

from __future__ import annotations

import argparse
import difflib
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # runnable from a source checkout without `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT))

from moonlight.config import Config, ModelConfig, format_size, parse_size  # noqa: E402
from moonlight.params import estimate_params  # noqa: E402

# ================================================================================================
# SPEC §10: the exact 100 names (order = SPEC order; used for the docs table)
# ================================================================================================

TIER_NAMES: Dict[str, Tuple[str, ...]] = {
    "test": tuple("""
        ml-1m-debug ml-1m-base ml-1m-chat ml-2m-base ml-2m-chat ml-2m-code ml-5m-base ml-5m-chat
        ml-5m-code ml-10m-base ml-10m-chat ml-10m-code ml-20m-base ml-20m-chat ml-20m-code ml-20m-vl
        ml-20m-colab-chat ml-20m-tools ml-50m-base ml-50m-chat ml-50m-code ml-50m-vl ml-50m-kaggle-chat
        ml-50m-tools ml-50m-think""".split()),
    "mini": tuple("""
        ml-100m-base ml-100m-chat ml-100m-code ml-150m-chat ml-250m-base ml-250m-chat ml-250m-code
        ml-250m-vl ml-250m-tools ml-250m-kaggle-chat ml-300m-base ml-300m-chat ml-300m-code ml-300m-vl
        ml-300m-long ml-300m-tools ml-300m-think ml-300m-uz""".split()),
    "si": tuple("""
        ml-1b-base ml-1b-chat ml-1b-code ml-1b-agent ml-1b-think ml-2b-chat ml-2b-code ml-3b-base
        ml-3b-chat ml-3b-code ml-3b-agent ml-7b-base ml-7b-chat ml-7b-code ml-7b-agent ml-7b-long ml-7b-think
        ml-13b-base ml-13b-chat ml-13b-code ml-13b-agent ml-30b-base ml-30b-chat ml-30b-code ml-30b-vl
        ml-30b-vl-chat ml-70b-base ml-70b-chat ml-70b-code ml-70b-vl ml-70b-agent ml-70b-think ml-175b-base
        ml-175b-chat ml-175b-vl ml-405b-base ml-405b-chat ml-405b-vl ml-1t-base ml-1t-chat ml-1t-vl
        ml-16b-a2b-chat ml-16b-a2b-code ml-30b-a3b-base ml-30b-a3b-chat ml-30b-a3b-vl ml-100b-a10b-chat
        ml-100b-a10b-vl ml-235b-a22b-chat ml-235b-a22b-code ml-235b-a22b-vl ml-671b-a37b-chat
        ml-671b-a37b-vl ml-1t-a32b-base ml-1t-a32b-chat ml-1t-a32b-code ml-1t-a32b-vl""".split()),
}
ALL_NAMES: Tuple[str, ...] = tuple(n for names in TIER_NAMES.values() for n in names)

_NAME_RE = re.compile(r"^ml-(?P<size>\d+[mbt])(?:-a(?P<active>\d+[mbt]))?-(?P<variant>.+)$")

#: variant (name suffix) -> config ``kind``
VARIANT_KIND: Dict[str, str] = {
    "base": "base", "chat": "chat", "code": "code", "vl": "vl", "vl-chat": "vl-chat", "tools": "tools",
    "agent": "agent", "think": "think", "long": "long", "debug": "debug", "colab-chat": "chat",
    "kaggle-chat": "chat", "uz": "chat",
}

# ================================================================================================
# SPEC §10 shapes
# ================================================================================================


@dataclass(frozen=True)
class DenseShape:
    """SPEC §10 base shape: (V, d, L, H, hd, retention_ratio, Hkv, ffn, ctx, window, global_every)."""

    vocab: int
    d: int
    layers: int
    heads: int
    head_dim: int
    ret_ratio: float
    kv: int
    ffn: int
    ctx: int
    window: Optional[int] = None
    global_every: int = 1


DENSE: Dict[str, DenseShape] = {
    "1M": DenseShape(2048, 96, 6, 4, 24, 0.5, 1, 256, 256),
    "2M": DenseShape(2048, 128, 6, 4, 32, 0.5, 1, 352, 256),
    "5M": DenseShape(4096, 192, 8, 6, 32, 0.5, 1, 512, 512),
    "10M": DenseShape(8192, 256, 8, 8, 32, 0.5, 2, 704, 512),
    "20M": DenseShape(8192, 384, 10, 6, 64, 0.5, 1, 1024, 1024),
    "50M": DenseShape(16384, 512, 12, 8, 64, 0.5, 2, 1408, 1024),
    "100M": DenseShape(32768, 768, 12, 12, 64, 0.5, 2, 2048, 2048),
    "150M": DenseShape(32768, 768, 18, 12, 64, 0.5, 2, 2048, 2048),
    "250M": DenseShape(32768, 1024, 18, 16, 64, 0.5, 4, 2816, 2048),
    "300M": DenseShape(32768, 1024, 22, 16, 64, 0.5, 4, 2816, 2048),
    "1B": DenseShape(65536, 2048, 16, 32, 64, 0.375, 4, 5632, 4096, 4096, 4),
    "2B": DenseShape(65536, 2304, 24, 36, 64, 0.375, 4, 6144, 4096, 4096, 4),
    "3B": DenseShape(65536, 3072, 28, 24, 128, 0.25, 4, 8192, 8192, 4096, 4),
    "7B": DenseShape(131072, 4096, 32, 32, 128, 0.25, 8, 14336, 8192, 4096, 4),
    "13B": DenseShape(131072, 5120, 40, 40, 128, 0.25, 8, 13824, 8192, 4096, 4),
    "30B": DenseShape(131072, 6656, 60, 52, 128, 0.25, 13, 17920, 8192, 4096, 4),
    "70B": DenseShape(131072, 8192, 80, 64, 128, 0.25, 8, 28672, 8192, 4096, 4),
    "175B": DenseShape(131072, 12288, 96, 96, 128, 0.25, 8, 49152, 8192, 4096, 4),
    "405B": DenseShape(131072, 16384, 126, 128, 128, 0.25, 16, 53248, 8192, 4096, 4),
    "1T": DenseShape(131072, 24576, 128, 192, 128, 0.25, 16, 66560, 8192, 4096, 4),
}


@dataclass(frozen=True)
class MoEShape:
    """SPEC §10 MoE shape. ``kv`` is not given by SPEC for every model; 4 is used when missing.
    ``ffn`` is the dense SwiGLU hidden size of the first ``first_dense`` layers (0 = none)."""

    d: int
    layers: int
    heads: int
    head_dim: int
    kv: int
    experts: int
    top_k: int
    shared: int
    expert_hidden: int
    first_dense: int
    ffn: int


#: SI MoE: vocab 131072, ctx 8192, retention_ratio 0.25, window 4096 / global_every 4 (SPEC §10).
MOE: Dict[str, MoEShape] = {
    "16b-a2b": MoEShape(2048, 28, 16, 128, 4, 64, 6, 2, 1408, 1, 10944),
    "30b-a3b": MoEShape(2048, 48, 32, 128, 4, 128, 8, 0, 768, 0, 0),
    "100b-a10b": MoEShape(4096, 48, 32, 128, 4, 64, 6, 1, 2560, 1, 14336),
    "235b-a22b": MoEShape(4096, 94, 64, 128, 4, 128, 8, 0, 1536, 0, 0),
    "671b-a37b": MoEShape(7168, 61, 128, 128, 8, 256, 8, 1, 2048, 3, 18432),
    "1t-a32b": MoEShape(7168, 61, 64, 128, 8, 384, 8, 1, 2048, 1, 18432),
}
MOE_VOCAB = 131072
MOE_CTX = 8192

# Fitting tolerances. KEEP_*: the SPEC shape is kept unchanged when inside them. LIMIT_*: hard
# limits every generated config must satisfy (SPEC §10: dense/MoE total ±10 %, active ±20 %).
KEEP_TOTAL_TOL = 0.05
KEEP_ACTIVE_TOL = 0.10
#: MoE search: candidates must have total within KEEP_TOTAL_TOL and active within this.
SEARCH_ACTIVE_TOL = 0.15
LIMIT_TOTAL_TOL = 0.10
LIMIT_ACTIVE_TOL = 0.20

#: SPEC §9 benchmark tiers: (suite, lowest nominal size). Highest lower bound <= nominal wins.
SUITE_LOWER_BOUNDS: Tuple[Tuple[str, int], ...] = (
    ("cc-001", 10**6), ("cc-002", 5 * 10**6), ("cc-003", 50 * 10**6), ("cc-004", 150 * 10**6),
    ("cc-005", 10**9), ("cc-006", 7 * 10**9), ("cc-007", 16 * 10**9), ("cc-008", 100 * 10**9),
    ("cc-009", 405 * 10**9), ("cc-010", 10**12),
)

# ================================================================================================
# data mixtures (SPEC §6 / §13.1 source names)
# ================================================================================================

#: pretrain weights (sum 1.0). General: ~55 % Uzbek, 30 % English, 15 % code.
PRETRAIN_GENERAL = (("uz-news", 0.20), ("uz-wiki", 0.08), ("uz-telegram", 0.07), ("uz-books", 0.10),
                    ("fineweb-edu", 0.25), ("cosmopedia", 0.15), ("stack-python", 0.10), ("codeparrot", 0.05))
#: code variants: 40 % code (SPEC §10).
PRETRAIN_CODE = (("uz-news", 0.15), ("uz-wiki", 0.06), ("uz-telegram", 0.04), ("uz-books", 0.07),
                 ("fineweb-edu", 0.18), ("cosmopedia", 0.10), ("stack-python", 0.25), ("codeparrot", 0.15))
#: uz variant: Uzbek only.
PRETRAIN_UZ = (("uz-news", 0.40), ("uz-wiki", 0.15), ("uz-telegram", 0.15), ("uz-books", 0.30))

UZ_SFT = ("slimorca-uz", "oasst2-uz", "alpaca-uz", "uzbek-instruct", "sft-uz-9k")
CONV_SFT_EN = ("smoltalk", "smoltalk-everyday", "smoltalk-constraints")
CODE_SFT = ("smoltalk-code", "python-18k", "codefeedback")
TOOL_SFT = ("smoltalk-tools", "glaive-tools")

#: SFT examples per source at scale 1.0 (the mini tier); test / SI tiers scale this.
SFT_BASE_COUNTS: Dict[str, int] = {
    "slimorca-uz": 60000, "oasst2-uz": 10000, "alpaca-uz": 40000, "uzbek-instruct": 20000, "sft-uz-9k": 9000,
    "smoltalk": 60000, "smoltalk-everyday": 2260, "smoltalk-constraints": 20000,
    "smoltalk-code": 20000, "python-18k": 18000, "codefeedback": 20000,
    "smoltalk-tools": 20000, "glaive-tools": 15000,
    "smoltalk-math": 20000, "openthoughts": 20000, "smoltalk-magpie": 40000,
}
#: approximate dataset sizes (never ask for more examples than a source has).
SFT_CAPS: Dict[str, int] = {"sft-uz-9k": 9000, "smoltalk-everyday": 2260, "python-18k": 18600,
                            "smoltalk-constraints": 34000, "smoltalk-tools": 80000, "glaive-tools": 112000,
                            "openthoughts": 114000,
                            "smoltalk-math": 100000}

#: test-tier pretrain character budgets (Colab T4; text is re-sampled by random windows).
TEST_PRETRAIN_CHARS: Dict[str, float] = {"1M": 4e7, "2M": 6e7, "5M": 1e8, "10M": 1.5e8, "20M": 2.5e8, "50M": 5e8}
CHARS_PER_TOKEN = 4.0

#: test tier: (pretrain, sft) minutes for chat-capable configs; base configs use the sum.
TEST_MINUTES: Dict[str, Tuple[int, int]] = {"1M": (7, 3), "2M": (10, 5), "5M": (17, 8), "10M": (28, 12),
                                             "20M": (55, 20), "50M": (90, 30)}

GRPO_TASKS: Dict[str, List[str]] = {
    "chat": ["constraints", "detail", "arithmetic"], "vl-chat": ["constraints", "detail", "arithmetic"],
    "code": ["code_lite", "constraints"], "agent": ["constraints", "detail", "code_lite"],
    "tools": ["constraints", "detail", "code_lite"], "think": ["gsm8k", "arithmetic", "code_lite"],
}

TRAIN_COMMANDS = "moonlight train {name}  |  python: ml.train(\"{name}\")"

# ================================================================================================
# small helpers
# ================================================================================================


def sig(x: float, digits: int = 2) -> float:
    """Round to ``digits`` significant digits (deterministic, float)."""
    if x == 0:
        return 0.0
    return float(f"{x:.{digits - 1}e}")


def round_count(x: float, digits: int = 2, minimum: int = 1) -> int:
    """Round a count to ``digits`` significant digits (int, >= ``minimum``)."""
    return max(minimum, int(sig(x, digits)))


def suite_for(nominal: int) -> str:
    """SPEC §9 suite for a NOMINAL total parameter count."""
    chosen = SUITE_LOWER_BOUNDS[0][0]
    for suite, low in SUITE_LOWER_BOUNDS:
        if nominal >= low:
            chosen = suite
    return chosen


def largest_divisor_at_most(n: int, cap: int) -> int:
    """Largest divisor of ``n`` that is <= ``cap`` (>= 1)."""
    for k in range(min(n, cap), 0, -1):
        if n % k == 0:
            return k
    return 1


def size_label(token: str) -> str:
    """``"20m"`` -> ``"20M"``."""
    return token.upper()


def parse_name(name: str) -> Tuple[str, Optional[str], str]:
    """``ml-30b-a3b-vl`` -> ``("30B", "3B", "vl")``."""
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"bad config name {name!r}")
    active = m.group("active")
    return size_label(m.group("size")), (size_label(active) if active else None), m.group("variant")


def _model_cfg(d: Dict[str, Any]) -> ModelConfig:
    return ModelConfig.from_dict(d)


def _rel(value: int, target: int) -> float:
    return value / target - 1.0


# ================================================================================================
# shape fitting
# ================================================================================================


@dataclass
class FitResult:
    model: Dict[str, Any]
    total: int
    active: int
    notes: List[str] = field(default_factory=list)


def _dense_model_dict(s: DenseShape, nominal: int) -> Dict[str, Any]:
    n_ret = int(round(s.heads * s.ret_ratio))
    n_attn = s.heads - n_ret
    kv = largest_divisor_at_most(n_attn, s.kv) if n_attn > 0 else s.kv
    return {
        "vocab_size": s.vocab, "d_model": s.d, "n_layers": s.layers, "n_heads": s.heads, "head_dim": s.head_dim,
        "retention_ratio": s.ret_ratio, "n_kv_heads": kv, "attn_window": s.window, "global_every": s.global_every,
        "ffn_hidden": s.ffn, "tie_embeddings": nominal < 7 * 10**9,
    }


def fit_dense(label: str) -> FitResult:
    """Fit a dense size class to its nominal TOTAL (see module docstring)."""
    s = DENSE[label]
    nominal = parse_size(label)
    md = _dense_model_dict(s, nominal)
    notes: List[str] = []
    if md["n_kv_heads"] != s.kv:
        notes.append(f"n_kv_heads {s.kv}->{md['n_kv_heads']} (must divide the attention heads)")
    total = estimate_params(_model_cfg(md)).total
    if abs(_rel(total, nominal)) > KEEP_TOTAL_TOL:
        # total(f) = a + b*f exactly (every layer has a dense SwiGLU of width f)
        t1 = estimate_params(_model_cfg({**md, "ffn_hidden": 1024})).total
        t2 = estimate_params(_model_cfg({**md, "ffn_hidden": 2048})).total
        b = (t2 - t1) / 1024.0
        a = t1 - 1024 * b
        mult = 64 if s.d < 2048 else 128
        f = max(mult, int(round((nominal - a) / b / mult)) * mult)
        notes.append(f"ffn_hidden {s.ffn}->{f} (SPEC shape was {total / nominal:.3f}x nominal)")
        md["ffn_hidden"] = f
        total = estimate_params(_model_cfg(md)).total
    return FitResult(md, total, total, notes)


def _moe_model_dict(s: MoEShape, layers: int, he: int, shared: int, tie: bool) -> Dict[str, Any]:
    n_ret = int(round(s.heads * 0.25))
    n_attn = s.heads - n_ret
    return {
        "vocab_size": MOE_VOCAB, "d_model": s.d, "n_layers": layers, "n_heads": s.heads, "head_dim": s.head_dim,
        "retention_ratio": 0.25, "n_kv_heads": largest_divisor_at_most(n_attn, s.kv), "attn_window": 4096,
        "global_every": 4, "ffn_hidden": s.ffn, "tie_embeddings": tie,
        "moe": {"enabled": True, "n_experts": s.experts, "top_k": s.top_k, "n_shared": shared,
                "expert_hidden": he, "first_dense_layers": s.first_dense},
    }


def fit_moe(tag: str) -> FitResult:
    """Fit an MoE size class to its nominal TOTAL and ``-aNb`` ACTIVE label (see module docstring)."""
    s = MOE[tag]
    total_label, active_label = (size_label(x) for x in tag.split("-a"))
    nom_t, nom_a = parse_size(total_label), parse_size(active_label)
    tie = True
    if s.first_dense == 0:  # ffn_hidden unused by MoE layers; keep a harmless valid value
        s = MoEShape(s.d, s.layers, s.heads, s.head_dim, s.kv, s.experts, s.top_k, s.shared, s.expert_hidden,
                     s.first_dense, 4 * s.d)

    def counts(layers: int, he: int, shared: int) -> Tuple[int, int]:
        rep = estimate_params(_model_cfg(_moe_model_dict(s, layers, he, shared, tie)))
        return rep.total, rep.active

    t0, a0 = counts(s.layers, s.expert_hidden, s.shared)
    if abs(_rel(t0, nom_t)) <= KEEP_TOTAL_TOL and abs(_rel(a0, nom_a)) <= KEEP_ACTIVE_TOL:
        return FitResult(_moe_model_dict(s, s.layers, s.expert_hidden, s.shared, tie), t0, a0)

    shared_opts = list(range(s.shared, -1, -1))
    he_opts = list(range(max(128, (s.expert_hidden // 2) // 128 * 128), 3 * s.expert_hidden + 1, 128))
    l_lo = max(s.first_dense + 2, (s.layers + 1) // 2)
    best: Optional[Tuple[float, int, int, int, int, int]] = None
    for shared in shared_opts:
        for he in he_opts:
            # totals are exactly linear in the number of MoE layers
            ta, aa = counts(l_lo, he, shared)
            tb, ab = counts(l_lo + 1, he, shared)
            for layers in range(l_lo, s.layers + 1):
                t = ta + (tb - ta) * (layers - l_lo)
                a = aa + (ab - aa) * (layers - l_lo)
                et, ea = abs(_rel(t, nom_t)), abs(_rel(a, nom_a))
                if et > KEEP_TOTAL_TOL or ea > SEARCH_ACTIVE_TOL:
                    continue
                score = (0.5 * max(et / KEEP_TOTAL_TOL, ea / KEEP_ACTIVE_TOL)
                         + 2.0 * abs(math.log(layers / s.layers)) + abs(math.log(he / s.expert_hidden))
                         + 0.5 * (s.shared - shared))
                cand = (round(score, 9), -layers, he, shared, t, a)
                if best is None or cand < best:
                    best = cand
    if best is None:  # pragma: no cover - the grid always contains a feasible point for SPEC shapes
        raise RuntimeError(f"no MoE shape for {tag} within tolerance")
    _, neg_layers, he, shared, _, _ = best
    layers = -neg_layers
    md = _moe_model_dict(s, layers, he, shared, tie)
    total, active = counts(layers, he, shared)
    notes = [f"SPEC shape was total {t0 / nom_t:.3f}x / active {a0 / nom_a:.3f}x"]
    if layers != s.layers:
        notes.append(f"n_layers {s.layers}->{layers}")
    if he != s.expert_hidden:
        notes.append(f"expert_hidden {s.expert_hidden}->{he}")
    if shared != s.shared:
        notes.append(f"n_shared {s.shared}->{shared}")
    return FitResult(md, total, active, notes)


_FIT_CACHE: Dict[str, FitResult] = {}


def fitted(key: str) -> FitResult:
    if key not in _FIT_CACHE:
        _FIT_CACHE[key] = fit_moe(key) if key in MOE else fit_dense(key)
    return _FIT_CACHE[key]


# ================================================================================================
# recipe building
# ================================================================================================


@dataclass
class Spec:
    """Everything the builders need to know about one config name."""

    name: str
    tier: str
    size: str                   # nominal TOTAL label, e.g. "20M", "30B"
    active: Optional[str]       # MoE active label, e.g. "3B"
    variant: str
    kind: str
    nominal: int

    @property
    def is_moe(self) -> bool:
        return self.active is not None

    @property
    def shape_key(self) -> str:
        return f"{self.size.lower()}-a{self.active.lower()}" if self.is_moe else self.size

    @property
    def chat_capable(self) -> bool:
        return self.kind != "base"

    @property
    def multimodal(self) -> bool:
        return self.kind in ("vl", "vl-chat")

    @property
    def active_params(self) -> int:
        return parse_size(self.active) if self.active else self.nominal


def make_spec(name: str, tier: str) -> Spec:
    size, active, variant = parse_name(name)
    if variant not in VARIANT_KIND:
        raise ValueError(f"{name}: unknown variant {variant!r}")
    return Spec(name, tier, size, active, variant, VARIANT_KIND[variant], parse_size(size))


# ---- hardware / distributed ---------------------------------------------------------------------


def hardware(spec: Spec) -> Tuple[str, Dict[str, Any], str]:
    """``(preset, distributed, human description)`` per SPEC §10."""
    n = spec.nominal
    if spec.tier == "test":
        if spec.variant == "debug":
            return "cpu", _dist("none"), "CPU only (offline, a few minutes)"
        if spec.variant == "kaggle-chat":
            return "kaggle-2xt4", _dist("none"), ("Kaggle 2x T4 (single-process by default; add torchrun "
                                                  "--nproc_per_node 2 + --set train.distributed.strategy=ddp)")
        return "colab-t4", _dist("none"), "Colab T4 16 GB (single GPU)"
    if spec.tier == "mini":
        if spec.variant == "kaggle-chat" or n <= 150 * 10**6:
            return "kaggle-2xt4", _dist("ddp"), "Kaggle 2x T4 (torchrun --nproc_per_node 2, DDP)"
        return "a100-1", _dist("ddp"), "1x A100 80 GB (DDP-ready: torchrun --nproc_per_node N)"
    if spec.is_moe:
        experts = MOE[spec.shape_key].experts
        ep = min(64, max(8, experts // 4))
        if n <= 100 * 10**9:
            gpus = 512
        elif n <= 235 * 10**9:
            gpus = 1024
        else:
            gpus = 2048
        return (f"h100-{gpus}", _dist("fsdp", tp=8, ep=ep),
                f"{gpus}x H100 (FSDP; tp_size 8 / ep_size {ep} are documented placeholders)")
    if n <= 3 * 10**9:
        return "a100-8", _dist("fsdp"), "8x A100 80 GB (FSDP)"
    if n <= 7 * 10**9:
        return "h100-8", _dist("fsdp"), "8x H100 (FSDP)"
    if n <= 13 * 10**9:
        return "h100-16", _dist("fsdp"), "16x H100, 2 nodes (FSDP)"
    if n <= 30 * 10**9:
        return "h100-64", _dist("fsdp", tp=8), "64x H100 (FSDP; tp_size 8 is a documented placeholder)"
    if n <= 70 * 10**9:
        return "h100-256", _dist("fsdp", tp=8), "256x H100 (FSDP; tp_size 8 is a documented placeholder)"
    gpus = 512 if n <= 175 * 10**9 else (1024 if n <= 405 * 10**9 else 2048)
    return f"h100-{gpus}", _dist("fsdp", tp=8), f"{gpus}x H100 (FSDP; tp_size 8 is a documented placeholder)"


def _dist(strategy: str, tp: int = 1, ep: int = 1) -> Dict[str, Any]:
    return {"strategy": strategy, "fsdp_shard": "full", "tp_size": tp, "ep_size": ep}


# ---- model ---------------------------------------------------------------------------------------


def model_section(spec: Spec) -> Dict[str, Any]:
    fit = fitted(spec.shape_key)
    md = dict(fit.model)
    if spec.is_moe:
        ctx = MOE_CTX
        md["moe"] = {**md["moe"], "aux_loss_weight": 0.001, "bias_update_rate": 0.001, "capacity_factor": 0.0}
    else:
        ctx = DENSE[spec.size].ctx
    rope = 10000.0 if ctx <= 2048 else 500000.0
    if spec.variant == "long":
        if spec.tier == "mini":
            ctx, rope = 8192, 1.0e6
        else:
            ctx, rope = 131072, 5.0e6
    d = md["d_model"]
    init_std = 0.02 if d <= 2304 else round(0.5 / math.sqrt(d), 4)
    out: Dict[str, Any] = {
        "vocab_size": md["vocab_size"], "d_model": d, "n_layers": md["n_layers"], "n_heads": md["n_heads"],
        "head_dim": md["head_dim"], "retention_ratio": md["retention_ratio"], "n_kv_heads": md["n_kv_heads"],
        "attn_window": md["attn_window"], "global_every": md["global_every"], "ffn_hidden": md["ffn_hidden"],
        "token_shift": 4, "value_residual": True, "qk_norm": True, "output_gate": True, "rope_theta": rope,
        "max_seq_len": ctx, "tie_embeddings": md["tie_embeddings"], "softcap": 30.0, "mtp_depth": 1,
        "mtp_block": False, "mtp_weight": 0.3, "z_loss": 1.0e-4 if spec.tier == "si" else 0.0,
        "init_std": init_std,
        "moe": md.get("moe", {"enabled": False, "n_experts": 0, "top_k": 0, "n_shared": 0, "expert_hidden": 0,
                              "first_dense_layers": 0, "aux_loss_weight": 0.001, "bias_update_rate": 0.001,
                              "capacity_factor": 0.0}),
        "activation_checkpointing": spec.tier == "si" and spec.nominal >= 7 * 10**9,
    }
    return out


def multimodal_section(spec: Spec) -> Dict[str, Any]:
    return {
        "enabled": spec.multimodal,
        "vision_encoder": "google/siglip-base-patch16-224",
        "vision_dim": 768,
        "n_image_tokens": 196 if spec.tier == "si" else 49,
        "freeze_encoder": True,
    }


def tokenizer_section(spec: Spec, vocab: int) -> Dict[str, Any]:
    if spec.variant == "debug":
        chars = 2e6
    else:
        chars = {2048: 1e7, 4096: 1e7, 8192: 2e7, 16384: 3e7, 32768: 5e7, 65536: 1e8}.get(vocab, 2e8)
    return {"vocab_size": vocab, "path": None, "train_chars": int(chars), "split_digits": True}


# ---- budgets ---------------------------------------------------------------------------------------


def tokens_per_param(spec: Spec) -> int:
    """Pretraining tokens per nominal TOTAL parameter (mini ~20-40, SI Chinchilla 20+)."""
    n = spec.nominal
    if spec.tier == "mini":
        return 20 if n <= 150 * 10**6 else 30
    if n < 7 * 10**9:
        return 50
    if n < 100 * 10**9:
        return 30
    return 20


def pretrain_tokens(spec: Spec) -> float:
    return float(tokens_per_param(spec) * spec.nominal)


def long_tokens(spec: Spec) -> float:
    return sig(0.05 * pretrain_tokens(spec)) if spec.variant == "long" else 0.0


def test_minutes(spec: Spec) -> Tuple[int, int, int, int]:
    """Test tier: (pretrain, align, sft, dpo) minutes (Colab T4 / Kaggle T4)."""
    pre, sft = TEST_MINUTES[spec.size]
    align = dpo = 0
    if spec.variant == "base":
        return pre + sft, 0, 0, 0
    if spec.variant == "colab-chat":
        return 60, 0, 25, 0
    if spec.variant == "kaggle-chat":
        return 240, 0, 60, 0
    if spec.variant == "vl":
        align = 5
        pre -= align
    if spec.name in ("ml-50m-chat", "ml-50m-tools", "ml-50m-think"):
        dpo = 5
        pre -= dpo
    return pre, align, sft, dpo


def total_pretrain_chars(spec: Spec) -> float:
    if spec.variant == "debug":
        return 4e6
    if spec.tier == "test":
        chars = TEST_PRETRAIN_CHARS[spec.size]
        return chars * 2 if spec.variant == "kaggle-chat" else chars
    return (pretrain_tokens(spec) + long_tokens(spec)) * CHARS_PER_TOKEN


# ---- data --------------------------------------------------------------------------------------------


def sft_scale(spec: Spec) -> float:
    if spec.tier == "test":
        return {"1M": 0.05, "2M": 0.05, "5M": 0.1, "10M": 0.1, "20M": 0.25, "50M": 0.5}[spec.size]
    return 1.0 if spec.tier == "mini" else 3.0


def sft_sources(spec: Spec) -> List[Dict[str, Any]]:
    """SFT mixture (``{source, max_examples}``); chat-capable variants always include conversation +
    code data (SPEC §10/§13.8). Base configs get a small set used for the anneal mix only."""
    scale = sft_scale(spec)
    v, si = spec.variant, spec.tier == "si"
    if v == "debug":
        names: List[str] = ["slimorca-uz", "alpaca-uz", "smoltalk-everyday", "python-18k", "smoltalk-tools"]
        mult: Dict[str, float] = {}
        scale = 0.01
    elif v == "base":
        names = ["slimorca-uz", "sft-uz-9k", "smoltalk-everyday", "python-18k"]
        mult = {}
        scale *= 0.2
    elif v == "uz":
        names = list(UZ_SFT)
        mult = {s: 1.5 for s in UZ_SFT}
    else:
        names = list(UZ_SFT) + list(CONV_SFT_EN) + list(CODE_SFT) + list(TOOL_SFT)
        mult = {}
        if v == "code":
            mult.update({s: 2.5 for s in CODE_SFT})
            mult.update({s: 0.5 for s in UZ_SFT + ("smoltalk",)})
        if v in ("tools", "agent"):
            mult.update({s: 3.0 for s in TOOL_SFT})
            mult["smoltalk-constraints"] = 1.5
        if v == "think" or (si and v in ("chat", "vl-chat")):
            names.append("smoltalk-math")
        if v == "think" or (si and v in ("chat", "code", "vl-chat")):
            names.append("openthoughts")
        if v == "think":
            mult.update({"smoltalk-math": 1.5, "openthoughts": 1.5})
        if si:
            names.append("smoltalk-magpie")
    out = []
    for s in names:
        n = SFT_BASE_COUNTS[s] * scale * mult.get(s, 1.0)
        n = min(n, SFT_CAPS.get(s, n))
        out.append({"source": s, "max_examples": round_count(n, minimum=50)})
    return out


def synthetic_section(spec: Spec) -> Dict[str, Any]:
    """``data.synthetic`` generator counts (keys of ``moonlight.train.trainer.SYNTHETIC_GENERATORS``)."""
    v = spec.variant
    if v == "debug":
        return {"identity": 40, "greetings": 40, "knowledge": 60, "arithmetic": 100, "tool_calls": 60, "code": 40,
                "think": 20, "plan": 10, "constraints": 20, "detail": 20, "projects": 4}
    if v == "base":
        return {"identity": True, "knowledge": True, "arithmetic": 500, "tool_calls": 200, "code": 200}
    m = {"test": 1, "mini": 2, "si": 4}[spec.tier]
    c = {"arithmetic": 2000, "tool_calls": 1000, "code": 600, "think": 0, "plan": 100, "constraints": 300,
         "detail": 200, "projects": 20}
    if v == "code":
        c.update(code=1800, projects=40)
    if v == "tools":
        c.update(tool_calls=3000, plan=200, constraints=450, detail=400, projects=60)
    if v == "agent":
        c.update(tool_calls=3000, plan=300, constraints=600, detail=400, projects=100)
    if v == "think":
        c.update(think=2000, arithmetic=3000, plan=150)
    out: Dict[str, Any] = {"identity": True, "knowledge": True}
    for k in ("arithmetic", "tool_calls", "code", "think", "plan", "constraints", "detail", "projects"):
        if c[k]:
            out[k] = c[k] * m
    return out


def data_section(spec: Spec) -> Dict[str, Any]:
    if spec.variant == "uz":
        mix = PRETRAIN_UZ
    elif spec.variant == "code":
        mix = PRETRAIN_CODE
    else:
        mix = PRETRAIN_GENERAL
    chars = total_pretrain_chars(spec)
    pretrain = [{"source": s, "weight": w, "max_chars": sig(w * chars)} for s, w in mix]
    vision: List[Dict[str, Any]] = []
    if spec.multimodal:
        if spec.tier == "test":
            vision = [{"source": "flickr8k", "max_images": 6000}, {"source": "flickr30k", "max_images": 4000}]
        else:
            vision = [{"source": "flickr8k", "max_images": 8091}, {"source": "flickr30k", "max_images": 31783}]
    return {
        "cache_dir": "~/.cache/moonlight",
        "offline": spec.variant == "debug",
        "languages": ["uz"] if spec.variant == "uz" else ["uz", "en"],
        "pretrain": pretrain,
        "sft": sft_sources(spec),
        "synthetic": synthetic_section(spec),
        "vision": vision,
    }


# ---- phases -------------------------------------------------------------------------------------------


def dpo_phase(spec: Spec, minutes: Optional[int] = None) -> Dict[str, Any]:
    if spec.tier == "test":
        n_uf, n_syn = 2000, 500
    elif spec.tier == "mini":
        n_uf, n_syn = 20000, 2000
    else:
        n_uf, n_syn = 60000, 5000
    p: Dict[str, Any] = {"name": "dpo", "sources": [{"source": "ultrafeedback", "max_examples": n_uf}],
                         "synthetic_pairs": n_syn, "beta": 0.1, "loss": "dpo"}
    if minutes is not None:
        p["minutes"] = minutes
    p["epochs"] = 1.0
    p["lr_scale"] = 0.1
    return p


def grpo_phase(spec: Spec) -> Dict[str, Any]:
    return {
        "name": "grpo", "tasks": list(GRPO_TASKS[spec.variant]), "group_size": 8,
        "prompts_per_step": 16, "max_new_tokens": 512 if spec.variant in ("think", "agent") else 256,
        "temperature": 1.0, "kl_coef": 0.02, "clip": 0.2,
        "steps": 300 if spec.nominal < 7 * 10**9 else 500, "lr_scale": 0.05,
    }


def phases(spec: Spec) -> List[Dict[str, Any]]:
    v, tier = spec.variant, spec.tier
    chat_mix = 0.1 if v == "base" else 0.25
    if v == "debug":
        return [
            {"name": "pretrain", "steps": 40, "lr_scale": 1.0, "sft_mix": 0.0, "anneal_sft_mix": 0.25},
            {"name": "sft", "steps": 20, "epochs": 1.0, "lr_scale": 0.5, "replay": 0.1},
        ]
    if tier == "test":
        pre, align, sft, dpo = test_minutes(spec)
        out = [{"name": "pretrain", "minutes": pre, "lr_scale": 1.0, "sft_mix": 0.0, "anneal_sft_mix": chat_mix}]
        if v == "base":
            return out
        if align:
            out.append({"name": "align", "minutes": align, "epochs": 1.0, "lr_scale": 1.0, "trainable": "vision_proj"})
        s = {"name": "sft", "minutes": sft, "epochs": 2.0, "lr_scale": 0.5, "replay": 0.1}
        if spec.multimodal:
            s["image_mix"] = 0.3
        out.append(s)
        if dpo:
            out.append(dpo_phase(spec, minutes=dpo))
        return out

    tokens = pretrain_tokens(spec)
    pre: Dict[str, Any] = {"name": "pretrain"}
    if tier == "si" and v == "base":
        pre["tokens"] = sig(0.9 * tokens, 3)
    else:
        pre["tokens"] = tokens
    if v == "kaggle-chat":
        pre["minutes"] = 540
    pre.update({"lr_scale": 1.0, "sft_mix": 0.0, "anneal_sft_mix": chat_mix})
    out = [pre]
    if v == "base":
        if tier == "si":
            out.append({"name": "anneal", "tokens": sig(0.1 * tokens, 3), "lr_scale": 1.0, "anneal_sft_mix": 0.1})
        return out
    if v == "long":
        out.append({"name": "long", "seq_len": 8192 if tier == "mini" else 131072, "tokens": long_tokens(spec),
                    "lr_scale": 0.1})
    if spec.multimodal:
        out.append({"name": "align", "epochs": 1.0, "lr_scale": 1.0, "trainable": "vision_proj"})
    s = {"name": "sft"}
    if v == "kaggle-chat":
        s["minutes"] = 120
    s.update({"epochs": 2.0, "lr_scale": 0.5 if tier == "mini" else 0.25, "replay": 0.1})
    if spec.multimodal:
        s["image_mix"] = 0.3
    out.append(s)
    if tier == "mini" and spec.nominal >= 150 * 10**6 and v in ("chat", "kaggle-chat", "tools", "think"):
        out.append(dpo_phase(spec, minutes=60 if v == "kaggle-chat" else None))
    if tier == "si" and v in GRPO_TASKS:
        out.append(dpo_phase(spec))
        out.append(grpo_phase(spec))
    return out


def train_section(spec: Spec, ctx: int) -> Dict[str, Any]:
    v, tier, n = spec.variant, spec.tier, spec.nominal
    preset, dist, _ = hardware(spec)
    if v == "debug":
        seq, bs, bt = 128, 8, 1024
    elif tier == "test":
        seq = min(ctx, 512)
        bs = {"1M": 64, "2M": 64, "5M": 32, "10M": 32, "20M": 32, "50M": 16}[spec.size]
        bt = 32768 if spec.size == "50M" else 16384
    elif tier == "mini":
        seq = 1024
        bs = 16 if n <= 150 * 10**6 else 8
        bt = 2**19
    else:
        seq = 4096
        if n <= 1 * 10**9:
            bs = 8
        elif n <= 3 * 10**9:
            bs = 4
        elif n <= 13 * 10**9 and not spec.is_moe:
            bs = 2
        else:
            bs = 1
        bt = 2**21 if n <= 3 * 10**9 else (2**22 if n <= 13 * 10**9 else 2**23)

    if tier == "test":
        opt = {"lr": 0.02, "adam_lr": 0.003, "embed_lr": 0.003, "weight_decay": 0.0}
        sched = {"warmup_steps": 50 if spec.size in ("1M", "2M") else 100, "decay_frac": 0.35, "min_lr_ratio": 0.02}
    elif tier == "mini":
        opt = {"lr": 0.015, "adam_lr": 0.002, "embed_lr": 0.002, "weight_decay": 0.0}
        sched = {"warmup_steps": 500, "decay_frac": 0.25, "min_lr_ratio": 0.02}
    else:
        if n <= 3 * 10**9:
            lr, alr = 0.01, 1.0e-3
        elif n < 100 * 10**9 and not spec.is_moe:
            lr, alr = 0.006, 6.0e-4
        else:
            lr, alr = 0.004, 4.0e-4
        opt = {"lr": lr, "adam_lr": alr, "embed_lr": alr, "weight_decay": 0.1}
        sched = {"warmup_steps": 2000, "decay_frac": 0.2, "min_lr_ratio": 0.01}

    if v == "debug":
        precision, compile_, ema = "fp32", False, 0.0
        curriculum = {"enabled": False, "start_seq_len": 64, "until_frac": 0.3}
        eval_every, eval_batches, log_every, ckpt = 20, 2, 5, 1.0
        sched = {"warmup_steps": 5, "decay_frac": 0.35, "min_lr_ratio": 0.02}
    elif tier == "test":
        precision = "fp16" if v in ("colab-chat", "kaggle-chat") else "auto"
        compile_, ema = True, 0.998
        curriculum = {"enabled": True, "start_seq_len": min(128, seq // 2), "until_frac": 0.3}
        eval_every, eval_batches, log_every = 250, 20, 50
        ckpt = 5.0 if v in ("colab-chat", "kaggle-chat") else 10.0
    elif tier == "mini":
        precision = "fp16" if v == "kaggle-chat" else "auto"
        compile_, ema = True, 0.999
        curriculum = {"enabled": True, "start_seq_len": 256, "until_frac": 0.2}
        eval_every, eval_batches, log_every, ckpt = 1000, 50, 100, 30.0
    else:
        precision, compile_, ema = "bf16", True, 0.0
        curriculum = {"enabled": True, "start_seq_len": 1024, "until_frac": 0.1}
        eval_every, eval_batches, log_every, ckpt = 2000, 100, 10, 60.0

    return {
        "phases": phases(spec),
        "seq_len": seq,
        "batch_size": bs,
        "batch_tokens": bt,
        "optimizer": {"name": "muon", "lr": opt["lr"], "momentum": 0.95, "adam_lr": opt["adam_lr"],
                      "embed_lr": opt["embed_lr"], "betas": [0.9, 0.95], "weight_decay": opt["weight_decay"],
                      "grad_clip": 1.0},
        "schedule": {"name": "wsd", **sched},
        "precision": precision,
        "compile": compile_,
        "ema": ema,
        "curriculum": curriculum,
        "eval_every": eval_every,
        "eval_batches": eval_batches,
        "log_every": log_every,
        "checkpoint_every_min": ckpt,
        "output_dir": f"runs/{spec.name}",
        "seed": 1337,
        "hardware": preset,
        "distributed": dist,
    }


def chat_section(spec: Spec) -> Dict[str, Any]:
    v, tier = spec.variant, spec.tier
    codeish = v in ("code", "tools", "agent", "think")
    tiny = tier == "test" and spec.nominal <= 10 * 10**6
    return {
        "name": "Moonlight",
        "system_prompt": ("Sen Moonlight — o'zbek va ingliz tillarida samimiy, aniq va foydali javob beradigan, "
                          "kod yoza oladigan va kerak bo'lganda vositalardan (tools) foydalanadigan sun'iy intellekt "
                          "yordamchisisan."),
        "best_of": 1 if tier == "si" else 3,
        "temperature": 0.2 if v == "code" else 0.7,
        "top_k": 40,
        "top_p": 0.95 if v == "code" else 0.9,
        "repetition_penalty": 1.05 if tier == "si" or codeish else 1.1,
        # n-gram blocking breaks code (identifiers repeat); only the tiniest chat models use it.
        "no_repeat_ngram": 3 if tiny and not codeish else 0,
        "max_new_tokens": 1024 if (tier == "si" or v in ("think", "agent")) else 256,
        "tools": True,
    }


# ---- descriptions -----------------------------------------------------------------------------------

_VARIANT_TEXT = {
    "base": "base model (pretraining only; a small SFT anneal mix keeps the chat format usable)",
    "chat": "chat model (Uzbek + English conversation, coding and tool calling)",
    "code": "code model (40% code pretraining + code-heavy SFT; still a chat model)",
    "vl": "vision-language model (image input via frozen SigLIP + projector; conversation + code SFT)",
    "vl-chat": "vision-language chat model (image input; full chat + alignment recipe)",
    "tools": "tool-calling model (tool-call-heavy SFT: glaive + apigen + synthetic tool loops)",
    "agent": "agentic model (tool use, planning and workspace project building)",
    "think": "reasoning model (<|think|> traces, math and verifiable rewards)",
    "long": "long-context chat model (extra long-sequence phase before SFT)",
    "uz": "Uzbek-only chat model (Uzbek pretraining + Uzbek SFT)",
    "debug": "debug recipe (CPU, offline bundled data, tiny step budgets; used by the test-suite)",
    "colab-chat": "chat model tuned for a free Colab T4 session (fp16, compile, checkpoints every 5 min)",
    "kaggle-chat": "chat model tuned for a Kaggle 2x T4 session (fp16, compile, time budgets)",
}


def describe(spec: Spec) -> str:
    size = f"{spec.size} ({spec.active} active) MoE" if spec.is_moe else spec.size
    text = f"Moonlight {size} hybrid attention+retention {_VARIANT_TEXT[spec.variant]}."
    if spec.tier == "si":
        text += " Reference recipe for a large cluster."
    return text


def header(spec: Spec, cfg: Config) -> str:
    _, _, hw = hardware(spec)
    if spec.tier == "test" and spec.variant != "debug":
        budget = f"~{sum(int(p.get('minutes', 0) or 0) for p in cfg.train.phases)} min"
    elif spec.variant == "debug":
        budget = f"{sum(int(p.get('steps', 0) or 0) for p in cfg.train.phases)} optimizer steps"
    else:
        budget = f"{format_size(train_tokens(cfg))} LM training tokens"
    return (f"# {spec.name}: {cfg.description} [generated by scripts/generate_configs.py - do not edit]\n"
            f"# Hardware: {hw}; budget {budget}; benchmark {cfg.benchmark}.\n"
            f"# Train: {TRAIN_COMMANDS.format(name=spec.name)}\n")


def train_tokens(cfg: Config) -> float:
    """Sum of explicit token budgets of the LM phases (pretrain / anneal / long)."""
    total = 0.0
    for p in cfg.train.phases:
        if str(p.get("name")) in ("pretrain", "anneal", "long") and p.get("tokens"):
            total += float(p["tokens"])
    return total


# ================================================================================================
# assembly, validation, docs
# ================================================================================================


@dataclass
class Built:
    spec: Spec
    config: Config
    text: str
    total: int
    active: int


def build_config(spec: Spec) -> Built:
    model = model_section(spec)
    raw = {
        "name": spec.name,
        "description": describe(spec),
        "tier": spec.tier,
        "kind": spec.kind,
        "benchmark": suite_for(spec.nominal),
        "nominal_params": spec.size,
        "model": model,
        "multimodal": multimodal_section(spec),
        "tokenizer": tokenizer_section(spec, model["vocab_size"]),
        "data": data_section(spec),
        "train": train_section(spec, model["max_seq_len"]),
        "chat": chat_section(spec),
    }
    cfg = Config.from_dict(raw)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # a nominal-size warning must never slip through
        cfg.validate(strict=True)
    rep = estimate_params(cfg)
    check_tolerances(spec, rep.total, rep.active)
    text = header(spec, cfg) + cfg.to_yaml()
    if not text.endswith("\n"):
        text += "\n"
    # round-trip guard: what we write must load back to the same config
    back = Config.from_yaml(text)
    if back.to_dict() != cfg.to_dict():
        raise RuntimeError(f"{spec.name}: YAML round-trip mismatch")
    return Built(spec, cfg, text, rep.total, rep.active)


def check_tolerances(spec: Spec, total: int, active: int) -> None:
    rt = _rel(total, spec.nominal)
    if abs(rt) > LIMIT_TOTAL_TOL:
        raise RuntimeError(f"{spec.name}: total {format_size(total)} is {1 + rt:.3f}x nominal {spec.size}")
    if spec.is_moe:
        ra = _rel(active, spec.active_params)
        if abs(ra) > LIMIT_ACTIVE_TOL:
            raise RuntimeError(f"{spec.name}: active {format_size(active)} is {1 + ra:.3f}x label a{spec.active}")


def build_all() -> List[Built]:
    out = [build_config(make_spec(name, tier)) for tier, names in TIER_NAMES.items() for name in names]
    if len(out) != 100 or len({b.spec.name for b in out}) != 100:
        raise RuntimeError(f"expected exactly 100 unique configs, built {len(out)}")
    return out


def _heads_text(m: ModelConfig) -> str:
    return f"{m.n_attention_heads}/{m.n_retention_heads}"


def _experts_text(m: ModelConfig) -> str:
    if not m.moe.enabled:
        return "–"
    shared = f"+{m.moe.n_shared}s" if m.moe.n_shared else ""
    return f"{m.moe.n_experts}×{m.moe.expert_hidden} (top-{m.moe.top_k}{shared})"


def _ctx_text(n: int) -> str:
    return f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n)


def _budget_text(b: Built) -> str:
    if b.spec.variant == "debug":
        steps = sum(int(p.get("steps", 0) or 0) for p in b.config.train.phases)
        return f"{steps} steps (CPU)"
    if b.spec.tier == "test":
        minutes = sum(int(p.get("minutes", 0) or 0) for p in b.config.train.phases)
        return f"{minutes} min (time)"
    text = format_size(train_tokens(b.config))
    if b.spec.variant == "kaggle-chat":
        text += " / 540 min"
    return text


def render_docs(built: Sequence[Built]) -> str:
    lines = [
        "# Moonlight configs",
        "",
        "<!-- Generated by scripts/generate_configs.py - do not edit by hand. "
        "Regenerate with `python scripts/generate_configs.py` (CI runs `--check`). -->",
        "",
        "The 100 training recipes shipped in `configs/{tier}/{name}.yaml` (SPEC §10). Every config is",
        "fully explicit (no `extends:`), validates, and its exact parameter count (from",
        "`moonlight.params.estimate_params`) is within ±10 % of the nominal size; MoE active counts are",
        "within ±20 % of the `-aNb` label.",
        "",
        "Train any of them with `moonlight train <name>` or `ml.train(\"<name>\")`; inspect one with",
        "`moonlight info <name>` / `moonlight params <name>`.",
        "",
        "* **test** (25): Colab / Kaggle T4 recipes with time budgets (plus `ml-1m-debug`, a CPU-only",
        "  offline recipe used by the test-suite).",
        "* **mini** (18): 100M–300M models, ~20–30 pretraining tokens per parameter, single-node DDP.",
        "* **si** (57): 1B–1T dense and MoE reference recipes for large clusters (FSDP; `tp_size` /",
        "  `ep_size` are documented placeholders - tensor and expert parallelism are not implemented).",
        "  These are recipes, not released checkpoints: nothing here claims measured quality.",
        "",
        "Every model except `*-base` is trained on conversation **and** code (Uzbek + English SFT,",
        "code SFT, tool calls, planning / constraint / detail / project synthetic data). SI chat-capable",
        "variants end with `sft → dpo → grpo` (SPEC §13.8); mini 150M–300M chat variants and the 50M",
        "chat / tools / think configs add a DPO phase.",
        "",
        "Columns: *heads* = attention / retention query heads, *kv* = KV heads of the attention heads,",
        "*experts* = routed experts × expert hidden size (top-k, `+Ns` shared experts), *train* =",
        "explicit pretraining token budget (test tier: total minutes on the target GPU).",
        "",
        "| name | tier | kind | nominal | est. total | active | d_model | layers | heads (attn/ret) | kv | experts "
        "| ctx | vocab | benchmark | hardware | train |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|---|---|---|",
    ]
    for b in built:
        m = b.config.model
        nominal = f"{b.spec.size} (a{b.spec.active})" if b.spec.is_moe else b.spec.size
        lines.append(
            f"| `{b.spec.name}` | {b.spec.tier} | {b.config.kind} | {nominal} | {format_size(b.total)} | "
            f"{format_size(b.active)} | {m.d_model} | {m.n_layers} | {_heads_text(m)} | {m.n_attn_kv_heads} | "
            f"{_experts_text(m)} | {_ctx_text(m.max_seq_len)} | {m.vocab_size} | {b.config.benchmark} | "
            f"{b.config.train.hardware} | {_budget_text(b)} |")
    lines += ["", "## Shape adjustments", "",
              "Shapes follow the SPEC §10 tables; the generator only changes what is needed to land on the",
              "nominal size (see the docstring of `scripts/generate_configs.py`):", ""]
    any_note = False
    for key in list(DENSE) + list(MOE):
        if key not in _FIT_CACHE:
            continue
        notes = _FIT_CACHE[key].notes
        if notes:
            any_note = True
            lines.append(f"* `{key}`: " + "; ".join(notes) + ".")
    if not any_note:
        lines.append("* none")
    lines.append("")
    return "\n".join(lines)


def summary_table(built: Sequence[Built]) -> str:
    rows = ["name                     total      ratio  active     ratio  bench   hardware"]
    for b in built:
        ratio_t = b.total / b.spec.nominal
        if b.spec.is_moe:
            ratio_a = f"{b.active / b.spec.active_params:6.3f}"
            act = format_size(b.active)
        else:
            ratio_a, act = "     -", "-"
        rows.append(f"{b.spec.name:24s} {format_size(b.total):>8s}  {ratio_t:6.3f}  {act:>8s}  {ratio_a}  "
                    f"{b.config.benchmark}  {b.config.train.hardware}")
    return "\n".join(rows)


# ================================================================================================
# CLI
# ================================================================================================


def expected_files(built: Sequence[Built]) -> Dict[str, str]:
    """Relative path -> content for every generated file."""
    files = {f"configs/{b.spec.tier}/{b.spec.name}.yaml": b.text for b in built}
    files["docs/CONFIGS.md"] = render_docs(built)
    return files


def _existing_config_files(root: Path) -> List[str]:
    out = []
    for tier in TIER_NAMES:
        d = root / "configs" / tier
        if d.is_dir():
            out += [f"configs/{tier}/{p.name}" for p in sorted(d.iterdir())
                    if p.is_file() and p.suffix.lower() in (".yaml", ".yml")]
    return out


def check(root: Path, files: Dict[str, str]) -> int:
    problems: List[str] = []
    for rel, content in files.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
            continue
        current = path.read_text(encoding="utf-8")
        if current != content:
            diff = list(difflib.unified_diff(current.splitlines(), content.splitlines(), f"{rel} (committed)",
                                             f"{rel} (generated)", lineterm="", n=1))
            problems.append(f"differs: {rel}\n" + "\n".join(diff[:30]))
    for rel in _existing_config_files(root):
        if rel not in files:
            problems.append(f"extra: {rel} (not produced by the generator)")
    if problems:
        print("\n".join(problems))
        print(f"\n{len(problems)} problem(s); run `python scripts/generate_configs.py` to regenerate.")
        return 1
    print(f"OK: {len(files) - 1} configs and docs/CONFIGS.md are up to date.")
    return 0


def write(root: Path, files: Dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8", newline="\n")
    removed = [rel for rel in _existing_config_files(root) if rel not in files]
    for rel in removed:  # keep the tier directories at exactly the generated set
        (root / rel).unlink()
    print(f"wrote {len(files) - 1} configs to {root / 'configs'} and {root / 'docs' / 'CONFIGS.md'}"
          + (f"; removed {len(removed)} stale file(s)" if removed else ""))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the 100 Moonlight configs (SPEC §10) and docs/CONFIGS.md.")
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory; exit 1 if any file differs or files are missing / extra")
    parser.add_argument("--out", default=str(REPO_ROOT), metavar="DIR",
                        help="output root: writes DIR/configs/{tier}/*.yaml and DIR/docs/CONFIGS.md "
                             "(default: the repository root)")
    parser.add_argument("--summary", action="store_true", help="print the parameter / tolerance table")
    args = parser.parse_args(argv)

    built = build_all()
    files = expected_files(built)
    if args.summary:
        print(summary_table(built))
    root = Path(args.out).expanduser().resolve()
    if args.check:
        return check(root, files)
    write(root, files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
