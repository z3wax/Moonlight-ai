"""Learning-rate, momentum, curriculum and budget schedules (SPEC §7). Pure python (no torch).

All schedules return a *multiplier* applied to every optimizer param group's ``base_lr``::

    mult = lr_multiplier(step, progress, cfg.train.schedule)
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * mult

* ``step``     - optimizer steps already taken in the current phase (0-based). Warmup is by step:
  ``min(1, (step + 1) / warmup_steps)``.
* ``progress`` - fraction of the phase budget consumed, in ``[0, 1]``:
  ``max(step / max_steps, tokens / token_budget, seconds / time_budget, epochs_done / epochs)``
  over whichever budgets the phase defines (see :func:`phase_progress`).

Schedules (``cfg.name``):

* ``wsd``      warmup -> stable (1.0) -> linear decay to ``min_lr_ratio`` over the last
  ``decay_frac`` of progress (Warmup-Stable-Decay; the decay window is also where the trainer
  anneals in SFT data, see :func:`in_decay`).
* ``cosine``   warmup, then ``min + (1 - min) * 0.5 * (1 + cos(pi * progress))``.
* ``constant`` warmup, then 1.0.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

__all__ = [
    "lr_multiplier",
    "warmup_factor",
    "decay_factor",
    "in_decay",
    "decay_start",
    "muon_momentum",
    "curriculum_lengths",
    "curriculum_seq_len",
    "phase_budgets",
    "phase_progress",
    "BUDGET_KEYS",
]

#: Phase keys that define a budget (any subset may be present).
BUDGET_KEYS = ("tokens", "minutes", "steps", "epochs")


def _get(cfg: Any, key: str, default: Any) -> Any:
    """Read ``key`` from a dataclass / mapping; ``TrainConfig`` objects resolve to ``.schedule``."""
    if cfg is None:
        return default
    if not isinstance(cfg, Mapping) and hasattr(cfg, "schedule") and not hasattr(cfg, "decay_frac"):
        cfg = cfg.schedule
    if isinstance(cfg, Mapping):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return default if value is None else value


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def warmup_factor(step: int, warmup_steps: int) -> float:
    """Linear warmup by optimizer step: ``min(1, (step + 1) / warmup_steps)`` (1.0 if no warmup)."""
    warmup_steps = int(warmup_steps or 0)
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (max(0, int(step)) + 1) / warmup_steps)


def decay_start(cfg: Any) -> float:
    """Progress at which the WSD decay window begins (``1 - decay_frac``)."""
    return 1.0 - _clamp01(float(_get(cfg, "decay_frac", 0.35)))


def in_decay(progress: float, cfg: Any) -> bool:
    """True when ``progress`` lies in the decay window ``[1 - decay_frac, 1]`` (wsd/cosine).

    For ``constant`` schedules the window is still defined by ``decay_frac`` so that SFT annealing
    (``anneal_sft_mix``) behaves the same regardless of the LR shape. ``decay_frac == 0`` -> never.
    """
    frac = _clamp01(float(_get(cfg, "decay_frac", 0.35)))
    if frac <= 0.0:
        return False
    return _clamp01(progress) >= 1.0 - frac


def decay_factor(progress: float, cfg: Any) -> float:
    """Progress-dependent part of the multiplier (no warmup), in ``[min_lr_ratio, 1]``."""
    name = str(_get(cfg, "name", "wsd")).lower()
    min_ratio = _clamp01(float(_get(cfg, "min_lr_ratio", 0.0)))
    p = _clamp01(progress)
    if name == "constant":
        return 1.0
    if name == "cosine":
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))
    if name == "wsd":
        frac = _clamp01(float(_get(cfg, "decay_frac", 0.35)))
        if frac <= 0.0:
            return 1.0
        start = 1.0 - frac
        if p <= start:
            return 1.0
        t = (p - start) / frac  # 0 -> 1 across the decay window
        return 1.0 - (1.0 - min_ratio) * min(1.0, t)
    raise ValueError(f"unknown schedule {name!r}; expected one of wsd, cosine, constant")


def lr_multiplier(step: int, progress: float, cfg: Any) -> float:
    """LR multiplier for optimizer step ``step`` at phase ``progress`` (SPEC §7).

    Args:
        step: optimizer steps taken in the current phase (0-based).
        progress: consumed fraction of the phase budget (clamped to [0, 1]).
        cfg: ``ScheduleConfig`` (or a mapping / ``TrainConfig`` with ``.schedule``) providing
            ``name``, ``warmup_steps``, ``decay_frac``, ``min_lr_ratio``.

    Returns:
        ``warmup_factor(step) * decay_factor(progress)`` (a float in ``(0, 1]``).
    """
    return warmup_factor(step, int(_get(cfg, "warmup_steps", 0))) * decay_factor(progress, cfg)


def muon_momentum(step: int, warmup_steps: int = 300, start: float = 0.85, end: float = 0.95) -> float:
    """Muon momentum warmup: linear ``start -> end`` over ``warmup_steps`` optimizer steps."""
    if warmup_steps <= 0:
        return float(end)
    frac = min(1.0, max(0, int(step)) / float(warmup_steps))
    return float(start + (end - start) * frac)


def curriculum_lengths(start_seq_len: int, final_seq_len: int, multiple: int = 64) -> List[int]:
    """Discrete curriculum stages: ``start`` doubled until ``final`` (all multiples of ``multiple``).

    ``start`` is rounded up to a multiple of ``multiple`` (and clipped to ``final``); the last
    stage is always ``final``. When ``final < multiple`` (tiny debug runs) a single stage is used.
    The small, fixed set of shapes keeps ``torch.compile`` recompilation bounded (the trainer
    warms all of them up before the phase timer starts).

    >>> curriculum_lengths(128, 512)
    [128, 256, 512]
    >>> curriculum_lengths(100, 384)
    [128, 256, 384]
    """
    final = int(final_seq_len)
    if final <= 0:
        raise ValueError(f"final_seq_len must be > 0, got {final_seq_len}")
    if final < multiple:
        return [final]
    s = max(multiple, int(math.ceil(max(1, int(start_seq_len)) / multiple)) * multiple)
    stages: List[int] = []
    while s < final:
        stages.append(s)
        s *= 2
    stages.append(final)
    return stages


def curriculum_seq_len(progress: float, start_seq_len: int, final_seq_len: int, until_frac: float,
                       enabled: bool = True, multiple: int = 64) -> int:
    """Sequence length for the current ``progress`` of a pretraining phase.

    Stages from :func:`curriculum_lengths` (excluding the final one) are spread evenly over
    ``[0, until_frac)``; from ``until_frac`` on the full ``final_seq_len`` is used.
    """
    if not enabled or until_frac <= 0:
        return int(final_seq_len)
    stages = curriculum_lengths(start_seq_len, final_seq_len, multiple)
    if len(stages) == 1:
        return stages[0]
    p = _clamp01(progress)
    if p >= until_frac:
        return stages[-1]
    n_growing = len(stages) - 1
    idx = min(n_growing - 1, int(p / until_frac * n_growing))
    return stages[idx]


def phase_budgets(phase: Mapping[str, Any]) -> Dict[str, float]:
    """Positive budgets present in a phase dict: subset of ``tokens/minutes/steps/epochs``."""
    out: Dict[str, float] = {}
    for key in BUDGET_KEYS:
        value = phase.get(key) if isinstance(phase, Mapping) else getattr(phase, key, None)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"phase budget {key!r} must be a number, got {value!r}") from None
        if value > 0:
            out[key] = value
    return out


def phase_progress(budgets: Mapping[str, float], step: int = 0, tokens: float = 0.0,
                   seconds: float = 0.0, steps_per_epoch: Optional[float] = None) -> float:
    """Progress in ``[0, 1]`` = max ratio over the defined budgets (SPEC §7).

    Args:
        budgets: output of :func:`phase_budgets`.
        step: optimizer steps done; tokens: tokens consumed; seconds: phase wall time (excluding
            compile warm-up); steps_per_epoch: optimizer steps per data epoch (needed for the
            ``epochs`` budget; ignored when ``None``).
    Returns 0.0 when no budget is defined (callers should supply a default budget).
    """
    ratios = [0.0]
    if "steps" in budgets:
        ratios.append(step / budgets["steps"])
    if "tokens" in budgets:
        ratios.append(tokens / budgets["tokens"])
    if "minutes" in budgets:
        ratios.append(seconds / (60.0 * budgets["minutes"]))
    if "epochs" in budgets and steps_per_epoch:
        ratios.append(step / (budgets["epochs"] * float(steps_per_epoch)))
    return _clamp01(max(ratios))
