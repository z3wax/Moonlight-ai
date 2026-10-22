"""Moonlight training package (SPEC §7, §13.5).

Submodules: :mod:`.optim` (Muon + AdamW), :mod:`.schedule` (LR / curriculum / budgets, pure python),
:mod:`.distributed` (torchrun / DDP / FSDP), :mod:`.checkpoint` (safetensors checkpoints + export),
:mod:`.align` (DPO / GRPO helpers), :mod:`.trainer` (``Trainer``).

Attributes are resolved lazily (PEP 562) so ``import moonlight.train`` and the torch-free parts
(``moonlight.train.schedule``, ``DistInfo``, checkpoint metadata readers) work without torch.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

_EXPORTS: Dict[str, str] = {
    # trainer
    "Trainer": "trainer",
    "ModelEMA": "trainer",
    "phase_type": "trainer",
    "PHASE_TYPES": "trainer",
    "resolve_precision": "trainer",
    "compile_supported": "trainer",
    # optim
    "Muon": "optim",
    "newton_schulz": "optim",
    "build_optimizers": "optim",
    "set_trainable": "optim",
    # schedule
    "lr_multiplier": "schedule",
    "muon_momentum": "schedule",
    "curriculum_seq_len": "schedule",
    "curriculum_lengths": "schedule",
    "phase_progress": "schedule",
    "phase_budgets": "schedule",
    # distributed
    "DistInfo": "distributed",
    "init_distributed": "distributed",
    "wrap_model": "distributed",
    "unwrap_model": "distributed",
    "all_reduce_mean": "distributed",
    "barrier": "distributed",
    # checkpoint
    "save_checkpoint": "checkpoint",
    "load_checkpoint": "checkpoint",
    "export_model": "checkpoint",
    "load_exported": "checkpoint",
    "read_metadata": "checkpoint",
    "read_checkpoint_state": "checkpoint",
    # align
    "dpo_loss": "align",
    "sequence_logprobs": "align",
    "grpo_step": "align",
}

__all__: List[str] = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'moonlight.train' has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))
