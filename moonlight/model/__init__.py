"""Moonlight model package (requires torch).

Public API::

    from moonlight.model import MoonlightForCausalLM, ModelOutput, InferenceCache, generate
    model = MoonlightForCausalLM(cfg.model, cfg.multimodal)      # or .from_config(cfg)
    out = model(input_ids, targets=targets)                      # ModelOutput
    res = generate(model, prompt_ids, max_new_tokens=64, n=3)    # GenerationResult

``transformers`` is only imported lazily by :class:`VisionEncoder`.
"""
from __future__ import annotations

from .attention import gqa_attention, repeat_kv, sliding_window_mask
from .cache import InferenceCache, LayerCache
from .generation import GenerationResult, generate, sample_next
from .layers import RMSNorm, RoPE, RotaryEmbedding, SwiGLU, TokenShift, apply_rope, rms, rotate_half
from .model import (
    IMAGE_TOKEN_ID,
    Block,
    HybridMixer,
    ModelOutput,
    MoonlightForCausalLM,
    MTPHead,
    head_counts,
    layer_window,
)
from .moe import MoE
from .retention import (
    decay_init_logits,
    retention,
    retention_chunkwise,
    retention_final_state,
    retention_parallel,
    retention_recurrent_step,
)
from .vision import VisionEncoder, VisionProjector, load_image

__all__ = [
    "MoonlightForCausalLM",
    "ModelOutput",
    "Block",
    "HybridMixer",
    "MTPHead",
    "MoE",
    "InferenceCache",
    "LayerCache",
    "GenerationResult",
    "generate",
    "sample_next",
    "RMSNorm",
    "rms",
    "RotaryEmbedding",
    "RoPE",
    "apply_rope",
    "rotate_half",
    "SwiGLU",
    "TokenShift",
    "gqa_attention",
    "repeat_kv",
    "sliding_window_mask",
    "retention",
    "retention_parallel",
    "retention_chunkwise",
    "retention_recurrent_step",
    "retention_final_state",
    "decay_init_logits",
    "VisionProjector",
    "VisionEncoder",
    "load_image",
    "IMAGE_TOKEN_ID",
    "head_counts",
    "layer_window",
]
