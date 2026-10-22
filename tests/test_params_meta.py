"""SPEC §3: the model's real parameter layout must equal ``moonlight.params`` EXACTLY.

Every registry config (1M ... 1T, dense + MoE + multimodal) is instantiated on the ``meta`` device,
so no real memory is allocated; the vision ENCODER (transformers) is never built, only the
``vision_proj`` projector that belongs to the language model.
"""

from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

from conftest import ALL_CONFIG_NAMES, make_tiny_config  # noqa: E402

from moonlight import params  # noqa: E402
from moonlight.model.model import MoonlightForCausalLM  # noqa: E402
from moonlight.params import estimate_params  # noqa: E402
from moonlight.registry import load  # noqa: E402


def _meta_model(cfg):
    with torch.device("meta"):
        model = MoonlightForCausalLM(cfg.model, cfg.multimodal)
    return model


def _check_exact(cfg):
    model = _meta_model(cfg)
    try:
        tensors = list(model.parameters()) + list(model.buffers())
        assert all(t.is_meta for t in tensors), f"{cfg.name}: some tensors were materialised (not meta)"
        total = sum(p.numel() for p in model.parameters())
        report = estimate_params(cfg)
        if total != report.total and hasattr(params, "param_shapes"):  # precise diagnostics
            real = {n: tuple(p.shape) for n, p in model.named_parameters()}
            est = {n: tuple(s) for n, s in params.param_shapes(cfg).items()}
            diff = sorted((set(real.items()) ^ set(est.items())))[:12]
            pytest.fail(f"{cfg.name}: model {total:,} != estimate {report.total:,}; differing entries: {diff}")
        assert total == report.total, f"{cfg.name}: model {total:,} != estimate {report.total:,}"
        names = {n for n, _ in model.named_parameters()}
        assert "embed.weight" in names and "norm.weight" in names
        assert ("lm_head.weight" in names) == (not cfg.model.tie_embeddings)
        assert any(n.startswith("vision_proj.") for n in names) == bool(cfg.multimodal.enabled)
        if cfg.model.moe.enabled:
            assert not any(n.endswith("router_bias") for n in names), "router_bias must be a buffer"
            assert any(n.endswith("router_bias") for n, _ in model.named_buffers())
    finally:
        del model


@pytest.mark.parametrize("name", ALL_CONFIG_NAMES)
def test_meta_param_count_matches_estimator(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = load(name, validate=False)
    _check_exact(cfg)


@pytest.mark.parametrize("model_kw, sections", [
    ({}, {}),
    ({"tie_embeddings": False, "mtp_depth": 2, "mtp_block": True, "value_residual": False, "token_shift": 0}, {}),
    ({"retention_ratio": 0.0, "output_gate": False, "qk_norm": False}, {}),
    ({"retention_ratio": 1.0}, {}),
    ({"moe": {"enabled": True, "n_experts": 4, "top_k": 2, "n_shared": 1, "expert_hidden": 32,
              "first_dense_layers": 1}, "mtp_block": True}, {}),
    ({}, {"multimodal": {"enabled": True, "vision_dim": 48, "n_image_tokens": 4}}),
])
def test_meta_tiny_variants_exact(model_kw, sections):
    cfg = make_tiny_config(model_kw, **sections)
    _check_exact(cfg)


def test_meta_real_construction_agrees():
    """A materialised tiny model has the same count as its meta twin and the estimator."""
    cfg = make_tiny_config({"moe": {"enabled": True, "n_experts": 4, "top_k": 2, "n_shared": 1, "expert_hidden": 32}})
    real = MoonlightForCausalLM(cfg.model, cfg.multimodal)
    assert sum(p.numel() for p in real.parameters()) == estimate_params(cfg).total
