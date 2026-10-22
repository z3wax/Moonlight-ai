"""Config schema, registry and the 100 generated configs (SPEC §2, §10, §13.8). Torch-free."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from conftest import ALL_CONFIG_NAMES, CONFIG_TIERS, REPO_ROOT, expected_suite, tiny_config_dict

from moonlight.config import Config, format_size, load_yaml, parse_size
from moonlight.params import estimate_params
from moonlight.registry import find_config, list_configs, load

CHAT_CAPABLE_KINDS = ("chat", "code", "vl-chat", "tools", "agent", "think", "long", "debug")

# SPEC §10 base shapes: size -> (V, d, H, hd)
DENSE_SHAPES = {
    "1m": (2048, 96, 4, 24), "2m": (2048, 128, 4, 32), "5m": (4096, 192, 6, 32), "10m": (8192, 256, 8, 32),
    "20m": (8192, 384, 6, 64), "50m": (16384, 512, 8, 64), "100m": (32768, 768, 12, 64),
    "150m": (32768, 768, 12, 64), "250m": (32768, 1024, 16, 64), "300m": (32768, 1024, 16, 64),
    "1b": (65536, 2048, 32, 64), "2b": (65536, 2304, 36, 64), "3b": (65536, 3072, 24, 128),
    "7b": (131072, 4096, 32, 128), "13b": (131072, 5120, 40, 128), "30b": (131072, 6656, 52, 128),
    "70b": (131072, 8192, 64, 128), "175b": (131072, 12288, 96, 128), "405b": (131072, 16384, 128, 128),
    "1t": (131072, 24576, 192, 128),
}
# SPEC §10 MoE shapes: tag -> (d, H, hd, E, top_k)
MOE_SHAPES = {
    "16b-a2b": (2048, 16, 128, 64, 6), "30b-a3b": (2048, 32, 128, 128, 8), "100b-a10b": (4096, 32, 128, 64, 6),
    "235b-a22b": (4096, 64, 128, 128, 8), "671b-a37b": (7168, 128, 128, 256, 8), "1t-a32b": (7168, 64, 128, 384, 8),
}
_NAME_RE = re.compile(r"^ml-(?P<size>\d+[mbt])(?:-a(?P<active>\d+[mbt]))?-(?P<variant>.+)$")


def _parse_name(name):
    m = _NAME_RE.match(name)
    assert m, f"config name {name!r} does not follow ml-<size>[-a<active>]-<variant>"
    return m.group("size"), m.group("active"), m.group("variant")


def _phase_types(cfg):
    return [str(p.get("type") or p.get("name")) for p in cfg.train.phases]


def _has(types, prefix):
    return any(t == prefix or t.startswith(prefix + "-") or re.fullmatch(prefix + r"\d+", t) for t in types)


def _index(types, prefix):
    for i, t in enumerate(types):
        if t == prefix or t.startswith(prefix + "-"):
            return i
    return -1


# ================================================================================================
# registry: exactly the 100 SPEC configs
# ================================================================================================

def test_exactly_100_configs_with_spec_names():
    assert len(ALL_CONFIG_NAMES) == 100 and len(set(ALL_CONFIG_NAMES)) == 100
    names = list_configs()
    assert names == sorted(names), "list_configs() must be sorted"
    assert sorted(names) == sorted(ALL_CONFIG_NAMES)
    files = sorted((REPO_ROOT / "configs").glob("*/*.yaml"))
    assert len(files) == 100, f"expected exactly 100 YAML files under configs/, found {len(files)}"


@pytest.mark.parametrize("tier", sorted(CONFIG_TIERS))
def test_list_configs_by_tier(tier):
    assert sorted(list_configs(tier=tier)) == sorted(CONFIG_TIERS[tier])


def test_files_are_named_after_config_and_tier(registry_configs):
    for tier, names in CONFIG_TIERS.items():
        for name in names:
            path = REPO_ROOT / "configs" / tier / f"{name}.yaml"
            assert path.is_file(), f"missing {path}"
            assert Path(find_config(name)).resolve() == path.resolve()
            cfg = registry_configs[name]
            assert cfg.name == path.stem == name
            assert cfg.tier == tier, f"{name}: tier {cfg.tier!r} but stored under configs/{tier}/"


def test_generated_configs_are_explicit(registry_configs):
    for name in ALL_CONFIG_NAMES:
        raw = load_yaml(find_config(name))
        assert "extends" not in raw, f"{name}: generated configs must be fully explicit (no extends)"
        for key in ("name", "tier", "kind", "benchmark", "nominal_params", "model", "tokenizer", "data", "train",
                    "chat"):
            assert key in raw, f"{name}: missing top-level key {key!r}"


def test_all_configs_validate():
    for name in ALL_CONFIG_NAMES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # vl warnings are allowed; sizes checked below
            cfg = load(name)  # validate=True
        assert cfg.tokenizer.vocab_size == cfg.model.vocab_size


# ================================================================================================
# sizes, shapes, benchmark mapping
# ================================================================================================

def test_estimator_within_nominal_tolerance(registry_configs):
    """Dense and MoE totals within +-10% of the nominal label; MoE active within +-20% of -aNb."""
    problems = []
    for name, cfg in registry_configs.items():
        size, active, _ = _parse_name(name)
        nominal = parse_size(cfg.nominal_params)
        if nominal != parse_size(size.upper()):
            problems.append(f"{name}: nominal_params {cfg.nominal_params!r} != name size {size}")
        rep = estimate_params(cfg)
        ratio = rep.total / nominal
        if abs(ratio - 1) > 0.10:
            problems.append(f"{name}: total {format_size(rep.total)} is {ratio:.3f}x nominal {cfg.nominal_params}")
        if active:
            a_ratio = rep.active / parse_size(active.upper())
            if abs(a_ratio - 1) > 0.20:
                problems.append(f"{name}: active {format_size(rep.active)} is {a_ratio:.3f}x the label a{active}")
            assert cfg.model.moe.enabled, f"{name}: -a<active> configs must enable MoE"
        else:
            assert rep.active == rep.total or cfg.model.moe.enabled
    assert not problems, "\n".join(problems)


def test_base_shapes_follow_spec(registry_configs):
    for name, cfg in registry_configs.items():
        size, active, _ = _parse_name(name)
        m = cfg.model
        if active:
            d, H, hd, E, k = MOE_SHAPES[f"{size}-a{active}"]
            assert (m.d_model, m.n_heads, m.head_dim, m.moe.n_experts, m.moe.top_k) == (d, H, hd, E, k), name
            assert m.retention_ratio == pytest.approx(0.25), name
            assert (m.attn_window, m.global_every) == (4096, 4), name
        else:
            V, d, H, hd = DENSE_SHAPES[size]
            assert (m.vocab_size, m.d_model, m.n_heads, m.head_dim) == (V, d, H, hd), name
            assert not m.moe.enabled, name
        n_ret = int(round(m.n_heads * m.retention_ratio))
        n_attn = m.n_heads - n_ret
        if n_attn > 0:
            assert m.n_kv_heads >= 1 and n_attn % m.n_kv_heads == 0, name


def test_benchmark_mapping_by_nominal_size(registry_configs):
    for name, cfg in registry_configs.items():
        assert cfg.benchmark == expected_suite(parse_size(cfg.nominal_params)), name


# ================================================================================================
# variants (SPEC §10 + §13.8)
# ================================================================================================

def test_variant_rules(registry_configs):
    for name, cfg in registry_configs.items():
        _, _, variant = _parse_name(name)
        types = _phase_types(cfg)
        assert types, f"{name}: no training phases"
        if cfg.kind in ("vl", "vl-chat") or variant in ("vl", "vl-chat"):
            assert cfg.multimodal.enabled, f"{name}: vl configs need multimodal.enabled"
        if variant == "base":
            assert cfg.kind == "base", name
            assert set(types) <= {"pretrain", "anneal"}, f"{name}: base configs only pretrain (+anneal): {types}"
        if cfg.kind in CHAT_CAPABLE_KINDS:
            assert _has(types, "sft"), f"{name}: chat-capable configs need an sft phase: {types}"
            assert cfg.data.sft or cfg.data.offline, f"{name}: chat-capable configs need SFT sources"
            syn = dict(cfg.data.synthetic or {})
            code_sft = any("code" in str(s.get("source", "")) or "python" in str(s.get("source", ""))
                           for s in cfg.data.sft)
            assert code_sft or int(syn.get("code", 0) or 0) > 0, f"{name}: chat models must train on code too"
        if variant == "debug":
            assert cfg.kind == "debug" and cfg.data.offline, name


def test_alignment_phases_v11(registry_configs):
    """SPEC §13.8: SI chat-capable -> sft, dpo, grpo; mini 150M-300M chat -> dpo; 50M chat/tools/think -> dpo."""
    for name, cfg in registry_configs.items():
        size, _, variant = _parse_name(name)
        types = _phase_types(cfg)
        if cfg.tier == "si" and cfg.kind in ("chat", "code", "tools", "agent", "think", "vl-chat"):
            i_sft, i_dpo, i_grpo = _index(types, "sft"), _index(types, "dpo"), _index(types, "grpo")
            assert -1 < i_sft < i_dpo < i_grpo, f"{name}: expected ... sft -> dpo -> grpo, got {types}"
            grpo = cfg.train.phases[i_grpo]
            assert grpo.get("tasks"), f"{name}: grpo phase needs tasks"
        if re.fullmatch(r"ml-(150m|250m|300m)-chat", name):
            assert -1 < _index(types, "sft") < _index(types, "dpo"), f"{name}: mini chat needs sft -> dpo: {types}"
        if name in ("ml-50m-chat", "ml-50m-tools", "ml-50m-think"):
            assert _index(types, "dpo") > _index(types, "sft") > -1, f"{name}: needs a short dpo phase: {types}"
        if variant == "long":
            assert -1 < _index(types, "long") < _index(types, "sft"), f"{name}: long phase must precede sft"


def test_long_variants_context(registry_configs):
    assert registry_configs["ml-300m-long"].model.max_seq_len == 8192
    assert registry_configs["ml-7b-long"].model.max_seq_len == 131072


def test_distributed_presets_by_tier(registry_configs):
    expected = {"test": "none", "mini": "ddp", "si": "fsdp"}
    for name, cfg in registry_configs.items():
        assert cfg.train.distributed.strategy == expected[cfg.tier], name


# ================================================================================================
# schema behaviour
# ================================================================================================

def test_parse_and_format_size():
    assert parse_size("20M") == 20_000_000
    assert parse_size("1.5B") == 1_500_000_000
    assert parse_size("1T") == 10**12
    assert parse_size(123) == 123
    assert format_size(20_100_000) == "20.1M"
    for bad in ("abc", "-5M", ""):
        with pytest.raises(ValueError):
            parse_size(bad)


def test_defaults_and_from_dict_roundtrip():
    cfg = Config.from_dict({"name": "x"})
    assert cfg.model.vocab_size == cfg.tokenizer.vocab_size
    again = Config.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()


def test_unknown_key_is_rejected_with_path():
    with pytest.raises(ValueError, match=r"model\.moe\.n_expert"):
        Config.from_dict({"name": "x", "model": {"moe": {"n_expert": 4}}})


def test_yaml_roundtrip(tmp_path, registry_configs):
    for name in ("ml-1m-debug", "ml-20m-chat", "ml-30b-vl-chat", "ml-671b-a37b-chat"):
        cfg = registry_configs[name]
        text = cfg.to_yaml()
        path = tmp_path / f"{name}.yaml"
        cfg.to_yaml(path)
        assert path.read_text(encoding="utf-8") == text
        back = Config.from_yaml(path)
        assert back.to_dict() == cfg.to_dict()


def test_overrides_and_copy():
    cfg = load("ml-20m-chat", overrides={"train.batch_size": 16})
    assert cfg.train.batch_size == 16
    c2 = cfg.copy()
    c2.apply_overrides({"model.attn_window": "null", "train.optimizer.lr": "1e-3", "data.offline": "true"})
    assert c2.model.attn_window is None and c2.train.optimizer.lr == pytest.approx(1e-3) and c2.data.offline
    assert cfg.train.optimizer.lr != pytest.approx(1e-3), "copy() must be deep"
    c2.apply_overrides(["train.phases.0.minutes=3"])
    assert float(c2.train.phases[0]["minutes"]) == 3
    with pytest.raises(ValueError, match="train.batch_sise"):
        c2.apply_overrides({"train.batch_sise": 1})


def test_public_config_entry_point():
    import moonlight as ml

    cfg = ml.config("ml-1m-chat", overrides={"train.seed": 7})
    assert isinstance(cfg, Config) and cfg.train.seed == 7
    assert ml.list_configs(tier="test") == sorted(CONFIG_TIERS["test"])


def test_extends(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text("extends: ml-1m-chat\nname: child\ntrain:\n  batch_size: 3\n", encoding="utf-8")
    cfg = Config.from_yaml(child)
    parent = load("ml-1m-chat")
    assert cfg.name == "child" and cfg.train.batch_size == 3
    assert cfg.model.to_dict() == parent.model.to_dict()


@pytest.mark.parametrize("change, pattern", [
    ({"model": {"n_kv_heads": 3}}, "n_kv_heads"),
    ({"tokenizer": {"vocab_size": 1024}}, "vocab_size"),
    ({"model": {"moe": {"enabled": True, "n_experts": 2, "top_k": 4, "expert_hidden": 32}}}, "top_k|n_experts"),
    ({"model": {"moe": {"enabled": True, "n_experts": 4, "top_k": 2, "expert_hidden": 0}}}, "expert_hidden"),
    ({"kind": "vl"}, "multimodal"),
])
def test_validation_errors(change, pattern):
    d = tiny_config_dict()
    for k, v in change.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    d[k][kk].update(vv)
                else:
                    d[k][kk] = vv
        else:
            d[k] = v
    with pytest.raises(ValueError, match=pattern):
        Config.from_dict(d).validate()


def test_benchmark_accessor():
    cfg = load("ml-1m-chat")
    suite = cfg.bnchmrks.type("cc-001")
    assert suite.name == "cc-001"
    assert cfg.benchmarks.type("cc-002").name == "cc-002"
    assert cfg.bnchmrks.default().name == cfg.benchmark
