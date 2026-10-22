"""Training smoke tests (SPEC §7, §11): optimizer / schedule / checkpoint units and an end-to-end offline
run of ``ml-1m-debug`` with tiny budgets: loss decreases, checkpoint resume, export -> from_pretrained -> chat."""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")

from conftest import make_tiny_config  # noqa: E402

import moonlight as ml  # noqa: E402
from moonlight.llm import LLM  # noqa: E402

SYNTHETIC_SMALL = {"identity": True, "knowledge": True, "arithmetic": 60, "tool_calls": 40, "code": 20}


def debug_overrides(out_dir, phases, **extra):
    """Shrink ml-1m-debug to a ~1 minute CPU run with an isolated output directory."""
    o = {
        "train.output_dir": str(out_dir),
        "train.phases": phases,
        "train.seq_len": 64,
        "train.batch_size": 8,
        "train.batch_tokens": 512,           # accum = 1
        "train.compile": False,
        "train.precision": "fp32",
        "train.ema": 0.0,
        "train.curriculum.enabled": False,
        "train.eval_every": 0,
        "train.eval_batches": 2,
        "train.log_every": 1,
        "train.checkpoint_every_min": 0,
        "train.schedule.warmup_steps": 3,
        "train.distributed.strategy": "none",
        "train.seed": 1234,
        "tokenizer.path": None,
        "tokenizer.train_chars": 300_000,
        "data.offline": True,
        "data.synthetic": dict(SYNTHETIC_SMALL),
        "chat.max_new_tokens": 8,
        "chat.best_of": 1,
    }
    o.update(extra)
    return o


def read_metrics(out_dir):
    path = out_dir / "metrics.jsonl"
    assert path.is_file(), "the trainer writes metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ================================================================================================
# units: optimizer, schedule, checkpoint
# ================================================================================================

def test_newton_schulz_orthogonalizes_batched():
    from moonlight.train.optim import newton_schulz

    torch.manual_seed(0)
    G = torch.randn(3, 16, 32)
    X = newton_schulz(G, steps=5).float()
    assert X.shape == G.shape and torch.isfinite(X).all()
    sv = torch.linalg.svdvals(X)
    assert sv.max() < 1.5 and sv.min() > 0.3, "NS iteration maps singular values to ~1"


def test_build_optimizers_groups():
    from moonlight.train.optim import Muon, build_optimizers

    cfg = make_tiny_config({"tie_embeddings": False, "moe": {"enabled": True, "n_experts": 4, "top_k": 2,
                                                             "expert_hidden": 32}},
                           multimodal={"enabled": True, "vision_dim": 24, "n_image_tokens": 4})
    from moonlight.model.model import MoonlightForCausalLM

    model = MoonlightForCausalLM(cfg.model, cfg.multimodal)
    opts = build_optimizers(model, cfg.train)
    names = {id(p): n for n, p in model.named_parameters()}
    seen = {}
    for opt in opts:
        for g in opt.param_groups:
            assert "base_lr" in g
            for p in g["params"]:
                seen[names[id(p)]] = (isinstance(opt, Muon), g)
    assert set(seen) == set(names.values()), "every parameter is optimized exactly once"
    for n, (is_muon, g) in seen.items():
        p = dict(model.named_parameters())[n]
        if n.startswith(("embed.", "lm_head.")) or ".router." in n or n.startswith("vision_proj.norm"):
            assert not is_muon, f"{n} must use AdamW"
        elif p.ndim >= 2:
            assert is_muon, f"{n} ({p.ndim}D) must use Muon"
        else:
            assert not is_muon and g.get("weight_decay", 0.0) == 0.0, f"{n}: no weight decay on norms/scalars"
    embed_group = seen["embed.weight"][1]
    assert embed_group["base_lr"] == pytest.approx(cfg.train.optimizer.embed_lr)
    trainable = build_optimizers(model, cfg.train, trainable="vision_proj")
    tparams = [p for o in trainable for g in o.param_groups for p in g["params"]]
    assert tparams and all(names[id(p)].startswith("vision_proj") for p in tparams)


def test_lr_schedules():
    from moonlight.train.schedule import lr_multiplier

    wsd = {"name": "wsd", "warmup_steps": 10, "decay_frac": 0.4, "min_lr_ratio": 0.1}
    assert lr_multiplier(0, 0.0, wsd) < lr_multiplier(5, 0.01, wsd) < 1.0
    assert lr_multiplier(50, 0.3, wsd) == pytest.approx(1.0)
    assert lr_multiplier(50, 0.59, wsd) == pytest.approx(1.0)
    assert lr_multiplier(90, 0.8, wsd) < 1.0
    assert lr_multiplier(100, 1.0, wsd) == pytest.approx(0.1, abs=1e-6)
    cos = {"name": "cosine", "warmup_steps": 0, "decay_frac": 0.35, "min_lr_ratio": 0.0}
    assert lr_multiplier(10, 0.5, cos) == pytest.approx(0.5, abs=0.05)
    const = {"name": "constant", "warmup_steps": 0, "decay_frac": 0.35, "min_lr_ratio": 0.0}
    assert lr_multiplier(10, 0.9, const) == pytest.approx(1.0)


def test_checkpoint_roundtrip(tmp_path):
    from moonlight.model.model import MoonlightForCausalLM
    from moonlight.train.checkpoint import load_checkpoint, save_checkpoint
    from moonlight.train.optim import build_optimizers

    cfg = make_tiny_config()
    torch.manual_seed(0)
    model = MoonlightForCausalLM(cfg.model)
    opts = build_optimizers(model, cfg.train)
    x = torch.randint(8, cfg.model.vocab_size, (2, 16))
    model(x, targets=x).loss.backward()
    for o in opts:
        o.step()
    path = tmp_path / "ck.safetensors"
    save_checkpoint(path, model, opts, {"step": 7, "phase": "pretrain"}, config=cfg)
    ref = {k: v.clone() for k, v in model.state_dict().items()}
    opt_ref = [o.state_dict() for o in opts]

    torch.manual_seed(1)
    model2 = MoonlightForCausalLM(cfg.model)
    opts2 = build_optimizers(model2, cfg.train)
    state = load_checkpoint(path, model2, opts2, map_location="cpu")
    assert state["step"] == 7 and state["phase"] == "pretrain"
    for k, v in model2.state_dict().items():
        torch.testing.assert_close(v, ref[k])
    for o2, ref_sd in zip(opts2, opt_ref):
        sd = o2.state_dict()
        assert len(sd["state"]) == len(ref_sd["state"])
        for pid, st in ref_sd["state"].items():
            for key, val in st.items():
                if torch.is_tensor(val):
                    torch.testing.assert_close(sd["state"][pid][key], val)


# ================================================================================================
# end-to-end: ml-1m-debug offline
# ================================================================================================

@pytest.fixture(scope="module")
def trained_debug(tmp_path_factory):
    out = tmp_path_factory.mktemp("ml-1m-debug")
    phases = [{"name": "pretrain", "steps": 30, "lr_scale": 1.0},
              {"name": "sft", "steps": 10, "lr_scale": 0.5}]
    cfg = ml.config("ml-1m-debug", overrides=debug_overrides(out, phases))
    llm = LLM(cfg, device="cpu")
    result = llm.train()
    return llm, result, out, phases


def test_debug_train_loss_decreases(trained_debug):
    llm, result, out, _ = trained_debug
    assert result["steps"] == 40
    assert [p["name"] for p in result["phases"]] == ["pretrain", "sft"]
    assert all(not p.get("skipped") for p in result["phases"])
    recs = [r for r in read_metrics(out) if r.get("phase") == "pretrain" and "loss" in r and "event" not in r]
    losses = [r["loss"] for r in sorted(recs, key=lambda r: r["step"])]
    assert len(losses) >= 25 and all(math.isfinite(v) for v in losses)
    first, last = sum(losses[:3]) / 3, sum(losses[-3:]) / 3
    assert last < first - 0.3, f"loss did not decrease: {first:.3f} -> {last:.3f}"
    assert math.isfinite(result["final_val_loss"])
    for name in ("checkpoint.safetensors", "model.safetensors", "tokenizer.json", "metrics.jsonl"):
        assert (out / name).is_file(), name
    assert llm.tokenizer is not None and llm.tokenizer.vocab_size <= llm.config.model.vocab_size


def test_debug_resume_continues_after_finished_phases(trained_debug):
    _, first, out, phases = trained_debug
    n_pretrain = sum(1 for r in read_metrics(out) if r.get("phase") == "pretrain" and "loss" in r
                     and "event" not in r)
    more = phases + [{"name": "anneal", "steps": 4, "lr_scale": 0.3}]
    cfg = ml.config("ml-1m-debug", overrides=debug_overrides(out, more))
    llm = LLM(cfg, device="cpu")
    result = llm.train()
    assert result["steps"] == first["steps"] + 4, "finished phases are skipped on resume"
    assert [p["name"] for p in result["phases"]] == ["pretrain", "sft", "anneal"]
    n_after = sum(1 for r in read_metrics(out) if r.get("phase") == "pretrain" and "loss" in r
                  and "event" not in r)
    assert n_after == n_pretrain, "pretrain must not run again after resume"


def test_export_from_pretrained_chat(trained_debug, tmp_path):
    llm, _, _, _ = trained_debug
    path = llm.save(tmp_path / "ml-1m-debug.safetensors")
    assert path.is_file()
    llm2 = LLM.from_pretrained(path, device="cpu")
    assert llm2.config.name == "ml-1m-debug"
    assert llm2.tokenizer.to_str() == llm.tokenizer.to_str()
    sd1, sd2 = llm.model.state_dict(), llm2.model.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        torch.testing.assert_close(sd2[k].float().cpu(), sd1[k].float().cpu())
    ids = torch.tensor([llm.tokenizer.encode("Salom, dunyo!")])
    with torch.no_grad():
        torch.testing.assert_close(llm2.model(ids).logits, llm.model(ids).logits, atol=1e-5, rtol=1e-5)
    reply = llm2.chat("Salom!", max_new_tokens=8)
    assert isinstance(reply, str)
    assert isinstance(llm2.chat("2+2?", tools=True, max_new_tokens=8), str)
    # the trainer's own export loads as well
    llm3 = LLM.from_pretrained(llm.train_result["model_path"], device="cpu")
    assert llm3.num_params() == llm.num_params()


def test_unknown_phase_rejected(tmp_path):
    cfg = ml.config("ml-1m-debug", overrides=debug_overrides(tmp_path, [{"name": "warp-drive", "steps": 1}]))
    with pytest.raises(ValueError):
        LLM(cfg, device="cpu").train()


def test_one_liner_train(tmp_path):
    phases = [{"name": "pretrain", "steps": 3}]
    llm = ml.train("ml-1m-debug", overrides=debug_overrides(tmp_path / "run", phases), output=tmp_path / "exp")
    assert isinstance(llm, LLM) and llm.train_result["steps"] == 3
    assert (tmp_path / "exp" / "model.safetensors").is_file()
