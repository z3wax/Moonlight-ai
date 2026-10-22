"""Alignment (SPEC §13.5): DPO loss math, sequence log-probs, and short offline dpo + grpo phases."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from conftest import make_tiny_config  # noqa: E402

from moonlight.llm import LLM  # noqa: E402
from moonlight.model.model import MoonlightForCausalLM  # noqa: E402
from moonlight.train import align  # noqa: E402


def _loss(out):
    return out[0] if isinstance(out, tuple) else out


def test_dpo_loss_math():
    pc = torch.tensor([-10.0, -12.0])
    pr = torch.tensor([-11.0, -11.0])
    rc = torch.tensor([-10.5, -12.0])
    rr = torch.tensor([-10.5, -12.0])
    beta = 0.1
    loss = _loss(align.dpo_loss(pc, pr, rc, rr, beta=beta))
    logits = beta * ((pc - rc) - (pr - rr))
    expected = -torch.nn.functional.logsigmoid(logits).mean()
    torch.testing.assert_close(loss, expected)
    # identical policy and reference -> log(2)
    same = _loss(align.dpo_loss(pc, pr, pc, pr, beta=beta))
    assert same.item() == pytest.approx(math.log(2), rel=1e-6)
    # preferring the chosen answer more lowers the loss
    better = _loss(align.dpo_loss(pc + 5, pr, rc, rr, beta=beta))
    assert better < loss


def test_sequence_logprobs_sums_response_tokens():
    torch.manual_seed(0)
    cfg = make_tiny_config()
    model = MoonlightForCausalLM(cfg.model).eval()
    x = torch.randint(8, cfg.model.vocab_size, (2, 12))
    y = torch.roll(x, -1, 1)
    y[:, :6] = -100
    y[:, -1] = -100
    with torch.no_grad():
        got = align.sequence_logprobs(model, x, y)
        logp = torch.log_softmax(model(x).logits.float(), -1)
    mask = y != -100
    ref = (logp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask).sum(-1)
    assert got.shape == (2,)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-5)
    assert torch.all(got < 0)


def test_dpo_and_grpo_phases_offline(tmp_path):
    phases = [
        {"name": "pretrain", "steps": 6},
        {"name": "sft", "steps": 4},
        {"name": "dpo", "steps": 3, "synthetic_pairs": 16, "beta": 0.1, "lr_scale": 0.1},
        {"name": "grpo", "steps": 2, "tasks": ["arithmetic", "constraints"], "group_size": 2,
         "prompts_per_step": 2, "max_new_tokens": 8, "temperature": 1.0, "kl_coef": 0.02, "clip": 0.2,
         "lr_scale": 0.05},
    ]
    cfg = make_tiny_config(train={"phases": phases, "output_dir": str(tmp_path / "run"), "log_every": 1,
                                  "seq_len": 128, "batch_size": 4, "batch_tokens": 512,
                                  "eval_every": 0, "eval_batches": 1, "checkpoint_every_min": 0,
                                  "schedule": {"warmup_steps": 1}},
                           tokenizer={"train_chars": 200_000})
    llm = LLM(cfg, device="cpu")
    result = llm.train()
    by_name = {p["name"]: p for p in result["phases"]}
    assert set(by_name) == {"pretrain", "sft", "dpo", "grpo"}
    for name in ("dpo", "grpo"):
        ph = by_name[name]
        assert not ph.get("skipped"), f"{name} phase was skipped offline: {ph}"
        assert ph["steps"] >= 1
        assert math.isfinite(float(ph["final_train_loss"]))
    for p in llm.model.parameters():
        assert torch.isfinite(p).all()
    assert isinstance(llm.chat("Salom", max_new_tokens=4), str)
