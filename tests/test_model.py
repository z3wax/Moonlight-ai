"""Model math (SPEC §3-§4): forward/backward, MTP, MoE, multimodal, retention forms, KV/retention cache
equivalence and generation. Tiny fp32 models on CPU only."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from conftest import make_tiny_config  # noqa: E402

from moonlight.model.cache import InferenceCache  # noqa: E402
from moonlight.model.generation import GenerationResult, generate  # noqa: E402
from moonlight.model.model import MoonlightForCausalLM  # noqa: E402
from moonlight.model.retention import (  # noqa: E402
    decay_init_logits,
    retention_chunkwise,
    retention_parallel,
    retention_recurrent_step,
)
from moonlight.params import estimate_params  # noqa: E402

IMAGE_ID = 7
MOE = {"enabled": True, "n_experts": 4, "top_k": 2, "n_shared": 1, "expert_hidden": 32, "first_dense_layers": 1,
       "aux_loss_weight": 0.01, "bias_update_rate": 0.01}
ATOL = 1e-4


def build(model_kw=None, seed=0, **sections):
    torch.manual_seed(seed)
    cfg = make_tiny_config(model_kw or {}, **sections)
    model = MoonlightForCausalLM(cfg.model, cfg.multimodal if cfg.multimodal.enabled else None)
    return cfg, model


def batch(cfg, B=2, T=24, seed=1, avoid_image=True):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(8 if avoid_image else 0, cfg.model.vocab_size, (B, T), generator=g)
    targets = torch.roll(ids, -1, dims=1)
    targets[:, -1] = -100
    return ids, targets


def assert_finite_grads(model, names_prefix=()):
    seen = 0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"
        seen += 1
    assert seen > 0
    for pref in names_prefix:
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(pref)]
        assert grads and all(g is not None and g.abs().sum() > 0 for g in grads), f"no gradient reached {pref}"


# ================================================================================================
# forward / backward
# ================================================================================================

@pytest.mark.parametrize("model_kw", [
    {},
    {"tie_embeddings": False, "softcap": 0.0, "z_loss": 1e-4},
    {"retention_ratio": 0.0},
    {"retention_ratio": 1.0},
    {"attn_window": 8, "global_every": 2},
    {"activation_checkpointing": True},
], ids=["hybrid", "untied", "attn-only", "retention-only", "sliding", "ckpt"])
def test_dense_forward_backward(model_kw):
    cfg, model = build(model_kw)
    model.train()
    ids, targets = batch(cfg)
    out = model(ids, targets=targets)
    V = cfg.model.vocab_size
    assert out.logits.shape == (2, 24, V) and out.logits.dtype == torch.float32
    assert out.loss.ndim == 0 and torch.isfinite(out.loss) and torch.isfinite(out.lm_loss)
    # random init -> loss close to ln(V)
    assert abs(out.lm_loss.item() - math.log(V)) < 1.5
    if cfg.model.softcap > 0:
        assert out.logits.abs().max() <= cfg.model.softcap + 1e-4
    out.loss.backward()
    assert_finite_grads(model, ["embed."])


def test_logits_without_targets_and_hidden():
    cfg, model = build()
    model.eval()
    ids, _ = batch(cfg, B=1, T=10)
    with torch.no_grad():
        out = model(ids, return_hidden=True)
        one_d = model(ids[0])
    assert out.loss is None and out.logits.shape == (1, 10, cfg.model.vocab_size)
    assert out.hidden.shape == (1, 10, cfg.model.d_model)
    torch.testing.assert_close(one_d.logits, out.logits)


def test_ignore_index_masks_loss():
    cfg, model = build()
    ids, targets = batch(cfg)
    all_ignored = torch.full_like(targets, -100)
    all_ignored[:, :3] = targets[:, :3]
    model.eval()
    with torch.no_grad():
        out = model(ids, targets=all_ignored)
        logits = model(ids).logits
    ref = torch.nn.functional.cross_entropy(logits[:, :3].reshape(-1, logits.shape[-1]), targets[:, :3].reshape(-1))
    torch.testing.assert_close(out.lm_loss, ref, atol=1e-5, rtol=1e-5)


def test_mtp_heads_trained():
    cfg, model = build({"mtp_depth": 2, "mtp_block": True, "mtp_weight": 0.5})
    assert len(model.mtp) == 2
    model.train()
    ids, targets = batch(cfg)
    out = model(ids, targets=targets)
    assert torch.isfinite(out.loss)
    assert not torch.allclose(out.loss, out.lm_loss), "MTP loss must contribute to the total loss"
    out.loss.backward()
    assert_finite_grads(model, ["mtp.0.proj", "mtp.1.proj", "mtp.1.block."])


def test_moe_forward_backward_and_router_bias():
    cfg, model = build({"moe": MOE, "n_layers": 3})
    names = dict(model.named_parameters())
    assert not any(n.endswith("router_bias") for n in names)
    biases = {n: b for n, b in model.named_buffers() if n.endswith("router_bias")}
    assert len(biases) == 2 and all(b.dtype == torch.float32 for b in biases.values())
    assert "router_bias" in "".join(model.state_dict().keys()), "router_bias must be persistent"
    model.train()
    ids, targets = batch(cfg, B=4, T=32)
    out = model(ids, targets=targets)
    assert torch.isfinite(out.loss) and out.aux_loss is not None and out.aux_loss.item() > 0
    out.loss.backward()
    assert_finite_grads(model, ["layers.1.ffn.w_gate_up", "layers.1.ffn.router", "layers.1.ffn.shared"])
    before = {n: b.clone() for n, b in biases.items()}
    model.update_router_biases()
    changed = [n for n, b in model.named_buffers() if n in before and not torch.equal(b, before[n])]
    assert changed, "update_router_biases() must move the aux-free balancing bias"
    rate = cfg.model.moe.bias_update_rate
    for n, b in model.named_buffers():
        if n in before:
            assert torch.all((b - before[n]).abs() <= rate + 1e-7)


def test_multimodal_forward_backward():
    mm = {"enabled": True, "vision_dim": 24, "n_image_tokens": 4}
    cfg, model = build(multimodal=mm)
    assert model.vision_proj is not None
    model.train()
    ids, targets = batch(cfg, B=2, T=16)
    ids[0, 2:6] = IMAGE_ID           # one image in row 0
    ids[1, 5:9] = IMAGE_ID           # one image in row 1
    feats = torch.randn(2, 4, 24)
    out = model(ids, targets=targets, image_features=feats)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert_finite_grads(model, ["vision_proj.fc1", "vision_proj.fc2"])
    model.eval()
    with torch.no_grad():
        a = model(ids, image_features=feats).logits
        b = model(ids, image_features=torch.randn(2, 4, 24)).logits
        c = model(ids).logits  # no features: <|image|> embeddings stay as they are
    assert not torch.allclose(a, b), "image features must influence the logits"
    assert not torch.allclose(a, c)
    with pytest.raises(Exception):  # 2 images in input_ids but features for 1
        model(ids, image_features=feats[:1])


def test_param_count_matches_estimator():
    for kw in ({}, {"moe": MOE, "mtp_block": True}, {"tie_embeddings": False, "mtp_depth": 2}):
        cfg, model = build(kw)
        assert model.num_params() == estimate_params(cfg).total


def test_init_rules():
    cfg, model = build({"n_layers": 4})
    mixer = model.layers[0].mixer
    if mixer.shift is not None:  # token shift starts as identity
        w = mixer.shift.weight
        assert torch.all(w[:, 0, -1] == 1) and torch.all(w[:, 0, :-1] == 0)
    gamma = torch.sigmoid(mixer.decay_logit.detach())
    expected = torch.sigmoid(torch.tensor(decay_init_logits(mixer.n_ret, 0)))
    torch.testing.assert_close(gamma, expected)
    assert torch.all(model.norm.weight == 1)
    std = model.layers[1].mixer.o_proj.weight.std().item()
    assert std == pytest.approx(cfg.model.init_std / math.sqrt(2 * cfg.model.n_layers), rel=0.25)


# ================================================================================================
# retention forms agree
# ================================================================================================

def _ret_inputs(B=2, H=3, T=70, dk=8, dv=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, dk, generator=g) * dk ** -0.5
    k = torch.randn(B, H, T, dk, generator=g)
    v = torch.randn(B, H, T, dv, generator=g)
    log_gamma = torch.nn.functional.logsigmoid(torch.tensor(decay_init_logits(H, 3)))
    return q, k, v, log_gamma


@pytest.mark.parametrize("chunk", [1, 16, 64, 128])
def test_retention_parallel_equals_chunkwise(chunk):
    q, k, v, lg = _ret_inputs()
    par = retention_parallel(q, k, v, lg)
    ch = retention_chunkwise(q, k, v, lg, chunk=chunk)
    ch = ch[0] if isinstance(ch, tuple) else ch
    torch.testing.assert_close(ch, par, atol=1e-4, rtol=1e-4)


def test_retention_recurrent_matches_parallel_and_state():
    q, k, v, lg = _ret_inputs(T=40)
    par = retention_parallel(q, k, v, lg)
    state = None
    outs = []
    for t in range(q.shape[2]):
        o, state = retention_recurrent_step(q[:, :, t], k[:, :, t], v[:, :, t], state, lg)
        outs.append(o)
    rec = torch.stack(outs, dim=2)
    torch.testing.assert_close(rec, par, atol=1e-4, rtol=1e-4)
    # S_T = sum_j gamma^(T-1-j) k_j^T v_j
    T = q.shape[2]
    w = torch.exp(lg[:, None] * torch.arange(T - 1, -1, -1, dtype=torch.float32))  # (H, T)
    expected = torch.einsum("ht,bhtk,bhtv->bhkv", w, k, v)
    assert state.dtype == torch.float32
    torch.testing.assert_close(state, expected, atol=1e-4, rtol=1e-4)
    _, ch_state = retention_chunkwise(q, k, v, lg, chunk=16, return_state=True)
    torch.testing.assert_close(ch_state, expected, atol=1e-4, rtol=1e-4)


def test_long_sequence_uses_chunkwise_consistently():
    """T > 256 takes the chunkwise path inside the model; it must agree with a cached decode."""
    cfg, model = build({"retention_ratio": 1.0, "max_seq_len": 512, "n_layers": 1}, train={"seq_len": 300})
    model.eval()
    ids, _ = batch(cfg, B=1, T=300)
    with torch.no_grad():
        full = model(ids).logits
        out = model(ids[:, :-1], use_cache=True)
        step = model(ids[:, -1:], cache=out.cache, use_cache=True)
    torch.testing.assert_close(out.logits[:, 0], full[:, -2], atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(step.logits[:, 0], full[:, -1], atol=ATOL, rtol=1e-4)


# ================================================================================================
# cached decode == full forward
# ================================================================================================

CACHE_CASES = {
    "global": {"retention_ratio": 0.0, "token_shift": 0},
    "sliding": {"retention_ratio": 0.0, "attn_window": 6, "global_every": 2, "token_shift": 0},
    "retention": {"retention_ratio": 1.0, "token_shift": 0},
    "token-shift": {"token_shift": 4},
    "hybrid-gqa": {"n_heads": 6, "retention_ratio": 0.34, "n_kv_heads": 2, "attn_window": 8, "global_every": 2},
    "moe": {"moe": MOE, "n_layers": 3},
}


@pytest.mark.parametrize("case", sorted(CACHE_CASES))
def test_cached_decode_matches_full_forward(case):
    cfg, model = build(CACHE_CASES[case])
    model.eval()
    B, T, P = 2, 20, 9
    ids, _ = batch(cfg, B=B, T=T, seed=5)
    with torch.no_grad():
        full = model(ids).logits                                   # (B, T, V)
        out = model(ids[:, :P], use_cache=True)
        assert out.logits.shape == (B, 1, cfg.model.vocab_size)
        torch.testing.assert_close(out.logits[:, 0], full[:, P - 1], atol=ATOL, rtol=1e-4)
        cache = out.cache
        assert isinstance(cache, InferenceCache) and cache.seen_tokens == P
        for t in range(P, T):
            step = model(ids[:, t:t + 1], cache=cache, use_cache=True)
            cache = step.cache
            assert torch.allclose(step.logits[:, 0], full[:, t], atol=ATOL, rtol=1e-4), (
                f"{case}: cached logits differ at position {t}: "
                f"max abs diff {(step.logits[:, 0] - full[:, t]).abs().max().item():.3e}")
    assert cache.seen_tokens == T
    hd, d = cfg.model.head_dim, cfg.model.d_model
    for i, layer in enumerate(cache.layers):
        n_ret = model.layers[i].mixer.n_ret
        if n_ret:
            assert tuple(layer.state.shape) == (B, n_ret, hd, hd) and layer.state.dtype == torch.float32
        window = model.layers[i].mixer.window
        if model.layers[i].mixer.n_attn and window:
            assert layer.k.shape[2] <= window, "local layers must trim their KV cache to the window"
        if cfg.model.token_shift:
            assert tuple(layer.conv_buf.shape) == (B, d, cfg.model.token_shift - 1)


def test_cache_reorder_and_repeat():
    cfg, model = build()
    model.eval()
    ids, _ = batch(cfg, B=1, T=8)
    with torch.no_grad():
        out = model(ids, use_cache=True)
        cache = out.cache.repeat(3)
        assert cache.batch_size == 3
        nxt = torch.tensor([[11], [12], [13]])
        step = model(nxt, cache=cache, use_cache=True)
        cache.reorder(torch.tensor([2, 2, 0]))
        again = model(torch.tensor([[20], [20], [20]]), cache=cache, use_cache=True)
        ref_cache = model(torch.cat([ids, torch.tensor([[13]])], dim=1), use_cache=True).cache
        ref = model(torch.tensor([[20]]), cache=ref_cache, use_cache=True)
    assert step.logits.shape[0] == 3
    torch.testing.assert_close(again.logits[0], ref.logits[0], atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(again.logits[1], ref.logits[0], atol=ATOL, rtol=1e-4)


# ================================================================================================
# generation
# ================================================================================================

def test_greedy_generation_matches_argmax_and_is_deterministic():
    cfg, model = build({"moe": MOE, "n_layers": 3})
    model.eval()
    prompt = [1, 4, 40, 41, 42, 6, 5]
    kw = dict(max_new_tokens=6, temperature=0.0, repetition_penalty=1.0, no_repeat_ngram=0, stop_ids=(), n=1)
    r1 = generate(model, prompt, **kw)
    r2 = generate(model, torch.tensor(prompt), **kw)
    assert isinstance(r1, GenerationResult) and r1.tokens == r2.tokens and len(r1.tokens) == 6
    seq = list(prompt)
    with torch.no_grad():
        for _ in range(6):
            seq.append(int(model(torch.tensor([seq])).logits[0, -1].argmax()))
    assert r1.tokens == seq[len(prompt):]


def test_sampling_best_of_n_and_stream():
    cfg, model = build()
    prompt = [1, 4, 50, 51, 6, 5]
    res = generate(model, prompt, max_new_tokens=5, temperature=0.9, top_k=20, top_p=0.9, repetition_penalty=1.1,
                   no_repeat_ngram=2, n=3, seed=123)
    assert len(res.all_tokens) == len(res.scores) == len(res.finished) == 3
    assert res.tokens == res.all_tokens[max(range(3), key=lambda i: res.scores[i])]
    assert all(math.isfinite(s) for s in res.scores)
    again = generate(model, prompt, max_new_tokens=5, temperature=0.9, top_k=20, top_p=0.9, repetition_penalty=1.1,
                     no_repeat_ngram=2, n=3, seed=123)
    assert again.all_tokens == res.all_tokens, "seeded sampling must be reproducible"
    streamed = []
    one = generate(model, prompt, max_new_tokens=5, n=1, seed=1, stream=streamed.append)
    assert streamed == one.tokens
    banned = generate(model, prompt, max_new_tokens=6, n=1, seed=2, banned_ids=list(range(0, 200)), stop_ids=())
    assert all(t >= 200 for t in banned.tokens)


def test_generation_stops_on_stop_id():
    cfg, model = build()
    prompt = [1, 4, 30, 31, 6, 5]
    greedy = dict(temperature=0.0, repetition_penalty=1.0, no_repeat_ngram=0)
    first = generate(model, prompt, max_new_tokens=1, stop_ids=(), **greedy).tokens[0]
    res = generate(model, prompt, max_new_tokens=10, stop_ids=(first,), **greedy)
    assert res.finished == [True] and res.tokens == [], "a stop token ends generation and is not returned"
    empty = generate(model, prompt, max_new_tokens=0)
    assert empty.tokens == []
