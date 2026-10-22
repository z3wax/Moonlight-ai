# Moonlight architecture

Moonlight is a decoder-only causal language model in which **every layer mixes two kinds of heads**:
softmax-attention heads (GQA, RoPE, optional sliding window) and **retention** heads (a linear
recurrence with a learned per-head exponential decay). The same code covers every size from 1M to
1T parameters, dense and Mixture-of-Experts, with optional image input.

This document mirrors SPEC §3–§5 and is the reference for `moonlight/model/*`. All shapes use
`B` batch, `T` sequence length, `V` vocabulary, `d` model width, `L` layers.

> Design goals, not measured results. The hybrid layout is meant to give O(1)-per-token decoding
> state on retention heads, a smaller KV cache, and efficient training with Muon and multi-token
> prediction. Whether a given Moonlight model beats a Transformer of the same size is an empirical
> question; the `cc-00x` benchmark suites (see [BENCHMARKS.md](BENCHMARKS.md)) exist to **measure** it.

---------------------------------------------------------------------------------------------------

## 1. Head budget

```
H   = n_heads                         total query heads
Hr  = round(H * retention_ratio)      retention heads
Ha  = H - Hr                          attention heads
Hkv = n_kv_heads if Ha > 0 else 0     key/value heads shared by the attention heads (GQA, Ha % Hkv == 0)
hd  = head_dim                        n_heads * head_dim need NOT equal d_model
```

Typical ratios: 0.5 for test/mini models, 0.375 for 1B–2B, and 0.25 from 3B upward.

## 2. Block

```
x = x + HybridMixer(RMSNorm(x))       # attn_norm
x = x + FFN(RMSNorm(x))               # ffn_norm; dense SwiGLU or MoE
```

`RMSNorm(x) = x / sqrt(mean(x²) + 1e-6) * w`. Activation checkpointing can wrap each block (training only).

## 3. HybridMixer (x: B×T×d, already normalised)

1. **Token shift**: `xs = conv1d_causal(x)`, a depthwise `Conv1d(d, d, k, groups=d)` with left
   padding `k-1`, initialised to identity (`w[:, 0, -1] = 1`). `qkv` reads `xs`; the output gate
   reads the unshifted `x`.
2. **Projections**: one fused `qkv = Linear(d, hd·(H + 2·Hkv + 2·Hr))` with split order
   `[q (Ha attention heads, then Hr retention heads) | k_attn (Hkv) | v_attn (Hkv) | k_ret (Hr) | v_ret (Hr)]`.
   With `qk_norm`, q and k get a parameter-free RMS normalisation over `head_dim`.
3. **Value residual** (`value_residual`): layer 0 stores `v_first = concat(v_attn, v_ret)`
   (B, Hkv+Hr, T, hd). Each later layer mixes `v ← v + σ(λ)·(v_first − v)`, where `λ` is a scalar
   parameter initialised to 0 (a 0.5 mix).
4. **Attention heads**: RoPE (rotate-half, `θ = rope_theta`, positions offset by the cache length)
   on `q_a, k_a`; GQA; causal SDPA. A layer `i` is *global* when `attn_window` is null or
   `i % global_every == global_every − 1`. Otherwise it is *local*, and query `t` sees keys in `(t − window, t]`.
5. **Retention heads**: `γ_h = σ(decay_logit_h)`, with `q` scaled by `hd^-1/2`:

   ```
   o_t = Σ_{j ≤ t} γ^{t−j} (q_t · k_j) v_j
   ```

   Initial decays spread per layer: `γ = 1 − 2^(−linspace(lo, hi, Hr))`, where `lo = 2 + 0.3·(i mod 8)` and
   `hi = 5 + 0.4·(i mod 8)`, so different layers start with different memory lengths. Three
   mathematically identical implementations are used (and tested against each other); the decay math runs in float32:

   | form | cost | used for |
   |---|---|---|
   | parallel: `(Q Kᵀ ⊙ D) V`, `D_tj = γ^{t−j}·[j ≤ t]` | O(T²) | training / prefill, T ≤ 256 |
   | chunkwise (chunk 64): intra-chunk parallel + inter-chunk state scan | O(T·C) | T > 256 |
   | recurrent: `S_t = γ S_{t−1} + k_tᵀ v_t`, `o_t = q_t S_t` | O(1)/token | decoding |

   After prefill the state is `S_T = Σ_j γ^{T−1−j} k_jᵀ v_j` (B, Hr, hd, hd). A per-head
   `RMSNorm(hd)` (`ret_norm`) follows.
6. **Output**: concatenate heads `[attn (Ha) | ret (Hr)]` → (B, T, H·hd), multiply by `silu(gate(x))`
   when `output_gate`, then apply `o_proj = Linear(H·hd, d)`.

### Why hybrid?

| cache per layer and sequence | size |
|---|---|
| attention (global) | `2 · Hkv · T · hd` values, grows with T |
| attention (local, window W) | `2 · Hkv · W · hd`, bounded |
| retention | `Hr · hd · hd` (float32), **constant in T** |

With `retention_ratio = 0.25` and sliding windows on 3 of every 4 layers (the SI configs), most
of the decoding state stays bounded, while the remaining global attention layers keep exact long-range recall.

## 4. Feed-forward

* **Dense SwiGLU**: `w_down(silu(a) ⊙ b)`, where `[a | b] = w_gate_up(x)`, `w_gate_up: d→2h`, `w_down: h→d`.
* **MoE** (DeepSeek-V3 style), applied to layers `i ≥ first_dense_layers`:
  * `scores = sigmoid(router(x))` (float32). Top-k **selection** uses `scores + router_bias`; the
    **combine weights** use the unbiased scores renormalised over the selected experts.
  * Experts are stored as 3-D parameters `w_gate_up (E, d, 2he)` and `w_down (E, he, d)`. Dispatch is
    dropless: tokens are sorted by expert, each expert with tokens runs once, and results are combined with `index_add_`.
  * Optional shared experts form one `SwiGLU(d, he·n_shared)` that is always active.
  * Balancing: a sequence-level auxiliary loss `aux_loss_weight · E · Σ_e f_e·P_e`, plus **aux-loss-free**
    bias updates. After each optimizer step, `model.update_router_biases()` applies
    `bias += rate · sign(mean_load − load)`. `router_bias` is a persistent float32 buffer, not a parameter.

## 5. Model head, losses and MTP

```
logits = softcap · tanh(z / softcap)    z = norm(h) @ Eᵀ (tied) or lm_head(norm(h)); float32
loss   = CE(logits, targets) + mtp_weight · mean_j CE_j + aux_moe + z_loss · mean(logsumexp(z)²)
```

`targets` are already shifted by the caller (`-100` = ignore).
**Multi-token prediction**: head `j` (0-based) predicts token `t+j+2`:
`h' = proj(cat(norm_h(h), norm_e(embed(x_{t+j+1}))))`, followed by a full `Block` when `mtp_block`. Its logits
reuse the final norm and output head. Depth 0 starts from the last layer's hidden state *before*
the final norm.

With `use_cache=True` and no targets, only the last position's logits are computed (B, 1, V).

## 6. Images (multimodal configs)

A frozen SigLIP encoder (`google/siglip-base-patch16-224`, CLIP fallback) produces patch features.
These are average-pooled to `n_image_tokens` (49 = 7×7) vectors of `vision_dim`. The trainable
`VisionProjector` (`RMSNorm → Linear(vision_dim, d) → GELU → Linear(d, d)`) maps them into the
embedding space, and they replace the `<|image|>` (id 7) placeholder embeddings in row-major order
(`masked_scatter`). The number of placeholders must equal `B_img · n_image_tokens`.

## 7. Exact parameter layout

`moonlight.params.estimate_params(cfg)` reproduces the model's parameter count **exactly**. This is
tested for all 100 configs by instantiating them on the `meta` device. All linear layers are
bias-free. Per block `i`:

| tensor | count |
|---|---|
| `attn_norm`, `ffn_norm` | `d` each |
| `mixer.qkv` | `d · hd · (H + 2Hkv + 2Hr)` |
| `mixer.gate` (output_gate) | `d · H · hd` |
| `mixer.o_proj` | `H · hd · d` |
| `mixer.shift` (k > 0) | `d · k` |
| `mixer.decay_logit`, `mixer.ret_norm` (Hr > 0) | `Hr`, `hd` |
| `mixer.v_lambda` (value_residual, i > 0) | `1` |
| dense FFN | `3 · d · h` |
| MoE FFN | `d·E` (router) `+ 3·E·d·he` `+ 3·d·he·n_shared` |

Plus `embed` (`V·d`), the final `norm` (`d`), `lm_head` (`V·d`, only when untied), each MTP head
(`2d + 2d²` + one block when `mtp_block`), and `vision_proj` (`vision_dim + vision_dim·d + d²`).
Buffers such as RoPE tables and router biases are not counted. The **active** count counts routed
experts at a `top_k/E` fraction.

## 8. Initialisation

Linear and embedding weights use `N(0, init_std)`. `o_proj` and every FFN down projection (dense,
shared, routed) use `N(0, init_std/√(2L))`. RMSNorm weights start at 1, token shift at identity,
retention decays at the spread above, `v_lambda` at 0, and the vision `fc2` at `N(0, init_std)` (not zero).

## 9. Inference cache and generation

`InferenceCache(n_layers)` holds the following per layer:

* attention `k, v` (B, Hkv, T, hd), trimmed to the window on local layers;
* retention `state` (B, Hr, hd, hd) in float32;
* the token-shift `conv_buf` (B, d, k−1);
* `seen_tokens`.

`reorder(index)` and `repeat(n)` support best-of-N sampling. A cached prefill followed by
single-token steps reproduces the full forward pass (tested to `atol 1e-4` in fp32 for global,
sliding-window, retention, token-shift and MoE layers).

`generate()` supports temperature (0 means greedy), top-k, top-p, repetition penalty, n-gram
blocking, banned ids, stop ids (`<|end|>`, `<|eos|>`) and streaming. Best-of-N samples `n`
continuations in one batch and scores each by its mean untempered log-probability, with penalties
for unfinished or too-short replies.

## 10. Tokenizer and chat template

Byte-level BPE (HF `tokenizers`) with an Uzbek-apostrophe-aware pre-tokenizer
(`o'zbek`, `g'alaba` stay one word; ʻ ʼ ‘ ’ ` ´ are normalised to `'`) and one token per digit.
32 special tokens have fixed ids:

```
0 <|pad|> 1 <|bos|> 2 <|eos|> 3 <|system|> 4 <|user|> 5 <|assistant|> 6 <|end|> 7 <|image|>
8 <|tool_call|> 9 <|tool_result|> 10 <|tools|> 11 <|think|> 12 <|/think|> 13 <|code|>
14 <|audio|> 15..31 <|reserved_0..16|>
```

```
<|bos|><|system|>{system}[<|tools|>{tool schemas JSON}]<|end|>
<|user|>[<|image|>×n_image_tokens per image]{content}<|end|>
<|assistant|>{content}<|end|>
<|assistant|><|tool_call|>{"name": "...", "arguments": {...}}<|end|>
<|tool_result|>{content}<|end|>
```

The loss covers only assistant content, tool calls, and the assistant's closing `<|end|>`.
`<|audio|>` is reserved for the planned speech-to-speech assistant.

## 11. Parallelism

DDP, plus FSDP (FSDP2 `fully_shard` per block when available, otherwise FSDP1 with transformer
auto-wrap and bf16 mixed precision), are implemented. `tp_size` and `ep_size` in the configs are
**documented placeholders**: tensor and expert parallelism are not implemented yet, which is why
the configs at or above 1B are reference recipes for large clusters.
