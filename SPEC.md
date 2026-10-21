# MOONLIGHT — Engineering Specification (single source of truth)

Moonlight is a family of hybrid **Attention + Retention** language models (1M → 1T params) with one
library for: model code, data pipelines, tokenizer, training (single GPU → multi-node), chat with
tool-calling, multimodal (image) input, and a tiered benchmark harness (`cc-001` … `cc-010`).
Primary languages: **Uzbek (Latin)** and **English**; all chat models are trained for
**conversation + coding + tool calling**.

Repository root: `/home/user/moonlight-llm` (git repo). NEVER touch `/home/user/moonlight` (unrelated repo).

Every module MUST follow the interfaces below exactly — other modules are written in parallel
against this document. If something is unspecified, choose the simplest robust option and document
it in a docstring. Python ≥ 3.9 syntax (no `match`, no `X | Y` types at runtime — use
`from __future__ import annotations`). torch ≥ 2.1. Code comments/docstrings in English; user-facing
CLI/log strings may be English (README has Uzbek).

**Import hygiene (important):** these modules must import WITHOUT torch installed (torch imported
lazily inside functions only): `moonlight.config`, `moonlight.params`, `moonlight.registry`,
`moonlight.utils`, `moonlight.tokenizer`, `moonlight.data.sources`, `moonlight.data.cleaning`,
`moonlight.data.adapters`, `moonlight.data.synthetic`, `moonlight.data.sft` (numpy only),
`moonlight.tools`, `moonlight.benchmarks.suites`. `moonlight/__init__.py` must import lazily
(PEP 562 `__getattr__`) so `import moonlight; moonlight.config("ml-1m-chat")` works without torch.

---------------------------------------------------------------------------------------------------
## 0. Package layout & ownership

```
pyproject.toml                      [A]
moonlight/__init__.py               [F]  public API (lazy)
moonlight/__main__.py               [F]  -> cli.main()
moonlight/version.py                [A]  __version__ = "0.1.0"
moonlight/config.py                 [A]  dataclasses + YAML load/save + overrides + validation
moonlight/params.py                 [A]  exact param-count estimator (pure python)
moonlight/registry.py               [A]  find/list the 100 configs
moonlight/utils.py                  [A]  helpers (seed, device, human-readable numbers, logging)
moonlight/model/__init__.py         [B]
moonlight/model/layers.py           [B]  RMSNorm, rms(), RoPE, SwiGLU, TokenShift
moonlight/model/retention.py        [B]  retention: parallel, chunkwise, recurrent
moonlight/model/attention.py        [B]  GQA attention (SDPA), sliding window, KV cache
moonlight/model/moe.py              [B]  MoE FFN (shared + routed experts, aux-free balancing)
moonlight/model/cache.py            [B]  InferenceCache
moonlight/model/model.py            [B]  HybridMixer, Block, MTPHead, MoonlightForCausalLM
moonlight/model/vision.py           [B]  VisionProjector, VisionEncoder (lazy transformers)
moonlight/model/generation.py       [B]  sampling, best-of-N, streaming generate()
moonlight/tokenizer.py              [C]  MoonlightTokenizer, train_tokenizer, chat template
moonlight/assets/sample_uz.txt      [C]  ~30–60 KB ORIGINAL Uzbek Latin text (written by agent)
moonlight/assets/sample_en.txt      [C]  ~20 KB ORIGINAL English text
moonlight/assets/sample_code.py     [C]  ~15 KB original clean Python code (functions, docstrings)
moonlight/data/__init__.py          [D]
moonlight/data/sources.py           [D]  DATASETS registry (HF urls/formats)
moonlight/data/download.py          [D]  cached HTTP download (HF_TOKEN), parquet/jsonl readers
moonlight/data/cleaning.py          [D]  Uzbek/English/code cleaning + filters
moonlight/data/adapters.py          [D]  row -> text  /  row -> Conversation adapters
moonlight/data/synthetic.py         [D]  identity, greetings, knowledge, arithmetic, tool-call, code
moonlight/data/pretrain.py          [D]  build token bins (memmap), PretrainLoader
moonlight/data/sft.py               [D]  pack conversations, SFTLoader
moonlight/data/vision.py            [D]  image-caption pairs -> cached features + conversations
moonlight/train/__init__.py         [E]
moonlight/train/optim.py            [E]  Muon (batched NS, 3D experts) + AdamW param groups
moonlight/train/schedule.py         [E]  WSD / cosine / constant schedules
moonlight/train/distributed.py      [E]  DDP / FSDP setup from torchrun env
moonlight/train/checkpoint.py       [E]  safetensors checkpoints + resume
moonlight/train/trainer.py          [E]  Trainer (phases, AMP, compile, accum, EMA, eval, budgets)
moonlight/llm.py                    [F]  LLM facade
moonlight/chat.py                   [F]  ChatSession (multi-turn, tools loop, images)
moonlight/tools.py                  [F]  Tool registry, parsing, built-in tools
moonlight/cli.py                    [F]  `moonlight ...` CLI
moonlight/benchmarks/__init__.py    [G]
moonlight/benchmarks/suites.py      [G]  Suite objects, tier mapping, `type()` accessor
moonlight/benchmarks/suites/cc-001.yaml … cc-010.yaml   [G]
moonlight/benchmarks/tasks.py       [G]  task implementations
moonlight/benchmarks/execution.py   [G]  sandboxed python execution (subprocess+timeout)
moonlight/benchmarks/runner.py      [G]  run suite -> Report (json/markdown)
scripts/generate_configs.py         [H]  writes the 100 YAMLs deterministically
configs/{test,mini,si}/*.yaml       [H]  EXACTLY 100 files
docs/CONFIGS.md                     [H]  generated table of all configs
tests/*.py                          [I]
.github/workflows/ci.yml            [I]
README.md, docs/*.md, notebooks/colab_quickstart.py, LICENSE (Apache-2.0), .gitignore  [I]
```

---------------------------------------------------------------------------------------------------
## 1. Public API (moonlight/__init__.py, lazy)

```python
import moonlight as ml
cnf = ml.config("ml-20m-chat")          # registry name OR path to .yaml  -> Config
cnf = ml.config("ml-20m-chat", overrides={"train.batch_size": 16})   # dotted overrides
ml.list_configs(tier=None)              # -> list[str] (sorted)
llm = ml.LLM(cnf)                       # build model + tokenizer (tokenizer trained/loaded lazily on train)
llm.train()                             # runs cnf.train.phases; returns TrainResult(dict-like)
llm.chat("Salom!")                      # -> str
llm.chat("2+2?", tools=[...])           # tool-calling loop
suite = cnf.bnchmrks.type("cc-001")     # also cnf.benchmarks.type(...) ; returns Suite
report = suite.run(llm)                 # -> Report ; report.to_markdown(); report.save(path)
llm.benchmark("cc-001")                 # shorthand
llm.save("out/ml-20m-chat.safetensors"); llm2 = ml.LLM.from_pretrained("out/ml-20m-chat.safetensors")
ml.train("ml-20m-chat")                 # one-liner: config -> LLM -> train -> save
ml.benchmarks.type("cc-001")            # suite without a config
ml.estimate_params(cnf)                 # -> ParamReport
```
Exports: `config, list_configs, LLM, train, Config, estimate_params, benchmarks, ChatSession, Tool,
tool (decorator), __version__`.

---------------------------------------------------------------------------------------------------
## 2. Config schema (moonlight/config.py)

Dataclasses (all fields have defaults; YAML may omit any). `Config.from_yaml(path)`,
`Config.from_dict(d)`, `cfg.to_dict()`, `cfg.to_yaml(path=None) -> str`, `cfg.apply_overrides({"a.b": v})`,
`cfg.validate()` (raises `ValueError` with clear message), `cfg.copy()`. Unknown keys -> `ValueError`
listing the key path. YAML may contain `extends: <name-or-path>` (deep-merge child over parent;
resolution via registry), but generated configs are fully explicit (no extends).
`Config.bnchmrks` and `Config.benchmarks` are properties returning a `BenchmarkAccessor` with method
`.type(name) -> Suite` (lazy import of moonlight.benchmarks.suites) and `.default()` -> suite named in
`cfg.benchmark`.

```yaml
name: ml-20m-chat               # str, required (must equal file stem for registry configs)
description: "..."
tier: test                      # test | mini | si
kind: chat                      # base | chat | code | vl | vl-chat | tools | agent | think | long | debug
benchmark: cc-002               # default benchmark suite
nominal_params: 20M             # human label used in name; estimator must be within ±25%
model:
  vocab_size: 8192
  d_model: 384
  n_layers: 10
  n_heads: 6                    # total query heads (attention + retention)
  head_dim: 64
  retention_ratio: 0.5          # fraction of heads that are retention heads (round to int, >=0)
  n_kv_heads: 1                 # KV heads for the ATTENTION heads (must divide n_attn_heads)
  attn_window: null             # int | null; sliding-window size for "local" layers
  global_every: 1               # every Nth layer (idx % N == N-1) uses full attention; others use window
                                # (if attn_window is null all layers are global)
  ffn_hidden: 1024              # dense SwiGLU hidden size
  token_shift: 4                # depthwise causal conv kernel (0 = off)
  value_residual: true
  qk_norm: true
  output_gate: true
  rope_theta: 10000.0
  max_seq_len: 1024
  tie_embeddings: true
  softcap: 30.0                 # final logit soft-cap (0 = off)
  mtp_depth: 1                  # number of multi-token-prediction heads (0 = off)
  mtp_block: false              # each MTP head contains a full Block (DeepSeek-V3 style)
  mtp_weight: 0.3
  z_loss: 0.0
  init_std: 0.02
  moe:
    enabled: false
    n_experts: 0
    top_k: 0
    n_shared: 0
    expert_hidden: 0
    first_dense_layers: 0       # first N layers use dense ffn_hidden SwiGLU
    aux_loss_weight: 0.001      # sequence-level balance loss
    bias_update_rate: 0.001     # aux-loss-free balancing (DeepSeek-V3); router bias is a buffer
    capacity_factor: 0.0        # 0 = dropless
  activation_checkpointing: false
multimodal:
  enabled: false
  vision_encoder: google/siglip-base-patch16-224
  vision_dim: 768
  n_image_tokens: 49
  freeze_encoder: true
tokenizer:
  vocab_size: 8192              # must equal model.vocab_size
  path: null                    # existing tokenizer.json; if null it is trained during llm.train()
  train_chars: 50000000
  split_digits: true
data:
  cache_dir: ~/.cache/moonlight
  offline: false                # true -> only bundled assets + synthetic data (used by tests/debug)
  languages: [uz, en]
  pretrain:                     # list of {source, weight, max_chars}
    - {source: uz-news, weight: 0.25, max_chars: 2.5e8}
  sft:                          # list of {source, max_examples}
    - {source: slimorca-uz, max_examples: 150000}
  synthetic:
    identity: true
    knowledge: true
    arithmetic: 3000
    tool_calls: 2000
    code: 1000
  vision: []                    # list of {source, max_images}
train:
  phases:                       # executed in order
    - {name: pretrain, tokens: 2.0e8, minutes: 70, lr_scale: 1.0, sft_mix: 0.0, anneal_sft_mix: 0.25}
    - {name: align, minutes: 6, lr_scale: 1.0, trainable: vision_proj}     # only if multimodal
    - {name: sft, minutes: 25, epochs: 2.0, lr_scale: 0.5, replay: 0.1, image_mix: 0.3}
  seq_len: 512
  batch_size: 32                # micro-batch (sequences per device per step)
  batch_tokens: 16384           # global tokens per optimizer step -> grad accumulation derived
  optimizer: {name: muon, lr: 0.02, momentum: 0.95, adam_lr: 0.003, embed_lr: 0.003,
              betas: [0.9, 0.95], weight_decay: 0.0, grad_clip: 1.0}
  schedule: {name: wsd, warmup_steps: 200, decay_frac: 0.35, min_lr_ratio: 0.02}
  precision: auto               # auto | bf16 | fp16 | fp32
  compile: true
  ema: 0.998                    # 0 = off
  curriculum: {enabled: true, start_seq_len: 128, until_frac: 0.3}
  eval_every: 1000
  eval_batches: 20
  log_every: 100
  checkpoint_every_min: 10
  output_dir: runs/ml-20m-chat
  seed: 1337
  hardware: colab-t4            # informational preset name
  distributed: {strategy: none, fsdp_shard: full, tp_size: 1, ep_size: 1}  # none|ddp|fsdp
chat:
  name: Moonlight
  system_prompt: "Sen Moonlight — ..."
  best_of: 3
  temperature: 0.7
  top_k: 40
  top_p: 0.9
  repetition_penalty: 1.1
  no_repeat_ngram: 3
  max_new_tokens: 256
  tools: true
```
Validation rules: `n_heads*head_dim` need NOT equal d_model (projections handle it);
`n_ret = round(n_heads*retention_ratio)`, `n_attn = n_heads - n_ret`; if `n_attn>0` then
`n_kv_heads>=1` and `n_attn % n_kv_heads == 0`; if `n_attn==0` n_kv_heads ignored;
`tokenizer.vocab_size == model.vocab_size`; `moe.enabled -> n_experts>=top_k>=1, expert_hidden>0`;
`multimodal.enabled` requires tier in (test, mini) or nominal ≥ 30B OR kind in (vl, vl-chat)
(just warn otherwise); `kind in ("vl","vl-chat") -> multimodal.enabled`.
`parse_size("20M") -> 20_000_000`, `"1.5B"`, `"1T"`, ints allowed. `format_size(n) -> "20.1M"`.

---------------------------------------------------------------------------------------------------
## 3. Exact parameter layout (model ↔ params.py MUST agree exactly; tested on meta device)

Let V=vocab_size, d=d_model, L=n_layers, H=n_heads, hd=head_dim, Hr=round(H*retention_ratio),
Ha=H-Hr, Hkv=n_kv_heads if Ha>0 else 0, k=token_shift.
All Linear layers have `bias=False`.

`MoonlightForCausalLM` attributes (state_dict names):
- `embed` : nn.Embedding(V, d)                                       → V·d
- `layers.{i}` : Block                                               (i = 0..L-1)
- `norm` : RMSNorm(d) (param `weight`)                              → d
- `lm_head` : nn.Linear(d, V) ONLY if not tie_embeddings             → V·d
- `mtp.{j}` : MTPHead (j = 0..mtp_depth-1)
- `vision_proj` : VisionProjector ONLY if multimodal.enabled

`Block` (layer index i):
- `attn_norm` RMSNorm(d)                                             → d
- `mixer` HybridMixer:
  - `qkv` Linear(d, hd·(H + 2·Hkv + 2·Hr))                           → d·hd·(H+2Hkv+2Hr)
    output split order: [q (H heads: first Ha attention heads, then Hr retention heads) |
    k_attn (Hkv) | v_attn (Hkv) | k_ret (Hr) | v_ret (Hr)]
  - `gate` Linear(d, H·hd) if output_gate                            → d·H·hd
  - `o_proj` Linear(H·hd, d)                                         → H·hd·d
  - `shift` nn.Conv1d(d, d, k, groups=d, bias=False) if k>0          → d·k
  - `decay_logit` Parameter(Hr) if Hr>0                              → Hr
  - `ret_norm` RMSNorm(hd) if Hr>0                                   → hd
  - `v_lambda` Parameter(()) scalar if value_residual and i>0        → 1
- `ffn_norm` RMSNorm(d)                                              → d
- `ffn`: dense `SwiGLU(d, ffn_hidden)` if not moe or i < first_dense_layers:
  - `w_gate_up` Linear(d, 2h), `w_down` Linear(h, d)                  → 3·d·h
  else `MoE`:
  - `router` Linear(d, E)                                            → d·E
  - buffer `router_bias` (E,) float32, persistent=True (NOT a parameter)
  - `w_gate_up` Parameter(E, d, 2·he), `w_down` Parameter(E, he, d)  → 3·E·d·he
  - `shared` SwiGLU(d, he·n_shared) if n_shared>0                    → 3·d·he·n_shared

`MTPHead` j: `norm_h` RMSNorm(d), `norm_e` RMSNorm(d), `proj` Linear(2d, d), and `block` Block (layer
index L+j, dense FFN unless moe, i.e. same rule as a non-first layer) if mtp_block → 2d + 2d² (+block).
MTP heads reuse `embed`, final `norm`, and the output head.

`VisionProjector`: `norm` RMSNorm(vision_dim), `fc1` Linear(vision_dim, d), `fc2` Linear(d, d)
→ vision_dim + vision_dim·d + d².

params.py API:
```python
@dataclass
class ParamReport: total:int; active:int; embedding:int; per_layer:list[int]; breakdown:dict[str,int]
    def pretty(self)->str
def estimate_params(cfg_or_model_cfg) -> ParamReport   # active = params used per token
                                                        # (routed experts counted top_k/E fraction)
def flops_per_token(cfg, seq_len=None) -> float          # ~6*active + attention term (for training)
def memory_estimate(cfg, dtype="bf16") -> dict           # weights / grads / optimizer GB
```
Buffers (router_bias, rope cos/sin) are NOT counted.

---------------------------------------------------------------------------------------------------
## 4. Model math (moonlight/model)

`MoonlightForCausalLM(cfg: ModelConfig, mm: MultimodalConfig|None=None)`
```python
out = model(input_ids, targets=None, image_features=None, cache=None, use_cache=False,
            return_hidden=False)
# out: ModelOutput dataclass(logits, loss, lm_loss, aux_loss, cache, hidden)
```
- `targets[b,t]` is the label for position t (ALREADY shifted by caller), -100 = ignore.
- if `targets` is None and not use_cache: logits for all positions; if use_cache (generation) logits
  only for the last position (shape B,1,V) to save memory.
- loss = lm_loss + mtp_weight·mean_j(mtp_loss_j) + aux_loss(moe) + z_loss·mean(logsumexp²).
  MTP head j (0-based) predicts token at t+j+2, i.e. its target at position t is targets[t+j+1]
  (pad -100 at the end); its input combines h_t (previous depth hidden; depth 0 uses final-layer
  hidden BEFORE final norm) and embed(input_ids[t+j+1]) (pad id 0 at the end, masked by -100):
  `h' = proj(cat(norm_h(h), norm_e(e)))`, then block if mtp_block; logits = head(norm(h')).
- logits computed in float32, soft-cap: `c*tanh(z/c)` when softcap>0.
- Output head: `F.linear(x, embed.weight)` if tied else lm_head.
- Image features (B_img, n_image_tokens, vision_dim): projected by vision_proj and written into the
  embedding sequence at positions where `input_ids == IMAGE_TOKEN_ID` (=7) in row-major order using
  `masked_scatter`; the number of image tokens in input_ids must equal B_img·n_image_tokens.

HybridMixer forward (x: B,T,d already normed):
1. token shift: `xs = conv1d(x)` causal (left pad k-1), initialized to identity (weight[:,0,-1]=1,
   others 0). qkv and gate read `xs`… gate reads `x` (not shifted).
2. split qkv; reshape heads; q,k parameter-free RMS-normalized over head_dim if qk_norm;
3. value residual: layer 0 stores `v_first` = concat(v_attn, v_ret) along head dim (B, Hkv+Hr, T, hd);
   layers>0: `v = v + sigmoid(v_lambda)*(v_first - v)` (v_lambda init 0 → 0.5 mix).
4. attention heads: RoPE on q_a,k_a (rotate-half convention, theta=rope_theta, positions offset by
   cache length), GQA via repeat_interleave or SDPA `enable_gqa` fallback, SDPA causal; sliding window
   when this layer is local: query t attends keys in (t-window, t].
5. retention heads: decay γ_h = sigmoid(decay_logit_h) (init: per-layer spread
   `γ = 1 - 2^(-linspace(lo,hi,Hr))`, lo=2+0.3·(i mod 8), hi=5+0.4·(i mod 8)), scale = hd^-0.5 on q.
   o_t = Σ_{j≤t} γ^{t-j} (q_t·k_j) v_j. Implementations that MUST agree (tests):
   `retention_parallel(q,k,v,log_gamma)` O(T²); `retention_chunkwise(..., chunk=64)` used when T>256;
   `retention_recurrent_step(q_t,k_t,v_t,state,log_gamma)` with `S = γS + k_tᵀ v_t`, `o = q_t S`.
   Decay math in float32. Then `ret_norm` (RMSNorm over hd, with weight) per head.
6. concat heads [attn(Ha), ret(Hr)] → (B,T,H·hd); multiply by `silu(gate(x))` if output_gate; o_proj.

Block: `x = x + mixer(attn_norm(x)); x = x + ffn(ffn_norm(x))`. Optional activation checkpointing
per block (training only).

MoE: logits = router(x) (float32); scores = sigmoid(logits); selection uses `scores + router_bias`
(top_k), combine weights use un-biased scores normalized to sum 1 over selected experts.
Dropless dispatch: sort tokens by expert, loop experts with tokens, `index_add_`. aux balance loss
(Switch-style, sequence level) `aux_loss_weight * E * Σ_e f_e·P_e`. During training (model.training)
after forward, store per-expert load; `model.update_router_biases()` (called by trainer after optimizer
step) does `bias += rate * sign(mean_load - load)`.

Init: Linear/Embedding normal(0, init_std); `o_proj` and FFN down-projections (incl. experts, shared)
scaled by `1/sqrt(2·n_layers)`; RMSNorm weights 1; token shift identity; vision fc2 zero-init? NO —
normal small (std init_std). Rope buffers non-persistent.

Cache (`moonlight/model/cache.py`): `InferenceCache(n_layers)` holds per layer: attention `k,v`
(B,Hkv,T,hd) (trimmed to window for local layers), retention `state` (B,Hr,hd,hd) float32, token-shift
`conv_buf` (B,d,k-1), and `seen_tokens:int`. `cache.reorder(index)` / `cache.repeat(n)` for best-of-N.
Prefill with full sequence + `use_cache=True` must produce identical next-token logits to a full
non-cached forward; subsequent single-token steps must match (tests, atol 1e-4 fp32).
Retention prefill state: `S_T = Σ_j γ^{T-1-j} k_jᵀ v_j`.

generation.py:
```python
@torch.no_grad()
def generate(model, input_ids: list[int] | Tensor, max_new_tokens=256, temperature=0.7, top_k=40,
             top_p=0.9, repetition_penalty=1.1, no_repeat_ngram=0, stop_ids=(END_ID, EOS_ID),
             n=1, image_features=None, banned_ids=(), stream=None, seed=None) -> GenerationResult
# GenerationResult(tokens: list[int] (best), all_tokens: list[list[int]], scores: list[float],
#                  finished: list[bool])
```
best-of-N: run N samples in a batch, score = mean log-prob under the untempered model distribution,
penalize unfinished/too-short; return the best. temperature 0 → greedy. `stream(token_id)` callback
for N==1. Always uses the KV/retention cache.

Vision (`model/vision.py`): `VisionProjector(vision_dim, d, init_std)`; `VisionEncoder(name, n_tokens,
device, dtype)` lazily loads `transformers` SiglipVisionModel (fallback CLIPVisionModel
"openai/clip-vit-base-patch32"), preprocesses PIL images (resize 224 / processor mean-std),
returns (B, n_tokens, vision_dim) by adaptive avg-pooling the patch grid to sqrt(n_tokens)².

---------------------------------------------------------------------------------------------------
## 5. Tokenizer & chat template (moonlight/tokenizer.py)

Special tokens (ids FIXED, in this order, total 32):
```
0 <|pad|> 1 <|bos|> 2 <|eos|> 3 <|system|> 4 <|user|> 5 <|assistant|> 6 <|end|> 7 <|image|>
8 <|tool_call|> 9 <|tool_result|> 10 <|tools|> 11 <|think|> 12 <|/think|> 13 <|code|>
14 <|audio|> 15..31 <|reserved_0|>..<|reserved_16|>
```
Constants exported: `PAD_ID, BOS_ID, EOS_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID, END_ID, IMAGE_ID,
TOOL_CALL_ID, TOOL_RESULT_ID, TOOLS_ID, THINK_ID, THINK_END_ID, CODE_ID, AUDIO_ID, SPECIAL_TOKENS`.
BPE via HF `tokenizers`: pre-tokenizer = Split(regex, isolated) + ByteLevel(add_prefix_space=False,
use_regex=False); decoder ByteLevel. Regex (Uzbek apostrophe aware, single digits):
`[ ]?\p{L}+(?:['ʻʼ‘’]\p{L}+)*| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
(when split_digits false use `\p{N}{1,3}`). Text normalization before encode: map ʻʼ‘’`´ → '.
```python
class MoonlightTokenizer:
    @classmethod
    def from_file(cls, path) ; def save(self, path) ; def to_str(self) ; @classmethod from_str(s)
    @classmethod
    def byte_level(cls) -> MoonlightTokenizer     # untrained: 32 specials + 256 bytes (tests/offline)
    vocab_size: int (property) ; special_ids: dict
    def encode(self, text, add_bos=False) -> list[int] ; def encode_batch(self, texts) -> list[list[int]]
    def decode(self, ids, skip_special=True) -> str
    def encode_chat(self, messages, tools=None, add_generation_prompt=False, n_image_tokens=0,
                    system=None) -> tuple[list[int], list[int]]     # (ids, loss_mask)
    def render_chat(self, messages, ...) -> str   # human-readable for debugging
def train_tokenizer(texts: Iterable[str], vocab_size: int, path=None, split_digits=True,
                    min_frequency=2) -> MoonlightTokenizer
```
Messages: list of dicts `{"role": "system"|"user"|"assistant"|"tool", "content": str,
"tool_calls": [{"name":..., "arguments": {...}}] (assistant, optional), "images": int (user, number
of images, optional), "name": tool name (tool role)}`.
Template:
```
<|bos|><|system|>{system}[<|tools|>{json.dumps(tool_schemas, ensure_ascii=False)}]<|end|>
<|user|>[<|image|>×n_image_tokens per image]{content}<|end|>
<|assistant|>{content}<|end|>                                 # plain answer
<|assistant|><|tool_call|>{"name": "...", "arguments": {...}}<|end|>   # tool call (one per message)
<|tool_result|>{content}<|end|>                                # tool role
```
(no newlines between parts). loss_mask = 1 on assistant content tokens, tool_call token + JSON, and
the closing `<|end|>` of assistant messages; 0 elsewhere. add_generation_prompt appends `<|assistant|>`.
If system is None and first message is not system → `DEFAULT_SYSTEM_PROMPT` (Uzbek) is used.
`DEFAULT_SYSTEM_PROMPT = "Sen Moonlight — o'zbek va ingliz tillarida samimiy, aniq va foydali javob
beradigan, kod yoza oladigan va kerak bo'lganda vositalardan (tools) foydalanadigan sun'iy intellekt
yordamchisisan."`

---------------------------------------------------------------------------------------------------
## 6. Data (moonlight/data)

`sources.py`: `@dataclass DatasetSpec(name, kind, urls: list[str], fmt, adapter, language, license,
gated=False, notes="")`; `DATASETS: dict[str, DatasetSpec]`; `get_source(name)`.
kind ∈ pretrain|sft|vision|bench. fmt ∈ parquet|jsonl|json|json.gz. Verified HF files (use
`https://huggingface.co/datasets/<repo>/resolve/main/<path>`):
- pretrain: `uz-news` tahrirchi/uz-crawl data/news-0000{0..6}-of-00007.parquet (col text);
  `uz-telegram` tahrirchi/uz-crawl data/telegram_blogs-00000-of-00001.parquet; `uz-wiki`
  wikimedia/wikipedia 20231101.uz/train-00000-of-00001.parquet; `uz-books`
  tahrirchi/uz-books data/lat-000{00..27}-of-00028.parquet; `fineweb-edu`
  HuggingFaceTB/smollm-corpus fineweb-edu-dedup/train-00000-of-00234.parquet (…, col text);
  `cosmopedia` HuggingFaceTB/smollm-corpus cosmopedia-v2/train-00000-of-00104.parquet (col text);
  `stack-python` bigcode/the-stack-smol data/python/data.json (field content);
  `codeparrot` codeparrot/codeparrot-clean file-000000000001.json.gz (jsonl.gz, field content).
- sft: `alpaca-uz` behbudiy/alpaca-cleaned-uz data/train-00000-of-00001.parquet (instruction,input,
  output); `uzbek-instruct` UAzimov/uzbek-instruct-llm uzbek-instruct-dataset-openai.jsonl (messages);
  `sft-uz-9k` MLDataScientist/SFT-UZ-9k output_uz_sft.json (prompt, answer [strip outer quotes and \"],
  system, history); `oasst2-uz` MLDataScientist/oasst2_uzbek_threads data/train-00000-of-00001.parquet
  (text in Llama-3 header format); `slimorca-uz` MLDataScientist/SlimOrca-Dedup-Uzbek-cleaned
  data/train-0000{0,1}-of-00002.parquet (conversations [{from: system|human|gpt, value}]);
  `smoltalk` HuggingFaceTB/smoltalk data/all/train-0000{0..8}-of-00009.parquet (messages);
  `smoltalk-everyday` data/everyday-conversations/...; `smoltalk-code` data/self-oss-instruct/...;
  `smoltalk-tools` data/apigen-80k/... (use `find`-free fixed names `train-00000-of-00001.parquet`
  and document that loaders try listing via HF API `https://huggingface.co/api/datasets/<repo>/tree/main/<dir>`
  to resolve shard names when a file 404s); `python-18k` iamtarun/python_code_instructions_18k_alpaca
  data/train-00000-of-00001-8b6e212f3e1ece96.parquet; `codefeedback` m-a-p/CodeFeedback-Filtered-
  Instruction CodeFeedback-Filtered-Instruction.jsonl (query, answer); `glaive-tools`
  glaiveai/glaive-function-calling-v2 glaive-function-calling-v2.json (system, chat with USER:/
  ASSISTANT:/FUNCTION RESPONSE: and `<functioncall> {...}`) → convert to our tool format.
- vision: `flickr8k` jxie/flickr8k data/{train-00000-of-00002-2f8f6bfa852eac4b,
  train-00001-of-00002-2173151d8cd6c7fb,validation-00000-of-00001-7025a2b596f14b7b,
  test-00000-of-00001-42a2661d12c73e48}.parquet (image{bytes}, caption_0..4); `flickr30k`
  lmms-lab/flickr30k data/test-0000{0..8}-of-00009.parquet (image, caption list).
- bench: `humaneval` openai/openai_humaneval openai_humaneval/test-00000-of-00001.parquet;
  `mbpp` google-research-datasets/mbpp sanitized/test-00000-of-00001.parquet; `mmlu` cais/mmlu
  all/test-00000-of-00001.parquet; `arc-easy`/`arc-challenge` allenai/ai2_arc ARC-Easy|ARC-Challenge/
  test-00000-of-00001.parquet; `hellaswag` Rowan/hellaswag data/validation-00000-of-00001.parquet;
  `gsm8k` openai/gsm8k main/test-00000-of-00001.parquet; `belebele-uz` facebook/belebele
  data/uzn_Latn.jsonl.

`download.py`: `download(url, dest_dir, token=os.environ.get("HF_TOKEN")) -> Path` (cache, .part,
resume-safe, progress), `iter_rows(path, fmt, columns=None, batch_size=256) -> Iterator[dict]`
(pyarrow for parquet; json/jsonl/json.gz streaming where possible), `resolve_hf_shards(repo, dir)`.

`cleaning.py` (from proven code): `norm_apostrophes`, `norm_text`, `clean_doc` (URLs, page numbers,
OCR line joins, hyphenation), `good_doc(text, min_chars, lang)` (Cyrillic ratio filter for uz, alpha
ratio), `clean_code(text)` (strip license headers >20 lines, drop files with long lines/minified),
`dedup_exact(iter)` (hash set).

`adapters.py`: `to_text(source_name, row) -> str|None`, `to_conversation(source_name, row) ->
list[message]|None` (messages per §5). Conversation = list[dict] messages.

`synthetic.py` (deterministic with seed): `identity_conversations(name, n)`, `greeting_conversations(n)`,
`knowledge_conversations(n)` (accurate Uzbek/English facts — O'zbekiston poytaxti Toshkent; mustaqillik
1991-yil 31-avgust e'lon, 1-sentabr nishonlanadi; so'm; 12 viloyat + Qoraqalpog'iston + Toshkent shahri;
Amudaryo/Sirdaryo; Navoiy 1441–1501 Xamsa; Amir Temur 1336–1405; Ulug'bek 1394–1449 rasadxona;
Ibn Sino 980–1037 Tib qonunlari; al-Xorazmiy algebra/algoritm; Bobur 1483–1530 Boburnoma; Registon;
Yupiter; Tinch okeani; 365 kun; suv 100°C; science basics; ≥40 facts, uz+en),
`arithmetic_conversations(n)` (+,-,×,÷ exact, some with tool calls), `tool_call_conversations(n)`
(calculator, current_time, unit_convert, weather (fake data clearly marked), search_wikipedia stub —
full loop: user → assistant tool_call → tool result → assistant answer), `code_conversations(n)`
(small verified Python tasks with solutions: reverse string, fizzbuzz, factorial, is_prime, etc. —
uz & en prompts), `think_conversations(n)` (arithmetic word problems with `<|think|>…<|/think|>`
reasoning then answer). All return list of conversations.

`pretrain.py`: `build_pretrain_tokens(cfg, tokenizer, out_dir) -> (train_bin, val_bin, stats)` —
streams sources by weight (round-robin by weight within char budgets × data_scale), cleans, filters,
tokenizes with `encode_batch`, appends EOS per doc, writes uint16 (V≤65535) else uint32 `.bin` +
`.json` meta; 0.5% docs → val. Offline mode: uses bundled assets repeated + synthetic text.
`class PretrainLoader(path, seq_len, batch_size, device, seed, rank=0, world=1)`: `.next(seq_len=None,
batch_size=None) -> (x, y)` LongTensors (random windows; y = x shifted); loads memmap; for small files
(< 2 GB) moves to device as int32 tensor.

`sft.py` (numpy): `encode_conversations(convs, tokenizer, n_image_tokens=0)` → list[(ids, mask)];
`pack(examples, seq_len) -> (ids int32 [N, seq_len+1], mask uint8 [N, seq_len+1])` greedy packing
with PAD; `class SFTLoader(ids, mask, batch_size, device, seed)`: `.next() -> (x, y)` with
`y = ids[:,1:]` masked to -100 where `mask[:,1:]==0`; `split_val(ids, mask, frac=0.02)`.

`vision.py` (data): `build_image_dataset(cfg, tokenizer, encoder, out_dir)` → cached
features (.npy fp16) + conversations (captions; optional NLLB translation en→uz via transformers
`facebook/nllb-200-distilled-600M` when `language` includes uz) + `ImageSFTLoader` producing
`(x, y, image_features)`.

---------------------------------------------------------------------------------------------------
## 7. Training (moonlight/train)

`optim.py`: `newton_schulz(G, steps=5)` batched over leading dims; `class Muon(Optimizer)`;
`build_optimizers(model, train_cfg, lr_scale=1.0, trainable=None) -> list[Optimizer]`:
Muon for params with ndim ≥ 2 except embedding / lm_head / router / vision_proj.norm (3D expert
weights: orthogonalize each expert matrix — treat dim0 as batch); AdamW for the rest
(embed/lm_head at `embed_lr`, others at `adam_lr`, no weight decay on norms/scalars). Each param group
stores `base_lr`. Muon lr scaled by `max(1, rows/cols)**0.5`.
`schedule.py`: `lr_multiplier(step, progress, cfg) -> float` for wsd|cosine|constant (warmup by step,
decay by progress ∈ [0,1] where progress = max(step/max_steps, tokens/token_budget, time/time_budget)).
`distributed.py`: `init_distributed() -> DistInfo(rank, local_rank, world, device, is_main)` from
torchrun env (no-op single process); `wrap_model(model, strategy, ...)` DDP or FSDP (prefer FSDP2
`torch.distributed.fsdp.fully_shard` per Block when available, else FSDP1 with transformer auto-wrap,
mixed precision bf16); `all_reduce_mean(t)`; `barrier()`.
`checkpoint.py`: `save_checkpoint(path, model, optimizers, state: dict, ema=None)` → safetensors with
`model/…`, `ema/…`, `opt{i}/{param_name}/{key}` tensors and metadata `{"state": json, "config": yaml}`;
atomic write; `load_checkpoint(path, model, optimizers=None, ema=None, map_location) -> state`;
`export_model(path, model, config, tokenizer)` → final single-file safetensors (fp32 or bf16 weights)
with metadata `config` (yaml string), `tokenizer` (json string), `format: moonlight-v1`.
`trainer.py`:
```python
class Trainer:
    def __init__(self, model, config, tokenizer, data=None, dist=None, callbacks=None)
    def fit(self) -> dict          # runs phases; resumes from output_dir/checkpoint.safetensors
    def run_phase(self, phase: dict) -> dict
    def evaluate(self, phase_name) -> float
```
Features: precision auto (bf16 if supported, else fp16 + GradScaler, cpu fp32), grad accumulation
(`accum = batch_tokens // (batch_size*seq_len*world)` ≥1), grad clip, Muon momentum warmup
0.85→0.95 over 300 steps (pretrain), MoE router bias update after each step, EMA (foreach lerp),
torch.compile with safe fallback (warm up all curriculum shapes before the timer), curriculum seq_len
(multiples of 64, constant tokens/step), budgets (tokens / minutes / steps / epochs), periodic eval +
sample chat, checkpoint every N minutes + phase end, JSONL metrics log at output_dir/metrics.jsonl,
optional wandb if installed and env WANDB_PROJECT set, phase `trainable` (freeze others), anneal mix
(SFT batches during decay), SFT replay, image batches. Phase skipped automatically if its data is
unavailable (e.g. align without multimodal). Data preparation (tokenizer training, bins, sft packing,
image features) happens in `Trainer.prepare_data()` (rank 0 builds, others wait at barrier).

---------------------------------------------------------------------------------------------------
## 8. LLM facade, chat, tools (moonlight/llm.py, chat.py, tools.py, cli.py)

`LLM(config, tokenizer=None, device=None, dtype=None, build=True)`; `.model .tokenizer .config`;
`.train(**overrides)`, `.chat(message, history=None, system=None, tools=None, images=None, **gen)
-> str`, `.generate(prompt, **gen) -> str`, `.benchmark(name="default"|"cc-00x", **kw) -> Report`,
`.save(path)`, `@classmethod from_pretrained(path, device=None)`, `.num_params()`, `.encode_images(imgs)`.
If tokenizer missing at chat time → raise RuntimeError("train first or load a checkpoint").
Offline/random models must still run chat (garbage text, no crash) for tests.

`tools.py`: `@dataclass Tool(name, description, parameters: dict(JSON schema), fn: Callable)`;
`tool(name=None, description=None)` decorator inferring schema from type hints/docstring;
`ToolRegistry` (`register`, `schemas()`, `call(name, arguments) -> str` (json-serialized result or
error)); built-ins: `calculator(expression)` (SAFE ast evaluator: numbers, + - * / // % **, parentheses,
unary ±, math funcs sqrt, sin, cos, log, exp, pi, e; limits on exponent size), `current_time(timezone="UTC")`,
`unit_convert(value, from_unit, to_unit)` (length/mass/temperature), `python_exec(code)` (DISABLED unless
`allow_code_exec=True`; subprocess, timeout 5s, no network not guaranteed — documented);
`parse_tool_call(text_or_ids_decoded) -> dict|None` (tolerant JSON extraction).
`chat.py`: `ChatSession(llm, system=None, tools=None, max_tool_rounds=4, **gen)`; `.send(message,
images=None) -> str`, `.history`, `.reset()`. Tool loop: generate with stop at `<|end|>`; if output
starts with `<|tool_call|>` (detect via token id) → parse → registry.call → append tool message →
generate again.
`cli.py` (argparse): `moonlight list [--tier T]`, `moonlight info NAME`, `moonlight params NAME`,
`moonlight train NAME [--set k=v ...] [--offline]`, `moonlight chat PATH_OR_NAME`, `moonlight bench
PATH_OR_NAME --suite cc-001 [--limit N] [--out report.md]`, `moonlight gen-configs`.

---------------------------------------------------------------------------------------------------
## 9. Benchmarks (moonlight/benchmarks)

Tiers (by NOMINAL total params):
| suite | range | purpose |
|---|---|---|
| cc-001 | 1M–2M | weakest: chat format, greeting sanity, identity, char-level perplexity |
| cc-002 | 5M–20M | + arithmetic-lite, knowledge-uz, tool-format |
| cc-003 | 50M–100M | + code-lite (bundled problems, execution), belebele-uz (loglik) |
| cc-004 | 150M–300M | + arc-easy, hellaswag(sub), humaneval(sub), tool-calls end-to-end |
| cc-005 | 1B–3B | + mmlu(sub), arc-challenge, gsm8k(sub), mbpp, humaneval |
| cc-006 | 7B–13B | full mmlu/arc/hellaswag/gsm8k/humaneval/mbpp + long-context needle |
| cc-007 | 16B–70B | cc-006 + vision captioning (if multimodal) + agent multi-tool |
| cc-008 | 100B–235B | cc-007 stricter thresholds |
| cc-009 | 405B–671B | cc-008 stricter |
| cc-010 | 1T | strictest |
Every suite includes `chat_format`, `chat_sanity`, `identity` (all models must be usable as chat
models) and `speed` (tokens/s, peak memory). Suite YAML:
```yaml
name: cc-001
description: ...
param_range: [1M, 2M]
tasks:
  - {task: chat_format, limit: 20, threshold: 0.8}
  - {task: chat_sanity, limit: 20, threshold: 0.5}
  ...
```
`suites.py`: `type(name) -> Suite`, `list_suites()`, `suite_for_params(n) -> name`,
`class Suite(name, description, param_range, tasks)`; `.run(llm, limit=None, out=None, offline=False)
-> Report`. `Report(suite, results: list[TaskResult], passed: bool)` with `.to_markdown()`,
`.to_dict()`, `.save(path)` (json or md by extension). `TaskResult(task, score, n, threshold, passed,
details, seconds)`.
`tasks.py`: registry `TASKS = {name: fn(llm, limit, offline) -> TaskResult-like dict}` implementing:
chat_format, chat_sanity, identity, perplexity (bundled sample_uz/en text), arithmetic, knowledge_uz,
tool_format, tool_e2e, code_lite (≥15 bundled problems with asserts, exec via execution.py),
belebele_uz, arc_easy, arc_challenge, hellaswag, mmlu, gsm8k, humaneval, mbpp, needle (long context),
vision_caption (skips with score None if not multimodal), speed. Loglikelihood MC scoring via a shared
`loglikelihood(llm, context, continuation)` helper (sum log-probs of continuation tokens, length-
normalized variant). Tasks needing downloads are skipped (score None, passed None) when offline or
download fails — never crash the suite.
`execution.py`: `run_python(code, timeout=5.0) -> (ok, stdout, stderr)` via subprocess with
resource limits where available.

---------------------------------------------------------------------------------------------------
## 10. The 100 configs (scripts/generate_configs.py → configs/{tier}/{name}.yaml)

Exactly these names (tier: test 1–25, mini 26–43, si 44–100):
```
test: ml-1m-debug ml-1m-base ml-1m-chat ml-2m-base ml-2m-chat ml-2m-code ml-5m-base ml-5m-chat
ml-5m-code ml-10m-base ml-10m-chat ml-10m-code ml-20m-base ml-20m-chat ml-20m-code ml-20m-vl
ml-20m-colab-chat ml-20m-tools ml-50m-base ml-50m-chat ml-50m-code ml-50m-vl ml-50m-kaggle-chat
ml-50m-tools ml-50m-think
mini: ml-100m-base ml-100m-chat ml-100m-code ml-150m-chat ml-250m-base ml-250m-chat ml-250m-code
ml-250m-vl ml-250m-tools ml-250m-kaggle-chat ml-300m-base ml-300m-chat ml-300m-code ml-300m-vl
ml-300m-long ml-300m-tools ml-300m-think ml-300m-uz
si: ml-1b-base ml-1b-chat ml-1b-code ml-1b-agent ml-1b-think ml-2b-chat ml-2b-code ml-3b-base
ml-3b-chat ml-3b-code ml-3b-agent ml-7b-base ml-7b-chat ml-7b-code ml-7b-agent ml-7b-long ml-7b-think
ml-13b-base ml-13b-chat ml-13b-code ml-13b-agent ml-30b-base ml-30b-chat ml-30b-code ml-30b-vl
ml-30b-vl-chat ml-70b-base ml-70b-chat ml-70b-code ml-70b-vl ml-70b-agent ml-70b-think ml-175b-base
ml-175b-chat ml-175b-vl ml-405b-base ml-405b-chat ml-405b-vl ml-1t-base ml-1t-chat ml-1t-vl
ml-16b-a2b-chat ml-16b-a2b-code ml-30b-a3b-base ml-30b-a3b-chat ml-30b-a3b-vl ml-100b-a10b-chat
ml-100b-a10b-vl ml-235b-a22b-chat ml-235b-a22b-code ml-235b-a22b-vl ml-671b-a37b-chat
ml-671b-a37b-vl ml-1t-a32b-base ml-1t-a32b-chat ml-1t-a32b-code ml-1t-a32b-vl
```
Base shapes (generator may adjust ffn_hidden/n_layers so estimator lands within ±10% of nominal;
MoE nominal = TOTAL params, `-aNb` = active, must land within ±20% of the active label):
| size | V | d | L | H | hd | ret_ratio | Hkv | ffn | ctx | window/global_every |
|---|---|---|---|---|---|---|---|---|---|---|
| 1M | 2048 | 96 | 6 | 4 | 24 | 0.5 | 1 | 256 | 256 | –
| 2M | 2048 | 128 | 6 | 4 | 32 | 0.5 | 1 | 352 | 256 | –
| 5M | 4096 | 192 | 8 | 6 | 32 | 0.5 | 1 | 512 | 512 | –
| 10M | 8192 | 256 | 8 | 8 | 32 | 0.5 | 2 | 704 | 512 | –
| 20M | 8192 | 384 | 10 | 6 | 64 | 0.5 | 1 | 1024 | 1024 | –
| 50M | 16384 | 512 | 12 | 8 | 64 | 0.5 | 2 | 1408 | 1024 | –
| 100M | 32768 | 768 | 12 | 12 | 64 | 0.5 | 2 | 2048 | 2048 | –
| 150M | 32768 | 768 | 18 | 12 | 64 | 0.5 | 2 | 2048 | 2048 | –
| 250M | 32768 | 1024 | 18 | 16 | 64 | 0.5 | 4 | 2816 | 2048 | –
| 300M | 32768 | 1024 | 22 | 16 | 64 | 0.5 | 4 | 2816 | 2048 | –
| 1B | 65536 | 2048 | 16 | 32 | 64 | 0.375 | 4 | 5632 | 4096 | 4096/4
| 2B | 65536 | 2304 | 24 | 36 | 64 | 0.375 | 4 | 6144 | 4096 | 4096/4
| 3B | 65536 | 3072 | 28 | 24 | 128 | 0.25 | 4 | 8192 | 8192 | 4096/4
| 7B | 131072 | 4096 | 32 | 32 | 128 | 0.25 | 8 | 14336 | 8192 | 4096/4
| 13B | 131072 | 5120 | 40 | 40 | 128 | 0.25 | 8 | 13824 | 8192 | 4096/4
| 30B | 131072 | 6656 | 60 | 52 | 128 | 0.25 | 13 | 17920 | 8192 | 4096/4
| 70B | 131072 | 8192 | 80 | 64 | 128 | 0.25 | 8 | 28672 | 8192 | 4096/4
| 175B | 131072 | 12288 | 96 | 96 | 128 | 0.25 | 8 | 49152 | 8192 | 4096/4
| 405B | 131072 | 16384 | 126 | 128 | 128 | 0.25 | 16 | 53248 | 8192 | 4096/4
| 1T | 131072 | 24576 | 128 | 192 | 128 | 0.25 | 16 | 66560 | 8192 | 4096/4
MoE: 16b-a2b d2048 L28 H16 hd128 E64 k6 shared2 he1408 first_dense1 (dense ffn 10944);
30b-a3b d2048 L48 H32 hd128 Hkv4 E128 k8 he768; 100b-a10b d4096 L48 H32 hd128 E64 k6 shared1 he2560
first_dense1 (ffn 14336); 235b-a22b d4096 L94 H64 hd128 Hkv4 E128 k8 he1536; 671b-a37b d7168 L61 H128
hd128 Hkv8 E256 k8 shared1 he2048 first_dense3 (ffn 18432); 1t-a32b d7168 L61 H64 hd128 Hkv8 E384 k8
shared1 he2048 first_dense1 (ffn 18432). SI MoE retention_ratio 0.25, window 4096/global_every 4.
(H and hd above are TOTAL query heads incl. retention; ensure Ha % Hkv == 0 — adjust Hkv if needed.)
Variants: base (pretrain only, phases pretrain[+anneal]); chat (pretrain + sft, chat+code+tool data);
code (code-heavy pretrain 40% + code SFT); vl / vl-chat (multimodal.enabled true, align + sft with
image_mix); tools / agent (tool-call heavy SFT + glaive/apigen); think (reasoning `<|think|>` data);
long (max_seq_len 8192 for 300m, 131072 for 7b with rope_theta 1e6/5e6 and long-context phase);
uz (Uzbek-only data); colab / kaggle (time budgets for T4 / 2×T4, compile on, fp16); debug (CPU, offline
true, tiny budgets). EVERY chat-capable config (all except *-base) trains on conversation + code.
Hardware presets & distributed: test → colab-t4 (none); mini → kaggle-2xt4 / a100-1 (ddp);
1B–3B a100-8 (fsdp); 7B–13B h100-8/h100-16 (fsdp); 30B–70B h100-64..256 (fsdp, tp_size 8 documented);
175B+ / MoE h100-512..2048 (fsdp, tp 8, ep_size up to 64 documented). Token budgets: test by minutes;
mini ~20–40 tokens/param; SI 20+ tokens/param (Chinchilla) with values written explicitly.
Benchmark mapping per §9 by nominal size.

---------------------------------------------------------------------------------------------------
## 11. Tests (tests/, pytest) & CI

Torch-dependent tests use `torch = pytest.importorskip("torch")`. Required tests:
`test_config.py` (load all 100; exactly 100; names = stems; validate; estimator within tolerance;
overrides; yaml roundtrip; bnchmrks.type works), `test_params_meta.py` (meta-device instantiation of
EVERY config — exact equality with estimator total; skip vision encoder), `test_model.py` (tiny dense +
MoE + multimodal forward/backward finite; MTP; retention parallel==chunkwise==recurrent; cached decode
== full forward for global, sliding-window, retention, token-shift, MoE; best-of-N generate runs),
`test_tokenizer.py`, `test_data.py` (cleaning, adapters on sample rows, synthetic generators,
packing/masks, loaders), `test_train_smoke.py` (ml-1m-debug offline: 30 steps loss decreases; checkpoint
save→resume; export→from_pretrained→chat), `test_tools.py` (calculator safety, parsing, decorator),
`test_benchmarks.py` (10 suites load; offline run of cc-001 on random tiny model produces report;
suite_for_params mapping), `test_api.py` (lazy import, ml.config, LLM, cli list/info).
CI: `.github/workflows/ci.yml` on push/PR: ubuntu-latest, python 3.11, `pip install torch --index-url
https://download.pytorch.org/whl/cpu`, `pip install -e .[dev]`, `pytest -q -x --timeout=900` (add
pytest-timeout to dev deps), plus `python scripts/generate_configs.py --check` (fails if regenerated
configs differ from committed).

## 12. Honesty rules for docs
Never claim measured superiority over Transformers. State design goals (hybrid retention for O(1)
decode state on retention heads, smaller KV cache, Muon optimizer efficiency, MTP) and that the
benchmark harness exists to MEASURE. Configs ≥1B are reference recipes requiring large clusters;
FSDP is implemented, tensor/expert parallel fields are documented placeholders (not implemented).

---------------------------------------------------------------------------------------------------
## 13. v1.1 additions — alignment (DPO/GRPO), planning, agentic project building, detail-following
(Implement in THIS build by the owners named below. Phase dicts and data.synthetic are free-form
mappings, so no config-schema change is needed.)

### 13.1 New data sources (D, moonlight/data/sources.py + adapters.py)
- `ultrafeedback` (kind `pref`): HuggingFaceH4/ultrafeedback_binarized
  `data/train_prefs-00000-of-00001.parquet` (columns: prompt, chosen [messages], rejected [messages],
  score_chosen, score_rejected) → adapter `to_preference(row) -> (chosen_msgs, rejected_msgs)|None`.
- `smoltalk-constraints` (sft): HuggingFaceTB/smoltalk `data/smol-constraints/train-00000-of-00001.parquet` (messages).
- `smoltalk-math` (sft): `data/numina-cot-100k/train-00000-of-00001.parquet` (messages).
- `smoltalk-magpie` (sft): `data/smol-magpie-ultra/train-0000{0..5}-of-00006.parquet` (messages).
- Correct smoltalk shard names: `data/everyday-conversations/train-00000-of-00001.parquet`,
  `data/self-oss-instruct/train-00000-of-00001.parquet`, `data/apigen-80k/train-00000-of-00001.parquet`,
  `data/all/train-0000{0..8}-of-00009.parquet`.
- `openthoughts` (sft, reasoning): open-thoughts/OpenThoughts-114k `data/train-0000{0..5}-of-00006.parquet`
  (columns system, conversations [{from: user|assistant, value}]); convert
  `<|begin_of_thought|>…<|end_of_thought|>` → `<|think|>…<|/think|>` and strip
  `<|begin_of_solution|>`/`<|end_of_solution|>` markers.
- `gsm8k-train` (rl prompts): openai/gsm8k `main/train-00000-of-00001.parquet` (question, answer "#### N").
- `ifeval` (bench): google/IFEval `ifeval_input_data.jsonl` (key, prompt, instruction_id_list, kwargs).

### 13.2 Verifiers (D owns NEW file moonlight/data/verifiers.py, torch-free)
Deterministic checkers shared by synthetic data, GRPO rewards and benchmarks:
`Constraint = dict(type=..., **params)`; `check(response: str, constraint) -> bool`;
`check_all(response, constraints) -> (n_passed, n_total, details)`; supported types:
`word_count_max/min`, `sentence_count_max/min`, `bullet_count` (exact, lines starting with "- " or "* "
or "N. "), `paragraph_count`, `include_keywords` (all, case-insensitive), `exclude_keywords`,
`keyword_frequency` (word, at_least), `json_format` (whole response parses as JSON, fences allowed),
`lowercase`, `uppercase`, `no_commas`, `ends_with` (phrase), `starts_with`, `title_brackets` (<<title>>),
`sections` (n sections with "SECTION X"), `language_uz` (≥ 85% Latin letters, Uzbek markers o'/g'
optional), `quotation_wrapped`, `max_chars`. Plus `ifeval_to_constraints(instruction_id_list, kwargs)`
mapping the supported IFEval ids (`length_constraints:number_words`, `…:number_sentences`,
`…:number_paragraphs`, `keywords:existence`, `keywords:frequency`, `keywords:forbidden_words`,
`detectable_format:json_format`, `…:number_bullet_lists`, `…:title`, `…:multiple_sections`,
`change_case:english_lowercase`, `change_case:english_capital`, `punctuation:no_comma`,
`startend:end_checker`, `startend:quotation`) and returning None for unsupported ids.
Reward helpers: `reward_exact_number(response, target)`, `reward_constraints(response, constraints)`
(fraction satisfied, 1.0 only if all), `reward_code(response, tests, entry_point)` (uses
moonlight.benchmarks.execution.run_python lazily), `extract_final_answer(text)` (after `<|/think|>` if
present; last number for math).

### 13.3 New synthetic generators (D, moonlight/data/synthetic.py)
- `plan_conversations(n)`: realistic planning/project prompts (uz + en: build a CLI app, plan a study
  schedule, design a REST API, organize an event, migrate a database, write a test plan…). Responses
  follow the Moonlight PLAN FORMAT: `<|think|>` short analysis: goal, constraints, unknowns, risks
  `<|/think|>` then **Reja / Plan** numbered steps (each step: action + expected output + check),
  then **Tekshiruv / Verification** checklist mapping every user requirement to the step that
  satisfies it. Template-composed from rich component pools so outputs are coherent and varied.
- `constraint_conversations(n)`: prompts carrying 1–4 verifiable constraints (from verifiers types)
  with programmatically constructed responses that provably satisfy all of them (assert with
  verifiers.check_all before emitting).
- `detail_conversations(n)`: prompts hiding small requirements (e.g. "use the variable name `total`",
  "end with the word 'Tayyor'", "mention the date 2024-05-01") and correct responses meeting each one.
- `project_conversations(n)`: agentic trajectories with workspace tools (`write_file`, `read_file`,
  `list_dir`, `run_tests`) building small verified Python projects (calculator module, string utils,
  todo list class, temperature converter, word counter…): plan → write module → write tests →
  run_tests result (realistic pytest-like output) → fix if a deliberately injected bug failed → final
  summary. All code in trajectories must actually pass its tests (verify by executing in scratch).
- `grpo_prompt_set(kinds, n, seed) -> list[dict(prompt_messages, reward_type, reward_data)]` with
  kinds: arithmetic, constraints, detail, code_lite (problems from benchmarks? NO — define a separate
  TRAIN problem pool in synthetic.py; benchmark problems must stay held-out), gsm8k (from source when
  online).
- `preference_pairs_synthetic(n)`: chosen = correct/complete/format-following answer, rejected = a
  realistic flawed variant (missed constraint, wrong number, missing plan step, language switch).

### 13.4 Preference & RL data (D, moonlight/data/sft.py)
`encode_preference(chosen_msgs, rejected_msgs, tokenizer, seq_len) -> dict(chosen_ids, chosen_mask,
rejected_ids, rejected_mask)` (truncate from the left of the prompt if needed, keep response);
`PreferenceLoader(pairs, batch_size, device, seed).next() -> (c_x, c_y, r_x, r_y)` right-padded with
PAD, targets -100 where mask 0.

### 13.5 New training phases (E, moonlight/train/trainer.py + new moonlight/train/align.py)
- `long`: like pretrain but with `seq_len` (phase key) > train.seq_len, documents filtered/concatenated
  to long sequences; lr_scale default 0.1.
- `dpo`: keys `sources` (list of {source, max_examples}), `synthetic_pairs` (int), `beta` (0.1),
  `minutes|steps|epochs`, `lr_scale` (0.1). Reference model = frozen deepcopy at phase start (eval
  mode, no grad; on CPU-offload is not required). Loss = −logσ(β[(logπ(c)−logπ_ref(c)) −
  (logπ(r)−logπ_ref(r))]) with sequence log-probs summed over response tokens; log rewards
  accuracy/margins. Optional `loss: ipo|dpo` (default dpo).
- `grpo`: keys `tasks` (list of kinds for grpo_prompt_set), `group_size` (8), `prompts_per_step` (4),
  `max_new_tokens` (256), `temperature` (1.0), `kl_coef` (0.02), `clip` (0.2), `minutes|steps`,
  `lr_scale` (0.05). Per step: sample G completions per prompt with moonlight.model.generation.generate
  (no grad, eval mode), rewards via verifiers, advantages A = (r − mean_group)/(std_group+1e-4), skip
  groups with zero variance, then recompute token log-probs with grad (train mode, same masks) and
  loss = −mean(min(ratio·A, clip(ratio)·A)) + kl_coef·KL_k3(π‖π_ref) over completion tokens
  (old logp = detached logp from the sampling pass recomputed in train mode → ratio starts at 1).
  Log mean reward, solve rate.
- align.py contains `dpo_loss`, `sequence_logprobs(model, x, y)`, `grpo_step(...)` helpers; trainer
  dispatches phases by name; unknown phase name → ValueError.

### 13.6 Planning & agentic project building (F, moonlight/tools.py + llm.py + chat.py)
- `Workspace(root, allow_exec=True, timeout=20)`: sandboxed to root (resolve paths, reject `..`
  escapes and absolute paths outside), tools: `read_file(path)`, `write_file(path, content)`,
  `list_dir(path=".")`, `run_tests(target=".")` (runs `python -m pytest -q` if pytest importable else
  unittest discovery, in a subprocess with cwd=root and timeout; returns trimmed output + passed bool),
  `run_python(code)`; `workspace_tools(ws) -> list[Tool]`.
- `LLM.plan(task, **gen) -> str` (forces PLAN FORMAT system prompt) and
  `LLM.build_project(task, workdir, max_steps=24, **gen) -> ProjectResult(workdir, files, tests_passed,
  steps, transcript)`: ChatSession with workspace tools and a builder system prompt (plan first, write
  code + tests, run tests, fix until green, final summary listing every requirement and where it is
  satisfied). Stops early when run_tests passes and the model gives a final answer.
- PLAN/BUILDER system prompts are constants in chat.py (Uzbek + English sentence).

### 13.7 New benchmark tasks (G)
`ifeval` (download; only supported ids via verifiers.ifeval_to_constraints; prompt-level strict
accuracy + instruction-level accuracy in details), `ifeval_lite` (bundled 30 prompts with constraints,
held-out from training generators by using a different seed and different templates), `detail_check`
(bundled 20 prompts; score = mean fraction of hidden requirements satisfied), `plan_bench` (bundled
20 verifiable planning problems: dependency ordering (topological order validity), scheduling with
time-window constraints, resource allocation with sums — answer parsed from a final "Javob:/Answer:"
line), `project_build` (bundled 4 mini-project specs with hidden pytest tests; runs
`llm.build_project` in a temp dir, then runs hidden tests; score = fraction passed; skipped on
cc-001..cc-003 tiers). Add ifeval_lite + detail_check from cc-002, plan_bench from cc-003, ifeval from
cc-004, project_build from cc-005 (thresholds scaled by tier).

### 13.8 Configs (H)
- chat/code/tools/agent/think/vl-chat variants of size ≥ 1B: phases … → sft → dpo
  (sources: ultrafeedback + synthetic_pairs) → grpo (tasks per variant: chat: [constraints, detail,
  arithmetic]; code: [code_lite, constraints]; agent: [constraints, detail, code_lite]; think:
  [gsm8k, arithmetic, code_lite]). Mini (150M–300M) chat variants: sft → dpo. Test tier: 50M chat/
  tools/think add a short dpo (minutes 5); others unchanged.
- SFT sources additionally include smoltalk-constraints (all chat-capable), smoltalk-math (think/
  si-chat), openthoughts (think variants and SI chat/code), smoltalk-magpie (SI), and data.synthetic
  gets `plan`, `constraints`, `detail`, `projects` counts (agent/tools variants heavier).
- *-long variants add a `long` phase before sft.

## 14. v2 model upgrade (NOT in this build — planned next, do not implement now)
YaRN RoPE scaling, attention sinks, int8 KV cache, chunked prefill, optional MLA attention,
MTP self-speculative decoding, 2M (2,097,152) token context/output support.
