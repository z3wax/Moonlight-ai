# Training Moonlight

```python
import moonlight as ml
llm = ml.LLM(ml.config("ml-20m-colab-chat"))
result = llm.train()          # runs cfg.train.phases, resumes automatically
llm.save("out/ml-20m-colab-chat.safetensors")
# or the one-liner:  llm = ml.train("ml-20m-colab-chat")
```

CLI: `moonlight train ml-20m-colab-chat [--set train.batch_size=16 ...] [--offline]`.
Multi-GPU: `torchrun --nproc_per_node 8 -m moonlight train ml-1b-chat`.

Everything below mirrors SPEC §6–§7 and §13 (`moonlight/data`, `moonlight/train`).

---------------------------------------------------------------------------------------------------

## 1. Data preparation (`Trainer.prepare_data`, rank 0 builds, other ranks wait)

| product | source | notes |
|---|---|---|
| tokenizer | `tokenizer.path`, `{output_dir}/tokenizer.json`, or newly trained | BPE trained on up to `tokenizer.train_chars` characters of the pretrain mix + synthetic chats |
| pretrain bins | `data.pretrain` (`{source, weight, max_chars}`) | streamed, cleaned (`cleaning.py`), filtered (Cyrillic ratio for uz, alpha ratio, code minification), deduplicated, tokenized; EOS after each doc; `uint16` (V ≤ 65535) memmaps; 0.5% of documents go to validation |
| SFT packs | `data.sft` (`{source, max_examples}`) + `data.synthetic` | chat template → `(ids, loss_mask)` → greedy packing into `seq_len + 1` rows |
| image features | `data.vision` (vl configs) | frozen SigLIP features cached as fp16 `.npy` plus caption conversations (optional NLLB en→uz) |
| preference pairs | dpo phase `sources` + `synthetic_pairs` | `ultrafeedback` + `preference_pairs_synthetic` |
| RL prompts | grpo phase `tasks` | `grpo_prompt_set(kinds, n, seed)` with verifier rewards |

Downloads use the Hugging Face `resolve/main` URLs from `moonlight/data/sources.py`. They are cached
in `data.cache_dir` and are resume-safe; set `HF_TOKEN` for gated or rate-limited repos.
`data.offline: true` (or `--offline`) uses only the bundled Uzbek, English and code samples plus
synthetic data. Tests, CI and `ml-1m-debug` run this way.

### Synthetic data (`data.synthetic`)

| key | content |
|---|---|
| `identity`, `greetings`, `knowledge` | persona, small talk, ≥ 40 verified uz/en facts |
| `arithmetic`, `tool_calls` | exact arithmetic (some via calculator calls); full tool loops (calculator, time, units, mock weather, wiki stub) |
| `code`, `think` | verified Python tasks; word problems with `<|think|>…<|/think|>` reasoning |
| `plan` | PLAN FORMAT answers: `<|think|>` analysis → **Reja / Plan** steps (action, output, check) → **Tekshiruv / Verification** |
| `constraints`, `detail` | prompts with 1–4 verifiable constraints or hidden details; every answer is checked with `verifiers.check_all` before it is emitted |
| `projects` | agentic trajectories with `write_file / read_file / list_dir / run_tests`; every project's code passes its tests |

Every chat-capable config trains on conversation **and** code.

## 2. Phases

`train.phases` is an ordered list of free-form dicts. Each phase is dispatched by `type` (default:
its `name`; `sft-code` resolves to `sft`). An unknown phase raises `ValueError`. A phase whose data
is unavailable (for example `align` without multimodal) is skipped and logged.

| phase | what it does | key options (defaults) |
|---|---|---|
| `pretrain` | next-token LM on the bins; seq-len curriculum; Muon momentum warmup 0.85→0.95 over 300 steps | `tokens`, `minutes`, `steps`, `lr_scale`, `sft_mix`, `anneal_sft_mix` |
| `anneal` | LM with LR decaying over the whole phase and SFT mixed in | `anneal_sft_mix` |
| `long` | LM at a long `seq_len` (documents concatenated), same tokens per step | `seq_len`, `lr_scale` (0.1) |
| `align` | vision projector only (`trainable: vision_proj`) on image-caption batches | `minutes` |
| `sft` | packed conversations + `replay` LM batches + `image_mix` image batches | `epochs`, `replay`, `image_mix`, `lr_scale` |
| `dpo` | preference optimisation against a frozen reference copy | `sources`, `synthetic_pairs`, `beta` (0.1), `loss: dpo/ipo`, `lr_scale` (0.1) |
| `grpo` | verifier-rewarded RL | `tasks`, `group_size` (8), `prompts_per_step` (4), `max_new_tokens` (256), `temperature` (1.0), `kl_coef` (0.02), `clip` (0.2), `lr_scale` (0.05) |

**Budgets**: any subset of `tokens`, `minutes`, `steps` and `epochs`. Phase progress is
`max(step/steps, tokens/token_budget, time/time_budget, epoch/epochs)`, and the phase ends at 1.
Test-tier configs are budgeted by minutes on a Colab T4. Mini and SI configs use explicit token budgets
(roughly 20–40 tokens per parameter for mini, and Chinchilla-style ≥ 20 tokens per parameter for SI).

### DPO

The reference model `π_ref` is a frozen deep copy taken at phase start. With summed response
log-probabilities:

```
L_DPO = −log σ( β [ (log π(c) − log π_ref(c)) − (log π(r) − log π_ref(r)) ] )
L_IPO = ( (log π(c) − log π_ref(c)) − (log π(r) − log π_ref(r)) − 1/(2β) )²
```

Reward accuracy and margins are logged. When encoding pairs, the prompt is truncated from the left
and the response is kept.

### GRPO

For each step, the trainer samples `G` completions for each of `P` prompts with `generate()` (no grad,
eval mode) and scores them with deterministic verifiers (exact number, constraints, unit tests).
Group-normalised advantages are `A = (r − mean_g) / (std_g + 1e-4)`, and groups with zero variance
are skipped. Token log-probs are then recomputed with gradients:

```
ratio = exp(logπ − logπ_old)
L = −mean( min(ratio·A, clip(ratio, 1−ε, 1+ε)·A) ) + kl_coef · KL_k3(π‖π_ref)
KL_k3 = exp(ref − logπ) − (ref − logπ) − 1
```

Mean reward and solve rate are logged. Benchmark problems are never used as GRPO prompts:
`code_lite` draws from a separate training pool.

## 3. Optimisation

* **Muon** for every matrix parameter (ndim ≥ 2) except embeddings, `lm_head`, MoE routers and the
  vision projector norm. The update is orthogonalised by a 5-step Newton–Schulz iteration (batched; 3-D
  expert weights orthogonalise each expert separately), and the learning rate is scaled by `max(1, rows/cols)^½`.
* **AdamW** for the rest: embeddings and `lm_head` at `embed_lr`, others at `adam_lr`, with no weight decay on
  norms or scalars.
* **Schedules**: `wsd` (warmup, then stable, then linear decay over the last `decay_frac`), `cosine`, or
  `constant`; warmup is counted in steps and decay in phase progress.
* Gradient accumulation is `batch_tokens // (batch_size · seq_len · world)`, with gradient clipping and
  an EMA of the weights (fp32, `foreach` lerp; at phase end the better of EMA and raw weights by
  validation loss is kept).
* Precision `auto` resolves to bf16 autocast when the GPU supports it, otherwise fp16 with a GradScaler; CPU runs in fp32.
  `torch.compile` warms up every curriculum shape before the phase timer starts and falls back to eager if compilation fails.
* MoE router biases are updated after every optimizer step.

## 4. Checkpoints, resume, export

* `{output_dir}/checkpoint.safetensors` is written every `checkpoint_every_min` minutes and at every phase
  end. It holds model, EMA and optimizer tensors, with JSON trainer state and YAML config as metadata,
  and is written atomically.
* `fit()` resumes automatically: finished phases are skipped, and an unfinished phase continues with
  its step, tokens, elapsed time, optimizer and EMA state.
* At the end the model is exported to `{output_dir}/model.safetensors`, containing the weights plus `config`
  (YAML), `tokenizer` (JSON) and `format: moonlight-v1` metadata. Load it with
  `ml.LLM.from_pretrained(path)`.
* Metrics are appended to `{output_dir}/metrics.jsonl`. wandb is used if it is installed and `WANDB_PROJECT` is set.

## 5. Hardware presets (informational `train.hardware`)

| tier | sizes | preset | strategy |
|---|---|---|---|
| test | 1M–50M | `colab-t4` (Kaggle 2×T4 variants) | single GPU |
| mini | 100M–300M | `kaggle-2xt4` / `a100-1` | DDP |
| si | 1B–3B | `a100-8` | FSDP |
| si | 7B–13B | `h100-8` / `h100-16` | FSDP |
| si | 30B–70B | `h100-64…256` | FSDP (+ TP 8 planned) |
| si | 175B+ and MoE | `h100-512…2048` | FSDP (+ TP 8, EP ≤ 64 planned) |

Configs at or above 1B are **reference recipes** that need multi-node GPU clusters. Tensor and
expert parallelism (`tp_size`, `ep_size`) are documented placeholders and are not implemented in this
release. Memory and FLOPs per config: `moonlight params NAME`.

## 6. Tips

* Debug the full pipeline on CPU in about a minute: `moonlight train ml-1m-debug --offline`.
* Override anything: `--set train.phases.0.minutes=20 --set train.batch_size=8`.
* On a T4, use the `*-colab-*` and `*-kaggle-*` configs: they use fp16, compile and time-based budgets.
