# Moonlight benchmarks (`cc-001` … `cc-010`)

The benchmark harness exists to **measure** what a Moonlight model can do. No results are claimed
here; run the suites and publish your own reports.

```python
import moonlight as ml
cnf = ml.config("ml-20m-chat")
report = cnf.bnchmrks.type("cc-002").run(llm)      # or llm.benchmark("cc-002") / llm.benchmark()
print(report.to_markdown()); report.save("reports/ml-20m-chat.md")   # .md or .json
suite = ml.benchmarks.type("cc-001")                # without a config
```

CLI: `moonlight bench PATH_OR_NAME --suite cc-002 [--limit 50] [--out report.md] [--offline]`.

## Tiers

Each model is mapped to one suite by its **nominal total** parameters (MoE models by total, not
active). Suites are cumulative: every tier includes the previous tier's tasks with equal or stricter thresholds.

| suite | nominal params | adds |
|---|---|---|
| cc-001 | 1M–2M | chat_format, chat_sanity, identity, perplexity (char-level), speed |
| cc-002 | 5M–20M | arithmetic (lite), knowledge_uz, tool_format, **ifeval_lite**, **detail_check** |
| cc-003 | 50M–100M | code_lite (≥ 15 bundled problems, executed), belebele_uz (log-likelihood), **plan_bench** |
| cc-004 | 150M–300M | arc_easy, hellaswag (subset), humaneval (subset), tool_e2e, **ifeval** |
| cc-005 | 1B–3B | mmlu (subset), arc_challenge, gsm8k (subset), mbpp, humaneval, **project_build** |
| cc-006 | 7B–13B | full mmlu / arc / hellaswag / gsm8k / humaneval / mbpp + needle (long context) |
| cc-007 | 16B–70B | cc-006 + vision_caption (multimodal models) + multi-tool agent tasks |
| cc-008 | 100B–235B | cc-007 with stricter thresholds |
| cc-009 | 405B–671B | stricter |
| cc-010 | 1T | strictest |

Every suite includes `chat_format`, `chat_sanity` and `identity` (every Moonlight model must be
usable as a chat model) and `speed` (tokens/s, peak memory).

Suite files live in `moonlight/benchmarks/suites/cc-00x.yaml`:

```yaml
name: cc-001
description: ...
param_range: [1M, 2M]
tasks:
  - {task: chat_format, limit: 20, threshold: 0.8}
  - {task: chat_sanity, limit: 20, threshold: 0.5}
```

A suite **passes** when every task that was not skipped passes its threshold (`lower_is_better`
tasks such as `perplexity` pass below it). Skipped tasks are reported with `score: None, passed: None`.

## Tasks

| task | metric |
|---|---|
| `chat_format` | replies follow the chat template (terminated turn, no leaked special tokens, non-empty) |
| `chat_sanity` | greetings and simple questions get non-degenerate, on-topic replies |
| `identity` | the model names itself (Moonlight) |
| `perplexity` | character-level perplexity (bits per byte, lower is better) on the bundled Uzbek / English samples |
| `arithmetic` | exact-match integer arithmetic |
| `knowledge_uz` | Uzbek facts (capital, independence day, currency, poets, scientists …) |
| `tool_format` | a valid `<|tool_call|>{"name", "arguments"}` when a tool is needed |
| `tool_e2e` | full loop: call → tool result → correct final answer |
| `code_lite` | bundled Python problems, run with asserts in a sandboxed subprocess |
| `belebele_uz`, `arc_easy`, `arc_challenge`, `hellaswag`, `mmlu` | multiple choice via `loglikelihood(llm, context, continuation)` (sum and length-normalised) |
| `gsm8k` | final number after the last `<|/think|>` / “Answer:” |
| `humaneval`, `mbpp` | pass@1 with execution |
| `needle` | retrieve a fact hidden in a long context |
| `vision_caption` | caption quality for multimodal models (skipped otherwise) |
| `ifeval` | Google IFEval, restricted to instruction ids supported by `verifiers.ifeval_to_constraints`; prompt-level strict accuracy (instruction-level in details) |
| `ifeval_lite` | 30 bundled prompts with verifiable constraints. They are held out from training: built with a different seed and different templates from the synthetic data |
| `detail_check` | 20 bundled prompts with hidden small requirements; score = mean fraction satisfied |
| `plan_bench` | 20 verifiable planning problems (dependency ordering = valid topological order, scheduling with time windows, resource allocation with sums); answer read from the final `Javob:` / `Answer:` line |
| `project_build` | 4 mini-project specs. `llm.build_project` runs in a temp dir, then **hidden** pytest tests run; score = fraction passed (only from cc-005) |
| `speed` | generation tokens/s and peak memory |

Tasks that need downloads (`mmlu`, `arc_*`, `hellaswag`, `gsm8k`, `humaneval`, `mbpp`,
`belebele_uz`, `ifeval`) are **skipped, not failed**, when offline or when a download fails. A
suite never crashes because of a missing dataset.

Code is executed by `moonlight.benchmarks.execution.run_python(code, timeout=5.0)` in a separate
Python subprocess with resource limits where the OS supports them. This isolates the benchmark
process but is not a security sandbox; run untrusted models in a container.

## Reports

`Report(suite, results, passed)`, where each `TaskResult` holds `task, score, n, threshold, passed,
details, seconds`. `report.to_markdown()`, `report.to_dict()`, and `report.save("x.md" | "x.json")`.

## What the suites measure

The architecture makes design bets: retention heads with constant decoding state, sliding-window
attention, Muon, multi-token prediction, verifier-driven alignment. The suites quantify their
effect at each scale:

* **quality per parameter**: the standard tasks per tier;
* **long-context recall** with bounded memory: `needle` versus `speed` (peak memory);
* **inference efficiency**: `speed` (tokens/s, memory) for the hybrid cache;
* **instruction and detail following**: `ifeval*`, `detail_check`;
* **planning and agentic building**: `plan_bench`, `project_build`.

Compare against baselines only with identical suites, limits and hardware, and report both.
