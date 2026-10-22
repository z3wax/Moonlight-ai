# Tools, planning and agentic project building

Every Moonlight chat model is trained to call tools. This page documents the wire format, the
Python API, the built-in tools, and the v1.1 planning and project-building features
(`moonlight/tools.py`, `moonlight/chat.py`).

## 1. Wire format

Tool schemas go into the system block; a call is one assistant message that starts with the
`<|tool_call|>` token (id 8); the result comes back as a `<|tool_result|>` message:

```
<|bos|><|system|>{system prompt}<|tools|>[{"name": "calculator", "description": "...", "parameters": {...}}]<|end|>
<|user|>12 * 37 nechchi?<|end|>
<|assistant|><|tool_call|>{"name": "calculator", "arguments": {"expression": "12 * 37"}}<|end|>
<|tool_result|>444<|end|>
<|assistant|>12 × 37 = 444.<|end|>
```

Message dicts (SPEC §5):

```python
{"role": "assistant", "content": "", "tool_calls": [{"name": "calculator", "arguments": {"expression": "12 * 37"}}]}
{"role": "tool", "name": "calculator", "content": "444"}
```

The chat loop (`ChatSession.send`) works as follows:

1. Generate until `<|end|>` or `<|eos|>`.
2. If the **first generated token id** is `<|tool_call|>`, parse the JSON with `parse_tool_call`,
   which is tolerant: it accepts fences, single quotes, arguments given as a JSON string, and glaive-style `<functioncall>`.
3. Run `registry.call(name, arguments)`. This never raises: errors come back as `{"error": "..."}`.
4. Append the call and its result to the history, then generate again.

After `max_tool_rounds` (default 4) calls, the model must answer.

## 2. Python API

```python
import moonlight as ml

@ml.tool                                   # schema inferred from type hints + docstring
def weather(city: str, unit: str = "C") -> dict:
    """Current weather for a city.

    Args:
        city: city name, e.g. Toshkent
        unit: C or F
    """
    return {"city": city, "temp": 31, "unit": unit}

llm.chat("Toshkentda havo qanday?", tools=[weather])          # one-shot
chat = ml.ChatSession(llm, tools=True)                         # built-in tools
chat.send("100 km necha mil?"); chat.send("Hozir soat nechchi?"); chat.history
```

`tools=` accepts `True` (the built-in tools), a `ToolRegistry`, or a list of `Tool` objects,
functions, schema dicts or built-in names. `ToolRegistry.register(fn_or_tool)`, `.schemas()`,
`.call(name, arguments) -> str`.

## 3. Built-in tools

| tool | description |
|---|---|
| `calculator(expression)` | **safe** AST evaluator: numbers, `+ - * / // % **`, parentheses, unary ±, `sqrt sin cos log exp`, `pi e`. No names, attributes, calls to anything else, lambdas or comprehensions; limits on exponent, magnitude, length and node count |
| `current_time(timezone="UTC")` | date / time / weekday for an IANA zone or a UTC offset |
| `unit_convert(value, from_unit, to_unit)` | length, mass and temperature |
| `python_exec(code)` | **disabled** unless `allow_code_exec=True`. Runs in a separate `python -I` subprocess with a 5 s timeout. This is process isolation, **not** a security sandbox (network and filesystem access are not prevented) |

## 4. Planning (`llm.plan`)

`llm.plan(task)` answers in the **Moonlight PLAN FORMAT**, which the `plan` synthetic data teaches
and `plan_bench` measures:

```
<|think|>Maqsad / goal, cheklovlar / constraints, noma'lumlar / unknowns, xavflar / risks<|/think|>
**Reja / Plan**
1. <action>. Natija / Output: <expected output>. Tekshiruv / Check: <how to verify>.
2. ...
**Tekshiruv / Verification**
- <user requirement 1> → step 2
- <user requirement 2> → steps 3, 5
```

The system prompt is `moonlight.chat.PLAN_SYSTEM_PROMPT`; it contains one Uzbek and one English sentence.

## 5. Agentic project building (`llm.build_project`)

```python
res = llm.build_project("CLI temperature converter with tests", "work/temp_conv", max_steps=24)
res.tests_passed, res.files, res.steps, res.transcript
```

* `Workspace(root, allow_exec=True, timeout=20)` is sandboxed to `root`. Paths are resolved, and `..`
  escapes, absolute paths outside the root and symlink escapes are rejected. Tools: `read_file(path)`,
  `write_file(path, content)`, `list_dir(path=".")`, `run_tests(target=".")` (`python -m pytest -q`
  if pytest is importable, otherwise unittest discovery; subprocess with `cwd=root` and timeout; returns
  trimmed output plus `passed`), and `run_python(code)`. `workspace_tools(ws)` returns them as `Tool`s.
* The agent uses `BUILDER_SYSTEM_PROMPT`: plan first, write code and tests, run the tests, fix until
  they pass, then summarise every requirement and where it is satisfied. If the model stops while tests
  are missing or failing, it is nudged to continue. The loop stops early once `run_tests` passes and the
  model gives its final answer.
* `ProjectResult(workdir, files, tests_passed, steps, transcript)`.
* CLI: `moonlight build MODEL "task" --workdir work/x --max-steps 24` and `moonlight plan MODEL "task"`.

Generated code runs on your machine. Use a container or VM for untrusted models, and
`allow_exec=False` to disable execution entirely.

## 6. Verifiers

`moonlight.data.verifiers` provides deterministic checkers shared by synthetic data generation,
GRPO rewards and benchmarks:

* `check(response, {"type": ..., **params}) -> bool` and `check_all(response, constraints) -> (passed, total, details)`;
* constraint types: `word_count_max/min`, `sentence_count_max/min`, `bullet_count`, `paragraph_count`,
  `include_keywords`, `exclude_keywords`, `keyword_frequency`, `json_format`, `lowercase`, `uppercase`,
  `no_commas`, `ends_with`, `starts_with`, `title_brackets`, `sections`, `language_uz`,
  `quotation_wrapped`, `max_chars`;
* `ifeval_to_constraints(ids, kwargs)` (None for unsupported ids);
* reward helpers: `reward_exact_number`, `reward_constraints`, `reward_code`, `extract_final_answer`.

Reasoning inside `<|think|>…<|/think|>` is ignored by every checker; only the final answer counts.
