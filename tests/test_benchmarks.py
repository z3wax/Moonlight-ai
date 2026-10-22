"""Benchmark harness (SPEC §9 + §13.7): suite definitions, tier mapping, sandboxed execution and an
offline run of cc-001 on a random tiny model."""

from __future__ import annotations

import json

import pytest

from conftest import SUITE_NAMES, SUITE_RANGES

from moonlight.benchmarks import suites
from moonlight.config import parse_size

ALWAYS = ("chat_format", "chat_sanity", "identity", "speed")
SPEC_TASKS = {"chat_format", "chat_sanity", "identity", "perplexity", "arithmetic", "knowledge_uz", "tool_format",
              "tool_e2e", "code_lite", "belebele_uz", "arc_easy", "arc_challenge", "hellaswag", "mmlu", "gsm8k",
              "humaneval", "mbpp", "needle", "vision_caption", "speed",
              # v1.1 (§13.7)
              "ifeval", "ifeval_lite", "detail_check", "plan_bench", "project_build"}
# §13.7: first suite (by tier index) that must contain each v1.1 task
V11_FROM = {"ifeval_lite": 2, "detail_check": 2, "plan_bench": 3, "ifeval": 4, "project_build": 5}
# §9: tasks each tier row adds ("+ ..."). Lower-tier "lite" tasks may be superseded later (e.g. tool_e2e
# replaces tool_format), so only the row's own tasks are required at that tier; cc-006's full set and
# cc-007's vision task carry on to cc-010 ("cc-008 = cc-007 stricter", ...).
TIER_TASKS = {
    2: {"arithmetic", "knowledge_uz", "tool_format"},
    3: {"code_lite", "belebele_uz"},
    4: {"arc_easy", "hellaswag", "humaneval", "tool_e2e"},
    5: {"mmlu", "arc_challenge", "gsm8k", "mbpp", "humaneval"},
    6: {"mmlu", "arc_easy", "arc_challenge", "hellaswag", "gsm8k", "humaneval", "mbpp", "needle"},
    7: {"vision_caption"},
}
CARRIED_FROM = (6, 7)


def _task_names(suite):
    out = []
    for t in suite.tasks:
        out.append(t["task"] if isinstance(t, dict) else getattr(t, "task", getattr(t, "name", None)))
    return out


def _task_entry(suite, name):
    for t in suite.tasks:
        d = t if isinstance(t, dict) else vars(t)
        if (d.get("task") or d.get("name")) == name:
            return d
    return None


def test_ten_suites_load():
    assert sorted(suites.list_suites()) == list(SUITE_NAMES)
    for name in SUITE_NAMES:
        s = suites.type(name)
        assert s.name == name and s.description
        lo, hi = (parse_size(x) for x in s.param_range)
        spec_lo, spec_hi = next((lo_, hi_) for n, lo_, hi_ in SUITE_RANGES if n == name)
        assert lo == spec_lo and hi >= spec_hi, f"{name}: param_range {s.param_range}"
    with pytest.raises((KeyError, ValueError)):
        suites.type("cc-999")


def test_every_suite_has_core_tasks_and_known_tasks():
    from moonlight.benchmarks.tasks import TASKS

    assert SPEC_TASKS <= set(TASKS), f"missing task implementations: {sorted(SPEC_TASKS - set(TASKS))}"
    for name in SUITE_NAMES:
        s = suites.type(name)
        names = _task_names(s)
        for core in ALWAYS:
            assert core in names, f"{name} lacks {core}"
        for t in names:
            assert t in TASKS, f"{name}: unknown task {t}"
        for t in s.tasks:
            d = t if isinstance(t, dict) else vars(t)
            thr = d.get("threshold")
            assert thr is None or 0.0 <= float(thr) or d.get("task") == "speed"


def test_tier_composition():
    for idx, name in enumerate(SUITE_NAMES, start=1):
        names = set(_task_names(suites.type(name)))
        required = set(TIER_TASKS.get(idx, set()))
        for tier in CARRIED_FROM:
            if idx >= tier:
                required |= TIER_TASKS[tier]
        assert required <= names, f"{name} must include {sorted(required - names)}"
        for task, first in V11_FROM.items():
            if idx >= first:
                assert task in names, f"{name} must include {task} (SPEC §13.7)"
            elif task == "project_build":
                assert task not in names, f"{name}: project_build only from cc-005"
    base = set(_task_names(suites.type("cc-007")))
    for name in SUITE_NAMES[7:]:
        assert base <= set(_task_names(suites.type(name))), f"{name} = cc-007 with stricter thresholds"


def test_large_tiers_get_stricter():
    """SPEC §9: cc-008 = cc-007 with stricter thresholds, cc-009 stricter, cc-010 strictest. For shared
    score tasks (higher = better) the threshold never decreases and at least one increases."""
    skip = {"speed", "perplexity"}
    for lower, upper in zip(SUITE_NAMES[6:], SUITE_NAMES[7:]):
        a, b = suites.type(lower), suites.type(upper)
        shared = (set(_task_names(a)) & set(_task_names(b))) - skip
        stricter = 0
        for t in shared:
            ta, tb = _task_entry(a, t).get("threshold"), _task_entry(b, t).get("threshold")
            if ta is None or tb is None:
                continue
            assert float(tb) >= float(ta) - 1e-9, f"{upper}: {t} threshold {tb} < {lower} {ta}"
            stricter += float(tb) > float(ta) + 1e-9
        assert stricter >= 1, f"{upper} must be stricter than {lower}"


@pytest.mark.parametrize("name, lo, hi", SUITE_RANGES)
def test_suite_for_params(name, lo, hi):
    assert suites.suite_for_params(int(lo)) == name
    assert suites.suite_for_params(int(hi)) == name


def test_suite_for_params_moe_uses_total():
    assert suites.suite_for_params(parse_size("16B")) == "cc-007"
    assert suites.suite_for_params(parse_size("671B")) == "cc-009"
    assert suites.suite_for_params(parse_size("1T")) == "cc-010"


def test_top_level_benchmarks_accessor():
    import moonlight as ml

    assert ml.benchmarks.type("cc-003").name == "cc-003"


# ================================================================================================
# execution sandbox
# ================================================================================================

def test_run_python():
    from moonlight.benchmarks.execution import run_python

    ok, out, err = run_python("print(sum(range(10)))")
    assert ok and out.strip() == "45"
    ok, out, err = run_python("assert 1 == 2, 'boom'")
    assert not ok and "boom" in err
    ok, out, err = run_python("while True:\n    pass\n", timeout=1.0)
    assert not ok
    ok, _, err = run_python("raise SystemExit(3)")
    assert not ok


# ================================================================================================
# offline run on a random tiny model
# ================================================================================================

@pytest.fixture(scope="module")
def tiny_llm():
    pytest.importorskip("torch")
    from conftest import make_tiny_config

    from moonlight.llm import LLM

    cfg = make_tiny_config(chat={"max_new_tokens": 6, "best_of": 1})
    return LLM(cfg, device="cpu")


def test_offline_cc001_report(tiny_llm, tmp_path):
    suite = suites.type("cc-001")
    report = suite.run(tiny_llm, limit=2, offline=True)
    assert getattr(report.suite, "name", report.suite) == "cc-001"
    assert len(report.results) == len(suite.tasks)
    names = set()
    for r in report.results:
        names.add(r.task)
        assert r.score is None or isinstance(r.score, (int, float))
        assert r.passed in (True, False, None) and r.seconds >= 0
        if r.task in ("chat_format", "chat_sanity", "identity"):
            assert r.score is not None and r.n >= 1, f"{r.task} must run offline"
    assert set(ALWAYS) <= names
    assert report.passed in (True, False)
    md = report.to_markdown()
    assert isinstance(md, str) and "cc-001" in md and "chat_format" in md
    d = report.to_dict()
    json.dumps(d)  # JSON-serializable
    for ext in ("json", "md"):
        path = tmp_path / f"report.{ext}"
        report.save(path)
        assert path.is_file() and path.stat().st_size > 0
    assert json.loads((tmp_path / "report.json").read_text())["results"]


def test_llm_benchmark_shorthand_and_offline_skips(tiny_llm):
    report = tiny_llm.benchmark("cc-001", limit=1, offline=True)
    assert report.results
    # a suite with download-only tasks never crashes offline: those tasks are skipped (score None)
    rep4 = suites.type("cc-004").run(tiny_llm, limit=1, offline=True)
    by_task = {r.task: r for r in rep4.results}
    for t in ("arc_easy", "hellaswag", "humaneval", "ifeval"):
        assert by_task[t].score is None and by_task[t].passed is None, f"{t} must be skipped offline"


def test_loglikelihood_helper(tiny_llm):
    from moonlight.benchmarks.tasks import loglikelihood

    out = loglikelihood(tiny_llm, "Toshkent O'zbekistonning", " poytaxti")
    vals = out if isinstance(out, (tuple, list)) else (out,)
    assert all(isinstance(v, float) or hasattr(v, "__float__") for v in vals)
    assert float(vals[0]) <= 0.0
