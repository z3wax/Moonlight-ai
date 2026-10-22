"""Benchmark suites ``cc-001`` ... ``cc-010`` (SPEC §9, §13.7). Torch-free.

```python
from moonlight.benchmarks import suites
suite = suites.type("cc-002")              # also cfg.bnchmrks.type("cc-002") / ml.benchmarks.type(...)
report = suite.run(llm, limit=None, out="report.md", offline=False)
print(report.to_markdown()); report.passed
suites.list_suites()                       # ['cc-001', ..., 'cc-010']
suites.suite_for_params("20M")             # 'cc-002'
```

Suite YAML (``moonlight/benchmarks/suites/<name>.yaml``)::

    name: cc-001
    description: ...
    param_range: [1M, 2M]                  # nominal total params (labels or ints)
    tasks:
      - {task: chat_format, limit: 20, threshold: 0.5}
      - {task: perplexity, name: perplexity_uz, limit: 4096, threshold: 3.5, lower_is_better: true,
         languages: [uz]}

Task entry keys: ``task`` (a name in :data:`TASK_NAMES`, required), ``limit`` (max examples; task
specific meaning, see :mod:`moonlight.benchmarks.tasks`), ``threshold`` (null = always passes when
it runs, e.g. ``speed``), ``lower_is_better`` (pass iff ``score <= threshold``; e.g. bits-per-byte),
``name`` (display name, defaults to ``task``; lets a suite run one task twice with different
parameters) and any other key, which is passed to the task function as a keyword parameter.

Pass rule: a task passes iff it ran and ``score >= threshold`` (``<=`` when lower is better).
Tasks that cannot run (offline / download failed / not multimodal ...) are *skipped*
(``score None, passed None``) and do not affect the verdict; a task that raised an exception is
*failed* (``passed False`` with the error in ``TaskResult.error``). ``Report.passed`` is True iff
every non-skipped task passed.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

__all__ = [
    "TASK_NAMES",
    "SUITES_DIR",
    "TaskSpec",
    "TaskResult",
    "Report",
    "Suite",
    "type",
    "get_suite",
    "load_suite",
    "list_suites",
    "suite_for_params",
]

# NOTE: this module defines a function called ``type`` (SPEC §9) which shadows the builtin below
# its definition; the module therefore never calls the builtin ``type``.

#: directory holding the bundled suite YAML files.
SUITES_DIR = Path(__file__).resolve().parent / "suites"

#: every task implemented in :mod:`moonlight.benchmarks.tasks` (kept here so suites load without
#: importing the task module; ``tasks.TASKS`` is checked against it by the test-suite).
TASK_NAMES: Tuple[str, ...] = (
    "chat_format", "chat_sanity", "identity", "perplexity", "arithmetic", "knowledge_uz", "tool_format",
    "tool_e2e", "code_lite", "belebele_uz", "arc_easy", "arc_challenge", "hellaswag", "mmlu", "gsm8k",
    "humaneval", "mbpp", "needle", "vision_caption", "speed",
    # SPEC §13.7
    "ifeval", "ifeval_lite", "detail_check", "plan_bench", "project_build",
)

_SPEC_KEYS = ("task", "name", "limit", "threshold", "lower_is_better")


def _parse_count(value: Any) -> int:
    from moonlight.config import parse_size  # torch-free

    return int(parse_size(value))


def _fmt_count(n: int) -> str:
    from moonlight.config import format_size

    return format_size(n)


def _num_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _jsonable(obj: Any) -> Any:
    """Recursively convert to JSON-serializable python (tuples -> lists, numpy scalars -> python,
    non-finite floats -> None, unknown objects -> str)."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    item = getattr(obj, "item", None)  # numpy / torch scalars
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError):
            pass
    return str(obj)


# ============================================================================================ specs

@dataclass
class TaskSpec:
    """One task entry of a suite (see the module docstring for the YAML keys)."""

    task: str
    limit: Optional[int] = None
    threshold: Optional[float] = None
    lower_is_better: bool = False
    name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Display name (``name`` or ``task``); also ``TaskResult.task``."""
        return self.name or self.task

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], where: str = "suite") -> "TaskSpec":
        if not isinstance(d, Mapping):
            raise ValueError(f"{where}: task entries must be mappings, got {d!r}")
        if "task" not in d:
            raise ValueError(f"{where}: task entry without 'task' key: {dict(d)!r}")
        task = str(d["task"])
        if task not in TASK_NAMES:
            raise ValueError(f"{where}: unknown task {task!r}; known: {', '.join(TASK_NAMES)}")
        limit = d.get("limit")
        if limit is not None:
            limit = int(limit)
            if limit < 1:
                raise ValueError(f"{where}: task {task!r} limit must be >= 1, got {limit}")
        threshold = d.get("threshold")
        if threshold is not None:
            threshold = float(threshold)
        params = {str(k): v for k, v in d.items() if k not in _SPEC_KEYS}
        return cls(task=task, limit=limit, threshold=threshold, lower_is_better=bool(d.get("lower_is_better", False)),
                   name=(str(d["name"]) if d.get("name") else None), params=params)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"task": self.task}
        if self.name:
            out["name"] = self.name
        out["limit"] = self.limit
        out["threshold"] = self.threshold
        if self.lower_is_better:
            out["lower_is_better"] = True
        out.update(copy.deepcopy(self.params))
        return out

    def judge(self, score: Optional[float]) -> Optional[bool]:
        """Pass / fail for ``score`` (None = skipped)."""
        return _judge(score, self.threshold, self.lower_is_better)


def _judge(score: Optional[float], threshold: Optional[float], lower_is_better: bool) -> Optional[bool]:
    if score is None:
        return None
    if threshold is None:
        return True
    return bool(score <= threshold) if lower_is_better else bool(score >= threshold)


# ============================================================================================ results

@dataclass
class TaskResult:
    """Result of one task (SPEC §9).

    Attributes:
        task: task (display) name.
        score: main metric (accuracy / pass rate in [0, 1], bits-per-byte for ``perplexity``,
            decode tokens/s for ``speed``); None when the task was skipped or errored.
        n: number of evaluated examples.
        threshold: pass threshold (None = no threshold).
        passed: True / False, or None when skipped.
        details: task specific extra metrics, per-example samples, skip reason ...
        seconds: wall-clock time of the task.
        lower_is_better: threshold direction.
        error: ``"ExcType: message"`` when the task raised (then ``passed`` is False).
    """

    task: str
    score: Optional[float]
    n: int
    threshold: Optional[float]
    passed: Optional[bool]
    details: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    lower_is_better: bool = False
    error: Optional[str] = None

    @property
    def skipped(self) -> bool:
        return self.passed is None

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        if self.passed is None:
            return "SKIP"
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        return {"task": self.task, "score": _num_or_none(self.score), "n": int(self.n),
                "threshold": _num_or_none(self.threshold), "passed": self.passed,
                "lower_is_better": bool(self.lower_is_better), "seconds": round(float(self.seconds), 3),
                "error": self.error, "details": _jsonable(self.details)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TaskResult":
        return cls(task=str(d["task"]), score=_num_or_none(d.get("score")), n=int(d.get("n") or 0),
                   threshold=_num_or_none(d.get("threshold")), passed=d.get("passed"),
                   details=dict(d.get("details") or {}), seconds=float(d.get("seconds") or 0.0),
                   lower_is_better=bool(d.get("lower_is_better", False)), error=d.get("error"))


def _fmt_score(x: Optional[float]) -> str:
    if x is None:
        return "–"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:.1f}"
    return f"{x:.3f}"


def _md_escape(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


@dataclass
class Report:
    """Outcome of :meth:`Suite.run` (SPEC §9).

    ``passed`` is True iff every non-skipped task passed. ``meta`` holds run information (model
    name, parameter count, device, offline flag, start time, total seconds, library version).
    """

    suite: str
    results: List[TaskResult]
    passed: bool
    description: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def verdict(results: Sequence[TaskResult]) -> bool:
        """True iff every non-skipped result passed (SPEC §9; vacuously True when all were skipped)."""
        return all(r.passed for r in results if r.passed is not None)

    def __getitem__(self, task: str) -> TaskResult:
        for r in self.results:
            if r.task == task:
                return r
        raise KeyError(task)

    @property
    def scores(self) -> Dict[str, Optional[float]]:
        return {r.task: r.score for r in self.results}

    def counts(self) -> Dict[str, int]:
        out = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
        for r in self.results:
            out[r.status] += 1
        return out

    def summary(self) -> str:
        c = self.counts()
        return (f"{self.suite}: {'PASSED' if self.passed else 'FAILED'} "
                f"({c['PASS']} passed, {c['FAIL']} failed, {c['ERROR']} errors, {c['SKIP']} skipped)")

    def to_dict(self) -> Dict[str, Any]:
        return {"suite": self.suite, "description": self.description, "passed": bool(self.passed),
                "meta": _jsonable(self.meta), "results": [r.to_dict() for r in self.results]}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Report":
        results = [TaskResult.from_dict(r) for r in d.get("results") or []]
        return cls(suite=str(d.get("suite", "")), results=results, passed=bool(d.get("passed", False)),
                   description=str(d.get("description", "")), meta=dict(d.get("meta") or {}))

    @classmethod
    def load(cls, path: Union[str, "Path"]) -> "Report":
        """Load a report saved as JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_markdown(self) -> str:
        m = self.meta
        lines = [f"# Moonlight benchmark `{self.suite}` — {'PASSED' if self.passed else 'FAILED'}", ""]
        if self.description:
            lines += [self.description.strip(), ""]
        info = []
        if m.get("model"):
            info.append(f"model `{m['model']}`")
        if m.get("params"):
            try:
                info.append(f"{_fmt_count(int(m['params']))} params")
            except (TypeError, ValueError):
                pass
        if m.get("device"):
            info.append(f"device `{m['device']}`")
        if m.get("offline"):
            info.append("offline")
        if m.get("limit"):
            info.append(f"limit {m['limit']}")
        if m.get("seconds") is not None:
            info.append(f"{float(m['seconds']):.1f}s")
        if m.get("started"):
            info.append(str(m["started"]))
        if info:
            lines += [" · ".join(info), ""]
        lines += ["| task | score | threshold | n | result | time |", "|---|---:|---:|---:|:---:|---:|"]
        for r in self.results:
            thr = "–" if r.threshold is None else ("≤ " if r.lower_is_better else "≥ ") + _fmt_score(r.threshold)
            lines.append(f"| {_md_escape(r.task)} | {_fmt_score(r.score)} | {thr} | {r.n} | {r.status} | "
                         f"{r.seconds:.1f}s |")
        lines += ["", self.summary()]
        notes = []
        for r in self.results:
            if r.error:
                notes.append(f"- **{_md_escape(r.task)}** error: `{_md_escape(r.error)[:300]}`")
            elif r.passed is None and r.details.get("skipped"):
                notes.append(f"- **{_md_escape(r.task)}** skipped: {_md_escape(r.details['skipped'])[:300]}")
            elif r.details.get("summary"):
                notes.append(f"- **{_md_escape(r.task)}**: {_md_escape(r.details['summary'])[:300]}")
        if notes:
            lines += ["", "Notes:", ""] + notes
        return "\n".join(lines) + "\n"

    def save(self, path: Union[str, "Path"]) -> Path:
        """Write the report: Markdown for ``.md`` / ``.markdown`` / ``.txt``, JSON otherwise.
        Parent directories are created. Returns the path."""
        p = Path(path).expanduser()
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() in (".md", ".markdown", ".txt"):
            p.write_text(self.to_markdown(), encoding="utf-8")
        else:
            p.write_text(self.to_json() + "\n", encoding="utf-8")
        return p

    def __str__(self) -> str:
        return self.to_markdown()


# ============================================================================================ suites

@dataclass
class Suite:
    """A named list of benchmark tasks with thresholds (SPEC §9).

    ``param_range`` is ``(low, high)`` nominal total parameter counts (ints).
    """

    name: str
    description: str
    param_range: Tuple[int, int]
    tasks: List[TaskSpec]

    def __post_init__(self) -> None:
        lo, hi = (int(x) for x in self.param_range)
        if lo > hi:
            raise ValueError(f"suite {self.name!r}: param_range low > high ({lo} > {hi})")
        self.param_range = (lo, hi)
        labels = [t.label for t in self.tasks]
        dup = sorted({x for x in labels if labels.count(x) > 1})
        if dup:
            raise ValueError(f"suite {self.name!r}: duplicate task names {dup} (use 'name:' to disambiguate)")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, d: Mapping[str, Any], where: Optional[str] = None) -> "Suite":
        where = where or str(d.get("name", "suite"))
        unknown = sorted(set(d) - {"name", "description", "param_range", "tasks"})
        if unknown:
            raise ValueError(f"{where}: unknown suite keys {unknown}")
        if not d.get("name"):
            raise ValueError(f"{where}: suite needs a 'name'")
        pr = d.get("param_range") or [0, 0]
        if not isinstance(pr, (list, tuple)) or len(pr) != 2:
            raise ValueError(f"{where}: param_range must be [low, high], got {pr!r}")
        tasks = [TaskSpec.from_dict(t, where) for t in (d.get("tasks") or [])]
        if not tasks:
            raise ValueError(f"{where}: suite has no tasks")
        return cls(name=str(d["name"]), description=str(d.get("description") or "").strip(),
                   param_range=(_parse_count(pr[0]), _parse_count(pr[1])), tasks=tasks)

    @classmethod
    def from_yaml(cls, path: Union[str, "Path"]) -> "Suite":
        p = Path(path).expanduser()
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(f"{p}: suite YAML must be a mapping")
        return cls.from_dict(data, where=str(p))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "param_range": [_fmt_count(self.param_range[0]), _fmt_count(self.param_range[1])],
                "tasks": [t.to_dict() for t in self.tasks]}

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, width=110)

    # ------------------------------------------------------------------ access
    @property
    def task_names(self) -> List[str]:
        return [t.label for t in self.tasks]

    def get(self, name: str) -> TaskSpec:
        for t in self.tasks:
            if t.label == name:
                return t
        raise KeyError(f"suite {self.name!r} has no task {name!r}; tasks: {', '.join(self.task_names)}")

    def select(self, names: Iterable[str]) -> "Suite":
        """A copy with only the tasks named (``label`` or ``task``) in ``names`` (suite order kept)."""
        wanted = list(names)
        missing = [w for w in wanted if not any(w in (t.label, t.task) for t in self.tasks)]
        if missing:
            raise KeyError(f"suite {self.name!r} has no task(s) {missing}; tasks: {', '.join(self.task_names)}")
        keep = [copy.deepcopy(t) for t in self.tasks if t.label in wanted or t.task in wanted]
        return Suite(self.name, self.description, self.param_range, keep)

    def covers(self, n_params: Union[int, str]) -> bool:
        n = _parse_count(n_params)
        return self.param_range[0] <= n <= self.param_range[1]

    def __len__(self) -> int:
        return len(self.tasks)

    def __repr__(self) -> str:
        lo, hi = self.param_range
        return f"Suite({self.name!r}, {_fmt_count(lo)}–{_fmt_count(hi)}, tasks=[{', '.join(self.task_names)}])"

    # ------------------------------------------------------------------ running
    def run(self, llm: Any, limit: Optional[int] = None, out: Union[str, "Path", None] = None,
            offline: bool = False, tasks: Optional[Sequence[str]] = None, verbose: bool = True) -> Report:
        """Run every task on ``llm`` and return a :class:`Report` (never raises for task failures).

        Args:
            llm: a ``moonlight.LLM``.
            limit: cap on each task's example count (``min(task.limit, limit)``).
            out: optional path; the report is saved there (``.md`` -> Markdown, else JSON).
            offline: skip tasks that need downloads (also implied by ``config.data.offline`` and
                ``HF_HUB_OFFLINE`` / ``MOONLIGHT_OFFLINE``).
            tasks: optional subset of task names to run.
            verbose: log one line per task.
        """
        from moonlight.benchmarks.runner import run_suite  # lazy: imports the task module

        return run_suite(self, llm, limit=limit, out=out, offline=offline, tasks=tasks, verbose=verbose)


# ============================================================================================ registry

_CACHE: Dict[str, Suite] = {}


def list_suites() -> List[str]:
    """Names of the bundled suites, sorted (``['cc-001', ..., 'cc-010']``)."""
    return sorted(p.stem for p in SUITES_DIR.glob("*.yaml"))


def load_suite(path: Union[str, "Path"]) -> Suite:
    """Load a suite from a YAML file path."""
    return Suite.from_yaml(path)


def get_suite(name: Union[str, "Path", Suite]) -> Suite:
    """Suite by bundled name (``"cc-001"``; ``"CC-1"`` / ``"cc1"`` / ``"1"`` are normalized) or by
    path to a suite YAML. Returns a fresh copy (callers may modify it)."""
    if isinstance(name, Suite):
        return copy.deepcopy(name)
    raw = str(name).strip()
    key = _normalize_name(raw)
    path = SUITES_DIR / f"{key}.yaml"
    if path.is_file():
        if key not in _CACHE:
            _CACHE[key] = Suite.from_yaml(path)
        return copy.deepcopy(_CACHE[key])
    p = Path(raw).expanduser()
    if raw.endswith((".yaml", ".yml")) and p.is_file():
        return Suite.from_yaml(p)
    raise KeyError(f"unknown benchmark suite {raw!r}; available: {', '.join(list_suites())}")


def _normalize_name(raw: str) -> str:
    s = raw.lower().replace("_", "-")
    digits = s[3:] if s.startswith("cc-") else (s[2:] if s.startswith("cc") else s)
    digits = digits.lstrip("-")
    if digits.isdigit():
        return f"cc-{int(digits):03d}"
    return s


def type(name: Union[str, "Path", Suite]) -> Suite:  # noqa: A001 - name fixed by SPEC §9
    """``moonlight.benchmarks.type("cc-001") -> Suite`` (see :func:`get_suite`)."""
    return get_suite(name)


def suite_for_params(n: Union[int, float, str]) -> str:
    """Suite name for a NOMINAL total parameter count (int or label like ``"20M"``).

    Rule: the suite with the largest lower bound ``param_range[0] <= n``; counts below the first
    suite map to ``cc-001`` and counts between two ranges (e.g. 3M, 30M) map to the lower suite.
    Example: ``1M, 2M -> cc-001``, ``5M..20M -> cc-002``, ``50M, 100M -> cc-003``, ``150M..300M ->
    cc-004``, ``1B..3B -> cc-005``, ``7B, 13B -> cc-006``, ``16B..70B -> cc-007``,
    ``100B..235B -> cc-008``, ``405B, 671B -> cc-009``, ``1T -> cc-010``.
    """
    count = _parse_count(n) if not isinstance(n, float) else int(round(n))
    suites = sorted((get_suite(s) for s in list_suites()), key=lambda s: (s.param_range[0], s.name))
    if not suites:
        raise RuntimeError(f"no suite YAML files found in {SUITES_DIR}")
    chosen = suites[0]
    for s in suites:
        if s.param_range[0] <= count:
            chosen = s
    return chosen.name


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

