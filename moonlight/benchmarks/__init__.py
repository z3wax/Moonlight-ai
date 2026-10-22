"""Moonlight benchmark harness: tiered suites ``cc-001`` ... ``cc-010`` (SPEC §9, §13.7).

```python
import moonlight as ml
suite = ml.benchmarks.type("cc-001")        # or cfg.bnchmrks.type("cc-001")
report = suite.run(llm)                     # -> Report
print(report.to_markdown()); report.save("report.json")
ml.benchmarks.list_suites(); ml.benchmarks.suite_for_params("20M")   # -> 'cc-002'
```

Modules: :mod:`.suites` (torch-free: YAML suites, ``Suite`` / ``Report`` / ``TaskResult``),
:mod:`.tasks` (task implementations, ``TASKS`` registry; torch imported lazily inside tasks),
:mod:`.execution` (subprocess Python execution with timeout and resource limits) and
:mod:`.runner` (runs a suite and builds the report). Importing this package does not import torch.
"""

from __future__ import annotations

import importlib
from typing import Any, List

from .suites import (
    SUITES_DIR,
    TASK_NAMES,
    Report,
    Suite,
    TaskResult,
    TaskSpec,
    get_suite,
    list_suites,
    load_suite,
    suite_for_params,
    type,
)

__all__ = [
    "type", "get_suite", "load_suite", "list_suites", "suite_for_params", "Suite", "Report", "TaskResult",
    "TaskSpec", "TASK_NAMES", "SUITES_DIR", "run_suite", "run_task", "TASKS", "run_python",
]

# lazily resolved attributes: name -> (module, attribute)
_LAZY = {
    "run_suite": (".runner", "run_suite"),
    "run_task": (".runner", "run_task"),
    "TASKS": (".tasks", "TASKS"),
    "loglikelihood": (".tasks", "loglikelihood"),
    "run_python": (".execution", "run_python"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'moonlight.benchmarks' has no attribute {name!r}")
    module = importlib.import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_LAZY))
