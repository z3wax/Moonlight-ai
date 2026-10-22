"""Run a benchmark suite on an LLM and build the :class:`~moonlight.benchmarks.suites.Report` (SPEC §9).

``run_suite(suite, llm, limit=None, out=None, offline=False)`` runs every task of the suite:

* each task is timed and run with a fixed seed (:data:`moonlight.benchmarks.tasks.EVAL_SEED`),
  the model in eval mode (its previous mode is restored at the end);
* exceptions never escape: a task that raises becomes ``TaskResult(score=None, passed=False,
  error="ExcType: message")`` with the traceback tail in ``details["traceback"]``
  (``KeyboardInterrupt`` still stops the run); after a CUDA OOM the allocator cache is emptied;
* a task returning ``score None`` is *skipped* (``passed None``);
* ``Report.passed`` = every non-skipped task passed; the report is saved to ``out`` (``.md`` ->
  Markdown, otherwise JSON) when given.

``offline`` is also implied by ``llm.config.data.offline`` and the ``HF_HUB_OFFLINE`` /
``MOONLIGHT_OFFLINE`` environment variables.
"""

from __future__ import annotations

import inspect
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from moonlight.utils import get_logger

from .suites import Report, Suite, TaskResult, TaskSpec, _now, get_suite

__all__ = ["run_suite", "run_task", "effective_offline"]

log = get_logger("moonlight.benchmarks")


def effective_offline(llm: Any, offline: bool = False) -> bool:
    """``offline`` or ``config.data.offline`` or the HF/Moonlight offline environment variables."""
    if offline:
        return True
    try:
        if bool(llm.config.data.offline):
            return True
    except AttributeError:
        pass
    try:
        from moonlight.data.download import is_offline_env

        return bool(is_offline_env())
    except Exception:  # pragma: no cover - data package import problems must not break benchmarks
        return False


def _seed() -> None:
    from moonlight.benchmarks.tasks import EVAL_SEED
    from moonlight.utils import seed_everything

    seed_everything(EVAL_SEED)


def _free_cuda() -> None:
    try:
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def _effective_limit(spec_limit: Optional[int], limit: Optional[int]) -> Optional[int]:
    if limit is None:
        return spec_limit
    limit = max(1, int(limit))
    return limit if spec_limit is None else min(int(spec_limit), limit)


def _call_task(fn: Any, llm: Any, limit: Optional[int], offline: bool, params: Mapping[str, Any]) -> Any:
    """Call ``fn(llm, limit, offline, **params)``.

    Only the task's explicitly named keyword parameters are accepted (task functions also take
    ``**_`` for forward compatibility, but a typo in a suite YAML must not be silently ignored), so
    unknown parameters raise ``TypeError`` (reported as a task error).
    """
    sig = inspect.signature(fn)
    explicit = {name for name, p in sig.parameters.items()
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    explicit -= {"llm", "limit", "offline"}
    unknown = sorted(k for k in params if k not in explicit)
    if unknown:
        raise TypeError(f"unknown parameter(s) {unknown} for task {getattr(fn, '__name__', fn)!r}; "
                        f"accepted: {sorted(explicit)}")
    return fn(llm, limit, offline, **dict(params))


def run_task(spec: Union[TaskSpec, Mapping[str, Any]], llm: Any, limit: Optional[int] = None,
             offline: bool = False) -> TaskResult:
    """Run one task spec and return its :class:`TaskResult` (never raises, except
    ``KeyboardInterrupt`` / ``SystemExit``)."""
    from moonlight.benchmarks.tasks import TASKS

    if not isinstance(spec, TaskSpec):
        spec = TaskSpec.from_dict(spec)
    lim = _effective_limit(spec.limit, limit)
    t0 = time.perf_counter()
    try:
        fn = TASKS[spec.task]
        _seed()
        raw = _call_task(fn, llm, lim, offline, spec.params)
        if not isinstance(raw, Mapping):
            raise TypeError(f"task {spec.task!r} returned {type(raw).__name__}, expected a mapping")
        score = raw.get("score")
        score = None if score is None else float(score)
        details: Dict[str, Any] = dict(raw.get("details") or {})
        passed = spec.judge(score)
        if score is not None and "passed" in raw and raw["passed"] is not None:
            passed = bool(raw["passed"]) and bool(passed)
        return TaskResult(task=spec.label, score=score, n=int(raw.get("n") or 0), threshold=spec.threshold,
                          passed=passed, details=details, seconds=time.perf_counter() - t0,
                          lower_is_better=spec.lower_is_better)
    except Exception as e:  # the suite must never crash
        if "OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower():
            _free_cuda()
        tb = traceback.format_exc(limit=8)
        return TaskResult(task=spec.label, score=None, n=0, threshold=spec.threshold, passed=False,
                          details={"traceback": tb[-4000:]}, seconds=time.perf_counter() - t0,
                          lower_is_better=spec.lower_is_better, error=f"{type(e).__name__}: {e}"[:1000])


def _model_meta(llm: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    cfg = getattr(llm, "config", None)
    if cfg is not None:
        meta["model"] = getattr(cfg, "name", None)
        meta["kind"] = getattr(cfg, "kind", None)
        meta["nominal_params"] = getattr(cfg, "nominal_params", None)
    try:
        meta["params"] = int(llm.num_params())
    except Exception:
        pass
    meta["device"] = str(getattr(llm, "device", "")) or None
    try:
        from moonlight.version import __version__

        meta["moonlight_version"] = __version__
    except Exception:  # pragma: no cover
        pass
    return {k: v for k, v in meta.items() if v is not None}


def run_suite(suite: Union[Suite, str], llm: Any, limit: Optional[int] = None,
              out: Union[str, "Path", None] = None, offline: bool = False,
              tasks: Optional[Sequence[str]] = None, verbose: bool = True) -> Report:
    """Run ``suite`` (a :class:`Suite` or a suite name) on ``llm`` and return the :class:`Report`.

    Args:
        limit: cap on every task's example count (``min(task.limit, limit)``).
        out: save the report there (``.md``/``.markdown``/``.txt`` -> Markdown, else JSON).
        offline: skip download-based tasks (see :func:`effective_offline`).
        tasks: optional subset of task names (labels) to run.
        verbose: log a line per task.
    """
    if not isinstance(suite, Suite):
        suite = get_suite(suite)
    if tasks:
        suite = suite.select(tasks)
    off = effective_offline(llm, offline)
    model = getattr(llm, "model", None)
    was_training = bool(getattr(model, "training", False)) if model is not None else False
    if model is not None and hasattr(model, "eval"):
        model.eval()
    started = _now()
    t0 = time.perf_counter()
    results = []
    try:
        for i, spec in enumerate(suite.tasks, start=1):
            if verbose:
                log.info("[%s %d/%d] %s ...", suite.name, i, len(suite.tasks), spec.label)
            res = run_task(spec, llm, limit=limit, offline=off)
            results.append(res)
            if verbose:
                score = "-" if res.score is None else f"{res.score:.4g}"
                extra = f" ({res.error})" if res.error else (
                    f" ({res.details.get('skipped')})" if res.passed is None and res.details.get("skipped") else "")
                log.info("[%s %d/%d] %s: %s score=%s n=%d %.1fs%s", suite.name, i, len(suite.tasks), spec.label,
                         res.status, score, res.n, res.seconds, extra)
    finally:
        model = getattr(llm, "model", None)
        if model is not None and hasattr(model, "train"):
            model.train(was_training)
    meta = _model_meta(llm)
    meta.update(offline=off, limit=limit, started=started, seconds=round(time.perf_counter() - t0, 2),
                param_range=list(suite.param_range))
    report = Report(suite=suite.name, results=results, passed=Report.verdict(results),
                    description=suite.description, meta=meta)
    if verbose:
        log.info("%s", report.summary())
    if out:
        path = report.save(out)
        if verbose:
            log.info("report saved to %s", path)
    return report
