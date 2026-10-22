"""Sandboxed-ish Python execution for code benchmarks and code rewards (SPEC §9, §13.2).

``run_python(code, timeout=5.0) -> (ok, stdout, stderr)`` runs ``code`` in a fresh, isolated
interpreter (``python -I``) inside a throw-away temporary directory:

* wall-clock ``timeout`` (the child runs in its own session / process group and the whole group is
  killed when the timeout expires),
* POSIX resource limits where available: CPU seconds (``timeout + 1``), address space
  (``memory_mb``), created file size (16 MiB), core dumps disabled,
* a minimal environment (no ``PYTHONPATH`` / user site, ``HOME`` = the temp dir, one BLAS thread),
* an optional *guard* prelude (default on) that disables process-spawning and destructive
  filesystem functions (``os.system``, ``subprocess.Popen``, ``os.remove``, ``shutil.rmtree``, ...),
  in the spirit of the HumanEval "reliability guard".

This is process isolation, NOT a security sandbox: there is no network or filesystem isolation and
a determined program can undo the guard. Only run model-generated code on machines where that is
acceptable (containers / CI runners).

Helpers used by :mod:`moonlight.benchmarks.tasks`: :func:`build_program` (solution + tests ->
program), :func:`check_program` (run it, ok iff exit code 0) and :func:`run_test_functions`
(import a test file next to a project and run each ``test_*`` function, reporting per-test results
as JSON; used by the ``project_build`` task so pytest is not required).

Pure python, torch-free.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "run_python",
    "run_python_detailed",
    "build_program",
    "check_program",
    "run_test_functions",
    "GUARD_PRELUDE",
    "DEFAULT_MEMORY_MB",
]

#: default address-space limit of the child process (MiB); 0 disables the limit.
DEFAULT_MEMORY_MB = 2048
#: default cap (characters) of the returned stdout / stderr.
DEFAULT_MAX_OUTPUT = 20_000
_FSIZE_LIMIT = 16 * 2**20

#: prepended to the program when ``guard=True``. Disables process creation and destructive file
#: operations; keeps everything a typical benchmark solution needs.
GUARD_PRELUDE = '''\
def __moonlight_guard():
    import builtins, os, shutil, subprocess
    def _blocked(*_a, **_k):
        raise PermissionError("disabled by the moonlight benchmark sandbox")
    for _name in ("system", "kill", "killpg", "fork", "forkpty", "remove", "unlink", "removedirs",
                  "rmdir", "rename", "renames", "replace", "truncate", "putenv", "unsetenv", "setuid",
                  "setgid", "chmod", "chown", "lchown", "fchmod", "fchown", "execv", "execve", "execl",
                  "execle", "execlp", "execlpe", "execvp", "execvpe", "spawnl", "spawnle", "spawnv",
                  "spawnve", "popen", "posix_spawn", "posix_spawnp"):
        if hasattr(os, _name):
            setattr(os, _name, _blocked)
    for _name in ("rmtree", "move", "chown"):
        if hasattr(shutil, _name):
            setattr(shutil, _name, _blocked)
    for _name in ("Popen", "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput"):
        if hasattr(subprocess, _name):
            setattr(subprocess, _name, _blocked)
    builtins.help = None
__moonlight_guard()
del __moonlight_guard
'''


# ============================================================================================ low level

def _preexec(cpu_seconds: float, memory_mb: int) -> Optional[Callable[[], None]]:
    """``preexec_fn`` applying POSIX rlimits in the child (None on non-POSIX systems)."""
    if os.name != "posix":
        return None

    def apply() -> None:  # runs in the forked child before exec; must never raise
        try:
            import resource

            cpu = int(max(1, math.ceil(cpu_seconds)))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        except Exception:
            pass
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_FSIZE, (_FSIZE_LIMIT, _FSIZE_LIMIT))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except Exception:
            pass
        if memory_mb and memory_mb > 0:
            try:
                import resource

                limit = int(memory_mb) * 2**20
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except Exception:  # e.g. macOS refuses RLIMIT_AS changes
                pass

    return apply


def _child_env(home: str, extra_env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": home,
        "TMPDIR": home,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if os.name == "nt":  # pragma: no cover - Windows needs these to start Python
        for key in ("SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "PATHEXT"):
            if key in os.environ:
                env[key] = os.environ[key]
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    return env


def _trim(text: str, limit: int) -> str:
    """Keep the head and the tail of long outputs (errors are usually at the end)."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} characters trimmed] ...\n{text[-tail:]}"


def _kill_group(proc: "subprocess.Popen[bytes]") -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def run_python_detailed(code: str, timeout: float = 5.0, memory_mb: int = DEFAULT_MEMORY_MB,
                        guard: bool = True, max_output: int = DEFAULT_MAX_OUTPUT,
                        cwd: Union[str, "os.PathLike[str]", None] = None,
                        extra_env: Optional[Mapping[str, str]] = None,
                        stdin: Optional[str] = None) -> Dict[str, Any]:
    """Run ``code`` and return ``{"ok", "returncode", "stdout", "stderr", "timed_out"}``.

    Args:
        code: Python source (str).
        timeout: wall-clock seconds; on expiry the process group is killed and ``ok`` is False.
        memory_mb: address-space limit in MiB (0 = unlimited).
        guard: prepend :data:`GUARD_PRELUDE`.
        max_output: cap (characters) of the returned stdout / stderr (head + tail kept).
        cwd: working directory; default a fresh temporary directory (deleted afterwards). The
            program file itself always lives in a private temporary directory.
        extra_env: additional environment variables for the child.
        stdin: text fed to the child's stdin (default: no stdin).

    ``ok`` is True iff the program exited with status 0 before the timeout.
    """
    if not isinstance(code, str):
        raise TypeError(f"code must be a str, got {type(code).__name__}")
    timeout = max(0.1, float(timeout))
    source = (GUARD_PRELUDE + "\n" + code) if guard else code
    with tempfile.TemporaryDirectory(prefix="moonlight_bench_") as tmp:
        script = Path(tmp) / "program.py"
        script.write_text(source, encoding="utf-8")
        work = str(cwd) if cwd is not None else tmp
        kwargs: Dict[str, Any] = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True
            pre = _preexec(timeout + 1.0, int(memory_mb or 0))
            if pre is not None:
                kwargs["preexec_fn"] = pre
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", str(script)], cwd=work,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_child_env(tmp, extra_env), **kwargs)
        except OSError as e:
            return {"ok": False, "returncode": None, "stdout": "", "stderr": f"failed to start python: {e}",
                    "timed_out": False}
        timed_out = False
        try:
            out, err = proc.communicate(input=stdin.encode("utf-8") if stdin is not None else None,
                                        timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            try:
                out, err = proc.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - pipes held open by a grandchild
                out, err = b"", b""
        finally:
            if proc.poll() is None:  # pragma: no cover - defensive
                _kill_group(proc)
    stdout = _trim(out.decode("utf-8", errors="replace"), max_output)
    stderr = _trim(err.decode("utf-8", errors="replace"), max_output)
    if timed_out:
        stderr = (stderr + f"\nTIMEOUT: execution exceeded {timeout:g}s").strip()
    rc = None if timed_out else proc.returncode
    return {"ok": (not timed_out) and rc == 0, "returncode": rc, "stdout": stdout, "stderr": stderr,
            "timed_out": timed_out}


def run_python(code: str, timeout: float = 5.0, memory_mb: int = DEFAULT_MEMORY_MB, guard: bool = True,
               max_output: int = DEFAULT_MAX_OUTPUT) -> Tuple[bool, str, str]:
    """Run Python ``code`` in a subprocess with a timeout and resource limits (SPEC §9).

    Returns ``(ok, stdout, stderr)`` where ``ok`` is True iff the program exited with status 0
    within ``timeout`` seconds. See the module docstring for the isolation that is (and is not)
    provided.
    """
    r = run_python_detailed(code, timeout=timeout, memory_mb=memory_mb, guard=guard, max_output=max_output)
    return bool(r["ok"]), str(r["stdout"]), str(r["stderr"])


# ============================================================================================ programs

def build_program(solution: str, tests: Union[str, Sequence[str]], entry_point: Optional[str] = None,
                  prefix: str = "") -> str:
    """Assemble ``prefix + solution + tests`` into one program.

    ``tests`` is test code (asserts, or a HumanEval-style ``def check(candidate)``) or a list of
    assert statements. When the tests define ``check`` and ``entry_point`` is given,
    ``check(<entry_point>)`` is appended.
    """
    test_src = "\n".join(str(t) for t in tests) if isinstance(tests, (list, tuple)) else str(tests or "")
    parts = [p for p in (prefix.rstrip("\n"), solution.rstrip("\n"), test_src.rstrip("\n")) if p.strip()]
    program = "\n\n".join(parts) + "\n"
    if entry_point and re.search(r"^def check\(", test_src, re.MULTILINE):
        program += f"\ncheck({entry_point})\n"
    return program


def check_program(solution: str, tests: Union[str, Sequence[str]], entry_point: Optional[str] = None,
                  prefix: str = "", timeout: float = 5.0) -> Tuple[bool, str]:
    """Run ``solution`` against ``tests`` (see :func:`build_program`).

    Returns ``(passed, info)`` with ``info`` a short reason (``"passed"``, ``"timeout"``, or the
    last line of stderr, e.g. ``"AssertionError"``).
    """
    if not solution or not solution.strip():
        return False, "empty solution"
    r = run_python_detailed(build_program(solution, tests, entry_point, prefix), timeout=timeout)
    if r["ok"]:
        return True, "passed"
    if r["timed_out"]:
        return False, "timeout"
    lines = [ln for ln in str(r["stderr"]).strip().splitlines() if ln.strip()]
    return False, (lines[-1][:300] if lines else f"exit code {r['returncode']}")


_TEST_RUNNER = r'''
import importlib.util, inspect, json, sys, traceback
project, test_file = sys.argv[1], sys.argv[2]
sys.path.insert(0, project)
results = {"tests": [], "import_error": None}
try:
    spec = importlib.util.spec_from_file_location("moonlight_hidden_tests", test_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except BaseException as e:  # the project module may be missing or broken
    results["import_error"] = "".join(traceback.format_exception_only(type(e), e)).strip()[-500:]
    mod = None
if mod is not None:
    for name, fn in sorted(vars(mod).items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if inspect.signature(fn).parameters:
                raise TypeError("test functions must not take arguments")
            fn()
            results["tests"].append({"name": name, "passed": True, "error": ""})
        except BaseException as e:
            msg = "".join(traceback.format_exception_only(type(e), e)).strip()
            results["tests"].append({"name": name, "passed": False, "error": msg[-300:]})
print("@@RESULTS@@" + json.dumps(results))
'''


def run_test_functions(test_source: str, project_dir: Union[str, "os.PathLike[str]"], timeout: float = 30.0,
                       n_expected: Optional[int] = None) -> Dict[str, Any]:
    """Run every ``test_*`` function of ``test_source`` against the project in ``project_dir``.

    The tests are written to a private temporary file (so the project cannot see or modify them),
    imported in a subprocess with ``project_dir`` first on ``sys.path`` and ``cwd=project_dir``,
    and each test function is called (no fixtures; plain asserts). pytest is not required.

    Returns ``{"total", "passed", "tests": [{"name", "passed", "error"}], "error": str|None}``.
    When the module cannot be imported (e.g. the project file is missing) every expected test counts
    as failed: ``total`` is ``n_expected`` (or the number of ``def test_`` in the source).
    """
    project = Path(project_dir).resolve()
    if n_expected is None:
        n_expected = len(re.findall(r"^def test_\w+\s*\(", test_source, re.MULTILINE))
    with tempfile.TemporaryDirectory(prefix="moonlight_hidden_") as tmp:
        test_file = Path(tmp) / "hidden_tests.py"
        test_file.write_text(test_source, encoding="utf-8")
        runner = _TEST_RUNNER.replace("project, test_file = sys.argv[1], sys.argv[2]",
                                      f"project, test_file = {str(project)!r}, {str(test_file)!r}")
        r = run_python_detailed(runner, timeout=timeout, guard=False, cwd=str(project), max_output=200_000)
    out = str(r["stdout"])
    marker = out.rfind("@@RESULTS@@")
    parsed: Optional[Dict[str, Any]] = None
    if marker >= 0:
        try:
            parsed = json.loads(out[marker + len("@@RESULTS@@"):].strip().splitlines()[0])
        except (ValueError, IndexError):
            parsed = None
    if parsed is None:
        reason = "timeout" if r["timed_out"] else (str(r["stderr"]).strip().splitlines() or ["no result"])[-1]
        return {"total": int(n_expected), "passed": 0, "tests": [], "error": reason[:500]}
    tests: List[Dict[str, Any]] = list(parsed.get("tests") or [])
    if parsed.get("import_error"):
        return {"total": int(n_expected), "passed": 0, "tests": tests, "error": parsed["import_error"]}
    total = max(len(tests), int(n_expected))
    return {"total": total, "passed": sum(1 for t in tests if t.get("passed")), "tests": tests,
            "error": None if len(tests) >= int(n_expected) else "some tests did not run"}
