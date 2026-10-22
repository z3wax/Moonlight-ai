"""Tools (SPEC §8 + §13.6): safe calculator, registry, parsing, decorator, workspace sandbox. Torch-free."""

from __future__ import annotations

import json
import math
import sys

import pytest

from moonlight import tools as T
from moonlight.tools import Tool, ToolRegistry, Workspace, calculator, parse_tool_call, tool, workspace_tools


def _value(out):
    """Registry results are text: JSON for structured values, strings verbatim."""
    assert isinstance(out, str)
    try:
        return json.loads(out)
    except ValueError:
        return out


def _disabled(fn, *args, **kw):
    """True if a disabled capability refuses to run (raises, or returns a not-ok / error result)."""
    try:
        res = fn(*args, **kw)
    except Exception:  # PermissionError, ToolError, ...
        return True
    return (not res.get("ok", False) and not res.get("passed", False)) if isinstance(res, dict) else False


# ================================================================================================
# calculator
# ================================================================================================

@pytest.mark.parametrize("expr, expected", [
    ("2+3*4", 14), ("(2+3)*4", 20), ("2**10", 1024), ("10//3", 3), ("7%3", 1), ("-(3)", -3), ("+5", 5),
    ("7/2", 3.5), ("sqrt(16)", 4), ("sin(0)", 0), ("cos(0)", 1), ("log(e)", 1), ("exp(0)", 1),
    ("2*pi", 2 * math.pi), ("1e3 + 1", 1001), ("3 × 4", 12),
])
def test_calculator_values(expr, expected):
    assert calculator(expr) == pytest.approx(expected, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hacked')",
    "open('/etc/passwd').read()",
    "().__class__.__bases__[0].__subclasses__()",
    "a.b",
    "[1, 2, 3]",
    "(lambda: 1)()",
    "x",
    "2**999999999",
    "9" * 600,
    "exec('1')",
    "eval('1+1')",
    "1 if True else 2",
    "f'{1}'",
])
def test_calculator_rejects_unsafe(expr):
    with pytest.raises((ValueError, SyntaxError, OverflowError, TypeError)):
        calculator(expr)


def test_calculator_never_runs_code(tmp_path):
    marker = tmp_path / "pwned"
    reg = T.default_registry()
    out = reg.call("calculator", {"expression": f"__import__('pathlib').Path({str(marker)!r}).touch()"})
    assert "error" in json.loads(out)
    assert not marker.exists()


def test_division_by_zero_is_reported_not_raised():
    out = json.loads(T.default_registry().call("calculator", {"expression": "1/0"}))
    assert "error" in out


# ================================================================================================
# other built-ins
# ================================================================================================

def test_unit_convert_and_current_time():
    assert T.unit_convert(1, "km", "m")["result"] == pytest.approx(1000)
    assert T.unit_convert(100, "C", "F")["result"] == pytest.approx(212)
    assert T.unit_convert(1, "kg", "g")["result"] == pytest.approx(1000)
    with pytest.raises((ValueError, KeyError)):
        T.unit_convert(1, "kg", "m")
    now = T.current_time("UTC")
    assert isinstance(now, dict) and "time" in json.dumps(now)
    assert T.current_time("Asia/Tashkent")


def test_python_exec_disabled_by_default():
    reg = T.default_registry()
    assert "python_exec" not in reg.names()
    assert set(reg.names()) >= {"calculator", "current_time", "unit_convert"}
    assert _disabled(T.python_exec, "print(1)")
    out = _value(T.ToolRegistry([T.python_exec_tool()]).call("python_exec", {"code": "print(1)"}))
    assert isinstance(out, dict) and "error" in out


def test_python_exec_enabled():
    reg = T.default_registry(allow_code_exec=True)
    assert "python_exec" in reg.names()
    out = json.loads(reg.call("python_exec", {"code": "print(6 * 7)"}))
    assert "42" in json.dumps(out)
    slow = T.python_exec("while True: pass", allow_code_exec=True, timeout=1.0)
    assert not slow.get("ok", True)


# ================================================================================================
# registry, decorator, schemas
# ================================================================================================

def test_decorator_infers_schema():
    @tool
    def add(a: int, b: int = 2) -> int:
        """Add two numbers.

        Args:
            a: first number
            b: second number
        """
        return a + b

    assert isinstance(add, Tool) and add.name == "add"
    assert add.description.startswith("Add two numbers")
    params = add.parameters
    assert params["type"] == "object" and params["required"] == ["a"]
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["a"]["description"] == "first number"
    assert add(1) == 3

    @tool(name="greet", description="Say hi")
    def g(name: str, loud: bool = False) -> str:
        return ("HI " if loud else "hi ") + name

    assert g.name == "greet" and g.description == "Say hi"
    assert g.parameters["properties"]["loud"]["type"] == "boolean"


def test_registry_call_and_errors():
    @tool
    def mul(a: float, b: float) -> float:
        """Multiply."""
        return a * b

    reg = ToolRegistry([mul])
    reg.register(lambda text: text[::-1], name="rev", description="reverse text")
    names = [s["name"] for s in reg.schemas()]
    assert names == ["mul", "rev"]
    assert json.loads(reg.call("mul", {"a": 3, "b": "4"})) == 12
    assert json.loads(reg.call("mul", '{"a": 2, "b": 5}')) == 10
    assert _value(reg.call("rev", {"text": "abc"})) == "cba"
    for bad_call in (("nope", {}), ("mul", {"a": 1}), ("mul", "not json")):
        out = json.loads(reg.call(*bad_call))
        assert isinstance(out, dict) and "error" in out


def test_registry_results_are_json():
    out = T.default_registry().call("unit_convert", {"value": 5, "from_unit": "m", "to_unit": "cm"})
    assert json.loads(out)["result"] == pytest.approx(500)


# ================================================================================================
# tool-call parsing
# ================================================================================================

@pytest.mark.parametrize("text", [
    '<|tool_call|>{"name": "calculator", "arguments": {"expression": "2+2"}}',
    '{"name": "calculator", "arguments": {"expression": "2+2"}}<|end|>',
    'Sure! ```json\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n```',
    '<|tool_call|>{"name": "calculator", "arguments": "{\\"expression\\": \\"2+2\\"}"}',
    "<|tool_call|>{'name': 'calculator', 'arguments': {'expression': '2+2'}}",
    '<functioncall> {"name": "calculator", "arguments": \'{"expression": "2+2"}\'}',
])
def test_parse_tool_call_tolerant(text):
    assert parse_tool_call(text) == {"name": "calculator", "arguments": {"expression": "2+2"}}


@pytest.mark.parametrize("text", ["", "Salom! Qalaysiz?", "{not json at all", '{"foo": 1}', None])
def test_parse_tool_call_none(text):
    assert parse_tool_call(text) is None


# ================================================================================================
# Workspace (SPEC §13.6)
# ================================================================================================

def test_workspace_file_ops(tmp_path):
    ws = Workspace(tmp_path / "proj")
    ws.write_file("pkg/mod.py", "def f():\n    return 1\n")
    assert ws.read_file("pkg/mod.py") == "def f():\n    return 1\n"
    assert any(e.rstrip("/") == "pkg" for e in ws.list_dir("."))
    assert any("mod.py" in e for e in ws.list_dir("pkg"))
    with pytest.raises(Exception):
        ws.read_file("missing.py")


@pytest.mark.parametrize("path", ["../escape.py", "a/../../escape.py", "/etc/passwd", "/tmp/evil.py"])
def test_workspace_rejects_escapes(tmp_path, path):
    root = tmp_path / "proj"
    ws = Workspace(root)
    with pytest.raises((PermissionError, ValueError)):
        ws.write_file(path, "x = 1\n")
    assert not (tmp_path / "escape.py").exists()
    reg = ToolRegistry(workspace_tools(ws))
    assert "error" in json.loads(reg.call("read_file", {"path": path}))


def test_workspace_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "proj"
    root.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported")
    ws = Workspace(root)
    with pytest.raises((PermissionError, ValueError)):
        ws.write_file("link/evil.py", "x = 1\n")
    assert not (outside / "evil.py").exists()


def test_workspace_run_tests_pass_and_fail(tmp_path):
    ws = Workspace(tmp_path / "proj", timeout=60)
    ws.write_file("calc.py", "def add(a, b):\n    return a + b\n")
    ws.write_file("test_calc.py", "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    ok = ws.run_tests()
    assert ok["passed"] is True and isinstance(ok["output"], str)
    ws.write_file("calc.py", "def add(a, b):\n    return a - b\n")
    bad = ws.run_tests()
    assert bad["passed"] is False and bad["output"]


def test_workspace_run_python_and_exec_switch(tmp_path):
    ws = Workspace(tmp_path / "p")
    res = ws.run_python("print(sum(range(5)))")
    assert res["ok"] and res["stdout"].strip() == "10"
    off = Workspace(tmp_path / "q", allow_exec=False)
    off.write_file("test_x.py", "def test_x():\n    assert True\n")
    assert _disabled(off.run_tests)
    assert _disabled(off.run_python, "print(1)")


def test_workspace_tools_registry(tmp_path):
    ws = Workspace(tmp_path / "w")
    tools = workspace_tools(ws)
    assert {t.name for t in tools} == {"read_file", "write_file", "list_dir", "run_tests", "run_python"}
    reg = ToolRegistry(tools)
    assert "error" not in str(_value(reg.call("write_file", {"path": "hello.py", "content": "print('salom')\n"})))
    assert _value(reg.call("read_file", {"path": "hello.py"})) == "print('salom')\n"
    assert "hello.py" in json.dumps(_value(reg.call("list_dir", {})))
    assert _value(reg.call("run_tests", {}))
    schemas = {s["name"]: s for s in reg.schemas()}
    assert schemas["write_file"]["parameters"]["required"] == ["path", "content"]


def test_workspace_timeout(tmp_path):
    ws = Workspace(tmp_path / "t", timeout=2)
    res = ws.run_python("import time\ntime.sleep(30)")
    assert not res.get("ok")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX paths")
def test_workspace_root_is_created(tmp_path):
    ws = Workspace(tmp_path / "new" / "dir")
    ws.write_file("a.txt", "x")
    assert (tmp_path / "new" / "dir" / "a.txt").read_text() == "x"
