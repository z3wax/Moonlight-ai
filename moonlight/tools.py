"""Tool calling for Moonlight chat models (SPEC §8, §13.6). Torch-free.

Main pieces:

* :class:`Tool` - ``Tool(name, description, parameters, fn)`` where ``parameters`` is a JSON schema
  ``{"type": "object", "properties": {...}, "required": [...]}``. ``tool.schema()`` returns the
  ``{"name", "description", "parameters"}`` dict rendered into the ``<|tools|>`` block.
* :func:`tool` - decorator building a :class:`Tool` from a function: the schema comes from the
  type hints (``int -> integer``, ``float -> number``, ``str -> string``, ``bool -> boolean``,
  ``list -> array``, ``dict -> object``; ``Optional[X]`` -> X; ``Literal[...]`` -> ``enum``), the
  description from the docstring summary and per-argument descriptions from its ``Args:`` section.
  Usable as ``@tool`` or ``@tool(name=..., description=...)``.
* :class:`ToolRegistry` - ``register``, ``schemas()``, ``call(name, arguments) -> str``. ``call``
  never raises: results are serialized (strings verbatim, everything else as JSON) and failures
  come back as ``{"error": "..."}`` JSON so the model can read them.
* Built-in tools: :data:`calculator` (safe AST evaluator), :data:`current_time`,
  :data:`unit_convert` and :func:`python_exec` (disabled unless ``allow_code_exec=True``).
* :func:`parse_tool_call` - tolerant extraction of ``{"name": ..., "arguments": {...}}`` from model
  output (raw JSON, JSON in code fences, single-quoted python-dict style, OpenAI style).
* :class:`Workspace` + :func:`workspace_tools` - a directory sandbox with ``read_file``,
  ``write_file``, ``list_dir``, ``run_tests`` and ``run_python`` used by agentic project building
  (``LLM.build_project``).

Security notes: the calculator never calls ``eval``; it walks a whitelisted AST. ``python_exec`` /
``Workspace.run_python`` / ``Workspace.run_tests`` run code in a separate Python *process* with a
timeout (the whole process group is killed on timeout) and, on POSIX, CPU-time / file-size
resource limits. This is NOT a security sandbox: there is no filesystem or network isolation.
Only enable code execution for trusted use.
"""

from __future__ import annotations

import ast
import datetime as _dt
import importlib.util
import inspect
import json
import math
import operator
import os
import re
import signal
import subprocess
import sys
import tempfile
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "Tool",
    "tool",
    "ToolRegistry",
    "ToolError",
    "as_registry",
    "parse_tool_call",
    "safe_eval",
    "calculator",
    "current_time",
    "unit_convert",
    "python_exec",
    "python_exec_tool",
    "builtin_tools",
    "default_registry",
    "BUILTIN_TOOL_NAMES",
    "Workspace",
    "workspace_tools",
    "run_process",
]


class ToolError(Exception):
    """Raised by tools for expected, user-facing failures (message is shown to the model)."""


# =================================================================================================
# JSON-schema inference
# =================================================================================================

_SIMPLE_TYPES: Dict[Any, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    tuple: "array",
    set: "array",
    frozenset: "array",
    dict: "object",
}

_STRING_HINTS: Dict[str, str] = {
    "int": "integer", "float": "number", "str": "string", "bool": "boolean",
    "list": "array", "tuple": "array", "set": "array", "dict": "object",
    "List": "array", "Tuple": "array", "Sequence": "array", "Set": "array", "Iterable": "array",
    "Dict": "object", "Mapping": "object", "MutableMapping": "object",
}


def _json_type_of_value(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    return "string"


def _schema_for_hint(hint: Any) -> Dict[str, Any]:
    """Python type hint -> JSON schema fragment (unknown / missing hints -> ``string``)."""
    if hint is inspect.Parameter.empty or hint is None or hint is Any:
        return {"type": "string"}
    if isinstance(hint, str):  # unresolved forward reference: best effort on the outer name
        outer = hint.split("[", 1)[0].strip().split(".")[-1]
        inner = hint[len(hint.split("[", 1)[0]):].strip("[] ")
        if outer == "Optional" and inner:
            return _schema_for_hint(inner)
        return {"type": _STRING_HINTS.get(outer, "string")}
    if hint in _SIMPLE_TYPES:
        return {"type": _SIMPLE_TYPES[hint]}
    origin = typing.get_origin(hint)
    args = [a for a in typing.get_args(hint)]
    if origin is typing.Union or (origin is not None and getattr(origin, "__name__", "") == "UnionType"):
        non_none = [a for a in args if a is not type(None)]
        return _schema_for_hint(non_none[0]) if len(non_none) >= 1 else {"type": "string"}
    if origin is typing.Literal:
        values = list(args)
        return {"type": _json_type_of_value(values[0]) if values else "string", "enum": values}
    if origin in (list, tuple, set, frozenset) or (
            origin is not None and getattr(origin, "__name__", "") in ("Sequence", "Iterable", "MutableSequence")):
        schema: Dict[str, Any] = {"type": "array"}
        item_args = [a for a in args if a is not Ellipsis]
        if item_args:
            schema["items"] = _schema_for_hint(item_args[0])
        return schema
    if origin in (dict,) or (origin is not None and getattr(origin, "__name__", "") in ("Mapping", "MutableMapping")):
        return {"type": "object"}
    return {"type": "string"}


_SECTION_RE = re.compile(r"^\s*(Args|Arguments|Parameters|Params|Returns|Return|Yields|Raises|Examples?|Notes?)\s*:\s*$")
_ARG_LINE_RE = re.compile(r"^\s*\*{0,2}([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")


def _parse_docstring(doc: Optional[str]) -> Tuple[str, Dict[str, str]]:
    """Google-style docstring -> (summary description, {arg name: description}).

    The summary is every line before the first section header (whitespace-joined). Argument
    entries look like ``name: text`` or ``name (type): text`` under ``Args:`` with optional
    indented continuation lines.
    """
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    summary: List[str] = []
    args: Dict[str, str] = {}
    section: Optional[str] = None
    current: Optional[str] = None
    arg_indent: Optional[int] = None
    for line in lines:
        header = _SECTION_RE.match(line)
        if header:
            section = header.group(1).lower()
            current, arg_indent = None, None
            continue
        if section is None:
            summary.append(line.strip())
            continue
        if section not in ("args", "arguments", "parameters", "params"):
            continue
        if not line.strip():
            current = None
            continue
        indent = len(line) - len(line.lstrip())
        m = _ARG_LINE_RE.match(line)
        if m and (arg_indent is None or indent <= arg_indent):
            arg_indent = indent
            current = m.group(1)
            args[current] = m.group(3).strip()
        elif current is not None:
            args[current] = (args[current] + " " + line.strip()).strip()
    text = " ".join(s for s in summary if s)
    return re.sub(r"\s+", " ", text).strip(), args


def _infer_parameters(fn: Callable[..., Any], arg_docs: Mapping[str, str]) -> Dict[str, Any]:
    """Build the ``parameters`` JSON schema of ``fn`` from its signature and type hints."""
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # unresolvable annotations: fall back to the raw (string) annotations
        hints = {}
    props: Dict[str, Any] = {}
    required: List[str] = []
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in ("self", "cls"):
            continue
        hint = hints.get(name, p.annotation)
        schema = _schema_for_hint(hint)
        if arg_docs.get(name):
            schema["description"] = arg_docs[name]
        if p.default is not inspect.Parameter.empty:
            if p.default is None or isinstance(p.default, (bool, int, float, str)):
                if p.default is not None:
                    schema["default"] = p.default
        else:
            required.append(name)
        props[name] = schema
    out: Dict[str, Any] = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return out


# =================================================================================================
# Tool / decorator
# =================================================================================================

@dataclass
class Tool:
    """A callable tool exposed to the model.

    Attributes:
        name: identifier the model uses in ``{"name": ..., "arguments": {...}}``.
        description: one-paragraph description shown in the ``<|tools|>`` block.
        parameters: JSON schema of the keyword arguments (``type: object``).
        fn: the implementation, called as ``fn(**arguments)``; None for schema-only tools
            (calling one reports an error to the model).
    """

    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    fn: Optional[Callable[..., Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tool.name must be a non-empty string")
        self.name = self.name.strip()
        self.description = (self.description or "").strip()
        if not isinstance(self.parameters, Mapping):
            raise TypeError(f"Tool {self.name!r}: parameters must be a JSON-schema dict")
        params = dict(self.parameters)
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        self.parameters = params
        if self.fn is not None and not callable(self.fn):
            raise TypeError(f"Tool {self.name!r}: fn must be callable")

    def schema(self) -> Dict[str, Any]:
        """``{"name", "description", "parameters"}`` for the chat template's tools block."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    to_schema = schema

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.fn is None:
            raise ToolError(f"tool {self.name!r} has no implementation")
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        props = ", ".join(self.parameters.get("properties", {}))
        return f"Tool({self.name}({props}))"


def _make_tool(fn: Callable[..., Any], name: Optional[str], description: Optional[str]) -> Tool:
    if isinstance(fn, Tool):
        if name is None and description is None:
            return fn
        return Tool(name or fn.name, description if description is not None else fn.description,
                    fn.parameters, fn.fn)
    if not callable(fn):
        raise TypeError(f"@tool expects a function, got {type(fn).__name__}")
    summary, arg_docs = _parse_docstring(inspect.getdoc(fn))
    tool_name = name or getattr(fn, "__name__", None)
    if not tool_name or tool_name == "<lambda>":
        raise ValueError("cannot infer a tool name (pass tool(name=...))")
    return Tool(name=tool_name, description=description if description is not None else summary,
                parameters=_infer_parameters(fn, arg_docs), fn=fn)


def tool(name: Union[str, Callable[..., Any], None] = None, description: Optional[str] = None) -> Any:
    """Decorator turning a function into a :class:`Tool` (schema inferred, see module docstring).

    ``@tool`` / ``@tool()`` / ``@tool(name="weather", description="...")``. The decorated object is
    the :class:`Tool` (still callable like the original function).
    """
    if callable(name) and description is None:  # bare @tool
        return _make_tool(name, None, None)
    if name is not None and not isinstance(name, str):
        raise TypeError("tool(name=...) must be a string")

    def deco(fn: Callable[..., Any]) -> Tool:
        return _make_tool(fn, name, description)

    return deco


# =================================================================================================
# Registry
# =================================================================================================

def _dump(value: Any) -> str:
    """Tool result -> text for the ``<|tool_result|>`` message (str verbatim, else JSON)."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


_BOOL_TRUE = {"true", "yes", "1", "on", "ha"}
_BOOL_FALSE = {"false", "no", "0", "off", "yo'q", "yoq"}


def _coerce(value: Any, schema: Mapping[str, Any]) -> Any:
    """Lenient coercion of a model-produced argument to its declared JSON type (best effort;
    values that cannot be coerced are passed through unchanged)."""
    t = schema.get("type") if isinstance(schema, Mapping) else None
    try:
        if t == "integer":
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str) and re.fullmatch(r"\s*[-+]?\d+\s*", value):
                return int(value)
            if isinstance(value, str) and re.fullmatch(r"\s*[-+]?\d+\.0*\s*", value):
                return int(float(value))
        elif t == "number":
            if isinstance(value, str) and value.strip():
                return float(value.strip().replace("_", ""))
        elif t == "boolean":
            if isinstance(value, str):
                low = value.strip().lower()
                if low in _BOOL_TRUE:
                    return True
                if low in _BOOL_FALSE:
                    return False
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return bool(value)
        elif t == "string":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
        elif t in ("array", "object") and isinstance(value, str):
            parsed = json.loads(value)
            if (t == "array" and isinstance(parsed, list)) or (t == "object" and isinstance(parsed, dict)):
                return parsed
    except (ValueError, TypeError):
        pass
    return value


class ToolRegistry:
    """Named collection of tools with JSON-in / text-out calling.

    ``ToolRegistry(tools=None)``; ``tools`` may contain :class:`Tool` objects, plain functions
    (wrapped with :func:`tool`), schema dicts (with an optional ``fn``/``function`` callable) or
    built-in tool names (``"calculator"`` ...).
    """

    def __init__(self, tools: Optional[Iterable[Any]] = None, allow_code_exec: bool = False) -> None:
        self._tools: Dict[str, Tool] = {}
        self.allow_code_exec = bool(allow_code_exec)
        for t in tools or ():
            self.register(t)

    # ------------------------------------------------------------------ registration
    def register(self, obj: Any, name: Optional[str] = None, description: Optional[str] = None,
                 replace: bool = True) -> Tool:
        """Add a tool (see class docstring for accepted objects); returns the stored :class:`Tool`.

        Can also be used as a decorator: ``@registry.register``.
        """
        if isinstance(obj, str):
            t = _builtin_by_name(obj, self.allow_code_exec)
            if name or description is not None:
                t = Tool(name or t.name, description if description is not None else t.description,
                         t.parameters, t.fn)
        elif isinstance(obj, Tool):
            t = obj if (name is None and description is None) else _make_tool(obj, name, description)
        elif isinstance(obj, Mapping):
            spec = dict(obj)
            fn = spec.get("fn") or spec.get("callable")
            src: Mapping[str, Any] = spec
            if isinstance(spec.get("function"), Mapping):  # OpenAI style {"type": "function", "function": {...}}
                src = spec["function"]
            elif callable(spec.get("function")):
                fn = fn or spec["function"]
            t = Tool(name=name or str(src.get("name", "")),
                     description=description if description is not None else str(src.get("description", "")),
                     parameters=dict(src.get("parameters") or {"type": "object", "properties": {}}),
                     fn=fn if callable(fn) else None)
        elif callable(obj):
            t = _make_tool(obj, name, description)
        else:
            raise TypeError(f"cannot register {type(obj).__name__} as a tool")
        if not replace and t.name in self._tools:
            raise ValueError(f"tool {t.name!r} is already registered")
        self._tools[t.name] = t
        return t

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ------------------------------------------------------------------ access
    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    @property
    def tools(self) -> List[Tool]:
        return list(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(list(self._tools.values()))

    def __bool__(self) -> bool:
        return bool(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({', '.join(self._tools)})"

    def schemas(self) -> List[Dict[str, Any]]:
        """JSON schemas of all tools (registration order) for ``encode_chat(tools=...)``."""
        return [t.schema() for t in self._tools.values()]

    # ------------------------------------------------------------------ calling
    def call(self, name: str, arguments: Any = None) -> str:
        """Run tool ``name`` with ``arguments`` (dict, JSON string or None) and return text.

        Never raises for tool-side problems: unknown tools, invalid / missing arguments and
        exceptions inside the tool are returned as ``{"error": "..."}`` JSON.
        """
        t = self._tools.get(str(name)) if name is not None else None
        if t is None:
            avail = ", ".join(self._tools) or "none"
            return _error(f"unknown tool {name!r}; available: {avail}")
        if t.fn is None:
            return _error(f"tool {t.name!r} is not executable here")
        if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
            arguments = {}
        if isinstance(arguments, str):
            parsed = _loads_lenient(arguments.strip())
            if parsed is None:
                props = list(t.parameters.get("properties", {}))
                if len(props) != 1:
                    return _error("arguments are not valid JSON")
                parsed = {props[0]: arguments}  # bare value for a one-parameter tool
            arguments = parsed
        if not isinstance(arguments, Mapping):
            # a single positional value for a one-parameter tool is accepted
            props = list(t.parameters.get("properties", {}))
            if len(props) == 1:
                arguments = {props[0]: arguments}
            else:
                return _error("arguments must be a JSON object")
        props = t.parameters.get("properties", {}) or {}
        kwargs: Dict[str, Any] = {}
        unknown = []
        for k, v in arguments.items():
            k = str(k)
            if props and k not in props:
                unknown.append(k)
                continue
            kwargs[k] = _coerce(v, props.get(k, {}))
        accepts_var_kw = False
        if t.fn is not None:
            try:
                accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                                     for p in inspect.signature(t.fn).parameters.values())
            except (TypeError, ValueError):
                accepts_var_kw = True
        if unknown and not accepts_var_kw:
            return _error(f"unknown argument(s) for {t.name}: {', '.join(unknown)}; "
                          f"expected: {', '.join(props) or 'none'}")
        missing = [r for r in t.parameters.get("required", []) or [] if r not in kwargs]
        if missing:
            return _error(f"missing required argument(s) for {t.name}: {', '.join(missing)}")
        try:
            result = t.fn(**kwargs)
        except ToolError as e:
            return _error(str(e))
        except (ValueError, TypeError, ArithmeticError, KeyError, LookupError, OSError, PermissionError) as e:
            return _error(f"{type(e).__name__}: {e}")
        except Exception as e:  # tools must never crash the chat loop
            return _error(f"{type(e).__name__}: {e}")
        return _dump(result)

    def __call__(self, name: str, arguments: Any = None) -> str:
        return self.call(name, arguments)


def as_registry(tools: Any, allow_code_exec: bool = False) -> Optional[ToolRegistry]:
    """Normalize the ``tools=`` argument of chat APIs.

    None / False / empty -> None (no tools); True -> :func:`default_registry`; a
    :class:`ToolRegistry` -> itself; a Tool / function / schema dict / built-in name or a list of
    those -> a new registry.
    """
    if tools is None or tools is False:
        return None
    if tools is True:
        return default_registry(allow_code_exec=allow_code_exec)
    if isinstance(tools, ToolRegistry):
        return tools if len(tools) else None
    if isinstance(tools, (Tool, Mapping, str)) or callable(tools):
        tools = [tools]
    reg = ToolRegistry(tools, allow_code_exec=allow_code_exec)
    return reg if len(reg) else None


# =================================================================================================
# Tool-call parsing
# =================================================================================================

_TOOL_CALL_MARKERS = ("<|tool_call|>", "<functioncall>", "<function_call>", "<tool_call>", "</tool_call>",
                      "<|end|>", "<|eos|>")
_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)


def _balanced_objects(text: str, limit: int = 32) -> Iterator[str]:
    """Yield substrings ``{...}`` with balanced braces (string-literal aware, ' and ") in order
    of their opening brace. An unterminated final object is yielded closed with the missing
    braces (models often stop one ``}`` short)."""
    n = len(text)
    starts = [i for i, c in enumerate(text) if c == "{"][:limit]
    for s in starts:
        depth = 0
        quote: Optional[str] = None
        escape = False
        end = None
        for i in range(s, n):
            c = text[i]
            if quote is not None:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == quote:
                    quote = None
                continue
            if c in ("'", '"'):
                quote = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is not None:
            yield text[s:end + 1]
        elif quote is None and 0 < depth <= 3:
            yield text[s:] + "}" * depth


_PY_LITERALS = (("true", "True"), ("false", "False"), ("null", "None"))


def _loads_lenient(s: str) -> Any:
    """JSON, then python-literal (single quotes, True/None) with JSON keywords mapped."""
    try:
        return json.loads(s)
    except ValueError:
        pass
    try:
        return ast.literal_eval(s)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    fixed = s
    for js, py in _PY_LITERALS:
        fixed = re.sub(rf"(?<![A-Za-z0-9_'\"]){js}(?![A-Za-z0-9_'\"])", py, fixed)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)  # trailing commas
    try:
        return ast.literal_eval(fixed)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def _normalize_call(obj: Any) -> Optional[Dict[str, Any]]:
    """Accept our ``{"name", "arguments"}``, OpenAI ``{"function": {"name", "arguments"}}`` and
    ``{"tool"|"tool_name", "args"|"parameters"|"input"}`` variants."""
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, Mapping):
        return None
    if isinstance(obj.get("function"), Mapping):
        obj = obj["function"]
    elif isinstance(obj.get("tool_call"), Mapping):
        obj = obj["tool_call"]
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name") or obj.get("function")
    if not isinstance(name, str) or not name.strip():
        return None
    args: Any = None
    for key in ("arguments", "args", "parameters", "input", "params"):
        if key in obj:
            args = obj[key]
            break
    if args is None:
        args = {}
    if isinstance(args, str):
        stripped = args.strip()
        if not stripped:
            args = {}
        else:
            parsed = _loads_lenient(stripped)
            args = parsed if isinstance(parsed, Mapping) else {"input": args}
    if not isinstance(args, Mapping):
        return None
    return {"name": name.strip(), "arguments": dict(args)}


def parse_tool_call(text: Any) -> Optional[Dict[str, Any]]:
    """Extract one tool call ``{"name": str, "arguments": dict}`` from model output, or None.

    Tolerates: leading ``<|tool_call|>`` / ``<functioncall>`` markers and a trailing ``<|end|>``,
    surrounding prose, ```` ```json ```` code fences, single-quoted python-dict syntax
    (``ast.literal_eval`` fallback), JSON ``true/false/null`` inside python syntax, a missing
    final ``}``, ``arguments`` given as a JSON string, and OpenAI-style ``{"function": {...}}``.
    A dict input is normalized directly.
    """
    if text is None:
        return None
    if isinstance(text, Mapping):
        return _normalize_call(text)
    if not isinstance(text, str):
        text = str(text)
    s = text
    for marker in _TOOL_CALL_MARKERS:
        s = s.replace(marker, " ")
    s = s.strip()
    if not s:
        return None
    candidates: List[str] = [s]
    candidates += [m.group(1).strip() for m in _FENCE_RE.finditer(s)]
    seen = set()
    for cand in candidates:
        for sub in [cand] + list(_balanced_objects(cand)):
            if sub in seen:
                continue
            seen.add(sub)
            if not sub.lstrip().startswith(("{", "[")):
                continue
            call = _normalize_call(_loads_lenient(sub))
            if call is not None:
                return call
    return None


# =================================================================================================
# Safe calculator
# =================================================================================================

MAX_EXPONENT = 10_000
MAX_MAGNITUDE = 1e100
MAX_EXPRESSION_CHARS = 500
MAX_AST_NODES = 200
MAX_FACTORIAL = 170

_CONSTANTS: Dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau, "π": math.pi}


def _factorial(x: Any) -> int:
    if isinstance(x, float):
        if not x.is_integer():
            raise ValueError("factorial() needs an integer")
        x = int(x)
    if x < 0:
        raise ValueError("factorial() of a negative number")
    if x > MAX_FACTORIAL:
        raise ValueError(f"factorial() argument too large (max {MAX_FACTORIAL})")
    return math.factorial(x)


def _round(x: Any, ndigits: Any = None) -> Any:
    if ndigits is None:
        return round(x)
    if not isinstance(ndigits, int) or not -20 <= ndigits <= 20:
        raise ValueError("round() ndigits must be an integer in [-20, 20]")
    return round(x, ndigits)


_FUNCTIONS: Dict[str, Tuple[Callable[..., Any], int, int]] = {
    # name: (function, min args, max args)
    "sqrt": (math.sqrt, 1, 1), "sin": (math.sin, 1, 1), "cos": (math.cos, 1, 1), "tan": (math.tan, 1, 1),
    "asin": (math.asin, 1, 1), "acos": (math.acos, 1, 1), "atan": (math.atan, 1, 1),
    "sinh": (math.sinh, 1, 1), "cosh": (math.cosh, 1, 1), "tanh": (math.tanh, 1, 1),
    "log": (math.log, 1, 2), "ln": (math.log, 1, 1), "log2": (math.log2, 1, 1), "log10": (math.log10, 1, 1),
    "exp": (math.exp, 1, 1), "abs": (abs, 1, 1), "round": (_round, 1, 2), "floor": (math.floor, 1, 1),
    "ceil": (math.ceil, 1, 1), "factorial": (_factorial, 1, 1), "degrees": (math.degrees, 1, 1),
    "radians": (math.radians, 1, 1), "min": (min, 1, 16), "max": (max, 1, 16), "hypot": (math.hypot, 1, 16),
}

_BINOPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARYOPS: Dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_EXPR_REPLACEMENTS = (("×", "*"), ("·", "*"), ("÷", "/"), ("−", "-"), ("–", "-"), ("^", "**"), ("√", "sqrt"))


def _check_magnitude(v: Any) -> Any:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("only real numbers are supported")
    if isinstance(v, float):
        if math.isnan(v):
            raise ValueError("result is not a number")
        if math.isinf(v) or abs(v) > MAX_MAGNITUDE:
            raise ValueError(f"result magnitude exceeds {MAX_MAGNITUDE:g}")
    elif abs(v) > 10 ** 100:
        raise ValueError(f"result magnitude exceeds {MAX_MAGNITUDE:g}")
    return v


def _safe_pow(base: Any, exp: Any) -> Any:
    if abs(exp) > MAX_EXPONENT:
        raise ValueError(f"exponent too large (|exponent| <= {MAX_EXPONENT})")
    if base != 0 and abs(base) != 1:
        # log10|result| = exp * log10|base|; refuse before computing a huge integer
        est = float(exp) * math.log10(abs(base))
        if est > 100.5:
            raise ValueError(f"result magnitude exceeds {MAX_MAGNITUDE:g}")
    if base == 0 and exp < 0:
        raise ZeroDivisionError("0 cannot be raised to a negative power")
    result = operator.pow(base, exp)
    if isinstance(result, complex):
        raise ValueError("result is a complex number")
    return result


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"unsupported literal {v!r}")
        return _check_magnitude(v)
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            return _check_magnitude(_safe_pow(left, right))
        return _check_magnitude(op(left, right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator {type(node.op).__name__}")
        return _check_magnitude(op(_eval_node(node.operand)))
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"unknown name {node.id!r}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            fname = node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
            raise ValueError(f"function {fname!r} is not allowed")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        fn, lo, hi = _FUNCTIONS[node.func.id]
        if not lo <= len(node.args) <= hi:
            raise ValueError(f"{node.func.id}() takes {lo}..{hi} arguments")
        args = [_eval_node(a) for a in node.args]
        return _check_magnitude(fn(*args))
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def _tidy_number(v: Any) -> Any:
    """Integral floats -> int (when exact), other floats rounded to 12 significant digits."""
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return int(v)
        if v != 0:
            digits = 12 - int(math.floor(math.log10(abs(v)))) - 1
            v = round(v, max(-300, min(300, digits)))
        return v
    return v


def safe_eval(expression: str) -> Union[int, float]:
    """Evaluate an arithmetic expression without ``eval`` (whitelisted AST walk).

    Supported: int/float literals, ``+ - * / // % **`` (``^`` is read as power, ``×``/``÷`` as
    ``*``/``/``), parentheses, unary ``+``/``-``, constants ``pi e tau`` and the functions
    ``sqrt sin cos tan asin acos atan sinh cosh tanh log ln log2 log10 exp abs round floor ceil
    factorial degrees radians min max hypot``. Rejected: names / attributes / subscripts /
    comprehensions / strings / keyword arguments, exponents with ``|exp| > 10000``, any
    (intermediate) value with magnitude ``> 1e100``, expressions over 500 characters or 200 AST
    nodes. Raises ``ValueError`` / ``ZeroDivisionError``.
    """
    if not isinstance(expression, str):
        expression = str(expression)
    expr = expression.strip()
    if not expr:
        raise ValueError("empty expression")
    if len(expr) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression too long (max {MAX_EXPRESSION_CHARS} characters)")
    for a, b in _EXPR_REPLACEMENTS:
        expr = expr.replace(a, b)
    expr = re.sub(r"(?<=\d)_(?=\d)", "", expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid expression: {e.msg}") from None
    n_nodes = sum(1 for _ in ast.walk(tree))
    if n_nodes > MAX_AST_NODES:
        raise ValueError(f"expression too complex (max {MAX_AST_NODES} nodes)")
    try:
        result = _eval_node(tree)
    except ZeroDivisionError:
        raise ZeroDivisionError("division by zero") from None
    except OverflowError:
        raise ValueError(f"result magnitude exceeds {MAX_MAGNITUDE:g}") from None
    return _tidy_number(result)


@tool(name="calculator")
def calculator(expression: str) -> Union[int, float]:
    """Evaluate an arithmetic expression exactly and safely (+, -, *, /, //, %, **, parentheses,
    sqrt, sin, cos, log, exp, pi, e).

    Args:
        expression: the expression to evaluate, e.g. "(12 + 7) * 3" or "sqrt(16) + 2**10".
    """
    return safe_eval(expression)


# =================================================================================================
# current_time
# =================================================================================================

_TZ_ALIASES = {
    "tashkent": "Asia/Tashkent", "toshkent": "Asia/Tashkent", "uzbekistan": "Asia/Tashkent",
    "o'zbekiston": "Asia/Tashkent", "samarkand": "Asia/Samarkand", "samarqand": "Asia/Samarkand",
    "london": "Europe/London", "moscow": "Europe/Moscow", "moskva": "Europe/Moscow",
    "new york": "America/New_York", "tokyo": "Asia/Tokyo", "berlin": "Europe/Berlin",
    "paris": "Europe/Paris", "dubai": "Asia/Dubai", "istanbul": "Europe/Istanbul",
    "beijing": "Asia/Shanghai", "seoul": "Asia/Seoul", "delhi": "Asia/Kolkata",
}
# zones without DST, usable when the system has no tz database (e.g. Windows without tzdata)
_FIXED_OFFSETS = {"Asia/Tashkent": 5, "Asia/Samarkand": 5, "Asia/Tokyo": 9, "Asia/Dubai": 4,
                  "Asia/Shanghai": 8, "Asia/Seoul": 9, "Asia/Kolkata": 5.5, "Europe/Moscow": 3,
                  "Europe/Istanbul": 3}
_OFFSET_RE = re.compile(r"^(?:UTC|GMT)?\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)


def _resolve_timezone(name: str) -> Tuple[_dt.tzinfo, str]:
    raw = (name or "UTC").strip()
    low = raw.lower()
    if low in ("utc", "gmt", "z", "zulu", ""):
        return _dt.timezone.utc, "UTC"
    m = _OFFSET_RE.match(raw)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        hours, minutes = int(m.group(2)), int(m.group(3) or 0)
        if hours > 14 or minutes >= 60:
            raise ValueError(f"invalid UTC offset {raw!r}")
        delta = _dt.timedelta(hours=hours, minutes=minutes) * sign
        return _dt.timezone(delta), f"UTC{m.group(1)}{hours:02d}:{minutes:02d}"
    zone = _TZ_ALIASES.get(low, raw)
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(zone), zone
    except Exception:
        if zone in _FIXED_OFFSETS:
            return _dt.timezone(_dt.timedelta(hours=_FIXED_OFFSETS[zone])), zone
        raise ValueError(f"unknown timezone {raw!r} (use an IANA name like 'Asia/Tashkent' or 'UTC+5')") from None


@tool(name="current_time")
def current_time(timezone: str = "UTC") -> Dict[str, str]:
    """Get the current date and time in a timezone.

    Args:
        timezone: IANA timezone name (e.g. "Asia/Tashkent", "Europe/London"), "UTC", or an offset
            like "UTC+5".
    """
    tz, label = _resolve_timezone(timezone)
    now = _dt.datetime.now(tz)
    offset = now.strftime("%z")
    offset = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
    return {"timezone": label, "datetime": now.isoformat(timespec="seconds"),
            "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A"), "utc_offset": offset}


# =================================================================================================
# unit_convert
# =================================================================================================

# category -> canonical unit -> factor to the category's base unit (metre / kilogram)
_LINEAR_UNITS: Dict[str, Dict[str, float]] = {
    "length": {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "um": 1e-6, "nm": 1e-9,
               "mi": 1609.344, "yd": 0.9144, "ft": 0.3048, "in": 0.0254, "nmi": 1852.0},
    "mass": {"kg": 1.0, "g": 0.001, "mg": 1e-6, "t": 1000.0, "lb": 0.45359237, "oz": 0.028349523125,
             "st": 6.35029318},
}
_UNIT_ALIASES: Dict[str, str] = {
    # length
    "meter": "m", "meters": "m", "metre": "m", "metres": "m", "metr": "m",
    "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km", "kilometr": "km",
    "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "santimetr": "cm", "sm": "cm",
    "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetr": "mm",
    "micrometer": "um", "micron": "um", "µm": "um", "nanometer": "nm",
    "mile": "mi", "miles": "mi", "mil": "mi", "yard": "yd", "yards": "yd",
    "foot": "ft", "feet": "ft", "fut": "ft", "inch": "in", "inches": "in", "dyuym": "in",
    "nautical mile": "nmi", "nautical miles": "nmi",
    # mass
    "kilogram": "kg", "kilograms": "kg", "kilogramm": "kg", "kilo": "kg",
    "gram": "g", "grams": "g", "gramm": "g", "milligram": "mg", "milligrams": "mg", "milligramm": "mg",
    "tonne": "t", "tonnes": "t", "ton": "t", "tons": "t", "tonna": "t",
    "pound": "lb", "pounds": "lb", "lbs": "lb", "funt": "lb", "ounce": "oz", "ounces": "oz",
    "stone": "st",
    # temperature
    "c": "C", "°c": "C", "celsius": "C", "selsiy": "C", "centigrade": "C", "degc": "C",
    "f": "F", "°f": "F", "fahrenheit": "F", "farengeyt": "F", "degf": "F",
    "k": "K", "kelvin": "K", "kelvins": "K",
}
_TEMPERATURE = ("C", "F", "K")


def _canonical_unit(unit: str) -> Tuple[str, str]:
    """-> (category, canonical unit) or ValueError."""
    raw = str(unit).strip()
    key = raw.lower().replace("degrees ", "").replace("degree ", "").strip()
    canon = _UNIT_ALIASES.get(key)
    if canon is None:
        canon = raw if raw in _TEMPERATURE else key
    if canon in _TEMPERATURE:
        return "temperature", canon
    for cat, table in _LINEAR_UNITS.items():
        if canon in table:
            return cat, canon
    known = sorted(set(_LINEAR_UNITS["length"]) | set(_LINEAR_UNITS["mass"]) | set(_TEMPERATURE))
    raise ValueError(f"unknown unit {raw!r}; supported: {', '.join(known)}")


def _to_kelvin(v: float, u: str) -> float:
    return v + 273.15 if u == "C" else (v - 32.0) * 5.0 / 9.0 + 273.15 if u == "F" else v


def _from_kelvin(k: float, u: str) -> float:
    return k - 273.15 if u == "C" else (k - 273.15) * 9.0 / 5.0 + 32.0 if u == "F" else k


@tool(name="unit_convert")
def unit_convert(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """Convert a value between units of length (m, km, cm, mm, mi, yd, ft, in), mass (kg, g, mg,
    t, lb, oz) or temperature (C, F, K).

    Args:
        value: the number to convert.
        from_unit: unit of the value, e.g. "km", "lb", "C".
        to_unit: target unit, e.g. "mi", "kg", "F".
    """
    if isinstance(value, str):
        value = float(value.strip())
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("value must be a finite number")
    cat_a, a = _canonical_unit(from_unit)
    cat_b, b = _canonical_unit(to_unit)
    if cat_a != cat_b:
        raise ValueError(f"cannot convert {cat_a} ({a}) to {cat_b} ({b})")
    if cat_a == "temperature":
        kelvin = _to_kelvin(float(value), a)
        if kelvin < -1e-9:
            raise ValueError("temperature below absolute zero")
        result = _from_kelvin(kelvin, b)
    else:
        table = _LINEAR_UNITS[cat_a]
        result = float(value) * table[a] / table[b]
    return {"value": value, "from_unit": a, "to_unit": b, "result": _tidy_number(float(result)),
            "category": cat_a}


# =================================================================================================
# Subprocess helper + python_exec
# =================================================================================================

OUTPUT_LIMIT = 4000


def _trim(text: str, limit: int = OUTPUT_LIMIT) -> str:
    """Keep head and tail of long outputs (the tail usually holds the error / summary)."""
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} characters trimmed] ...\n{text[-tail:]}"


def _limits(cpu_seconds: float) -> Optional[Callable[[], None]]:
    if os.name != "posix":
        return None

    def apply() -> None:  # runs in the child before exec
        try:
            import resource

            cpu = int(max(1, math.ceil(cpu_seconds)))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            fsize = 64 * 2**20
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        except Exception:
            pass

    return apply


def run_process(cmd: Sequence[str], cwd: Union[str, os.PathLike, None] = None, timeout: float = 5.0,
                env: Optional[Mapping[str, str]] = None) -> Tuple[Optional[int], str, str, bool]:
    """Run ``cmd`` with a wall-clock timeout; returns ``(returncode, stdout, stderr, timed_out)``.

    The child runs in its own session/process group (POSIX) so the whole group is killed on
    timeout; CPU-time and file-size rlimits are applied where available. ``returncode`` is None
    when the process timed out.
    """
    kwargs: Dict[str, Any] = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
        pre = _limits(timeout + 1.0)
        if pre is not None:
            kwargs["preexec_fn"] = pre
    proc = subprocess.Popen(list(cmd), cwd=None if cwd is None else str(cwd), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=dict(env) if env is not None else None, **kwargs)
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        out, err = proc.communicate()
    return (None if timed_out else proc.returncode,
            out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace"), timed_out)


def _child_env(extra_path: Optional[str] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_path:
        prev = env.get("PYTHONPATH")
        env["PYTHONPATH"] = extra_path if not prev else extra_path + os.pathsep + prev
    return env


def python_exec(code: str, allow_code_exec: bool = False, timeout: float = 5.0) -> Dict[str, Any]:
    """Run Python ``code`` in a fresh isolated interpreter (``python -I``) inside a temporary
    directory; returns ``{"ok", "stdout", "stderr"}`` (outputs trimmed).

    Disabled unless ``allow_code_exec=True`` (raises ``PermissionError``). Timeout ``timeout``
    seconds (default 5). Network access and filesystem access outside the temp dir are NOT
    prevented - this is process isolation, not a security sandbox.
    """
    if not allow_code_exec:
        raise PermissionError("python_exec is disabled (create the tools with allow_code_exec=True)")
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    with tempfile.TemporaryDirectory(prefix="moonlight_exec_") as tmp:
        script = Path(tmp) / "main.py"
        script.write_text(code, encoding="utf-8")
        rc, out, err, timed_out = run_process([sys.executable, "-I", str(script)], cwd=tmp, timeout=timeout,
                                              env=_child_env())
    if timed_out:
        return {"ok": False, "stdout": _trim(out), "stderr": f"timeout after {timeout:g}s"}
    return {"ok": rc == 0, "stdout": _trim(out), "stderr": _trim(err)}


def python_exec_tool(allow_code_exec: bool = False, timeout: float = 5.0) -> Tool:
    """The ``python_exec`` :class:`Tool` (its calls fail with an error unless enabled)."""

    def _run(code: str) -> Dict[str, Any]:
        return python_exec(code, allow_code_exec=allow_code_exec, timeout=timeout)

    return Tool(name="python_exec",
                description="Run a short Python program and return its stdout/stderr (print the results).",
                parameters={"type": "object",
                            "properties": {"code": {"type": "string", "description": "Python source code to run."}},
                            "required": ["code"]},
                fn=_run)


BUILTIN_TOOL_NAMES = ("calculator", "current_time", "unit_convert", "python_exec")


def _builtin_by_name(name: str, allow_code_exec: bool) -> Tool:
    key = name.strip().lower()
    if key == "calculator":
        return calculator
    if key == "current_time":
        return current_time
    if key == "unit_convert":
        return unit_convert
    if key == "python_exec":
        return python_exec_tool(allow_code_exec)
    raise ValueError(f"unknown built-in tool {name!r}; available: {', '.join(BUILTIN_TOOL_NAMES)}")


def builtin_tools(allow_code_exec: bool = False) -> List[Tool]:
    """``[calculator, current_time, unit_convert]`` (+ ``python_exec`` when ``allow_code_exec``)."""
    tools = [calculator, current_time, unit_convert]
    if allow_code_exec:
        tools.append(python_exec_tool(True))
    return tools


def default_registry(allow_code_exec: bool = False) -> ToolRegistry:
    """A :class:`ToolRegistry` with the built-in tools."""
    return ToolRegistry(builtin_tools(allow_code_exec), allow_code_exec=allow_code_exec)


# =================================================================================================
# Workspace (agentic project building, SPEC §13.6)
# =================================================================================================

_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".mypy_cache", ".ruff_cache", ".venv", "venv"}
MAX_READ_CHARS = 64_000
MAX_WRITE_CHARS = 1_000_000


class Workspace:
    """A directory sandbox for project-building agents.

    ``Workspace(root, allow_exec=True, timeout=20)``. Every path argument is interpreted relative
    to ``root``; absolute paths are only accepted when they point inside ``root`` and any path
    that resolves outside it (``..`` escapes, symlinks) is rejected with ``PermissionError``.
    ``allow_exec=False`` disables :meth:`run_tests` / :meth:`run_python`.

    Attributes updated as tools run: ``written`` (relative paths written, in order),
    ``last_test`` (the latest :meth:`run_tests` result dict or None).
    """

    def __init__(self, root: Union[str, os.PathLike], allow_exec: bool = True, timeout: float = 20) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.allow_exec = bool(allow_exec)
        self.timeout = float(timeout)
        self.written: List[str] = []
        self.last_test: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return f"Workspace({str(self.root)!r}, allow_exec={self.allow_exec})"

    # ------------------------------------------------------------------ paths
    def resolve(self, path: Union[str, os.PathLike] = ".") -> Path:
        """Path inside the workspace (absolute, resolved) or ``PermissionError`` if it escapes."""
        raw = os.fspath(path) if path is not None else "."
        if not isinstance(raw, str):
            raise TypeError("path must be a string")
        if "\x00" in raw:
            raise ValueError("path contains a NUL byte")
        raw = raw.strip() or "."
        p = Path(raw).expanduser()
        target = (p if p.is_absolute() else self.root / p).resolve()
        if target != self.root and self.root not in target.parents:
            raise PermissionError(f"path {raw!r} is outside the workspace")
        return target

    def relpath(self, path: Union[str, os.PathLike]) -> str:
        rel = self.resolve(path).relative_to(self.root)
        return rel.as_posix() if str(rel) != "." else "."

    def files(self) -> List[str]:
        """All files under root (relative POSIX paths, sorted; caches and VCS dirs skipped)."""
        out: List[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for f in filenames:
                if f.endswith((".pyc", ".pyo")):
                    continue
                out.append((Path(dirpath) / f).relative_to(self.root).as_posix())
        return sorted(out)

    def test_files(self) -> List[str]:
        return [f for f in self.files() if f.endswith(".py") and
                (Path(f).name.startswith("test_") or Path(f).name.endswith("_test.py"))]

    # ------------------------------------------------------------------ tools
    def read_file(self, path: str) -> str:
        """Text of a file in the workspace (long files are truncated with a note)."""
        p = self.resolve(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + f"\n... [truncated, {len(text) - MAX_READ_CHARS} more characters]"
        return text

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a text file (parent directories are created)."""
        if not isinstance(content, str):
            content = _dump(content)
        if len(content) > MAX_WRITE_CHARS:
            raise ToolError(f"content too large (max {MAX_WRITE_CHARS} characters)")
        p = self.resolve(path)
        if p == self.root or p.is_dir():
            raise ToolError(f"{path} is a directory")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        rel = p.relative_to(self.root).as_posix()
        if rel not in self.written:
            self.written.append(rel)
        n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"wrote {rel} ({len(content)} characters, {n_lines} lines)"

    def list_dir(self, path: str = ".") -> List[str]:
        """Entries of a directory (sub-directories end with "/")."""
        p = self.resolve(path)
        if not p.is_dir():
            raise ToolError(f"not a directory: {path}")
        entries = []
        for child in sorted(p.iterdir(), key=lambda c: c.name):
            if child.name in _SKIP_DIRS or child.name.endswith((".pyc", ".pyo")):
                continue
            entries.append(child.name + ("/" if child.is_dir() else ""))
        return entries

    def run_tests(self, target: str = ".") -> Dict[str, Any]:
        """Run the test suite (``python -m pytest -q`` when pytest is importable, else unittest
        discovery) in a subprocess with ``cwd=root`` and the workspace timeout.

        Returns ``{"passed": bool, "returncode": int|None, "runner": "pytest"|"unittest",
        "output": trimmed combined output}``. "no tests ran" counts as not passed.
        """
        if not self.allow_exec:
            raise ToolError("running code is disabled in this workspace")
        tgt = self.resolve(target)
        rel = tgt.relative_to(self.root).as_posix() if tgt != self.root else "."
        env = _child_env(str(self.root))
        if importlib.util.find_spec("pytest") is not None:
            runner = "pytest"
            cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(self.root), rel]
        else:
            runner = "unittest"
            if tgt.is_file():
                module = ".".join(Path(rel).with_suffix("").parts)
                cmd = [sys.executable, "-m", "unittest", "-q", module]
            else:
                cmd = [sys.executable, "-m", "unittest", "discover", "-q", "-s", rel, "-t", ".", "-p", "*test*.py"]
        rc, out, err, timed_out = run_process(cmd, cwd=self.root, timeout=self.timeout, env=env)
        text = (out + ("\n" + err if err.strip() else "")).strip()
        if timed_out:
            text = (text + f"\nTIMEOUT: tests did not finish within {self.timeout:g}s").strip()
        passed = (not timed_out) and rc == 0
        if runner == "unittest" and passed and re.search(r"Ran 0 tests", text):
            passed = False
        result = {"passed": bool(passed), "returncode": rc, "runner": runner, "output": _trim(text)}
        self.last_test = result
        return result

    def run_python(self, code: str) -> Dict[str, Any]:
        """Run a Python snippet with ``cwd=root`` (workspace modules importable); returns
        ``{"ok", "stdout", "stderr"}``."""
        if not self.allow_exec:
            raise ToolError("running code is disabled in this workspace")
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        rc, out, err, timed_out = run_process([sys.executable, "-c", code], cwd=self.root, timeout=self.timeout,
                                              env=_child_env(str(self.root)))
        if timed_out:
            return {"ok": False, "stdout": _trim(out), "stderr": f"timeout after {self.timeout:g}s"}
        return {"ok": rc == 0, "stdout": _trim(out), "stderr": _trim(err)}


def workspace_tools(ws: Workspace) -> List[Tool]:
    """Tools bound to ``ws``: ``read_file``, ``write_file``, ``list_dir`` and, when
    ``ws.allow_exec``, ``run_tests`` and ``run_python``."""

    def read_file(path: str) -> str:
        """Read a text file from the project directory.

        Args:
            path: file path relative to the project root, e.g. "calc.py".
        """
        return ws.read_file(path)

    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file in the project directory.

        Args:
            path: file path relative to the project root, e.g. "test_calc.py".
            content: the complete new file content.
        """
        return ws.write_file(path, content)

    def list_dir(path: str = ".") -> List[str]:
        """List the files and folders of a project directory.

        Args:
            path: directory relative to the project root (default ".").
        """
        return ws.list_dir(path)

    def run_tests(target: str = ".") -> Dict[str, Any]:
        """Run the project's tests (pytest) and report whether they passed, with the output.

        Args:
            target: test file or directory relative to the project root (default ".").
        """
        return ws.run_tests(target)

    def run_python(code: str) -> Dict[str, Any]:
        """Run a short Python snippet inside the project directory and return stdout/stderr.

        Args:
            code: Python source code to run.
        """
        return ws.run_python(code)

    fns: List[Callable[..., Any]] = [read_file, write_file, list_dir]
    if ws.allow_exec:
        fns += [run_tests, run_python]
    return [tool(fn) for fn in fns]
