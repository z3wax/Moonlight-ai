"""Moonlight configuration schema (SPEC §2): dataclasses, YAML load/save, overrides, validation.

Torch-free by design (SPEC import hygiene): only the standard library, PyYAML and
``moonlight.utils`` are imported at module level; ``moonlight.registry`` (for ``extends:`` /
registry names), ``moonlight.params`` (nominal-size check) and ``moonlight.benchmarks.suites``
(``cfg.bnchmrks``) are imported lazily.

Overview::

    cfg = Config.from_yaml("configs/test/ml-20m-chat.yaml")   # or Config.from_dict(d)
    cfg.apply_overrides({"train.batch_size": 16, "model.attn_window": "null"})
    cfg.validate()                    # ValueError listing every problem
    text = cfg.to_yaml()              # clean, stable, sort_keys=False, SPEC section order
    suite = cfg.bnchmrks.type("cc-001")

Parsing rules (all dataclass fields have defaults; YAML may omit any key):

* Unknown keys raise ``ValueError`` naming the full dotted path (``model.moe.n_expert``) plus a
  "did you mean" suggestion. All problems of one load are reported together.
* Typed scalar fields are coerced: ``int`` accepts ints, integral floats (``2.0e8``) and numeric
  strings incl. scientific notation and K/M/B/T suffixes (``"2.5e8"``, ``"50M"``); ``float`` accepts
  numbers and numeric strings; ``bool`` accepts ``true/false/yes/no/on/off/1/0``; ``Optional`` fields
  accept ``null/none/~``; list fields accept YAML text (``"[uz, en]"``) or comma separated strings.
* YAML is read with a SafeLoader variant that resolves YAML-1.2 floats: PyYAML (YAML 1.1) returns
  ``2.5e8`` / ``1e-4`` as STRINGS, here they are floats. Free-form sections (``train.phases``,
  ``data.pretrain``/``sft``/``vision``/``synthetic``) additionally coerce scientific-notation strings
  (``"2.5e8"``) to floats when built from already-parsed dicts.
* Free-form mappings (each phase, each data-source entry, ``data.synthetic``) are stored as
  :class:`AttrDict` - a plain ``dict`` subclass that also allows attribute access
  (``phase["name"] == phase.name``).
* ``extends: <registry-name | path.yaml>`` deep-merges the file over its parent (mappings merge
  recursively; lists and scalars are replaced). Relative paths resolve against the child file.
"""

from __future__ import annotations

import copy
import dataclasses
import difflib
import math
import numbers
import os
import re
import typing
import warnings
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import yaml

from moonlight.utils import human_int

__all__ = [
    "Config",
    "ModelConfig",
    "MoEConfig",
    "MultimodalConfig",
    "TokenizerConfig",
    "DataConfig",
    "TrainConfig",
    "OptimizerConfig",
    "ScheduleConfig",
    "CurriculumConfig",
    "DistributedConfig",
    "ChatConfig",
    "BenchmarkAccessor",
    "AttrDict",
    "parse_size",
    "format_size",
    "load_yaml",
    "dump_yaml",
    "deep_merge",
    "TIERS",
    "KINDS",
    "PRECISIONS",
    "OPTIMIZERS",
    "SCHEDULES",
    "STRATEGIES",
    "SECTION_ORDER",
]

TIERS: Tuple[str, ...] = ("test", "mini", "si")
KINDS: Tuple[str, ...] = ("base", "chat", "code", "vl", "vl-chat", "tools", "agent", "think", "long",
                          "debug")
PRECISIONS: Tuple[str, ...] = ("auto", "bf16", "fp16", "fp32")
OPTIMIZERS: Tuple[str, ...] = ("muon", "adamw")
SCHEDULES: Tuple[str, ...] = ("wsd", "cosine", "constant")
STRATEGIES: Tuple[str, ...] = ("none", "ddp", "fsdp")
#: Top-level key order used by ``to_dict`` / ``to_yaml`` (== ``Config`` field order).
SECTION_ORDER: Tuple[str, ...] = ("name", "description", "tier", "kind", "benchmark", "nominal_params",
                                  "model", "multimodal", "tokenizer", "data", "train", "chat")

#: Fraction by which the estimated total may deviate from ``nominal_params`` (SPEC §2).
NOMINAL_TOLERANCE = 0.25

# Same text as ``moonlight.tokenizer.DEFAULT_SYSTEM_PROMPT`` (SPEC §5); duplicated so this module
# stays independent of the tokenizer module.
_DEFAULT_SYSTEM_PROMPT = (
    "Sen Moonlight — o'zbek va ingliz tillarida samimiy, aniq va foydali javob beradigan, kod yoza "
    "oladigan va kerak bo'lganda vositalardan (tools) foydalanadigan sun'iy intellekt yordamchisisan."
)


# ================================================================================================
# Sizes
# ================================================================================================

_SIZE_RE = re.compile(
    r"^(?P<num>(?:[0-9][0-9_,]*(?:\.[0-9_]*)?|\.[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)?)\s*(?P<unit>[kKmMbBgGtT]?)$"
)
_SIZE_MULT = {"": 1, "k": 10**3, "m": 10**6, "b": 10**9, "g": 10**9, "t": 10**12}


def _parse_decimal(text: str) -> Decimal:
    """Parse a non-negative number with optional K/M/B(G)/T suffix into an exact ``Decimal``."""
    m = _SIZE_RE.match(text.strip())
    if not m:
        raise ValueError(f"not a number/size: {text!r}")
    try:
        num = Decimal(m.group("num").replace("_", "").replace(",", ""))
    except InvalidOperation as exc:  # pragma: no cover - regex already guarantees the syntax
        raise ValueError(f"not a number/size: {text!r}") from exc
    return num * _SIZE_MULT[m.group("unit").lower()]


def parse_size(value: Union[str, int, float]) -> int:
    """Parse a human size label into an int (decimal units).

    ``"20M" -> 20_000_000``, ``"1.5B" -> 1_500_000_000``, ``"1T" -> 10**12``, ``"512K"``,
    ``"2.5e8"``, ``"20_000_000"``, ``20_000_000`` (int) and floats are accepted (case-insensitive,
    ``G`` == ``B``). Fractional results are rounded to the nearest integer. Raises ``ValueError``
    for negative, non-finite or malformed input.
    """
    if isinstance(value, bool):
        raise ValueError(f"not a size: {value!r}")
    if isinstance(value, numbers.Integral):
        n = int(value)
    elif isinstance(value, numbers.Real):
        f = float(value)
        if not math.isfinite(f):
            raise ValueError(f"not a size: {value!r}")
        n = int(round(f))
    elif isinstance(value, str):
        n = int(_parse_decimal(value).to_integral_value(rounding=ROUND_HALF_EVEN))
    else:
        raise ValueError(f"not a size: {value!r}")
    if n < 0:
        raise ValueError(f"size must be non-negative: {value!r}")
    return n


def format_size(n: Union[int, float], precision: int = 1) -> str:
    """Human label for a count: ``20_132_736 -> "20.1M"``, ``20_000_000 -> "20M"``,
    ``1.5e9 -> "1.5B"``, ``999 -> "999"`` (inverse of :func:`parse_size` up to rounding)."""
    return human_int(n, precision)


# ================================================================================================
# YAML I/O
# ================================================================================================

_FLOAT_TAG = "tag:yaml.org,2002:float"
# YAML-1.1 float regex from PyYAML extended with YAML-1.2 style exponents: the exponent sign is
# optional and a mantissa without "." is allowed when an exponent is present (2.5e8, 1e-4, 3E5).
_FLOAT_RE = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)[eE][-+]?[0-9]+
    |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.X,
)
# Strings PyYAML would have returned for scientific notation (used for free-form sections).
_SCI_RE = re.compile(r"^[-+]?(?:[0-9][0-9_]*\.?[0-9_]*|\.[0-9][0-9_]*)[eE][-+]?[0-9]+$")


def _patched_resolvers(base: Any) -> Dict[str, list]:
    """Copy of ``base.yaml_implicit_resolvers`` with the float regex swapped in place (order kept)."""
    return {
        ch: [(tag, _FLOAT_RE if tag == _FLOAT_TAG else rx) for tag, rx in resolvers]
        for ch, resolvers in base.yaml_implicit_resolvers.items()
    }


class _Loader(yaml.SafeLoader):
    """SafeLoader that resolves ``2.5e8`` / ``1e-4`` as floats (YAML 1.2 behaviour)."""


_Loader.yaml_implicit_resolvers = _patched_resolvers(yaml.SafeLoader)


class AttrDict(dict):
    """A plain ``dict`` that also supports attribute access (``d.key`` == ``d["key"]``).

    Used for free-form config mappings (training phases, data-source entries, ``data.synthetic``)
    so consumers may use either style. Missing attributes raise ``AttributeError``. It pickles,
    deep-copies, JSON-serialises and (thanks to registered representers) ``yaml.safe_dump``-s
    like a normal dict.
    """

    __slots__ = ()

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __dir__(self) -> List[str]:
        return list(super().__dir__()) + [k for k in self.keys() if isinstance(k, str)]

    def copy(self) -> "AttrDict":
        return AttrDict(self)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "AttrDict":
        return AttrDict((copy.deepcopy(k, memo), copy.deepcopy(v, memo)) for k, v in self.items())

    def __repr__(self) -> str:
        return f"AttrDict({dict.__repr__(self)})"


# Let plain yaml.dump / yaml.safe_dump serialise AttrDict like a dict (benign global registration).
yaml.SafeDumper.add_representer(AttrDict, yaml.SafeDumper.represent_dict)
yaml.Dumper.add_representer(AttrDict, yaml.Dumper.represent_dict)


class _FlowList(list):
    """Rendered as an inline ``[a, b]`` sequence by ``_Dumper``."""


class _FlowDict(dict):
    """Rendered as an inline ``{k: v, ...}`` mapping by ``_Dumper``."""


class _Dumper(yaml.SafeDumper):
    """SafeDumper with indented block sequences, compact floats and quoting consistent with
    ``_Loader`` (strings such as ``"1e5"`` are quoted because they would load as floats)."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


_Dumper.yaml_implicit_resolvers = _patched_resolvers(yaml.SafeDumper)


def _float_text(value: float) -> str:
    """YAML text for a float that both PyYAML's SafeLoader and ``_Loader`` read back exactly.

    Large/small magnitudes use the shortest round-tripping scientific form with a "." in the
    mantissa and a signed exponent (``2.0e+8``, ``2.5e+8``, ``1.0e-5``); others use ``repr``.
    """
    if value != value:
        return ".nan"
    if value in (math.inf, -math.inf):
        return ".inf" if value > 0 else "-.inf"
    if value != 0.0 and (abs(value) >= 1e6 or abs(value) < 1e-4):
        text = f"{value:.16e}"
        for digits in range(17):
            candidate = f"{value:.{digits}e}"
            if float(candidate) == value:
                text = candidate
                break
        mantissa, exponent = text.split("e")
        if "." not in mantissa:
            mantissa += ".0"
        exp = int(exponent)
        return f"{mantissa}e{'+' if exp >= 0 else '-'}{abs(exp)}"
    text = repr(value)  # 1e-4 <= |v| < 1e6 (or 0): repr never uses an exponent here
    return text if "." in text else text + ".0"


_Dumper.add_representer(float, lambda d, v: d.represent_scalar(_FLOAT_TAG, _float_text(v)))
_Dumper.add_representer(AttrDict, yaml.SafeDumper.represent_dict)
_Dumper.add_representer(
    _FlowList, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True))
_Dumper.add_representer(
    _FlowDict, lambda d, v: d.represent_mapping("tag:yaml.org,2002:map", v, flow_style=True))
_Dumper.add_representer(tuple, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", list(v)))


def load_yaml(source: Union[str, os.PathLike]) -> Any:
    """Parse YAML from a file path (or from YAML text if ``source`` contains a newline) with the
    YAML-1.2-float-aware safe loader. Returns the raw Python object (``{}`` for an empty doc)."""
    if isinstance(source, str) and "\n" in source:
        text = source
    else:
        text = Path(source).expanduser().read_text(encoding="utf-8")
    data = yaml.load(text, Loader=_Loader)
    return {} if data is None else data


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, bool, int, float))


def _styled(obj: Any) -> Any:
    """Mark collections for human-friendly YAML: scalar lists inline, flat mappings inside lists
    inline (one line per phase / data source), everything else block style."""
    if isinstance(obj, Mapping):
        return {k: _styled(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if all(_is_scalar(x) for x in obj):
            return _FlowList(obj)
        out = []
        for item in obj:
            if isinstance(item, Mapping) and all(
                    _is_scalar(v) or (isinstance(v, (list, tuple)) and all(_is_scalar(x) for x in v))
                    for v in item.values()):
                out.append(_FlowDict((k, list(v) if isinstance(v, tuple) else v) for k, v in item.items()))
            else:
                out.append(_styled(item))
        return out
    return obj


def dump_yaml(data: Mapping[str, Any]) -> str:
    """Render a (plain) mapping as clean YAML: insertion order kept, unicode kept, no line wrapping,
    indented block lists, flat list items inline, blank line before each nested top-level section."""
    chunks: List[str] = []
    for key, value in data.items():
        text = yaml.dump({key: _styled(value)}, Dumper=_Dumper, sort_keys=False, allow_unicode=True,
                         default_flow_style=False, width=4096, indent=2)
        if isinstance(value, Mapping) and value and chunks:
            chunks.append("\n")
        chunks.append(text)
    return "".join(chunks)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``base`` deep-merged with ``override`` (override wins). Mappings merge recursively;
    lists and scalars are replaced. Inputs are not modified."""
    out: Dict[str, Any] = {k: copy.deepcopy(v) for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# ================================================================================================
# Strict dataclass construction & coercion
# ================================================================================================

_INVALID = object()
_NoneType = type(None)
_TRUE = {"true", "yes", "y", "on", "1"}
_FALSE = {"false", "no", "n", "off", "0"}
_NULL = {"null", "none", "~", ""}
_HINTS: Dict[type, Dict[str, Any]] = {}


def _hints(cls: type) -> Dict[str, Any]:
    hints = _HINTS.get(cls)
    if hints is None:
        hints = typing.get_type_hints(cls)
        _HINTS[cls] = hints
    return hints


def _join(path: str, key: Any) -> str:
    return f"{path}.{key}" if path else str(key)


def _type_name(tp: Any) -> str:
    if isinstance(tp, type):
        return tp.__name__
    return str(tp).replace("typing.", "")


def _parse_text(text: str) -> Any:
    """YAML-parse an override/CLI string (``"16"`` -> 16, ``"[a, b]"`` -> list); on YAML syntax
    errors the original string is returned."""
    try:
        value = yaml.load(text, Loader=_Loader)
    except yaml.YAMLError:
        return text
    return value


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in _NULL)


def _free(value: Any) -> Any:
    """Normalise free-form data: mappings -> AttrDict, tuples -> lists, ``"2.5e8"`` -> 2.5e8."""
    if isinstance(value, Mapping):
        return AttrDict((k, _free(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return [_free(v) for v in value]
    if isinstance(value, str) and _SCI_RE.match(value.strip()):
        return float(value.strip().replace("_", ""))
    return value


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        f = float(value)
        if math.isfinite(f) and f.is_integer():
            return int(f)
        raise ValueError
    if isinstance(value, str):
        s = value.strip()
        negative = s.startswith("-")
        if s and s[0] in "+-":
            s = s[1:]
        d = _parse_decimal(s)
        if d != d.to_integral_value():
            raise ValueError
        return -int(d) if negative else int(d)
    raise ValueError


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError
    if isinstance(value, numbers.Real):
        f = float(value)
    elif isinstance(value, str):
        f = float(value.strip().replace("_", ""))
    else:
        raise ValueError
    if not math.isfinite(f):
        raise ValueError
    return f


def _coerce(value: Any, tp: Any, path: str, errors: List[str]) -> Any:
    """Coerce ``value`` to annotation ``tp``; on failure append a message and return ``_INVALID``."""
    if tp is Any:
        return _free(value)
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is Union:
        options = [a for a in args if a is not _NoneType]
        if len(options) < len(args) and _is_null(value):
            return None
        if len(options) == 1:
            return _coerce(value, options[0], path, errors)
        if set(options) == {str, int}:
            if isinstance(value, bool):
                pass
            elif isinstance(value, numbers.Integral):
                return int(value)
            elif isinstance(value, numbers.Real) and math.isfinite(value) and float(value).is_integer():
                return int(value)
            elif isinstance(value, str):
                return value.strip()
            errors.append(f"{path}: expected a string or an integer, got {value!r}")
            return _INVALID
        for option in options:  # generic fallback: first option that coerces cleanly
            trial: List[str] = []
            out = _coerce(value, option, path, trial)
            if not trial:
                return out
        errors.append(f"{path}: cannot interpret {value!r} as {_type_name(tp)}")
        return _INVALID

    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        if isinstance(value, str):
            value = _parse_text(value)
        return _build(tp, value, path, errors)

    if origin in (list, List) or tp is list:
        item_tp = args[0] if args else Any
        if isinstance(value, str):
            parsed = _parse_text(value)
            if isinstance(parsed, list):
                value = parsed
            elif _is_null(value):
                value = []
            else:
                value = [part.strip() for part in value.split(",") if part.strip()]
        if value is None:
            return []
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            errors.append(f"{path}: expected a list, got {type(value).__name__} {value!r}")
            return _INVALID
        out_list = []
        for i, item in enumerate(value):
            item_out = _coerce(item, item_tp, f"{path}[{i}]", errors)
            if item_out is not _INVALID:
                out_list.append(item_out)
        return out_list

    if origin in (dict, Dict) or tp is dict:
        if isinstance(value, str):
            value = None if _is_null(value) else _parse_text(value)
        if value is None:
            return AttrDict()
        if not isinstance(value, Mapping):
            errors.append(f"{path}: expected a mapping, got {type(value).__name__} {value!r}")
            return _INVALID
        bad = [k for k in value if not isinstance(k, str)]
        if bad:
            errors.append(f"{path}: mapping keys must be strings, got {bad!r}")
            return _INVALID
        return _free(value)

    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, numbers.Integral) and int(value) in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            return value.strip().lower() in _TRUE
        errors.append(f"{path}: expected a boolean, got {value!r}")
        return _INVALID

    if tp is int:
        try:
            return _to_int(value)
        except ValueError:
            errors.append(f"{path}: expected an integer, got {value!r}")
            return _INVALID

    if tp is float:
        try:
            return _to_float(value)
        except ValueError:
            errors.append(f"{path}: expected a number, got {value!r}")
            return _INVALID

    if tp is str:
        if isinstance(value, str):
            return value
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            return str(value)
        errors.append(f"{path}: expected a string, got {value!r}")
        return _INVALID

    raise TypeError(f"unsupported config annotation {tp!r} at {path}")  # programming error


def _unknown_key_message(path: str, key: Any, known: List[str]) -> str:
    full = _join(path, key)
    match = difflib.get_close_matches(str(key), known, n=1, cutoff=0.6)
    if match:
        return f"unknown key '{full}' (did you mean '{_join(path, match[0])}'?)"
    return f"unknown key '{full}' (valid keys under '{path or '<root>'}': {', '.join(known)})"


def _build(cls: type, data: Any, path: str, errors: List[str]) -> Any:
    """Build dataclass ``cls`` from a mapping, strictly (unknown keys / bad types -> ``errors``)."""
    if data is None:
        return cls()
    if isinstance(data, cls):
        return copy.deepcopy(data)
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        data = _plain(data)
    if not isinstance(data, Mapping):
        errors.append(f"{path or '<root>'}: expected a mapping, got {type(data).__name__} {data!r}")
        return cls()
    hints = _hints(cls)
    known = [f.name for f in dataclasses.fields(cls) if f.init]
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str) or key not in known:
            errors.append(_unknown_key_message(path, key, known))
            continue
        out = _coerce(value, hints[key], _join(path, key), errors)
        if out is not _INVALID:
            kwargs[key] = out
    return cls(**kwargs)


def _plain(value: Any) -> Any:
    """Recursively convert dataclasses / AttrDicts / tuples into plain dicts and lists."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _format_errors(title: str, errors: List[str]) -> str:
    return title + ":\n" + "\n".join(f"  - {e}" for e in errors)


# ================================================================================================
# Section base class
# ================================================================================================


class _Section:
    """Shared behaviour of every config dataclass (strict ``from_dict``, ``to_dict``, YAML, copy)."""

    def __post_init__(self) -> None:
        """Normalise values passed directly to the constructor (not via ``from_dict``):
        mappings given for section fields are built strictly, tuples become lists and free-form
        mappings / lists of mappings become :class:`AttrDict` - so programmatically built configs
        behave (and compare) exactly like loaded ones."""
        hints = _hints(type(self))
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            tp = hints[f.name]
            if isinstance(tp, type) and dataclasses.is_dataclass(tp):
                if isinstance(value, Mapping):
                    object.__setattr__(self, f.name, tp.from_dict(value))
                continue
            origin = typing.get_origin(tp)
            if origin in (dict, Dict) and isinstance(value, Mapping) and type(value) is not AttrDict:
                object.__setattr__(self, f.name, _free(value))
            elif origin in (list, List) and isinstance(value, (list, tuple)):
                item_tp = _item_type(tp)
                if typing.get_origin(item_tp) in (dict, Dict) or item_tp is Any:
                    if any(isinstance(v, Mapping) and type(v) is not AttrDict for v in value) \
                            or isinstance(value, tuple):
                        object.__setattr__(self, f.name, [_free(v) for v in value])
                elif isinstance(value, tuple):
                    object.__setattr__(self, f.name, list(value))

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]] = None):
        """Strictly build this section from a mapping (``ValueError`` lists unknown keys by path)."""
        errors: List[str] = []
        obj = _build(cls, data or {}, "", errors)
        if errors:
            raise ValueError(_format_errors(f"Invalid {cls.__name__}", errors))
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """Plain nested ``dict``/``list`` representation in field (= SPEC) order."""
        return _plain(self)

    def to_yaml(self, path: Union[str, os.PathLike, None] = None) -> str:
        """Render as YAML (see :func:`dump_yaml`); also write it to ``path`` if given."""
        text = dump_yaml(self.to_dict())
        if path is not None:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return text

    def copy(self):
        """Deep copy."""
        return copy.deepcopy(self)

    def validate(self):
        """Raise ``ValueError`` listing every problem; return ``self`` when valid."""
        problems = self._problems("")
        if problems:
            raise ValueError(_format_errors(f"Invalid {type(self).__name__}", problems))
        return self

    def _problems(self, prefix: str) -> List[str]:  # overridden by sections with rules
        return []


class _Checker:
    """Accumulates validation messages with dotted paths."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.problems: List[str] = []

    def add(self, key: str, message: str) -> None:
        self.problems.append(f"{_join(self.prefix, key)}: {message}")

    def check(self, ok: bool, key: str, message: str) -> None:
        if not ok:
            self.add(key, message)

    def positive(self, obj: Any, *keys: str) -> None:
        for key in keys:
            value = getattr(obj, key)
            self.check(value > 0, key, f"must be > 0 (got {value!r})")

    def non_negative(self, obj: Any, *keys: str) -> None:
        for key in keys:
            value = getattr(obj, key)
            self.check(value >= 0, key, f"must be >= 0 (got {value!r})")

    def fraction(self, obj: Any, key: str, upper_open: bool = False) -> None:
        value = getattr(obj, key)
        ok = 0.0 <= value < 1.0 if upper_open else 0.0 <= value <= 1.0
        self.check(ok, key, f"must be in [0, {'1)' if upper_open else '1]'} (got {value!r})")

    def choice(self, obj: Any, key: str, options: Iterable[str]) -> None:
        value = getattr(obj, key)
        options = tuple(options)
        self.check(value in options, key, f"must be one of {', '.join(options)} (got {value!r})")

    def non_empty(self, obj: Any, *keys: str) -> None:
        for key in keys:
            value = getattr(obj, key)
            self.check(isinstance(value, str) and value.strip() != "", key, "must be a non-empty string")


def _is_number(v: Any) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


# ================================================================================================
# Sections (field names, order and defaults follow SPEC §2 exactly)
# ================================================================================================


@dataclass
class MoEConfig(_Section):
    """Mixture-of-experts FFN (shared + routed experts, aux-loss-free balancing)."""

    enabled: bool = False
    n_experts: int = 0
    top_k: int = 0
    n_shared: int = 0
    expert_hidden: int = 0
    first_dense_layers: int = 0      # first N layers use the dense ffn_hidden SwiGLU
    aux_loss_weight: float = 0.001   # sequence-level balance loss
    bias_update_rate: float = 0.001  # aux-loss-free balancing; router bias is a buffer
    capacity_factor: float = 0.0     # 0 = dropless

    def _problems(self, prefix: str) -> List[str]:
        c = _Checker(prefix)
        if self.enabled:
            c.check(self.top_k >= 1, "top_k", f"must be >= 1 when moe.enabled (got {self.top_k})")
            c.check(self.n_experts >= self.top_k, "n_experts",
                    f"must be >= top_k ({self.top_k}) (got {self.n_experts})")
            c.positive(self, "expert_hidden")
            c.non_negative(self, "n_shared", "first_dense_layers", "aux_loss_weight",
                           "bias_update_rate", "capacity_factor")
        return c.problems


@dataclass
class ModelConfig(_Section):
    """Hybrid attention + retention decoder (SPEC §2/§3/§4).

    Head bookkeeping (SPEC §3): ``Hr = round(n_heads * retention_ratio)`` (Python ``round``, i.e.
    round-half-to-even), ``Ha = n_heads - Hr``, ``Hkv = n_kv_heads if Ha > 0 else 0`` - exposed
    as :attr:`n_retention_heads`, :attr:`n_attention_heads`, :attr:`n_attn_kv_heads`.
    """

    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 10
    n_heads: int = 6                       # total query heads (attention + retention)
    head_dim: int = 64
    retention_ratio: float = 0.5           # fraction of heads that are retention heads
    n_kv_heads: int = 1                    # KV heads of the ATTENTION heads
    attn_window: Optional[int] = None      # sliding window for "local" layers; None = all global
    global_every: int = 1                  # layer i is global iff i % N == N - 1
    ffn_hidden: int = 1024                 # dense SwiGLU hidden size
    token_shift: int = 4                   # depthwise causal conv kernel (0 = off)
    value_residual: bool = True
    qk_norm: bool = True
    output_gate: bool = True
    rope_theta: float = 10000.0
    max_seq_len: int = 1024
    tie_embeddings: bool = True
    softcap: float = 30.0                  # final logit soft-cap (0 = off)
    mtp_depth: int = 1                     # number of multi-token-prediction heads (0 = off)
    mtp_block: bool = False                # each MTP head contains a full Block
    mtp_weight: float = 0.3
    z_loss: float = 0.0
    init_std: float = 0.02
    moe: MoEConfig = field(default_factory=MoEConfig)
    activation_checkpointing: bool = False

    # ---- derived helpers (not serialised) ------------------------------------------------------
    @property
    def n_retention_heads(self) -> int:
        """``Hr = round(n_heads * retention_ratio)`` (Python round-half-even, SPEC §2/§3)."""
        return int(round(self.n_heads * self.retention_ratio))

    @property
    def n_attention_heads(self) -> int:
        """``Ha = n_heads - Hr``."""
        return self.n_heads - self.n_retention_heads

    @property
    def n_attn_kv_heads(self) -> int:
        """``Hkv``: ``n_kv_heads`` if there are attention heads, else 0 (KV heads are unused)."""
        return self.n_kv_heads if self.n_attention_heads > 0 else 0

    @property
    def qkv_dim(self) -> int:
        """Output features of ``mixer.qkv``: ``head_dim * (H + 2*Hkv + 2*Hr)``."""
        return self.head_dim * (self.n_heads + 2 * self.n_attn_kv_heads + 2 * self.n_retention_heads)

    def is_global_layer(self, i: int) -> bool:
        """True if layer ``i`` uses full causal attention (``attn_window`` None, or
        ``i % global_every == global_every - 1``)."""
        if self.attn_window is None:
            return True
        n = max(1, self.global_every)
        return i % n == n - 1

    def layer_window(self, i: int) -> Optional[int]:
        """Sliding-window size for layer ``i`` (``None`` for global layers)."""
        return None if self.is_global_layer(i) else self.attn_window

    def is_moe_layer(self, i: int) -> bool:
        """True if layer ``i`` (MTP blocks use ``i = n_layers + j``) has an MoE FFN."""
        return bool(self.moe.enabled) and i >= self.moe.first_dense_layers

    def _problems(self, prefix: str) -> List[str]:
        c = _Checker(prefix)
        c.positive(self, "d_model", "n_layers", "n_heads", "head_dim", "max_seq_len", "rope_theta",
                   "init_std")
        c.check(self.vocab_size >= 32, "vocab_size",
                f"must be >= 32 (the 32 special tokens have fixed ids) (got {self.vocab_size})")
        c.check(0.0 <= self.retention_ratio <= 1.0, "retention_ratio",
                f"must be in [0, 1] (got {self.retention_ratio})")
        n_attn = self.n_attention_heads
        if n_attn > 0:
            if self.n_kv_heads < 1:
                c.add("n_kv_heads", f"must be >= 1 when there are attention heads (got {self.n_kv_heads})")
            elif n_attn % self.n_kv_heads != 0:
                c.add("n_kv_heads", f"must divide the number of attention heads {n_attn} "
                                    f"(= n_heads {self.n_heads} - retention heads "
                                    f"{self.n_retention_heads}) (got {self.n_kv_heads})")
            c.check(self.head_dim % 2 == 0, "head_dim",
                    f"must be even for RoPE when attention heads exist (got {self.head_dim})")
        c.check(self.attn_window is None or self.attn_window >= 1, "attn_window",
                f"must be null or >= 1 (got {self.attn_window})")
        c.check(self.global_every >= 1, "global_every", f"must be >= 1 (got {self.global_every})")
        has_dense = (not self.moe.enabled) or self.moe.first_dense_layers > 0
        if has_dense:
            c.check(self.ffn_hidden >= 1, "ffn_hidden",
                    f"must be >= 1 when dense FFN layers exist (got {self.ffn_hidden})")
        c.non_negative(self, "token_shift", "softcap", "mtp_depth", "mtp_weight", "z_loss")
        if self.moe.enabled and self.moe.first_dense_layers > self.n_layers:
            c.add("moe.first_dense_layers",
                  f"must be <= n_layers ({self.n_layers}) (got {self.moe.first_dense_layers})")
        c.problems.extend(self.moe._problems(_join(prefix, "moe")))
        return c.problems


@dataclass
class MultimodalConfig(_Section):
    """Image input via a frozen vision encoder + trainable projector."""

    enabled: bool = False
    vision_encoder: str = "google/siglip-base-patch16-224"
    vision_dim: int = 768
    n_image_tokens: int = 49
    freeze_encoder: bool = True

    def _problems(self, prefix: str) -> List[str]:
        c = _Checker(prefix)
        if self.enabled:
            c.non_empty(self, "vision_encoder")
            c.positive(self, "vision_dim", "n_image_tokens")
            if self.n_image_tokens > 0:
                root = math.isqrt(self.n_image_tokens)
                c.check(root * root == self.n_image_tokens, "n_image_tokens",
                        "must be a perfect square (the patch grid is pooled to sqrt(n)^2) "
                        f"(got {self.n_image_tokens})")
        return c.problems


@dataclass
class TokenizerConfig(_Section):
    """Byte-level BPE tokenizer settings (SPEC §5)."""

    vocab_size: int = 8192
    path: Optional[str] = None        # existing tokenizer.json; None -> trained during llm.train()
    train_chars: int = 50000000
    split_digits: bool = True

    def _problems(self, prefix: str) -> List[str]:
        c = _Checker(prefix)
        c.positive(self, "vocab_size", "train_chars")
        return c.problems


def _default_pretrain() -> List[Dict[str, Any]]:
    return [AttrDict(source="uz-news", weight=0.25, max_chars=2.5e8)]


def _default_sft() -> List[Dict[str, Any]]:
    return [AttrDict(source="slimorca-uz", max_examples=150000)]


def _default_synthetic() -> Dict[str, Any]:
    return AttrDict(identity=True, knowledge=True, arithmetic=3000, tool_calls=2000, code=1000)


@dataclass
class DataConfig(_Section):
    """Data sources. ``pretrain``: list of ``{source, weight, max_chars}``; ``sft``: list of
    ``{source, max_examples}``; ``vision``: list of ``{source, max_images}``; ``synthetic``:
    generator name -> bool/count. Entries are free-form :class:`AttrDict` (consumers define keys)."""

    cache_dir: str = "~/.cache/moonlight"
    offline: bool = False             # True -> only bundled assets + synthetic data
    languages: List[str] = field(default_factory=lambda: ["uz", "en"])
    pretrain: List[Dict[str, Any]] = field(default_factory=_default_pretrain)
    sft: List[Dict[str, Any]] = field(default_factory=_default_sft)
    synthetic: Dict[str, Any] = field(default_factory=_default_synthetic)
    vision: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def cache_path(self) -> Path:
        """``cache_dir`` with ``~`` and environment variables expanded."""
        return Path(os.path.expandvars(os.path.expanduser(self.cache_dir)))

    def _problems(self, prefix: str) -> List[str]:
        c = _Checker(prefix)
        c.non_empty(self, "cache_dir")
        for i, lang in enumerate(self.languages):
            c.check(isinstance(lang, str) and lang.strip() != "", f"languages[{i}]",
                    f"must be a non-empty string (got {lang!r})")
        for section, count_keys in (("pretrain", ("weight", "max_chars")), ("sft", ("max_examples",)),
                                    ("vision", ("max_images",))):
            for i, entry in enumerate(getattr(self, section)):
                key = f"{section}[{i}]"
                if not isinstance(entry, Mapping):
                    c.add(key, f"must be a mapping with a 'source' key (got {entry!r})")
                    continue
                source = entry.get("source")
                c.check(isinstance(source, str) and source.strip() != "", key,
                        f"needs a non-empty 'source' (got {source!r})")
                for ck in count_keys:
                    if ck in entry and entry[ck] is not None:
                        v = entry[ck]
                        c.check(_is_number(v) and v >= 0, f"{key}.{ck}",
                                f"must be a number >= 0 (got {v!r})")
        for name, v in self.synthetic.items():
            if _is_number(v):
                c.check(v >= 0, f"synthetic.{name}", f"must be >= 0 (got {v!r})")
        return c.problems


@dataclass
class OptimizerConfig(_Section):
    """Muon (matrices) + AdamW (embeddings/norms/scalars) settings (SPEC §7)."""

    name: str = "muon"
    lr: float = 0.02
    momentum: float = 0.95
    adam_lr: float = 0.003
    embed_lr: float = 0.003
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    def _problems(self, prefix: str) -> List[str]:
        c = _Checker(prefix)
        c.choice(self, "name", OPTIMIZERS)
        c.positive(self, "lr", "adam_lr", "embed_lr")
        c.fraction(self, "momentum", upper_open=True)
        c.check(len(self.betas) == 2 and all(0.0 <= b < 1.0 for b in self.betas), "betas",
                f"must be two numbers in [0, 1) (got {self.betas!r})")
        c.non_negative(self, "weight_decay", "grad_clip")
        return c.problems


@dataclass
class ScheduleConfig(_Section):
    """LR schedule: ``wsd`` | ``cosine`` | ``constant`` (warmup by step, decay by progress)."""

    name: str = "wsd"
    warmup_steps: int = 200
    decay_frac: float = 0.35
    min_lr_ratio: float = 0.02

    def _problems(self, prefix: str) -> List[str]:
        c = _Checker(prefix)
        c.choice(self, "name", SCHEDULES)
        c.non_negative(self, "warmup_steps")
        c.fraction(self, "decay_frac")
        c.fraction(self, "min_lr_ratio")
        return c.problems


@dataclass
class CurriculumConfig(_Section):
    """Sequence-length curriculum: grow from ``start_seq_len`` until ``until_frac`` of the phase."""

    enabled: bool = True
    start_seq_len: int = 128
    until_frac: float = 0.3

    def _problems(self, prefix: str) -> List[str]:
        c = _Checker(prefix)
        c.positive(self, "start_seq_len")
        c.fraction(self, "until_frac")
        return c.problems


@dataclass
class DistributedConfig(_Section):
    """Parallelism. ``strategy``: none | ddp | fsdp. ``tp_size``/``ep_size`` are documented
    placeholders (tensor / expert parallelism are NOT implemented, SPEC §12)."""

    strategy: str = "none"
    fsdp_shard: str = "full"
    tp_size: int = 1
    ep_size: int = 1

    def _problems(self, prefix: str) -> List[str]:
        c = _Checker(prefix)
        c.choice(self, "strategy", STRATEGIES)
        c.non_empty(self, "fsdp_shard")
        c.positive(self, "tp_size", "ep_size")
        return c.problems


def _default_phases() -> List[Dict[str, Any]]:
    return [
        AttrDict(name="pretrain", tokens=2.0e8, minutes=70, lr_scale=1.0, sft_mix=0.0, anneal_sft_mix=0.25),
        AttrDict(name="align", minutes=6, lr_scale=1.0, trainable="vision_proj"),
        AttrDict(name="sft", minutes=25, epochs=2.0, lr_scale=0.5, replay=0.1, image_mix=0.3),
    ]


# Phase keys that, when present, must be non-negative numbers (budgets and mixing ratios).
_PHASE_NUMERIC_KEYS = ("tokens", "minutes", "steps", "epochs", "lr_scale", "sft_mix", "anneal_sft_mix",
                       "replay", "image_mix", "seq_len", "batch_size")


@dataclass
class TrainConfig(_Section):
    """Training recipe. ``phases`` is an ordered list of free-form phase dicts (interpreted by
    ``Trainer.run_phase``), e.g. ``{name: pretrain, tokens: 2.0e8, minutes: 70, lr_scale: 1.0}``."""

    phases: List[Dict[str, Any]] = field(default_factory=_default_phases)
    seq_len: int = 512
    batch_size: int = 32              # micro-batch (sequences per device per step)
    batch_tokens: int = 16384         # global tokens per optimizer step
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    precision: str = "auto"           # auto | bf16 | fp16 | fp32
    compile: bool = True
    ema: float = 0.998                # 0 = off
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    eval_every: int = 1000
    eval_batches: int = 20
    log_every: int = 100
    checkpoint_every_min: float = 10.0
    output_dir: str = "runs/ml-20m-chat"
    seed: int = 1337
    hardware: str = "colab-t4"        # informational preset name
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    def _problems(self, prefix: str) -> List[str]:
        c = _Checker(prefix)
        for i, phase in enumerate(self.phases):
            key = f"phases[{i}]"
            if not isinstance(phase, Mapping):
                c.add(key, f"must be a mapping with a 'name' key (got {phase!r})")
                continue
            pname = phase.get("name")
            c.check(isinstance(pname, str) and pname.strip() != "", key,
                    f"needs a non-empty 'name' (got {pname!r})")
            for nk in _PHASE_NUMERIC_KEYS:
                if nk in phase and phase[nk] is not None:
                    v = phase[nk]
                    c.check(_is_number(v) and v >= 0, f"{key}.{nk}", f"must be a number >= 0 (got {v!r})")
        c.positive(self, "seq_len", "batch_size", "batch_tokens")
        c.choice(self, "precision", PRECISIONS)
        c.fraction(self, "ema", upper_open=True)
        c.non_negative(self, "eval_every", "eval_batches", "log_every", "checkpoint_every_min")
        c.non_empty(self, "output_dir")
        c.problems.extend(self.optimizer._problems(_join(prefix, "optimizer")))
        c.problems.extend(self.schedule._problems(_join(prefix, "schedule")))
        c.problems.extend(self.curriculum._problems(_join(prefix, "curriculum")))
        c.problems.extend(self.distributed._problems(_join(prefix, "distributed")))
        return c.problems


@dataclass
class ChatConfig(_Section):
    """Chat persona and default generation settings."""

    name: str = "Moonlight"
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    best_of: int = 3
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    no_repeat_ngram: int = 3
    max_new_tokens: int = 256
    tools: bool = True

    def _problems(self, prefix: str) -> List[str]:
        c = _Checker(prefix)
        c.non_empty(self, "name")
        c.positive(self, "best_of", "repetition_penalty", "max_new_tokens")
        c.non_negative(self, "temperature", "top_k", "no_repeat_ngram")
        c.check(0.0 < self.top_p <= 1.0, "top_p", f"must be in (0, 1] (got {self.top_p})")
        return c.problems


# ================================================================================================
# Benchmarks accessor
# ================================================================================================


class BenchmarkAccessor:
    """``cfg.bnchmrks`` / ``cfg.benchmarks``: lazy access to benchmark suites (SPEC §1/§2).

    ``.type("cc-001")`` returns a ``moonlight.benchmarks.suites.Suite`` (imported lazily);
    ``.default()`` returns the suite named by ``cfg.benchmark``; calling the accessor is a shortcut
    for ``.type`` (``cfg.benchmarks("cc-003")``; no argument -> default suite).
    """

    __slots__ = ("_cfg",)

    def __init__(self, cfg: "Config") -> None:
        self._cfg = cfg

    def type(self, name: Optional[str] = None) -> Any:
        from moonlight.benchmarks.suites import type as _suite_type

        return _suite_type(self._cfg.benchmark if name is None else name)

    def default(self) -> Any:
        return self.type(self._cfg.benchmark)

    def list(self) -> List[str]:
        """Names of all available suites."""
        from moonlight.benchmarks.suites import list_suites

        return list_suites()

    def __call__(self, name: Optional[str] = None) -> Any:
        return self.type(name)

    def __repr__(self) -> str:
        return f"BenchmarkAccessor(config={self._cfg.name!r}, default={self._cfg.benchmark!r})"


# ================================================================================================
# Top-level config
# ================================================================================================


def _resolve_parent(ref: str, base_dir: Optional[Path]) -> Path:
    """Locate an ``extends:`` target: a path (relative to the child's directory, then CWD) or a
    registry name."""
    ref_path = Path(ref).expanduser()
    looks_like_path = ref_path.suffix.lower() in (".yaml", ".yml") or "/" in ref or os.sep in ref
    if looks_like_path:
        candidates = []
        if base_dir is not None and not ref_path.is_absolute():
            candidates.append(base_dir / ref_path)
        candidates.append(ref_path)
        for cand in candidates:
            if cand.is_file():
                return cand.resolve()
    from moonlight.registry import find_config  # lazy: registry imports this module

    return find_config(ref)


def _resolve_extends(data: Dict[str, Any], base_dir: Optional[Path], chain: Tuple[Path, ...]) -> Dict[str, Any]:
    """Recursively apply ``extends:`` (child deep-merged over parent; cycles are an error)."""
    data = dict(data)
    ref = data.pop("extends", None)
    if ref is None:
        return data
    where = f"{chain[-1]}: " if chain else ""
    if not isinstance(ref, (str, os.PathLike)) or not os.fspath(ref).strip():
        raise ValueError(f"{where}'extends' must be a config name or a path, got {ref!r}")
    parent_path = _resolve_parent(os.fspath(ref).strip(), base_dir)
    if parent_path in chain:
        cycle = " -> ".join(str(p) for p in chain + (parent_path,))
        raise ValueError(f"circular 'extends' chain: {cycle}")
    parent = load_yaml(parent_path)
    if not isinstance(parent, Mapping):
        raise ValueError(f"{parent_path}: top level of a config must be a mapping")
    parent = _resolve_extends(dict(parent), parent_path.parent, chain + (parent_path,))
    return deep_merge(parent, data)


@dataclass
class Config(_Section):
    """A complete Moonlight recipe (SPEC §2). Top-level field order == YAML section order."""

    name: str = "ml-20m-chat"         # must equal the file stem for registry configs
    description: str = ""
    tier: str = "test"                # test | mini | si
    kind: str = "chat"                # base | chat | code | vl | vl-chat | tools | agent | think | long | debug
    benchmark: str = "cc-002"         # default benchmark suite
    nominal_params: Union[str, int] = "20M"   # human label; estimator must be within +-25%
    model: ModelConfig = field(default_factory=ModelConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)

    # ---- construction --------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]] = None, base_dir: Union[str, os.PathLike, None] = None
                  ) -> "Config":
        """Strictly build a Config from a (possibly partial) nested mapping.

        Missing keys take defaults; unknown keys / uncoercible values raise one ``ValueError``
        listing every problem with its dotted path. ``extends:`` is resolved (relative paths
        against ``base_dir`` or the CWD, bare names via the registry). When ``data`` itself does
        not set ``train.output_dir`` (not even through ``extends``-inherited values, which are
        ignored for this key) it defaults to ``runs/{name}`` so distinct configs never share a
        run directory / checkpoint. Does not validate.
        """
        if isinstance(data, Config):
            return data.copy()
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ValueError(f"config must be a mapping, got {type(data).__name__}")
        base = Path(base_dir).expanduser().resolve() if base_dir is not None else None
        return cls._from_raw(dict(data), base, ())

    @classmethod
    def _from_raw(cls, data: Dict[str, Any], base_dir: Optional[Path], chain: Tuple[Path, ...]) -> "Config":
        """Resolve ``extends``, apply the ``runs/{name}`` output_dir default, build strictly."""
        train_raw = data.get("train")
        explicit_dir = isinstance(train_raw, Mapping) and train_raw.get("output_dir") not in (None, "")
        merged = _resolve_extends(data, base_dir, chain)
        name = merged.get("name")
        if not explicit_dir and isinstance(name, str) and name.strip():
            train_merged = merged.get("train")
            if train_merged is None or isinstance(train_merged, Mapping):
                merged["train"] = {**(train_merged or {}), "output_dir": f"runs/{name.strip()}"}
        errors: List[str] = []
        cfg = _build(cls, merged, "", errors)
        if errors:
            label = merged.get("name")
            raise ValueError(_format_errors(f"Invalid config {label!r}" if label else "Invalid config", errors))
        return cfg

    @classmethod
    def from_yaml(cls, path: Union[str, os.PathLike, Any]) -> "Config":
        """Load a YAML config from a file path (``extends:`` resolved relative to it).

        Also accepted: YAML text (a ``str`` containing a newline), an open text file object, or a
        registry name (a string that is not an existing file and has no ``.yaml`` suffix). When the
        file has no ``name`` key, the file stem is used.
        """
        if hasattr(path, "read"):
            return cls._from_text(path.read())
        if isinstance(path, str) and "\n" in path:
            return cls._from_text(path)
        p = Path(path).expanduser()
        if not p.is_file():
            if p.suffix.lower() not in (".yaml", ".yml") and isinstance(path, str):
                from moonlight.registry import find_config

                p = find_config(path)
            else:
                raise FileNotFoundError(f"config file not found: {p}")
        p = p.resolve()
        raw = load_yaml(p)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{p}: top level of a config must be a mapping, got {type(raw).__name__}")
        raw = dict(raw)
        if not raw.get("name"):
            raw["name"] = p.stem
        return cls._from_raw(raw, p.parent, (p,))

    @classmethod
    def _from_text(cls, text: str) -> "Config":
        raw = yaml.load(text, Loader=_Loader)
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"top level of a config must be a mapping, got {type(raw).__name__}")
        return cls.from_dict(raw)

    @classmethod
    def load(cls, name_or_path: Union[str, os.PathLike], overrides: Any = None, validate: bool = True) -> "Config":
        """Registry name or path -> Config (same as ``moonlight.registry.load``)."""
        from moonlight.registry import load as _load

        return _load(name_or_path, overrides=overrides, validate=validate)

    # ---- overrides -----------------------------------------------------------------------------
    def apply_overrides(self, overrides: Union[Mapping[str, Any], Iterable[str], str, None]) -> "Config":
        """Apply dotted-key overrides IN PLACE and return ``self``.

        ``overrides`` is a mapping ``{"train.batch_size": 16, "model.attn_window": "null"}`` or an
        iterable of ``"key=value"`` strings (CLI ``--set``). String values are coerced to the
        field type (``"16"`` -> 16, ``"2.5e8"``/``"50M"`` -> int, ``"true"`` -> True,
        ``"null"`` -> None, ``"[uz, en]"`` -> list via YAML). Numeric path segments index lists
        (``train.phases.0.minutes``); keys inside free-form mappings may be new. A mapping value
        on a section/mapping is deep-merged. Unknown keys raise ``ValueError`` with the full path;
        all overrides are checked before anything is modified (atomic).
        """
        items = _override_items(overrides)
        if not items:
            return self
        work = self.copy()
        errors: List[str] = []
        for key, value in items:
            _apply_one(work, key, value, errors)
        if errors:
            raise ValueError(_format_errors(f"Invalid override(s) for config {self.name!r}", errors))
        for f in dataclasses.fields(self):
            setattr(self, f.name, getattr(work, f.name))
        return self

    # ---- validation ----------------------------------------------------------------------------
    def validate(self, strict: bool = False) -> "Config":
        """Check SPEC §2 rules; raise ``ValueError`` listing every problem, else return ``self``.

        Errors: field ranges / enumerations, ``n_attn % n_kv_heads``, MoE ``n_experts >= top_k >= 1``
        and ``expert_hidden > 0``, ``tokenizer.vocab_size == model.vocab_size``,
        ``kind in (vl, vl-chat) -> multimodal.enabled``, ``train.seq_len <= model.max_seq_len``.
        Warnings (``UserWarning``): multimodal enabled outside test/mini tiers for a non-vl kind
        below 30B nominal; estimated total params outside ``nominal_params`` +-25% (an error when
        ``strict=True``).
        """
        c = _Checker("")
        c.non_empty(self, "name", "benchmark")
        c.choice(self, "tier", TIERS)
        c.choice(self, "kind", KINDS)
        nominal: Optional[int] = None
        try:
            nominal = parse_size(self.nominal_params)
            c.check(nominal > 0, "nominal_params", f"must be > 0 (got {self.nominal_params!r})")
        except ValueError:
            c.add("nominal_params", f"must be a size like '20M', '1.5B', '1T' (got {self.nominal_params!r})")
        problems = c.problems
        problems += self.model._problems("model")
        problems += self.multimodal._problems("multimodal")
        problems += self.tokenizer._problems("tokenizer")
        problems += self.data._problems("data")
        problems += self.train._problems("train")
        problems += self.chat._problems("chat")
        if self.tokenizer.vocab_size != self.model.vocab_size:
            problems.append(f"tokenizer.vocab_size: must equal model.vocab_size "
                            f"({self.tokenizer.vocab_size} != {self.model.vocab_size})")
        if self.kind in ("vl", "vl-chat") and not self.multimodal.enabled:
            problems.append(f"multimodal.enabled: must be true for kind {self.kind!r}")
        if self.train.seq_len > self.model.max_seq_len > 0:
            problems.append(f"train.seq_len: must be <= model.max_seq_len "
                            f"({self.train.seq_len} > {self.model.max_seq_len})")

        if not problems:
            if (self.multimodal.enabled and self.tier not in ("test", "mini")
                    and self.kind not in ("vl", "vl-chat") and (nominal or 0) < 30 * 10**9):
                warnings.warn(f"config {self.name!r}: multimodal.enabled for a {self.tier} {self.kind!r} "
                              f"model below 30B nominal params (SPEC recommends vl/vl-chat kinds, "
                              f"test/mini tiers or >= 30B)", UserWarning, stacklevel=2)
            if nominal:
                from moonlight.params import estimate_params  # lazy: params imports this module

                total = estimate_params(self).total
                ratio = total / nominal
                if abs(ratio - 1.0) > NOMINAL_TOLERANCE:
                    message = (f"estimated params {format_size(total)} ({total:,}) are {ratio:.2f}x the "
                               f"nominal {self.nominal_params} (allowed +-{NOMINAL_TOLERANCE:.0%})")
                    if strict:
                        problems.append(f"nominal_params: {message}")
                    else:
                        warnings.warn(f"config {self.name!r}: {message}", UserWarning, stacklevel=2)
        if problems:
            raise ValueError(_format_errors(f"Invalid config {self.name!r}", problems))
        return self

    # ---- conveniences --------------------------------------------------------------------------
    @property
    def bnchmrks(self) -> BenchmarkAccessor:
        """Benchmark suites accessor (``cfg.bnchmrks.type("cc-001")``)."""
        return BenchmarkAccessor(self)

    @property
    def benchmarks(self) -> BenchmarkAccessor:
        """Alias of :attr:`bnchmrks`."""
        return BenchmarkAccessor(self)

    @property
    def nominal_param_count(self) -> int:
        """``parse_size(nominal_params)``."""
        return parse_size(self.nominal_params)

    @property
    def n_retention_heads(self) -> int:
        """Shortcut for ``model.n_retention_heads``."""
        return self.model.n_retention_heads

    @property
    def n_attention_heads(self) -> int:
        """Shortcut for ``model.n_attention_heads``."""
        return self.model.n_attention_heads

    def __str__(self) -> str:
        return self.to_yaml()


# ================================================================================================
# Override helpers
# ================================================================================================


def _override_items(overrides: Union[Mapping[str, Any], Iterable[str], str, None]) -> List[Tuple[str, Any]]:
    if overrides is None:
        return []
    if isinstance(overrides, Mapping):
        return [(str(k), v) for k, v in overrides.items()]
    if isinstance(overrides, str):
        overrides = [overrides]
    items: List[Tuple[str, Any]] = []
    for entry in overrides:
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            items.append((str(entry[0]), entry[1]))
            continue
        if not isinstance(entry, str) or "=" not in entry:
            raise ValueError(f"override must look like 'key=value', got {entry!r}")
        key, value = entry.split("=", 1)
        items.append((key.strip(), value))
    return items


def _set_typed_field(obj: Any, name: str, value: Any, path: str, errors: List[str]) -> None:
    """Coerce ``value`` to the annotated type of dataclass field ``name`` and assign it."""
    tp = _hints(type(obj))[name]
    current = getattr(obj, name)
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        if isinstance(value, str):
            value = _parse_text(value)
        if isinstance(value, Mapping):  # deep-merge into the existing section
            value = deep_merge(_plain(current), value)
    elif typing.get_origin(tp) in (dict, Dict) and isinstance(current, Mapping):
        if isinstance(value, str) and not _is_null(value):
            value = _parse_text(value)
        if isinstance(value, Mapping):
            value = deep_merge(current, value)
    out = _coerce(value, tp, path, errors)
    if out is not _INVALID:
        setattr(obj, name, out)


def _item_type(tp: Any) -> Any:
    """Element annotation of ``List[X]`` / value annotation of ``Dict[str, X]`` (else ``Any``)."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is Union:
        options = [a for a in args if a is not _NoneType]
        return _item_type(options[0]) if len(options) == 1 else Any
    if origin in (list, List) and args:
        return args[0]
    if origin in (dict, Dict) and len(args) == 2:
        return args[1]
    return Any


def _coerce_slot(value: Any, tp: Any, path: str, errors: List[str]) -> Any:
    """Coerce a value stored inside a list/mapping slot. Free-form (``Any``) slots YAML-parse
    strings (``"500"`` -> 500, ``"true"`` -> True, ``"[a, b]"`` -> list)."""
    if tp is Any:
        if isinstance(value, str):
            value = _parse_text(value)
        return _free(value)
    return _coerce(value, tp, path, errors)


def _apply_one(root: Config, key: str, value: Any, errors: List[str]) -> None:
    """Apply one dotted override to ``root`` (descending dataclasses, mappings and lists)."""
    parts = [p.strip() for p in key.strip().split(".")]
    if not key.strip() or any(p == "" for p in parts):
        errors.append(f"invalid override key {key!r}")
        return
    obj: Any = root
    tp: Any = Config
    path = ""
    for depth, part in enumerate(parts):
        last = depth == len(parts) - 1
        here = _join(path, part)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            known = [f.name for f in dataclasses.fields(obj)]
            if part not in known:
                errors.append(_unknown_key_message(path, part, known))
                return
            if last:
                _set_typed_field(obj, part, value, here, errors)
                return
            child, child_tp = getattr(obj, part), _hints(type(obj))[part]
        elif isinstance(obj, Mapping):
            child_tp = _item_type(tp)
            if last:
                current = obj.get(part)
                new = deep_merge(current, value) if isinstance(value, Mapping) and isinstance(current, Mapping) \
                    else value
                out = _coerce_slot(new, child_tp, here, errors)
                if out is not _INVALID:
                    obj[part] = out
                return
            if obj.get(part) is None:
                obj[part] = AttrDict()
            child = obj[part]
        elif isinstance(obj, list):
            child_tp = _item_type(tp)
            try:
                index = int(part)
            except ValueError:
                errors.append(f"{here}: list index must be an integer")
                return
            if not -len(obj) <= index < len(obj):
                errors.append(f"{here}: list index out of range (length {len(obj)})")
                return
            if last:
                out = _coerce_slot(value, child_tp, here, errors)
                if out is not _INVALID:
                    obj[index] = out
                return
            child = obj[index]
        else:  # pragma: no cover - guarded below
            errors.append(f"{here}: cannot descend into {type(obj).__name__}")
            return
        if not ((dataclasses.is_dataclass(child) and not isinstance(child, type))
                or isinstance(child, (Mapping, list))):
            errors.append(f"{here}: is a {type(child).__name__} value, cannot set "
                          f"'{'.'.join(parts[depth + 1:])}' inside it")
            return
        obj, tp, path = child, child_tp, here
