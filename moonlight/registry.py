"""Config registry: locate, list and load the shipped YAML recipes (``configs/{tier}/{name}.yaml``).

Torch-free. The config directory is resolved at call time:

1. env ``MOONLIGHT_CONFIG_DIR`` (if set),
2. :data:`CONFIG_ROOT` = ``<repo root>/configs`` located relative to this package
   (works for source checkouts and editable installs),
3. ``./configs`` in the current working directory (fallback for non-editable installs run from a
   repository checkout).

Names are file stems (``ml-20m-chat``); stems are unique across tiers. Files or directories whose
name starts with ``_`` or ``.`` (e.g. shared ``extends:`` bases) are findable but not listed.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from moonlight.config import TIERS, Config

__all__ = [
    "CONFIG_ROOT",
    "ENV_VAR",
    "ConfigNotFoundError",
    "config_root",
    "find_config",
    "list_configs",
    "load",
    "load_config",
    "get_config",
    "describe",
]

#: Default config directory: ``<repo root>/configs`` (sibling of the ``moonlight`` package).
CONFIG_ROOT: Path = Path(__file__).resolve().parents[1] / "configs"
#: Environment variable overriding the config directory.
ENV_VAR = "MOONLIGHT_CONFIG_DIR"

_SUFFIXES = (".yaml", ".yml")


class ConfigNotFoundError(FileNotFoundError, KeyError, ValueError):
    """Unknown config name / missing file. Subclasses FileNotFoundError, KeyError and ValueError so
    callers may catch whichever is natural."""

    def __str__(self) -> str:  # KeyError would repr()-quote the message
        return str(self.args[0]) if self.args else ""


def config_root() -> Path:
    """The directory searched for configs (see module docstring for the resolution order)."""
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if CONFIG_ROOT.is_dir():
        return CONFIG_ROOT
    cwd = Path.cwd() / "configs"
    if cwd.is_dir():
        return cwd.resolve()
    return CONFIG_ROOT


def _hidden(rel: Path) -> bool:
    return any(part.startswith((".", "_")) for part in rel.parts)


def _config_files(root: Path) -> List[Path]:
    """All ``*.yaml``/``*.yml`` files under ``root`` (sorted, recursive)."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _SUFFIXES and p.is_file())


def _not_found(name: str, root: Path, candidates: List[str]) -> ConfigNotFoundError:
    close = difflib.get_close_matches(name, candidates, n=5, cutoff=0.5)
    if not close:  # also try a substring match ("20m-chat" -> ml-20m-chat, ml-20m-colab-chat)
        close = [c for c in candidates if name.lower() in c.lower()][:5]
    message = f"config {name!r} not found in {root}"
    if close:
        message += f"; did you mean: {', '.join(close)}?"
    elif not candidates:
        message += (f" (no configs there; generate them with `python scripts/generate_configs.py` "
                    f"or set {ENV_VAR})")
    else:
        message += " (see `moonlight list`)"
    return ConfigNotFoundError(message)


def find_config(name: Union[str, os.PathLike]) -> Path:
    """Resolve a registry name (``"ml-20m-chat"``), a tier-qualified name (``"test/ml-20m-chat"``)
    or a path to a ``.yaml`` file into an absolute ``Path``.

    Raises :class:`ConfigNotFoundError` (a FileNotFoundError / KeyError / ValueError) with close
    matches when nothing is found, and ``ValueError`` if a stem is ambiguous.
    """
    raw = os.fspath(name).strip()
    if not raw:
        raise ConfigNotFoundError("empty config name")
    root = config_root()
    path = Path(raw).expanduser()
    is_path_like = path.suffix.lower() in _SUFFIXES or "/" in raw or os.sep in raw

    if is_path_like:
        candidates = [path]
        if not path.is_absolute():
            candidates.append(root / path)
            if path.suffix.lower() not in _SUFFIXES:
                candidates += [root / (raw + suffix) for suffix in _SUFFIXES]
        for cand in candidates:
            if cand.is_file():
                return cand.resolve()
        if path.is_absolute() or (len(path.parts) > 1 and path.suffix.lower() in _SUFFIXES):
            raise ConfigNotFoundError(f"config file not found: {raw} (also looked under {root})")
        # a bare "name.yaml" or "tier/name" falls through to the stem search below

    stem = path.name[: -len(path.suffix)] if path.suffix.lower() in _SUFFIXES else path.name
    files = _config_files(root)
    matches = [p for p in files if p.stem == stem]
    if not matches:  # case-insensitive fallback
        matches = [p for p in files if p.stem.lower() == stem.lower()]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        listing = ", ".join(str(p.relative_to(root)) for p in matches)
        raise ValueError(f"config name {stem!r} is ambiguous under {root}: {listing}")
    names = sorted({p.stem for p in files if not _hidden(p.relative_to(root))})
    raise _not_found(stem, root, names)


def _tier_of(path: Path, root: Path) -> Optional[str]:
    rel = path.relative_to(root)
    if len(rel.parts) > 1 and rel.parts[0] in TIERS:
        return rel.parts[0]
    try:  # flat layouts: read the tier from the file itself
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    tier = data.get("tier") if isinstance(data, dict) else None
    return tier if isinstance(tier, str) else None


def list_configs(tier: Optional[str] = None) -> List[str]:
    """Sorted config names, optionally restricted to a tier (``test`` | ``mini`` | ``si``).

    The tier is taken from the directory (``configs/{tier}/...``), or from the file's ``tier:`` key
    for files outside a tier directory.
    """
    if tier is not None and tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {', '.join(TIERS)}")
    root = config_root()
    names = set()
    for p in _config_files(root):
        if _hidden(p.relative_to(root)):
            continue
        if tier is not None and _tier_of(p, root) != tier:
            continue
        names.add(p.stem)
    return sorted(names)


def load(name: Union[str, os.PathLike], overrides: Any = None, validate: bool = True) -> Config:
    """Registry name or YAML path -> :class:`~moonlight.config.Config`.

    ``overrides``: mapping of dotted keys or iterable of ``"key=value"`` strings, applied after
    loading. With ``validate=True`` (default) the result is validated (``ValueError`` on problems).
    """
    if isinstance(name, Config):
        cfg = name.copy()
    else:
        cfg = Config.from_yaml(find_config(name))
    if overrides:
        cfg.apply_overrides(overrides)
    if validate:
        cfg.validate()
    return cfg


#: Aliases of :func:`load`.
load_config = load
get_config = load


def describe(name: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Small summary dict of a config (name, tier, kind, nominal/estimated params, path)."""
    from moonlight.params import estimate_params

    path = find_config(name)
    cfg = Config.from_yaml(path)
    report = estimate_params(cfg)
    return {"name": cfg.name, "tier": cfg.tier, "kind": cfg.kind, "benchmark": cfg.benchmark,
            "nominal_params": cfg.nominal_params, "params": report.total, "active_params": report.active,
            "path": str(path)}
