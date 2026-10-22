"""Moonlight: hybrid Attention + Retention language models (1M -> 1T params) for Uzbek and English.

```python
import moonlight as ml
cfg = ml.config("ml-20m-chat", overrides={"train.batch_size": 16})
llm = ml.LLM(cfg); llm.train(); print(llm.chat("Salom!"))
ml.train("ml-1m-debug")                 # one-liner: config -> LLM -> train -> export
ml.benchmarks.type("cc-001")            # a benchmark suite without a config
```

Everything is imported lazily (PEP 562 ``__getattr__``): ``import moonlight`` and
``moonlight.config(...)`` / ``moonlight.list_configs()`` / ``moonlight.estimate_params(...)`` work
without torch; torch is only needed to build / train / run a model.

Name clashes, resolved deliberately: ``moonlight.config`` and ``moonlight.train`` are both public
*functions* (SPEC §1) and *submodules* (``moonlight/config.py``, ``moonlight/train/``). Importing a
submodule normally rebinds the package attribute to the module, which would silently replace the
function. Here the attribute is always a callable proxy that

* calls the function: ``ml.config("ml-20m-chat")``, ``ml.train("ml-1m-debug")``;
* forwards every other attribute to the submodule: ``ml.config.Config``,
  ``ml.train.Trainer``, ``import moonlight.train.checkpoint as ck`` and
  ``from moonlight.train.trainer import Trainer`` all keep working;

and the package refuses to let the import system rebind those two names to the submodules (the
modules stay in ``sys.modules`` and are reachable through the proxies), so the behaviour does not
depend on import order.
"""

from __future__ import annotations

import functools
import importlib
import sys
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

from moonlight.version import __version__

# export name -> (module, attribute); attribute None -> the module itself
_EXPORTS: Dict[str, Tuple[str, Optional[str]]] = {
    "list_configs": ("moonlight.registry", "list_configs"),
    "find_config": ("moonlight.registry", "find_config"),
    "Config": ("moonlight.config", "Config"),
    "estimate_params": ("moonlight.params", "estimate_params"),
    "ParamReport": ("moonlight.params", "ParamReport"),
    "LLM": ("moonlight.llm", "LLM"),
    "TrainResult": ("moonlight.llm", "TrainResult"),
    "train_config": ("moonlight.llm", "train_config"),
    "ChatSession": ("moonlight.chat", "ChatSession"),
    "ProjectResult": ("moonlight.chat", "ProjectResult"),
    "Tool": ("moonlight.tools", "Tool"),
    "tool": ("moonlight.tools", "tool"),
    "ToolRegistry": ("moonlight.tools", "ToolRegistry"),
    "Workspace": ("moonlight.tools", "Workspace"),
    "MoonlightTokenizer": ("moonlight.tokenizer", "MoonlightTokenizer"),
    "benchmarks": ("moonlight.benchmarks", None),
}

__all__ = ["config", "list_configs", "LLM", "train", "Config", "estimate_params", "benchmarks", "ChatSession",
           "Tool", "tool", "__version__"]


# -------------------------------------------------------------------------------------------------
# public functions that share their name with a submodule
# -------------------------------------------------------------------------------------------------

def _config(name_or_path: Any, overrides: Any = None, validate: bool = True) -> Any:
    """Load a config: registry name (``"ml-20m-chat"``) or path to a ``.yaml`` -> ``Config``.

    ``overrides``: dotted keys ``{"train.batch_size": 16}`` or ``["train.batch_size=16"]``.
    Validated unless ``validate=False``. A ``Config`` argument is copied.
    """
    from moonlight.registry import load

    return load(name_or_path, overrides=overrides, validate=validate)


def _train(name_or_cfg: Any, overrides: Any = None, output: Any = None, **kw: Any) -> Any:
    """One-liner training: config -> ``LLM`` -> ``train()`` -> export; returns the trained ``LLM``
    (see :func:`moonlight.llm.train_config`)."""
    from moonlight.llm import train_config

    return train_config(name_or_cfg, overrides=overrides, output=output, **kw)


_config.__name__ = _config.__qualname__ = "config"
_train.__name__ = _train.__qualname__ = "train"

def _is_local(name: str) -> bool:
    """Proxy-own attributes: its two slots and every dunder (``__wrapped__``, ``__doc__``,
    ``__type_params__`` ... set by ``functools.update_wrapper``)."""
    return name in ("_ml_func", "_ml_module") or (name.startswith("__") and name.endswith("__"))


class _CallableModuleProxy:
    """A function that also exposes a submodule's attributes (see the package docstring).

    ``proxy(*a, **k)`` calls the function; attribute reads / writes / deletes that are not the
    proxy's own go to the submodule (imported on first use). ``inspect.signature(proxy)`` is the
    function's signature (via ``__wrapped__``).
    """

    def __init__(self, func: Callable[..., Any], module_name: str) -> None:
        object.__setattr__(self, "_ml_func", func)
        object.__setattr__(self, "_ml_module", module_name)
        functools.update_wrapper(self, func)  # __wrapped__, __doc__, __name__, __qualname__, __module__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._ml_func(*args, **kwargs)

    def _module(self) -> types.ModuleType:
        return importlib.import_module(object.__getattribute__(self, "_ml_module"))

    def __getattr__(self, name: str) -> Any:  # only called when normal lookup fails
        if name.startswith("_ml_"):
            raise AttributeError(name)
        return getattr(self._module(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if _is_local(name):
            object.__setattr__(self, name, value)
        else:
            setattr(self._module(), name, value)

    def __delattr__(self, name: str) -> None:
        if _is_local(name):
            object.__delattr__(self, name)
        else:
            delattr(self._module(), name)

    def __dir__(self) -> List[str]:
        return sorted(set(dir(self._module())) | {"__call__", "__wrapped__"})

    def __repr__(self) -> str:
        return (f"<function moonlight.{self.__name__} (callable; attributes of module "
                f"{object.__getattribute__(self, '_ml_module')})>")


config = _CallableModuleProxy(_config, "moonlight.config")
train = _CallableModuleProxy(_train, "moonlight.train")

_PROTECTED = {"config": "moonlight.config", "train": "moonlight.train"}


class _MoonlightPackage(types.ModuleType):
    """Module type of the ``moonlight`` package: keeps the ``config`` / ``train`` proxies bound
    when the import system tries to set the same-named submodules as package attributes."""

    def __setattr__(self, name: str, value: Any) -> None:
        if (name in _PROTECTED and isinstance(value, types.ModuleType)
                and value.__name__ == _PROTECTED[name]
                and isinstance(self.__dict__.get(name), _CallableModuleProxy)):
            return  # the submodule stays importable (sys.modules) and reachable through the proxy
        super().__setattr__(name, value)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy exports (see ``_EXPORTS``); values are cached in the module namespace."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'moonlight' has no attribute {name!r}")
    module_name, attr = target
    module = importlib.import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    public = {n for n in globals() if n.startswith("__") and n.endswith("__")}
    return sorted(public | set(_EXPORTS) | set(__all__))


sys.modules[__name__].__class__ = _MoonlightPackage
