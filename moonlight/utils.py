"""Small, dependency-light helpers shared across Moonlight.

Contents: seeding, device selection, distributed rank helpers, human-readable numbers/bytes/durations,
a ``Timer``, a rank-aware logger and a rank-aware ``print0``.

Import hygiene (SPEC): this module must import WITHOUT torch. torch is only imported lazily inside
the functions that need it (``seed_everything``, ``get_device``), and rank helpers only consult
``torch.distributed`` when torch has already been imported by someone else.
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

__all__ = [
    "seed_everything",
    "get_device",
    "get_rank",
    "get_local_rank",
    "get_world_size",
    "is_main_process",
    "human_int",
    "human_bytes",
    "human_time",
    "Timer",
    "get_logger",
    "print0",
    "print_rank0",
    "count_parameters",
    "ensure_dir",
]


# ------------------------------------------------------------------------------------------------
# Seeding & devices
# ------------------------------------------------------------------------------------------------


def seed_everything(seed: int = 1337, deterministic: bool = False) -> int:
    """Seed Python ``random``, numpy and torch (all devices) if they are installed.

    Args:
        seed: the seed (any int; it is reduced modulo 2**32 for numpy).
        deterministic: additionally request deterministic torch kernels
            (``torch.use_deterministic_algorithms(True, warn_only=True)``, cuDNN deterministic,
            ``CUBLAS_WORKSPACE_CONFIG``). Slower; meant for tests and debugging.

    Returns:
        The seed that was applied.

    ``PYTHONHASHSEED`` is exported for child processes only (it cannot change the hash seed of the
    already running interpreter).
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)  # seeds CPU and every CUDA device (lazily)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed


def get_device(prefer: Optional[str] = None) -> str:
    """Return the best available device as a string usable by ``torch.device`` / ``.to()``.

    Resolution order: explicit ``prefer`` (anything but ``None``/``"auto"`` is returned unchanged),
    env ``MOONLIGHT_DEVICE``, then auto-detection ``cuda`` > ``mps`` > ``cpu``. Under ``torchrun``
    (env ``LOCAL_RANK`` set) CUDA resolves to ``"cuda:{LOCAL_RANK}"``. Without torch installed the
    answer is always ``"cpu"`` (auto mode).
    """
    choice = prefer if prefer not in (None, "", "auto") else os.environ.get("MOONLIGHT_DEVICE", "auto")
    choice = str(choice).strip() or "auto"
    if choice != "auto":
        return choice
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None and local_rank.strip().isdigit():
            return f"cuda:{int(local_rank)}"
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------------------------------------------------------------------------
# Distributed rank helpers (torch-free unless torch.distributed is already initialised)
# ------------------------------------------------------------------------------------------------


def _initialized_dist() -> Any:
    """``torch.distributed`` if torch is ALREADY imported and a process group exists, else None."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        dist = torch.distributed
        if dist.is_available() and dist.is_initialized():
            return dist
    except Exception:  # pragma: no cover - defensive (partially initialised torch)
        return None
    return None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    try:
        return int(value)
    except ValueError:
        return default


def get_rank() -> int:
    """Global rank: ``torch.distributed`` if initialised, else env ``RANK`` (torchrun), else 0."""
    dist = _initialized_dist()
    if dist is not None:
        return int(dist.get_rank())
    return _env_int("RANK", 0)


def get_local_rank() -> int:
    """Node-local rank from env ``LOCAL_RANK`` (torchrun), else 0."""
    return _env_int("LOCAL_RANK", 0)


def get_world_size() -> int:
    """World size: ``torch.distributed`` if initialised, else env ``WORLD_SIZE``, else 1."""
    dist = _initialized_dist()
    if dist is not None:
        return int(dist.get_world_size())
    return max(1, _env_int("WORLD_SIZE", 1))


def is_main_process() -> bool:
    """True on global rank 0 (and in every non-distributed run)."""
    return get_rank() == 0


# ------------------------------------------------------------------------------------------------
# Human-readable formatting
# ------------------------------------------------------------------------------------------------

# Ascending decimal units used for parameter / token counts.
_COUNT_UNITS = (("", 1), ("K", 10**3), ("M", 10**6), ("B", 10**9), ("T", 10**12))
_BYTE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")


def _strip_zeros(text: str) -> str:
    return text.rstrip("0").rstrip(".") if "." in text else text


def human_int(n: Union[int, float], precision: int = 1) -> str:
    """Format a count with a decimal K/M/B/T suffix.

    Examples: ``1_234_567 -> "1.2M"``, ``20_000_000 -> "20M"``, ``20_100_000 -> "20.1M"``,
    ``1.5e9 -> "1.5B"``, ``999 -> "999"``, ``999_960 -> "1M"`` (rounding never produces "1000K").
    Trailing zeros are stripped. Negative numbers keep their sign; non-finite floats are returned
    as ``str(n)``.
    """
    x = float(n)
    if not math.isfinite(x):
        return str(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    idx = 0
    for i, (_, scale) in enumerate(_COUNT_UNITS):
        if x >= scale:
            idx = i
    while True:
        suffix, scale = _COUNT_UNITS[idx]
        value = round(x / scale, precision)
        if value >= 1000 and idx + 1 < len(_COUNT_UNITS):
            idx += 1
            continue
        break
    if idx == 0 and x.is_integer():
        return f"{sign}{int(x)}"
    return f"{sign}{_strip_zeros(f'{value:.{precision}f}')}{suffix}"


def human_bytes(n: Union[int, float], precision: int = 1) -> str:
    """Format a byte count with binary units: ``1536 -> "1.5 KiB"``, ``3 * 2**30 -> "3 GiB"``."""
    x = float(n)
    if not math.isfinite(x):
        return str(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    idx = 0
    while x >= 1024 and idx + 1 < len(_BYTE_UNITS):
        x /= 1024.0
        idx += 1
    if round(x, precision) >= 1024 and idx + 1 < len(_BYTE_UNITS):
        x /= 1024.0
        idx += 1
    if idx == 0:
        return f"{sign}{int(round(x))} B"
    return f"{sign}{_strip_zeros(f'{x:.{precision}f}')} {_BYTE_UNITS[idx]}"


def human_time(seconds: float) -> str:
    """Format a duration: ``4.2 -> "4.2s"``, ``725 -> "12m 05s"``, ``3723 -> "1h 02m 03s"``,
    ``183600 -> "2d 03h 00m"``."""
    s = float(seconds)
    if not math.isfinite(s):
        return str(s)
    sign = "-" if s < 0 else ""
    s = abs(s)
    if s < 60:
        return f"{sign}{s:.1f}s"
    total = int(round(s))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{sign}{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{sign}{hours}h {minutes:02d}m {secs:02d}s"
    return f"{sign}{minutes}m {secs:02d}s"


# ------------------------------------------------------------------------------------------------
# Timer
# ------------------------------------------------------------------------------------------------


class Timer:
    """Wall-clock stopwatch (``time.perf_counter``), usable manually or as a context manager.

    >>> with Timer("tokenize") as t:      # logs "tokenize: 1.2s" at INFO on exit if ``log=True``
    ...     work()
    >>> t.elapsed                         # seconds (float)

    Manual use: ``t = Timer().start(); ...; t.stop()``; ``t.lap()`` returns seconds since the last
    lap (or start). ``elapsed`` includes the currently running interval; ``stop()`` accumulates,
    so start/stop can be repeated to time disjoint intervals. With ``sync_cuda=True`` the timer
    calls ``torch.cuda.synchronize()`` (only if torch is already imported and CUDA is available)
    before reading the clock, so asynchronous GPU work is included.
    """

    def __init__(self, name: Optional[str] = None, log: bool = False, sync_cuda: bool = False,
                 logger: Optional[logging.Logger] = None) -> None:
        self.name = name or "timer"
        self.log = log
        self.sync_cuda = sync_cuda
        self.logger = logger
        self._accum = 0.0
        self._start: Optional[float] = None
        self._lap: Optional[float] = None

    def _now(self) -> float:
        if self.sync_cuda:
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
        return time.perf_counter()

    @property
    def running(self) -> bool:
        return self._start is not None

    def start(self) -> "Timer":
        if self._start is None:
            self._start = self._now()
            self._lap = self._start
        return self

    def stop(self) -> float:
        """Stop and return the total accumulated seconds."""
        if self._start is not None:
            self._accum += self._now() - self._start
            self._start = None
        return self._accum

    def reset(self) -> "Timer":
        self._accum = 0.0
        self._start = None
        self._lap = None
        return self

    def lap(self) -> float:
        """Seconds since the previous ``lap()`` (or ``start()``); starts the timer if needed."""
        if self._start is None:
            self.start()
            return 0.0
        now = self._now()
        delta = now - (self._lap if self._lap is not None else now)
        self._lap = now
        return delta

    @property
    def elapsed(self) -> float:
        """Accumulated seconds including the currently running interval."""
        if self._start is None:
            return self._accum
        return self._accum + (self._now() - self._start)

    @property
    def minutes(self) -> float:
        return self.elapsed / 60.0

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()
        if self.log:
            (self.logger or get_logger()).info("%s: %s", self.name, human_time(self._accum))

    def __repr__(self) -> str:
        state = "running" if self.running else "stopped"
        return f"Timer(name={self.name!r}, elapsed={self.elapsed:.3f}s, {state})"


# ------------------------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------------------------

_ROOT_LOGGER = "moonlight"
_HANDLER_FLAG = "_moonlight_handler"


class _StdoutHandler(logging.StreamHandler):
    """StreamHandler that always writes to the CURRENT ``sys.stdout``.

    Looking the stream up at emit time keeps logging working when ``sys.stdout`` is swapped
    (pytest capture, notebooks) instead of writing to a stale/closed stream.
    """

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property  # type: ignore[override]
    def stream(self) -> Any:
        return sys.stdout

    @stream.setter
    def stream(self, value: Any) -> None:  # ignore assignments from StreamHandler.__init__/setStream
        pass


def _parse_level(level: Union[int, str, None]) -> int:
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    value = logging.getLevelName(str(level).strip().upper())
    return value if isinstance(value, int) else logging.INFO


def _configure_root_logger() -> logging.Logger:
    root = logging.getLogger(_ROOT_LOGGER)
    if any(getattr(h, _HANDLER_FLAG, False) for h in root.handlers):
        return root
    rank = get_rank()
    world = get_world_size()
    rank_tag = f"[rank {rank}] " if world > 1 or rank > 0 else ""
    handler = _StdoutHandler()
    setattr(handler, _HANDLER_FLAG, True)
    handler.setFormatter(logging.Formatter(
        f"%(asctime)s %(levelname).1s {rank_tag}%(name)s: %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.propagate = False
    level = _parse_level(os.environ.get("MOONLIGHT_LOG_LEVEL", "INFO"))
    if rank != 0:  # non-main ranks only report problems
        level = max(level, logging.WARNING)
    root.setLevel(level)
    return root


def get_logger(name: str = _ROOT_LOGGER, level: Union[int, str, None] = None) -> logging.Logger:
    """Return a logger in the ``moonlight`` hierarchy (configured once, rank-aware).

    ``name`` may be a module ``__name__`` (``"moonlight.train.trainer"``) or a short name
    (``"bench"`` -> ``"moonlight.bench"``). The ``moonlight`` root logger gets one stdout handler,
    level from env ``MOONLIGHT_LOG_LEVEL`` (default INFO); ranks != 0 log WARNING and above only and
    are tagged ``[rank N]``. ``level`` (if given) sets the level of the returned logger.
    """
    _configure_root_logger()
    if name != _ROOT_LOGGER and not name.startswith(_ROOT_LOGGER + "."):
        name = f"{_ROOT_LOGGER}.{name}"
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(_parse_level(level))
    return logger


def print0(*args: Any, **kwargs: Any) -> None:
    """``print`` on global rank 0 only (flushes by default)."""
    if get_rank() == 0:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


#: Alias of :func:`print0`.
print_rank0 = print0


# ------------------------------------------------------------------------------------------------
# Misc
# ------------------------------------------------------------------------------------------------


def count_parameters(module: Any, trainable_only: bool = False) -> int:
    """Number of (unique) parameters of a torch module; tied weights are counted once."""
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad or not trainable_only)


def ensure_dir(path: Union[str, os.PathLike]) -> Path:
    """Create ``path`` (with parents, ``~`` expanded) if missing and return it as a ``Path``."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p
