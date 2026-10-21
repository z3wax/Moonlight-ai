"""Single source of truth for the Moonlight package version.

``pyproject.toml`` reads ``__version__`` from here (setuptools dynamic metadata), so keep the
assignment a plain string literal.
"""

__version__ = "0.1.0"

#: Parsed version tuple, e.g. ``(0, 1, 0)``.
version_info = tuple(int(part) for part in __version__.split("."))

__all__ = ["__version__", "version_info"]
