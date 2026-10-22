"""``python -m moonlight ...`` -> :func:`moonlight.cli.main`."""

import sys

from moonlight.cli import main

if __name__ == "__main__":
    sys.exit(main())
