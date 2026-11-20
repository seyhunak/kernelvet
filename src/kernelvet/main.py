"""Console entry point (``kernelvet``)."""

from __future__ import annotations

import sys


def main() -> None:
    from kernelvet.cli.app import app

    try:
        app()
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
