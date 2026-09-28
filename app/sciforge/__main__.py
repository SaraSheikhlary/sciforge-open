"""Allow ``python -m sciforge``."""

from sciforge.cli import main

if __name__ == "__main__":  # pragma: no cover - thin wrapper
    raise SystemExit(main())
