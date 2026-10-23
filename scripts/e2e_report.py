"""Entry point: regenerates the e2e sprint-delivery report (see tests/e2e/report.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from e2e.report import main

if __name__ == "__main__":
    raise SystemExit(main())
