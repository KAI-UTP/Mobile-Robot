"""Make `shared/` and `services/` importable from the tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT / "shared", ROOT / "services", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
