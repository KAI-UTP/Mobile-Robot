"""Make `shared/` and `services/` importable from the tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT / "shared", ROOT / "services", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# The Omniverse scene's stub-USD harness lives in test_kit_measured_room.py.
# Re-exported here so more than one module can load the scene — the geometry
# tests need it too, and duplicating a hundred lines of USD stubs to get at a
# fixture is how two copies of a harness start disagreeing.
import pytest  # noqa: E402
from test_kit_measured_room import FakeStage, _load_kit_module  # noqa: E402


@pytest.fixture
def kit():
    stage = FakeStage()
    module = _load_kit_module(stage)
    # No test reaches the network; see the note in test_kit_measured_room.py.
    module.real_room_has_furniture = module.room_has_furniture
    module.room_has_furniture = lambda: True
    return module, stage
