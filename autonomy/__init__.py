"""How the robot decides where to drive.

Separate from `simulator/` on purpose. These controllers consume `RangeReading`s
and emit velocity commands and nothing else — no world model, no ground truth,
no simulator types anywhere in their signatures. That is what lets the identical
code drive the virtual robot in CI and the real servo bus in a room, which in
turn is what makes a passing test say something about the hardware.

They lived under `simulator/` while only the simulator drove them, and the name
then said the opposite of the truth: it read as though autonomy were a
simulation artefact, and the real robot had no path to an autonomous scan at
all.

Two controllers, run in this order by `services/pilot`:

* `WallFollower` — hugs the boundary. Close range and good incidence angles are
  what an outline needs. Learns nothing about the middle of the room.
* `CoveragePlanner` — sweeps the interior row by row. Slower, and the only way
  to find a table standing in open floor.
"""

from .coverage import (
    CoverageCommand,
    CoverageConfig,
    CoveragePlanner,
    CoverageState,
    Obstacle,
)
from .explorer import DriveCommand, ExploreConfig, ExploreState, WallFollower

__all__ = [
    "CoverageCommand",
    "CoverageConfig",
    "CoveragePlanner",
    "CoverageState",
    "DriveCommand",
    "ExploreConfig",
    "ExploreState",
    "Obstacle",
    "WallFollower",
]
