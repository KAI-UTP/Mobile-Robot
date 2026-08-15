"""Log-odds occupancy grid built from range-sensor rays.

Why log-odds
------------
Each cell holds the log of the odds that it is occupied, not a probability.
Two properties make this the standard representation:

1. Bayesian updates become *addition*, which is fast and cannot underflow.
2. Evidence accumulates symmetrically — ten "free" observations genuinely
   cancel ten "occupied" ones, so a person walking through the room while
   mapping does not leave a permanent wall behind them.

The alternative, storing a raw probability and multiplying, saturates at 0 or
1 after a few dozen readings and can then never be corrected.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from robotmap_common.geometry import bresenham_line, project_ray
from robotmap_common.models import (
    GridMetadata,
    MapSnapshot,
    PoseEstimate,
    RangeReading,
)

logger = logging.getLogger(__name__)


def _log_odds(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


class OccupancyGrid:
    """A 2-D metric map of the room, grown on demand.

    The grid auto-expands when the robot drives past its edge, so no one has
    to guess the room size before mapping starts.
    """

    # Probability that a cell is occupied given one "hit" / "pass-through".
    # Deliberately not 0.9/0.1: a single ultrasonic reading is weak evidence,
    # and overconfident updates produce walls made of noise.
    P_OCCUPIED = 0.70
    P_FREE = 0.35

    # Clamp the accumulated log-odds. Without this, a cell observed a
    # thousand times becomes infinitely certain and can never be revised
    # when the world changes.
    LOG_ODDS_MAX = 6.0
    LOG_ODDS_MIN = -6.0

    # A cell this confidently occupied (~88 %) is treated as opaque: rays
    # stop there instead of carving free space out of whatever lies behind
    # it. See `_integrate_ray` for why this matters.
    LOG_ODDS_BLOCKING = 2.0

    def __init__(
        self,
        resolution_m: float = 0.05,
        initial_size_m: float = 10.0,
        max_range_m: float = 4.0,
    ) -> None:
        if resolution_m <= 0:
            raise ValueError("resolution_m must be positive")

        self.resolution_m = resolution_m
        self.max_range_m = max_range_m

        cells = int(initial_size_m / resolution_m)
        self.grid = np.zeros((cells, cells), dtype=np.float32)

        # World coordinate of cell (0, 0). Starts centred so the robot has
        # room to drive in every direction from its origin.
        self.origin_x_m = -initial_size_m / 2.0
        self.origin_y_m = -initial_size_m / 2.0

        self.l_occupied = _log_odds(self.P_OCCUPIED)
        self.l_free = _log_odds(self.P_FREE)

        self.updates_applied = 0
        self.rays_rejected = 0

    # ── Coordinate conversion ─────────────────────────────────────────────

    @property
    def height_cells(self) -> int:
        return self.grid.shape[0]

    @property
    def width_cells(self) -> int:
        return self.grid.shape[1]

    def world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Return (col, row) for a world coordinate. May be out of bounds."""
        col = int(math.floor((x_m - self.origin_x_m) / self.resolution_m))
        row = int(math.floor((y_m - self.origin_y_m) / self.resolution_m))
        return col, row

    def cell_to_world(self, col: int, row: int) -> tuple[float, float]:
        """Return the world coordinate of a cell's centre."""
        return (
            self.origin_x_m + (col + 0.5) * self.resolution_m,
            self.origin_y_m + (row + 0.5) * self.resolution_m,
        )

    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.width_cells and 0 <= row < self.height_cells

    # ── Growth ────────────────────────────────────────────────────────────

    def ensure_contains(self, x_m: float, y_m: float, margin_m: float = 2.0) -> None:
        """Expand the grid so (x_m, y_m) plus a margin fits inside it."""
        col, row = self.world_to_cell(x_m, y_m)
        margin_cells = int(math.ceil(margin_m / self.resolution_m))

        pad_left = max(0, margin_cells - col)
        pad_right = max(0, col + margin_cells - (self.width_cells - 1))
        pad_bottom = max(0, margin_cells - row)
        pad_top = max(0, row + margin_cells - (self.height_cells - 1))

        if not (pad_left or pad_right or pad_bottom or pad_top):
            return

        # Pad with zeros — log-odds 0 is exactly "unknown", which is the
        # correct prior for ground we have never seen.
        self.grid = np.pad(
            self.grid,
            ((pad_bottom, pad_top), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0,
        )
        self.origin_x_m -= pad_left * self.resolution_m
        self.origin_y_m -= pad_bottom * self.resolution_m

        logger.debug(
            "Grid grown to %dx%d cells (origin %.2f, %.2f)",
            self.width_cells,
            self.height_cells,
            self.origin_x_m,
            self.origin_y_m,
        )

    # ── Ray integration ───────────────────────────────────────────────────

    def integrate_scan(self, pose: PoseEstimate, ranges: list[RangeReading]) -> int:
        """Fold one posed scan into the grid. Returns the number of rays used.

        Scans taken from a pose the filter no longer trusts are dropped
        outright: writing them in would smear the walls of an otherwise good
        map, and a smaller correct map beats a larger corrupted one.
        """
        if pose.position_confidence < 0.2:
            self.rays_rejected += len(ranges)
            return 0

        self.ensure_contains(pose.x_m, pose.y_m, margin_m=self.max_range_m + 1.0)

        used = 0
        for reading in ranges:
            if self._integrate_ray(pose, reading):
                used += 1

        self.updates_applied += 1
        return used

    def _integrate_ray(self, pose: PoseEstimate, reading: RangeReading) -> bool:
        if not reading.valid:
            return False

        # A reading at or beyond max range means "nothing out there", not
        # "wall at exactly 4 m". Marking the endpoint occupied would build a
        # phantom wall in a ring around the robot — a classic sonar artefact.
        is_max_range = reading.distance_m >= self.max_range_m - 1e-6
        distance = min(reading.distance_m, self.max_range_m)

        if distance <= 0.0:
            return False

        hit_x, hit_y = project_ray(
            pose.x_m, pose.y_m, pose.heading_deg, reading.angle_deg, distance
        )
        self.ensure_contains(hit_x, hit_y, margin_m=1.0)

        start_col, start_row = self.world_to_cell(pose.x_m, pose.y_m)
        end_col, end_row = self.world_to_cell(hit_x, hit_y)

        if not self.in_bounds(start_col, start_row):
            return False

        cells = bresenham_line(start_col, start_row, end_col, end_row)

        # Everything the ray passed through is free space — but only up to the
        # first cell already known to be solid.
        #
        # Without this stop, one physical effect wrecks the whole map:
        # ultrasonic pulses striking a wall at a shallow angle reflect away
        # instead of returning, so the sensor reports max range. Taken at face
        # value that carves a 4 m corridor of "free space" straight through
        # the wall, and the room's boundary springs a leak that the flood fill
        # then pours out of. Every corner pivot produces such a reading.
        #
        # The blocking cell itself still receives the free update before the
        # ray stops, so genuine change is still learnable: an obstacle that
        # moves away decays past the threshold after enough observations and
        # rays resume passing through it.
        blocked = False
        for col, row in cells[:-1]:
            if not self.in_bounds(col, row):
                continue
            was_blocking = self.grid[row, col] >= self.LOG_ODDS_BLOCKING
            self._update_cell(col, row, self.l_free)
            if was_blocking:
                blocked = True
                break

        # The endpoint is a wall, unless the ray ran out of range or never
        # reached the endpoint because something solid stopped it first.
        if not is_max_range and not blocked and self.in_bounds(end_col, end_row):
            self._update_cell(end_col, end_row, self.l_occupied)

        return True

    def _update_cell(self, col: int, row: int, delta: float) -> None:
        value = self.grid[row, col] + delta
        self.grid[row, col] = min(self.LOG_ODDS_MAX, max(self.LOG_ODDS_MIN, value))

    def mark_robot_footprint(self, pose: PoseEstimate, radius_m: float = 0.12) -> None:
        """Mark the cells under the robot as free.

        The robot is demonstrably standing there, so those cells cannot be
        walls. This closes the small unknown gaps that range sensors mounted
        above floor level leave behind.
        """
        radius_cells = int(math.ceil(radius_m / self.resolution_m))
        centre_col, centre_row = self.world_to_cell(pose.x_m, pose.y_m)

        for d_row in range(-radius_cells, radius_cells + 1):
            for d_col in range(-radius_cells, radius_cells + 1):
                if d_col * d_col + d_row * d_row > radius_cells * radius_cells:
                    continue
                col, row = centre_col + d_col, centre_row + d_row
                if self.in_bounds(col, row):
                    self._update_cell(col, row, self.l_free)

    # ── Queries ───────────────────────────────────────────────────────────

    def probability_map(self) -> np.ndarray:
        """Convert log-odds back to probabilities in [0, 1]."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.grid))

    def occupancy_percent(self) -> np.ndarray:
        """ROS-style int8 map: -1 unknown, 0-100 percent occupied."""
        prob = self.probability_map()
        out = np.rint(prob * 100).astype(np.int16)
        out[self.grid == 0.0] = -1  # never observed
        return out

    def explored_cells(self) -> int:
        return int(np.count_nonzero(self.grid))

    def free_mask(self, threshold: float = -0.4) -> np.ndarray:
        """Cells confidently observed as free."""
        return self.grid < threshold

    def occupied_mask(self, threshold: float = 0.4) -> np.ndarray:
        """Cells confidently observed as occupied."""
        return self.grid > threshold

    def metadata(self) -> GridMetadata:
        return GridMetadata(
            resolution_m=self.resolution_m,
            width_cells=self.width_cells,
            height_cells=self.height_cells,
            origin_x_m=self.origin_x_m,
            origin_y_m=self.origin_y_m,
        )

    def snapshot(self, robot_id: str, timestamp: str) -> MapSnapshot:
        return MapSnapshot(
            robot_id=robot_id,
            timestamp=timestamp,
            metadata=self.metadata(),
            data=self.occupancy_percent().flatten().tolist(),
            explored_cells=self.explored_cells(),
        )

    def clear(self) -> None:
        self.grid[:] = 0.0
        self.updates_applied = 0
        self.rays_rejected = 0
