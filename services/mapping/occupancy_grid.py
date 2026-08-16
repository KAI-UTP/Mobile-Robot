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

    # What one bumper contact is worth.
    #
    # Far stronger than a range reading, and deliberately so. An ultrasonic
    # pulse infers an obstacle from an echo and is wrong often — off a shallow
    # wall, a soft sofa, a chair leg narrower than the beam. A bumper is a
    # switch closed by the object itself: the robot is touching it. There is no
    # inference to be wrong about.
    #
    # Applied by ASSIGNMENT rather than accumulation, and set above
    # LOG_ODDS_BLOCKING so one contact immediately makes the cell opaque.
    #
    # Accumulating does not work, which is not obvious until measured: a cell
    # in a well-observed room sits at the clamp, LOG_ODDS_MIN, and adding one
    # contact's worth of evidence to -6.0 leaves it at -3.06 — still firmly
    # "free", so the object the robot is touching does not appear on the map at
    # all. It only ever worked for objects the sonar had never seen.
    #
    # Assignment is also the honest model. Those free readings were echoes,
    # inferred and often wrong; this is a switch closed by the object itself.
    # Direct evidence supersedes inference rather than being averaged with it.
    P_CONTACT = 0.95

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

        # Which cells the robot has physically run into.
        #
        # Kept apart from the log-odds because the two answer different
        # questions. The grid says how likely a cell is to be occupied; this
        # says how the evidence was obtained, and contact evidence is treated
        # differently downstream: an object the robot has touched is furniture
        # on the floor whether or not free space has been observed all the way
        # around it, which is what the hole-filling test would otherwise
        # require. One touch and a retreat never encloses anything.
        self.contact_mask = np.zeros((cells, cells), dtype=bool)

        # World coordinate of cell (0, 0). Starts centred so the robot has
        # room to drive in every direction from its origin.
        self.origin_x_m = -initial_size_m / 2.0
        self.origin_y_m = -initial_size_m / 2.0

        self.l_occupied = _log_odds(self.P_OCCUPIED)
        self.l_free = _log_odds(self.P_FREE)
        self.l_contact = _log_odds(self.P_CONTACT)

        self.updates_applied = 0
        self.rays_rejected = 0
        self.contacts_recorded = 0

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
        padding = ((pad_bottom, pad_top), (pad_left, pad_right))
        self.grid = np.pad(
            self.grid, padding, mode="constant", constant_values=0.0
        )
        # Grown in lockstep, or every recorded contact silently shifts by the
        # padding the moment the robot drives past the edge of the map.
        self.contact_mask = np.pad(
            self.contact_mask, padding, mode="constant", constant_values=False
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

        Cells already known solid are left alone. This circle is a little wider
        than the chassis and the bumper sits just outside it, so a robot
        stopped against an obstacle would otherwise scrub out the contact it
        has just recorded — at ten packets a second, within about half a
        second of touching it. The robot's own position is an estimate; a
        closed bumper switch is not, so the contact wins.
        """
        radius_cells = int(math.ceil(radius_m / self.resolution_m))
        centre_col, centre_row = self.world_to_cell(pose.x_m, pose.y_m)

        for d_row in range(-radius_cells, radius_cells + 1):
            for d_col in range(-radius_cells, radius_cells + 1):
                if d_col * d_col + d_row * d_row > radius_cells * radius_cells:
                    continue
                col, row = centre_col + d_col, centre_row + d_row
                if not self.in_bounds(col, row):
                    continue
                if self.grid[row, col] >= self.LOG_ODDS_BLOCKING:
                    continue
                self._update_cell(col, row, self.l_free)

    def mark_contact(
        self,
        pose: PoseEstimate,
        bumper_offset_m: float = 0.10,
        contact_width_m: float = 0.16,
    ) -> int:
        """Record something the robot has physically run into.

        This is what puts a red patch on the map for an obstacle the range
        sensors never saw. Ultrasonic misses plenty of real furniture: a chair
        leg narrower than the beam, a sofa that absorbs the pulse, anything
        angled enough to reflect the echo away, and everything below the
        sensor's mounting height. The bumper is what catches those, and until
        the contact is written into the grid it exists only in the controller's
        head — the robot avoids the object but never draws it.

        Geometry. The switch is on the robot's nose, so the object is
        `bumper_offset_m` ahead of the reported centre, not underneath it.
        Marking the centre would put the obstacle where the robot is standing,
        which is the one place it certainly is not.

        Only the contact patch is marked, not a guess at the object's extent.
        The robot knows it touched something roughly its own width across; it
        knows nothing about how far back the thing goes. Repeated contacts
        along a sofa fill the rest in honestly, one touch at a time.

        Returns the number of cells marked.
        """
        heading = math.radians(pose.heading_deg)
        contact_x = pose.x_m + bumper_offset_m * math.cos(heading)
        contact_y = pose.y_m + bumper_offset_m * math.sin(heading)

        self.ensure_contains(contact_x, contact_y)

        radius_cells = max(1, int(round(contact_width_m / 2.0 / self.resolution_m)))
        centre_col, centre_row = self.world_to_cell(contact_x, contact_y)

        marked = 0
        for d_row in range(-radius_cells, radius_cells + 1):
            for d_col in range(-radius_cells, radius_cells + 1):
                if d_col * d_col + d_row * d_row > radius_cells * radius_cells:
                    continue
                col, row = centre_col + d_col, centre_row + d_row
                if self.in_bounds(col, row):
                    # Assigned, not added — see P_CONTACT. `max` so a contact
                    # never argues DOWN a cell that other evidence has already
                    # made more certainly occupied.
                    self.grid[row, col] = max(
                        float(self.grid[row, col]), self.l_contact
                    )
                    self.contact_mask[row, col] = True
                    marked += 1

        self.contacts_recorded += 1
        self.updates_applied += marked
        return marked

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
        self.contact_mask[:] = False
        self.updates_applied = 0
        self.rays_rejected = 0
        self.contacts_recorded = 0
