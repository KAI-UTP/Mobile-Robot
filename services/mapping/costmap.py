"""Inflating obstacles by the robot's own size, the way Nav2 does.

The idea, borrowed rather than the framework
--------------------------------------------
Nav2 draws a halo around every obstacle in its costmap — the coloured rings you
see around each traffic cone in an RViz screenshot. This module is that idea,
without ROS.

Why it matters, and it is not decoration. A planner reasons about the robot as
a *point*, because reasoning about a rotating rectangle against every obstacle
is expensive and fiddly. That is only safe if the obstacles are first grown by
the robot's own radius: a point that stays out of the grown region is a robot
whose whole body stays out of the real one. Skip it and the robot clips corners
it planned straight through — which is exactly the failure this project kept
producing by hand, one obstacle at a time.

Three bands, following Nav2's vocabulary
----------------------------------------
    LETHAL      the obstacle itself. The robot's centre here means contact.
    INSCRIBED   within one robot radius. Any pose here is in collision
                whatever the heading, so it is as forbidden as LETHAL.
    INFLATED    beyond the radius, decaying with distance. Not forbidden —
                *discouraged*. This is what makes a robot prefer the middle of
                a doorway to scraping the frame.

The decay is exponential, again as Nav2 does it, because what matters is a
steep penalty close in and a gentle preference further out. A linear ramp
either makes distant floor needlessly expensive or leaves the near margin too
cheap to respect.

Why this project needs it specifically
--------------------------------------
This robot maps by *touching* things. Every obstacle it knows about, it knows
because it drove into it. Inflation is what turns "I hit something here" into
"do not come within a robot radius of here", which is the difference between a
map that records history and a map that prevents a repeat.
"""

from __future__ import annotations

import math

import numpy as np

# Cost values, following Nav2's 0-254 convention so the numbers mean the same
# thing to anyone who has read that codebase.
FREE = 0
INSCRIBED = 253
LETHAL = 254
UNKNOWN = 255


class Costmap:
    """An occupancy grid inflated by the robot's radius.

    Deliberately separate from `OccupancyGrid`. That class answers "what is
    there?", which is a question about evidence. This one answers "where may
    the robot go?", which is a question about the robot — change the chassis
    and this changes while the evidence does not.
    """

    def __init__(
        self,
        robot_radius_m: float = 0.11,
        inflation_radius_m: float = 0.35,
        cost_scaling_factor: float = 3.0,
    ) -> None:
        if inflation_radius_m < robot_radius_m:
            raise ValueError(
                "inflation_radius_m must be at least the robot radius, or the "
                "inscribed band would extend past the inflated one"
            )

        self.robot_radius_m = robot_radius_m
        self.inflation_radius_m = inflation_radius_m
        # How sharply cost falls with distance. Higher hugs obstacles more
        # tightly; Nav2's default of 3.0 is a reasonable middle and is kept so
        # the tuning advice written for Nav2 applies here too.
        self.cost_scaling_factor = cost_scaling_factor

    # ── Building ──────────────────────────────────────────────────────────

    def build(self, occupied: np.ndarray, resolution_m: float) -> np.ndarray:
        """Inflate a boolean occupancy mask into a 0-254 cost field."""
        cost = np.zeros(occupied.shape, dtype=np.uint8)
        if not occupied.any():
            return cost

        distance_m = self._distance_to_obstacle(occupied, resolution_m)

        cost[distance_m <= 0.0] = LETHAL
        inscribed = (distance_m > 0.0) & (distance_m <= self.robot_radius_m)
        cost[inscribed] = INSCRIBED

        band = (distance_m > self.robot_radius_m) & (
            distance_m <= self.inflation_radius_m
        )
        if band.any():
            # Nav2's exponential decay, measured from the inscribed edge so the
            # curve starts where "definitely colliding" stops.
            decay = np.exp(
                -self.cost_scaling_factor
                * (distance_m[band] - self.robot_radius_m)
            )
            cost[band] = (INSCRIBED - 1) * decay

        return cost

    def _distance_to_obstacle(
        self, occupied: np.ndarray, resolution_m: float
    ) -> np.ndarray:
        """Metres from each cell to the nearest occupied cell.

        A two-pass chamfer transform rather than an exact Euclidean one. The
        exact version needs SciPy, which this project deliberately does not
        depend on, and the chamfer approximation is within a few percent —
        far inside the accuracy of a map built by bumping into things.
        """
        height, width = occupied.shape
        # Only cells within the inflation radius can matter, so the sweep can
        # stop there instead of computing a distance field for the whole map.
        limit = int(math.ceil(self.inflation_radius_m / resolution_m)) + 1

        large = float(limit + 2)
        distance = np.full(occupied.shape, large, dtype=np.float32)
        distance[occupied] = 0.0

        # Chamfer weights: 1 for an edge step, sqrt(2) for a diagonal.
        straight, diagonal = 1.0, math.sqrt(2.0)

        for _ in range(2):
            # Forward pass, then the same backwards. Two passes is what makes
            # a chamfer transform correct in both directions.
            for row in range(height):
                for col in range(width):
                    self._relax(distance, row, col, straight, diagonal, forward=True)
            for row in range(height - 1, -1, -1):
                for col in range(width - 1, -1, -1):
                    self._relax(distance, row, col, straight, diagonal, forward=False)
            break   # one forward + one backward pass is the standard sweep

        return distance * resolution_m

    @staticmethod
    def _relax(distance, row, col, straight, diagonal, forward: bool) -> None:
        height, width = distance.shape
        best = distance[row, col]
        if best == 0.0:
            return

        offsets = (
            ((-1, -1, diagonal), (-1, 0, straight), (-1, 1, diagonal), (0, -1, straight))
            if forward
            else ((1, 1, diagonal), (1, 0, straight), (1, -1, diagonal), (0, 1, straight))
        )
        for d_row, d_col, weight in offsets:
            r, c = row + d_row, col + d_col
            if 0 <= r < height and 0 <= c < width:
                candidate = distance[r, c] + weight
                if candidate < best:
                    best = candidate
        distance[row, col] = best

    # ── Reading ───────────────────────────────────────────────────────────

    @staticmethod
    def is_forbidden(cost: int) -> bool:
        """Whether a planner must refuse this cell outright.

        INSCRIBED counts as forbidden, not merely expensive. A cell within one
        robot radius of an obstacle is in collision at every heading, so there
        is no orientation that rescues it.
        """
        return cost >= INSCRIBED

    def summarise(self, cost: np.ndarray, resolution_m: float) -> dict:
        cell_area = resolution_m**2
        return {
            "lethal_m2": round(float((cost == LETHAL).sum()) * cell_area, 3),
            "inscribed_m2": round(float((cost == INSCRIBED).sum()) * cell_area, 3),
            "inflated_m2": round(
                float(((cost > FREE) & (cost < INSCRIBED)).sum()) * cell_area, 3
            ),
            "robot_radius_m": self.robot_radius_m,
            "inflation_radius_m": self.inflation_radius_m,
        }
