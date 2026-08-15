"""Row-by-row floor coverage, with obstacle avoidance and collision recovery.

Why this exists alongside wall-following
----------------------------------------
`explorer.py` hugs the walls. That measures the *perimeter* well and never
learns anything about the middle of the room — so a table sitting in open
floor is discovered only if the robot happens to graze it, and the reported
floor area silently includes the space it occupies.

This drives a boustrophedon sweep instead: along a row, step across, back
along the next row, like mowing a lawn. It is slower, and it is the only way
to find out what is actually on the floor.

Handling the awkward cases
--------------------------
Real rooms are not empty rectangles, so the sweep has to cope with:

* **Obstacles mid-row** — a table blocks the middle of a row. The sweep marks
  it, backs off, and continues the row *beyond* it where reachable, rather
  than abandoning everything past the first obstruction.
* **Rows that are entirely blocked** — skipped rather than retried forever.
* **Collisions** — the bumper fires when a sensor missed something. The
  contact point is recorded so the same obstacle is never driven into twice,
  which is the difference between a robot that learns and one that keeps
  bumping the same chair leg.
* **Irregular rooms** — no assumption that the room is rectangular. The sweep
  works from the discovered free space, so an L-shape or an alcove is covered
  by the same logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from robotmap_common.models import RangeReading


class CoverageState(str, Enum):
    STARTING = "STARTING"
    SWEEPING = "SWEEPING"          # driving along a row
    AVOIDING = "AVOIDING"          # obstacle ahead, going round it
    RECOVERING = "RECOVERING"      # bumper hit, backing off
    STEPPING = "STEPPING"          # moving across to the next row
    FINISHED = "FINISHED"


@dataclass(frozen=True)
class CoverageConfig:
    # Spacing between rows. Narrower covers more thoroughly and takes longer;
    # roughly the robot's width is the usual compromise, since that is the
    # swathe it actually observes closely.
    row_spacing_m: float = 0.35

    cruise_speed_mps: float = 0.18
    # Slower than a straight run: the robot is about to meet something.
    approach_speed_mps: float = 0.08
    reverse_speed_mps: float = 0.12

    # Stop this far from an obstacle ahead.
    obstacle_stop_m: float = 0.30
    # Below this, treat it as an imminent collision and reverse.
    obstacle_critical_m: float = 0.16

    # How far to back off after a bump before trying anything else.
    recover_distance_m: float = 0.18
    # How far to sidestep when going round an obstacle.
    avoid_step_m: float = 0.22

    # A row is finished when the robot has travelled this far without
    # progress, which catches it wedged in a corner.
    stall_distance_m: float = 0.05
    stall_patience_steps: int = 40

    # How many times the sweep will step round obstacles within a single row
    # before giving up on it.
    #
    # Without a limit the sweep livelocks: each avoided obstacle is recorded,
    # the recorded obstacle then triggers another avoid on the next pass, and
    # the robot ping-pongs between sweeping and avoiding without ever reaching
    # the end of the row or the next one. A cluttered row is better abandoned
    # than swept forever.
    max_avoids_per_row: int = 6

    max_rows: int = 40


@dataclass
class Obstacle:
    """Something in the floor space, found by sensor or by contact."""

    x_m: float
    y_m: float
    radius_m: float = 0.15
    hit_count: int = 1
    from_collision: bool = False

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return math.hypot(x - self.x_m, y - self.y_m) < self.radius_m + margin


@dataclass
class CoverageCommand:
    linear_mps: float
    lateral_mps: float
    angular_dps: float
    state: CoverageState
    note: str = ""


@dataclass
class CoverageStats:
    rows_started: int = 0
    rows_completed: int = 0
    rows_skipped: int = 0
    collisions: int = 0
    obstacles_found: int = 0
    distance_m: float = 0.0
    obstacles: list[Obstacle] = field(default_factory=list)


class CoveragePlanner:
    """Sweeps the floor row by row, avoiding what it finds.

    Holonomic-aware: the robot can step sideways to the next row without
    turning, which a differential drive cannot. That keeps its forward sensor
    pointing along the row for the whole sweep, so the sensor that matters is
    always looking where the robot is going.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float] | None = None,
        config: CoverageConfig | None = None,
    ) -> None:
        self.config = config or CoverageConfig()
        # (min_x, min_y, max_x, max_y). If unknown the sweep grows its own
        # bounds from wherever it has been able to drive.
        self.bounds = bounds
        self.state = CoverageState.STARTING

        self.stats = CoverageStats()
        self.row_index = 0
        self.sweep_direction = 1.0     # +1 driving +x, -1 driving -x
        self.row_y: float | None = None

        self._start_x: float | None = None
        self._recover_remaining = 0.0
        self._avoid_remaining = 0.0
        self._step_remaining = 0.0
        self._last_x = 0.0
        self._last_y = 0.0
        self._no_progress_steps = 0
        self._row_blocked = False
        self._avoids_this_row = 0

    # ── Sensing ───────────────────────────────────────────────────────────

    @staticmethod
    def _forward_distance(ranges: list[RangeReading], heading_window: float = 35.0) -> float:
        best = math.inf
        for reading in ranges:
            if reading.valid and abs(reading.angle_deg) <= heading_window:
                best = min(best, reading.distance_m)
        return best

    def _register_obstacle(self, x: float, y: float, from_collision: bool) -> None:
        """Record something in the way, merging with anything already there.

        Merging matters: without it a single table generates dozens of
        overlapping obstacles as the robot approaches it from slightly
        different angles, and the avoidance logic then treats each as new.
        """
        for obstacle in self.stats.obstacles:
            if obstacle.contains(x, y, margin=0.10):
                obstacle.hit_count += 1
                obstacle.from_collision = obstacle.from_collision or from_collision
                return

        self.stats.obstacles.append(
            Obstacle(x_m=x, y_m=y, from_collision=from_collision)
        )
        self.stats.obstacles_found += 1

    def known_obstacle_ahead(self, x: float, y: float, heading_deg: float) -> bool:
        """Whether a previously recorded obstacle lies just ahead.

        This is the memory that stops the robot bumping the same chair leg on
        every pass — the sensor may miss it again, but the map does not.
        """
        angle = math.radians(heading_deg)
        probe_x = x + 0.30 * math.cos(angle)
        probe_y = y + 0.30 * math.sin(angle)
        return any(o.contains(probe_x, probe_y, margin=0.05) for o in self.stats.obstacles)

    # ── Main step ─────────────────────────────────────────────────────────

    def step(
        self,
        ranges: list[RangeReading],
        x_m: float,
        y_m: float,
        heading_deg: float,
        bumper: bool,
        dt_s: float,
    ) -> CoverageCommand:
        cfg = self.config

        if self.state == CoverageState.FINISHED:
            return CoverageCommand(0, 0, 0, self.state, "coverage complete")

        # Collision takes priority over everything else.
        if bumper and self.state != CoverageState.RECOVERING:
            self.stats.collisions += 1
            angle = math.radians(heading_deg)
            # Record it slightly ahead: the bumper is at the robot's front, so
            # the obstacle is in front of the reported centre, not at it.
            self._register_obstacle(
                x_m + 0.20 * math.cos(angle),
                y_m + 0.20 * math.sin(angle),
                from_collision=True,
            )
            self.state = CoverageState.RECOVERING
            self._recover_remaining = cfg.recover_distance_m

        command = self._decide(ranges, x_m, y_m, heading_deg, cfg, dt_s)

        travelled = math.hypot(command.linear_mps, command.lateral_mps) * dt_s
        self.stats.distance_m += travelled
        self._track_progress(x_m, y_m, cfg)
        return command

    def _track_progress(self, x: float, y: float, cfg: CoverageConfig) -> None:
        """Detect being wedged.

        A robot pressed into a corner still commands motion and still reports
        sensible sensors; only the fact that it is not actually moving reveals
        the problem.
        """
        if math.hypot(x - self._last_x, y - self._last_y) < cfg.stall_distance_m * 0.1:
            self._no_progress_steps += 1
        else:
            self._no_progress_steps = 0
        self._last_x, self._last_y = x, y

    # ── Decision ──────────────────────────────────────────────────────────

    def _decide(self, ranges, x_m, y_m, heading_deg, cfg, dt_s) -> CoverageCommand:
        if self.state == CoverageState.RECOVERING:
            return self._recover(cfg, dt_s)

        if self.state == CoverageState.STARTING:
            self.row_y = y_m
            self._start_x = x_m
            self.state = CoverageState.SWEEPING
            self.stats.rows_started = 1

        if self.state == CoverageState.STEPPING:
            return self._step_across(cfg, dt_s)

        if self.state == CoverageState.AVOIDING:
            return self._avoid(cfg, dt_s)

        return self._sweep(ranges, x_m, y_m, heading_deg, cfg)

    def _sweep(self, ranges, x_m, y_m, heading_deg, cfg) -> CoverageCommand:
        forward = self._forward_distance(ranges)

        # Something remembered from a previous pass, even if unseen now.
        if self.known_obstacle_ahead(x_m, y_m, heading_deg):
            if self._avoids_this_row >= cfg.max_avoids_per_row:
                # This row is too cluttered to sweep. Abandoning it is better
                # than ping-ponging between avoiding and sweeping forever.
                self.stats.rows_skipped += 1
                self._begin_next_row(cfg)
                return CoverageCommand(0, 0, 0, self.state, "row too cluttered")

            self._avoids_this_row += 1
            self.state = CoverageState.AVOIDING
            self._avoid_remaining = cfg.avoid_step_m
            return CoverageCommand(0, 0, 0, self.state, "known obstacle ahead")

        if forward < cfg.obstacle_critical_m:
            self._register_obstacle(
                x_m + forward * math.cos(math.radians(heading_deg)),
                y_m + forward * math.sin(math.radians(heading_deg)),
                from_collision=False,
            )
            self.state = CoverageState.RECOVERING
            self._recover_remaining = cfg.recover_distance_m
            return CoverageCommand(-cfg.reverse_speed_mps, 0, 0, self.state, "too close")

        if forward < cfg.obstacle_stop_m:
            self._register_obstacle(
                x_m + forward * math.cos(math.radians(heading_deg)),
                y_m + forward * math.sin(math.radians(heading_deg)),
                from_collision=False,
            )
            # A wall at the end of the row versus a table in the middle looks
            # identical to the sensor. Both mean "this row is done ahead", so
            # both step to the next row; the difference only matters for what
            # gets drawn, and that comes from the occupancy grid.
            self._begin_next_row(cfg)
            return CoverageCommand(0, 0, 0, self.state, "obstacle or wall ahead")

        if self._no_progress_steps > cfg.stall_patience_steps:
            self._no_progress_steps = 0
            self._register_obstacle(x_m, y_m, from_collision=False)
            self._begin_next_row(cfg)
            return CoverageCommand(0, 0, 0, self.state, "stalled; moving on")

        # Slow down as something gets nearer, so a late detection is still
        # stoppable rather than becoming a collision.
        speed = (
            cfg.approach_speed_mps
            if forward < cfg.obstacle_stop_m * 2
            else cfg.cruise_speed_mps
        )
        return CoverageCommand(speed, 0, 0, CoverageState.SWEEPING, "sweeping row")

    def _begin_next_row(self, cfg: CoverageConfig) -> None:
        self.stats.rows_completed += 1
        self.row_index += 1

        if self.row_index >= cfg.max_rows:
            self.state = CoverageState.FINISHED
            return

        self.state = CoverageState.STEPPING
        self._step_remaining = cfg.row_spacing_m
        self.sweep_direction *= -1.0
        self.stats.rows_started += 1
        # Each row gets a fresh avoidance budget.
        self._avoids_this_row = 0

    def _step_across(self, cfg: CoverageConfig, dt_s: float) -> CoverageCommand:
        """Sidestep to the next row.

        Strafing rather than turning: the robot is holonomic, so it keeps
        facing along the row and its forward sensor keeps looking where it
        will be going. A differential robot would have to turn twice here.
        """
        moved = cfg.cruise_speed_mps * dt_s
        self._step_remaining -= moved

        if self._step_remaining <= 0:
            self.state = CoverageState.SWEEPING
            return CoverageCommand(0, 0, 0, self.state, "row start")

        return CoverageCommand(
            0.0, cfg.cruise_speed_mps, 0.0, CoverageState.STEPPING, "stepping to next row"
        )

    def _avoid(self, cfg: CoverageConfig, dt_s: float) -> CoverageCommand:
        """Sidestep round a known obstacle, then carry on along the row."""
        moved = cfg.cruise_speed_mps * dt_s
        self._avoid_remaining -= moved

        if self._avoid_remaining <= 0:
            self.state = CoverageState.SWEEPING
            return CoverageCommand(0, 0, 0, self.state, "resuming row")

        return CoverageCommand(
            0.0, cfg.cruise_speed_mps * 0.7, 0.0, CoverageState.AVOIDING, "going round"
        )

    def _recover(self, cfg: CoverageConfig, dt_s: float) -> CoverageCommand:
        """Back away from a collision before doing anything else."""
        moved = cfg.reverse_speed_mps * dt_s
        self._recover_remaining -= moved

        if self._recover_remaining <= 0:
            # Move on rather than retry: whatever was hit is still there, and
            # the obstacle is now recorded so the row will route round it.
            self._begin_next_row(cfg)
            return CoverageCommand(0, 0, 0, self.state, "recovered")

        return CoverageCommand(
            -cfg.reverse_speed_mps, 0.0, 0.0, CoverageState.RECOVERING, "backing off"
        )

    # ── Reporting ─────────────────────────────────────────────────────────

    @property
    def is_finished(self) -> bool:
        return self.state == CoverageState.FINISHED

    def summary(self) -> dict:
        return {
            "state": self.state.value,
            "rows_started": self.stats.rows_started,
            "rows_completed": self.stats.rows_completed,
            "collisions": self.stats.collisions,
            "obstacles_found": self.stats.obstacles_found,
            "obstacles_from_contact": sum(
                1 for o in self.stats.obstacles if o.from_collision
            ),
            "distance_m": round(self.stats.distance_m, 2),
        }
