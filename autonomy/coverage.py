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
    TURNING = "TURNING"            # about-facing to sweep the next row back
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
    turn_speed_dps: float = 60.0
    # How near the target heading counts as turned. Tighter than the heading
    # error the robot can actually hold would leave it turning forever; the
    # residual is taken out by the heading hold below while it drives.
    turn_tolerance_deg: float = 2.0

    # Give up on an about-face after this long and get on with the row.
    #
    # The turn is closed on the measured heading, so a heading that never
    # updates — a failed IMU, a stalled servo, a robot held by hand — would
    # otherwise leave it turning for ever. The heading hold takes out whatever
    # the abandoned turn left behind.
    turn_timeout_s: float = 8.0

    # Proportional heading hold along a row, in degrees per second per degree
    # of error, and its authority limit. Capped so a bad heading estimate
    # cannot spin the robot mid-row.
    heading_hold_gain: float = 1.2
    heading_hold_max_dps: float = 12.0

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

    # How close to the edge of the known room a row may run before it counts
    # as finished. Roughly the stopping distance, so the sweep turns rather
    # than relying on the wall to stop it.
    bounds_margin_m: float = 0.30

    max_rows: int = 40


def _wrap180(degrees: float) -> float:
    """Signed angle in (-180, 180]: the short way round to a heading."""
    return (degrees + 180.0) % 360.0 - 180.0


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

    The cycle for one row is: drive along it, sidestep across to the next, then
    about-face so the next row is driven back the other way.

    Why about-face rather than simply reversing
    -------------------------------------------
    The robot is holonomic and *could* drive backwards along the next row
    without turning. It must not, because the range sensors face forward: the
    ±90° pair sees the sides and nothing at all watches the rear. Reversing
    down a row would mean driving blind into whatever the previous pass did not
    already cover, which is exactly where the unmapped furniture is.

    Holonomy still earns its keep in the *sidestep*: the robot translates to
    the next row without rotating, so the row spacing is exactly the commanded
    distance. A differential base has to arc across, and the arc is what makes
    its rows drift out of parallel.

    Knowing where the room ends
    ---------------------------
    `bounds` is normally the bounding box measured by the perimeter lap. Given
    it, the sweep turns at the last row inside the room and stops when it has
    crossed to the far side. Without it the sweep relies on the sensors alone,
    which works but drives into a wall at the end of every row to find out it
    is there — and on real hardware each of those is a collision.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float] | None = None,
        config: CoverageConfig | None = None,
    ) -> None:
        self.config = config or CoverageConfig()
        # (min_x, min_y, max_x, max_y), normally from the measured outline.
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
        # The direction the rows run, taken from wherever the robot happens to
        # be facing when the sweep begins. Nothing here assumes rows run along
        # x: the perimeter lap ends with the robot alongside a wall, pointing
        # whichever way that wall runs, and squaring up to a global axis first
        # would be a turn for no reason.
        self._row_axis_deg: float | None = None
        # Absolute heading the current turn is aiming at. Turning is closed on
        # the measured heading rather than run open-loop for a fixed time: an
        # open-loop turn is short by one control step every row, and after six
        # rows the sweep is visibly fanning out rather than sweeping.
        self._target_heading_deg = 0.0
        self._turning_for_s = 0.0

        # Which way the sweep advances across the room, perpendicular to the
        # rows: +1 is 90° left of the row axis. Fixed for the whole sweep, and
        # chosen at the first step to head into the room rather than at the
        # nearest wall.
        #
        # Held in world terms rather than body terms because the robot
        # about-faces between rows: a constant body-frame sidestep advances one
        # row and then retraces it, which stops the sweep after two rows.
        self._advance_sign = 1.0
        self._last_x = 0.0
        self._last_y = 0.0
        self._last_tracked_x = 0.0
        self._last_tracked_y = 0.0
        self._no_progress_steps = 0
        self._avoids_this_row = 0

    # ── Sensing ───────────────────────────────────────────────────────────

    @staticmethod
    def _forward_distance(ranges: list[RangeReading], heading_window: float = 35.0) -> float:
        best = math.inf
        for reading in ranges:
            if reading.valid and abs(reading.angle_deg) <= heading_window:
                best = min(best, reading.distance_m)
        return best

    @staticmethod
    def _side_distance(ranges: list[RangeReading], sign: float) -> float:
        """Clearance to the robot's left (`sign` +1) or right (-1)."""
        centre = 90.0 * (1.0 if sign >= 0 else -1.0)
        best = math.inf
        for reading in ranges:
            if reading.valid and abs(reading.angle_deg - centre) <= 40.0:
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

        Counted only while sweeping. An about-face is a deliberate half-minute
        of standing still, and letting it feed this counter made every row
        after the first abort as "stalled" on its opening step.
        """
        self._last_x, self._last_y = x, y

        if self.state != CoverageState.SWEEPING:
            self._no_progress_steps = 0
            return

        if math.hypot(x - self._last_tracked_x, y - self._last_tracked_y) < (
            cfg.stall_distance_m * 0.1
        ):
            self._no_progress_steps += 1
        else:
            self._no_progress_steps = 0
        self._last_tracked_x, self._last_tracked_y = x, y

    # ── Decision ──────────────────────────────────────────────────────────

    def _decide(self, ranges, x_m, y_m, heading_deg, cfg, dt_s) -> CoverageCommand:
        if self.state == CoverageState.RECOVERING:
            return self._recover(cfg, dt_s)

        if self.state == CoverageState.STARTING:
            self.row_y = y_m
            self._start_x = x_m
            self._row_axis_deg = heading_deg
            self._choose_advance(x_m, y_m)
            self.state = CoverageState.SWEEPING
            self.stats.rows_started = 1

        if self.state == CoverageState.STEPPING:
            return self._step_across(ranges, x_m, y_m, heading_deg, cfg, dt_s)

        if self.state == CoverageState.TURNING:
            return self._turn(heading_deg, cfg, dt_s)

        if self.state == CoverageState.AVOIDING:
            return self._avoid(heading_deg, cfg, dt_s)

        return self._sweep(ranges, x_m, y_m, heading_deg, cfg)

    # ── Bounds ────────────────────────────────────────────────────────────

    def _outside(self, x: float, y: float, margin: float) -> bool:
        """Whether a point is outside the known room by more than `margin`."""
        if self.bounds is None:
            return False
        min_x, min_y, max_x, max_y = self.bounds
        return (
            x < min_x + margin
            or x > max_x - margin
            or y < min_y + margin
            or y > max_y - margin
        )

    def _room_to_advance(self, x: float, y: float, margin: float) -> float:
        """How far the sweep can still step across before leaving the room.

        Measured along the advance direction only. Testing the whole bounding
        box instead ends the sweep as soon as the robot is near *any* wall —
        and it is near one at the end of every single row, which is exactly
        where this gets asked. Rooms were being abandoned a metre short.
        """
        if self.bounds is None:
            return math.inf

        min_x, min_y, max_x, max_y = self.bounds
        step_x, step_y = self._advance_vector()

        limits = []
        if abs(step_x) > 1e-9:
            edge = (max_x - margin) if step_x > 0 else (min_x + margin)
            limits.append((edge - x) / step_x)
        if abs(step_y) > 1e-9:
            edge = (max_y - margin) if step_y > 0 else (min_y + margin)
            limits.append((edge - y) / step_y)

        return min(limits) if limits else math.inf

    def _advance_vector(self) -> tuple[float, float]:
        """World-frame unit vector from one row to the next."""
        axis = math.radians((self._row_axis_deg or 0.0) + 90.0)
        return (
            self._advance_sign * math.cos(axis),
            self._advance_sign * math.sin(axis),
        )

    def _choose_advance(self, x_m: float, y_m: float) -> None:
        """Point the sweep at the bulk of the room, not at the nearest wall.

        A sweep starts wherever the perimeter lap left the robot, which is
        alongside a wall. Advancing towards that wall would end the sweep after
        one row, so the side with more room to cross wins.
        """
        if self.bounds is None:
            return

        min_x, min_y, max_x, max_y = self.bounds
        centre_x, centre_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

        self._advance_sign = 1.0
        step_x, step_y = self._advance_vector()
        # Positive if stepping this way heads towards the middle of the room.
        towards_centre = (centre_x - x_m) * step_x + (centre_y - y_m) * step_y
        if towards_centre < 0:
            self._advance_sign = -1.0

    def _lateral_sign(self, heading_deg: float) -> float:
        """Body-frame sidestep direction that advances the sweep in world terms.

        The robot's own left points one way along the advance axis when it
        faces up the row and the other way once it has about-faced. Deriving
        the sign from the measured heading each time is what keeps the sweep
        marching across the room instead of oscillating between two rows.
        """
        left = math.radians(heading_deg + 90.0)
        step_x, step_y = self._advance_vector()
        # Project the robot's own left onto the direction the sweep advances.
        return 1.0 if (math.cos(left) * step_x + math.sin(left) * step_y) >= 0 else -1.0

    def _sweep(self, ranges, x_m, y_m, heading_deg, cfg) -> CoverageCommand:
        forward = self._forward_distance(ranges)
        angle = math.radians(heading_deg)

        # The end of the row, known from the measured outline rather than
        # discovered by driving into the wall. On real hardware the difference
        # is one collision per row.
        probe_x = x_m + cfg.bounds_margin_m * math.cos(angle)
        probe_y = y_m + cfg.bounds_margin_m * math.sin(angle)
        if self._outside(probe_x, probe_y, margin=0.0):
            self._begin_next_row(x_m, y_m, cfg)
            return CoverageCommand(0, 0, 0, self.state, "end of row")

        # Something remembered from a previous pass, even if unseen now.
        if self.known_obstacle_ahead(x_m, y_m, heading_deg):
            if self._avoids_this_row >= cfg.max_avoids_per_row:
                # This row is too cluttered to sweep. Abandoning it is better
                # than ping-ponging between avoiding and sweeping forever.
                self.stats.rows_skipped += 1
                self._begin_next_row(x_m, y_m, cfg)
                return CoverageCommand(0, 0, 0, self.state, "row too cluttered")

            self._avoids_this_row += 1
            self.state = CoverageState.AVOIDING
            self._avoid_remaining = cfg.avoid_step_m
            return CoverageCommand(0, 0, 0, self.state, "known obstacle ahead")

        if forward < cfg.obstacle_critical_m:
            self._register_obstacle(
                x_m + forward * math.cos(angle),
                y_m + forward * math.sin(angle),
                from_collision=False,
            )
            self.state = CoverageState.RECOVERING
            self._recover_remaining = cfg.recover_distance_m
            return CoverageCommand(-cfg.reverse_speed_mps, 0, 0, self.state, "too close")

        if forward < cfg.obstacle_stop_m:
            self._register_obstacle(
                x_m + forward * math.cos(angle),
                y_m + forward * math.sin(angle),
                from_collision=False,
            )
            # A wall at the end of the row versus a table in the middle looks
            # identical to the sensor. Both mean "this row is done ahead", so
            # both step to the next row; the difference only matters for what
            # gets drawn, and that comes from the occupancy grid.
            self._begin_next_row(x_m, y_m, cfg)
            return CoverageCommand(0, 0, 0, self.state, "obstacle or wall ahead")

        if self._no_progress_steps > cfg.stall_patience_steps:
            self._no_progress_steps = 0
            self._register_obstacle(x_m, y_m, from_collision=False)
            self._begin_next_row(x_m, y_m, cfg)
            return CoverageCommand(0, 0, 0, self.state, "stalled; moving on")

        # Slow down as something gets nearer, so a late detection is still
        # stoppable rather than becoming a collision.
        speed = (
            cfg.approach_speed_mps
            if forward < cfg.obstacle_stop_m * 2
            else cfg.cruise_speed_mps
        )
        return CoverageCommand(
            speed,
            0,
            self._heading_hold(heading_deg, cfg),
            CoverageState.SWEEPING,
            "sweeping row",
        )

    def _heading_hold(self, heading_deg: float, cfg: CoverageConfig) -> float:
        """A gentle correction back onto the row axis while driving.

        Whatever the about-face leaves behind — a degree or two — becomes
        lateral drift multiplied by the length of the row: two degrees over a
        five metre row is nearly 20 cm, which eats over half of a 35 cm row
        spacing and leaves strips of floor unswept between passes.

        Proportional only. There is no integral term because the error being
        corrected is a fixed offset, not a persistent disturbance, and no
        derivative term because the heading estimate is far too noisy to
        differentiate.
        """
        if self._row_axis_deg is None:
            return 0.0

        # Whichever end of the axis the robot is currently driving.
        error = _wrap180(self._row_axis_deg - heading_deg)
        if abs(error) > 90.0:
            error = _wrap180(self._row_axis_deg + 180.0 - heading_deg)

        return max(
            -cfg.heading_hold_max_dps,
            min(cfg.heading_hold_max_dps, error * cfg.heading_hold_gain),
        )

    def _begin_next_row(self, x_m: float, y_m: float, cfg: CoverageConfig) -> None:
        self.stats.rows_completed += 1
        self.row_index += 1

        if self.row_index >= cfg.max_rows:
            self.state = CoverageState.FINISHED
            return

        # Has the sweep run out of room to advance into? Checked here rather
        # than after the sidestep, so the robot never commits to a step that
        # would put it in a wall.
        if self._room_to_advance(x_m, y_m, cfg.bounds_margin_m * 0.5) < (
            cfg.row_spacing_m
        ):
            self.state = CoverageState.FINISHED
            return

        self.state = CoverageState.STEPPING
        self._step_remaining = cfg.row_spacing_m
        self.sweep_direction *= -1.0
        self.stats.rows_started += 1
        # Each row gets a fresh avoidance budget.
        self._avoids_this_row = 0

    def _step_across(
        self,
        ranges: list[RangeReading],
        x_m: float,
        y_m: float,
        heading_deg: float,
        cfg: CoverageConfig,
        dt_s: float,
    ) -> CoverageCommand:
        """Sidestep to the next row, then about-face to drive it back.

        Strafing rather than arcing across: the robot is holonomic, so the row
        spacing is exactly the distance commanded. The about-face that follows
        is not a limitation of the base but of the sensors, which all face
        forward — see the class docstring.
        """
        moved = cfg.cruise_speed_mps * dt_s
        self._step_remaining -= moved

        if self._step_remaining <= 0:
            self.state = CoverageState.TURNING
            self._turning_for_s = 0.0
            # Aim at whichever end of the row axis the robot is NOT currently
            # facing. Deriving it from the measured heading rather than from a
            # count of rows matters because the count also advances on
            # recoveries and skipped rows: parity drifts out of step with the
            # robot, and the sweep then "turns" to the heading it already has
            # and drives the row it has just finished all over again.
            axis = self._row_axis_deg if self._row_axis_deg is not None else heading_deg
            forward_error = abs(_wrap180(axis - heading_deg))
            self._target_heading_deg = (
                axis if forward_error > 90.0 else axis + 180.0
            ) % 360.0
            return CoverageCommand(0, 0, 0, self.state, "reached the next row")

        sign = self._lateral_sign(heading_deg)

        # The sensors have the final say on whether there is another row.
        #
        # `bounds` comes from the perimeter lap and is expressed in the pose
        # estimate's own frame, so it drifts with the pose over a long sweep —
        # by the far side of the room the robot can believe there is space left
        # where there is a wall. The range readings do not drift, so a wall
        # this close alongside means the sweep has genuinely finished,
        # whatever the arithmetic says.
        if self._side_distance(ranges, sign) < cfg.row_spacing_m:
            self.state = CoverageState.FINISHED
            return CoverageCommand(0, 0, 0, self.state, "wall alongside; sweep done")

        # Abort mid-step if the room runs out. The prediction in
        # `_begin_next_row` is made before the step starts and can be beaten by
        # drift over the course of it.
        if self._room_to_advance(x_m, y_m, 0.0) <= 0.0:
            self.state = CoverageState.FINISHED
            return CoverageCommand(0, 0, 0, self.state, "no room for another row")

        return CoverageCommand(
            0.0,
            cfg.cruise_speed_mps * sign,
            0.0,
            CoverageState.STEPPING,
            "stepping to next row",
        )

    def _turn(
        self, heading_deg: float, cfg: CoverageConfig, dt_s: float
    ) -> CoverageCommand:
        """About-face so the next row is driven with the sensors leading.

        Closed on the measured heading. Commanding a fixed rate for a fixed
        time instead leaves the turn a little short every row — one control
        step's worth — and six rows later the sweep is fanning out across the
        room rather than covering it in parallel strips.
        """
        error = _wrap180(self._target_heading_deg - heading_deg)
        self._turning_for_s += dt_s

        if abs(error) <= cfg.turn_tolerance_deg:
            self.state = CoverageState.SWEEPING
            return CoverageCommand(0, 0, 0, self.state, "row start")

        if self._turning_for_s > cfg.turn_timeout_s:
            # Never spin in place indefinitely because a heading estimate has
            # stopped arriving. Better a row driven slightly askew, which the
            # heading hold will straighten, than a robot that never moves again.
            self.state = CoverageState.SWEEPING
            return CoverageCommand(0, 0, 0, self.state, "turn timed out; carrying on")

        # Ease off near the target so the turn settles instead of hunting
        # around it, which on a holonomic base it will otherwise do. The floor
        # is low enough that the last step cannot overshoot by more than the
        # tolerance — otherwise every row lands slightly past its target, all
        # in the same direction, and the error accumulates across the sweep.
        rate = cfg.turn_speed_dps * min(1.0, max(0.12, abs(error) / 45.0))
        return CoverageCommand(
            0.0,
            0.0,
            math.copysign(rate, error),
            CoverageState.TURNING,
            "about-facing",
        )

    def _avoid(
        self, heading_deg: float, cfg: CoverageConfig, dt_s: float
    ) -> CoverageCommand:
        """Sidestep round a known obstacle, then carry on along the row."""
        moved = cfg.cruise_speed_mps * dt_s
        self._avoid_remaining -= moved

        if self._avoid_remaining <= 0:
            self.state = CoverageState.SWEEPING
            return CoverageCommand(0, 0, 0, self.state, "resuming row")

        return CoverageCommand(
            0.0,
            cfg.cruise_speed_mps * 0.7 * self._lateral_sign(heading_deg),
            0.0,
            CoverageState.AVOIDING,
            "going round",
        )

    def _recover(self, cfg: CoverageConfig, dt_s: float) -> CoverageCommand:
        """Back away from a collision before doing anything else."""
        moved = cfg.reverse_speed_mps * dt_s
        self._recover_remaining -= moved

        if self._recover_remaining <= 0:
            # Move on rather than retry: whatever was hit is still there, and
            # the obstacle is now recorded so the row will route round it.
            self._begin_next_row(self._last_x, self._last_y, cfg)
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
