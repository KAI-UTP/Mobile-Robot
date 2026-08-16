"""Wall-following exploration controller.

Why wall following
------------------
To draw a room's outline you must observe its entire boundary, and the
shortest reliable way to do that with cheap forward-facing sensors is to hug
the wall all the way round. A lawnmower pattern covers the *floor* well but
observes the walls only from a distance and at bad incidence angles, which is
where sonar is weakest.

The controller is a bang-bang loop with hysteresis rather than a PID: with
5 cm-quantised ultrasonic readings arriving at 10 Hz, a PID's derivative term
is mostly differentiated noise. This is simpler and, at these speeds, steadier.

Runs equally well against the simulator and against real hardware, since it
only consumes `RangeReading`s and emits velocity commands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from robotmap_common.models import RangeReading


class ExploreState(str, Enum):
    SEEKING_WALL = "SEEKING_WALL"  # driving out to find any wall
    FOLLOWING = "FOLLOWING"  # tracking the wall on the chosen side
    TURNING_CORNER = "TURNING_CORNER"  # inside corner: wall ahead
    ROUNDING_CORNER = "ROUNDING_CORNER"  # outside corner: wall fell away
    RECOVERING = "RECOVERING"  # ran into something the sensors missed
    FINISHED = "FINISHED"


@dataclass(frozen=True)
class ExploreConfig:
    target_wall_distance_m: float = 0.35
    # Hysteresis band. Correcting inside this band would make the robot
    # oscillate against sensor noise instead of driving smoothly.
    tolerance_m: float = 0.06

    front_stop_distance_m: float = 0.35
    # Corner turns end only once the front is clear by a wider margin than
    # the one that started them. Without this gap the robot chatters between
    # turning and driving on the same noisy reading.
    front_clear_distance_m: float = 0.55

    # A side reading closer than this while seeking counts as finding a wall.
    wall_detect_distance_m: float = 0.65
    # A side reading beyond this means the wall has ended (outside corner).
    wall_lost_distance_m: float = 0.85

    cruise_speed_mps: float = 0.18
    turn_speed_dps: float = 55.0
    correction_dps: float = 30.0

    follow_right_wall: bool = True

    # Loop-closure detection: once the robot has driven at least this far and
    # returned near where wall-following began, the boundary is complete.
    min_loop_distance_m: float = 3.0
    loop_close_radius_m: float = 0.5

    # ...and has actually been round something.
    #
    # Distance and proximity alone are far too weak a test. Measured in the
    # furnished room, the robot drove a 3.18 m hook near the bin, curled back
    # to within 0.49 m of where following began, and declared the boundary
    # complete — 10.19 m2 of a 27 m2 room, 6.9 % of the floor ever visited.
    # Both thresholds were satisfied by a curl that went nowhere.
    #
    # A circuit of a room turns through a full revolution; a hook does not.
    # Accumulating the path's own course change costs nothing, needs no
    # heading input, and makes no assumption about the room's size or shape —
    # which matters, because the shape is exactly what is being measured.
    #
    # The sign says *what* was circled. Keeping the wall on the right, a lap
    # round the inside of a room turns consistently left; a lap round the
    # outside of a table turns consistently right. So the sign rejects the
    # robot circling the furniture and calling it the room.
    min_loop_winding_deg: float = 240.0

    # There is deliberately NO upper bound on winding.
    #
    # An obvious-looking one was tried and removed. A simple closed curve winds
    # exactly +-360 degrees, so refusing a lap that had wound much further
    # looked like a clean way to reject a path that wove among furniture and
    # curled back to its start. It failed on both counts:
    #
    #   * it did not catch the case it was for — the furnished lap simply found
    #     a closure at +424 degrees instead of +495, just inside any bound
    #     loose enough to be safe;
    #   * it broke a case that worked — outdoors, GNSS noise makes the path
    #     wander, the extra turning accumulates, the lap was refused, and the
    #     outline grew to 34.8 m2 of a 27 m2 room.
    #
    # The evidence that does separate them is not available here: it is the
    # distance driven against the perimeter of the outline the lap produced,
    # 0.90 for both rooms that work and 0.50 for the furnished one. That needs
    # the extracted room, so it lives in `assess_quality`. See
    # LAP_COVERAGE_LIMIT in services/mapper/storage.py.

    # Course is taken between points this far apart. Too fine and steering
    # jitter dominates the sum; too coarse and a small room stops registering.
    winding_sample_m: float = 0.15

    # A lap that never closes has to end anyway.
    #
    # Refusing an implausible closure means the robot keeps driving, and
    # without a backstop the mapper's loop never returns. Deliberately far
    # above any real perimeter — this room's is about 21 m and the lap takes
    # 19 — because it is a backstop and not the intended finish. Reaching it
    # leaves `loop_closed` False, so the outline is reported as never closed
    # and graded unusable, which is the honest outcome for a lap that failed.
    max_lap_distance_m: float = 150.0

    # Recovery from running into something the range sensors missed.
    #
    # Needed because real rooms have furniture against the walls — a bin, a
    # sofa, a cabinet — and the ±30° forward cone does not see a low bin at
    # all. Without recovery the robot simply leans on it: measured against a
    # fully furnished room the lap never completed and the scan came back
    # 10.16 m2 of a 27 m2 room.
    recover_distance_m: float = 0.20
    # Turning away afterwards matters as much as backing off. Reversing alone
    # leaves the robot pointed at the same obstacle, and it drives into it
    # again on the next cycle.
    recover_turn_deg: float = 35.0


@dataclass
class DriveCommand:
    linear_mps: float
    angular_dps: float
    state: ExploreState
    note: str = ""


class WallFollower:
    """Drives the robot around the boundary of a room."""

    def __init__(self, config: ExploreConfig | None = None) -> None:
        self.config = config or ExploreConfig()
        self.state = ExploreState.SEEKING_WALL
        # Total distance driven, including the initial hunt for a wall.
        self.distance_travelled_m = 0.0
        # Distance driven since wall-following began. Loop closure is judged
        # on this, not the total: the approach to the wall is not part of the
        # circuit, and counting it would satisfy the minimum before the lap
        # had started.
        self.lap_distance_m = 0.0
        self.start_x: float | None = None
        self.start_y: float | None = None
        self.loop_closed = False
        # How far the path has turned since following began, in degrees, signed.
        # A full circuit is about +-360; see `min_loop_winding_deg`.
        self.winding_deg = 0.0
        self._winding_x: float | None = None
        self._winding_y: float | None = None
        self._winding_course_deg: float | None = None
        # Counts consecutive cycles with the wall missing, so one dropped
        # ultrasonic reading does not trigger a corner manoeuvre.
        self._wall_lost_cycles = 0

        # Recovery bookkeeping: metres still to reverse, then degrees still to
        # turn away from whatever was hit.
        self._recover_remaining_m = 0.0
        self._recover_turn_remaining_deg = 0.0
        self.collisions = 0

    # ── Sensor helpers ────────────────────────────────────────────────────

    @staticmethod
    def _nearest(ranges: list[RangeReading], centre: float, span: float) -> float:
        """Closest valid reading within a window of sensor angles."""
        best = math.inf
        for r in ranges:
            if not r.valid:
                continue
            if abs(r.angle_deg - centre) <= span:
                best = min(best, r.distance_m)
        return best

    def _front_distance(self, ranges: list[RangeReading]) -> float:
        return self._nearest(ranges, 0.0, 30.0)

    def _side_distance(self, ranges: list[RangeReading]) -> float:
        side = -90.0 if self.config.follow_right_wall else 90.0
        return self._nearest(ranges, side, 50.0)

    # ── Control loop ──────────────────────────────────────────────────────

    def step(
        self,
        ranges: list[RangeReading],
        x_m: float,
        y_m: float,
        dt_s: float,
        blocked: bool = False,
    ) -> DriveCommand:
        """Return the next velocity command.

        `blocked` says the robot has run into something — inferred from the
        servo bus, since this robot has no bumper. It defaults to False so
        callers with no collision detection behave exactly as before.
        """
        cfg = self.config

        if self._check_loop_closed(x_m, y_m):
            self.state = ExploreState.FINISHED
            return DriveCommand(0.0, 0.0, self.state, "loop closed")

        # A lap that will not close still has to end. Refusing an implausible
        # closure means the robot keeps driving, and the mapper's loop waits
        # for FINISHED — so without this it waits for ever. `loop_closed` stays
        # False, which is what makes the outline report itself as never closed
        # rather than as a measurement.
        if self.lap_distance_m >= cfg.max_lap_distance_m:
            self.state = ExploreState.FINISHED
            return DriveCommand(
                0.0, 0.0, self.state, "gave up: the lap never closed"
            )

        recovery = self._recover(blocked, dt_s)
        if recovery is not None:
            step_distance = abs(recovery.linear_mps) * dt_s
            self.distance_travelled_m += step_distance
            if self.start_x is not None:
                self.lap_distance_m += step_distance
            return recovery

        front = self._front_distance(ranges)
        side = self._side_distance(ranges)

        # Sign convention: turning away from the followed wall is positive
        # when following the right wall, negative when following the left.
        away = 1.0 if cfg.follow_right_wall else -1.0

        command = self._decide(front, side, away, cfg)

        # The circuit is measured from where wall-following actually began,
        # not from wherever the robot happened to be switched on. Starting the
        # clock mid-room would compare the robot's position against a point it
        # never returns to, and the loop would never close.
        if self.start_x is None and self.state in (
            ExploreState.FOLLOWING,
            ExploreState.ROUNDING_CORNER,
        ):
            self.start_x, self.start_y = x_m, y_m
            self.lap_distance_m = 0.0

        step_distance = abs(command.linear_mps) * dt_s
        self.distance_travelled_m += step_distance
        if self.start_x is not None:
            self.lap_distance_m += step_distance
            self._accumulate_winding(x_m, y_m)
        return command

    def _decide(
        self, front: float, side: float, away: float, cfg: ExploreConfig
    ) -> DriveCommand:
        # ── Seeking: no wall has been latched onto yet ────────────────────
        if self.state == ExploreState.SEEKING_WALL:
            if side < cfg.wall_detect_distance_m:
                # A wall appeared alongside — start following it.
                self.state = ExploreState.FOLLOWING
                self._wall_lost_cycles = 0
            elif front < cfg.front_stop_distance_m:
                # Ran into a wall head-on. Turn away until it is beside us.
                self.state = ExploreState.TURNING_CORNER
                return DriveCommand(
                    0.0, away * cfg.turn_speed_dps, self.state, "wall ahead"
                )
            else:
                return DriveCommand(
                    cfg.cruise_speed_mps, 0.0, self.state, "seeking wall"
                )

        # ── Corner turn in progress ───────────────────────────────────────
        # Keep pivoting until the front is clear by the wider margin. Exiting
        # on the same threshold that triggered the turn would restart it on
        # the next noisy reading.
        if self.state == ExploreState.TURNING_CORNER:
            if front < cfg.front_clear_distance_m:
                return DriveCommand(
                    0.0, away * cfg.turn_speed_dps, self.state, "turning corner"
                )
            self.state = ExploreState.FOLLOWING
            self._wall_lost_cycles = 0

        # ── Wall directly ahead: an inside corner ─────────────────────────
        if front < cfg.front_stop_distance_m:
            self.state = ExploreState.TURNING_CORNER
            self._wall_lost_cycles = 0
            return DriveCommand(
                0.0, away * cfg.turn_speed_dps, self.state, "wall ahead"
            )

        # ── Wall fell away: an outside corner ─────────────────────────────
        # Curve around it rather than pivoting, so the side sensor keeps the
        # wall in view the whole way round. Requiring several consecutive
        # cycles stops a single dropped reading from starting a manoeuvre.
        if side > cfg.wall_lost_distance_m:
            self._wall_lost_cycles += 1
            if self._wall_lost_cycles >= 3:
                self.state = ExploreState.ROUNDING_CORNER
                return DriveCommand(
                    cfg.cruise_speed_mps * 0.6,
                    -away * cfg.turn_speed_dps * 0.8,
                    self.state,
                    "rounding corner",
                )
        else:
            self._wall_lost_cycles = 0
            self.state = ExploreState.FOLLOWING

        # ── Normal following ──────────────────────────────────────────────
        error = side - cfg.target_wall_distance_m

        if abs(error) <= cfg.tolerance_m:
            return DriveCommand(cfg.cruise_speed_mps, 0.0, self.state, "on track")

        # Too far from the wall: steer toward it. Too close: steer away.
        direction = -away if error > 0 else away
        return DriveCommand(
            cfg.cruise_speed_mps,
            direction * cfg.correction_dps,
            self.state,
            "too far" if error > 0 else "too close",
        )

    def _recover(self, blocked: bool, dt_s: float) -> DriveCommand | None:
        """Back off and turn away from something the sensors did not see.

        Returns the command to send while recovering, or None to carry on
        following the wall.

        Wall-following assumes the range sensors saw whatever is ahead. The
        ±30° forward cone does not see a low bin at all, and a sofa pushed
        against the wall is met head-on. Without this the robot leans on it for
        the rest of the lap: against a fully furnished room the circuit never
        closed and the scan reported 10.16 m2 of a 27 m2 room.

        Turning away matters as much as reversing. Backing straight out leaves
        the robot pointed at the same obstacle and it drives into it again.
        """
        cfg = self.config
        away = 1.0 if cfg.follow_right_wall else -1.0

        if blocked and self.state != ExploreState.RECOVERING:
            self.collisions += 1
            self.state = ExploreState.RECOVERING
            self._recover_remaining_m = cfg.recover_distance_m
            self._recover_turn_remaining_deg = cfg.recover_turn_deg

        if self.state != ExploreState.RECOVERING:
            return None

        if self._recover_remaining_m > 0.0:
            self._recover_remaining_m -= cfg.cruise_speed_mps * dt_s
            return DriveCommand(
                -cfg.cruise_speed_mps * 0.7, 0.0, self.state, "backing off"
            )

        if self._recover_turn_remaining_deg > 0.0:
            self._recover_turn_remaining_deg -= cfg.turn_speed_dps * dt_s
            return DriveCommand(
                0.0, away * cfg.turn_speed_dps, self.state, "turning away"
            )

        # Re-acquire the wall rather than resuming mid-manoeuvre: the robot is
        # no longer where the follower thought it was.
        self.state = ExploreState.SEEKING_WALL
        self._wall_lost_cycles = 0
        return None

    def _accumulate_winding(self, x_m: float, y_m: float) -> None:
        """Add this step's course change to the running total.

        Sampled over a fixed distance rather than every cycle: the follower
        steers constantly to hold its offset from the wall, and at 0.1 s
        intervals that jitter swamps the turn actually being made. Over 0.15 m
        the corrections cancel and the corners survive.
        """
        if self._winding_x is None:
            self._winding_x, self._winding_y = x_m, y_m
            return

        dx, dy = x_m - self._winding_x, y_m - self._winding_y
        if math.hypot(dx, dy) < self.config.winding_sample_m:
            return

        course = math.degrees(math.atan2(dy, dx))
        if self._winding_course_deg is not None:
            change = (course - self._winding_course_deg + 180.0) % 360.0 - 180.0
            self.winding_deg += change

        self._winding_course_deg = course
        self._winding_x, self._winding_y = x_m, y_m

    def _check_loop_closed(self, x_m: float, y_m: float) -> bool:
        if self.loop_closed:
            return True
        if self.start_x is None or self.start_y is None:
            return False
        if self.lap_distance_m < self.config.min_loop_distance_m:
            return False

        # Been round something, the right way round. Keeping the wall on the
        # right, a circuit of a room's inside turns left throughout; circling a
        # table turns right. Without this a 3.18 m hook closed the loop.
        expected = 1.0 if self.config.follow_right_wall else -1.0
        if self.winding_deg * expected < self.config.min_loop_winding_deg:
            return False

        back_home = (
            math.hypot(x_m - self.start_x, y_m - self.start_y)
            < self.config.loop_close_radius_m
        )
        if back_home:
            self.loop_closed = True
        return self.loop_closed

    def reset(self) -> None:
        self.state = ExploreState.SEEKING_WALL
        self.distance_travelled_m = 0.0
        self.lap_distance_m = 0.0
        self.start_x = self.start_y = None
        self.loop_closed = False
        self.winding_deg = 0.0
        self._winding_x = self._winding_y = self._winding_course_deg = None
        self._wall_lost_cycles = 0
        self._recover_remaining_m = 0.0
        self._recover_turn_remaining_deg = 0.0
        self.collisions = 0
