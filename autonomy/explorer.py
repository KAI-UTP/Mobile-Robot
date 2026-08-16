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

    def _check_loop_closed(self, x_m: float, y_m: float) -> bool:
        if self.loop_closed:
            return True
        if self.start_x is None or self.start_y is None:
            return False
        if self.lap_distance_m < self.config.min_loop_distance_m:
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
        self._wall_lost_cycles = 0
        self._recover_remaining_m = 0.0
        self._recover_turn_remaining_deg = 0.0
        self.collisions = 0
