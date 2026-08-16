"""Mapping a room by driving into it, for a robot with no range sensors.

Why this exists
---------------
The physical robot has a servo bus, BLE beacons and GNSS. It has no ultrasonic
ring and no lidar. `explorer.py` cannot run on it at all — wall-following works
by holding a *measured distance* to a wall, and there is nothing to measure it
with. Every map this project makes today is built from range readings the
hardware will not produce.

So the robot has exactly three things to work with:

    where it drove      servo encoders, dead reckoning
    where it stopped    a stalled wheel means something solid is there
    roughly where       BLE trilateration, 2.71 m of error, but bounded

That is enough. It is how bumper-only floor robots mapped rooms for years, and
it produces a genuinely different kind of map:

    FREE      swept by the chassis as it drove — certain, because it was there
    BLOCKED   a contact point — certain, because it touched it
    UNKNOWN   everywhere else, and honestly labelled as such

Compare that with a range-sensor map, which infers free space along a ray it
never visited. This map claims less and is wrong less often. What it costs is
time: the only way to learn about a square metre of floor is to drive over it.

How the room is traced
----------------------
Drive in a straight line until something stops the robot; back off; turn; drive
again. The contact points land on the boundary, and the floor the chassis has
swept is the room. This is how a bumper-only vacuum covers a room, and it is
the only strategy available without a way to see.

Two details decide whether it works, and I got both wrong first time:

**Drive until you actually hit something.** My first version turned back after
1.2 m of open floor, reasoning that a long free run meant the robot had
wandered off the boundary. It meant the robot never reached the far wall at
all: it circled near its start, declared the loop closed after 12 m and four
contacts, and reported 0.38 m2 of a 27 m2 room. Crossing open floor is not a
failure — it is the only way to discover how far away the other side is.

**Vary the turn.** A constant turn angle puts the robot into a limit cycle: it
retraces the same triangle for ever, and coverage stops improving while the
robot looks busy. A small seeded jitter breaks the symmetry and is what real
bumper robots do. Seeded, so a run is still reproducible.

Termination is coverage saturation, not loop closure. "I have returned to where
I started" means nothing to a robot bouncing across a room — it happens within
the first few metres. "I have stopped discovering new floor" is the honest
signal that the room is mapped.

This is deliberately shape-agnostic. Nothing assumes the room is rectangular or
convex: an L-shape, an alcove, a corridor all work the same way, because the
robot is only ever answering "can I go this way?".
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum


class BumpState(str, Enum):
    DRIVING = "DRIVING"        # heading out until something stops us
    BACKING = "BACKING"        # easing off what we just touched
    TURNING = "TURNING"        # swinging to a new heading
    FINISHED = "FINISHED"


@dataclass(frozen=True)
class BumpConfig:
    """Everything the strategy needs, and the reasoning for each number."""

    cruise_speed_mps: float = 0.15
    reverse_speed_mps: float = 0.10
    turn_speed_dps: float = 45.0

    # How far to reverse after a contact. Enough to clear the object so the
    # turn does not scrub against it, and no further — every centimetre back
    # is floor being re-covered.
    backoff_m: float = 0.12

    # How far to turn after each contact, and how much that varies.
    #
    # A CONSTANT angle puts the robot into a limit cycle — it retraces one
    # triangle for ever while coverage flatlines. The jitter breaks that
    # symmetry. Around 110 degrees turns the robot back into the room rather
    # than grazing along the wall it just met.
    turn_after_contact_deg: float = 110.0
    turn_jitter_deg: float = 45.0
    turn_seed: int = 17

    # Turning consistently one way sweeps the room in a rotating pattern
    # instead of oscillating between two headings. +1 anticlockwise.
    turn_direction: float = 1.0

    # ── Termination ──────────────────────────────────────────────────────
    # Coverage saturation: stop once the robot has stopped discovering floor.
    # Loop closure is meaningless here — a bouncing robot passes its start
    # within the first few metres.
    saturation_contacts: int = 25
    min_new_cells: int = 40

    # Hard stops, so a run always ends.
    #
    # These are backstops, NOT the intended finish. That distinction was lost
    # when the distance cap sat at 200 m: measured on a 6.0 x 4.5 m room the
    # robot needs about 435 m of bouncing to cover it, so every run stopped at
    # 200 m and reported itself FINISHED while coverage was still climbing —
    # 87.7 % of the floor, and rising 6 % over the previous 50 m.
    #
    #     cap 200 m   87.7 % covered, saturated=False   cut off
    #     cap 400 m   97.7 % covered, saturated=False   cut off
    #     cap 600 m   97.7 % covered, saturated=True    finished at 435 m
    #
    # Set well clear of that so coverage saturation is what ends a run.
    # `summary()["saturated"]` still says which of the two it was, and a scan
    # that hit a cap should be read as incomplete.
    max_contacts: int = 600
    max_distance_m: float = 600.0


@dataclass
class BumpCommand:
    linear_mps: float
    angular_dps: float
    state: BumpState
    note: str = ""


@dataclass
class BumpStats:
    contacts: int = 0
    distance_m: float = 0.0
    free_runs: int = 0          # legs that ended in open floor, not a contact
    contact_points: list[tuple[float, float]] = field(default_factory=list)


class BumpExplorer:
    """Traces a room's boundary by driving into it.

    Consumes only a contact flag and a pose, so it runs unchanged against the
    simulator, against Isaac Sim, and against the real servo bus.
    """

    def __init__(self, config: BumpConfig | None = None) -> None:
        self.config = config or BumpConfig()
        self.state = BumpState.DRIVING
        self.stats = BumpStats()

        self.start_x: float | None = None
        self.start_y: float | None = None
        self.saturated = False

        self._rng = random.Random(self.config.turn_seed)
        self._backoff_remaining_m = 0.0
        self._turn_remaining_deg = 0.0
        self._last_x: float | None = None
        self._last_y: float | None = None

        # Coverage saturation: how many cells were known at the last contact,
        # and how many contacts in a row have added almost nothing.
        self._cells_at_last_contact = 0
        self._barren_contacts = 0

    # ── Main step ─────────────────────────────────────────────────────────

    def step(
        self,
        x_m: float,
        y_m: float,
        heading_deg: float,
        blocked: bool,
        dt_s: float,
        explored_cells: int = 0,
    ) -> BumpCommand:
        """One control cycle. `blocked` comes from the servo bus, not a bumper.

        `explored_cells` is how much floor the map knows about. It is what
        decides when to stop: a robot bouncing across a room passes its own
        start within the first few metres, so returning home says nothing,
        while "I have stopped finding new floor" says everything.
        """
        cfg = self.config

        if self.state == BumpState.FINISHED:
            return BumpCommand(0.0, 0.0, self.state, "room covered")

        if self.start_x is None:
            self.start_x, self.start_y = x_m, y_m
            self._last_x, self._last_y = x_m, y_m

        self._accumulate(x_m, y_m)

        if self._should_finish():
            self.state = BumpState.FINISHED
            note = "coverage saturated" if self.saturated else "limit reached"
            return BumpCommand(0.0, 0.0, self.state, note)

        # Contact takes priority: the robot is against something and must stop
        # pushing against it.
        if blocked and self.state == BumpState.DRIVING:
            self._record_contact(x_m, y_m, heading_deg, explored_cells)
            self.state = BumpState.BACKING
            self._backoff_remaining_m = cfg.backoff_m
            self._turn_remaining_deg = self._next_turn()

        if self.state == BumpState.BACKING:
            return self._back_off(dt_s)

        if self.state == BumpState.TURNING:
            return self._turn(dt_s)

        return self._drive()

    def _next_turn(self) -> float:
        """A turn angle that varies, so the robot cannot settle into a cycle."""
        cfg = self.config
        return cfg.turn_after_contact_deg + self._rng.uniform(
            -cfg.turn_jitter_deg, cfg.turn_jitter_deg
        )

    # ── The three behaviours ──────────────────────────────────────────────

    def _drive(self) -> BumpCommand:
        """Straight on until something stops us.

        Deliberately no free-run limit. Crossing open floor is not the robot
        going wrong, it is the only way to find out how far away the far wall
        is — turning back after 1.2 m left it circling its start and reporting
        0.38 m2 of a 27 m2 room.
        """
        return BumpCommand(
            self.config.cruise_speed_mps, 0.0, BumpState.DRIVING,
            "driving until blocked",
        )

    def _back_off(self, dt_s: float) -> BumpCommand:
        cfg = self.config
        self._backoff_remaining_m -= cfg.reverse_speed_mps * dt_s

        if self._backoff_remaining_m <= 0.0:
            self.state = BumpState.TURNING
            return BumpCommand(0.0, 0.0, self.state, "backed off")

        return BumpCommand(
            -cfg.reverse_speed_mps, 0.0, BumpState.BACKING, "easing off contact"
        )

    def _turn(self, dt_s: float) -> BumpCommand:
        cfg = self.config
        self._turn_remaining_deg -= cfg.turn_speed_dps * dt_s

        if self._turn_remaining_deg <= 0.0:
            self.state = BumpState.DRIVING
            self._leg_distance_m = 0.0
            return BumpCommand(0.0, 0.0, self.state, "new heading")

        return BumpCommand(
            0.0,
            cfg.turn_direction * cfg.turn_speed_dps,
            BumpState.TURNING,
            "turning away",
        )

    # ── Bookkeeping ───────────────────────────────────────────────────────

    def _record_contact(
        self, x_m: float, y_m: float, heading_deg: float, explored_cells: int
    ) -> None:
        """Note where the boundary is, and whether we are still learning.

        The contact is recorded slightly ahead of the reported pose: the robot
        stopped with its leading edge against the object, so the object is at
        its nose, not under its middle.
        """
        angle = math.radians(heading_deg)
        self.stats.contacts += 1
        self.stats.contact_points.append(
            (x_m + 0.10 * math.cos(angle), y_m + 0.10 * math.sin(angle))
        )

        gained = explored_cells - self._cells_at_last_contact
        self._cells_at_last_contact = explored_cells
        if explored_cells and gained < self.config.min_new_cells:
            self._barren_contacts += 1
        else:
            self._barren_contacts = 0

    def _accumulate(self, x_m: float, y_m: float) -> None:
        if self._last_x is None:
            self._last_x, self._last_y = x_m, y_m
            return
        self.stats.distance_m += math.hypot(x_m - self._last_x, y_m - self._last_y)
        self._last_x, self._last_y = x_m, y_m

    def _should_finish(self) -> bool:
        cfg = self.config

        if self._barren_contacts >= cfg.saturation_contacts:
            self.saturated = True
            return True
        if self.stats.contacts >= cfg.max_contacts:
            return True
        return self.stats.distance_m >= cfg.max_distance_m

    # ── Reporting ─────────────────────────────────────────────────────────

    @property
    def is_finished(self) -> bool:
        return self.state == BumpState.FINISHED

    def summary(self) -> dict:
        return {
            "state": self.state.value,
            "contacts": self.stats.contacts,
            "distance_m": round(self.stats.distance_m, 2),
            "saturated": self.saturated,
        }
