"""Cover a room systematically, by touch, on a robot that cannot see.

Why this exists
---------------
`bump_explorer` drives straight until something stops it, turns by roughly 110
degrees and drives again. That is how a first-generation bumper vacuum worked
and it has the coverage efficiency of one. Measured on this project:

    empty room       97.7 % of the floor, after 435 m of driving
    furnished room   58.8 % of the floor, after 142 m — and then it stopped,
                     because it had stopped *discovering* floor even though two
                     fifths of the room had never been touched

Random bouncing revisits the middle of a room over and over while corners and
the far sides of furniture go unvisited, and more driving does not reliably fix
it: it is a coupon-collector problem. The trail it leaves is a few diagonal
streaks across an otherwise untouched floor.

The room is not random, so the search should not be either.

Each phase does what it is actually good at
-------------------------------------------
**Find the walls by bouncing.** Bouncing is bad at covering a room and good at
*finding its edges* — it runs into things constantly, all over, which is
exactly the evidence a bounding box needs. So the first phase is
`BumpExplorer`, run for a fixed distance, and the box comes from where it hit
things.

Probing straight out in four directions was tried first and is worse. In a
furnished room the probes meet a table, a sofa and a cabinet rather than walls,
so the box comes back as the free space around wherever the robot started: it
measured 20.4 % of the room and stopped satisfied. Making a probe feel its way
past obstacles fixed that case and broke the empty one, thrashing sideways
against a wall it had already found. Bouncing needs neither special case.

**Cover it in rows.** Boustrophedon inside the measured box: across, shift over
by one robot width, back, shift, repeat. Coverage is then guaranteed by
construction rather than hoped for statistically.

The holonomic base is what makes the sweep cheap
------------------------------------------------
Three omni wheels at 120 degrees can strafe sideways without turning, so the
robot never rotates during a sweep — it drives along one axis, slides sideways
by a row, drives back. A differential robot needs two 90-degree turns at every
row end, forty-odd turns across a room, each one a fresh chance for the heading
estimate to drift. Dead reckoning is all this robot has.

Holonomic drive has been in this project from the start and no mapping strategy
ever used it for anything. This is what it is for.

The cost of not being able to see
---------------------------------
An obstacle is only found by touching it, so a row that runs into a table ends
there and the floor beyond it is unreachable along that axis. A second pass at
right angles reaches most of it, which is why `passes=2` is the default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from autonomy.bump_explorer import BumpConfig, BumpExplorer


class CoverState(str, Enum):
    FINDING = "FINDING"      # bouncing around to find where the walls are
    SWEEPING = "SWEEPING"    # driving a row
    SHIFTING = "SHIFTING"    # sliding sideways to the next row
    BACKING = "BACKING"      # easing off something just touched
    FINISHED = "FINISHED"


@dataclass(frozen=True)
class CoverConfig:
    cruise_speed_mps: float = 0.15
    reverse_speed_mps: float = 0.10

    # How far to bounce before deciding where the walls are.
    #
    # Long enough to have reached every wall of a domestic room a few times —
    # bouncing covers ground fast even though it covers it badly — and short
    # enough that most of the run is the part that actually sweeps.
    find_distance_m: float = 70.0

    # How far apart the rows are. The chassis is 0.22 m across, so rows this
    # far apart leave no gap between what one sweeps and the next.
    row_spacing_m: float = 0.20

    # Keeps a row clear of the wall it runs alongside, so it does not begin by
    # driving into it.
    #
    # Tighter is better right up until it is much worse, and 0.16 m is the
    # turn. The chassis radius is 0.11 m, so the margin decides how much of the
    # outer ring of the room the robot can reach at all. Measured on both
    # rooms, with the contact-tuned extractor:
    #
    #     margin   empty room            furnished room
    #     0.20     25.74 m2, 91.0 %      14.64 m2, 57.2 %
    #     0.16     26.37 m2, 99.8 %      14.73 m2, 60.0 %
    #     0.13     26.91 m2, 99.8 %       0.45 m2, 60.0 %   <- extraction fails
    #
    # At 0.13 the robot scrapes along everything it passes. Contact cells get
    # painted down every wall and around every piece of furniture, the free
    # space between them is carved into strips, and the morphological opening
    # in the extractor removes what is left: the best-covered furnished run of
    # the lot reports 0.45 m2 of floor. Coverage and a usable map are not the
    # same thing.
    wall_margin_m: float = 0.16

    # Easing off after a contact. Enough to unstick the wheels and no more —
    # every centimetre back is floor being re-covered.
    backoff_m: float = 0.10

    # Sweep along one axis, then the other. The second pass reaches floor the
    # first could not: a row that runs into a table stops there, and what is
    # behind it is only reachable from another direction.
    passes: int = 2

    # Backstops, so a run always ends. Not the intended finish — the sweep
    # finishes because it has covered its rows.
    max_distance_m: float = 500.0
    max_contacts: int = 600


@dataclass
class CoverCommand:
    """A holonomic command: forward, sideways and turn, in the BODY frame."""

    linear_mps: float
    lateral_mps: float
    angular_dps: float
    state: CoverState
    note: str = ""


@dataclass
class CoverStats:
    contacts: int = 0
    distance_m: float = 0.0
    rows_done: int = 0
    passes_done: int = 0
    contact_points: list[tuple[float, float]] = field(default_factory=list)
    bounds: tuple[float, float, float, float] | None = None


class ContactCoverage:
    """Covers a room by touch: bounce to find the walls, then sweep in rows.

    Consumes a pose and a contact flag — the same two things `bump_explorer`
    needs, and the same two the real hardware can produce.
    """

    def __init__(self, config: CoverConfig | None = None) -> None:
        self.config = config or CoverConfig()
        self.state = CoverState.FINDING
        self.stats = CoverStats()

        # Phase 1 is delegated. Its own limits are set far beyond what this
        # uses, so the distance below is what ends the phase.
        self._finder = BumpExplorer(
            BumpConfig(max_distance_m=self.config.find_distance_m * 4,
                       max_contacts=9999)
        )

        self.start_x: float | None = None
        self.start_y: float | None = None

        # Phase 2
        self._axis = 0                  # 0 = rows run along x, 1 = along y
        self._rows: list[float] = []
        self._row_index = 0
        self._direction = 1.0
        self._row_target: float | None = None
        self._shift_target: float | None = None

        self._backoff_remaining_m = 0.0
        self._retry_from_other_end = False
        # Whether this row has already had its one retry. Without it, a row
        # blocked at BOTH ends grants itself a fresh retry on every contact and
        # ping-pongs between the two obstacles for ever.
        self._retried_this_row = False
        self._last_x: float | None = None
        self._last_y: float | None = None

    # ── Main step ─────────────────────────────────────────────────────────

    def step(
        self, x_m: float, y_m: float, heading_deg: float, blocked: bool, dt_s: float
    ) -> CoverCommand:
        """One control cycle. `blocked` comes from the servo bus, not a bumper."""
        if self.state == CoverState.FINISHED:
            return CoverCommand(0.0, 0.0, 0.0, self.state, "room covered")

        if self.start_x is None:
            self.start_x, self.start_y = x_m, y_m
            self._last_x, self._last_y = x_m, y_m

        self._accumulate(x_m, y_m)

        if self._hit_a_limit():
            self.state = CoverState.FINISHED
            return CoverCommand(0.0, 0.0, 0.0, self.state, "limit reached")

        if self.state == CoverState.FINDING:
            return self._find(x_m, y_m, heading_deg, blocked, dt_s)

        if blocked and self.state in (CoverState.SWEEPING, CoverState.SHIFTING):
            return self._on_contact(x_m, y_m, heading_deg)

        if self.state == CoverState.BACKING:
            return self._back_off(heading_deg, dt_s)
        if self.state == CoverState.SHIFTING:
            return self._shift(x_m, y_m, heading_deg)
        return self._sweep(x_m, y_m, heading_deg)

    # ── Phase 1: where are the walls? ─────────────────────────────────────

    def _find(
        self, x_m: float, y_m: float, heading_deg: float, blocked: bool, dt_s: float
    ) -> CoverCommand:
        """Bounce around, collecting the places something stopped us."""
        if self.stats.distance_m >= self.config.find_distance_m:
            return self._start_sweeping()

        command = self._finder.step(x_m, y_m, heading_deg, blocked, dt_s)
        if self._finder.is_finished:
            return self._start_sweeping()

        # The finder turns; the sweep does not. Its commands pass through
        # unchanged, including the rotation.
        return CoverCommand(
            command.linear_mps, 0.0, command.angular_dps,
            CoverState.FINDING, "finding the walls",
        )

    def _start_sweeping(self) -> CoverCommand:
        self.stats.contact_points = list(self._finder.stats.contact_points)
        self.stats.contacts = self._finder.stats.contacts
        self._plan_rows()
        self.state = CoverState.SWEEPING
        return CoverCommand(0.0, 0.0, 0.0, self.state, "walls found; sweeping")

    def _bounds(self) -> tuple[float, float, float, float]:
        """The box the contacts fall in.

        Contacts include furniture as well as walls, but the box is the extreme
        of them, and a wall is always further out than the furniture standing
        against it — so the furniture does not shrink the answer.
        """
        points = self.stats.contact_points
        if len(points) < 3:
            # Nothing to go on. A sweep of the wrong size still beats no sweep.
            return (self.start_x - 3.0, self.start_x + 3.0,
                    self.start_y - 3.0, self.start_y + 3.0)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), max(xs), min(ys), max(ys)

    # ── Phase 2: the rows ─────────────────────────────────────────────────

    def _plan_rows(self) -> None:
        """Lay the rows out across the measured box, for the current pass."""
        cfg = self.config
        x_min, x_max, y_min, y_max = self._bounds()
        self.stats.bounds = (x_min, y_min, x_max, y_max)

        low, high = (y_min, y_max) if self._axis == 0 else (x_min, x_max)
        low += cfg.wall_margin_m
        high -= cfg.wall_margin_m

        if high <= low:
            self._rows = [(low + high) / 2.0]
        else:
            count = max(1, int(math.ceil((high - low) / cfg.row_spacing_m)))
            step = (high - low) / count
            self._rows = [low + step * i for i in range(count + 1)]

        self._row_index = 0
        self._direction = 1.0
        self._row_target = None

    def _sweep(self, x_m: float, y_m: float, heading_deg: float) -> CoverCommand:
        cfg = self.config
        x_min, x_max, y_min, y_max = self._bounds()

        if self._row_index >= len(self._rows):
            return self._next_pass()

        along_low, along_high = (
            (x_min + cfg.wall_margin_m, x_max - cfg.wall_margin_m) if self._axis == 0
            else (y_min + cfg.wall_margin_m, y_max - cfg.wall_margin_m)
        )
        here = x_m if self._axis == 0 else y_m

        if self._row_target is None:
            self._row_target = along_high if self._direction > 0 else along_low

        reached = (
            here >= self._row_target - 0.06 if self._direction > 0
            else here <= self._row_target + 0.06
        )
        if reached:
            return self._end_of_row()

        speed = cfg.cruise_speed_mps * self._direction
        world = (speed, 0.0) if self._axis == 0 else (0.0, speed)
        return self._world_command(world, heading_deg, CoverState.SWEEPING, "row")

    def _end_of_row(self) -> CoverCommand:
        """Row done: turn around and slide across to the next one."""
        self.stats.rows_done += 1
        self._row_index += 1
        self._direction *= -1.0
        self._row_target = None
        self._retry_from_other_end = False
        self._retried_this_row = False

        if self._row_index >= len(self._rows):
            return self._next_pass()

        self._shift_target = self._rows[self._row_index]
        self.state = CoverState.SHIFTING
        return CoverCommand(0.0, 0.0, 0.0, self.state, "next row")

    def _shift(self, x_m: float, y_m: float, heading_deg: float) -> CoverCommand:
        """Slide onto the next row. No turning — this is a kiwi drive."""
        across = y_m if self._axis == 0 else x_m
        gap = self._shift_target - across

        if abs(gap) < 0.05:
            self.state = CoverState.SWEEPING
            return CoverCommand(0.0, 0.0, 0.0, self.state, "on the next row")

        speed = math.copysign(self.config.cruise_speed_mps, gap)
        world = (0.0, speed) if self._axis == 0 else (speed, 0.0)
        return self._world_command(world, heading_deg, CoverState.SHIFTING, "shifting")

    def _next_pass(self) -> CoverCommand:
        """Rows exhausted. Sweep again at right angles, or stop."""
        self.stats.passes_done += 1

        if self.stats.passes_done >= self.config.passes:
            self.state = CoverState.FINISHED
            return CoverCommand(0.0, 0.0, 0.0, self.state, "room covered")

        # Across the other axis, so floor that was behind furniture — a row
        # stops at whatever it touches — is approached from a direction that
        # can reach it.
        self._axis = 1 - self._axis
        self._plan_rows()
        self.state = CoverState.SWEEPING
        return CoverCommand(0.0, 0.0, 0.0, self.state, "second pass, across")

    # ── Contact during the sweep ──────────────────────────────────────────

    def _on_contact(self, x_m: float, y_m: float, heading_deg: float) -> CoverCommand:
        """Something solid in the row. Note it, ease off, take the next row."""
        self.stats.contacts += 1
        self.stats.contact_points.append((x_m, y_m))

        self._row_target = None
        # One retry per row, not one per contact. A row blocked at both ends
        # would otherwise grant itself another every time it touched either of
        # them, and ping-pong between the two for ever — measured on the live
        # stack, the robot spent 451 m bouncing between (3.82, 3.67) and
        # (4.70, 3.67) and never reached another row.
        self._retry_from_other_end = not self._retried_this_row
        self._backoff_remaining_m = self.config.backoff_m
        self.state = CoverState.BACKING
        return self._reverse(heading_deg)

    def _back_off(self, heading_deg: float, dt_s: float) -> CoverCommand:
        self._backoff_remaining_m -= self.config.reverse_speed_mps * dt_s
        if self._backoff_remaining_m > 0.0:
            return self._reverse(heading_deg)

        self.state = CoverState.SWEEPING

        # A row that ran into something has only covered the side of it the
        # robot approached from. Run the same row back the other way before
        # moving on, and the far side gets covered too.
        #
        # Without this each row covered one side of the table and the next row
        # covered the other, so half of every obstructed band was missed —
        # 57.6 % of a furnished room, no better than bouncing. Once per row, so
        # a robot wedged against something cannot sit here trading directions.
        if self._retry_from_other_end:
            self._retry_from_other_end = False
            self._retried_this_row = True
            self._direction *= -1.0
            self._row_target = None
            return CoverCommand(0.0, 0.0, 0.0, self.state, "same row, other way")

        return self._end_of_row()

    def _reverse(self, heading_deg: float) -> CoverCommand:
        """Back along the row, away from whatever was touched."""
        speed = -self.config.reverse_speed_mps * self._direction
        world = (speed, 0.0) if self._axis == 0 else (0.0, speed)
        return self._world_command(world, heading_deg, CoverState.BACKING, "easing off")

    # ── Frames ────────────────────────────────────────────────────────────

    @staticmethod
    def _world_command(
        world: tuple[float, float], heading_deg: float,
        state: CoverState, note: str,
    ) -> CoverCommand:
        """Turn a world-frame velocity into the body-frame one the wheels take.

        The sweep thinks in the room's axes; the wheels work in the robot's. It
        would be tempting to assume the two are aligned — the sweep never turns
        — but the phase before it does, and it hands over pointing wherever it
        finished. Assuming otherwise would send every row off at that angle.
        """
        angle = math.radians(heading_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        vx, vy = world
        return CoverCommand(
            linear_mps=vx * cos_a + vy * sin_a,
            lateral_mps=-vx * sin_a + vy * cos_a,
            angular_dps=0.0,
            state=state,
            note=note,
        )

    # ── Bookkeeping ───────────────────────────────────────────────────────

    def _accumulate(self, x_m: float, y_m: float) -> None:
        if self._last_x is None:
            self._last_x, self._last_y = x_m, y_m
            return
        self.stats.distance_m += math.hypot(x_m - self._last_x, y_m - self._last_y)
        self._last_x, self._last_y = x_m, y_m

    def _hit_a_limit(self) -> bool:
        return (
            self.stats.distance_m >= self.config.max_distance_m
            or self.stats.contacts >= self.config.max_contacts
        )

    @property
    def is_finished(self) -> bool:
        return self.state == CoverState.FINISHED

    def summary(self) -> dict:
        return {
            "state": self.state.value,
            "contacts": self.stats.contacts,
            "distance_m": round(self.stats.distance_m, 2),
            "rows": self.stats.rows_done,
            "passes": self.stats.passes_done,
            "bounds": self.stats.bounds,
            # Finished its rows, as opposed to stopping on a limit — a run that
            # hit a limit should be read as incomplete.
            "completed": self.stats.passes_done >= self.config.passes,
        }
