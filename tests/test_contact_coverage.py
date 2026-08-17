"""Covering a room on purpose, rather than by bouncing off it.

`bump_explorer` drives straight until something stops it, turns about 110
degrees and goes again. It is how a first-generation bumper vacuum worked and
it covers a room the way one did: measured here, 97.7 % of an empty floor after
435 m of driving, and 58.8 % of a furnished one before deciding it was done.
Random bouncing revisits the middle over and over while corners go untouched.

This covers the room in rows instead, and the rows are what make coverage a
property of the plan rather than a hope.
"""

from __future__ import annotations

import math

import pytest

from autonomy.contact_coverage import ContactCoverage, CoverConfig, CoverState

DT = 0.1


def _drive(explorer, cycles, *, blocked_at=None, start=(1.0, 1.0), heading=0.0):
    """Run the explorer, moving the robot as it asks.

    `blocked_at` is a predicate on (x, y) standing in for the room's walls and
    furniture — contact is the only thing this robot learns from.

    A blocked robot is stopped at the obstacle, NOT frozen in place: it has to
    be able to reverse off whatever it touched. Freezing it deadlocks — contact
    stays true, the explorer keeps commanding a reverse, nothing moves, and the
    run never ends.
    """
    x, y = start
    command = None
    blocked = False
    for _ in range(cycles):
        command = explorer.step(x, y, heading, blocked, DT)
        if explorer.is_finished:
            break

        angle = math.radians(heading)
        vx = command.linear_mps * math.cos(angle) - command.lateral_mps * math.sin(angle)
        vy = command.linear_mps * math.sin(angle) + command.lateral_mps * math.cos(angle)
        next_x, next_y = x + vx * DT, y + vy * DT

        # Solid: the move is refused, and contact is what the robot feels.
        blocked = bool(blocked_at and blocked_at(next_x, next_y))
        if not blocked:
            x, y = next_x, next_y

        heading = (heading + command.angular_dps * DT) % 360
    return command, (x, y)


def _room(width=6.0, height=4.5, margin=0.11):
    """Walls only. Contact when the chassis would be outside them."""
    return lambda x, y: not (
        margin < x < width - margin and margin < y < height - margin
    )


# ── It finds the room before it tries to cover it ────────────────────────────


def test_it_starts_by_finding_the_walls():
    """Bouncing is bad at covering a room and good at finding its edges, so
    that is what the first phase is for."""
    explorer = ContactCoverage()
    command = explorer.step(1.0, 1.0, 0.0, False, DT)
    assert command.state == CoverState.FINDING


def test_the_box_comes_from_where_it_hit_things():
    explorer = ContactCoverage(CoverConfig(find_distance_m=40.0))
    _drive(explorer, 20000, blocked_at=_room())

    bounds = explorer.stats.bounds
    assert bounds is not None, "never worked out how big the room is"
    x_min, y_min, x_max, y_max = bounds
    assert x_max - x_min > 3.0
    assert y_max - y_min > 2.0


def test_furniture_does_not_shrink_the_box():
    """Contacts include tables as well as walls, but the box is the extreme of
    them and a wall is always further out than the furniture against it."""
    explorer = ContactCoverage()
    explorer.start_x, explorer.start_y = 1.0, 1.0
    explorer.stats.contact_points = [
        (0.1, 2.0), (5.9, 2.0), (3.0, 0.1), (3.0, 4.4),   # walls
        (2.5, 2.0), (3.5, 2.3), (2.8, 1.8),               # a table in the middle
    ]
    x_min, x_max, y_min, y_max = explorer._bounds()

    assert x_min == pytest.approx(0.1)
    assert x_max == pytest.approx(5.9)
    assert y_min == pytest.approx(0.1)
    assert y_max == pytest.approx(4.4)


# ── The sweep itself ─────────────────────────────────────────────────────────


def test_the_sweep_never_turns():
    """A kiwi drive strafes. A differential robot would need two 90-degree
    turns at every row end — forty-odd across a room, each one a fresh chance
    for the heading estimate to drift, and dead reckoning is all this has."""
    explorer = ContactCoverage(CoverConfig(find_distance_m=25.0))
    explorer.state = CoverState.SWEEPING
    explorer.start_x, explorer.start_y = 1.0, 1.0
    explorer.stats.contact_points = [(0.1, 0.1), (5.9, 4.4), (0.1, 4.4)]
    explorer._plan_rows()

    for _ in range(3000):
        command = explorer.step(2.0, 2.0, 0.0, False, DT)
        if explorer.is_finished:
            break
        assert command.angular_dps == 0.0, "the sweep turned"


def test_rows_are_spaced_so_nothing_is_missed_between_them():
    """The chassis is 0.22 m across, so rows further apart than that leave
    strips of floor no pass ever touches."""
    config = CoverConfig()
    assert config.row_spacing_m <= 0.22


def test_it_sweeps_both_ways_round():
    """A row that runs into a table stops there, and what is behind it can only
    be reached from another direction."""
    assert CoverConfig().passes == 2


def test_the_margin_keeps_the_rows_off_the_walls():
    """0.13 m was measured and rejected: the robot scrapes along everything,
    contact cells get painted down every wall, and the extractor's opening
    removes the strips of floor left between them — the best-covered furnished
    run of the lot reported 0.45 m2."""
    assert CoverConfig().wall_margin_m >= 0.15


# ── Frames ───────────────────────────────────────────────────────────────────


def test_commands_are_in_the_body_frame():
    """The sweep thinks in the room's axes and the wheels work in the robot's.
    The phase before the sweep turns, so it hands over pointing wherever it
    finished — assuming the two frames agree would send every row off at that
    angle."""
    explorer = ContactCoverage()
    straight = explorer._world_command((0.15, 0.0), 0.0, CoverState.SWEEPING, "")
    assert straight.linear_mps == pytest.approx(0.15)
    assert straight.lateral_mps == pytest.approx(0.0, abs=1e-9)

    # Same intent, robot turned 90 degrees: it must now strafe, not drive.
    turned = explorer._world_command((0.15, 0.0), 90.0, CoverState.SWEEPING, "")
    assert turned.linear_mps == pytest.approx(0.0, abs=1e-9)
    assert turned.lateral_mps == pytest.approx(-0.15)


# ── Ending ───────────────────────────────────────────────────────────────────


def test_it_always_terminates():
    explorer = ContactCoverage(CoverConfig(max_distance_m=30.0))
    _drive(explorer, 40000, blocked_at=_room())
    assert explorer.is_finished


def test_a_finished_explorer_commands_nothing():
    explorer = ContactCoverage(CoverConfig(max_distance_m=1.0))
    _drive(explorer, 5000, blocked_at=_room())

    command = explorer.step(2.0, 2.0, 0.0, False, DT)
    assert command.linear_mps == 0.0
    assert command.lateral_mps == 0.0
    assert command.angular_dps == 0.0


def test_the_summary_says_whether_it_covered_its_rows():
    """Stopping on a limit is not the same as finishing, and a scan that hit
    one should not read like a scan that completed."""
    explorer = ContactCoverage(CoverConfig(max_distance_m=5.0))
    _drive(explorer, 5000, blocked_at=_room())

    assert explorer.summary()["completed"] is False


def test_it_needs_no_range_readings():
    """The whole point: a pose and a contact flag, which is all this robot can
    produce. If this signature ever grows a ranges argument, the strategy has
    stopped being usable on the actual hardware."""
    import inspect

    parameters = set(inspect.signature(ContactCoverage.step).parameters)
    assert "ranges" not in parameters
    assert {"x_m", "y_m", "heading_deg", "blocked"} <= parameters


# ── Not getting stuck ────────────────────────────────────────────────────────


def test_a_row_blocked_at_both_ends_moves_on():
    """The failure seen on the live stack: the robot ping-ponged between
    (3.82, 3.67) and (4.70, 3.67) for 451 m and never reached another row.

    A row that runs into something gets ONE retry from the other end, because
    it has only covered the side it approached from. Granting a fresh retry on
    every contact — which is what "set the flag in _on_contact" quietly did —
    means a row blocked at both ends trades directions for ever.
    """
    explorer = ContactCoverage(CoverConfig(find_distance_m=1.0))
    explorer.state = CoverState.SWEEPING
    explorer.start_x, explorer.start_y = 1.0, 1.0
    explorer.stats.contact_points = [(0.1, 0.1), (5.9, 4.4), (0.1, 4.4)]
    explorer._plan_rows()

    # A corridor blocked at both ends, so every row is a trap.
    blocked = lambda x, y: not (2.5 < x < 3.5)      # noqa: E731

    rows_at_start = explorer._row_index
    _drive(explorer, 30000, blocked_at=blocked, start=(3.0, 1.0))

    assert explorer.is_finished or explorer._row_index > rows_at_start, (
        "never left the first row"
    )


def test_the_retry_is_granted_once_per_row():
    explorer = ContactCoverage(CoverConfig(find_distance_m=1.0))
    explorer.state = CoverState.SWEEPING
    explorer.start_x, explorer.start_y = 1.0, 1.0
    explorer.stats.contact_points = [(0.1, 0.1), (5.9, 4.4), (0.1, 4.4)]
    explorer._plan_rows()

    explorer._on_contact(3.0, 1.0, 0.0)
    assert explorer._retry_from_other_end, "first contact should earn a retry"

    explorer._back_off(0.0, 10.0)                   # consumes it
    explorer._on_contact(3.0, 1.0, 0.0)
    assert not explorer._retry_from_other_end, "a second retry on the same row"


def test_a_fresh_row_gets_its_own_retry():
    """Once per row, not once per run — the next row has its own obstacles."""
    explorer = ContactCoverage(CoverConfig(find_distance_m=1.0))
    explorer.state = CoverState.SWEEPING
    explorer.start_x, explorer.start_y = 1.0, 1.0
    explorer.stats.contact_points = [(0.1, 0.1), (5.9, 4.4), (0.1, 4.4)]
    explorer._plan_rows()

    explorer._on_contact(3.0, 1.0, 0.0)
    explorer._back_off(0.0, 10.0)
    explorer._end_of_row()

    explorer._on_contact(3.0, 1.2, 0.0)
    assert explorer._retry_from_other_end
