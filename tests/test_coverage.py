"""Row-by-row coverage, obstacle avoidance and collision recovery.

Wall-following measures a perimeter and learns nothing about the middle of the
room. These tests cover the sweep that does — and the awkward cases that make
real rooms harder than empty rectangles.
"""

from __future__ import annotations

import pytest
from robotmap_common.models import RangeReading

from simulator.coverage import (
    CoverageConfig,
    CoveragePlanner,
    CoverageState,
)

DT = 0.1


def _clear(distance: float = 4.0) -> list[RangeReading]:
    return [
        RangeReading(angle_deg=0.0, distance_m=distance),
        RangeReading(angle_deg=90.0, distance_m=distance),
        RangeReading(angle_deg=-90.0, distance_m=distance),
    ]


def _blocked(distance: float) -> list[RangeReading]:
    return [
        RangeReading(angle_deg=0.0, distance_m=distance),
        RangeReading(angle_deg=90.0, distance_m=4.0),
        RangeReading(angle_deg=-90.0, distance_m=4.0),
    ]


# ── Sweeping ─────────────────────────────────────────────────────────────────


def test_it_starts_sweeping_forward():
    planner = CoveragePlanner()
    cmd = planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    assert cmd.linear_mps > 0
    assert planner.state == CoverageState.SWEEPING


def test_it_slows_as_something_gets_nearer():
    """A late detection at full speed becomes a collision."""
    planner = CoveragePlanner()
    fast = planner.step(_clear(4.0), 1.0, 1.0, 0.0, False, DT)
    slow = planner.step(_blocked(0.5), 1.0, 1.0, 0.0, False, DT)
    assert slow.linear_mps < fast.linear_mps


def test_an_obstacle_ahead_ends_the_row():
    planner = CoveragePlanner()
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner.step(_blocked(0.25), 3.0, 1.0, 0.0, False, DT)
    assert planner.state == CoverageState.STEPPING
    assert planner.stats.obstacles_found == 1


def test_it_steps_sideways_not_by_turning():
    """The robot is holonomic, so it keeps facing along the row and its
    forward sensor keeps looking where it is going."""
    planner = CoveragePlanner()
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner.step(_blocked(0.25), 3.0, 1.0, 0.0, False, DT)

    cmd = planner.step(_clear(), 3.0, 1.0, 0.0, False, DT)
    assert cmd.state == CoverageState.STEPPING
    assert abs(cmd.lateral_mps) > 0
    assert cmd.angular_dps == 0


def test_rows_alternate_direction():
    """A boustrophedon sweep reverses each row rather than driving back to the
    start, which would double the distance."""
    planner = CoveragePlanner()
    first = planner.sweep_direction
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner.step(_blocked(0.25), 3.0, 1.0, 0.0, False, DT)
    assert planner.sweep_direction == -first


def test_it_about_faces_between_rows():
    """The base could drive the next row backwards without turning, but every
    range sensor faces forward — so it turns and keeps them leading."""
    planner = CoveragePlanner(config=CoverageConfig(row_spacing_m=0.2))
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner.step(_blocked(0.25), 3.0, 1.0, 0.0, False, DT)

    heading = 0.0
    for _ in range(200):
        cmd = planner.step(_clear(), 3.0, 1.0, heading, False, DT)
        heading = (heading + cmd.angular_dps * DT) % 360
        if cmd.state == CoverageState.TURNING:
            break
    assert planner.state == CoverageState.TURNING


def test_it_returns_to_sweeping_after_turning_round():
    planner = CoveragePlanner(config=CoverageConfig(row_spacing_m=0.2))
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner.step(_blocked(0.25), 3.0, 1.0, 0.0, False, DT)

    heading = 0.0
    for _ in range(400):
        cmd = planner.step(_clear(), 3.0, 1.0, heading, False, DT)
        heading = (heading + cmd.angular_dps * DT) % 360
        if planner.row_index > 0 and cmd.state == CoverageState.SWEEPING:
            break

    assert planner.state == CoverageState.SWEEPING
    # It should now be facing back down the row it came from.
    assert abs(((heading - 180.0 + 180.0) % 360.0) - 180.0) < 15.0


def test_a_turn_that_never_completes_is_abandoned():
    """Closing the turn on the measured heading means a heading that stops
    updating — a failed IMU, a robot picked up — would otherwise spin for
    ever. It must give up and get on with the row."""
    config = CoverageConfig(row_spacing_m=0.2, turn_timeout_s=1.0)
    planner = CoveragePlanner(config=config)
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner.step(_blocked(0.25), 3.0, 1.0, 0.0, False, DT)

    # Heading pinned at 0 throughout: the turn can never reach its target.
    for _ in range(400):
        planner.step(_clear(), 3.0, 1.0, 0.0, False, DT)
        if planner.row_index > 0 and planner.state == CoverageState.SWEEPING:
            break

    assert planner.state != CoverageState.TURNING


# ── Collisions ───────────────────────────────────────────────────────────────


def test_a_bump_triggers_recovery():
    """The bumper fires when a sensor missed something."""
    planner = CoveragePlanner()
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)

    cmd = planner.step(_clear(), 2.0, 1.0, 0.0, True, DT)
    assert planner.state == CoverageState.RECOVERING
    assert cmd.linear_mps < 0, "must reverse away from what it hit"
    assert planner.stats.collisions == 1


def test_a_collision_records_the_obstacle():
    """Otherwise the robot bumps the same chair leg on every pass."""
    planner = CoveragePlanner()
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner.step(_clear(), 2.0, 1.0, 0.0, True, DT)

    assert planner.stats.obstacles_found == 1
    assert planner.stats.obstacles[0].from_collision is True


def test_the_recorded_obstacle_is_in_front_not_underneath():
    """The bumper is on the robot's nose, so what it hit is ahead of the
    reported centre."""
    planner = CoveragePlanner()
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner.step(_clear(), 2.0, 1.0, 0.0, True, DT)

    obstacle = planner.stats.obstacles[0]
    assert obstacle.x_m > 2.0


def test_it_backs_off_a_real_distance_before_continuing():
    planner = CoveragePlanner()
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner.step(_clear(), 2.0, 1.0, 0.0, True, DT)

    reversing = 0
    for _ in range(30):
        cmd = planner.step(_clear(), 2.0, 1.0, 0.0, False, DT)
        if cmd.linear_mps < 0:
            reversing += 1
        else:
            break
    assert reversing >= 3


def test_it_does_not_resume_into_the_thing_it_just_hit():
    planner = CoveragePlanner()
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner.step(_clear(), 2.0, 1.0, 0.0, True, DT)

    for _ in range(60):
        cmd = planner.step(_clear(), 2.0, 1.0, 0.0, False, DT)
        if cmd.state not in (CoverageState.RECOVERING,):
            break
    assert planner.state != CoverageState.SWEEPING or planner.row_index > 0


# ── Obstacle memory ──────────────────────────────────────────────────────────


def test_obstacles_at_the_same_spot_are_merged():
    """Without merging, one table becomes dozens of overlapping obstacles as
    the robot approaches from slightly different angles."""
    planner = CoveragePlanner()
    for _ in range(8):
        planner._register_obstacle(3.0, 2.0, from_collision=False)
    assert planner.stats.obstacles_found == 1
    assert planner.stats.obstacles[0].hit_count == 8


def test_obstacles_far_apart_stay_separate():
    planner = CoveragePlanner()
    planner._register_obstacle(1.0, 1.0, from_collision=False)
    planner._register_obstacle(4.0, 3.0, from_collision=False)
    assert planner.stats.obstacles_found == 2


def test_a_remembered_obstacle_is_avoided_even_when_unseen():
    """The sensor may miss it again; the map does not."""
    planner = CoveragePlanner()
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner._register_obstacle(2.3, 1.0, from_collision=True)

    cmd = planner.step(_clear(4.0), 2.0, 1.0, 0.0, False, DT)
    assert planner.state == CoverageState.AVOIDING
    assert "known obstacle" in cmd.note


def test_avoiding_sidesteps_rather_than_stopping():
    planner = CoveragePlanner()
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner._register_obstacle(2.3, 1.0, from_collision=True)
    planner.step(_clear(), 2.0, 1.0, 0.0, False, DT)

    cmd = planner.step(_clear(), 2.0, 1.0, 0.0, False, DT)
    assert abs(cmd.lateral_mps) > 0


# ── Awkward rooms ────────────────────────────────────────────────────────────


def test_a_wedged_robot_gives_up_on_the_row():
    """Pressed into a corner the robot still commands motion and still reports
    sensible sensors — only the lack of actual movement reveals it."""
    config = CoverageConfig(stall_patience_steps=5)
    planner = CoveragePlanner(config=config)
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)

    for _ in range(20):
        planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)   # never moves

    assert planner.row_index > 0 or planner.state != CoverageState.SWEEPING


def test_the_sweep_terminates():
    """It must finish rather than sweep forever."""
    planner = CoveragePlanner(config=CoverageConfig(max_rows=3, row_spacing_m=0.1))

    heading = 0.0
    for step in range(8000):
        x = 1.0 + (step % 30) * 0.05
        ranges = _clear() if (step % 30) < 25 else _blocked(0.2)
        cmd = planner.step(ranges, x, 1.0, heading, False, DT)
        heading = (heading + cmd.angular_dps * DT) % 360
        if planner.is_finished:
            break

    assert planner.is_finished


def test_a_wall_alongside_ends_the_sweep():
    """`bounds` is expressed in the pose estimate's own frame, so it drifts
    over a long sweep. The range readings do not, so they get the final say."""
    planner = CoveragePlanner(bounds=(0.0, 0.0, 100.0, 100.0))
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner.step(_blocked(0.25), 3.0, 1.0, 0.0, False, DT)

    # Plenty of room according to the (stale) bounds, but a wall 10 cm to the
    # left, which is the way the sweep is trying to advance.
    hemmed_in = [
        RangeReading(angle_deg=0.0, distance_m=4.0),
        RangeReading(angle_deg=90.0, distance_m=0.10),
        RangeReading(angle_deg=-90.0, distance_m=0.10),
    ]
    for _ in range(30):
        planner.step(hemmed_in, 3.0, 1.0, 0.0, False, DT)
        if planner.is_finished:
            break

    assert planner.is_finished


def test_rows_stay_parallel_over_a_long_sweep():
    """An open-loop about-face falls short by one control step every row. Six
    rows later the sweep is fanning out across the room instead of covering
    it, and the gaps between passes are unswept floor."""
    planner = CoveragePlanner(config=CoverageConfig(row_spacing_m=0.2))

    heading = 0.0
    row_headings: list[float] = []
    last_row = 0

    for _ in range(6000):
        # End each row after a fixed distance, alternating like a real sweep.
        blocked = len(row_headings) < 8 and (_ % 400) > 380
        ranges = _blocked(0.2) if blocked else _clear()
        cmd = planner.step(ranges, 3.0, 1.0, heading, False, DT)
        heading = (heading + cmd.angular_dps * DT) % 360

        if cmd.state == CoverageState.SWEEPING and planner.row_index != last_row:
            row_headings.append(heading)
            last_row = planner.row_index
        if len(row_headings) >= 6:
            break

    # Every row must run along the axis or exactly against it — never at a
    # steadily growing angle to it.
    for h in row_headings:
        off_axis = min(abs(_signed(h)), abs(_signed(h - 180.0)))
        assert off_axis < 10.0, f"row heading {h:.1f} has drifted off the axis"


def _signed(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def test_a_cluttered_row_is_abandoned_rather_than_swept_forever():
    """Guards a livelock.

    Every avoided obstacle gets recorded, and the record then triggers another
    avoid on the next pass — so without a per-row budget the robot ping-pongs
    between sweeping and avoiding and never reaches the end of the row or the
    next one. It looks busy and achieves nothing.
    """
    config = CoverageConfig(max_avoids_per_row=3, max_rows=10)
    planner = CoveragePlanner(config=config)
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)

    # A line of obstacles straight down the row.
    for x in (1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3):
        planner._register_obstacle(x, 1.0, from_collision=False)

    for _ in range(400):
        planner.step(_clear(), 1.2, 1.0, 0.0, False, DT)
        if planner.row_index > 0:
            break

    assert planner.row_index > 0, "never gave up on an impassable row"
    assert planner.stats.rows_skipped >= 1


def test_no_assumption_that_the_room_is_rectangular():
    """Bounds are optional: the sweep works from what it can actually drive,
    so an L-shape or an alcove needs no special case."""
    planner = CoveragePlanner(bounds=None)
    cmd = planner.step(_clear(), 0.0, 0.0, 0.0, False, DT)
    assert cmd.linear_mps > 0


# ── Reporting ────────────────────────────────────────────────────────────────


def test_the_summary_reports_what_happened():
    planner = CoveragePlanner()
    planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    planner.step(_clear(), 2.0, 1.0, 0.0, True, DT)

    summary = planner.summary()
    assert summary["collisions"] == 1
    assert summary["obstacles_found"] == 1
    assert summary["obstacles_from_contact"] == 1
    assert summary["distance_m"] >= 0


def test_distance_accumulates():
    planner = CoveragePlanner()
    for _ in range(10):
        planner.step(_clear(), 1.0, 1.0, 0.0, False, DT)
    assert planner.stats.distance_m == pytest.approx(0.18 * 10 * DT, rel=0.2)
