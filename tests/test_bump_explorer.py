"""Mapping with no range sensors at all.

The physical robot has a servo bus, BLE and GNSS. No ultrasonic ring, no lidar.
`explorer.py` cannot run on it — wall-following holds a *measured distance* to a
wall and there is nothing to measure with. So the robot has three facts:

    where it drove      servo encoders
    where it stopped    a stalled wheel means something solid is there
    roughly where       BLE, 2.71 m error, bounded

These tests cover the strategy that turns those into a room, and pin down the
two mistakes that made the first version useless.
"""

from __future__ import annotations

import pytest

from autonomy.bump_explorer import BumpConfig, BumpExplorer, BumpState

DT = 0.1


def _drive(explorer, cycles, *, blocked=False, cells=0, start=(1.0, 1.0)):
    """Run the explorer, moving the robot as it asks."""
    import math

    x, y, heading = start[0], start[1], 0.0
    command = None
    for _ in range(cycles):
        command = explorer.step(x, y, heading, blocked, DT, cells)
        angle = math.radians(heading)
        x += command.linear_mps * DT * math.cos(angle)
        y += command.linear_mps * DT * math.sin(angle)
        heading = (heading + command.angular_dps * DT) % 360
    return command, (x, y, heading)


# ── Driving ──────────────────────────────────────────────────────────────────


def test_it_drives_forward_to_begin_with():
    explorer = BumpExplorer()
    command = explorer.step(1.0, 1.0, 0.0, False, DT)
    assert command.linear_mps > 0
    assert command.state == BumpState.DRIVING


def test_it_keeps_driving_across_open_floor():
    """The first version turned back after 1.2 m, reasoning that a long free
    run meant the robot had wandered off the boundary. It meant the robot never
    reached the far wall: it circled its start, called the loop closed after
    12 m and four contacts, and reported 0.38 m2 of a 27 m2 room.

    Crossing open floor is not a failure. It is the only way to find out how
    far away the other side is.
    """
    explorer = BumpExplorer()
    command, _ = _drive(explorer, 300)      # ~4.5 m with nothing in the way

    assert command.state == BumpState.DRIVING
    assert command.linear_mps > 0
    assert explorer.stats.contacts == 0


# ── Contact ──────────────────────────────────────────────────────────────────


def test_contact_stops_it_pushing():
    explorer = BumpExplorer()
    explorer.step(1.0, 1.0, 0.0, False, DT)

    command = explorer.step(1.0, 1.0, 0.0, True, DT)
    assert command.linear_mps <= 0
    assert explorer.stats.contacts == 1


def test_a_contact_is_recorded_ahead_of_the_robot():
    """The robot stopped with its leading edge against the object, so the
    object is at its nose — not underneath it."""
    explorer = BumpExplorer()
    explorer.step(2.0, 2.0, 0.0, False, DT)
    explorer.step(2.0, 2.0, 0.0, True, DT)

    x, y = explorer.stats.contact_points[0]
    assert x > 2.0
    assert y == pytest.approx(2.0, abs=0.01)


def test_it_backs_off_then_turns():
    """Turning while still pressed against the object only scrubs."""
    explorer = BumpExplorer()
    explorer.step(1.0, 1.0, 0.0, False, DT)
    explorer.step(1.0, 1.0, 0.0, True, DT)

    saw_reverse = saw_turn = False
    for _ in range(120):
        command = explorer.step(1.0, 1.0, 0.0, False, DT)
        if command.linear_mps < 0:
            saw_reverse = True
        if command.angular_dps != 0:
            saw_turn = True
            assert saw_reverse, "turned before backing off"
            break
    assert saw_reverse and saw_turn


def test_it_returns_to_driving_after_a_contact():
    explorer = BumpExplorer()
    explorer.step(1.0, 1.0, 0.0, False, DT)
    explorer.step(1.0, 1.0, 0.0, True, DT)

    for _ in range(300):
        command = explorer.step(1.0, 1.0, 0.0, False, DT)
        if command.state == BumpState.DRIVING and command.linear_mps > 0:
            return
    pytest.fail("never resumed driving")


# ── Not getting stuck in a cycle ─────────────────────────────────────────────


def test_the_turn_angle_varies():
    """A constant turn puts the robot into a limit cycle: it retraces one
    triangle for ever while coverage flatlines and it looks busy."""
    explorer = BumpExplorer()
    angles = {round(explorer._next_turn(), 3) for _ in range(20)}
    assert len(angles) > 15, "the turn angle is effectively constant"


def test_the_turn_is_reproducible():
    """Varying must not mean unrepeatable — a run has to be comparable with
    the one before it."""
    first = [BumpExplorer()._next_turn() for _ in range(5)]
    second = [BumpExplorer()._next_turn() for _ in range(5)]
    assert first == second


def test_it_always_turns_the_same_way():
    """Sweeping the room in a rotating pattern, rather than oscillating
    between two headings."""
    explorer = BumpExplorer()
    explorer.step(1.0, 1.0, 0.0, False, DT)
    explorer.step(1.0, 1.0, 0.0, True, DT)

    for _ in range(200):
        command = explorer.step(1.0, 1.0, 0.0, False, DT)
        if command.angular_dps != 0:
            assert command.angular_dps > 0
            return
    pytest.fail("never turned")


# ── Knowing when to stop ─────────────────────────────────────────────────────


def test_it_stops_once_it_stops_finding_new_floor():
    """Loop closure is meaningless for a robot bouncing across a room — it
    passes its own start within the first few metres. "I have stopped
    discovering floor" is the signal that means something."""
    config = BumpConfig(saturation_contacts=3, min_new_cells=10)
    explorer = BumpExplorer(config)

    # Repeated contacts, and the map never grows.
    for _ in range(400):
        explorer.step(1.0, 1.0, 0.0, True, DT, 500)
        explorer.step(1.0, 1.0, 0.0, False, DT, 500)
        if explorer.is_finished:
            break

    assert explorer.is_finished
    assert explorer.saturated


def test_it_keeps_going_while_it_is_still_learning():
    config = BumpConfig(saturation_contacts=3, min_new_cells=10)
    explorer = BumpExplorer(config)

    cells = 0
    for _ in range(30):
        cells += 200        # plenty of new floor every contact
        explorer.step(1.0, 1.0, 0.0, True, DT, cells)
        explorer.step(1.0, 1.0, 0.0, False, DT, cells)

    assert not explorer.saturated


def test_it_always_terminates():
    """A room it cannot cover must not run for ever."""
    explorer = BumpExplorer(BumpConfig(max_contacts=5))
    for _ in range(2000):
        explorer.step(1.0, 1.0, 0.0, True, DT)
        explorer.step(1.0, 1.0, 0.0, False, DT)
        if explorer.is_finished:
            break
    assert explorer.is_finished


def test_a_finished_explorer_commands_nothing():
    explorer = BumpExplorer(BumpConfig(max_contacts=1))
    for _ in range(50):
        explorer.step(1.0, 1.0, 0.0, True, DT)

    command = explorer.step(1.0, 1.0, 0.0, False, DT)
    assert command.linear_mps == 0
    assert command.angular_dps == 0


# ── Reporting ────────────────────────────────────────────────────────────────


def test_the_distance_cap_is_a_backstop_not_the_intended_finish():
    """It used to be the thing that actually ended every run.

    Measured on a 6.0 x 4.5 m room the robot needs about 435 m of bouncing to
    cover it, and the cap sat at 200 m — so runs reported themselves FINISHED
    at 87.7 % coverage while still discovering 6 % of new floor every 50 m.
    The cap has to sit well clear of the distance a real room needs.
    """
    assert BumpConfig().max_distance_m >= 500.0


def test_the_summary_says_whether_it_finished_or_gave_up():
    """Saturated means the room was covered. Hitting a limit does not, and a
    scan that stopped early should not look the same as one that finished."""
    explorer = BumpExplorer(BumpConfig(max_contacts=2))
    for _ in range(50):
        explorer.step(1.0, 1.0, 0.0, True, DT)

    summary = explorer.summary()
    assert summary["contacts"] >= 1
    assert summary["saturated"] is False


def test_it_needs_no_range_readings_at_all():
    """The entire point: `step` takes a pose and a contact flag. If this
    signature ever grows a ranges argument, the strategy has stopped being
    usable on the actual hardware."""
    import inspect

    parameters = set(inspect.signature(BumpExplorer.step).parameters)
    assert "ranges" not in parameters
    assert {"x_m", "y_m", "heading_deg", "blocked"} <= parameters
