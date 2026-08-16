"""Wall-following has to survive furniture standing against the wall.

Wall-following assumes the range sensors saw whatever is ahead. They often did
not: the forward cone is ±30°, which misses a low bin entirely, and a sofa
pushed against the wall the robot is following is met head-on with nothing
useful in view beforehand.

Without recovery the robot simply leans on it for the rest of the lap. Measured
against the fully furnished room, the circuit never closed and the scan came
back as 10.16 m2 of a 27 m2 room — not wrong so much as abandoned.

There is no bumper on this robot, so `blocked` is inferred from the servo bus;
that inference is covered in tests/test_collision_detection.py.
"""

from __future__ import annotations

import pytest
from robotmap_common.models import RangeReading

from autonomy.explorer import ExploreConfig, ExploreState, WallFollower

DT = 0.1


def _ranges(front=4.0, right=0.35, left=4.0):
    return [
        RangeReading(angle_deg=0.0, distance_m=front),
        RangeReading(angle_deg=-90.0, distance_m=right),
        RangeReading(angle_deg=90.0, distance_m=left),
    ]


def _settle(follower, cycles=40, **kwargs):
    command = None
    for _ in range(cycles):
        command = follower.step(_ranges(**kwargs), 1.0, 1.0, DT)
    return command


# ── Nothing changes when nothing is hit ──────────────────────────────────────


def test_it_behaves_exactly_as_before_when_nothing_is_hit():
    """`blocked` defaults to False, so every existing caller is unaffected."""
    follower = WallFollower()
    for _ in range(50):
        command = follower.step(_ranges(), 1.0, 1.0, DT)
        assert command.state != ExploreState.RECOVERING
    assert follower.collisions == 0


# ── Recovering ───────────────────────────────────────────────────────────────


def test_a_collision_starts_a_recovery():
    follower = WallFollower()
    _settle(follower, 5)

    command = follower.step(_ranges(), 1.0, 1.0, DT, blocked=True)
    assert command.state == ExploreState.RECOVERING
    assert follower.collisions == 1


def test_it_reverses_first():
    """Turning while still pressed against the object just grinds."""
    follower = WallFollower()
    _settle(follower, 5)

    command = follower.step(_ranges(), 1.0, 1.0, DT, blocked=True)
    assert command.linear_mps < 0


def test_it_backs_off_a_real_distance():
    """One reversed cycle moves under two centimetres, which clears nothing."""
    follower = WallFollower()
    _settle(follower, 5)
    follower.step(_ranges(), 1.0, 1.0, DT, blocked=True)

    reversing = 0
    for _ in range(40):
        command = follower.step(_ranges(), 1.0, 1.0, DT)
        if command.linear_mps < 0:
            reversing += 1
        else:
            break
    assert reversing >= 3


def test_it_turns_away_after_backing_off():
    """Reversing alone leaves the robot pointed at the same obstacle, and it
    drives straight back into it on the next cycle."""
    follower = WallFollower()
    _settle(follower, 5)
    follower.step(_ranges(), 1.0, 1.0, DT, blocked=True)

    turned = 0.0
    for _ in range(80):
        command = follower.step(_ranges(), 1.0, 1.0, DT)
        if command.state != ExploreState.RECOVERING:
            break
        turned += abs(command.angular_dps) * DT
    assert turned > 10.0, "it never turned away from what it hit"


def test_it_turns_away_from_the_wall_it_follows():
    follower = WallFollower(ExploreConfig(follow_right_wall=True))
    _settle(follower, 5)
    follower.step(_ranges(), 1.0, 1.0, DT, blocked=True)

    for _ in range(80):
        command = follower.step(_ranges(), 1.0, 1.0, DT)
        if command.angular_dps != 0:
            assert command.angular_dps > 0, "turned into the wall, not away"
            return
    pytest.fail("never turned")


def test_it_goes_back_to_looking_for_the_wall():
    """Resuming mid-manoeuvre would be wrong: the robot is no longer where the
    follower thought it was."""
    follower = WallFollower()
    _settle(follower, 5)
    follower.step(_ranges(), 1.0, 1.0, DT, blocked=True)

    for _ in range(200):
        command = follower.step(_ranges(), 1.0, 1.0, DT)
        if command.state not in (ExploreState.RECOVERING,):
            break
    assert follower.state != ExploreState.RECOVERING


def test_one_contact_counts_once_while_it_is_still_touching():
    """Contact persists for many cycles as the robot backs away."""
    follower = WallFollower()
    _settle(follower, 5)
    for _ in range(15):
        follower.step(_ranges(), 1.0, 1.0, DT, blocked=True)

    assert follower.collisions == 1


def test_hitting_something_else_later_counts_again():
    follower = WallFollower()
    _settle(follower, 5)
    follower.step(_ranges(), 1.0, 1.0, DT, blocked=True)

    # Long enough to finish recovering, short enough that the lap does not
    # close — the robot is standing still here, so 3 m of commanded travel
    # would otherwise satisfy loop closure and freeze the follower.
    for _ in range(60):
        follower.step(_ranges(), 1.0, 1.0, DT)
    assert follower.state != ExploreState.FINISHED

    follower.step(_ranges(), 1.0, 1.0, DT, blocked=True)
    assert follower.collisions == 2


def test_recovery_distance_counts_towards_the_lap():
    """Loop closure is judged on distance driven. Reversing that is not
    counted makes the lap look shorter than it was, and the boundary can close
    early on a robot that has been shuffling back and forth."""
    follower = WallFollower()
    _settle(follower, 30)
    before = follower.distance_travelled_m

    follower.step(_ranges(), 1.0, 1.0, DT, blocked=True)
    assert follower.distance_travelled_m > before


def test_reset_clears_the_recovery_state():
    follower = WallFollower()
    _settle(follower, 5)
    follower.step(_ranges(), 1.0, 1.0, DT, blocked=True)
    follower.reset()

    assert follower.collisions == 0
    assert follower.state == ExploreState.SEEKING_WALL


# ── The reason it exists ─────────────────────────────────────────────────────


def test_the_furnished_room_can_be_mapped_again():
    """The end-to-end consequence: with furniture against the walls and no
    recovery, the lap never closes and the scan reports a third of the room."""
    from test_integration import run_pipeline

    from simulator.virtual_robot import VirtualWorld

    world = VirtualWorld.room_with_furniture(6.0, 4.5)
    result = run_pipeline(world, start_x=0.6, start_y=0.6)

    assert result.outline.is_closed
    assert result.outline.area_m2 > 20.0, "the lap was abandoned"


# ── Closing the loop on something that is actually a loop ────────────────────
#
# Distance and proximity alone are far too weak a test for "I have been round
# the room". Measured in the furnished room, the robot drove a 3.18 m hook near
# the bin, curled back to within 0.49 m of where following began, and declared
# the boundary complete — 10.19 m2 of a 27 m2 room, 6.9 % of the floor ever
# visited. Both thresholds were satisfied by a curl that went nowhere.


def _follow_path(follower, points):
    """Walk the follower along a path, bypassing the sensor model.

    The winding accumulator only reads the pose, so a path is enough to
    exercise it, and a synthetic path is far clearer than steering a simulated
    robot into the right shape.
    """
    follower.start_x, follower.start_y = points[0]
    follower.lap_distance_m = 999.0        # distance is not what is under test
    for x, y in points:
        follower._accumulate_winding(x, y)
    return follower


def _ring(cx, cy, radius, turns=1.0, steps=180, clockwise=False):
    import math

    sign = -1.0 if clockwise else 1.0
    return [
        (
            cx + radius * math.cos(sign * 2 * math.pi * turns * i / steps),
            cy + radius * math.sin(sign * 2 * math.pi * turns * i / steps),
        )
        for i in range(steps + 1)
    ]


def test_a_full_circuit_winds_a_full_turn():
    follower = WallFollower()
    _follow_path(follower, _ring(3.0, 2.0, 2.0))

    assert follower.winding_deg == pytest.approx(360.0, abs=15.0)


def test_a_hook_does_not_close_the_loop():
    """The 3.18 m curl that reported 10.19 m2 of a 27 m2 room."""
    follower = WallFollower()
    # A quarter turn and back to near the start: short, curled, going nowhere.
    _follow_path(follower, _ring(1.0, 1.0, 0.4, turns=0.25))

    assert abs(follower.winding_deg) < WallFollower().config.min_loop_winding_deg
    assert not follower._check_loop_closed(*_ring(1.0, 1.0, 0.4, turns=0.25)[0])


def test_a_full_circuit_does_close_the_loop():
    """The guard must not stop the rooms that work from finishing."""
    follower = WallFollower()
    path = _ring(3.0, 2.0, 2.0)
    _follow_path(follower, path)

    assert follower._check_loop_closed(*path[0])


def test_circling_furniture_the_wrong_way_is_not_a_room():
    """Keeping the wall on the right, a lap round the inside of a room turns
    left throughout; a lap round the outside of a table turns right. The sign
    is what tells the two apart."""
    follower = WallFollower()
    assert follower.config.follow_right_wall
    path = _ring(3.0, 2.0, 0.6, clockwise=True)
    _follow_path(follower, path)

    assert follower.winding_deg < 0
    assert not follower._check_loop_closed(*path[0])


def test_steering_jitter_does_not_accumulate():
    """The follower corrects constantly to hold its offset from the wall. If
    that jitter counted, a straight run along a wall would wind its way to a
    false closure on its own."""
    import random

    rng = random.Random(3)
    follower = WallFollower()
    path = [(x * 0.05, rng.uniform(-0.01, 0.01)) for x in range(400)]
    _follow_path(follower, path)

    assert abs(follower.winding_deg) < 90.0


def test_a_reset_forgets_the_winding():
    follower = WallFollower()
    _follow_path(follower, _ring(3.0, 2.0, 2.0))
    assert follower.winding_deg != 0.0

    follower.reset()
    assert follower.winding_deg == 0.0
