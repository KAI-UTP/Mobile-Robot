"""A scan can be internally perfect and still be less than half the room.

`is_closed`, `coverage_pct` and pose confidence all measure the map's
consistency with itself, and a robot that maps half a room very tidily scores
well on every one of them. A path that leaves the outline is the one piece of
evidence that cannot be argued with — the robot cannot drive through walls —
and it is evidence the robot collects itself, which matters because with real
hardware there is no ground truth to check against.

Scope, stated plainly: this catches an outline that EXCLUDES where the robot
drove. It does not catch a lap that closed early, because then the outline and
the path were cut short together — the furnished room reports 12.89 m2 of 27.0
with 0.3 % of the path outside. That failure is still open.
"""

from __future__ import annotations

from mapper.storage import FLOOR_OUTSIDE_LIMIT_PCT, assess_quality


def test_a_tidy_outline_around_half_a_room_is_not_good():
    """The exact failure: closed boundary, high coverage, and wrong."""
    quality = assess_quality(
        is_closed=True, coverage_pct=94.0, pose_confidence=0.9,
        floor_outside_pct=45.0,
    )
    assert quality.grade == "POOR"
    assert not quality.is_usable


def test_it_says_which_way_the_number_is_wrong():
    """"Too small" is actionable; "poor" on its own is not."""
    quality = assess_quality(
        is_closed=True, coverage_pct=94.0, pose_confidence=0.9,
        floor_outside_pct=45.0,
    )
    reason = " ".join(quality.reasons).lower()
    assert "outside" in reason
    assert "too small" in reason


def test_a_properly_measured_room_still_grades_good():
    """The check must not punish the scans that work. A little overshoot is
    normal — the robot's own footprint paints free cells slightly past the wall
    it is hugging, and the outline is simplified."""
    quality = assess_quality(
        is_closed=True, coverage_pct=96.8, pose_confidence=0.9,
        floor_outside_pct=3.0,
    )
    assert quality.grade == "GOOD"


def test_the_limit_leaves_room_for_normal_overshoot():
    assert 5.0 < FLOOR_OUTSIDE_LIMIT_PCT < 30.0


def test_an_unclosed_boundary_still_dominates():
    """A lap that never closed is unusable whatever else is true."""
    quality = assess_quality(
        is_closed=False, coverage_pct=99.0, pose_confidence=1.0,
        floor_outside_pct=0.0,
    )
    assert quality.grade == "UNUSABLE"


def test_the_figure_is_recorded_not_just_acted_on():
    """So the scan library can show why a room was marked down."""
    quality = assess_quality(
        is_closed=True, coverage_pct=90.0, pose_confidence=0.9,
        floor_outside_pct=22.5,
    )
    assert quality.floor_outside_pct == 22.5


def test_callers_that_do_not_measure_it_are_unaffected():
    """Defaults to zero so existing callers keep their previous grade rather
    than being silently marked down by a check they never ran."""
    quality = assess_quality(
        is_closed=True, coverage_pct=90.0, pose_confidence=0.9,
    )
    assert quality.grade == "GOOD"
