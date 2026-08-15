"""Tests for twin-fidelity comparison.

These decide what the two-screen display claims, so each metric is tested
against a case where it *should* fail — a measure that never fails is not a
measure.
"""

from __future__ import annotations

import pytest
from robotmap_common.comparison import (
    align_polygons,
    centroid,
    compare_rooms,
    intersection_over_union,
    point_in_polygon,
    reference_room,
    rotate_polygon,
    translate_polygon,
)

ROOM = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.5), (0.0, 4.5)]  # 27 m2


def _scaled(polygon, factor, about=(3.0, 2.25)):
    ox, oy = about
    return [(ox + (x - ox) * factor, oy + (y - oy) * factor) for x, y in polygon]


# ── Centroid ─────────────────────────────────────────────────────────────────


def test_centroid_of_a_rectangle_is_its_middle():
    assert centroid(ROOM) == pytest.approx((3.0, 2.25))


def test_centroid_is_area_weighted_not_vertex_averaged():
    """A traced map has far more vertices along the walls it saw closely, so a
    vertex average would be dragged toward them."""
    lopsided = [
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0),
        (4.0, 0.0), (5.0, 0.0), (6.0, 0.0),
        (6.0, 4.5), (0.0, 4.5),
    ]
    vertex_mean_y = sum(p[1] for p in lopsided) / len(lopsided)
    assert centroid(lopsided)[1] != pytest.approx(vertex_mean_y)


def test_degenerate_polygons_do_not_crash():
    assert centroid([]) == (0.0, 0.0)
    assert centroid([(1.0, 1.0)]) == pytest.approx((1.0, 1.0))


# ── Point in polygon ─────────────────────────────────────────────────────────


def test_point_inside_and_outside():
    assert point_in_polygon((3.0, 2.0), ROOM)
    assert not point_in_polygon((7.0, 2.0), ROOM)
    assert not point_in_polygon((3.0, -1.0), ROOM)


def test_concave_notch_is_excluded():
    """An L-shaped room's notch must read as outside; a convex-hull style test
    would wrongly include it."""
    l_room = reference_room("l-shaped")
    assert point_in_polygon((1.0, 4.0), l_room)      # in the tall leg
    assert not point_in_polygon((5.0, 4.5), l_room)  # in the notch


# ── IoU ──────────────────────────────────────────────────────────────────────


def test_identical_rooms_score_one():
    assert intersection_over_union(ROOM, ROOM, 0.05) == pytest.approx(1.0, abs=0.01)


def test_disjoint_rooms_score_zero():
    far = translate_polygon(ROOM, 100.0, 100.0)
    assert intersection_over_union(ROOM, far, 0.05) == 0.0


def test_half_overlap_scores_a_third():
    """Two equal rectangles overlapping by half: intersection is half of one,
    union is one and a half, so IoU is 1/3."""
    shifted = translate_polygon(ROOM, 3.0, 0.0)
    assert intersection_over_union(ROOM, shifted, 0.02) == pytest.approx(1 / 3, abs=0.02)


def test_iou_falls_as_size_error_grows():
    for factor, ceiling in ((0.95, 1.0), (0.85, 0.9), (0.70, 0.75)):
        score = intersection_over_union(_scaled(ROOM, factor), ROOM, 0.03)
        assert score < ceiling


def test_iou_catches_a_wrong_shape_with_the_right_area():
    """The failure area error alone cannot see.

    A room measured too wide and too short can have exactly the correct area
    while being the wrong shape. IoU must penalise it.
    """
    same_area_wrong_shape = [(0.0, 0.0), (9.0, 0.0), (9.0, 3.0), (0.0, 3.0)]  # 27 m2

    from robotmap_common.geometry import polygon_area_m2

    assert polygon_area_m2(same_area_wrong_shape) == pytest.approx(27.0)
    assert intersection_over_union(same_area_wrong_shape, ROOM, 0.03) < 0.75


def test_iou_is_symmetric():
    other = _scaled(ROOM, 0.9)
    forward = intersection_over_union(ROOM, other, 0.05)
    backward = intersection_over_union(other, ROOM, 0.05)
    assert forward == pytest.approx(backward, abs=0.02)


def test_iou_handles_degenerate_input():
    assert intersection_over_union([], ROOM) == 0.0
    assert intersection_over_union([(0.0, 0.0), (1.0, 1.0)], ROOM) == 0.0


# ── Alignment ────────────────────────────────────────────────────────────────


def test_translation_is_removed_before_comparison():
    """The robot's map origin is wherever it was switched on. Comparing raw
    coordinates would mostly measure that arbitrary offset."""
    displaced = translate_polygon(ROOM, 10.0, -5.0)
    aligned = align_polygons(displaced, ROOM)
    assert centroid(aligned) == pytest.approx(centroid(ROOM), abs=1e-6)


def test_rotation_is_removed_before_comparison():
    """The map's axes point wherever the robot was facing at startup."""
    rotated = rotate_polygon(ROOM, 30.0, centroid(ROOM))
    aligned = align_polygons(rotated, ROOM)
    assert intersection_over_union(aligned, ROOM, 0.03) > 0.95


def test_alignment_does_not_hide_a_size_error():
    """Alignment is rigid motion only. Allowing scale would erase exactly the
    error worth finding: a map uniformly 20 % small is 20 % wrong."""
    small = _scaled(ROOM, 0.8)
    aligned = align_polygons(small, ROOM)

    from robotmap_common.geometry import polygon_area_m2

    assert polygon_area_m2(aligned) == pytest.approx(27.0 * 0.64, rel=0.01)


# ── Whole comparison ─────────────────────────────────────────────────────────


def test_a_perfect_map_scores_perfectly():
    result = compare_rooms(ROOM, ROOM)
    assert result.area_error_pct == pytest.approx(0.0, abs=0.1)
    assert result.iou > 0.99
    assert result.grade == "EXCELLENT"


def test_a_realistic_map_grades_well():
    """The accuracy the simulator actually achieves: about 1 % on area."""
    realistic = _scaled(ROOM, 0.995)
    result = compare_rooms(realistic, ROOM)
    assert abs(result.area_error_pct) < 2.0
    assert result.grade in ("EXCELLENT", "GOOD")


def test_a_bad_map_grades_poorly():
    result = compare_rooms(_scaled(ROOM, 0.6), ROOM)
    assert result.grade == "POOR"


def test_dimension_errors_are_reported_per_side():
    stretched = [(0.0, 0.0), (7.0, 0.0), (7.0, 4.5), (0.0, 4.5)]
    result = compare_rooms(stretched, ROOM)

    assert result.long_side_error_pct == pytest.approx(100 * (7.0 - 6.0) / 6.0, abs=1.0)
    assert abs(result.short_side_error_pct) < 2.0


def test_centroid_offset_is_measured_before_alignment():
    """Alignment removes the offset, so measuring afterwards would always
    report zero and the metric would be useless."""
    displaced = translate_polygon(ROOM, 3.0, 4.0)
    result = compare_rooms(displaced, ROOM)
    assert result.centroid_offset_m == pytest.approx(5.0, abs=0.01)


def test_area_can_be_right_while_the_grade_is_not():
    """Guards the reason IoU exists rather than grading on area."""
    same_area_wrong_shape = [(0.0, 0.0), (9.0, 0.0), (9.0, 3.0), (0.0, 3.0)]
    result = compare_rooms(same_area_wrong_shape, ROOM)

    assert abs(result.area_error_pct) < 1.0     # area says perfect
    assert result.grade in ("FAIR", "POOR")     # shape says otherwise


def test_l_shaped_room_compares():
    l_room = reference_room("l-shaped")
    result = compare_rooms(l_room, l_room)
    assert result.iou > 0.99
    assert result.truth_area_m2 == pytest.approx(25.0, rel=0.01)


def test_summary_is_human_readable():
    text = compare_rooms(_scaled(ROOM, 0.98), ROOM).summary()
    assert "m2" in text
    assert "IoU" in text


def test_empty_input_is_survivable():
    result = compare_rooms([], ROOM)
    assert result.iou == 0.0
    assert result.grade == "POOR"


def test_reference_rooms_are_available():
    assert reference_room("rectangular") is not None
    assert reference_room("nonexistent") is None

    from robotmap_common.geometry import polygon_area_m2

    assert polygon_area_m2(reference_room("rectangular")) == pytest.approx(27.0)
    assert polygon_area_m2(reference_room("l-shaped")) == pytest.approx(25.0)
