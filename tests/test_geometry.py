"""Unit tests for the kinematics and geometry core.

These are the calculations every downstream number depends on, so they are
tested against closed-form expected values rather than golden outputs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

from robotmap_common.geometry import (  # noqa: E402
    RobotGeometry,
    angle_difference_deg,
    bresenham_line,
    convex_hull,
    differential_drive_delta,
    douglas_peucker,
    gps_to_local_xy,
    haversine_m,
    local_xy_to_gps,
    min_area_rect,
    normalize_deg,
    polygon_area_m2,
    polygon_perimeter_m,
    project_ray,
    wrap_tick_delta,
)

GEOM = RobotGeometry(wheel_diameter_m=0.065, wheel_base_m=0.150, ticks_per_revolution=20)


# ── Wheel constants ───────────────────────────────────────────────────────────


def test_metres_per_tick_matches_circumference():
    assert GEOM.wheel_circumference_m == pytest.approx(math.pi * 0.065)
    assert GEOM.metres_per_tick == pytest.approx(math.pi * 0.065 / 20)


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError):
        RobotGeometry(wheel_diameter_m=0.0).validate()
    with pytest.raises(ValueError):
        RobotGeometry(wheel_base_m=-1.0).validate()


# ── Straight-line motion ──────────────────────────────────────────────────────


def test_one_full_revolution_travels_one_circumference():
    d = differential_drive_delta(20, 20, 0.0, GEOM)
    assert d.delta_x_m == pytest.approx(GEOM.wheel_circumference_m)
    assert d.delta_y_m == pytest.approx(0.0, abs=1e-12)
    assert d.delta_heading_deg == pytest.approx(0.0, abs=1e-12)


def test_straight_motion_follows_heading():
    d = differential_drive_delta(20, 20, 90.0, GEOM)
    assert d.delta_x_m == pytest.approx(0.0, abs=1e-12)
    assert d.delta_y_m == pytest.approx(GEOM.wheel_circumference_m)


def test_reverse_produces_negative_displacement():
    d = differential_drive_delta(-20, -20, 0.0, GEOM)
    assert d.delta_x_m == pytest.approx(-GEOM.wheel_circumference_m)
    assert d.distance_m == pytest.approx(GEOM.wheel_circumference_m)


# ── Rotation ──────────────────────────────────────────────────────────────────


def test_spin_in_place_produces_no_translation():
    d = differential_drive_delta(-20, 20, 0.0, GEOM)
    assert d.delta_x_m == pytest.approx(0.0, abs=1e-9)
    assert d.delta_y_m == pytest.approx(0.0, abs=1e-9)
    # Each wheel travels one circumference in opposite directions.
    expected_rad = 2 * GEOM.wheel_circumference_m / GEOM.wheel_base_m
    assert d.delta_heading_deg == pytest.approx(math.degrees(expected_rad))


def test_right_wheel_faster_turns_left():
    """Counter-clockwise is positive, so the outer wheel being the right
    wheel must produce a positive heading change."""
    d = differential_drive_delta(10, 20, 0.0, GEOM)
    assert d.delta_heading_deg > 0


# ── Arc integration ───────────────────────────────────────────────────────────


def test_arc_step_matches_closed_form_geometry():
    """A single curved step must land exactly on the arc about the
    instantaneous centre of curvature.

    Starting at the origin heading east and turning left, the centre of
    curvature sits at (0, R).  Rotating the robot about it by d_theta puts it
    at (R sin d, R(1 - cos d)) — the exact answer this asserts against.

    A straight-line (chord) approximation places the robot short of this
    point, so this is the test that fails if the arc integration is removed.
    """
    left_ticks, right_ticks = 40, 80  # strongly curved, so the error is large
    d = differential_drive_delta(left_ticks, right_ticks, 0.0, GEOM)

    d_center = (d.left_distance_m + d.right_distance_m) / 2.0
    d_theta = math.radians(d.delta_heading_deg)
    radius = d_center / d_theta

    assert d.delta_x_m == pytest.approx(radius * math.sin(d_theta))
    assert d.delta_y_m == pytest.approx(radius * (1.0 - math.cos(d_theta)))

    # And confirm the chord approximation really would have differed, so this
    # test cannot silently pass against a broken implementation.
    chord_x = d_center * math.cos(0.0)
    assert abs(d.delta_x_m - chord_x) > 1e-3


def test_arc_integration_closes_a_circle():
    """Integrating a constant-curvature arc through a full turn must bring
    the robot back to where it started."""
    x = y = 0.0
    heading = 0.0
    total_turn = 0.0

    step = differential_drive_delta(10, 11, 0.0, GEOM)
    step_deg = step.delta_heading_deg
    steps = round(360.0 / step_deg)

    for _ in range(steps):
        d = differential_drive_delta(10, 11, heading, GEOM)
        x += d.delta_x_m
        y += d.delta_y_m
        heading = normalize_deg(heading + d.delta_heading_deg)
        total_turn += d.delta_heading_deg

    # The loop can only close to a whole number of steps, so allow one step.
    assert total_turn == pytest.approx(360.0, abs=step_deg)

    # Residual gap should be no worse than the arc left undriven by that
    # partial step: 2 * R * sin(leftover / 2).
    radius = (step.left_distance_m + step.right_distance_m) / 2 / math.radians(step_deg)
    leftover_rad = math.radians(abs(360.0 - total_turn))
    max_gap = 2 * abs(radius) * math.sin(leftover_rad / 2) + 1e-6
    assert math.hypot(x, y) <= max_gap * 1.05


def test_left_wheel_trim_corrects_drift():
    """A robot whose left wheel is 5 % small curves when told to go straight;
    the trim factor must cancel that."""
    biased = RobotGeometry(ticks_per_revolution=20, left_wheel_trim=1.05)
    d = differential_drive_delta(20, 21, 0.0, biased)
    # Left travels 20*1.05 = 21 effective ticks, matching the right wheel.
    assert d.delta_heading_deg == pytest.approx(0.0, abs=1e-9)


# ── Counter overflow ──────────────────────────────────────────────────────────


def test_tick_delta_handles_32bit_wraparound():
    assert wrap_tick_delta(5, 4_294_967_290, 32) == 11


def test_tick_delta_handles_negative_wraparound():
    assert wrap_tick_delta(4_294_967_290, 5, 32) == -11


def test_tick_delta_normal_case():
    assert wrap_tick_delta(150, 100, 32) == 50


# ── Angles ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (350.0, 10.0, -20.0),  # must go backwards across the wrap
        (10.0, 350.0, 20.0),
        (180.0, 0.0, 180.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_angle_difference_takes_short_way_round(a, b, expected):
    assert angle_difference_deg(a, b) == pytest.approx(expected)


def test_normalize_wraps_into_range():
    assert normalize_deg(370.0) == pytest.approx(10.0)
    assert normalize_deg(-10.0) == pytest.approx(350.0)


# ── GPS projection ────────────────────────────────────────────────────────────

UTP_LAT, UTP_LON = 4.3852, 100.9739  # Universiti Teknologi PETRONAS


def test_gps_projection_roundtrip_is_sub_millimetre():
    target_lat, target_lon = UTP_LAT + 0.0001, UTP_LON + 0.0001
    east, north = gps_to_local_xy(target_lat, target_lon, UTP_LAT, UTP_LON)
    back_lat, back_lon = local_xy_to_gps(east, north, UTP_LAT, UTP_LON)
    assert haversine_m(target_lat, target_lon, back_lat, back_lon) < 0.001


def test_gps_anchor_is_origin():
    east, north = gps_to_local_xy(UTP_LAT, UTP_LON, UTP_LAT, UTP_LON)
    assert east == pytest.approx(0.0, abs=1e-9)
    assert north == pytest.approx(0.0, abs=1e-9)


def test_north_offset_is_positive_y():
    _, north = gps_to_local_xy(UTP_LAT + 0.001, UTP_LON, UTP_LAT, UTP_LON)
    assert north > 0
    # 0.001 degrees latitude is very close to 111 m everywhere on Earth.
    assert north == pytest.approx(111.0, abs=1.0)


# ── Ray projection ────────────────────────────────────────────────────────────


def test_forward_ray_projects_along_heading():
    hit = project_ray(1.0, 1.0, 0.0, 0.0, 2.0)
    assert hit == pytest.approx((3.0, 1.0))


def test_sensor_mount_angle_is_added_to_heading():
    hit = project_ray(0.0, 0.0, 90.0, -90.0, 2.0)
    # Robot faces north, sensor points 90 degrees right => due east.
    assert hit[0] == pytest.approx(2.0)
    assert hit[1] == pytest.approx(0.0, abs=1e-9)


# ── Grid traversal ────────────────────────────────────────────────────────────


def test_bresenham_includes_both_endpoints():
    cells = bresenham_line(0, 0, 3, 2)
    assert cells[0] == (0, 0)
    assert cells[-1] == (3, 2)


def test_bresenham_is_contiguous():
    cells = bresenham_line(0, 0, 10, 7)
    for (x1, y1), (x2, y2) in zip(cells, cells[1:], strict=False):
        assert max(abs(x2 - x1), abs(y2 - y1)) == 1


def test_bresenham_single_cell():
    assert bresenham_line(4, 4, 4, 4) == [(4, 4)]


# ── Polygon measurement ───────────────────────────────────────────────────────


def test_rectangle_area_and_perimeter():
    rect = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    assert polygon_area_m2(rect) == pytest.approx(12.0)
    assert polygon_perimeter_m(rect) == pytest.approx(14.0)


def test_area_is_orientation_independent():
    rect = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    assert polygon_area_m2(rect) == pytest.approx(polygon_area_m2(rect[::-1]))


def test_degenerate_polygon_has_zero_area():
    assert polygon_area_m2([(0.0, 0.0), (1.0, 1.0)]) == 0.0


# ── Simplification ────────────────────────────────────────────────────────────


# ── Oriented bounding rectangle ───────────────────────────────────────────────


def test_min_area_rect_of_axis_aligned_rectangle():
    rect = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.5), (0.0, 4.5)]
    long_side, short_side, _ = min_area_rect(rect)
    assert long_side == pytest.approx(6.0)
    assert short_side == pytest.approx(4.5)


@pytest.mark.parametrize("rotation_deg", [10.0, 15.0, 30.0, 45.0, 75.0])
def test_min_area_rect_recovers_sides_of_a_rotated_room(rotation_deg):
    """The whole reason this exists: a rotated room must still measure
    6.0 x 4.5, not the larger axis-aligned extent of its corners."""
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rect = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.5), (0.0, 4.5)]
    rotated = [(x * cos_t - y * sin_t, x * sin_t + y * cos_t) for x, y in rect]

    long_side, short_side, _ = min_area_rect(rotated)
    assert long_side == pytest.approx(6.0, abs=1e-6)
    assert short_side == pytest.approx(4.5, abs=1e-6)

    # And confirm the naive axis-aligned box really would have been wrong:
    # its area always exceeds the true 27 m2 for any non-right-angle rotation.
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    axis_aligned_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    assert axis_aligned_area > 27.0 + 1e-6


def test_min_area_rect_reports_orientation():
    theta = 30.0
    rad = math.radians(theta)
    rect = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.5), (0.0, 4.5)]
    rotated = [
        (x * math.cos(rad) - y * math.sin(rad), x * math.sin(rad) + y * math.cos(rad))
        for x, y in rect
    ]
    _, _, angle = min_area_rect(rotated)
    # Rectangle symmetry means the angle is only defined modulo 90 degrees.
    assert angle % 90.0 == pytest.approx(30.0, abs=0.01) or angle % 90.0 == pytest.approx(
        60.0, abs=0.01
    )


def test_min_area_rect_handles_degenerate_input():
    assert min_area_rect([]) == (0.0, 0.0, 0.0)
    long_side, short_side, _ = min_area_rect([(1.0, 1.0), (3.0, 1.0)])
    assert long_side == pytest.approx(2.0)


def test_convex_hull_drops_interior_points():
    points = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (2.0, 2.0)]
    hull = convex_hull(points)
    assert (2.0, 2.0) not in hull
    assert len(hull) == 4


def test_douglas_peucker_removes_collinear_points():
    line = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    assert douglas_peucker(line, 0.01) == [(0.0, 0.0), (3.0, 0.0)]


def test_douglas_peucker_keeps_real_corners():
    corner = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)]
    simplified = douglas_peucker(corner, 0.01)
    assert (2.0, 0.0) in simplified
    assert len(simplified) == 3


def test_douglas_peucker_collapses_staircase_noise():
    """A traced grid contour is a staircase; simplification should reduce it
    to roughly a straight line when the steps are below tolerance."""
    staircase = []
    for i in range(20):
        staircase.append((float(i) * 0.05, 0.0))
        staircase.append((float(i) * 0.05, 0.02))
    simplified = douglas_peucker(staircase, 0.05)
    assert len(simplified) < len(staircase) / 4
