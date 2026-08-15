"""Tests for kiwi-drive kinematics.

These run without Isaac Sim installed, which is the point: all the movement
maths lives in pure Python so it can be verified on any machine, and the Isaac
layer stays a thin wrapper that only applies numbers this module produced.
"""

from __future__ import annotations

import math

import pytest
from robotmap_common.holonomic import (
    BodyTwist,
    HolonomicGeometry,
    WheelSpeeds,
    forward_kinematics,
    integrate_twist,
    inverse_kinematics,
    odometry_from_ticks,
    scale_to_limit,
    twist_from_direction,
    wheel_deltas_to_speeds,
)

GEOM = HolonomicGeometry(
    wheel_radius_m=0.029,
    wheel_offset_m=0.100,
    wheel_angles_deg=(0.0, 120.0, 240.0),
    ticks_per_revolution=4096,
)


# ── Configuration validation ─────────────────────────────────────────────────


def test_valid_geometry_passes():
    GEOM.validate()


def test_negative_dimensions_rejected():
    with pytest.raises(ValueError):
        HolonomicGeometry(wheel_radius_m=0.0).validate()
    with pytest.raises(ValueError):
        HolonomicGeometry(wheel_offset_m=-0.1).validate()


def test_wrong_wheel_count_rejected():
    with pytest.raises(ValueError):
        HolonomicGeometry(wheel_angles_deg=(0.0, 180.0)).validate()


def test_nearly_collinear_wheels_rejected():
    """Three wheels bunched together cannot span the plane, and the forward
    kinematics matrix would be singular."""
    with pytest.raises(ValueError, match="singular"):
        HolonomicGeometry(wheel_angles_deg=(0.0, 5.0, 240.0)).validate()


# ── Inverse kinematics ───────────────────────────────────────────────────────


def test_stationary_command_stops_every_wheel():
    speeds = inverse_kinematics(BodyTwist(), GEOM)
    assert speeds.values == pytest.approx((0.0, 0.0, 0.0))


def test_pure_rotation_drives_all_wheels_equally():
    """Spinning on the spot is the one motion where all three wheels do the
    same thing — a good sanity check on the omega term."""
    speeds = inverse_kinematics(BodyTwist(omega_dps=90.0), GEOM)
    expected = math.radians(90.0) * GEOM.wheel_offset_m
    for v in speeds.values:
        assert v == pytest.approx(expected)


def test_forward_motion_leaves_front_wheel_still():
    """Wheel 1 sits at 0 degrees, so its rolling direction is perpendicular to
    'forward'. Driving straight ahead should spin it not at all — the rollers
    carry that motion instead."""
    speeds = inverse_kinematics(BodyTwist(vx_mps=0.2), GEOM)
    assert speeds.values[0] == pytest.approx(0.0, abs=1e-12)
    # The other two must oppose each other to produce net forward motion.
    assert speeds.values[1] < 0
    assert speeds.values[2] > 0


def test_strafing_requires_no_rotation_command():
    """The defining property of a holonomic base: it can move sideways without
    turning. On a differential drive this command is impossible."""
    speeds = inverse_kinematics(BodyTwist(vy_mps=0.2), GEOM)
    recovered = forward_kinematics(speeds, GEOM)

    assert recovered.vy_mps == pytest.approx(0.2)
    assert recovered.vx_mps == pytest.approx(0.0, abs=1e-12)
    assert recovered.omega_dps == pytest.approx(0.0, abs=1e-12)


def test_wheel_speeds_scale_linearly_with_command():
    slow = inverse_kinematics(BodyTwist(vx_mps=0.1), GEOM)
    fast = inverse_kinematics(BodyTwist(vx_mps=0.2), GEOM)
    for a, b in zip(slow.values, fast.values, strict=True):
        assert b == pytest.approx(a * 2.0)


# ── Round trip ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "vx,vy,omega",
    [
        (0.2, 0.0, 0.0),
        (0.0, 0.2, 0.0),
        (0.0, 0.0, 45.0),
        (0.15, -0.1, 30.0),
        (-0.2, 0.05, -60.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_inverse_then_forward_is_identity(vx, vy, omega):
    """IK followed by FK must return the original command exactly.

    This is the strongest single check on both functions: an error in either
    the wheel equation or the matrix inverse breaks it.
    """
    twist = BodyTwist(vx_mps=vx, vy_mps=vy, omega_dps=omega)
    recovered = forward_kinematics(inverse_kinematics(twist, GEOM), GEOM)

    assert recovered.vx_mps == pytest.approx(vx, abs=1e-12)
    assert recovered.vy_mps == pytest.approx(vy, abs=1e-12)
    assert recovered.omega_dps == pytest.approx(omega, abs=1e-9)


@pytest.mark.parametrize("angles", [(0.0, 120.0, 240.0), (60.0, 180.0, 300.0), (30.0, 150.0, 270.0)])
def test_round_trip_holds_for_other_wheel_layouts(angles):
    """The maths must not be hard-wired to one mounting arrangement."""
    geom = HolonomicGeometry(wheel_angles_deg=angles)
    twist = BodyTwist(vx_mps=0.12, vy_mps=-0.07, omega_dps=25.0)
    recovered = forward_kinematics(inverse_kinematics(twist, geom), geom)

    assert recovered.vx_mps == pytest.approx(0.12, abs=1e-12)
    assert recovered.vy_mps == pytest.approx(-0.07, abs=1e-12)
    assert recovered.omega_dps == pytest.approx(25.0, abs=1e-9)


def test_wheel_trim_cancels_in_the_round_trip():
    """Calibration must not leak into the odometry estimate as phantom motion."""
    trimmed = HolonomicGeometry(wheel_trim=(1.03, 0.98, 1.01))
    twist = BodyTwist(vx_mps=0.2, vy_mps=0.1, omega_dps=15.0)
    recovered = forward_kinematics(inverse_kinematics(twist, trimmed), trimmed)

    assert recovered.vx_mps == pytest.approx(0.2, abs=1e-12)
    assert recovered.vy_mps == pytest.approx(0.1, abs=1e-12)
    assert recovered.omega_dps == pytest.approx(15.0, abs=1e-9)


# ── Speed limiting ───────────────────────────────────────────────────────────


def test_speeds_below_limit_are_untouched():
    speeds = WheelSpeeds((0.1, -0.2, 0.15))
    scaled, was_scaled = scale_to_limit(speeds, 0.5)
    assert not was_scaled
    assert scaled.values == speeds.values


def test_scaling_preserves_direction_of_travel():
    """The crucial property: scaling must not change *where* the robot goes.

    Clipping each wheel independently would alter the ratios between them, and
    on a holonomic base those ratios are the direction of travel — a robot
    asked to drive diagonally would set off at the wrong angle rather than
    simply more slowly.
    """
    twist = BodyTwist(vx_mps=1.5, vy_mps=0.9, omega_dps=40.0)
    speeds = inverse_kinematics(twist, GEOM)
    scaled, was_scaled = scale_to_limit(speeds, 0.3)

    assert was_scaled
    assert scaled.max_abs == pytest.approx(0.3)

    original_dir = math.atan2(twist.vy_mps, twist.vx_mps)
    recovered = forward_kinematics(scaled, GEOM)
    scaled_dir = math.atan2(recovered.vy_mps, recovered.vx_mps)
    assert scaled_dir == pytest.approx(original_dir, abs=1e-9)


def test_scaling_reduces_translation_and_rotation_together():
    twist = BodyTwist(vx_mps=1.0, vy_mps=0.0, omega_dps=60.0)
    speeds = inverse_kinematics(twist, GEOM)
    scaled, _ = scale_to_limit(speeds, 0.2)
    recovered = forward_kinematics(scaled, GEOM)

    ratio_linear = recovered.vx_mps / twist.vx_mps
    ratio_angular = recovered.omega_dps / twist.omega_dps
    assert ratio_linear == pytest.approx(ratio_angular, abs=1e-9)


def test_invalid_limit_rejected():
    with pytest.raises(ValueError):
        scale_to_limit(WheelSpeeds((0.1, 0.1, 0.1)), 0.0)


# ── Odometry integration ─────────────────────────────────────────────────────


def test_straight_motion_integrates_along_heading():
    delta = integrate_twist(BodyTwist(vx_mps=1.0), heading_deg=0.0, dt_s=1.0)
    assert delta.delta_x_m == pytest.approx(1.0)
    assert delta.delta_y_m == pytest.approx(0.0, abs=1e-12)
    assert delta.delta_heading_deg == pytest.approx(0.0)


def test_body_frame_rotates_into_world_frame():
    """Driving 'forward' while facing 90 degrees must move the robot north."""
    delta = integrate_twist(BodyTwist(vx_mps=1.0), heading_deg=90.0, dt_s=1.0)
    assert delta.delta_x_m == pytest.approx(0.0, abs=1e-12)
    assert delta.delta_y_m == pytest.approx(1.0)


def test_strafe_is_perpendicular_to_heading():
    delta = integrate_twist(BodyTwist(vy_mps=1.0), heading_deg=0.0, dt_s=1.0)
    assert delta.delta_x_m == pytest.approx(0.0, abs=1e-12)
    assert delta.delta_y_m == pytest.approx(1.0)


def test_rotation_in_place_produces_no_translation():
    delta = integrate_twist(BodyTwist(omega_dps=90.0), heading_deg=0.0, dt_s=1.0)
    assert delta.delta_x_m == pytest.approx(0.0, abs=1e-12)
    assert delta.delta_y_m == pytest.approx(0.0, abs=1e-12)
    assert delta.delta_heading_deg == pytest.approx(90.0)


def test_arc_integration_matches_closed_form():
    """Translating while rotating traces a circle of radius v/omega.

    Starting at the origin heading east and turning left, the exact endpoint
    after sweeping d radians is (R sin d, R(1 - cos d)). This is the test that
    fails if the arc integration is replaced by the straight-line shortcut.
    """
    v = 0.5
    omega_dps = 90.0
    dt = 1.0

    delta = integrate_twist(
        BodyTwist(vx_mps=v, omega_dps=omega_dps), heading_deg=0.0, dt_s=dt
    )

    omega_rad = math.radians(omega_dps)
    radius = v / omega_rad
    swept = omega_rad * dt

    assert delta.delta_x_m == pytest.approx(radius * math.sin(swept))
    assert delta.delta_y_m == pytest.approx(radius * (1.0 - math.cos(swept)))

    # Confirm the straight-line approximation really would have differed, so
    # this test cannot silently pass against a broken implementation.
    assert abs(delta.delta_x_m - v * dt) > 1e-3


def test_integrating_a_full_rotation_returns_to_start():
    """Four 90-degree arcs must close the loop."""
    x = y = heading = 0.0
    for _ in range(4):
        delta = integrate_twist(
            BodyTwist(vx_mps=0.5, omega_dps=90.0), heading_deg=heading, dt_s=1.0
        )
        x += delta.delta_x_m
        y += delta.delta_y_m
        heading = (heading + delta.delta_heading_deg) % 360.0

    assert math.hypot(x, y) < 1e-9
    assert heading == pytest.approx(0.0, abs=1e-9)


def test_holonomic_square_without_turning():
    """Drive a 2 m square by strafing, never rotating.

    A differential robot cannot do this at all; it would have to stop and turn
    at every corner. Ending back at the origin with the original heading is the
    clearest demonstration that the kinematics are genuinely holonomic.
    """
    x = y = 0.0
    heading = 0.0
    for direction in (0.0, 90.0, 180.0, 270.0):
        twist = twist_from_direction(direction, speed_mps=1.0)
        delta = integrate_twist(twist, heading_deg=heading, dt_s=2.0)
        x += delta.delta_x_m
        y += delta.delta_y_m

    assert x == pytest.approx(0.0, abs=1e-12)
    assert y == pytest.approx(0.0, abs=1e-12)
    assert heading == pytest.approx(0.0)


# ── Ticks to motion ──────────────────────────────────────────────────────────


def test_one_wheel_revolution_equals_one_circumference():
    speeds = wheel_deltas_to_speeds((4096, 0, 0), dt_s=1.0, geometry=GEOM)
    assert speeds.values[0] == pytest.approx(GEOM.wheel_circumference_m)


def test_zero_interval_yields_zero_speed():
    speeds = wheel_deltas_to_speeds((100, 100, 100), dt_s=0.0, geometry=GEOM)
    assert speeds.values == (0.0, 0.0, 0.0)


def test_odometry_from_ticks_recovers_commanded_motion():
    """The full loop: command a twist, work out the ticks it would produce,
    then recover the motion from those ticks."""
    twist = BodyTwist(vx_mps=0.2, vy_mps=0.1, omega_dps=20.0)
    dt = 0.1

    speeds = inverse_kinematics(twist, GEOM)
    ticks = tuple(
        round(v * dt / GEOM.metres_per_tick) for v in speeds.values
    )

    delta = odometry_from_ticks(ticks, heading_deg=0.0, dt_s=dt, geometry=GEOM)
    expected = integrate_twist(twist, heading_deg=0.0, dt_s=dt)

    # Quantisation to whole ticks costs a little precision; at 4096 counts per
    # revolution that is well under a millimetre.
    assert delta.delta_x_m == pytest.approx(expected.delta_x_m, abs=1e-4)
    assert delta.delta_y_m == pytest.approx(expected.delta_y_m, abs=1e-4)
    assert delta.delta_heading_deg == pytest.approx(expected.delta_heading_deg, abs=0.05)


def test_bus_servo_encoder_resolution_is_ample():
    """The differential build needed >=360 counts/rev. A bus servo's 12-bit
    absolute encoder gives 4096, so the encoder problem that dominated the
    previous design does not arise on this platform."""
    assert GEOM.ticks_per_revolution >= 360
    # Sub-millimetre resolution at the wheel rim.
    assert GEOM.metres_per_tick < 0.001


# ── Named motions ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "direction,expected_x,expected_y",
    [
        (0.0, 1.0, 0.0),      # forward
        (90.0, 0.0, 1.0),     # left
        (180.0, -1.0, 0.0),   # backward
        (-90.0, 0.0, -1.0),   # right
    ],
)
def test_named_directions(direction, expected_x, expected_y):
    twist = twist_from_direction(direction, speed_mps=1.0)
    assert twist.vx_mps == pytest.approx(expected_x, abs=1e-12)
    assert twist.vy_mps == pytest.approx(expected_y, abs=1e-12)


def test_direction_and_rotation_combine():
    """Strafing right while spinning left — the motion that justifies the
    platform."""
    twist = twist_from_direction(-90.0, speed_mps=0.2, omega_dps=45.0)
    assert twist.vy_mps == pytest.approx(-0.2)
    assert twist.omega_dps == pytest.approx(45.0)

    recovered = forward_kinematics(inverse_kinematics(twist, GEOM), GEOM)
    assert recovered.vy_mps == pytest.approx(-0.2, abs=1e-12)
    assert recovered.omega_dps == pytest.approx(45.0, abs=1e-9)
