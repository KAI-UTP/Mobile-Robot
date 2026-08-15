"""The pose filter driven by three-wheel holonomic packets.

Confirms the filter routes holonomic packets to the kiwi-drive maths, that a
differential packet is unaffected, and that the extra uncertainty of omni
wheels is actually reflected in the covariance.
"""

from __future__ import annotations

import math

import pytest
from localization.fusion import FilterConfig, PoseFilter
from robotmap_common.holonomic import (
    BodyTwist,
    HolonomicGeometry,
    inverse_kinematics,
)
from robotmap_common.models import (
    DriveKind,
    EncoderData,
    ImuData,
    SensorPacket,
)

HOLO = HolonomicGeometry(
    wheel_radius_m=0.029,
    wheel_offset_m=0.100,
    wheel_angles_deg=(0.0, 120.0, 240.0),
    ticks_per_revolution=4096,
)
DT_S = 0.1


def _holo_packet(seq: int, ticks: tuple[int, int, int], imu_heading: float | None = None):
    imu = (
        ImuData(heading_deg=imu_heading % 360, gyro_z_dps=0.0, calibrated=True)
        if imu_heading is not None
        else None
    )
    return SensorPacket(
        robot_id="TEST01",
        timestamp="2026-08-15T00:00:00Z",
        sequence=seq,
        drive=DriveKind.HOLONOMIC_3WHEEL,
        encoders=EncoderData(
            left_ticks=ticks[0], right_ticks=ticks[1], dt_ms=int(DT_S * 1000)
        ),
        wheel_ticks=list(ticks),
        imu=imu,
    )


def _ticks_for(twist: BodyTwist, dt_s: float) -> tuple[int, int, int]:
    """Encoder counts the wheels would produce for a commanded twist."""
    speeds = inverse_kinematics(twist, HOLO)
    return tuple(  # type: ignore[return-value]
        round(v * dt_s / HOLO.metres_per_tick) for v in speeds.values
    )


def _drive(filter_, twist: BodyTwist, steps: int):
    """Feed the filter a run of packets for a steady twist."""
    cumulative = [0, 0, 0]
    pose = filter_.update(_holo_packet(0, tuple(cumulative)))
    per_step = _ticks_for(twist, DT_S)

    for step in range(1, steps + 1):
        cumulative = [cumulative[i] + per_step[i] for i in range(3)]
        pose = filter_.update(_holo_packet(step, tuple(cumulative)))
    return pose


# ── Routing ──────────────────────────────────────────────────────────────────


def test_holonomic_packet_is_detected():
    packet = _holo_packet(0, (0, 0, 0))
    assert packet.is_holonomic is True


def test_differential_packet_is_not_treated_as_holonomic():
    packet = SensorPacket(
        robot_id="TEST01",
        timestamp="2026-08-15T00:00:00Z",
        sequence=0,
        encoders=EncoderData(left_ticks=0, right_ticks=0, dt_ms=100),
    )
    assert packet.is_holonomic is False


def test_wheel_ticks_length_is_validated():
    with pytest.raises(ValueError):
        SensorPacket(
            robot_id="TEST01",
            timestamp="2026-08-15T00:00:00Z",
            sequence=0,
            encoders=EncoderData(left_ticks=0, right_ticks=0, dt_ms=100),
            wheel_ticks=[1],
        )


def test_first_packet_only_sets_a_baseline():
    f = PoseFilter("TEST01", holonomic_geometry=HOLO)
    pose = f.update(_holo_packet(0, (5000, 5000, 5000)))
    assert pose.x_m == 0.0
    assert pose.y_m == 0.0


# ── Motion ───────────────────────────────────────────────────────────────────


def test_driving_forward_moves_along_x():
    f = PoseFilter("TEST01", holonomic_geometry=HOLO)
    pose = _drive(f, BodyTwist(vx_mps=0.2), steps=50)

    assert pose.x_m == pytest.approx(0.2 * 50 * DT_S, rel=0.02)
    assert pose.y_m == pytest.approx(0.0, abs=0.01)


def test_strafing_moves_sideways_without_turning():
    """The behaviour a differential drive cannot produce at all."""
    f = PoseFilter("TEST01", holonomic_geometry=HOLO)
    pose = _drive(f, BodyTwist(vy_mps=0.2), steps=50)

    assert pose.y_m == pytest.approx(0.2 * 50 * DT_S, rel=0.02)
    assert pose.x_m == pytest.approx(0.0, abs=0.01)
    assert pose.heading_deg == pytest.approx(0.0, abs=1.0)


def test_rotation_changes_heading_without_translating():
    f = PoseFilter("TEST01", holonomic_geometry=HOLO)
    pose = _drive(f, BodyTwist(omega_dps=45.0), steps=40)

    assert pose.heading_deg == pytest.approx(45.0 * 40 * DT_S, rel=0.05)
    assert math.hypot(pose.x_m, pose.y_m) < 0.02


def test_diagonal_motion_combines_both_axes():
    f = PoseFilter("TEST01", holonomic_geometry=HOLO)
    pose = _drive(f, BodyTwist(vx_mps=0.15, vy_mps=0.15), steps=40)

    expected = 0.15 * 40 * DT_S
    assert pose.x_m == pytest.approx(expected, rel=0.03)
    assert pose.y_m == pytest.approx(expected, rel=0.03)


def test_strafing_square_returns_to_origin():
    """Drive a square by strafing only, never rotating. A differential robot
    would have to turn at each corner."""
    f = PoseFilter("TEST01", holonomic_geometry=HOLO)
    cumulative = [0, 0, 0]
    f.update(_holo_packet(0, tuple(cumulative)))

    seq = 1
    for twist in (
        BodyTwist(vx_mps=0.2),
        BodyTwist(vy_mps=0.2),
        BodyTwist(vx_mps=-0.2),
        BodyTwist(vy_mps=-0.2),
    ):
        per_step = _ticks_for(twist, DT_S)
        for _ in range(25):
            cumulative = [cumulative[i] + per_step[i] for i in range(3)]
            pose = f.update(_holo_packet(seq, tuple(cumulative)))
            seq += 1

    assert math.hypot(pose.x_m, pose.y_m) < 0.05
    assert pose.heading_deg == pytest.approx(0.0, abs=1.0)


def test_encoder_overflow_does_not_teleport_the_robot():
    f = PoseFilter("TEST01", holonomic_geometry=HOLO)
    f.update(_holo_packet(0, (4_294_967_290, 4_294_967_290, 4_294_967_290)))
    pose = f.update(_holo_packet(1, (5, 5, 5)))

    # Eleven counts of real motion, not four billion. All three wheels moving
    # together is pure rotation, so no translation either.
    assert math.hypot(pose.x_m, pose.y_m) < 0.01


def test_missing_wheel_ticks_is_ignored_safely():
    """A malformed holonomic packet must not crash or corrupt the pose."""
    f = PoseFilter("TEST01", holonomic_geometry=HOLO)
    packet = _holo_packet(0, (0, 0, 0))
    packet.wheel_ticks = None
    pose = f.update(packet)
    assert pose.x_m == 0.0


# ── Uncertainty ──────────────────────────────────────────────────────────────


def test_holonomic_odometry_is_trusted_less_than_differential():
    """Omni rollers slide by design, and that sliding is invisible to the
    encoders, so the filter must be correspondingly less confident."""
    config = FilterConfig(holonomic_noise_multiplier=2.0)

    holo = PoseFilter("TEST01", config=config, holonomic_geometry=HOLO)
    pose = _drive(holo, BodyTwist(vx_mps=0.2), steps=100)

    plain = PoseFilter("TEST01", config=FilterConfig(holonomic_noise_multiplier=1.0),
                       holonomic_geometry=HOLO)
    plain_pose = _drive(plain, BodyTwist(vx_mps=0.2), steps=100)

    assert pose.std_x_m > plain_pose.std_x_m
    # Sigma scales with the multiplier, variance with its square.
    assert pose.std_x_m == pytest.approx(plain_pose.std_x_m * 2.0, rel=0.01)


def test_uncertainty_still_grows_with_distance():
    f = PoseFilter("TEST01", holonomic_geometry=HOLO)
    short = _drive(f, BodyTwist(vx_mps=0.2), steps=10)
    short_std = short.std_x_m

    long = _drive(f, BodyTwist(vx_mps=0.2), steps=200)
    assert long.std_x_m > short_std


def test_imu_still_corrects_heading_on_holonomic():
    """The IMU path must work regardless of drive geometry."""
    f = PoseFilter("TEST01", holonomic_geometry=HOLO)
    cumulative = [0, 0, 0]
    f.update(_holo_packet(0, tuple(cumulative), imu_heading=0.0))

    per_step = _ticks_for(BodyTwist(vx_mps=0.2), DT_S)
    for step in range(1, 60):
        cumulative = [cumulative[i] + per_step[i] for i in range(3)]
        pose = f.update(_holo_packet(step, tuple(cumulative), imu_heading=0.0))

    # The IMU insists we never turned, so the heading must stay near zero.
    assert abs(pose.heading_deg) < 5.0 or abs(pose.heading_deg - 360.0) < 5.0


def test_reset_clears_holonomic_state():
    f = PoseFilter("TEST01", holonomic_geometry=HOLO)
    _drive(f, BodyTwist(vx_mps=0.2), steps=30)
    assert f.distance_travelled_m > 0

    f.reset()
    pose = _drive(f, BodyTwist(vx_mps=0.2), steps=10)
    assert pose.x_m == pytest.approx(0.2 * 10 * DT_S, rel=0.05)
