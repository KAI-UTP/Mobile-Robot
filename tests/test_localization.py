"""Tests for sensor fusion, with emphasis on GPS rejection.

The central safety property of this system is that a bad GPS fix must never
be allowed to move the robot on the map. Most of these tests exist to prove
that property holds under the conditions that actually occur indoors.
"""

from __future__ import annotations

import math

import pytest
from localization.fusion import FilterConfig, PoseFilter
from robotmap_common.geometry import RobotGeometry, gps_to_local_xy, local_xy_to_gps
from robotmap_common.models import (
    EncoderData,
    GpsData,
    GpsFixQuality,
    ImuData,
    PoseSource,
    SensorPacket,
)

GEOM = RobotGeometry(wheel_diameter_m=0.065, wheel_base_m=0.150, ticks_per_revolution=20)
UTP_LAT, UTP_LON = 4.3852, 100.9739


def _packet(
    seq: int,
    left: int,
    right: int,
    imu: ImuData | None = None,
    gps: GpsData | None = None,
    dt_ms: int = 100,
) -> SensorPacket:
    return SensorPacket(
        robot_id="TEST01",
        timestamp="2026-08-15T00:00:00Z",
        sequence=seq,
        encoders=EncoderData(left_ticks=left, right_ticks=right, dt_ms=dt_ms),
        imu=imu,
        gps=gps,
    )


def _good_gps(lat: float = UTP_LAT, lon: float = UTP_LON) -> GpsData:
    return GpsData(
        latitude=lat,
        longitude=lon,
        fix_quality=GpsFixQuality.GPS,
        satellites=9,
        hdop=0.9,
    )


def _rtk_gps(lat: float = UTP_LAT, lon: float = UTP_LON) -> GpsData:
    """A centimetre-grade fix — the only kind precise enough to correct a
    room-scale map."""
    return GpsData(
        latitude=lat,
        longitude=lon,
        fix_quality=GpsFixQuality.RTK_FIXED,
        satellites=14,
        hdop=0.6,
    )


def _indoor_gps(lat: float = UTP_LAT, lon: float = UTP_LON) -> GpsData:
    """What a receiver actually reports inside a building: a fix exists, but
    few satellites and terrible geometry."""
    return GpsData(
        latitude=lat,
        longitude=lon,
        fix_quality=GpsFixQuality.GPS,
        satellites=3,
        hdop=12.5,
    )


# ── Odometry integration ──────────────────────────────────────────────────────


def test_first_packet_establishes_baseline_without_moving():
    f = PoseFilter("TEST01", GEOM)
    pose = f.update(_packet(0, 1000, 1000))
    assert pose.x_m == 0.0
    assert pose.y_m == 0.0


def test_straight_drive_accumulates_distance():
    f = PoseFilter("TEST01", GEOM)
    f.update(_packet(0, 0, 0))
    pose = f.update(_packet(1, 20, 20))

    assert pose.x_m == pytest.approx(GEOM.wheel_circumference_m, rel=1e-6)
    assert pose.distance_travelled_m == pytest.approx(GEOM.wheel_circumference_m)


def test_velocity_is_computed_from_dt():
    f = PoseFilter("TEST01", GEOM)
    f.update(_packet(0, 0, 0))
    pose = f.update(_packet(1, 20, 20, dt_ms=1000))
    assert pose.linear_velocity_mps == pytest.approx(GEOM.wheel_circumference_m, rel=1e-6)


def test_encoder_overflow_does_not_teleport_the_robot():
    f = PoseFilter("TEST01", GEOM)
    f.update(_packet(0, 4_294_967_290, 4_294_967_290))
    pose = f.update(_packet(1, 5, 5))
    # 11 ticks of real motion, not four billion.
    assert pose.distance_travelled_m == pytest.approx(11 * GEOM.metres_per_tick)


def test_uncertainty_grows_with_distance():
    f = PoseFilter("TEST01", GEOM)
    f.update(_packet(0, 0, 0))

    first = f.update(_packet(1, 20, 20))
    for i in range(2, 40):
        latest = f.update(_packet(i, 20 * i, 20 * i))

    assert latest.std_x_m > first.std_x_m
    assert latest.position_confidence < first.position_confidence


def test_pure_odometry_is_flagged_degraded_once_drifted():
    """Drive until predicted drift passes the threshold, then check the filter
    admits it. With the default 3 cm/m random walk, half a metre of one-sigma
    error arrives after (0.5 / 0.03)^2 = 278 m of driving."""
    cfg = FilterConfig()
    f = PoseFilter("TEST01", GEOM, cfg)
    f.update(_packet(0, 0, 0))

    step_m = GEOM.wheel_circumference_m
    steps_needed = int((cfg.degraded_position_std_m / cfg.odometry_position_noise_per_m) ** 2 / step_m) + 5

    for i in range(1, steps_needed):
        pose = f.update(_packet(i, 20 * i, 20 * i))

    assert pose.std_x_m > cfg.degraded_position_std_m
    assert pose.source == PoseSource.DEAD_RECKONING_DEGRADED


def test_uncertainty_is_independent_of_sampling_rate():
    """One metre driven must yield the same uncertainty whether it arrives in
    one packet or a hundred — otherwise the filter's confidence is an artefact
    of the telemetry rate."""
    coarse = PoseFilter("TEST01", GEOM)
    coarse.update(_packet(0, 0, 0))
    coarse.update(_packet(1, 100, 100))

    fine = PoseFilter("TEST01", GEOM)
    fine.update(_packet(0, 0, 0))
    for i in range(1, 101):
        fine.update(_packet(i, i, i))

    assert fine.distance_travelled_m == pytest.approx(coarse.distance_travelled_m)
    assert math.sqrt(fine.covariance.var_x) == pytest.approx(
        math.sqrt(coarse.covariance.var_x), rel=1e-6
    )


def test_reset_returns_to_origin():
    f = PoseFilter("TEST01", GEOM)
    f.update(_packet(0, 0, 0))
    f.update(_packet(1, 100, 100))
    f.reset()
    pose = f.update(_packet(2, 200, 200))
    assert pose.x_m == 0.0
    assert pose.distance_travelled_m == 0.0


# ── IMU ───────────────────────────────────────────────────────────────────────


def test_uncalibrated_imu_is_ignored():
    f = PoseFilter("TEST01", GEOM)
    imu = ImuData(heading_deg=90.0, gyro_z_dps=0.0, calibrated=False)
    f.update(_packet(0, 0, 0, imu=imu))
    pose = f.update(_packet(1, 20, 20, imu=imu))
    assert pose.source == PoseSource.ODOMETRY_ONLY


def test_imu_first_reading_sets_offset_not_heading():
    """The IMU's zero is arbitrary; adopting it directly would rotate the map."""
    f = PoseFilter("TEST01", GEOM)
    imu = ImuData(heading_deg=137.0, gyro_z_dps=0.0, calibrated=True)
    pose = f.update(_packet(0, 0, 0, imu=imu))
    assert pose.heading_deg == pytest.approx(0.0)


def test_imu_corrects_odometry_heading_drift():
    f = PoseFilter("TEST01", GEOM)
    imu = ImuData(heading_deg=0.0, gyro_z_dps=0.0, calibrated=True)
    f.update(_packet(0, 0, 0, imu=imu))

    # Drive far enough that odometry heading uncertainty grows large.
    for i in range(1, 30):
        f.update(_packet(i, 20 * i, 20 * i, imu=imu))

    # IMU insists we are still pointing at zero; heading should stay near zero.
    pose = f.update(_packet(30, 600, 600, imu=imu))
    assert abs(pose.heading_deg) < 5.0 or abs(pose.heading_deg - 360.0) < 5.0
    assert pose.source in (PoseSource.ODOMETRY_IMU, PoseSource.ODOMETRY_IMU_GPS)


def test_imu_ignored_during_fast_rotation():
    f = PoseFilter("TEST01", GEOM)
    calm = ImuData(heading_deg=0.0, gyro_z_dps=0.0, calibrated=True)
    f.update(_packet(0, 0, 0, imu=calm))

    spinning = ImuData(heading_deg=45.0, gyro_z_dps=500.0, calibrated=True)
    pose = f.update(_packet(1, 20, 20, imu=spinning))
    assert pose.source == PoseSource.ODOMETRY_ONLY


# ── GPS gating: the important part ────────────────────────────────────────────


def test_indoor_gps_is_rejected():
    """A weak indoor fix must not be accepted at all."""
    f = PoseFilter("TEST01", GEOM)
    gps = _indoor_gps()
    f.update(_packet(0, 0, 0, gps=gps))
    for i in range(1, 20):
        pose = f.update(_packet(i, 0, 0, gps=gps))

    assert f.anchor_lat is None
    assert f.gps_acceptances == 0
    assert f.gps_rejections >= 19
    assert pose.source != PoseSource.ODOMETRY_IMU_GPS


def test_no_fix_is_rejected():
    f = PoseFilter("TEST01", GEOM)
    gps = GpsData(
        latitude=UTP_LAT,
        longitude=UTP_LON,
        fix_quality=GpsFixQuality.NO_FIX,
        satellites=0,
        hdop=99.9,
    )
    f.update(_packet(0, 0, 0, gps=gps))
    f.update(_packet(1, 0, 0, gps=gps))
    assert f.gps_acceptances == 0
    assert "no satellite fix" in f.last_gps_reason


def test_high_hdop_is_rejected_with_reason():
    f = PoseFilter("TEST01", GEOM)
    gps = GpsData(
        latitude=UTP_LAT,
        longitude=UTP_LON,
        fix_quality=GpsFixQuality.GPS,
        satellites=8,
        hdop=5.0,
    )
    f.update(_packet(0, 0, 0, gps=gps))
    assert f.gps_acceptances == 0
    assert "HDOP" in f.last_gps_reason


def test_too_few_satellites_is_rejected():
    f = PoseFilter("TEST01", GEOM)
    gps = GpsData(
        latitude=UTP_LAT,
        longitude=UTP_LON,
        fix_quality=GpsFixQuality.GPS,
        satellites=4,
        hdop=1.0,
    )
    f.update(_packet(0, 0, 0, gps=gps))
    assert f.gps_acceptances == 0
    assert "satellites" in f.last_gps_reason


def test_anchor_requires_sustained_good_fixes():
    """One lucky fix must not define where the map is on Earth."""
    f = PoseFilter("TEST01", GEOM, FilterConfig(gps_anchor_required_fixes=5))
    f.update(_packet(0, 0, 0, gps=_good_gps()))
    assert f.anchor_lat is None  # one fix is not enough

    for i in range(1, 5):
        f.update(_packet(i, 0, 0, gps=_good_gps()))
    assert f.anchor_lat == pytest.approx(UTP_LAT)


def test_anchor_run_is_broken_by_a_bad_fix():
    f = PoseFilter("TEST01", GEOM, FilterConfig(gps_anchor_required_fixes=5))
    for i in range(3):
        f.update(_packet(i, 0, 0, gps=_good_gps()))
    f.update(_packet(3, 0, 0, gps=_indoor_gps()))  # resets the run
    for i in range(4, 7):
        f.update(_packet(i, 0, 0, gps=_good_gps()))
    assert f.anchor_lat is None


def test_anchoring_does_not_move_the_robot():
    """The map origin is where the robot started; GPS georeferences it but
    must not shift it."""
    f = PoseFilter("TEST01", GEOM, FilterConfig(gps_anchor_required_fixes=3))
    f.update(_packet(0, 0, 0))
    for i in range(1, 6):
        pose = f.update(_packet(i, 0, 0, gps=_good_gps()))

    assert f.anchor_lat is not None
    assert pose.x_m == pytest.approx(0.0, abs=1e-9)
    assert pose.y_m == pytest.approx(0.0, abs=1e-9)


def test_wild_outlier_fix_is_gated_out():
    """A fix hundreds of metres away must be rejected, not averaged in."""
    f = PoseFilter("TEST01", GEOM, FilterConfig(gps_anchor_required_fixes=3))
    for i in range(5):
        f.update(_packet(i, 0, 0, gps=_rtk_gps()))

    corrections_before = f.gps_corrections
    # A fix 0.005 degrees away is roughly 500 m — a multipath reflection.
    f.update(_packet(6, 0, 0, gps=_rtk_gps(UTP_LAT + 0.005, UTP_LON)))

    assert f.gps_corrections == corrections_before
    assert "outlier" in f.last_gps_reason
    assert abs(f.y_m) < 5.0


# ── Accuracy gate: coarse fixes georeference but must not steer ───────────────


def test_consumer_gps_anchors_but_never_corrects():
    """The project's central design rule.

    A 4 m-accurate fix is as wide as the room being mapped. It is good enough
    to say which building the robot is in, and useless for saying where in the
    room it stands. It must therefore anchor the map and nothing more.
    """
    f = PoseFilter("TEST01", GEOM, FilterConfig(gps_anchor_required_fixes=3))
    for i in range(40):
        f.update(_packet(i, 20 * i, 20 * i, gps=_good_gps()))

    assert f.anchor_lat is not None, "should still georeference the map"
    assert f.gps_acceptances > 0, "fixes are valid, just imprecise"
    assert f.gps_corrections == 0, "must not steer the map with a 4 m fix"
    assert "too coarse" in f.last_gps_reason


def _gps_at_map_position(f: PoseFilter, x_m: float, y_m: float) -> GpsData:
    """An RTK fix reporting a given position in the filter's own map frame."""
    lat, lon = local_xy_to_gps(
        x_m - f.anchor_map_x, y_m - f.anchor_map_y, f.anchor_lat, f.anchor_lon
    )
    return _rtk_gps(lat, lon)


def test_rtk_gps_is_allowed_to_correct():
    """A centimetre-grade fix is more precise than the map and may steer it."""
    f = PoseFilter("TEST01", GEOM, FilterConfig(gps_anchor_required_fixes=3))
    # Stationary, so the fix and the encoders agree and nothing is gated out.
    for i in range(10):
        f.update(_packet(i, 0, 0, gps=_rtk_gps()))

    assert f.gps_corrections > 0


def test_stationary_gps_disagreeing_with_encoders_is_rejected():
    """If the wheels say the robot moved 8 m and GPS says it did not, one of
    them is wrong and the filter must not silently split the difference."""
    f = PoseFilter("TEST01", GEOM, FilterConfig(gps_anchor_required_fixes=3))
    for i in range(5):
        f.update(_packet(i, 0, 0, gps=_rtk_gps()))

    corrections_before = f.gps_corrections
    for i in range(5, 45):
        f.update(_packet(i, 20 * i, 20 * i, gps=_rtk_gps()))

    assert f.gps_corrections == corrections_before
    assert "outlier" in f.last_gps_reason


def test_accuracy_estimate_reflects_correction_grade():
    """Fix quality, not just HDOP, determines usable accuracy."""
    assert _good_gps().estimated_accuracy_m > 1.0
    assert _rtk_gps().estimated_accuracy_m < 0.1


def test_anchor_records_map_position():
    """The anchor is set after several fixes, by which time the robot has
    moved; that offset must be recorded or every GPS position is shifted."""
    f = PoseFilter("TEST01", GEOM, FilterConfig(gps_anchor_required_fixes=5))
    f.update(_packet(0, 0, 0))
    for i in range(1, 6):
        f.update(_packet(i, 20 * i, 20 * i, gps=_rtk_gps()))

    assert f.anchor_lat is not None
    assert f.anchor_map_x == pytest.approx(f.x_m, abs=0.3)
    assert f.anchor_map_x > 0.0, "robot had moved before the anchor was set"


def test_correction_is_clamped_to_maximum():
    """Even an accepted correction cannot teleport the robot."""
    cfg = FilterConfig(
        gps_anchor_required_fixes=1,
        gps_max_correction_m=1.0,
        gps_outlier_sigma=1e9,  # disable the gate to isolate the clamp
    )
    f = PoseFilter("TEST01", GEOM, cfg)
    f.update(_packet(0, 0, 0, gps=_rtk_gps()))

    # Drive a long way so the filter is uncertain and the gain is high.
    for i in range(1, 60):
        f.update(_packet(i, 20 * i, 20 * i))

    x_before, y_before = f.x_m, f.y_m
    f.update(_packet(100, 20 * 59, 20 * 59, gps=_rtk_gps(UTP_LAT + 0.001, UTP_LON)))
    assert math.hypot(f.x_m - x_before, f.y_m - y_before) <= 1.0 + 1e-6


def test_precise_gps_reduces_uncertainty():
    f = PoseFilter("TEST01", GEOM, FilterConfig(gps_anchor_required_fixes=1))
    f.update(_packet(0, 0, 0, gps=_rtk_gps()))
    for i in range(1, 40):
        f.update(_packet(i, 20 * i, 20 * i))

    std_before = math.sqrt(f.covariance.var_x)
    # A fix that agrees with where the robot believes it is: no correction to
    # apply, but the independent confirmation still reduces uncertainty.
    confirming = _gps_at_map_position(f, f.x_m, f.y_m)
    f.update(_packet(50, 20 * 39, 20 * 39, gps=confirming))
    std_after = math.sqrt(f.covariance.var_x)

    assert std_after < std_before
    assert f.gps_corrections >= 1


def test_gps_corrects_toward_true_position():
    """With a precise fix, the estimate should move toward it."""
    cfg = FilterConfig(gps_anchor_required_fixes=1)
    f = PoseFilter("TEST01", GEOM, cfg)
    f.update(_packet(0, 0, 0, gps=_rtk_gps()))

    # Accumulate drift by driving.
    for i in range(1, 40):
        f.update(_packet(i, 20 * i, 20 * i))

    # A fix a couple of metres north of the anchor, expressed in the map frame.
    target = _rtk_gps(UTP_LAT + 0.00002, UTP_LON)
    _, north = gps_to_local_xy(target.latitude, target.longitude, UTP_LAT, UTP_LON)
    target_y = north + f.anchor_map_y

    y_before = f.y_m
    f.update(_packet(60, 20 * 39, 20 * 39, gps=target))

    # Moved toward the fix, without overshooting past it.
    assert abs(f.y_m - target_y) <= abs(y_before - target_y)


def test_diagnostics_report_rejection_counts():
    f = PoseFilter("TEST01", GEOM)
    for i in range(5):
        f.update(_packet(i, 0, 0, gps=_indoor_gps()))
    diag = f.diagnostics()
    assert diag["gps_rejections"] == 5
    assert diag["anchored"] is False


# ── GpsData helpers ───────────────────────────────────────────────────────────


def test_usability_helper_matches_gate():
    assert _good_gps().is_usable_for_position is True
    assert _indoor_gps().is_usable_for_position is False


def test_accuracy_estimate_scales_with_hdop():
    good = _good_gps()
    bad = _indoor_gps()
    assert bad.estimated_accuracy_m > good.estimated_accuracy_m
    # An indoor fix should be estimated as tens of metres — far worse than a room.
    assert bad.estimated_accuracy_m > 20.0
