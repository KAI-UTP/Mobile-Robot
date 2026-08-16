"""Folding BLE beacons into the pose, and why it is switched off.

The beacons are one of only three sensors this robot has, and the code to use
them works. It is disabled by default anyway, because measuring it showed it
makes the pose worse — and these tests exist so that decision cannot be quietly
reversed without someone re-measuring.
"""

from __future__ import annotations

import pytest
from localization.fusion import FilterConfig, PoseFilter
from robotmap_common.models import (
    BeaconSample,
    DriveKind,
    EncoderData,
    SensorPacket,
)
from robotmap_common.rssi import BeaconLayout, distance_to_rssi

LAYOUT = BeaconLayout.room_corners(6.0, 4.5)


def _packet(sequence: int, beacons=None, ticks=(0, 0, 0)) -> SensorPacket:
    return SensorPacket(
        robot_id="TEST01",
        timestamp="2026-08-16T00:00:00Z",
        sequence=sequence,
        drive=DriveKind.HOLONOMIC_3WHEEL,
        wheel_ticks=list(ticks),
        encoders=EncoderData(left_ticks=ticks[0], right_ticks=ticks[1], dt_ms=100),
        beacons=beacons or [],
    )


def _samples_at(x: float, y: float) -> list[BeaconSample]:
    """Perfect, noiseless RSSI as heard from (x, y)."""
    return [
        BeaconSample(
            beacon_id=b.beacon_id,
            rssi_dbm=max(-120.0, min(0.0, distance_to_rssi(b.distance_to(x, y)))),
            sample_count=8,
        )
        for b in LAYOUT.as_list()
    ]


def _filter(**config):
    return PoseFilter(
        "TEST01", config=FilterConfig(**config), beacon_layout=LAYOUT.beacons
    )


# ── The decision ─────────────────────────────────────────────────────────────


def test_ble_is_off_by_default():
    """Measured on a 200 m contact-only run in a 6.0 x 4.5 m room:

        BLE off   mean error 0.51 m, worst 1.59 m
        BLE on    mean error 1.42 m, worst 2.78 m

    Odometry is about five times better than a BLE fix at this scale, so
    folding BLE in adds noise rather than removing drift. If this default is
    ever flipped, it should be because someone re-ran that measurement.
    """
    assert FilterConfig().ble_enabled is False


def test_beacons_are_ignored_while_disabled():
    pose_filter = _filter(ble_enabled=False)
    pose_filter.update(_packet(1))
    before = (pose_filter.x_m, pose_filter.y_m)

    for seq in range(2, 40):
        pose_filter.update(_packet(seq, beacons=_samples_at(4.0, 3.0)))

    assert (pose_filter.x_m, pose_filter.y_m) == before
    assert pose_filter.ble_acceptances == 0


# ── It works when asked ──────────────────────────────────────────────────────


def test_enabled_beacons_pull_a_drifted_pose_back():
    """The behaviour the feature exists for: an estimate that has wandered is
    dragged back towards where the beacons say the robot is."""
    pose_filter = _filter(ble_enabled=True)
    pose_filter.update(_packet(1))

    # Pretend the estimate has drifted badly, and widen its covariance to
    # match — a filter that does not know it is lost will not accept help.
    pose_filter.x_m, pose_filter.y_m = 5.0, 4.0
    pose_filter.covariance.var_x = pose_filter.covariance.var_y = 9.0

    for seq in range(2, 30):
        pose_filter.update(_packet(seq, beacons=_samples_at(1.5, 1.5)))

    assert pose_filter.x_m < 5.0
    assert pose_filter.y_m < 4.0
    assert pose_filter.ble_acceptances > 0


def test_a_confident_pose_is_barely_moved():
    """The gain is var_odom / (var_odom + var_ble). A fresh, certain pose must
    not be dragged around by a metre-scale fix."""
    pose_filter = _filter(ble_enabled=True)
    pose_filter.update(_packet(1))
    pose_filter.covariance.var_x = pose_filter.covariance.var_y = 0.0001

    start = (pose_filter.x_m, pose_filter.y_m)
    for seq in range(2, 20):
        pose_filter.update(_packet(seq, beacons=_samples_at(4.0, 3.0)))

    moved = abs(pose_filter.x_m - start[0]) + abs(pose_filter.y_m - start[1])
    assert moved < 0.2


def test_a_single_correction_is_clamped():
    """Even a formally valid correction must not teleport the robot across the
    room in one cycle."""
    pose_filter = _filter(ble_enabled=True, ble_max_correction_m=0.5)
    pose_filter.update(_packet(1))
    pose_filter.x_m, pose_filter.y_m = 5.5, 4.2
    pose_filter.covariance.var_x = pose_filter.covariance.var_y = 100.0

    before = (pose_filter.x_m, pose_filter.y_m)
    pose_filter.update(_packet(2, beacons=_samples_at(0.5, 0.5)))

    moved = (
        (pose_filter.x_m - before[0]) ** 2 + (pose_filter.y_m - before[1]) ** 2
    ) ** 0.5
    assert moved <= 0.5 + 1e-6


# ── Refusing bad fixes ───────────────────────────────────────────────────────


def test_too_few_beacons_is_not_a_fix():
    """Trilateration needs three; two leaves a pair of solutions and no way to
    choose between them."""
    pose_filter = _filter(ble_enabled=True)
    pose_filter.update(_packet(1))

    partial = _samples_at(2.0, 2.0)[:2]
    for seq in range(2, 20):
        pose_filter.update(_packet(seq, beacons=partial))

    assert pose_filter.ble_acceptances == 0


def test_no_beacon_layout_means_no_corrections():
    """A reading without a surveyed position is a distance to an unknown
    point, which is worth nothing."""
    pose_filter = PoseFilter("TEST01", config=FilterConfig(ble_enabled=True))
    pose_filter.update(_packet(1))

    for seq in range(2, 20):
        pose_filter.update(_packet(seq, beacons=_samples_at(2.0, 2.0)))

    assert pose_filter.ble_acceptances == 0


def test_a_wild_fix_is_rejected_as_an_outlier():
    """A fix far outside what the current uncertainty can explain is a bad
    solve, not a robot that teleported."""
    pose_filter = _filter(ble_enabled=True, ble_outlier_sigma=1.0)
    pose_filter.update(_packet(1))
    pose_filter.covariance.var_x = pose_filter.covariance.var_y = 0.01

    pose_filter.update(_packet(2, beacons=_samples_at(5.8, 4.3)))
    assert pose_filter.ble_rejections > 0


# ── Reporting ────────────────────────────────────────────────────────────────


def test_diagnostics_report_the_beacon_state():
    """Whether BLE is contributing has to be visible, or a silently disabled
    sensor looks identical to a working one."""
    pose_filter = _filter(ble_enabled=True)
    pose_filter.update(_packet(1, beacons=_samples_at(1.0, 1.0)))

    diagnostics = pose_filter.diagnostics()
    assert diagnostics["beacons_installed"] == 4
    assert "ble_acceptances" in diagnostics
    assert diagnostics["last_ble_reason"]


def test_packets_carry_beacon_samples():
    packet = _packet(1, beacons=_samples_at(2.0, 2.0))
    restored = SensorPacket.model_validate_json(packet.model_dump_json())

    assert len(restored.beacons) == 4
    assert restored.beacons[0].beacon_id


def test_nonsense_rssi_is_refused_at_the_contract():
    """+20 dBm is not a signal strength any receiver reports."""
    with pytest.raises(ValueError):
        BeaconSample(beacon_id="B1", rssi_dbm=20.0)
