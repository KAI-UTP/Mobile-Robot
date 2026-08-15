"""Robot agent tests: PC-side sensor packet assembly.

The agent replaces what the ESP32 firmware used to do. Every source is
injected, so all of this runs with no servo bus, no GPS and no broker.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "robot-agent"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "gps-reader"))

from agent import AgentConfig, RobotAgent  # noqa: E402
from robotmap_common.models import (  # noqa: E402
    DriveKind,
    GpsData,
    GpsFixQuality,
    RangeReading,
)


class FakeServo:
    def __init__(self, ticks=(100, 200, 300), fail=False):
        self._ticks = ticks
        self.fail = fail
        self.last_readings = []

    def wheel_ticks(self):
        if self.fail:
            raise OSError("bus disconnected")
        return self._ticks


class FakeGps:
    def __init__(self, fix=None, fresh=True):
        self._fix = fix
        self._fresh = fresh

    def read(self):
        return self._fix

    @property
    def is_fresh(self):
        return self._fresh


def _outdoor_fix():
    return GpsData(
        latitude=4.3852,
        longitude=100.9739,
        fix_quality=GpsFixQuality.GPS,
        satellites=9,
        hdop=0.9,
    )


def _indoor_fix():
    return GpsData(
        latitude=4.3852,
        longitude=100.9739,
        fix_quality=GpsFixQuality.GPS,
        satellites=3,
        hdop=12.5,
    )


# ── Wheel data ───────────────────────────────────────────────────────────────


def test_three_wheel_ticks_are_carried():
    agent = RobotAgent(servo_driver=FakeServo((100, 200, 300)))
    packet = agent.build_packet(dt_s=0.1)

    assert packet.wheel_ticks == [100, 200, 300]
    assert packet.drive == DriveKind.HOLONOMIC_3WHEEL
    assert packet.is_holonomic is True


def test_two_wheel_field_stays_populated_for_older_readers():
    agent = RobotAgent(servo_driver=FakeServo((100, 200, 300)))
    packet = agent.build_packet(dt_s=0.1)
    assert packet.encoders.left_ticks == 100
    assert packet.encoders.right_ticks == 200


def test_missing_servo_bus_is_not_reported_as_holonomic():
    """With no wheel data there is no third wheel to report, and claiming
    otherwise would send the pose filter down the holonomic path with nothing
    to integrate."""
    agent = RobotAgent(servo_driver=None)
    packet = agent.build_packet(dt_s=0.1)

    assert packet.wheel_ticks is None
    assert packet.is_holonomic is False


def test_servo_bus_failure_does_not_crash_the_agent():
    """A disconnected bus must degrade, not take the process down."""
    agent = RobotAgent(servo_driver=FakeServo(fail=True))
    packet = agent.build_packet(dt_s=0.1)
    assert packet.wheel_ticks is None


# ── GPS ──────────────────────────────────────────────────────────────────────


def test_fresh_fix_is_included():
    agent = RobotAgent(servo_driver=FakeServo(), gps_reader=FakeGps(_outdoor_fix()))
    packet = agent.build_packet(dt_s=0.1)

    assert packet.gps is not None
    assert packet.gps.satellites == 9
    assert agent.gps_fixes_included == 1


def test_stale_fix_is_dropped():
    """A minute-old position reported as current would be treated by the pose
    filter as a live measurement. Since going quiet is exactly what a receiver
    does when the robot drives indoors, this case is the normal one."""
    agent = RobotAgent(
        servo_driver=FakeServo(), gps_reader=FakeGps(_outdoor_fix(), fresh=False)
    )
    packet = agent.build_packet(dt_s=0.1)

    assert packet.gps is None
    assert agent.gps_fixes_stale == 1
    assert agent.gps_fixes_included == 0


def test_indoor_fix_is_still_forwarded_for_the_filter_to_reject():
    """The agent does not second-guess quality — it forwards the fix with its
    quality fields intact and lets the filter's gate decide. Two places making
    the same judgement would eventually disagree."""
    agent = RobotAgent(servo_driver=FakeServo(), gps_reader=FakeGps(_indoor_fix()))
    packet = agent.build_packet(dt_s=0.1)

    assert packet.gps is not None
    assert packet.gps.is_usable_for_position is False


def test_no_gps_reader_is_fine():
    agent = RobotAgent(servo_driver=FakeServo(), gps_reader=None)
    assert agent.build_packet(dt_s=0.1).gps is None


def test_gps_reader_with_no_fix_yet():
    agent = RobotAgent(servo_driver=FakeServo(), gps_reader=FakeGps(None))
    assert agent.build_packet(dt_s=0.1).gps is None


# ── IMU ──────────────────────────────────────────────────────────────────────


def test_no_imu_is_reported_honestly():
    """The ESP32 had an IMU; this platform does not.

    Synthesising a heading from the wheels and labelling it an IMU reading
    would tell the pose filter the heading had been independently confirmed
    when it had not, and the filter weights IMU headings heavily.
    """
    agent = RobotAgent(servo_driver=FakeServo())
    packet = agent.build_packet(dt_s=0.1)

    assert packet.imu is None
    assert agent.diagnostics()["has_imu"] is False


# ── Ranges ───────────────────────────────────────────────────────────────────


def test_range_readings_are_included():
    def ranges():
        return [RangeReading(angle_deg=0.0, distance_m=1.5)]

    agent = RobotAgent(servo_driver=FakeServo(), range_source=ranges)
    packet = agent.build_packet(dt_s=0.1)

    assert len(packet.ranges) == 1
    assert packet.ranges[0].distance_m == pytest.approx(1.5)


def test_range_source_failure_is_survivable():
    def broken():
        raise RuntimeError("sensor unplugged")

    agent = RobotAgent(servo_driver=FakeServo(), range_source=broken)
    assert agent.build_packet(dt_s=0.1).ranges == []


def test_no_ranges_yields_an_empty_list_not_none():
    """The schema expects a list; None would fail validation downstream."""
    agent = RobotAgent(servo_driver=FakeServo())
    assert agent.build_packet(dt_s=0.1).ranges == []


# ── Packet mechanics ─────────────────────────────────────────────────────────


def test_sequence_increments():
    agent = RobotAgent(servo_driver=FakeServo())
    first = agent.build_packet(dt_s=0.1)
    second = agent.build_packet(dt_s=0.1)
    assert second.sequence == first.sequence + 1


def test_interval_is_measured_when_not_supplied():
    agent = RobotAgent(servo_driver=FakeServo())
    agent.build_packet()
    time.sleep(0.05)
    packet = agent.build_packet()
    assert packet.encoders.dt_ms >= 40


def test_interval_is_clamped_into_the_valid_range():
    """A long stall between packets must not produce a dt the schema rejects
    and take the whole stream down."""
    agent = RobotAgent(servo_driver=FakeServo())
    packet = agent.build_packet(dt_s=9999.0)
    assert 1 <= packet.encoders.dt_ms <= 10_000


def test_zero_interval_is_clamped_to_one_ms():
    agent = RobotAgent(servo_driver=FakeServo())
    packet = agent.build_packet(dt_s=0.0)
    assert packet.encoders.dt_ms >= 1


def test_packet_round_trips_through_json():
    """It has to survive MQTT to be worth anything."""
    from robotmap_common.models import SensorPacket

    agent = RobotAgent(servo_driver=FakeServo(), gps_reader=FakeGps(_outdoor_fix()))
    packet = agent.build_packet(dt_s=0.1)

    restored = SensorPacket.model_validate_json(packet.model_dump_json())
    assert restored.wheel_ticks == packet.wheel_ticks
    assert restored.gps.satellites == 9
    assert restored.is_holonomic is True


def test_agent_works_with_no_sources_at_all():
    """Used by the smoke test, and keeps the agent testable in isolation."""
    agent = RobotAgent()
    packet = agent.build_packet(dt_s=0.1)
    assert packet.robot_id == "MR3W01"
    assert packet.wheel_ticks is None


def test_robot_id_is_configurable():
    agent = RobotAgent(config=AgentConfig(robot_id="MR3W02"))
    assert agent.build_packet(dt_s=0.1).robot_id == "MR3W02"


def test_diagnostics_report_what_is_attached():
    agent = RobotAgent(servo_driver=FakeServo(), gps_reader=FakeGps(_outdoor_fix()))
    agent.build_packet(dt_s=0.1)

    diagnostics = agent.diagnostics()
    assert diagnostics["has_servo_bus"] is True
    assert diagnostics["has_gps"] is True
    assert diagnostics["has_ranges"] is False
    assert diagnostics["packets_built"] == 1
