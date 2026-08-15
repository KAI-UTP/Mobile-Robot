"""Tests for the twin controller — one command driving both robots.

Runs with no hardware and no broker: the servo driver and the pose publisher
are injected, so both are substituted with stand-ins here.

The central property under test is that the twin tracks the REAL robot, not
the command. A twin that mirrors the command always looks perfect and is
therefore useless for measuring anything.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "twin-control"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "servo-bus"))

from robotmap_common.holonomic import (  # noqa: E402
    BodyTwist,
    HolonomicGeometry,
    inverse_kinematics,
    wrap_tick_delta_holonomic,
)
from twin import MirrorMode, TwinController  # noqa: E402

GEOM = HolonomicGeometry(
    wheel_radius_m=0.029,
    wheel_offset_m=0.100,
    wheel_angles_deg=(0.0, 120.0, 240.0),
    ticks_per_revolution=4096,
)
DT = 0.1


class FakeProtocol:
    counts_per_revolution = 4096


class FakeServoDriver:
    """A servo bus that optionally loses a fraction of every command to slip.

    `slip` is the share of commanded wheel motion that does NOT happen — the
    thing an encoder on the motor cannot see and the twin must therefore be
    told about some other way.
    """

    def __init__(self, geometry=GEOM, slip: float = 0.0):
        self.geometry = geometry
        self.protocol = FakeProtocol()
        self.slip = slip

        self._ticks = [0.0, 0.0, 0.0]
        self.last_twist = BodyTwist()
        self.stopped = False
        self.drive_calls = 0

    def drive(self, twist: BodyTwist):
        self.last_twist = twist
        self.drive_calls += 1
        speeds = inverse_kinematics(twist, self.geometry)
        for i, v in enumerate(speeds.values):
            travelled = v * DT * (1.0 - self.slip)
            self._ticks[i] += travelled / self.geometry.metres_per_tick
        return speeds

    def stop(self):
        self.stopped = True
        self.last_twist = BodyTwist()

    def wheel_ticks(self):
        span = self.protocol.counts_per_revolution
        return tuple(int(round(t)) % span for t in self._ticks)

    def measured_twist(self):
        return self.last_twist


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def __call__(self, topic, payload):
        self.messages.append((topic, payload))


# ── Counter wraparound ───────────────────────────────────────────────────────


def test_wrap_handles_non_power_of_two_span():
    """Bus servos wrap at 4096, which the power-of-two helper cannot handle."""
    assert wrap_tick_delta_holonomic(10, 4090, 4096) == 16
    assert wrap_tick_delta_holonomic(4090, 10, 4096) == -16


def test_wrap_normal_case():
    assert wrap_tick_delta_holonomic(500, 400, 4096) == 100


def test_wrap_rejects_invalid_span():
    with pytest.raises(ValueError):
        wrap_tick_delta_holonomic(1, 0, 0)


# ── Mode selection ───────────────────────────────────────────────────────────


def test_feedback_mode_without_hardware_falls_back():
    """Feedback needs encoders. Silently reporting the command as if measured
    would be the worst outcome, so the fallback is explicit."""
    twin = TwinController(servo_driver=None, geometry=GEOM, mode=MirrorMode.FEEDBACK)
    assert twin.mode == MirrorMode.COMMAND
    assert twin.state.hardware_connected is False


def test_feedback_mode_used_when_hardware_present():
    twin = TwinController(FakeServoDriver(), geometry=GEOM, mode=MirrorMode.FEEDBACK)
    assert twin.mode == MirrorMode.FEEDBACK
    assert twin.state.hardware_connected is True


# ── Command reaches the hardware ─────────────────────────────────────────────


def test_command_is_sent_to_the_servos():
    servo = FakeServoDriver()
    twin = TwinController(servo, geometry=GEOM)
    twin.step(BodyTwist(vx_mps=0.2), DT)

    assert servo.drive_calls == 1
    assert servo.last_twist.vx_mps == pytest.approx(0.2)


def test_stop_reaches_the_servos():
    servo = FakeServoDriver()
    twin = TwinController(servo, geometry=GEOM)
    twin.step(BodyTwist(vx_mps=0.2), DT)
    twin.stop()
    assert servo.stopped is True


# ── Poses ────────────────────────────────────────────────────────────────────


def test_both_poses_agree_when_nothing_slips():
    """With a perfect robot the twin and the ideal reference must coincide.

    This is the property the divergence measurement rests on: any error
    reported must come from the robot, not from the bookkeeping.
    """
    servo = FakeServoDriver(slip=0.0)
    twin = TwinController(servo, geometry=GEOM, mode=MirrorMode.FEEDBACK)

    for _ in range(50):
        state = twin.step(BodyTwist(vx_mps=0.2), DT)

    assert state.position_error_m < 0.01
    assert state.real_pose.x_m == pytest.approx(state.ideal_pose.x_m, abs=0.01)


def test_no_bias_accumulates_between_the_two_poses():
    """Guards a bug that was present in an earlier version.

    The ideal pose used to advance on the very first call while the real pose
    was still establishing its encoder baseline, leaving both permanently one
    interval apart. With zero slip that read as a steadily growing error that
    had nothing to do with the robot — quietly poisoning the one number this
    class exists to produce.

    Divergence must stay flat across a long run when nothing slips.
    """
    servo = FakeServoDriver(slip=0.0)
    twin = TwinController(servo, geometry=GEOM, mode=MirrorMode.FEEDBACK)

    for _ in range(200):
        twin.step(BodyTwist(vx_mps=0.2), DT)

    errors = [s.position_error_m for s in twin.state.divergence]
    early = max(errors[10:20])
    late = max(errors[-10:])

    assert late < 0.01
    # The late error must not exceed the early one by any meaningful margin.
    assert late <= early + 0.005


def test_both_poses_lag_the_command_by_exactly_one_interval():
    """The lag is intentional and symmetric — documented so it is not mistaken
    for an error later. Encoders cannot report a delta until two samples
    exist, so the first call only establishes a baseline."""
    servo = FakeServoDriver(slip=0.0)
    twin = TwinController(servo, geometry=GEOM, mode=MirrorMode.FEEDBACK)

    steps = 50
    for _ in range(steps):
        state = twin.step(BodyTwist(vx_mps=0.2), DT)

    # (steps - 1) intervals of motion, not `steps`.
    assert state.ideal_pose.x_m == pytest.approx(0.2 * (steps - 1) * DT, rel=0.01)


def test_slip_makes_the_real_pose_lag_the_ideal():
    """The measurement the whole module exists for.

    A robot losing 10 % of its motion to slip must end up demonstrably short
    of where the commands alone would put it.
    """
    servo = FakeServoDriver(slip=0.10)
    twin = TwinController(servo, geometry=GEOM, mode=MirrorMode.FEEDBACK)

    for _ in range(100):
        state = twin.step(BodyTwist(vx_mps=0.2), DT)

    # 100 steps x 0.1 s x 0.2 m/s = 2.0 m ideal; 10 % slip loses about 0.2 m.
    assert state.ideal_pose.x_m == pytest.approx(2.0, rel=0.02)
    assert state.real_pose.x_m == pytest.approx(1.8, rel=0.03)
    assert state.position_error_m == pytest.approx(0.2, rel=0.15)


def test_command_mirror_hides_slip():
    """Guards the design decision.

    In command-mirror mode the twin follows the command, so slip is invisible
    and the divergence reads as zero even though the real robot fell short.
    This is exactly why feedback is the default, and why the mode is recorded
    in the gap report.
    """
    servo = FakeServoDriver(slip=0.10)
    twin = TwinController(servo, geometry=GEOM, mode=MirrorMode.COMMAND)

    for _ in range(100):
        state = twin.step(BodyTwist(vx_mps=0.2), DT)

    assert state.position_error_m < 1e-9


def test_strafing_tracks_on_both_poses():
    servo = FakeServoDriver()
    twin = TwinController(servo, geometry=GEOM, mode=MirrorMode.FEEDBACK)

    steps = 50
    for _ in range(steps):
        state = twin.step(BodyTwist(vy_mps=0.2), DT)

    assert state.real_pose.y_m == pytest.approx(0.2 * (steps - 1) * DT, rel=0.05)
    assert state.real_pose.x_m == pytest.approx(0.0, abs=0.02)
    assert state.real_pose.heading_deg == pytest.approx(0.0, abs=1.0)


def test_rotation_tracks_heading():
    servo = FakeServoDriver()
    twin = TwinController(servo, geometry=GEOM, mode=MirrorMode.FEEDBACK)

    steps = 40
    for _ in range(steps):
        state = twin.step(BodyTwist(omega_dps=45.0), DT)

    assert state.real_pose.heading_deg == pytest.approx(
        45.0 * (steps - 1) * DT, abs=3.0
    )
    assert math.hypot(state.real_pose.x_m, state.real_pose.y_m) < 0.03


def test_encoder_wraparound_does_not_tear_the_pose():
    """Drive long enough that the 4096-count servo counters wrap repeatedly."""
    servo = FakeServoDriver()
    twin = TwinController(servo, geometry=GEOM, mode=MirrorMode.FEEDBACK)

    for _ in range(300):
        state = twin.step(BodyTwist(vx_mps=0.3), DT)

    # Confirm the counters really did wrap, so the test is meaningful.
    assert max(servo._ticks) > 4096

    expected = 0.3 * 300 * DT
    assert state.real_pose.x_m == pytest.approx(expected, rel=0.05)


# ── Publishing ───────────────────────────────────────────────────────────────


def test_pose_is_published_for_omniverse():
    publisher = RecordingPublisher()
    twin = TwinController(FakeServoDriver(), publisher, geometry=GEOM)

    for _ in range(5):
        twin.step(BodyTwist(vx_mps=0.2), DT)

    assert len(publisher.messages) == 5
    topic, payload = publisher.messages[0]
    assert "pose" in topic
    assert '"x_m"' in payload


def test_publish_failure_does_not_stop_the_robot():
    """A broker going down must not take the control loop with it."""

    def broken_publisher(topic, payload):
        raise ConnectionError("broker gone")

    servo = FakeServoDriver()
    twin = TwinController(servo, broken_publisher, geometry=GEOM)

    for _ in range(10):
        twin.step(BodyTwist(vx_mps=0.2), DT)

    assert servo.drive_calls == 10


def test_works_with_no_publisher_at_all():
    twin = TwinController(FakeServoDriver(), None, geometry=GEOM)
    # Two steps: the first only establishes the encoder baseline.
    twin.step(BodyTwist(vx_mps=0.2), DT)
    state = twin.step(BodyTwist(vx_mps=0.2), DT)
    assert state.real_pose.x_m > 0


# ── Reporting ────────────────────────────────────────────────────────────────


def test_gap_report_is_empty_before_any_motion():
    twin = TwinController(FakeServoDriver(), geometry=GEOM)
    assert twin.gap_report()["samples"] == 0


def test_gap_report_quantifies_the_drift():
    servo = FakeServoDriver(slip=0.08)
    twin = TwinController(servo, geometry=GEOM, mode=MirrorMode.FEEDBACK)

    for _ in range(100):
        twin.step(BodyTwist(vx_mps=0.2), DT)

    report = twin.gap_report()
    assert report["samples"] == 100
    assert report["final_position_error_m"] > 0.1
    assert report["mode"] == "feedback"
    assert report["hardware_connected"] is True
    # Error normalised by distance, so runs of different lengths compare.
    assert report["error_per_metre"] == pytest.approx(0.08, rel=0.25)


def test_divergence_history_is_bounded():
    """A long run must not grow memory without limit."""
    twin = TwinController(FakeServoDriver(), geometry=GEOM)
    for _ in range(25_000):
        twin.step(BodyTwist(vx_mps=0.1), DT)
    assert len(twin.state.divergence) <= 20_000


def test_reset_clears_both_poses():
    servo = FakeServoDriver()
    twin = TwinController(servo, geometry=GEOM, mode=MirrorMode.FEEDBACK)
    for _ in range(20):
        twin.step(BodyTwist(vx_mps=0.2), DT)
    assert twin.state.real_pose.x_m > 0

    twin.reset()
    assert twin.state.real_pose.x_m == 0.0
    assert twin.state.ideal_pose.x_m == 0.0
    assert twin.gap_report()["samples"] == 0


def test_divergence_csv_is_written(tmp_path):
    servo = FakeServoDriver(slip=0.05)
    twin = TwinController(servo, geometry=GEOM, mode=MirrorMode.FEEDBACK)
    for _ in range(20):
        twin.step(BodyTwist(vx_mps=0.2), DT)

    out = twin.export_divergence_csv(tmp_path / "gap.csv")
    lines = out.read_text(encoding="utf-8").splitlines()

    assert lines[0].startswith("t_s,position_error_m")
    assert len(lines) == 21  # header plus one row per step
