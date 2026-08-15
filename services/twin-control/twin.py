"""One command, two robots: the real one and its twin in Omniverse.

This is the shape Shebaro described — type an instruction, the physical robot
moves and the Omniverse robot moves with it.

Two ways to mirror, and the difference matters
-----------------------------------------------
**Command mirror** sends the same twist to both. Omniverse then shows what the
robot was *told* to do. It is simple and it always looks perfect, which is
precisely the problem: the simulated robot never slips, so after a minute of
driving the two have quietly diverged and the display is confidently wrong.

**Feedback mirror** sends the twist only to the real robot, reads its encoders
back, and moves the Omniverse robot to where the real one actually *is*. The
display can now look untidy — wheels slipping, the robot lagging a command —
and that untidiness is the honest part. It is also the only mode in which the
difference between the two is measurable, which is what makes it a digital
twin rather than an animation.

Feedback is the default. Command mirroring is kept because it is genuinely
useful when the robot is on a bench with its wheels off the ground, or when
demonstrating with no hardware present at all.

The divergence between the two is recorded either way, and is the number the
sim-to-real gap analysis is built on.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "shared", ROOT / "services", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robotmap_common.holonomic import (
    BodyTwist,
    HolonomicGeometry,
    integrate_twist,
    odometry_from_ticks,
    wrap_tick_delta_holonomic,
)
from robotmap_common.models import PoseEstimate, PoseSource
from robotmap_common.topics import Topics

logger = logging.getLogger("twin-control")


class MirrorMode(str, Enum):
    FEEDBACK = "feedback"  # Omniverse follows the real robot's encoders
    COMMAND = "command"    # Omniverse follows the command (no hardware needed)


@dataclass
class TwinPose:
    """A pose in the map frame, plus where it came from."""

    x_m: float = 0.0
    y_m: float = 0.0
    heading_deg: float = 0.0
    distance_m: float = 0.0

    def copy(self) -> TwinPose:
        return TwinPose(self.x_m, self.y_m, self.heading_deg, self.distance_m)


@dataclass
class DivergenceSample:
    """One measurement of how far the twin has drifted from the real robot."""

    t_s: float
    position_error_m: float
    heading_error_deg: float
    commanded: BodyTwist
    measured: BodyTwist


@dataclass
class TwinState:
    """Everything the operator and the report need."""

    commanded_twist: BodyTwist = field(default_factory=BodyTwist)
    measured_twist: BodyTwist = field(default_factory=BodyTwist)
    # Where the real robot is, from its own encoders.
    real_pose: TwinPose = field(default_factory=TwinPose)
    # Where the robot would be if nothing ever slipped.
    ideal_pose: TwinPose = field(default_factory=TwinPose)
    divergence: list[DivergenceSample] = field(default_factory=list)
    hardware_connected: bool = False

    @property
    def position_error_m(self) -> float:
        return math.hypot(
            self.real_pose.x_m - self.ideal_pose.x_m,
            self.real_pose.y_m - self.ideal_pose.y_m,
        )

    @property
    def heading_error_deg(self) -> float:
        diff = (self.real_pose.heading_deg - self.ideal_pose.heading_deg + 180.0) % 360.0 - 180.0
        return diff


def make_file_publisher(robot_id: str = "MR3W01"):
    """A publisher that writes the pose to a file instead of MQTT.

    Some Omniverse Kit builds sandbox pip and cannot install paho-mqtt. Rather
    than leave the twin unusable there, the follower script can read the pose
    from a file. Crude, but it works everywhere and needs nothing installed.

    The write is atomic — to a temporary file, then renamed — because the
    follower polls this path and would otherwise occasionally read a
    half-written file and log a parse error every second.
    """
    path = os.path.join(tempfile.gettempdir(), f"roommapper_{robot_id}_pose.json")
    logger.info("Publishing pose to %s", path)

    def publish(_topic: str, payload: str) -> None:
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temp_path, path)

    return publish


def make_mqtt_publisher(host: str = "localhost", robot_id: str = "MR3W01"):
    """A publisher backed by MQTT."""
    os.environ.setdefault("MQTT_HOST", host)
    from robotmap_common.mqtt_client import build_client

    client = build_client(client_id=f"twin-control-{robot_id}")

    def publish(topic: str, payload: str) -> None:
        client.publish(topic, payload)

    return publish


class TwinController:
    """Drives the real robot and keeps the Omniverse twin in step.

    The servo driver and the pose publisher are injected rather than
    constructed here, so this runs with no hardware and no broker — which is
    how it is tested.
    """

    def __init__(
        self,
        servo_driver=None,
        pose_publisher=None,
        geometry: HolonomicGeometry | None = None,
        mode: MirrorMode = MirrorMode.FEEDBACK,
        robot_id: str = "MR3W01",
    ) -> None:
        self.servo = servo_driver
        self.publisher = pose_publisher
        self.geometry = geometry or HolonomicGeometry()
        self.geometry.validate()
        self.mode = mode
        self.robot_id = robot_id

        self.state = TwinState(hardware_connected=servo_driver is not None)
        self.sequence = 0
        self._prev_ticks: tuple[int, int, int] | None = None
        # The command in force during the interval about to be integrated.
        self._prev_command: BodyTwist | None = None
        self._start_time = time.monotonic()
        self._lock = threading.Lock()

        if self.mode == MirrorMode.FEEDBACK and servo_driver is None:
            logger.warning(
                "Feedback mode needs hardware; falling back to command mirror. "
                "The twin will show the commanded motion, not the real one."
            )
            self.mode = MirrorMode.COMMAND

    # ── The main step ─────────────────────────────────────────────────────

    def step(self, twist: BodyTwist, dt_s: float) -> TwinState:
        """Apply one command and advance both robots by dt_s.

        Both poses are advanced using the interval that has just *elapsed*,
        not the command being issued now. That ordering matters more than it
        looks: encoders can only report a delta once two samples exist, so the
        real pose is inherently one interval behind. Integrating the incoming
        command immediately would run the ideal pose one step ahead of it and
        bake a constant offset into every divergence figure — which is the one
        number this class exists to measure.

        The cost is that both poses lag the command by a single interval. That
        is harmless and symmetric; a bias between them would not be.
        """
        with self._lock:
            self.state.commanded_twist = twist

            if self._prev_command is None:
                # First call. Capture the encoder baseline; no interval has
                # elapsed yet, so neither pose moves.
                if self.servo is not None:
                    try:
                        self._prev_ticks = self.servo.wheel_ticks()
                    except Exception:
                        logger.debug("Could not read initial ticks", exc_info=True)
            else:
                ideal_delta = integrate_twist(
                    self._prev_command, self.state.ideal_pose.heading_deg, dt_s
                )
                self._apply(self.state.ideal_pose, ideal_delta)

                if self.mode == MirrorMode.FEEDBACK and self.servo is not None:
                    self._advance_from_encoders(dt_s)
                else:
                    # No hardware to read: the real pose is the ideal pose.
                    self.state.real_pose = self.state.ideal_pose.copy()
                    self.state.measured_twist = self._prev_command

            if self.servo is not None:
                self.servo.drive(twist)
            self._prev_command = twist

            self._record_divergence()
            self._publish()

            return self.state

    def _advance_from_encoders(self, dt_s: float) -> None:
        """Move the real pose according to what the wheels actually did."""
        ticks = self.servo.wheel_ticks()

        if self._prev_ticks is None:
            self._prev_ticks = ticks
            return

        # Bus servos report absolute position within one turn, so the counter
        # wraps at counts_per_revolution rather than at a power of two.
        span = self.servo.protocol.counts_per_revolution
        deltas = tuple(
            wrap_tick_delta_holonomic(ticks[i], self._prev_ticks[i], span)
            for i in range(3)
        )
        self._prev_ticks = ticks

        delta = odometry_from_ticks(
            deltas, self.state.real_pose.heading_deg, dt_s, self.geometry
        )
        self._apply(self.state.real_pose, delta)

        try:
            self.state.measured_twist = self.servo.measured_twist()
        except Exception:
            logger.debug("Could not read measured twist", exc_info=True)

    @staticmethod
    def _apply(pose: TwinPose, delta) -> None:
        pose.x_m += delta.delta_x_m
        pose.y_m += delta.delta_y_m
        pose.heading_deg = (pose.heading_deg + delta.delta_heading_deg) % 360.0
        pose.distance_m += delta.distance_m

    def _record_divergence(self) -> None:
        self.state.divergence.append(
            DivergenceSample(
                t_s=time.monotonic() - self._start_time,
                position_error_m=self.state.position_error_m,
                heading_error_deg=self.state.heading_error_deg,
                commanded=self.state.commanded_twist,
                measured=self.state.measured_twist,
            )
        )
        # Bounded: a long run must not grow without limit.
        if len(self.state.divergence) > 20_000:
            del self.state.divergence[:10_000]

    # ── Publishing to Omniverse ───────────────────────────────────────────

    def _publish(self) -> None:
        if self.publisher is None:
            return

        pose = PoseEstimate(
            robot_id=self.robot_id,
            timestamp=datetime.now(UTC).isoformat(),
            sequence=self.sequence,
            x_m=self.state.real_pose.x_m,
            y_m=self.state.real_pose.y_m,
            heading_deg=self.state.real_pose.heading_deg % 360.0,
            linear_velocity_mps=max(-3.0, min(3.0, math.hypot(
                self.state.measured_twist.vx_mps, self.state.measured_twist.vy_mps
            ))),
            angular_velocity_dps=max(
                -360.0, min(360.0, self.state.measured_twist.omega_dps)
            ),
            distance_travelled_m=self.state.real_pose.distance_m,
            source=(
                PoseSource.ODOMETRY_ONLY
                if self.mode == MirrorMode.FEEDBACK
                else PoseSource.DEAD_RECKONING_DEGRADED
            ),
        )
        self.sequence += 1

        # The ideal pose rides along so the Omniverse follower can draw a
        # ghost robot where slip-free execution would have put it. Seeing the
        # two separate is far more legible than reading a divergence number.
        payload = pose.model_dump()
        payload.update(
            {
                "ideal_x_m": self.state.ideal_pose.x_m,
                "ideal_y_m": self.state.ideal_pose.y_m,
                "ideal_heading_deg": self.state.ideal_pose.heading_deg % 360.0,
                "position_error_m": self.state.position_error_m,
                "heading_error_deg": self.state.heading_error_deg,
                "mirror_mode": self.mode.value,
            }
        )

        try:
            self.publisher(Topics.POSE, json.dumps(payload))
        except Exception:
            logger.debug("Pose publish failed", exc_info=True)

    # ── Control ───────────────────────────────────────────────────────────

    def stop(self) -> None:
        if self.servo is not None:
            self.servo.stop()
        self.step(BodyTwist(), 0.0)

    def reset(self) -> None:
        with self._lock:
            self.state.real_pose = TwinPose()
            self.state.ideal_pose = TwinPose()
            self.state.divergence.clear()
            self._prev_ticks = None
            self._prev_command = None
            self._start_time = time.monotonic()

    # ── Reporting ─────────────────────────────────────────────────────────

    def gap_report(self) -> dict:
        """Summarise the sim-to-real gap over the run so far.

        These are the numbers the research claim rests on: how far the real
        robot ended up from where perfect, slip-free execution of the same
        commands would have put it.
        """
        samples = self.state.divergence
        if not samples:
            return {"samples": 0}

        position_errors = [s.position_error_m for s in samples]
        heading_errors = [abs(s.heading_error_deg) for s in samples]

        return {
            "samples": len(samples),
            "duration_s": round(samples[-1].t_s, 2),
            "distance_travelled_m": round(self.state.real_pose.distance_m, 3),
            "final_position_error_m": round(position_errors[-1], 4),
            "max_position_error_m": round(max(position_errors), 4),
            "final_heading_error_deg": round(samples[-1].heading_error_deg, 2),
            "max_heading_error_deg": round(max(heading_errors), 2),
            # Error as a share of distance driven — comparable across runs of
            # different lengths, unlike the raw figure.
            "error_per_metre": (
                round(position_errors[-1] / self.state.real_pose.distance_m, 4)
                if self.state.real_pose.distance_m > 0.01
                else None
            ),
            "mode": self.mode.value,
            "hardware_connected": self.state.hardware_connected,
        }

    def export_divergence_csv(self, path: str | Path) -> Path:
        """Write the divergence trace for plotting in the report."""
        path = Path(path)
        lines = ["t_s,position_error_m,heading_error_deg,cmd_vx,cmd_vy,cmd_omega,meas_vx,meas_vy,meas_omega"]
        for s in self.state.divergence:
            lines.append(
                f"{s.t_s:.3f},{s.position_error_m:.5f},{s.heading_error_deg:.3f},"
                f"{s.commanded.vx_mps:.4f},{s.commanded.vy_mps:.4f},{s.commanded.omega_dps:.3f},"
                f"{s.measured.vx_mps:.4f},{s.measured.vy_mps:.4f},{s.measured.omega_dps:.3f}"
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote %d divergence samples to %s", len(self.state.divergence), path)
        return path
