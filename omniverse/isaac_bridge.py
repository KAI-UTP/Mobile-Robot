"""Bridge Isaac Sim to the existing mapping stack.

Synthesises the same `SensorPacket` the real robot will publish — wheel
encoder counts, IMU heading, ultrasonic ranges — from Isaac's physics state,
and publishes it on the same MQTT topic. The mapper, localization filter and
web viewer therefore run unchanged against the simulated robot.

That shared path is the point. Anything demonstrated in Isaac Sim is
demonstrated through the code that will run on the real robot, so the only
untested difference when the servo bus arrives is the transport.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from robotmap_common.holonomic import BodyTwist
from robotmap_common.models import (
    EncoderData,
    ImuData,
    LinkType,
    PowerData,
    RangeReading,
    SensorPacket,
)
from robotmap_common.topics import Topics

logger = logging.getLogger("isaac-bridge")

# Where the ultrasonic sensors point, relative to the robot's forward axis.
SENSOR_ANGLES_DEG = (0.0, 90.0, -90.0)
SENSOR_MAX_RANGE_M = 4.0
SENSOR_HEIGHT_M = 0.08


class IsaacSensorBridge:
    """Turns Isaac physics state into telemetry, and drives wall-following."""

    def __init__(
        self,
        robot,
        controller,
        mqtt_host: str = "localhost",
        publish: bool = True,
        robot_id: str = "MR3W01",
        telemetry_interval_s: float = 0.1,
    ) -> None:
        self.robot = robot
        self.controller = controller
        self.robot_id = robot_id
        self.telemetry_interval_s = telemetry_interval_s
        self.publish = publish

        self.sequence = 0
        self._time_since_publish = 0.0
        self._elapsed = 0.0

        self.mqtt = None
        if publish:
            import os

            os.environ.setdefault("MQTT_HOST", mqtt_host)
            from robotmap_common.mqtt_client import build_client

            self.mqtt = build_client(client_id=f"isaac-bridge-{robot_id}")
            logger.info("Publishing to %s", Topics.SENSORS_RAW)

        # Wall-following runs on the same controller the real robot will use.
        from simulator.explorer import WallFollower

        self.follower = WallFollower()

        self._scene_query = self._acquire_scene_query()
        self._last_ranges: list[RangeReading] = []

    # ── Physics queries ───────────────────────────────────────────────────

    @staticmethod
    def _acquire_scene_query():
        """PhysX raycast interface, used to emulate the ultrasonic sensors."""
        try:
            from omni.physx import get_physx_scene_query_interface

            return get_physx_scene_query_interface()
        except ImportError:
            logger.warning(
                "omni.physx scene query unavailable; range sensors will report "
                "max range and no map will be built"
            )
            return None

    def _pose(self) -> tuple[float, float, float]:
        """Robot pose as (x, y, heading_deg) in the world frame."""
        position, orientation = self.robot.get_world_pose()
        w, x, y, z = (float(v) for v in orientation)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(position[0]), float(position[1]), math.degrees(yaw) % 360.0

    def _read_ranges(self) -> list[RangeReading]:
        """Cast one ray per sensor and return the distances.

        A raycast is a much better sensor than a real HC-SR04: no beam cone, no
        specular dropout. That flattering difference is called out in
        docs/OMNIVERSE.md, because a map that only works in simulation because
        the simulated sensors are unrealistically good is a trap.
        """
        if self._scene_query is None:
            return [
                RangeReading(angle_deg=a, distance_m=SENSOR_MAX_RANGE_M)
                for a in SENSOR_ANGLES_DEG
            ]

        x, y, heading_deg = self._pose()
        readings: list[RangeReading] = []

        for angle_deg in SENSOR_ANGLES_DEG:
            world_angle = math.radians(heading_deg + angle_deg)
            origin = (x, y, SENSOR_HEIGHT_M)
            direction = (math.cos(world_angle), math.sin(world_angle), 0.0)

            hit = self._scene_query.raycast_closest(
                origin, direction, SENSOR_MAX_RANGE_M
            )

            if hit and hit.get("hit"):
                distance = min(float(hit["distance"]), SENSOR_MAX_RANGE_M)
            else:
                # No hit means nothing within range, which the mapper reads as
                # free space rather than a wall at the range limit.
                distance = SENSOR_MAX_RANGE_M

            readings.append(
                RangeReading(
                    angle_deg=angle_deg,
                    distance_m=distance,
                    valid=True,
                    sensor_id=f"s{int(angle_deg)}",
                )
            )

        return readings

    # ── Telemetry ─────────────────────────────────────────────────────────

    def build_packet(self, dt_s: float) -> SensorPacket:
        ticks = self.controller.encoder_ticks()
        _, _, heading_deg = self._pose()
        state = self.controller.read_state()

        return SensorPacket(
            robot_id=self.robot_id,
            timestamp=datetime.now(UTC).isoformat(),
            sequence=self.sequence,
            link=LinkType.SIMULATED,
            # The differential schema carries two wheels; the third is sent in
            # `wheel_ticks` below. See models.py for why both exist.
            encoders=EncoderData(
                left_ticks=ticks[0],
                right_ticks=ticks[1],
                dt_ms=max(1, int(dt_s * 1000)),
            ),
            wheel_ticks=list(ticks),
            imu=ImuData(
                heading_deg=heading_deg,
                gyro_z_dps=state.measured.omega_dps,
                calibrated=True,
            ),
            ranges=self._last_ranges,
            power=PowerData(battery_v=7.4, battery_soc=100.0, current_a=0.5),
        )

    # ── Main step ─────────────────────────────────────────────────────────

    def step(self, dt_s: float) -> BodyTwist:
        """Advance one physics step. Returns the twist to command."""
        self._elapsed += dt_s
        self._time_since_publish += dt_s

        self._last_ranges = self._read_ranges()
        self.controller.read_state()

        x, y, _ = self._pose()

        if self._time_since_publish >= self.telemetry_interval_s:
            self.sequence += 1
            packet = self.build_packet(self._time_since_publish)
            if self.mqtt is not None:
                self.mqtt.publish(Topics.SENSORS_RAW, packet.model_dump_json())
            self._time_since_publish = 0.0

        command = self.follower.step(self._last_ranges, x, y, dt_s)

        if command.state.value == "FINISHED":
            logger.info(
                "Circuit complete after %.1f m", self.follower.lap_distance_m
            )
            return BodyTwist()

        # The wall follower was written for a differential robot: it emits a
        # forward speed and a turn rate, never a sideways component. Mapping
        # that onto a holonomic base drives it exactly like a differential one,
        # which keeps the mapping behaviour identical to the tested pipeline.
        #
        # Strafing would cover ground faster, but it would also mean the
        # forward-facing sensor no longer points where the robot is going, so
        # the explorer's assumptions would need revisiting first.
        return BodyTwist(
            vx_mps=command.linear_mps,
            vy_mps=0.0,
            omega_dps=command.angular_dps,
        )
