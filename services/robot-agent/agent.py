"""Assemble sensor packets on the PC. Replaces the ESP32 firmware's job.

What changed and why
--------------------
The earlier design put an ESP32 on the robot: it read the encoders, the IMU
and the GPS, packed them into a `SensorPacket` and transmitted it. There is no
microcontroller on this robot, so that job moves to the PC:

    ESP32 firmware                     ->  this module
    -------------------------------        -----------------------------
    read quadrature encoders           ->  read servo bus positions
    read MPU6050 over I2C              ->  optional IMU on USB, or none
    read GPS over UART                 ->  GPS on USB (services/gps-reader)
    build JSON, publish over WiFi      ->  build SensorPacket, publish MQTT

The output schema is unchanged, so the mapper, the pose filter and the web
viewer all work exactly as they already do.

The one thing genuinely lost
----------------------------
The ESP32 had an IMU wired to it. This robot's sensing is whatever the servo
bus reports plus whatever else is plugged into the PC. Without an IMU, heading
comes only from the three wheel encoders — and omni wheels slip sideways by
design, which is precisely the motion that corrupts a heading estimate. The
agent reports `imu=None` honestly in that case rather than fabricating a
heading, and the pose filter widens its uncertainty accordingly.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "shared", ROOT / "services", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robotmap_common.holonomic import HolonomicGeometry
from robotmap_common.models import (
    DriveKind,
    EncoderData,
    GpsData,
    ImuData,
    LinkType,
    PowerData,
    RangeReading,
    SensorPacket,
)

logger = logging.getLogger("robot-agent")


@dataclass
class AgentConfig:
    robot_id: str = "MR3W01"
    telemetry_hz: float = 10.0
    # Link is recorded so the map viewer can show how the robot is connected.
    link: LinkType = LinkType.WIFI_MQTT


class RobotAgent:
    """Polls the robot's sensors and produces SensorPackets.

    Every source is optional and injected, so the agent runs with a servo bus
    alone, or with GPS added, or with nothing at all for testing.
    """

    def __init__(
        self,
        servo_driver=None,
        gps_reader=None,
        range_source=None,
        geometry: HolonomicGeometry | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.servo = servo_driver
        self.gps = gps_reader
        self.range_source = range_source
        self.geometry = geometry or HolonomicGeometry()
        self.config = config or AgentConfig()

        self.sequence = 0
        self._last_packet_at = time.monotonic()

        self.packets_built = 0
        self.gps_fixes_included = 0
        self.gps_fixes_stale = 0

    # ── Individual sources ────────────────────────────────────────────────

    def _read_wheel_ticks(self) -> tuple[int, int, int] | None:
        if self.servo is None:
            return None
        try:
            return self.servo.wheel_ticks()
        except Exception:
            logger.debug("Could not read wheel ticks", exc_info=True)
            return None

    def _read_gps(self) -> GpsData | None:
        """Return a fix only if it is present AND current.

        A stale fix reported as live is worse than none: the pose filter would
        treat a minute-old position as a fresh measurement. Since a receiver
        going quiet is exactly what happens when the robot drives indoors, the
        freshness check matters more here than it looks.
        """
        if self.gps is None:
            return None

        fix = self.gps.read()
        if fix is None:
            return None

        if not self.gps.is_fresh:
            self.gps_fixes_stale += 1
            return None

        self.gps_fixes_included += 1
        return fix

    def _read_ranges(self) -> list[RangeReading]:
        if self.range_source is None:
            return []
        try:
            return self.range_source()
        except Exception:
            logger.debug("Could not read ranges", exc_info=True)
            return []

    def _read_imu(self) -> ImuData | None:
        """No IMU on this platform by default.

        Returning None is deliberate. Synthesising a heading from the wheel
        encoders and presenting it as an IMU reading would tell the pose filter
        the heading had been independently confirmed when it had not, and the
        filter would then trust it far more than it should.
        """
        return None

    # ── Packet assembly ───────────────────────────────────────────────────

    def build_packet(self, dt_s: float | None = None) -> SensorPacket:
        now = time.monotonic()
        interval = dt_s if dt_s is not None else (now - self._last_packet_at)
        self._last_packet_at = now
        self.sequence += 1

        ticks = self._read_wheel_ticks()
        wheel_ticks = list(ticks) if ticks else None

        packet = SensorPacket(
            robot_id=self.config.robot_id,
            timestamp=datetime.now(UTC).isoformat(),
            sequence=self.sequence,
            link=self.config.link,
            drive=(
                DriveKind.HOLONOMIC_3WHEEL if wheel_ticks else DriveKind.DIFFERENTIAL
            ),
            # The two-wheel field is kept populated so older readers still
            # parse; `wheel_ticks` is what a holonomic consumer should use.
            encoders=EncoderData(
                left_ticks=ticks[0] if ticks else 0,
                right_ticks=ticks[1] if ticks else 0,
                dt_ms=max(1, min(10_000, int(interval * 1000))),
            ),
            wheel_ticks=wheel_ticks,
            imu=self._read_imu(),
            ranges=self._read_ranges(),
            gps=self._read_gps(),
            power=self._read_power(),
        )

        self.packets_built += 1
        return packet

    def _read_power(self) -> PowerData | None:
        """Battery voltage, if the servos report it.

        Bus servos usually expose supply voltage, which is a free battery
        gauge. Not every family does, so this stays optional.
        """
        if self.servo is None:
            return None

        readings = getattr(self.servo, "last_readings", None)
        if not readings:
            return None

        voltages = [r.voltage_v for r in readings if r.voltage_v is not None]
        if not voltages:
            return None

        voltage = sum(voltages) / len(voltages)
        # 2S lithium: 8.4 V full, 6.0 V empty.
        soc = max(0.0, min(100.0, (voltage - 6.0) / 2.4 * 100.0))
        return PowerData(battery_v=voltage, battery_soc=soc, current_a=0.0)

    def diagnostics(self) -> dict:
        return {
            "packets_built": self.packets_built,
            "gps_fixes_included": self.gps_fixes_included,
            "gps_fixes_stale": self.gps_fixes_stale,
            "has_servo_bus": self.servo is not None,
            "has_gps": self.gps is not None,
            "has_ranges": self.range_source is not None,
            "has_imu": False,
        }
