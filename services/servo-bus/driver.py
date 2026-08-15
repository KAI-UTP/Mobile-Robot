"""Drive the three wheel servos over the USB serial bus.

Takes the same `BodyTwist` the Isaac Sim controller takes, runs the same
`inverse_kinematics`, and writes velocities to the daisy chain. The maths that
moves the robot in simulation is literally the maths that moves it on the
bench — no second implementation to drift out of step.

Safety
------
A mobile robot that keeps its last command when the program dies will drive
into a wall. The driver is a context manager and stops the wheels on exit,
including on exception, and a watchdog stops them if no command arrives.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "shared", ROOT / "services", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import serial
from protocols import ServoProtocol, ServoReading, get_protocol
from robotmap_common.holonomic import (
    BodyTwist,
    DriveLimits,
    HolonomicGeometry,
    WheelSpeeds,
    forward_kinematics,
    inverse_kinematics,
    scale_to_limit,
)

logger = logging.getLogger("servo-bus")


@dataclass
class BusConfig:
    port: str
    baud: int = 1_000_000
    # Confirmed hardware: Feetech STS3215, 12 V. Note this is NOT the same as
    # "feetech", which is the older SCS series with different wheel-mode
    # handling and different speed units.
    protocol: str = "sts3215"
    # Servo IDs in the SAME ORDER as HolonomicGeometry.wheel_angles_deg.
    # Getting this wrong makes the robot drive off at the wrong angle while
    # looking entirely healthy — run `calibrate.py --check-order` after wiring.
    servo_ids: tuple[int, int, int] = (1, 2, 3)
    # Some wheels are mounted mirrored, so positive command means backwards.
    invert: tuple[bool, bool, bool] = (False, False, False)
    timeout_s: float = 0.05
    watchdog_s: float = 0.5


class ServoBusDriver:
    """Holonomic drive over a daisy-chained servo bus."""

    def __init__(
        self,
        config: BusConfig,
        geometry: HolonomicGeometry | None = None,
        limits: DriveLimits | None = None,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.geometry = geometry or HolonomicGeometry()
        self.geometry.validate()
        self.limits = limits or DriveLimits()
        self.dry_run = dry_run

        self.protocol: ServoProtocol = get_protocol(config.protocol)
        self.serial: serial.Serial | None = None

        self._lock = threading.Lock()
        self._last_command_at = 0.0
        self._watchdog: threading.Thread | None = None
        self._running = False

        self.last_readings: list[ServoReading] = []
        self.commands_sent = 0
        self.replies_received = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def __enter__(self) -> ServoBusDriver:
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def open(self) -> None:
        if self.dry_run:
            logger.info("Dry run: no serial port opened")
        else:
            self.serial = serial.Serial(
                self.config.port,
                self.config.baud,
                timeout=self.config.timeout_s,
                write_timeout=self.config.timeout_s,
            )
            logger.info(
                "Opened %s at %d baud (%s)",
                self.config.port,
                self.config.baud,
                self.protocol.name,
            )

        self.configure_wheel_mode()

        self._running = True
        self._last_command_at = time.monotonic()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="servo-watchdog"
        )
        self._watchdog.start()

    def close(self) -> None:
        """Stop the wheels, then release the port. Order matters."""
        self._running = False
        try:
            self.stop()
        except Exception:
            logger.exception("Failed to stop wheels during shutdown")

        if self._watchdog is not None:
            self._watchdog.join(timeout=1.0)

        if self.serial is not None:
            try:
                self.serial.close()
            finally:
                self.serial = None
        logger.info("Servo bus closed")

    # ── Low level ─────────────────────────────────────────────────────────

    def _write(self, packet: bytes) -> None:
        if self.dry_run or self.serial is None:
            logger.debug("dry-run tx: %s", packet.hex(" "))
            return
        self.serial.write(packet)
        self.commands_sent += 1

    def _write_read(self, packet: bytes, reply_bytes: int) -> bytes:
        if self.dry_run or self.serial is None:
            return b""
        # Clear stale bytes first: a half-duplex bus echoes what was just sent,
        # and leftovers from a previous exchange would be parsed as this reply.
        self.serial.reset_input_buffer()
        self.serial.write(packet)
        self.commands_sent += 1
        data = self.serial.read(reply_bytes)
        if data:
            self.replies_received += 1
        return data

    # ── Configuration ─────────────────────────────────────────────────────

    def configure_wheel_mode(self) -> None:
        """Put every wheel servo into continuous rotation.

        Skipping this is the most common reason a freshly wired wheel twitches
        and stops: a servo configured for an arm joint refuses to turn past its
        angle limits.
        """
        for servo_id in self.config.servo_ids:
            for packet in self.protocol.set_wheel_mode(servo_id):
                self._write(packet)
                time.sleep(0.01)
        logger.info("Servos %s set to wheel mode", list(self.config.servo_ids))

    # ── Driving ───────────────────────────────────────────────────────────

    def drive(self, twist: BodyTwist) -> WheelSpeeds:
        """Command a body twist. Returns the wheel speeds actually sent."""
        clamped = self._clamp(twist)
        speeds = inverse_kinematics(clamped, self.geometry)
        speeds, _ = scale_to_limit(speeds, self.limits.max_wheel_mps)

        units_per_rad_s = self.protocol.velocity_units_per_rad_s()

        with self._lock:
            for index, servo_id in enumerate(self.config.servo_ids):
                rim_speed = speeds.values[index]
                if self.config.invert[index]:
                    rim_speed = -rim_speed

                angular = rim_speed / self.geometry.wheel_radius_m
                raw = int(round(angular * units_per_rad_s))
                self._write(self.protocol.set_velocity(servo_id, raw))

            self._last_command_at = time.monotonic()

        return speeds

    def stop(self) -> None:
        with self._lock:
            for servo_id in self.config.servo_ids:
                self._write(self.protocol.set_velocity(servo_id, 0))

    def _clamp(self, twist: BodyTwist) -> BodyTwist:
        """Clamp translation by magnitude so diagonals keep their direction."""
        import math

        speed = math.hypot(twist.vx_mps, twist.vy_mps)
        vx, vy = twist.vx_mps, twist.vy_mps
        if speed > self.limits.max_linear_mps and speed > 0:
            factor = self.limits.max_linear_mps / speed
            vx *= factor
            vy *= factor

        omega = max(
            -self.limits.max_angular_dps,
            min(self.limits.max_angular_dps, twist.omega_dps),
        )
        return BodyTwist(vx_mps=vx, vy_mps=vy, omega_dps=omega)

    # ── Feedback ──────────────────────────────────────────────────────────

    def read_wheels(self) -> list[ServoReading]:
        """Poll each servo for position and speed.

        Bus servos carry a 12-bit absolute magnetic encoder, so this replaces
        the separate encoders the differential design needed — at far better
        resolution than the 360 counts/rev that design required.
        """
        readings: list[ServoReading] = []
        with self._lock:
            for servo_id in self.config.servo_ids:
                data = self._write_read(
                    self.protocol.read_state(servo_id),
                    self.protocol.expected_reply_length(),
                )
                reading = self.protocol.parse_state_reply(data, servo_id)
                readings.append(reading or ServoReading(servo_id=servo_id))

        self.last_readings = readings
        return readings

    def measured_twist(self) -> BodyTwist:
        """Recover the body twist from servo speed feedback."""
        readings = self.read_wheels()
        units_per_rad_s = self.protocol.velocity_units_per_rad_s()

        rim_speeds = []
        for index, reading in enumerate(readings):
            if reading.speed_raw is None:
                rim_speeds.append(0.0)
                continue
            angular = reading.speed_raw / units_per_rad_s
            rim = angular * self.geometry.wheel_radius_m
            if self.config.invert[index]:
                rim = -rim
            rim_speeds.append(rim)

        return forward_kinematics(WheelSpeeds(tuple(rim_speeds)), self.geometry)  # type: ignore[arg-type]

    def wheel_ticks(self) -> tuple[int, int, int]:
        """Raw encoder positions, for the SensorPacket.

        Note these WRAP at `counts_per_revolution` — they are absolute within
        one turn, not cumulative. `wrap_tick_delta` in the pose filter handles
        that, which is exactly why it exists.
        """
        readings = self.last_readings or self.read_wheels()
        return tuple(  # type: ignore[return-value]
            r.position_raw if r.position_raw is not None else 0 for r in readings
        )

    # ── Watchdog ──────────────────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        """Stop the wheels if commands stop arriving.

        Without this, a crashed or blocked control loop leaves the robot
        driving at its last commanded speed until it hits something.
        """
        while self._running:
            time.sleep(0.05)
            if time.monotonic() - self._last_command_at > self.config.watchdog_s:
                try:
                    self.stop()
                except Exception:
                    pass
                # Reset so the stop is sent once per lapse, not continuously.
                self._last_command_at = time.monotonic()
                logger.warning("Watchdog: no command received, wheels stopped")

    def diagnostics(self) -> dict:
        return {
            "port": self.config.port,
            "baud": self.config.baud,
            "protocol": self.protocol.name,
            "servo_ids": list(self.config.servo_ids),
            "commands_sent": self.commands_sent,
            "replies_received": self.replies_received,
            "dry_run": self.dry_run,
        }
