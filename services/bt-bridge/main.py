"""Bluetooth bridge: robot serial link -> MQTT.

Why this exists
---------------
An HC-05 (classic Bluetooth SPP) or a BLE UART presents itself to the laptop
as an ordinary serial port. This service reads newline-delimited JSON from
that port and republishes it on MQTT, so a robot with no WiFi produces
exactly the same topic stream as an ESP32 with WiFi. Nothing downstream needs
to know which link is in use.

Finding the port
----------------
Run with ``--list`` to see the available ports. On Windows the HC-05 appears
as a COM port after pairing (use the *outgoing* one); on Linux it is usually
/dev/rfcomm0 after ``rfcomm bind``; on macOS /dev/tty.HC-05-*.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "shared", ROOT / "services", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import serial
import serial.tools.list_ports
from pydantic import ValidationError
from robotmap_common.models import SensorPacket
from robotmap_common.topics import Topics

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("bt-bridge")


def list_ports() -> None:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        print("Pair the robot's Bluetooth module first, then run this again.")
        return

    print(f"{'PORT':<16} {'DESCRIPTION':<40} HARDWARE ID")
    for port in ports:
        print(f"{port.device:<16} {(port.description or '')[:40]:<40} {port.hwid}")

    print()
    print("Bluetooth modules usually show up with 'Bluetooth', 'HC-05', "
          "'BT', or the robot's name in the description.")


class BluetoothBridge:
    """Reads framed JSON from a serial port and republishes it."""

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        robot_id: str = "MR3W01",
        publish: bool = True,
    ) -> None:
        self.port_name = port
        self.baud = baud
        self.robot_id = robot_id
        self.publish = publish

        self.serial: serial.Serial | None = None
        self.mqtt = None

        self.packets_ok = 0
        self.packets_bad = 0
        self.bytes_read = 0

    # ── Connections ───────────────────────────────────────────────────────

    def open_serial(self) -> None:
        # A read timeout keeps the loop responsive to Ctrl-C; without it a
        # silent robot blocks the process indefinitely.
        self.serial = serial.Serial(self.port_name, self.baud, timeout=1.0)
        logger.info("Opened %s at %d baud", self.port_name, self.baud)

    def open_mqtt(self) -> None:
        if not self.publish:
            return
        from robotmap_common.mqtt_client import build_client

        self.mqtt = build_client(client_id=f"bt-bridge-{self.robot_id}")
        logger.info("Publishing to %s", Topics.SENSORS_RAW)

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        self.open_serial()
        self.open_mqtt()

        buffer = bytearray()
        last_report = time.time()

        while True:
            try:
                chunk = self.serial.read(256)
            except serial.SerialException as exc:
                logger.error("Serial link lost: %s", exc)
                self._reconnect()
                continue

            if chunk:
                self.bytes_read += len(chunk)
                buffer.extend(chunk)

                # Frames are newline-delimited. Anything after the last
                # newline is a partial frame and stays buffered for next time.
                while b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer = bytearray(rest)
                    self._handle_line(bytes(line).strip())

                # A runaway buffer means the robot is emitting data with no
                # newlines — usually a baud mismatch producing garbage.
                if len(buffer) > 8192:
                    logger.warning(
                        "Discarding %d unframed bytes — check the baud rate",
                        len(buffer),
                    )
                    buffer.clear()

            if time.time() - last_report > 10.0:
                logger.info(
                    "%d packets ok, %d malformed, %.1f kB read",
                    self.packets_ok,
                    self.packets_bad,
                    self.bytes_read / 1024,
                )
                last_report = time.time()

    def _handle_line(self, line: bytes) -> None:
        if not line:
            return

        try:
            packet = SensorPacket.model_validate_json(line)
        except ValidationError as exc:
            self.packets_bad += 1
            # Log sparsely: a baud mismatch would otherwise flood the console.
            if self.packets_bad <= 5 or self.packets_bad % 100 == 0:
                logger.warning(
                    "Malformed packet #%d: %s", self.packets_bad, exc.errors()[:1]
                )
            return
        except Exception:
            self.packets_bad += 1
            return

        self.packets_ok += 1

        if self.mqtt is not None:
            self.mqtt.publish(Topics.SENSORS_RAW, packet.model_dump_json())
        else:
            print(json.dumps({"seq": packet.sequence, "ranges": len(packet.ranges)}))

    def _reconnect(self) -> None:
        if self.serial:
            try:
                self.serial.close()
            except Exception:
                pass

        for delay in (1, 2, 5, 10, 30):
            logger.info("Reconnecting in %ds…", delay)
            time.sleep(delay)
            try:
                self.open_serial()
                return
            except serial.SerialException as exc:
                logger.warning("Reconnect failed: %s", exc)

        raise RuntimeError(f"Could not reopen {self.port_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bluetooth to MQTT bridge")
    parser.add_argument("--list", action="store_true", help="List serial ports and exit")
    parser.add_argument("--port", help="Serial port, e.g. COM5 or /dev/rfcomm0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--robot-id", default=os.environ.get("ROBOT_ID", "MR3W01"))
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Print packet summaries instead of publishing (link testing)",
    )
    args = parser.parse_args()

    if args.list:
        list_ports()
        return

    if not args.port:
        parser.error("--port is required (use --list to find it)")

    bridge = BluetoothBridge(
        port=args.port,
        baud=args.baud,
        robot_id=args.robot_id,
        publish=not args.no_publish,
    )
    try:
        bridge.run()
    except KeyboardInterrupt:
        logger.info(
            "Stopped. %d packets forwarded, %d malformed.",
            bridge.packets_ok,
            bridge.packets_bad,
        )


if __name__ == "__main__":
    main()
