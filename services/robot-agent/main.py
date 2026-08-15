"""Run the robot agent: sensors in, telemetry out.

    python services/robot-agent/main.py --servo-port COM5 --gps-port COM7

Find the ports first:

    python services/servo-bus/scan.py --list
    python services/robot-agent/main.py --find-gps

This is the process that used to be the ESP32 firmware. It polls the servo bus
and the GNSS receiver, assembles SensorPackets and publishes them, so the
mapper works unchanged against the real robot:

    python services/mapper/main.py --source mqtt
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "shared",
    ROOT / "services",
    ROOT / "services" / "gps-reader",
    ROOT / "services" / "servo-bus",
    ROOT,
    Path(__file__).parent,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent import AgentConfig, RobotAgent
from robotmap_common.holonomic import HolonomicGeometry
from robotmap_common.models import LinkType
from robotmap_common.topics import Topics

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("robot-agent")

_running = True


def _handle_signal(signum, frame):
    global _running
    _running = False


def find_gps() -> None:
    from reader import find_gps_port

    print()
    print("Probing serial ports for an NMEA stream...")
    print("(a GNSS receiver talks unprompted, so it identifies itself)")
    print()
    found = find_gps_port(verbose=True)
    if found:
        port, baud = found
        print()
        print(f"  Use:  --gps-port {port} --gps-baud {baud}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot sensor agent")
    parser.add_argument("--find-gps", action="store_true", help="Locate the receiver and exit")

    parser.add_argument("--servo-port", help="Servo bus port, e.g. COM5")
    parser.add_argument("--servo-baud", type=int, default=1_000_000)
    parser.add_argument("--protocol", default="sts3215")
    parser.add_argument("--ids", default="1,2,3", help="Servo IDs in wheel order")

    parser.add_argument("--gps-port", help="GNSS receiver port, e.g. COM7")
    parser.add_argument("--gps-baud", type=int, default=9600)

    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--robot-id", default="MR3W01")
    parser.add_argument("--rate", type=float, default=10.0, help="Telemetry Hz")
    parser.add_argument("--wheel-radius", type=float, default=0.029)
    parser.add_argument("--wheel-offset", type=float, default=0.100)
    parser.add_argument(
        "--bluetooth",
        action="store_true",
        help="Servo port is a Bluetooth SPP port (affects the reported link only)",
    )
    args = parser.parse_args()

    if args.find_gps:
        find_gps()
        return

    geometry = HolonomicGeometry(
        wheel_radius_m=args.wheel_radius, wheel_offset_m=args.wheel_offset
    )

    servo = None
    gps = None
    mqtt = None

    try:
        if args.servo_port:
            from driver import BusConfig, ServoBusDriver

            ids = tuple(int(v) for v in args.ids.split(","))
            if len(ids) != 3:
                raise SystemExit("--ids needs exactly three servo IDs")

            servo = ServoBusDriver(
                BusConfig(
                    port=args.servo_port,
                    baud=args.servo_baud,
                    protocol=args.protocol,
                    servo_ids=ids,
                ),
                geometry,
            )
            servo.open()
        else:
            logger.warning("No --servo-port: packets will carry no wheel data")

        if args.gps_port:
            from reader import GpsReader

            gps = GpsReader(args.gps_port, args.gps_baud)
            gps.start()
        else:
            logger.info("No --gps-port: running without GNSS (normal indoors)")

        if not args.no_publish:
            import os

            os.environ.setdefault("MQTT_HOST", args.mqtt_host)
            from robotmap_common.mqtt_client import build_client

            mqtt = build_client(client_id=f"robot-agent-{args.robot_id}")

        agent = RobotAgent(
            servo_driver=servo,
            gps_reader=gps,
            geometry=geometry,
            config=AgentConfig(
                robot_id=args.robot_id,
                telemetry_hz=args.rate,
                link=(
                    LinkType.BLUETOOTH_SERIAL if args.bluetooth else LinkType.WIFI_MQTT
                ),
            ),
        )

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        print()
        print("=" * 62)
        print(" Robot agent running")
        print(f"   servo bus : {args.servo_port or 'not connected'}")
        print(f"   gnss      : {args.gps_port or 'not connected'}")
        print(f"   publishing: {'no' if args.no_publish else Topics.SENSORS_RAW}")
        print(f"   rate      : {args.rate} Hz")
        print()
        print("   Start the mapper in another terminal:")
        print("     python services/mapper/main.py --source mqtt")
        print("=" * 62)

        interval = 1.0 / max(args.rate, 0.1)
        last_report = time.monotonic()

        while _running:
            started = time.monotonic()
            packet = agent.build_packet()

            if mqtt is not None:
                mqtt.publish(Topics.SENSORS_RAW, packet.model_dump_json())

            if time.monotonic() - last_report > 10.0:
                diagnostics = agent.diagnostics()
                if gps is not None:
                    diagnostics.update(gps.diagnostics())
                logger.info("%s", diagnostics)
                last_report = time.monotonic()

            # Sleep the remainder of the period, not the whole period, so the
            # rate holds regardless of how long polling took.
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, interval - elapsed))

    finally:
        # Stop the wheels before releasing anything else.
        if servo is not None:
            try:
                servo.close()
            except Exception:
                logger.exception("Error closing the servo bus")
        if gps is not None:
            gps.stop()
        print(" agent stopped")


if __name__ == "__main__":
    main()
