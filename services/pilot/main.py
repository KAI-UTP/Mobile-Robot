"""Run an autonomous room scan on the real robot.

    # one terminal — the robot's sensors
    python services/robot-agent/main.py --servo-port COM5

    # another — the map
    python services/mapper/main.py --source mqtt

    # another — drive it
    python services/pilot/main.py --servo-port COM5

The pilot subscribes to the fused pose the mapper publishes and to the raw
sensor packets the agent publishes, and writes velocity commands straight to
the servo bus over USB-C.

Why the driver is opened here rather than commanded over MQTT
-------------------------------------------------------------
A broker hiccup between "stop" and the wheels is a robot that keeps driving.
The pilot holds the serial port itself so the stop path is a local function
call, and the driver's watchdog then halts the wheels even if this process is
killed outright.

That does mean the agent and the pilot cannot share one serial port. Run the
agent with `--no-servo` and let the pilot own the bus; it publishes the wheel
telemetry the agent would have.

Dry run
-------
    python services/pilot/main.py --dry-run

Prints the commands it would send and touches no hardware. Worth doing first,
in a room, with the robot on a box.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "shared",
    ROOT / "services",
    ROOT / "services" / "servo-bus",
    ROOT,
    Path(__file__).parent,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pilot import Pilot, PilotConfig, ScanPhase
from robotmap_common.holonomic import BodyTwist, HolonomicGeometry
from robotmap_common.models import PoseEstimate, RoomOutline, SensorPacket
from robotmap_common.topics import Topics

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("pilot-main")

_running = True


def _handle_signal(signum, frame):
    global _running
    _running = False


class Latest:
    """Most recent message of each kind, with the time it arrived.

    Guarded, because MQTT delivers on its own thread while the control loop
    reads on this one. The arrival time is what lets the pilot notice that
    telemetry has stopped — a stale packet looks exactly like a fresh one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.packet: SensorPacket | None = None
        self.packet_at = 0.0
        self.pose: PoseEstimate | None = None
        self.pose_at = 0.0
        self.bounds: tuple[float, float, float, float] | None = None

    def on_packet(self, payload: bytes) -> None:
        try:
            packet = SensorPacket.model_validate_json(payload)
        except Exception:
            return
        with self._lock:
            self.packet, self.packet_at = packet, time.monotonic()

    def on_pose(self, payload: bytes) -> None:
        try:
            pose = PoseEstimate.model_validate_json(payload)
        except Exception:
            return
        with self._lock:
            self.pose, self.pose_at = pose, time.monotonic()

    def on_room(self, payload: bytes) -> None:
        try:
            room = RoomOutline.model_validate_json(payload)
        except Exception:
            return
        if not room.polygon:
            return
        xs = [p.x_m for p in room.polygon]
        ys = [p.y_m for p in room.polygon]
        with self._lock:
            self.bounds = (min(xs), min(ys), max(xs), max(ys))

    def read(self):
        with self._lock:
            now = time.monotonic()
            age = now - min(
                self.packet_at or now,
                self.pose_at or now,
            )
            return self.packet, self.pose, age, self.bounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous room scan")
    parser.add_argument("--servo-port", help="Servo bus port, e.g. COM5")
    parser.add_argument("--servo-baud", type=int, default=1_000_000)
    parser.add_argument("--protocol", default="sts3215")
    parser.add_argument("--ids", default="1,2,3", help="Servo IDs in wheel order")
    parser.add_argument("--wheel-radius", type=float, default=0.029)
    parser.add_argument("--wheel-offset", type=float, default=0.100)

    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--rate", type=float, default=10.0, help="Control Hz")
    parser.add_argument("--max-speed", type=float, default=0.20, help="m/s")
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Measure the outline only; skip the row-by-row interior sweep",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of driving. Touches no hardware.",
    )
    args = parser.parse_args()

    if not args.servo_port and not args.dry_run:
        raise SystemExit("--servo-port is required (or use --dry-run)")

    geometry = HolonomicGeometry(
        wheel_radius_m=args.wheel_radius, wheel_offset_m=args.wheel_offset
    )

    latest = Latest()
    driver = None
    mqtt = None

    try:
        import os

        os.environ.setdefault("MQTT_HOST", args.mqtt_host)
        from robotmap_common.mqtt_client import build_client

        def on_message(client, userdata, msg) -> None:
            if msg.topic == Topics.SENSORS_RAW:
                latest.on_packet(msg.payload)
            elif msg.topic == Topics.POSE:
                latest.on_pose(msg.payload)
            elif msg.topic == Topics.ROOM:
                latest.on_room(msg.payload)

        mqtt = build_client(
            client_id="pilot",
            on_message=on_message,
            subscriptions=[Topics.SENSORS_RAW, Topics.POSE, Topics.ROOM],
        )

        if not args.dry_run:
            from driver import BusConfig, ServoBusDriver

            ids = tuple(int(v) for v in args.ids.split(","))
            if len(ids) != 3:
                raise SystemExit("--ids needs exactly three servo IDs")

            driver = ServoBusDriver(
                BusConfig(
                    port=args.servo_port,
                    baud=args.servo_baud,
                    protocol=args.protocol,
                    servo_ids=ids,
                ),
                geometry,
            )
            driver.open()
            driver.configure_wheel_mode()

        pilot = Pilot(
            PilotConfig(max_linear_mps=args.max_speed, sweep=not args.no_sweep)
        )

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        print()
        print("=" * 64)
        print(" Autonomous scan")
        print(f"   servo bus : {args.servo_port if driver else 'DRY RUN — not driving'}")
        print(f"   max speed : {args.max_speed} m/s")
        print(f"   sweep     : {'no (outline only)' if args.no_sweep else 'yes'}")
        print()
        print("   Ctrl-C stops the wheels and exits.")
        print("=" * 64)
        print()

        interval = 1.0 / max(args.rate, 0.1)
        last_report = 0.0

        while _running and pilot.is_running:
            started = time.monotonic()

            packet, pose, age, bounds = latest.read()
            if bounds is not None:
                pilot.set_bounds(bounds)

            twist = pilot.step(packet, pose, interval, age)

            if driver is not None:
                driver.drive(twist)
            elif _moving(twist):
                logger.info(
                    "would drive vx=%+.3f vy=%+.3f w=%+.1f  [%s] %s",
                    twist.vx_mps, twist.vy_mps, twist.omega_dps,
                    pilot.status.phase.value, pilot.status.note,
                )

            if started - last_report > 5.0:
                logger.info("%s", pilot.status.as_dict())
                last_report = started

            time.sleep(max(0.0, interval - (time.monotonic() - started)))

        print()
        if pilot.status.phase == ScanPhase.DONE:
            print(f" Scan complete — {pilot.status.note}")
            print(" See the measurement at http://localhost:8080/scans")
        elif pilot.status.phase == ScanPhase.STOPPED:
            print(f" Scan STOPPED — {pilot.status.stopped_reason}")
        else:
            print(" Interrupted.")

    finally:
        # Stop the wheels before anything else, on every path out of here
        # including an exception. A crash that leaves a robot driving is not an
        # acceptable failure.
        if driver is not None:
            try:
                driver.stop()
            except Exception:
                logger.exception("Could not stop the wheels — CUT THE POWER")
            try:
                driver.close()
            except Exception:
                logger.exception("Error closing the servo bus")
        if mqtt is not None:
            try:
                mqtt.loop_stop()
                mqtt.disconnect()
            except Exception:
                pass
        print(" wheels stopped")


def _moving(twist: BodyTwist) -> bool:
    return not twist.is_stationary(1e-6)


if __name__ == "__main__":
    main()
