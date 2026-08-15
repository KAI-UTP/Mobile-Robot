"""Type an instruction, the robot moves, and Omniverse moves with it.

    python services/twin-control/main.py --port COM5

Without hardware, to see the whole thing work first:

    python services/twin-control/main.py --dry-run

Commands (type and press Enter)
-------------------------------
    w [sec]        forward
    s [sec]        backward
    a [sec]        strafe LEFT      <- no turning; this is the holonomic bit
    d [sec]        strafe RIGHT
    q [sec]        rotate left
    e [sec]        rotate right
    go <deg> [sec] travel along a bearing, robot frame
    square         drive a square by strafing, never rotating
    spin           rotate on the spot through 360 degrees
    stop           stop
    report         print the sim-to-real gap so far
    save           write the divergence trace to CSV
    reset          zero both poses
    quit

Run `omniverse/kit_twin_follower.py` in the Omniverse Script Editor at the same
time and the twin will track every command.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "shared", ROOT / "services", ROOT, Path(__file__).parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robotmap_common.holonomic import (
    BodyTwist,
    DriveLimits,
    HolonomicGeometry,
    twist_from_direction,
)
from twin import MirrorMode, TwinController, make_file_publisher, make_mqtt_publisher

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("twin-control")

CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ


def build_servo_driver(args, geometry, limits):
    """Open the servo bus, or return None when running without hardware."""
    if args.dry_run:
        logger.info("Dry run: no servo bus, twin mirrors the command")
        return None

    from driver import BusConfig, ServoBusDriver

    ids = tuple(int(v) for v in args.ids.split(","))
    if len(ids) != 3:
        raise SystemExit("--ids needs exactly three servo IDs")

    config = BusConfig(
        port=args.port,
        baud=args.baud,
        protocol=args.protocol,
        servo_ids=ids,
    )
    driver = ServoBusDriver(config, geometry, limits)
    driver.open()
    return driver


def run_for(twin: TwinController, twist: BodyTwist, seconds: float) -> None:
    """Hold a twist for a duration, stepping at the control rate."""
    steps = max(1, int(seconds * CONTROL_HZ))
    for _ in range(steps):
        twin.step(twist, DT)
        time.sleep(DT)
    twin.step(BodyTwist(), DT)


def drive_square(twin: TwinController, speed: float, side_s: float) -> None:
    """A square driven by strafing, with no rotation at any corner.

    The clearest demonstration of what this platform can do: a differential
    robot would have to stop and turn four times.
    """
    print("  square by strafing — the robot never turns")
    for bearing in (0.0, 90.0, 180.0, 270.0):
        run_for(twin, twist_from_direction(bearing, speed), side_s)


def print_report(twin: TwinController) -> None:
    report = twin.gap_report()
    if report.get("samples", 0) == 0:
        print("  nothing recorded yet")
        return

    print()
    print("  " + "-" * 52)
    print("   Sim-to-real gap")
    print("  " + "-" * 52)
    for key, value in report.items():
        print(f"   {key:<28} {value}")
    print("  " + "-" * 52)
    if not report.get("hardware_connected"):
        print("   NOTE: no hardware connected, so the twin mirrored the")
        print("   command. These figures show zero drift by construction,")
        print("   not because the robot is accurate.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive the robot and its twin")
    parser.add_argument("--port", help="Servo bus serial port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--protocol", default="sts3215")
    parser.add_argument("--ids", default="1,2,3", help="Servo IDs in wheel order")
    parser.add_argument("--dry-run", action="store_true", help="No hardware")
    parser.add_argument(
        "--publish",
        choices=["mqtt", "file", "none"],
        default="file",
        help="How Omniverse receives the pose. 'file' needs nothing installed.",
    )
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--speed", type=float, default=0.15, help="m/s")
    parser.add_argument("--turn-rate", type=float, default=45.0, help="deg/s")
    parser.add_argument("--wheel-radius", type=float, default=0.029)
    parser.add_argument("--wheel-offset", type=float, default=0.100)
    args = parser.parse_args()

    if not args.dry_run and not args.port:
        parser.error("--port is required (or use --dry-run)")

    geometry = HolonomicGeometry(
        wheel_radius_m=args.wheel_radius,
        wheel_offset_m=args.wheel_offset,
    )
    limits = DriveLimits(
        max_linear_mps=max(0.05, args.speed * 2),
        max_angular_dps=max(15.0, args.turn_rate * 2),
        max_wheel_mps=0.6,
    )

    publisher = None
    if args.publish == "mqtt":
        publisher = make_mqtt_publisher(args.mqtt_host)
    elif args.publish == "file":
        publisher = make_file_publisher()

    servo = build_servo_driver(args, geometry, limits)

    twin = TwinController(
        servo_driver=servo,
        pose_publisher=publisher,
        geometry=geometry,
        mode=MirrorMode.FEEDBACK,
    )

    print()
    print("=" * 62)
    print(" Twin control — one command, two robots")
    print(f"   hardware : {'none (dry run)' if servo is None else args.port}")
    print(f"   mirror   : {twin.mode.value}")
    print(f"   pose out : {args.publish}")
    print()
    print("   Run omniverse/kit_twin_follower.py in the Script Editor to")
    print("   see the twin follow along.")
    print()
    print("   w/s forward/back   a/d STRAFE   q/e rotate")
    print("   square  spin  stop  report  save  reset  quit")
    print("=" * 62)

    single = {
        "w": lambda: BodyTwist(vx_mps=args.speed),
        "s": lambda: BodyTwist(vx_mps=-args.speed),
        "a": lambda: BodyTwist(vy_mps=args.speed),
        "d": lambda: BodyTwist(vy_mps=-args.speed),
        "q": lambda: BodyTwist(omega_dps=args.turn_rate),
        "e": lambda: BodyTwist(omega_dps=-args.turn_rate),
    }

    try:
        while True:
            try:
                # Strip a leading byte-order mark: it survives copy-paste and
                # piped input, and turns "square" into an unknown command for
                # no visible reason.
                raw = input("robot> ").lstrip("﻿").strip().lower()
            except EOFError:
                break
            if not raw:
                continue

            parts = raw.split()
            command = parts[0]

            if command in ("quit", "exit"):
                break

            if command in single:
                seconds = float(parts[1]) if len(parts) > 1 else 1.0
                run_for(twin, single[command](), seconds)
                state = twin.state
                print(
                    f"  at ({state.real_pose.x_m:+.2f}, {state.real_pose.y_m:+.2f}) "
                    f"hdg {state.real_pose.heading_deg:.0f}  "
                    f"drift {state.position_error_m:.3f} m"
                )

            elif command == "go":
                if len(parts) < 2:
                    print("  usage: go <bearing_deg> [seconds]")
                    continue
                bearing = float(parts[1])
                seconds = float(parts[2]) if len(parts) > 2 else 1.0
                run_for(twin, twist_from_direction(bearing, args.speed), seconds)

            elif command == "square":
                drive_square(twin, args.speed, 2.0)

            elif command == "spin":
                run_for(twin, BodyTwist(omega_dps=args.turn_rate), 360.0 / args.turn_rate)

            elif command == "stop":
                twin.stop()
                print("  stopped")

            elif command == "report":
                print_report(twin)

            elif command == "save":
                out = twin.export_divergence_csv(ROOT / "divergence.csv")
                print(f"  wrote {out}")

            elif command == "reset":
                twin.reset()
                print("  poses zeroed")

            else:
                print(f"  unknown command {command!r}")

    except KeyboardInterrupt:
        print()
    finally:
        # Stop the wheels before anything else. A robot left driving because
        # the program exited is the failure mode worth engineering against.
        try:
            twin.stop()
        finally:
            if servo is not None:
                servo.close()
        print_report(twin)
        print(" stopped cleanly")


if __name__ == "__main__":
    main()
