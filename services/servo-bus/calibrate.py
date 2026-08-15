"""Work out which servo is which wheel, and which way each turns.

Run after `scan.py`, before driving the robot for the first time:

    python services/servo-bus/calibrate.py --port COM5 --spin-each

Why this is a separate step
---------------------------
Two mistakes are almost impossible to spot by eye once the robot is moving,
because in both cases all three wheels turn smoothly and the robot glides off
confidently in the wrong direction:

1. **Wrong servo order.** `servo_ids` must be listed in the same order as
   `wheel_angles_deg`. Servo ID 1 is not necessarily the front wheel.
2. **Wrong rotation direction.** Two of the three wheels are typically mounted
   mirrored, so a positive command drives them backwards.

Both are settled here, one wheel at a time, with the robot lifted off the
ground so nothing can run away.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "shared", ROOT / "services", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from driver import BusConfig, ServoBusDriver
from robotmap_common.holonomic import BodyTwist, HolonomicGeometry

logging.basicConfig(level=logging.INFO, format="%(message)s")


def spin_each(driver: ServoBusDriver, seconds: float) -> None:
    """Turn one wheel at a time so each can be identified physically."""
    print()
    print("=" * 64)
    print(" LIFT THE ROBOT OFF THE GROUND before continuing.")
    print(" Each wheel spins in turn; note which physical wheel moves.")
    print("=" * 64)
    input(" Press Enter when the robot is lifted... ")

    geometry = driver.geometry
    protocol = driver.protocol
    units = protocol.velocity_units_per_rad_s()
    # A gentle speed: fast enough to see, slow enough to stop by hand.
    raw = int(round((0.05 / geometry.wheel_radius_m) * units))

    for index, servo_id in enumerate(driver.config.servo_ids):
        angle = geometry.wheel_angles_deg[index]
        print()
        print(f"  Wheel slot {index}  (servo ID {servo_id}, mounted at {angle:.0f} deg)")
        print("    spinning FORWARD ...")

        driver._write(protocol.set_velocity(servo_id, raw))
        time.sleep(seconds)
        driver._write(protocol.set_velocity(servo_id, 0))
        time.sleep(0.4)

        print("    spinning BACKWARD ...")
        driver._write(protocol.set_velocity(servo_id, -raw))
        time.sleep(seconds)
        driver._write(protocol.set_velocity(servo_id, 0))

        print("    Which physical wheel moved, and did FORWARD turn it the way")
        print("    that would push the robot anticlockwise about its centre?")
        input("    Press Enter for the next wheel... ")

    print()
    print("  If the wheels moved in a different order than slots 0, 1, 2,")
    print("  reorder servo_ids in BusConfig to match.")
    print("  If FORWARD turned a wheel the wrong way, set invert[slot] = True.")


def check_order(driver: ServoBusDriver, seconds: float) -> None:
    """Command pure rotation — the one motion with an unambiguous signature.

    Spinning on the spot is the only motion where all three wheels should turn
    the same way at the same speed. If one opposes the others, its `invert`
    flag is wrong. This isolates direction errors from ordering errors, which
    is why it is worth checking before anything more complex.
    """
    print()
    print("=" * 64)
    print(" ROTATION CHECK — robot still lifted.")
    print(" All three wheels should now turn the SAME way at the SAME speed.")
    print(" Any wheel opposing the others has its invert flag wrong.")
    print("=" * 64)
    input(" Press Enter to spin... ")

    driver.drive(BodyTwist(omega_dps=45.0))
    deadline = time.time() + seconds
    while time.time() < deadline:
        readings = driver.read_wheels()
        speeds = [r.speed_raw for r in readings]
        print(f"    raw speeds: {speeds}")
        time.sleep(0.4)
    driver.stop()

    print()
    print("  Signs should all match. If one differs, flip its invert flag.")


def check_translation(driver: ServoBusDriver, seconds: float) -> None:
    """Drive forward, and confirm the front wheel barely turns.

    With a wheel at 0 degrees, driving straight ahead should leave it almost
    stationary — its rollers absorb that motion. If it is spinning hard while
    the other two are slow, the wheel order is wrong.
    """
    print()
    print("=" * 64)
    print(" TRANSLATION CHECK — robot still lifted.")
    print(" Driving 'forward': the wheel at 0 degrees should turn LEAST.")
    print("=" * 64)
    input(" Press Enter to drive... ")

    driver.drive(BodyTwist(vx_mps=0.1))
    deadline = time.time() + seconds
    slowest_counts = [0, 0, 0]
    while time.time() < deadline:
        readings = driver.read_wheels()
        speeds = [abs(r.speed_raw or 0) for r in readings]
        slowest = speeds.index(min(speeds))
        slowest_counts[slowest] += 1
        print(f"    raw speeds: {speeds}   slowest = slot {slowest}")
        time.sleep(0.4)
    driver.stop()

    expected = 0  # the slot whose wheel sits at 0 degrees
    actual = slowest_counts.index(max(slowest_counts))
    print()
    if actual == expected:
        print("  PASS — the 0 degree wheel turned least, as expected.")
    else:
        print(f"  FAIL — slot {actual} turned least, expected slot {expected}.")
        print("  servo_ids are probably in the wrong order.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Servo bus calibration")
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--protocol", default="sts3215")
    parser.add_argument(
        "--ids", default="1,2,3", help="Servo IDs in wheel order, comma separated"
    )
    parser.add_argument("--spin-each", action="store_true")
    parser.add_argument("--check-order", action="store_true")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ids = tuple(int(v) for v in args.ids.split(","))
    if len(ids) != 3:
        parser.error("--ids needs exactly three servo IDs")

    config = BusConfig(
        port=args.port, baud=args.baud, protocol=args.protocol, servo_ids=ids
    )

    with ServoBusDriver(config, HolonomicGeometry(), dry_run=args.dry_run) as driver:
        if args.spin_each:
            spin_each(driver, args.seconds)
        if args.check_order or not args.spin_each:
            check_order(driver, args.seconds)
            check_translation(driver, args.seconds)

    print()
    print(" Calibration done. Update BusConfig with anything you changed.")


if __name__ == "__main__":
    main()
