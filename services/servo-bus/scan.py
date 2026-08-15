"""Identify the servo bus: which port, which protocol, which baud, which IDs.

Run this first, before anything else touches the hardware:

    python services/servo-bus/scan.py --port COM5

The servo brand has not been confirmed, and the three candidate families use
incompatible packet formats at different default baud rates. Rather than guess
and get silence, this sweeps every combination and reports what answers.

It only ever sends ping packets — nothing that could make a wheel turn — so it
is safe to run with the robot on the bench.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "shared", ROOT / "services", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import serial
import serial.tools.list_ports
from protocols import ALL_PROTOCOLS, COMMON_BAUDS, get_protocol


def list_ports() -> None:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        print()
        print("Checklist:")
        print("  1. Is the servo bus board powered from its adapter?")
        print("     Most boards do NOT enumerate over USB without it.")
        print("  2. Is the USB-C cable a data cable? Many are charge-only.")
        print("  3. Windows may need a CH340 or CP210x driver.")
        return

    print(f"{'PORT':<10} {'DESCRIPTION':<44} HARDWARE ID")
    print("-" * 100)
    for port in ports:
        print(f"{port.device:<10} {(port.description or '')[:44]:<44} {port.hwid}")

    print()
    print("Servo bus boards usually appear as a USB-serial bridge:")
    print("  CH340 / CH341   very common on budget boards")
    print("  CP2102 / CP210x  Silicon Labs")
    print("  FT232 / FTDI")


def scan_bus(
    port: str,
    protocols: list[str],
    bauds: list[int],
    max_id: int,
    verbose: bool,
) -> list[dict]:
    """Sweep protocol x baud x id, returning everything that replied."""
    found: list[dict] = []

    for protocol_name in protocols:
        protocol = get_protocol(protocol_name)

        for baud in bauds:
            try:
                connection = serial.Serial(port, baud, timeout=0.03)
            except serial.SerialException as exc:
                print(f"  cannot open {port} at {baud}: {exc}")
                continue

            ids_here = []
            try:
                # Let the adapter settle; some assert DTR and reset the board.
                time.sleep(0.15)
                connection.reset_input_buffer()

                for servo_id in range(1, max_id + 1):
                    connection.reset_input_buffer()
                    connection.write(protocol.ping(servo_id))
                    reply = connection.read(16)

                    if reply and protocol.parse_ping_reply(reply, servo_id):
                        ids_here.append(servo_id)
                        if verbose:
                            print(
                                f"      id {servo_id:>3} replied: {reply.hex(' ')}"
                            )
            finally:
                connection.close()

            status = f"{len(ids_here)} servo(s): {ids_here}" if ids_here else "-"
            print(f"  {protocol_name:<12} {baud:>9} baud   {status}")

            if ids_here:
                found.append(
                    {"protocol": protocol_name, "baud": baud, "ids": ids_here}
                )

    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify the servo bus")
    parser.add_argument("--list", action="store_true", help="List serial ports and exit")
    parser.add_argument("--port", help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument(
        "--protocol",
        choices=sorted(ALL_PROTOCOLS) + ["all"],
        default="all",
    )
    parser.add_argument("--baud", type=int, help="Probe one baud rate only")
    parser.add_argument("--max-id", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", help="Show raw replies")
    args = parser.parse_args()

    if args.list or not args.port:
        list_ports()
        if not args.port:
            print()
            print("Then run:  python services/servo-bus/scan.py --port <PORT>")
        return

    protocols = sorted(ALL_PROTOCOLS) if args.protocol == "all" else [args.protocol]
    bauds = [args.baud] if args.baud else COMMON_BAUDS

    print()
    print("=" * 66)
    print(f" Scanning {args.port} — {len(protocols)} protocol(s) x {len(bauds)} baud")
    print(" Only ping packets are sent; no wheel will move.")
    print("=" * 66)

    found = scan_bus(args.port, protocols, bauds, args.max_id, args.verbose)

    print()
    print("=" * 66)
    if not found:
        print(" No servos replied.")
        print()
        print(" Most likely causes, in order:")
        print("   1. The bus board is not powered from its adapter.")
        print("      USB alone powers the bridge chip but often not the bus.")
        print("   2. Wrong port — try --list and pick another.")
        print("   3. The daisy chain is not connected, or connected to the")
        print("      board's output rather than its input.")
        print("   4. A servo family this tool does not know. Ask Shebaro for")
        print("      the exact servo model printed on the case.")
        raise SystemExit(1)

    print(" FOUND:")
    for entry in found:
        print(
            f"   protocol={entry['protocol']}  baud={entry['baud']}  "
            f"ids={entry['ids']}"
        )

    best = found[0]
    print()
    print(" Put this in your BusConfig:")
    print()
    print("     BusConfig(")
    print(f'         port="{args.port}",')
    print(f"         baud={best['baud']},")
    print(f'         protocol="{best["protocol"]}",')
    ids = best["ids"][:3]
    if len(ids) == 3:
        print(f"         servo_ids=({ids[0]}, {ids[1]}, {ids[2]}),")
    else:
        print(f"         servo_ids=(...),  # expected 3 wheels, found {len(best['ids'])}")
    print("     )")

    if len(best["ids"]) != 3:
        print()
        print(f" WARNING: found {len(best['ids'])} servos but the robot has three")
        print(" wheels. Check the daisy chain, or the extra servos belong to")
        print(" something else on the same bus.")

    print()
    print(" Next: confirm which servo is which wheel —")
    print(f"   python services/servo-bus/calibrate.py --port {args.port} --spin-each")
    print("=" * 66)


if __name__ == "__main__":
    main()
