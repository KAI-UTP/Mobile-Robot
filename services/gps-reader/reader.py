"""Read a USB GNSS receiver on a background thread.

The receiver plugs into the PC, not the robot — there is no microcontroller to
host it. It appears as a serial port and streams NMEA continuously; this class
drains it in the background and keeps the latest fix available, so the control
loop never blocks waiting for a satellite.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "shared", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import serial
import serial.tools.list_ports
from nmea import NmeaState, parse_sentence
from robotmap_common.models import GpsData

logger = logging.getLogger("gps-reader")

# Almost every consumer GNSS module ships at 9600.
COMMON_GPS_BAUDS = [9600, 4800, 38400, 115200, 57600]


class GpsReader:
    """Keeps the latest fix from a USB GNSS receiver."""

    def __init__(
        self,
        port: str,
        baud: int = 9600,
        timeout_s: float = 1.0,
        max_fix_age_s: float = 5.0,
    ) -> None:
        self.port = port
        self.baud = baud
        self.timeout_s = timeout_s
        # Beyond this age a fix is treated as gone rather than current.
        self.max_fix_age_s = max_fix_age_s

        self.state = NmeaState()
        self.serial: serial.Serial | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self.last_fix_at: float | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def __enter__(self) -> GpsReader:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def start(self) -> None:
        self.serial = serial.Serial(self.port, self.baud, timeout=self.timeout_s)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gps")
        self._thread.start()
        logger.info("GPS reader started on %s at %d baud", self.port, self.baud)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.serial is not None:
            try:
                self.serial.close()
            finally:
                self.serial = None

    # ── Reading ───────────────────────────────────────────────────────────

    def _loop(self) -> None:
        buffer = ""
        while self._running:
            try:
                chunk = self.serial.read(256).decode("ascii", errors="ignore")
            except serial.SerialException as exc:
                logger.error("GPS serial error: %s", exc)
                time.sleep(1.0)
                continue

            if not chunk:
                continue

            buffer += chunk
            # Sentences are newline-delimited; keep any partial tail for next
            # time rather than parsing half a sentence.
            while "\n" in buffer:
                line, _, buffer = buffer.partition("\n")
                with self._lock:
                    if parse_sentence(line, self.state):
                        self.last_fix_at = time.monotonic()

            if len(buffer) > 4096:
                # No newlines arriving usually means the wrong baud rate.
                logger.warning("Discarding unframed GPS data; check the baud rate")
                buffer = ""

    def read(self) -> GpsData | None:
        """The latest fix, or None if no position has been received yet."""
        with self._lock:
            return self.state.to_gps_data()

    @property
    def is_fresh(self) -> bool:
        """False if the receiver has gone quiet.

        A stale fix that keeps being reported as current is worse than none:
        the filter would treat a minute-old position as a live measurement.
        Since going quiet is exactly what a receiver does when the robot drives
        indoors, this is the normal case rather than an edge case.

        The threshold is a constructor argument, not a parameter here — a
        property cannot take one, and an earlier version declared one anyway,
        where it silently had no effect.
        """
        if self.last_fix_at is None:
            return False
        return (time.monotonic() - self.last_fix_at) < self.max_fix_age_s

    def diagnostics(self) -> dict:
        with self._lock:
            return {
                "port": self.port,
                "baud": self.baud,
                "sentences_seen": self.state.sentences_seen,
                "sentences_bad": self.state.sentences_bad,
                "checksum_failures": self.state.checksum_failures,
                "fix_quality": self.state.fix_quality.value,
                "satellites": self.state.satellites,
                "hdop": self.state.hdop,
                "fresh": self.is_fresh,
            }


def find_gps_port(verbose: bool = True) -> tuple[str, int] | None:
    """Probe the serial ports for something emitting NMEA.

    A GNSS receiver is easy to identify without any guessing: it talks
    unprompted, and its sentences start with '$'. Nothing else on a typical
    machine does that.
    """
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        if verbose:
            print("No serial ports found.")
        return None

    for port in ports:
        for baud in COMMON_GPS_BAUDS:
            try:
                with serial.Serial(port.device, baud, timeout=1.5) as connection:
                    # Two seconds is enough for at least one sentence burst at
                    # the 1 Hz these modules default to.
                    data = connection.read(512).decode("ascii", errors="ignore")
                    if "$" in data and any(
                        tag in data for tag in ("GGA", "RMC", "GSA", "GSV")
                    ):
                        if verbose:
                            print(f"  Found NMEA on {port.device} at {baud} baud")
                        return port.device, baud
            except (serial.SerialException, OSError):
                continue

    if verbose:
        print("No NMEA stream found on any port.")
        print("  - Is the receiver plugged in?")
        print("  - A cold module can take 5-15 minutes for its first fix, but")
        print("    it should emit sentences (with empty fields) immediately.")
    return None
