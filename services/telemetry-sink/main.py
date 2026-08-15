"""Persist telemetry into InfluxDB so Grafana has something to chart.

Subscribes to every topic the system publishes and writes them as InfluxDB
measurements. Deliberately the only component that knows InfluxDB exists — if
the store is swapped later, nothing else changes.

Measurements written
--------------------
    robot_pose      x, y, heading, uncertainty, distance travelled
    room_estimate   area, dimensions, coverage, closure
    robot_sensors   wheel ticks, battery, range readings
    gps_status      fix quality, satellites, HDOP, accept/reject
    twin_gap        divergence between the real robot and the ideal

Batching
--------
Points are buffered and flushed together. At 10 Hz across five measurements a
per-point write would be fifty HTTP round trips a second, which InfluxDB
handles poorly and which makes the sink the slowest thing in the stack.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "shared", ROOT / "services", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robotmap_common.topics import Topics

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("telemetry-sink")

INFLUX_URL = os.environ.get("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN", "roommapper-super-secret-token")
INFLUX_ORG = os.environ.get("INFLUX_ORG", "roommapper")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "roommapper")

FLUSH_INTERVAL_S = float(os.environ.get("FLUSH_INTERVAL_S", "1.0"))
MAX_BUFFER = int(os.environ.get("MAX_BUFFER", "500"))

_running = True


def _handle_signal(signum, frame):
    global _running
    _running = False


class InfluxWriter:
    """Buffered line-protocol writer."""

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.lock = threading.Lock()
        self.points_written = 0
        self.write_failures = 0
        self.client = None
        self.write_api = None

    def connect(self) -> bool:
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS
        except ImportError:
            logger.error(
                "influxdb-client not installed. pip install influxdb-client"
            )
            return False

        for attempt in range(1, 11):
            try:
                self.client = InfluxDBClient(
                    url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
                )
                # ready() is the cheapest call that actually proves the server
                # is up; constructing the client alone connects to nothing.
                self.client.ready()
                self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
                logger.info("Connected to InfluxDB at %s", INFLUX_URL)
                return True
            except Exception as exc:
                wait = min(2**attempt, 30)
                logger.warning(
                    "InfluxDB not ready (attempt %d): %s — retrying in %ds",
                    attempt, exc, wait,
                )
                time.sleep(wait)

        logger.error("Giving up on InfluxDB after 10 attempts")
        return False

    @staticmethod
    def _escape(value: str) -> str:
        """Escape a tag value for line protocol."""
        return str(value).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")

    def add(self, measurement: str, tags: dict, fields: dict) -> None:
        """Buffer one point. Non-numeric and empty fields are dropped."""
        clean = {}
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, bool):
                # Booleans must be written as InfluxDB booleans, not as the
                # integers Python would otherwise coerce them to.
                clean[key] = "true" if value else "false"
            elif isinstance(value, (int, float)):
                clean[key] = f"{float(value)}"
            else:
                clean[key] = f'"{value}"'

        if not clean:
            return

        tag_text = "".join(
            f",{key}={self._escape(value)}" for key, value in tags.items() if value
        )
        field_text = ",".join(f"{key}={value}" for key, value in clean.items())

        with self.lock:
            self.buffer.append(f"{measurement}{tag_text} {field_text}")
            if len(self.buffer) >= MAX_BUFFER:
                self._flush_locked()

    def flush(self) -> None:
        with self.lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self.buffer or self.write_api is None:
            return

        batch = self.buffer
        self.buffer = []

        try:
            self.write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=batch)
            self.points_written += len(batch)
        except Exception as exc:
            self.write_failures += 1
            # Dropped rather than retried: telemetry is a continuous stream, so
            # a growing backlog would consume memory indefinitely and the stale
            # points would be worth less than the live ones replacing them.
            logger.warning("Dropped %d points: %s", len(batch), exc)

    def close(self) -> None:
        self.flush()
        if self.client is not None:
            self.client.close()


writer = InfluxWriter()


# ── Message handlers ─────────────────────────────────────────────────────────


def handle_pose(payload: dict) -> None:
    robot_id = payload.get("robot_id", "unknown")
    writer.add(
        "robot_pose",
        {"robot_id": robot_id, "source": payload.get("source", "")},
        {
            "x_m": payload.get("x_m"),
            "y_m": payload.get("y_m"),
            "heading_deg": payload.get("heading_deg"),
            "std_x_m": payload.get("std_x_m"),
            "std_y_m": payload.get("std_y_m"),
            "linear_velocity_mps": payload.get("linear_velocity_mps"),
            "angular_velocity_dps": payload.get("angular_velocity_dps"),
            "distance_travelled_m": payload.get("distance_travelled_m"),
        },
    )

    # The twin-control publisher adds the ideal pose alongside the real one.
    if "position_error_m" in payload:
        writer.add(
            "twin_gap",
            {"robot_id": robot_id, "mode": payload.get("mirror_mode", "")},
            {
                "position_error_m": payload.get("position_error_m"),
                "heading_error_deg": payload.get("heading_error_deg"),
                "ideal_x_m": payload.get("ideal_x_m"),
                "ideal_y_m": payload.get("ideal_y_m"),
            },
        )


def handle_room(payload: dict) -> None:
    writer.add(
        "room_estimate",
        {"robot_id": payload.get("robot_id", "unknown")},
        {
            "area_m2": payload.get("area_m2"),
            "perimeter_m": payload.get("perimeter_m"),
            "long_side_m": payload.get("bounding_width_m"),
            "short_side_m": payload.get("bounding_height_m"),
            "coverage_pct": payload.get("coverage_pct"),
            "is_closed": payload.get("is_closed"),
            "vertices": len(payload.get("polygon", [])),
        },
    )


def handle_sensors(payload: dict) -> None:
    robot_id = payload.get("robot_id", "unknown")

    encoders = payload.get("encoders") or {}
    wheel_ticks = payload.get("wheel_ticks") or []
    fields = {
        "dt_ms": encoders.get("dt_ms"),
        "bumper_active": payload.get("bumper_active"),
        "sequence": payload.get("sequence"),
    }
    for index, ticks in enumerate(wheel_ticks[:3]):
        fields[f"wheel{index}_ticks"] = ticks

    power = payload.get("power") or {}
    fields["battery_v"] = power.get("battery_v")
    fields["battery_soc"] = power.get("battery_soc")

    ranges = payload.get("ranges") or []
    for reading in ranges:
        if reading.get("valid"):
            fields[f"range_{int(reading.get('angle_deg', 0))}_m"] = reading.get(
                "distance_m"
            )

    writer.add(
        "robot_sensors",
        {"robot_id": robot_id, "link": payload.get("link", "")},
        fields,
    )

    gps = payload.get("gps")
    if gps:
        writer.add(
            "gps_status",
            {"robot_id": robot_id, "fix_quality": gps.get("fix_quality", "")},
            {
                "satellites": gps.get("satellites"),
                "hdop": gps.get("hdop"),
                "latitude": gps.get("latitude"),
                "longitude": gps.get("longitude"),
                # Reproduces the filter's own gate, so the dashboard can show
                # how much of the GPS stream was actually usable.
                "usable": (
                    gps.get("fix_quality") not in (None, "NO_FIX")
                    and (gps.get("satellites") or 0) >= 5
                    and (gps.get("hdop") or 99.9) <= 2.0
                ),
            },
        )


HANDLERS = {
    Topics.POSE: handle_pose,
    Topics.ROOM: handle_room,
    Topics.SENSORS_RAW: handle_sensors,
}


def on_message(client, userdata, message) -> None:
    handler = HANDLERS.get(message.topic)
    if handler is None:
        return
    try:
        handler(json.loads(message.payload))
    except json.JSONDecodeError:
        logger.debug("Non-JSON payload on %s", message.topic)
    except Exception:
        logger.exception("Handler failed for %s", message.topic)


def main() -> None:
    if not writer.connect():
        raise SystemExit(1)

    from robotmap_common.mqtt_client import build_client

    build_client(
        client_id="telemetry-sink",
        on_message=on_message,
        subscriptions=list(HANDLERS),
    )

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("Sink running. Subscribed to %s", list(HANDLERS))

    last_report = time.monotonic()
    try:
        while _running:
            time.sleep(FLUSH_INTERVAL_S)
            writer.flush()

            if time.monotonic() - last_report > 30.0:
                logger.info(
                    "%d points written, %d failed batches",
                    writer.points_written,
                    writer.write_failures,
                )
                last_report = time.monotonic()
    finally:
        writer.close()
        logger.info("Sink stopped after %d points", writer.points_written)


if __name__ == "__main__":
    main()
