"""Mapper service: ingests sensor packets, serves the live map.

Run modes
---------
* ``--source mqtt`` (default) subscribes to the broker and maps whatever the
  robot publishes.
* ``--source sim`` runs the virtual robot in-process, so the whole system can
  be demonstrated with no broker, no hardware and no Docker.

Both drive the identical `MappingPipeline`, so the demo exercises the real
code path rather than a parallel one written for show.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

# Make the sibling packages importable when run directly from a checkout.
ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "shared", ROOT / "services", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import uvicorn
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from robotmap_common.models import SensorPacket, ServiceHealth
from robotmap_common.topics import Topics

from mapper.pipeline import MappingPipeline

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("mapper")

STATIC_DIR = Path(__file__).parent / "static"

@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    """Run the map broadcast loop for the lifetime of the server."""
    task = asyncio.create_task(_broadcast_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="Room Mapper", lifespan=lifespan)
pipeline = MappingPipeline(robot_id=os.environ.get("ROBOT_ID", "MR3W01"))
_start_time = datetime.now(UTC)

# Which reference room the map is scored against on the /compare screen.
# Set from --room in simulator mode, or REFERENCE_ROOM when running against
# hardware in a room whose real dimensions are known.
app.state.reference_room_name = os.environ.get("REFERENCE_ROOM", "rectangular")

# Every connected browser. Guarded because packets arrive on an MQTT thread.
_clients: set[WebSocket] = set()
_clients_lock = threading.Lock()


# ── HTTP ──────────────────────────────────────────────────────────────────────


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> ServiceHealth:
    uptime = (datetime.now(UTC) - _start_time).total_seconds()
    return ServiceHealth(
        service="mapper",
        status="healthy",
        timestamp=datetime.now(UTC).isoformat(),
        uptime_s=uptime,
        details=pipeline.state()["diagnostics"],
    )


@app.get("/api/room")
async def get_room() -> JSONResponse:
    """The current room measurement — the project's headline output."""
    room = pipeline.refresh_room()
    if room is None:
        return JSONResponse(
            {"error": "no room mapped yet"}, status_code=404
        )
    return JSONResponse(room.model_dump())


@app.get("/api/state")
async def get_state() -> JSONResponse:
    return JSONResponse(pipeline.state())


@app.get("/compare")
async def compare_page() -> FileResponse:
    """Two screens: the real room, and the room the robot drew."""
    return FileResponse(STATIC_DIR / "compare.html")


@app.get("/api/compare")
async def get_comparison() -> JSONResponse:
    """Score the robot's map against the reference room.

    Returns both polygons so the page can draw them, plus the fidelity
    metrics. Without a reference room configured this reports the measured
    room alone rather than inventing a truth to compare against.
    """
    from robotmap_common.comparison import compare_rooms, reference_room

    room = pipeline.room
    measured = (
        [(p.x_m, p.y_m) for p in room.polygon] if room and room.polygon else []
    )

    truth = reference_room(app.state.reference_room_name)

    payload: dict = {
        "reference_name": app.state.reference_room_name,
        "measured_polygon": [{"x_m": x, "y_m": y} for x, y in measured],
        "truth_polygon": (
            [{"x_m": x, "y_m": y} for x, y in truth] if truth else []
        ),
        "has_reference": truth is not None,
        "room": room.model_dump() if room else None,
    }

    if truth and measured:
        result = compare_rooms(measured, truth)
        aligned = _align_for_display(measured, truth)
        payload["comparison"] = {
            "truth_area_m2": result.truth_area_m2,
            "measured_area_m2": result.measured_area_m2,
            "area_error_pct": result.area_error_pct,
            "truth_dimensions_m": list(result.truth_dimensions_m),
            "measured_dimensions_m": list(result.measured_dimensions_m),
            "long_side_error_pct": result.long_side_error_pct,
            "short_side_error_pct": result.short_side_error_pct,
            "iou": result.iou,
            "centroid_offset_m": result.centroid_offset_m,
            "perimeter_error_pct": result.perimeter_error_pct,
            "grade": result.grade,
            "summary": result.summary(),
        }
        payload["aligned_polygon"] = [{"x_m": x, "y_m": y} for x, y in aligned]

    return JSONResponse(payload)


def _align_for_display(measured, truth):
    """Overlay position for the measured room, for the shape-overlap panel."""
    from robotmap_common.comparison import align_polygons

    return align_polygons(measured, truth)


@app.get("/api/grid")
async def get_grid() -> Response:
    """Raw occupancy bytes, for the polling fallback."""
    return Response(content=pipeline.grid_bytes(), media_type="application/octet-stream")


@app.post("/api/command")
async def post_command(body: dict) -> JSONResponse:
    """Same actions as the WebSocket, for clients without one."""
    await _handle_command(body)
    return JSONResponse({"status": "ok", "action": body.get("action")})


@app.post("/api/reset")
async def reset() -> JSONResponse:
    pipeline.reset(clear_map=True)
    return JSONResponse({"status": "reset"})


# ── WebSocket ─────────────────────────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_endpoint(socket: WebSocket) -> None:
    await socket.accept()
    with _clients_lock:
        _clients.add(socket)
    logger.info("Browser connected (%d total)", len(_clients))

    try:
        # Send a full frame immediately so a page opened mid-run is not blank.
        await _send_frame(socket)
        while True:
            message = await socket.receive_text()
            await _handle_command(json.loads(message))
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error")
    finally:
        with _clients_lock:
            _clients.discard(socket)


async def _handle_command(message: dict) -> None:
    action = message.get("action")
    if action == "reset_pose":
        pipeline.reset(clear_map=False)
    elif action == "clear_map":
        pipeline.reset(clear_map=True)
    elif action == "refresh":
        pipeline.refresh_room()
    else:
        logger.warning("Unknown command from browser: %r", action)


async def _send_frame(socket: WebSocket) -> None:
    await socket.send_text(json.dumps({"type": "state", "data": pipeline.state()}))
    await socket.send_bytes(pipeline.grid_bytes())


def _publish_derived() -> None:
    """Publish the pose and room estimate for anything downstream.

    Without this the telemetry sink has nothing to persist and Grafana stays
    empty in the default simulator configuration — the pipeline runs
    in-process there, so the derived values never touch the broker unless they
    are put there deliberately.

    Republished at the broadcast rate rather than per packet: a 10 Hz pose is
    more resolution than a dashboard can show, and the room outline changes
    far more slowly still.
    """
    publisher = getattr(app.state, "mqtt_publisher", None)
    if publisher is None:
        return

    try:
        if pipeline.pose is not None:
            publisher(Topics.POSE, pipeline.pose.model_dump_json())
        if pipeline.room is not None:
            publisher(Topics.ROOM, pipeline.room.model_dump_json())
    except Exception:
        # A broker outage must not stop the map from being served.
        logger.debug("Could not publish derived state", exc_info=True)


async def _broadcast_loop(interval_s: float = 0.25) -> None:
    """Push the map to every browser a few times a second.

    Broadcasting on a timer rather than per packet decouples the UI frame rate
    from the telemetry rate, so a fast robot cannot flood the browser.
    """
    while True:
        await asyncio.sleep(interval_s)

        # Runs regardless of whether a browser is connected: the dashboards
        # and the time-series history must not depend on someone watching.
        _publish_derived()

        with _clients_lock:
            targets = list(_clients)
        if not targets:
            continue

        state_text = json.dumps({"type": "state", "data": pipeline.state()})
        grid = pipeline.grid_bytes()

        for socket in targets:
            try:
                await socket.send_text(state_text)
                await socket.send_bytes(grid)
            except Exception:
                with _clients_lock:
                    _clients.discard(socket)


# ── Sources ───────────────────────────────────────────────────────────────────


def _ingest(payload: bytes) -> None:
    """Validate and process one raw packet."""
    try:
        packet = SensorPacket.model_validate_json(payload)
    except ValidationError as exc:
        pipeline.packets_rejected += 1
        logger.warning("Rejected malformed packet: %s", exc.errors()[:2])
        return
    pipeline.process(packet)


def connect_publisher() -> None:
    """Attach an MQTT publisher for the derived pose and room.

    Optional by design: the offline demo must keep working on a laptop with no
    broker, so a failure here is logged and the map still serves.
    """
    try:
        from robotmap_common.mqtt_client import build_client

        client = build_client(client_id="mapper-publisher")
        app.state.mqtt_publisher = lambda topic, payload: client.publish(
            topic, payload
        )
        logger.info("Publishing pose and room to MQTT for the telemetry sink")
    except Exception as exc:
        app.state.mqtt_publisher = None
        logger.warning(
            "No MQTT publisher (%s). Grafana will have no data; the live map "
            "still works.", exc,
        )


def start_mqtt_source() -> None:
    from robotmap_common.mqtt_client import build_client

    def on_message(client, userdata, msg) -> None:
        _ingest(msg.payload)

    build_client(
        client_id="mapper",
        on_message=on_message,
        subscriptions=[Topics.SENSORS_RAW],
    )
    logger.info("Subscribed to %s", Topics.SENSORS_RAW)


def start_sim_source(room: str, indoor: bool, speed: float) -> None:
    """Drive the virtual robot on a background thread."""
    from simulator.explorer import ExploreState, WallFollower
    from simulator.virtual_robot import VirtualRobot, VirtualWorld

    worlds = {
        "rectangular": lambda: VirtualWorld.rectangular_room(6.0, 4.5),
        "l-shaped": VirtualWorld.l_shaped_room,
        "furnished": lambda: VirtualWorld.room_with_furniture(6.0, 4.5),
    }
    world = worlds.get(room, worlds["rectangular"])()
    world.indoor = indoor

    robot = VirtualRobot(world=world)
    robot.true_x, robot.true_y = 1.0, 1.0
    follower = WallFollower()
    dt_s = 0.1

    def loop() -> None:
        import time

        logger.info("Simulator running (room=%s, indoor=%s)", room, indoor)
        while True:
            packet = robot.build_packet(dt_ms=int(dt_s * 1000))
            pipeline.process(packet)

            command = follower.step(
                packet.ranges, pipeline.pose.x_m, pipeline.pose.y_m, dt_s
            )
            if command.state == ExploreState.FINISHED:
                logger.info(
                    "Circuit complete after %.1f m", follower.distance_travelled_m
                )
                pipeline.refresh_room()
                room_result = pipeline.room
                if room_result:
                    logger.info(
                        "Room: %.2f m2, %.2f x %.2f m, closed=%s",
                        room_result.area_m2,
                        room_result.bounding_width_m,
                        room_result.bounding_height_m,
                        room_result.is_closed,
                    )
                break

            robot.drive(command.linear_mps, command.angular_dps, dt_s)
            # speed > 1 fast-forwards the demo; the physics step is unchanged.
            time.sleep(dt_s / max(speed, 0.01))

    threading.Thread(target=loop, daemon=True, name="simulator").start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Room Mapper service")
    parser.add_argument(
        "--source",
        choices=["mqtt", "sim"],
        default=os.environ.get("SOURCE", "mqtt"),
        help="Where sensor packets come from",
    )
    parser.add_argument(
        "--room",
        choices=["rectangular", "l-shaped", "furnished"],
        default="rectangular",
        help="Which virtual room to map (--source sim only)",
    )
    parser.add_argument(
        "--outdoor",
        action="store_true",
        help="Simulate an outdoor GNSS environment (--source sim only)",
    )
    parser.add_argument(
        "--speed", type=float, default=4.0, help="Simulation speed multiplier"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--no-mqtt-publish",
        action="store_true",
        help="Do not publish pose and room (offline laptop demo with no broker)",
    )
    args = parser.parse_args()

    app.state.mqtt_publisher = None
    if not args.no_mqtt_publish:
        connect_publisher()

    if args.source == "sim":
        # The simulated room IS the ground truth, so score against it.
        app.state.reference_room_name = args.room
        start_sim_source(args.room, indoor=not args.outdoor, speed=args.speed)
    else:
        start_mqtt_source()

    logger.info("Open http://localhost:%d          — live map", args.port)
    logger.info("Open http://localhost:%d/compare  — real room vs robot's map", args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
