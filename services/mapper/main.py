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
import math
import os
import sys
import tempfile
import threading
from dataclasses import asdict
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

from mapper.exports import scan_to_json, scan_to_svg, scans_to_csv
from mapper.pipeline import MappingPipeline
from mapper.storage import Scan, ScanStore, assess_quality, safe_name
from mapper.world import describe_world

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

# Saved scans. Defaults to ./scans; set SCAN_DIR to a mounted volume in Docker
# so a container rebuild does not take the user's measurements with it.
store = ScanStore(os.environ.get("SCAN_DIR"))

# Whether the world description is genuine ground truth. Only the simulator
# knows the real room; with hardware it is an expectation.
app.state.source_mode = os.environ.get("SOURCE", "sim")

# Where in the room the robot started, which is the origin of the pose
# estimate's frame — the filter zeroes itself wherever the robot is standing.
# Any view that draws a pose into a room laid out from a corner must add this,
# or the robot is drawn outside its own room. With hardware it is wherever the
# operator set the robot down, so it is configurable.
app.state.robot_start = (
    float(os.environ.get("ROBOT_START_X_M", "1.0")),
    float(os.environ.get("ROBOT_START_Y_M", "1.0")),
)

# How the current scan was set up, so "scan again" can repeat it exactly.
# None in hardware mode: there is no simulator to restart, and a rescan there
# can only clear the map and wait for the robot to be driven round again.
app.state.sim_settings = None
app.state.coverage = None
app.state.collisions = None

# Mirror the pose into the file the Omniverse scene polls, so the robot shown
# there is the one actually driving the map — collisions, contacts and all —
# rather than Kit's scripted fallback lap.
app.state.write_pose_file = os.environ.get("POSE_FILE", "1") not in ("0", "")
app.state.pose_file_path = os.environ.get(
    "POSE_FILE_PATH",
    os.path.join(
        tempfile.gettempdir(),
        f"roommapper_{os.environ.get('ROBOT_ID', 'MR3W01')}_pose.json",
    ),
)

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


@app.get("/twin")
async def twin_page() -> FileResponse:
    """The digital twin: the physical room in 3D beside the robot's own map."""
    return FileResponse(STATIC_DIR / "twin.html")


@app.get("/api/world")
async def get_world() -> JSONResponse:
    """The physical room — the other half of the twin.

    Flagged as ground truth only in simulator mode. With a real robot the true
    room is unknown, and presenting a reference layout as if it were measured
    would make the comparison look authoritative when it is an assumption.
    """
    world = describe_world(
        app.state.reference_room_name,
        is_ground_truth=app.state.source_mode == "sim",
        robot_start=app.state.robot_start,
    )
    return JSONResponse(world.to_dict())


@app.get("/api/coverage")
async def get_coverage() -> JSONResponse:
    """How the interior sweep is going, and what it has bumped into.

    Separate from `/api/room` because the two answer different questions: the
    room outline is the *result*, this is the *process*. It is also the only
    place the robot's own account of an obstacle appears — the map's obstacles
    come from the occupancy grid, and the two agreeing is a good sign.
    """
    planner = getattr(app.state, "coverage", None)
    if planner is None:
        return JSONResponse({"active": False, "state": "NOT_STARTED"})

    payload = planner.summary()
    payload["active"] = not planner.is_finished
    payload["row_index"] = planner.row_index

    # How contact was detected, which is not a detail: with no bumper switch
    # every collision here is an inference, and the two detectors fail in
    # different ways. Reporting the split is what makes a red patch on the map
    # auditable rather than something the robot merely asserts.
    detector = getattr(app.state, "collisions", None)
    if detector is not None:
        payload["collision_detection"] = detector.summary()
        payload["collision_events"] = [
            {
                "x_m": round(e.x_m, 3),
                "y_m": round(e.y_m, 3),
                "reason": e.reason,
                "detector": e.detector,
            }
            for e in detector.events[-20:]
        ]
    payload["contacts"] = [
        {
            "x_m": round(o.x_m, 3),
            "y_m": round(o.y_m, 3),
            "radius_m": o.radius_m,
            "hits": o.hit_count,
            "from_collision": o.from_collision,
        }
        for o in planner.stats.obstacles
    ]
    return JSONResponse(payload)


@app.get("/scans")
async def scans_page() -> FileResponse:
    """The saved-scan library."""
    return FileResponse(STATIC_DIR / "scans.html")


@app.get("/compare")
async def compare_page() -> FileResponse:
    """Two screens: the real room, and the room the robot drew."""
    return FileResponse(STATIC_DIR / "compare.html")


# ── Saved scans ───────────────────────────────────────────────────────────────


_GRADE_ORDER = {"UNUSABLE": 0, "POOR": 1, "ACCEPTABLE": 2, "GOOD": 3}


def save_current_scan(name: str | None = None, replace: str | None = None) -> dict:
    """Persist whatever the robot has mapped so far.

    Called automatically when a circuit closes, and manually from the UI. The
    grade is computed here rather than trusted from the caller so a scan
    cannot be saved claiming to be better than it was.

    `replace` re-saves over an existing scan, which is how the two-phase run
    works: the perimeter lap saves the outline immediately so an interrupted
    sweep cannot lose it, and the sweep then adds what it found to that same
    record. One room should produce one scan, not two rows in the library that
    the user has to work out are the same room.

    When replacing, the outline is kept from whichever pass graded better. The
    perimeter lap measures the boundary from close range and usually wins; the
    sweep contributes the obstacles, which only it can see. Taking the best of
    each is not flattery — it is using each pass for what it actually measured
    well, and the grade reported is still the one that outline earned.
    """
    room = pipeline.refresh_room()
    if room is None or not room.polygon:
        raise ValueError("nothing mapped yet")

    pose = pipeline.pose
    quality = assess_quality(
        is_closed=room.is_closed,
        coverage_pct=room.coverage_pct,
        pose_confidence=pose.position_confidence if pose else 0.0,
    )

    previous = store.load(replace) if replace else None

    scan = Scan(
        scan_id=previous.scan_id if previous else store.new_id(),
        name=(
            previous.name if previous
            else safe_name(name or "", fallback=f"Room {len(store.list_scans()) + 1}")
        ),
        created_at=previous.created_at if previous else datetime.now(UTC).isoformat(),
        robot_id=pipeline.robot_id,
        area_m2=room.area_m2,
        perimeter_m=room.perimeter_m,
        long_side_m=room.bounding_width_m,
        short_side_m=room.bounding_height_m,
        polygon=[{"x_m": p.x_m, "y_m": p.y_m} for p in room.polygon],
        quality=quality,
        obstacles=[o.model_dump() for o in room.obstacles],
        blocked_area_m2=room.blocked_area_m2,
        distance_travelled_m=pose.distance_travelled_m if pose else 0.0,
        grid_b64=ScanStore.encode_grid(pipeline.grid_bytes()),
        grid_meta=pipeline.state()["grid"],
    )

    if previous is not None and _GRADE_ORDER.get(
        previous.quality.grade, 0
    ) > _GRADE_ORDER.get(quality.grade, 0):
        # The earlier pass measured the boundary better. Keep its outline and
        # its honest grade; only the obstacle findings carry forward.
        scan.area_m2 = previous.area_m2
        scan.perimeter_m = previous.perimeter_m
        scan.long_side_m = previous.long_side_m
        scan.short_side_m = previous.short_side_m
        scan.polygon = previous.polygon
        scan.quality = previous.quality

    store.save(scan)
    return scan.summary()


@app.get("/api/scans")
async def list_scans() -> JSONResponse:
    return JSONResponse({"scans": store.list_scans(), "totals": store.totals()})


@app.post("/api/scans")
async def create_scan(body: dict | None = None) -> JSONResponse:
    try:
        summary = save_current_scan((body or {}).get("name"))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(summary, status_code=201)


@app.get("/api/scans/{scan_id}")
async def get_scan(scan_id: str) -> JSONResponse:
    try:
        scan = store.load(scan_id)
    except ValueError:
        return JSONResponse({"error": "invalid scan id"}, status_code=400)
    if scan is None:
        return JSONResponse({"error": "not found"}, status_code=404)

    payload = scan.summary()
    payload["polygon"] = scan.polygon
    payload["quality"] = asdict(scan.quality)
    payload["notes"] = scan.notes
    payload["distance_travelled_m"] = scan.distance_travelled_m
    return JSONResponse(payload)


@app.patch("/api/scans/{scan_id}")
async def update_scan(scan_id: str, body: dict) -> JSONResponse:
    try:
        if "name" in body and not store.rename(scan_id, body["name"]):
            return JSONResponse({"error": "not found"}, status_code=404)
        if "notes" in body and not store.set_notes(scan_id, body["notes"]):
            return JSONResponse({"error": "not found"}, status_code=404)
    except ValueError:
        return JSONResponse({"error": "invalid scan id"}, status_code=400)
    return JSONResponse({"status": "ok"})


@app.delete("/api/scans/{scan_id}")
async def delete_scan(scan_id: str) -> JSONResponse:
    try:
        deleted = store.delete(scan_id)
    except ValueError:
        return JSONResponse({"error": "invalid scan id"}, status_code=400)
    if not deleted:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"status": "deleted"})


# ── Exports ───────────────────────────────────────────────────────────────────


@app.get("/api/scans/{scan_id}/floorplan.svg")
async def export_svg(scan_id: str) -> Response:
    try:
        scan = store.load(scan_id)
    except ValueError:
        return JSONResponse({"error": "invalid scan id"}, status_code=400)
    if scan is None:
        return JSONResponse({"error": "not found"}, status_code=404)

    filename = f"{safe_name(scan.name, 'floorplan').replace(' ', '-')}.svg"
    return Response(
        content=scan_to_svg(scan),
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/scans/{scan_id}/scan.json")
async def export_json(scan_id: str) -> Response:
    try:
        scan = store.load(scan_id)
    except ValueError:
        return JSONResponse({"error": "invalid scan id"}, status_code=400)
    if scan is None:
        return JSONResponse({"error": "not found"}, status_code=404)

    filename = f"{safe_name(scan.name, 'scan').replace(' ', '-')}.json"
    return Response(
        content=scan_to_json(scan),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/scans.csv")
async def export_csv() -> Response:
    """Every scan as a spreadsheet — where a quote actually gets written."""
    return Response(
        content=scans_to_csv(store.list_scans()),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="room-scans.csv"'},
    )


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


@app.post("/api/rescan")
async def post_rescan() -> JSONResponse:
    """Measure the room again, from scratch.

    Run off the event loop: stopping the previous scan means joining its
    thread, and blocking the loop for that would stall every websocket and
    freeze the map for everyone watching.
    """
    result = await asyncio.to_thread(rescan)
    status = 409 if result["status"] in ("busy", "error") else 200
    return JSONResponse(result, status_code=status)


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
    elif action == "rescan":
        await asyncio.to_thread(rescan)
    else:
        logger.warning("Unknown command from browser: %r", action)


async def _send_frame(socket: WebSocket) -> None:
    await socket.send_text(json.dumps({"type": "state", "data": pipeline.state()}))
    await socket.send_bytes(pipeline.grid_bytes())


def _write_pose_file() -> None:
    """Publish the pose where the Omniverse scene can read it.

    This is what makes the robot in Omniverse the *real* one. Without it the
    Kit scene falls back to its built-in demo lap, which is a scripted
    rectangle with no collision model — so the robot glides through the sofa,
    the cabinet and the bin, and the whole scene stops being a twin of
    anything. Every collision behaviour built into the simulator is invisible
    until the two are connected.

    A plain file rather than MQTT because Kit's Python is not this project's
    and cannot be assumed to have paho installed. `urllib` and `json` are all
    the scene needs, and both are standard library.

    Written atomically. The follower polls this path, and a half-written file
    is a parse error several times a second.
    """
    if not app.state.write_pose_file or pipeline.pose is None:
        return

    try:
        path = app.state.pose_file_path
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write(pipeline.pose.model_dump_json())
        os.replace(temp_path, path)
    except Exception:
        # Losing the twin feed must never take the map down with it.
        logger.debug("Could not write the pose file", exc_info=True)


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
    _write_pose_file()

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
    """Attach an MQTT publisher for the derived pose and room, in the background.

    Deliberately threaded. `build_client` retries ten times with exponential
    backoff, which is up to 210 seconds when no broker is listening — and
    doing that on the startup path meant the map served nothing for three and
    a half minutes on any machine without MQTT. It also failed CI, where there
    is no broker and the health check waits 60 seconds.

    The publisher is an optional extra: it feeds Grafana. The map is the
    product, so it must come up immediately whether or not a broker ever
    appears, and simply start publishing later if one does.
    """

    def connect() -> None:
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
                "No MQTT publisher (%s). Grafana will have no data; the live "
                "map is unaffected.", exc,
            )

    threading.Thread(target=connect, daemon=True, name="mqtt-publisher").start()


def start_mqtt_source() -> None:
    """Subscribe to the robot's telemetry, without blocking startup.

    Threaded for the same reason as the publisher: a broker that is slow to
    appear — or a robot switched on after the dashboard — must not stop the
    map from being served in the meantime.
    """
    def subscribe() -> None:
        from robotmap_common.mqtt_client import build_client

        def on_message(client, userdata, msg) -> None:
            _ingest(msg.payload)

        try:
            build_client(
                client_id="mapper",
                on_message=on_message,
                subscriptions=[Topics.SENSORS_RAW],
            )
            logger.info("Subscribed to %s", Topics.SENSORS_RAW)
        except Exception as exc:
            logger.error("Could not subscribe to telemetry: %s", exc)

    threading.Thread(target=subscribe, daemon=True, name="mqtt-source").start()


def start_sim_source(
    room: str,
    indoor: bool,
    speed: float,
    sweep: bool = True,
    stop: threading.Event | None = None,
) -> None:
    """Drive the virtual robot on a background thread.

    `stop` lets a run be abandoned part-way, which is what "scan again" needs:
    the previous robot has to stop touching the pipeline before a new one
    starts, or two simulators interleave packets into one map and the result is
    a room drawn by two robots in different places.

    Two phases, in this order and for different reasons:

    1. **Perimeter lap.** Wall-following observes the boundary from close
       range and at good incidence angles, which is what the room *outline*
       needs. It learns nothing about the middle of the room.
    2. **Row-by-row sweep.** A boustrophedon pass over the interior, which is
       the only way to find a table standing in open floor. Without it the
       reported floor area silently includes the space furniture occupies.

    The outline is measured and saved after phase 1 regardless, so a sweep
    that is interrupted still leaves a usable room measurement behind.
    """
    from robotmap_common.collision import CollisionDetector
    from robotmap_common.holonomic import BodyTwist, inverse_kinematics

    from autonomy.coverage import CoveragePlanner, CoverageState
    from autonomy.explorer import ExploreState, WallFollower
    from simulator.virtual_robot import VirtualRobot, VirtualWorld

    worlds = {
        "rectangular": lambda: VirtualWorld.rectangular_room(6.0, 4.5),
        "l-shaped": VirtualWorld.l_shaped_room,
        "furnished": lambda: VirtualWorld.room_with_furniture(6.0, 4.5),
    }
    world = worlds.get(room, worlds["rectangular"])()
    world.indoor = indoor

    robot = VirtualRobot(world=world)
    # The pose filter zeroes itself here, so this point is the origin of every
    # pose coordinate the system reports. The 3D views need it to place the
    # robot inside the room rather than beside it.
    robot.true_x, robot.true_y = app.state.robot_start
    follower = WallFollower()
    dt_s = 0.1

    # No bumper switch on this robot, so a collision is inferred from the servo
    # bus: wheels not delivering the speed they were asked for, or the robot
    # not covering the ground it was commanded over. See collision.py.
    detector = CollisionDetector()
    app.state.collisions = detector

    def check_contact(twist: BodyTwist) -> bool:
        """Fold one control cycle into the detector and draw any collision.

        Returns True when a NEW collision was recorded.
        """
        commanded = inverse_kinematics(twist, robot.holonomic_geometry).values
        event = detector.update(
            commanded_speed_mps=math.hypot(twist.vx_mps, twist.vy_mps),
            commanded_wheel_speeds=commanded,
            measured_wheel_speeds=robot.measured_wheel_speeds,
            wheel_loads=robot.wheel_loads,
            x_m=pipeline.pose.x_m,
            y_m=pipeline.pose.y_m,
            heading_deg=pipeline.pose.heading_deg,
            dt_s=dt_s,
        )
        if event is None:
            return False
        pipeline.record_contact(pipeline.pose, event.reason)
        return True

    # Deliberately NO bespoke recovery here.
    #
    # An obvious-looking addition — reverse for a dozen cycles on every
    # contact — was tried and made the map far worse: 122 m2 reported for a
    # 27 m2 room, with obstacles scattered fourteen metres out. Reversing
    # outside the control loop fights the wall-follower's own state machine,
    # and the lap closes against a robot that has wandered.
    #
    # None is needed. Contact detection is edge-triggered, so leaning on an
    # obstacle is recorded once, and the follower's front sensor turns it away
    # under its own logic. The pilot, which drives real hardware, has proper
    # recovery built into its control loop rather than bolted beside it.

    def report(stage: str) -> None:
        pipeline.refresh_room()
        result = pipeline.room
        if result is None:
            return
        logger.info(
            "%s — %.2f m2 floor, %.2f m2 blocked by %d obstacle(s), "
            "%.2f m2 usable, %.2f x %.2f m, closed=%s",
            stage,
            result.area_m2,
            result.blocked_area_m2,
            len(result.obstacles),
            result.usable_area_m2,
            result.bounding_width_m,
            result.bounding_height_m,
            result.is_closed,
        )

    saved_scan_id: dict[str, str | None] = {"id": None}

    def save(stage: str) -> None:
        # Saved automatically. Finishing a scan and then losing it because
        # nobody pressed a button is the worst possible outcome for someone who
        # just drove a robot round a room.
        try:
            saved = save_current_scan(replace=saved_scan_id["id"])
            saved_scan_id["id"] = saved["scan_id"]
            logger.info(
                "Saved '%s' (%s) after %s — %s",
                saved["name"], saved["grade"], stage,
                f"{saved['usable_area_m2']} m2 usable of {saved['area_m2']} m2",
            )
        except Exception:
            logger.exception("Could not save the scan after %s", stage)

    def loop() -> None:
        import time

        logger.info("Simulator running (room=%s, indoor=%s)", room, indoor)

        # ── Phase 1: the perimeter ────────────────────────────────────────
        while True:
            if stop is not None and stop.is_set():
                logger.info("Scan abandoned during the perimeter lap")
                return
            packet = robot.build_packet(dt_ms=int(dt_s * 1000))
            pipeline.process(packet)

            command = follower.step(
                packet.ranges,
                pipeline.pose.x_m,
                pipeline.pose.y_m,
                dt_s,
                # Inferred from the servo bus — this robot has no bumper.
                # Without it the lap leans on the bin and never closes.
                blocked=detector.in_contact,
            )
            if command.state == ExploreState.FINISHED:
                break

            robot.drive(command.linear_mps, command.angular_dps, dt_s)
            check_contact(
                BodyTwist(vx_mps=command.linear_mps, omega_dps=command.angular_dps)
            )
            # speed > 1 fast-forwards the demo; the physics step is unchanged.
            time.sleep(dt_s / max(speed, 0.01))

        logger.info("Perimeter complete after %.1f m", follower.distance_travelled_m)
        report("Outline")
        save("the perimeter lap")

        if not sweep:
            return

        # ── Phase 2: the interior ─────────────────────────────────────────
        # The sweep is given the outline phase 1 just measured, so it turns at
        # the end of each row instead of discovering the wall by hitting it.
        # This is the payoff for doing the perimeter first.
        outline = pipeline.room
        bounds = None
        if outline and outline.polygon:
            xs = [p.x_m for p in outline.polygon]
            ys = [p.y_m for p in outline.polygon]
            bounds = (min(xs), min(ys), max(xs), max(ys))

        logger.info(
            "Sweeping the interior row by row to find what is on the floor "
            "(bounds %s)",
            "unknown" if bounds is None else
            f"{bounds[0]:.1f},{bounds[1]:.1f} to {bounds[2]:.1f},{bounds[3]:.1f}",
        )
        planner = CoveragePlanner(bounds=bounds)
        app.state.coverage = planner

        while not planner.is_finished:
            if stop is not None and stop.is_set():
                logger.info("Scan abandoned during the interior sweep")
                return
            packet = robot.build_packet(dt_ms=int(dt_s * 1000))
            pipeline.process(packet)
            pose = pipeline.pose

            command = planner.step(
                packet.ranges,
                pose.x_m,
                pose.y_m,
                pose.heading_deg,
                # Inferred from the servo bus, not read from a switch: this
                # robot has no bumper, so `packet.bumper_active` is always
                # False and using it would leave the sweep blind to contact.
                detector.in_contact,
                dt_s,
            )
            if command.state == CoverageState.FINISHED:
                break

            # Strafing, not turning: the base is holonomic, so it changes rows
            # without ever taking its forward sensor off the direction of
            # travel. This is the one manoeuvre a differential robot cannot do.
            robot.drive_holonomic(
                command.linear_mps, command.lateral_mps, command.angular_dps, dt_s
            )
            check_contact(
                BodyTwist(
                    vx_mps=command.linear_mps,
                    vy_mps=command.lateral_mps,
                    omega_dps=command.angular_dps,
                )
            )
            time.sleep(dt_s / max(speed, 0.01))

        summary = planner.summary()
        logger.info(
            "Sweep complete: %d rows, %.1f m driven, %d obstacle(s) found "
            "(%d by contact), %d collision(s)",
            summary["rows_completed"],
            summary["distance_m"],
            summary["obstacles_found"],
            summary["obstacles_from_contact"],
            summary["collisions"],
        )
        report("Final")
        save("the interior sweep")

    thread = threading.Thread(target=loop, daemon=True, name="simulator")
    thread.start()
    return thread


# ── Re-running a scan ────────────────────────────────────────────────────────
#
# A scan is a one-shot: the robot drives its lap, sweeps, saves, and the thread
# ends. Measuring the room a second time meant restarting the container, which
# is a strange thing to ask of someone comparing two runs of the same room.

_scan_lock = threading.Lock()
_scan_stop = threading.Event()
_scan_thread: threading.Thread | None = None


def rescan() -> dict:
    """Wipe the map and drive the whole scan again from the start.

    Serialised on a lock, and the running scan is stopped and *joined* before
    the next one starts. Both matter: two simulator threads interleaving
    packets into one pipeline would draw a room from two robots standing in
    different places, and it would look like a mapping bug rather than like two
    scans fighting.
    """
    global _scan_thread

    if not _scan_lock.acquire(blocking=False):
        return {"status": "busy", "detail": "a restart is already in progress"}

    try:
        settings = getattr(app.state, "sim_settings", None)

        if _scan_thread is not None and _scan_thread.is_alive():
            _scan_stop.set()
            # Generous next to a 0.1 s control step, but the thread may be
            # inside a sleep scaled by --speed. Joining rather than assuming is
            # the whole point.
            _scan_thread.join(timeout=10.0)
            if _scan_thread.is_alive():
                return {
                    "status": "error",
                    "detail": "the previous scan did not stop; map left untouched",
                }

        _scan_stop.clear()
        _scan_thread = None

        # Clear only once the old robot has definitely stopped writing.
        pipeline.reset(clear_map=True)
        app.state.coverage = None
        app.state.collisions = None

        if settings is None:
            # Hardware mode: there is no simulator to restart, so the honest
            # thing is to clear the map and say the robot has to be sent round
            # again — which this service cannot do on its own.
            logger.info("Map cleared; waiting for the robot to drive again")
            return {
                "status": "cleared",
                "detail": "map cleared — drive the robot again to build a new one",
            }

        _scan_thread = start_sim_source(stop=_scan_stop, **settings)
        logger.info("Scan restarted (room=%s)", settings["room"])
        return {"status": "restarted", "room": settings["room"]}
    finally:
        _scan_lock.release()


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
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Measure the outline only; skip the row-by-row interior sweep",
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
    app.state.source_mode = args.source
    if not args.no_mqtt_publish:
        connect_publisher()

    if args.source == "sim":
        # The simulated room IS the ground truth, so score against it.
        app.state.reference_room_name = args.room
        # Remembered so "scan again" repeats this exact run rather than a
        # default one — a comparison between two scans is only meaningful if
        # both were driven the same way.
        app.state.sim_settings = {
            "room": args.room,
            "indoor": not args.outdoor,
            "speed": args.speed,
            "sweep": not args.no_sweep,
        }
        global _scan_thread
        _scan_thread = start_sim_source(stop=_scan_stop, **app.state.sim_settings)
    else:
        start_mqtt_source()

    logger.info("Open http://localhost:%d          — live map", args.port)
    logger.info("Open http://localhost:%d/compare  — real room vs robot's map", args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
