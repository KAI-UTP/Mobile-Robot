"""A furnished 3D room in Omniverse, with the robot and its BLE beacons.

    Window > Script Editor, paste this whole file, Ctrl+Enter.

Builds a realistic room rather than an empty box: four solid walls, a shut door
and a window, a table and chairs, a sofa, a cabinet, a rug — and the four
Bluetooth
beacons in the corners, so the positioning setup is visible rather than
implied.

Two robots are drawn, and the gap between them is the point
-----------------------------------------------------------
* **Solid blue** — where the robot actually is.
* **Green outline** — where its own dead reckoning thinks it is.

The gap between them is the drift, and watching it open up over a run is the
most useful thing this scene shows: it is why the map comes back larger than
the room, and why a long contact-only scan grades poorly.

A third marker used to show where Bluetooth thought the robot was. It is gone
with the fusion that fed it — BLE measured worse than dead reckoning, 0.51 m of
error becoming 1.42 m, so nothing uses it and drawing it implied otherwise.

Furniture is not decoration
---------------------------
It attenuates radio and blocks ultrasonic pulses, which is exactly why real
rooms are harder than empty ones. The layout below is deliberately awkward:
a table in the middle of the floor and a sofa across one corner.

Data source
-----------
Reads the pose the mapper publishes — the file first, then /api/state over
plain HTTP — so nothing needs installing inside Kit beyond the standard
library.
"""

import json
import math
import os
import random
import tempfile
import time

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, Vt

# ── Configuration ───────────────────────────────────────────────────────────

ROOM_W = 6.0
ROOM_H = 4.5
WALL_HEIGHT = 2.4
WALL_THICK = 0.12

ROBOT_ID = "MR3W01"
POSE_FILE = os.environ.get(
    "POSE_FILE_PATH",
    os.path.join(tempfile.gettempdir(), f"roommapper_{ROBOT_ID}_pose.json"),
)

# Where in this room the robot started.
#
# This is the origin of the pose estimate's frame: the filter zeroes itself
# wherever the robot is standing when a run begins, so every pose it reports is
# relative to that point and is routinely NEGATIVE. The room below is laid out
# from a corner at (0, 0), so a pose drawn straight into it puts the robot
# outside its own room — which is exactly what it did.
#
# Must match the mapper, which serves the same figure as `robot_start_x_m` on
# /api/world. With hardware it is wherever the operator set the robot down.
ROBOT_START_X_M = 1.0
ROBOT_START_Y_M = 1.0

# A pose file older than this is ignored in favour of the demo lap. The mapper
# rewrites it several times a second while a run is going, so anything this
# stale is left over from a previous session — and silently animating a robot from
# yesterday's file, frozen whereever it stopped, looks exactly like a bug in the
# scene.
POSE_STALE_AFTER_S = 15.0

SHOW_ODOMETRY_GHOST = True


# Give the walls and furniture colliders, and the robot a kinematic body.
#
# This is what makes the room solid to PhysX, so anything dynamic collides with
# it and Isaac Sim can drive the robot against it for real. It does NOT make
# the blue marker stop on its own — that marker mirrors the pose the mapper
# reports, and collision for it is decided in the mapper's simulator. See
# `make_robot_physical`.
ENABLE_PHYSICS = True


# Which mapper the robot is followed from.
#
# Settable from the environment so a mapper on a non-default port can be
# followed without editing this file.
MAPPER_URL = os.environ.get("MAPPER_URL", "http://localhost:8080")


UPDATE_HZ = 60.0
SMOOTHING = 0.25

# How many breadcrumbs the trail keeps.
#
# One dot every 0.15 m, and a contact-only run drives 300 m or more, so an
# uncapped trail is two thousand prims and climbing — it buries the room it is
# drawn on and costs frame time for the privilege. Past this the oldest dot is
# moved to the newest position rather than a new prim being made, so the count
# is fixed and the trail becomes a rolling window of where the robot has just
# been. The full path is on the 2D map, which is a better place to read it.
MAX_TRAIL_DOTS = 900

# ── Palette ─────────────────────────────────────────────────────────────────

COL_FLOOR      = (0.74, 0.68, 0.58)
COL_RUG        = (0.35, 0.42, 0.50)
COL_WALL       = (0.90, 0.89, 0.86)
COL_SKIRTING   = (0.82, 0.80, 0.76)
COL_TABLE      = (0.45, 0.31, 0.20)
COL_CHAIR      = (0.30, 0.28, 0.26)
COL_SOFA       = (0.32, 0.38, 0.45)
COL_CABINET    = (0.52, 0.38, 0.26)
COL_DOOR       = (0.65, 0.50, 0.36)
COL_WINDOW     = (0.62, 0.78, 0.88)
COL_ROBOT      = (0.15, 0.55, 0.90)
COL_ROBOT_NOSE = (0.98, 0.78, 0.15)
COL_ODOM       = (0.25, 0.80, 0.35)
COL_BEACON     = (0.85, 0.25, 0.75)
COL_TRAIL      = (0.95, 0.60, 0.20)

ROOT = "/World/RoomScan"


# ── Primitive helpers ───────────────────────────────────────────────────────


def _colour(prim, rgb, opacity=None):
    prim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))
    if opacity is not None:
        UsdGeom.Gprim(prim).CreateDisplayOpacityAttr(Vt.FloatArray([opacity]))
    return prim


def box(stage, path, centre, size, rgb, opacity=None):
    """An axis-aligned box, given its centre and full extents."""
    prim = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    prim.CreateSizeAttr(1.0)
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*centre))
    api.SetScale(Gf.Vec3f(*size))
    return _colour(prim, rgb, opacity)


def cylinder(stage, path, centre, radius, height, rgb, rotate=None, opacity=None):
    prim = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    prim.CreateRadiusAttr(radius)
    prim.CreateHeightAttr(height)
    prim.CreateAxisAttr("Z")
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*centre))
    if rotate:
        api.SetRotate(Gf.Vec3f(*rotate))
    return _colour(prim, rgb, opacity)


def sphere(stage, path, centre, radius, rgb, opacity=None):
    prim = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    prim.CreateRadiusAttr(radius)
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*centre))
    return _colour(prim, rgb, opacity)


# ── The room ────────────────────────────────────────────────────────────────


def build_shell(stage):
    """Floor, four solid walls, a shut door and a window.

    The south wall used to be split around an opening, on the grounds that a
    doorway is realistic for the range sensors and the radio model. It was not
    realistic, it was a disagreement: the world the robot drives in is
    `VirtualWorld.rectangular_room`, which is four solid walls, so the robot
    met an invisible wall exactly where the opening was drawn.
    """
    UsdGeom.Xform.Define(stage, Sdf.Path(f"{ROOT}/Room"))

    box(stage, f"{ROOT}/Room/Floor",
        (ROOM_W / 2, ROOM_H / 2, -0.02), (ROOM_W, ROOM_H, 0.04), COL_FLOOR)

    h = WALL_HEIGHT / 2
    t = WALL_THICK

    # South wall, solid, with the door shut in it.
    #
    # It used to be split around a 0.9 m opening with the door swung inwards.
    # The robot's world has four solid walls — `VirtualWorld.rectangular_room`
    # builds exactly four — so the robot met an invisible wall precisely where
    # the doorway was drawn, and a viewer watching the 3D scene saw it refuse
    # to drive through an obvious gap.
    #
    # A gap is also the wrong thing to measure. The flood fill that finds the
    # room deliberately stops at walls, and `_is_enclosed` treats an open
    # doorway as a room whose boundary was never found — so an opening here
    # would leak the map into whatever is beyond it and grade every scan as
    # unusable. A closed room is what the mapping assumes, and now what it is
    # shown.
    door_x, door_w = 4.3, 0.9
    box(stage, f"{ROOT}/Room/WallS",
        (ROOM_W / 2, 0, h), (ROOM_W, t, WALL_HEIGHT), COL_WALL)
    # The door itself: shut, set into the face of the wall rather than cut
    # through it, so the room still reads as a room you could walk out of.
    box(stage, f"{ROOT}/Room/Door",
        (door_x, t / 2 + 0.02, 1.0), (door_w, 0.05, 2.0), COL_DOOR)

    box(stage, f"{ROOT}/Room/WallN",
        (ROOM_W / 2, ROOM_H, h), (ROOM_W, t, WALL_HEIGHT), COL_WALL)
    box(stage, f"{ROOT}/Room/WallW",
        (0, ROOM_H / 2, h), (t, ROOM_H, WALL_HEIGHT), COL_WALL)
    box(stage, f"{ROOT}/Room/WallE",
        (ROOM_W, ROOM_H / 2, h), (t, ROOM_H, WALL_HEIGHT), COL_WALL)

    # A window in the north wall.
    box(stage, f"{ROOT}/Room/Window",
        (2.0, ROOM_H, 1.45), (1.6, 0.02, 1.1), COL_WINDOW, opacity=0.35)

    # Skirting, purely so the scene reads as a room rather than a box.
    for name, centre, size in (
        ("N", (ROOM_W / 2, ROOM_H - t / 2, 0.05), (ROOM_W, 0.03, 0.10)),
        ("W", (t / 2, ROOM_H / 2, 0.05), (0.03, ROOM_H, 0.10)),
        ("E", (ROOM_W - t / 2, ROOM_H / 2, 0.05), (0.03, ROOM_H, 0.10)),
    ):
        box(stage, f"{ROOT}/Room/Skirting{name}", centre, size, COL_SKIRTING)

    print(f"[room] shell built: {ROOM_W} x {ROOM_H} x {WALL_HEIGHT} m, closed room")


def build_furniture(stage):
    """Obstacles that make the room hard, in the places that make it hardest.

    Every item here blocks ultrasonic pulses and attenuates radio. The table
    sits in open floor so the robot must go round it; the sofa cuts a corner,
    which is where wall-following most often loses the wall.
    """
    UsdGeom.Xform.Define(stage, Sdf.Path(f"{ROOT}/Furniture"))
    f = f"{ROOT}/Furniture"

    # Rug — visual only, but it is where wheel slip would be worst.
    box(stage, f"{f}/Rug", (2.6, 2.3, 0.005), (2.4, 1.6, 0.01), COL_RUG)

    # Dining table with four legs and four chairs.
    table_x, table_y = 2.6, 2.3
    box(stage, f"{f}/TableTop", (table_x, table_y, 0.74), (1.4, 0.85, 0.06), COL_TABLE)
    for index, (dx, dy) in enumerate(((-0.62, -0.36), (0.62, -0.36), (-0.62, 0.36), (0.62, 0.36))):
        cylinder(stage, f"{f}/TableLeg_{index}",
                 (table_x + dx, table_y + dy, 0.36), 0.035, 0.72, COL_TABLE)

    for index, (cx, cy, rot) in enumerate((
        (table_x - 1.0, table_y, 0), (table_x + 1.0, table_y, 180),
        (table_x, table_y - 0.75, 90), (table_x, table_y + 0.75, 270),
    )):
        box(stage, f"{f}/Chair_{index}_seat", (cx, cy, 0.45), (0.42, 0.42, 0.05), COL_CHAIR)
        back_dx = 0.19 * math.cos(math.radians(rot))
        back_dy = 0.19 * math.sin(math.radians(rot))
        box(stage, f"{f}/Chair_{index}_back",
            (cx - back_dx, cy - back_dy, 0.68), (0.42, 0.05, 0.45), COL_CHAIR)
        for leg, (lx, ly) in enumerate(((-0.17, -0.17), (0.17, -0.17), (-0.17, 0.17), (0.17, 0.17))):
            cylinder(stage, f"{f}/Chair_{index}_leg{leg}",
                     (cx + lx, cy + ly, 0.22), 0.02, 0.44, COL_CHAIR)

    # Sofa across the north-west corner.
    box(stage, f"{f}/SofaBase", (1.1, 3.7, 0.22), (2.0, 0.85, 0.44), COL_SOFA)
    box(stage, f"{f}/SofaBack", (1.1, 4.05, 0.58), (2.0, 0.18, 0.72), COL_SOFA)
    box(stage, f"{f}/SofaArmL", (0.15, 3.7, 0.42), (0.18, 0.85, 0.28), COL_SOFA)
    box(stage, f"{f}/SofaArmR", (2.05, 3.7, 0.42), (0.18, 0.85, 0.28), COL_SOFA)

    # Cabinet against the east wall.
    box(stage, f"{f}/Cabinet", (5.6, 2.6, 0.55), (0.45, 1.5, 1.10), COL_CABINET)

    # Bin — small, low, and exactly the sort of thing sonar misses.
    cylinder(stage, f"{f}/Bin", (5.55, 0.6, 0.18), 0.16, 0.36, COL_CHAIR)

    print("[room] furniture placed: table, 4 chairs, sofa, cabinet, bin, rug")


def build_beacons(stage):
    """The four BLE beacons, mounted high in the corners.

    Corners maximise the spread of bearings from anywhere inside, which is
    what keeps the geometry well conditioned. Four rather than three means one
    blocked beacon still leaves a solvable fix.

    Putting them all along one wall makes trilateration mathematically
    impossible, not merely worse — a mistake worth being able to point at in
    the scene.
    """
    UsdGeom.Xform.Define(stage, Sdf.Path(f"{ROOT}/Beacons"))
    height = 2.1

    for index, (x, y) in enumerate(
        ((0.15, 0.15), (ROOM_W - 0.15, 0.15), (ROOM_W - 0.15, ROOM_H - 0.15), (0.15, ROOM_H - 0.15))
    ):
        path = f"{ROOT}/Beacons/B{index + 1}"
        box(stage, f"{path}_body", (x, y, height), (0.09, 0.09, 0.03), COL_BEACON)
        sphere(stage, f"{path}_glow", (x, y, height), 0.14, COL_BEACON, opacity=0.28)

        light = UsdLux.SphereLight.Define(stage, Sdf.Path(f"{path}_light"))
        light.CreateRadiusAttr(0.05)
        light.CreateIntensityAttr(2000.0)
        light.CreateColorAttr(Gf.Vec3f(*COL_BEACON))
        UsdGeom.XformCommonAPI(light).SetTranslate(Gf.Vec3d(x, y, height))

    print("[room] 4 BLE beacons at the corners, 2.1 m high")


def build_lighting(stage):
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path(f"{ROOT}/Lighting/Sky"))
    dome.CreateIntensityAttr(400.0)

    ceiling = UsdLux.RectLight.Define(stage, Sdf.Path(f"{ROOT}/Lighting/Ceiling"))
    ceiling.CreateWidthAttr(2.5)
    ceiling.CreateHeightAttr(2.0)
    ceiling.CreateIntensityAttr(3000.0)
    api = UsdGeom.XformCommonAPI(ceiling)
    api.SetTranslate(Gf.Vec3d(ROOM_W / 2, ROOM_H / 2, WALL_HEIGHT - 0.1))
    api.SetRotate(Gf.Vec3f(180.0, 0.0, 0.0))


# ── The robots ──────────────────────────────────────────────────────────────


def build_robot(stage, path, rgb, opacity=None, with_wheels=True):
    UsdGeom.Xform.Define(stage, Sdf.Path(path))
    cylinder(stage, f"{path}/Body", (0, 0, 0.06), 0.14, 0.06, rgb, opacity=opacity)
    # Nose marker: without it, strafing and driving forward look identical.
    box(stage, f"{path}/Nose", (0.11, 0, 0.07), (0.07, 0.025, 0.025),
        COL_ROBOT_NOSE, opacity=opacity)

    if with_wheels:
        for index, angle_deg in enumerate((0.0, 120.0, 240.0)):
            a = math.radians(angle_deg)
            cylinder(stage, f"{path}/Wheel_{index}",
                     (0.10 * math.cos(a), 0.10 * math.sin(a), 0.029),
                     0.029, 0.018, (0.12, 0.12, 0.14),
                     rotate=(90.0, 0.0, angle_deg + 90.0))
    return stage.GetPrimAtPath(path)


def build_all_robots(stage):
    build_robot(stage, f"{ROOT}/RobotTrue", COL_ROBOT)

    if SHOW_ODOMETRY_GHOST:
        build_robot(stage, f"{ROOT}/RobotOdometry", COL_ODOM,
                    opacity=0.45, with_wheels=False)

    UsdGeom.Xform.Define(stage, Sdf.Path(f"{ROOT}/Trail"))


# ── Pose sources ────────────────────────────────────────────────────────────


class FilePose:
    """Reads the pose file twin-control writes."""

    def __init__(self, path=POSE_FILE):
        self.path = path
        self.latest = None
        self._mtime = 0.0

    def read(self):
        try:
            mtime = os.path.getmtime(self.path)
            # A file left over from a previous session is worse than no file:
            # it pins the robot wherever that run happened to stop, which looks
            # like the scene being broken rather than like nothing running.
            if time.time() - mtime > POSE_STALE_AFTER_S:
                return None
            if mtime != self._mtime:
                self._mtime = mtime
                with open(self.path, encoding="utf-8") as handle:
                    self.latest = json.load(handle)
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[room] pose read failed: {exc}")
        return self.latest


#: Furniture prims that are decoration rather than obstruction. A rug is not
#: something a robot bumps into, and reporting it as an obstacle would put a
#: red patch on the 2D map where the floor is perfectly clear.
NOT_SOLID = ("Rug", "Shadow", "Label")


def furniture_footprints(stage):
    """Where every solid piece of furniture is on the stage *right now*.

    Read from the prims rather than from the layout constants they were built
    from, which is the entire point: drag a table across the viewport and this
    returns its new place, so the robot bumps into it where you just put it.

    Returns axis-aligned `(min_x, min_y, max_x, max_y)` boxes in metres, in the
    room's own frame. A box footprint is what the simulator's raycaster and
    contact test already understand, and flattening a 3D prim to its floor
    footprint is right for a robot that is 0.2 m tall and drives underneath
    nothing.
    """
    root = stage.GetPrimAtPath(Sdf.Path(f"{ROOT}/Furniture"))
    if not root:
        return []

    boxes = []
    for prim in root.GetChildren():
        path = prim.GetPath().pathString
        if any(skip in path for skip in NOT_SOLID):
            continue

        placement = _prim_placement(prim)
        if placement is None:
            continue
        (cx, cy), (half_w, half_d) = placement
        if half_w <= 0.0 or half_d <= 0.0:
            continue
        boxes.append((cx - half_w, cy - half_d, cx + half_w, cy + half_d))

    return boxes


def _prim_placement(prim):
    """Centre and half-extents on the floor plane, or None if it has neither.

    Cubes carry their size in the scale; cylinders carry it in a radius. Both
    are read through `XformCommonAPI` so that a prim someone has *moved* in the
    viewport reports where it now is rather than where the script first put it.
    """
    try:
        vectors = UsdGeom.XformCommonAPI(prim).GetXformVectors(0)
    except Exception:
        return None
    if not vectors:
        return None

    translate, _rotate, scale = vectors[0], vectors[1], vectors[2]
    cx, cy = float(translate[0]), float(translate[1])

    # Read through GetAttribute rather than the typed schema getters.
    #
    # `GetChildren()` hands back plain `Usd.Prim` objects, which have no
    # `GetRadiusAttr` — only the typed `UsdGeom.Cylinder` wrapper does. Calling
    # it therefore failed for every cylinder, and each one fell through to the
    # box branch and was measured by its scale, which a cylinder never sets.
    # Every table and chair leg was reported as a 1.0 x 1.0 m obstacle: the
    # live push contained a 1 m box for a 0.035 m leg, and a room furnished
    # like that has almost no floor left to drive on.
    radius = _attr(prim, "radius")

    if radius:
        # Cylinders are round; the footprint is the square that contains them.
        return (cx, cy), (float(radius) * abs(float(scale[0])),
                          float(radius) * abs(float(scale[1])))

    # Cubes carry a unit size scaled to the extents they were drawn with.
    size = _attr(prim, "size") or 1.0
    return (cx, cy), (abs(float(scale[0])) * float(size) / 2.0,
                      abs(float(scale[1])) * float(size) / 2.0)


def _attr(prim, name):
    """One USD attribute, or None if this prim has no such thing."""
    try:
        attribute = prim.GetAttribute(name)
    except Exception:
        return None
    if not attribute:
        return None
    try:
        return attribute.Get()
    except Exception:
        return None


class GeometryPublisher:
    """Tells the mapper what is in the room, whenever it changes.

    The scene is the physical world in this twin, so when someone rearranges it
    the simulated robot has to be rearranged with it. Without this the renderer
    and the world the robot drives in were two descriptions that merely looked
    alike, and they drifted apart repeatedly — the robot gliding through a sofa
    that existed only on screen was the visible symptom.

    Only posts when the layout actually differs, because this runs off the
    render loop at 60 Hz and the mapper does not need 60 identical messages a
    second. Dragging a prim changes the signature; nothing else does.
    """

    #: Re-send the layout this often even when nothing has moved.
    #
    #: Change-detection alone is not enough: the mapper holds the room in
    #: memory, so restarting its container throws the furniture away, and a
    #: publisher that only speaks when something moves never mentions it again.
    #: The room silently empties and the robot drives through where the table
    #: is on screen — the exact failure this whole mechanism exists to prevent.
    HEARTBEAT_S = 20.0

    def __init__(self, stage, url=None, interval_s=0.5, rebuild=None,
                 clear_trail=None):
        self.stage = stage
        self.url = url or MAPPER_URL
        self.interval_s = interval_s
        # Called to put the furniture back where it started, when the page asks
        # for a reset. Pull rather than push: the mapper cannot call into Kit,
        # and a scene that asks recovers on its own after either end restarts.
        self.rebuild = rebuild
        # Called when a fresh run starts, to wipe the breadcrumbs left by the
        # last one. A trail from a previous run lying over a new one shows the
        # robot having been somewhere it has not been this time.
        self.clear_trail = clear_trail
        self._reset_token = None
        self._run_id = None
        self._signature = None
        self._next_check = 0.0
        self._next_heartbeat = 0.0
        self._complained = False

    def poll(self, now):
        if now < self._next_check:
            return False
        self._next_check = now + self.interval_s

        if self._check_for_reset():
            # The furniture has just moved back to where it started, so the
            # signature below is about to differ and the new layout goes out on
            # this same tick.
            pass

        boxes = furniture_footprints(self.stage)
        signature = tuple(tuple(round(v, 3) for v in b) for b in boxes)

        due = now >= self._next_heartbeat
        if signature == self._signature and not due:
            return False

        self._signature = signature
        self._next_heartbeat = now + self.HEARTBEAT_S

        sent = self._post(boxes)
        if not sent:
            # Say it again next time rather than believing a layout the mapper
            # never received.
            self._signature = None
        return sent

    def _check_for_reset(self):
        """Has the page asked for anything to be put back?

        One request answers both questions — the room going back to how it
        started, and a fresh run needing a clean trail — because the scene
        polls this on a timer and two endpoints would be two round trips for
        one answer.
        """
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"{self.url}/api/scan/status", timeout=1.5
            ) as reply:
                status = json.loads(reply.read().decode("utf-8"))
        except Exception:
            return False

        token = status.get("scene_reset_token")
        run_id = status.get("run_id")

        first_look = self._reset_token is None and self._run_id is None
        reset_changed = token is not None and token != self._reset_token
        run_changed = run_id is not None and run_id != self._run_id
        self._reset_token, self._run_id = token, run_id

        # Not on the first look: the scene has only just built the room, and
        # tearing it down to build it again would be a visible flicker for
        # nothing.
        if first_look:
            return False

        if run_changed and self.clear_trail is not None:
            print("[room] new run — clearing the trail")
            self.clear_trail()

        if reset_changed and self.rebuild is not None:
            print("[room] resetting the furniture to where it started")
            self.rebuild()
            return True
        return False

    def _post(self, boxes):
        import urllib.request

        payload = json.dumps({
            "boxes": [
                {"min_x_m": a, "min_y_m": b, "max_x_m": c, "max_y_m": d}
                for a, b, c, d in boxes
            ]
        }).encode("utf-8")

        request = urllib.request.Request(
            f"{self.url}/api/world/geometry", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=1.5):
                pass
        except Exception as exc:
            if not self._complained:
                print(f"[room] could not push room geometry: {exc}")
                self._complained = True
            return False

        print(f"[room] pushed {len(boxes)} piece(s) of furniture to the simulator")
        return True


class HttpPose:
    """Reads the pose from the same mapper the room comes from.

    The file is faster and needs no web server, but it needs the file — and
    that turned out to be the thing nobody has. The default path is this
    machine's temp directory, the mapper runs in a container and writes to its
    own, and the scene is normally *pasted into the Script Editor*, where there
    is no opportunity to set an environment variable and no `__file__` to find
    the repository from. So the robot silently fell back to the canned demo lap
    while the room beside it was live: the two halves of one twin, again coming
    from two different places.

    `/api/state` already serves the pose, and `MAPPER_URL` is already needed
    for the room. One source answers both.
    """

    def __init__(self, url=None):
        self.url = url or MAPPER_URL
        self.latest = None
        self._complained = False

    def read(self):
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.url}/api/state", timeout=1.5) as reply:
                state = json.loads(reply.read().decode("utf-8"))
        except Exception as exc:
            if not self._complained:
                print(f"[room] pose fetch failed: {exc}")
                self._complained = True
            return self.latest

        pose = state.get("pose")
        if not pose:
            return self.latest

        # Passed through whole rather than picked apart field by field.
        #
        # Copying three keys is how this quietly lost the rest: `sequence`,
        # which is the only way to tell a robot that is running from one that
        # stopped an hour ago, and `true_x_m`, without which the scene goes
        # back to drawing the drifted estimate and the robot walks through
        # walls again. Both were added to the mapper and neither reached the
        # scene, because this function decided in advance what mattered.
        self.latest = dict(pose)
        return self.latest


class LiveOrDemo:
    """Follow the real robot whenever there is one, and keep checking.

    The source used to be chosen once, at startup: whatever was available in
    that instant was what the scene followed for the rest of the session. That
    was survivable while the mapper began driving the moment it booted.

    It stopped being survivable when the page gained a Start button. The robot
    is now deliberately idle until someone presses it, so at the moment Kit
    launches there is never a live pose — and the scene fell back to its canned
    demo lap permanently. Press Start, and the web app showed the real robot
    while Omniverse carried on driving a scripted rectangle. Not two views
    drifting apart: two different robots.

    Live means the pose is MOVING. A sequence number that has not advanced for
    a few seconds is a robot that is not publishing, which is a robot that is
    not running — the mapper holds its last pose indefinitely, so "is there a
    pose?" is answered yes forever once anything has ever run.
    """

    #: How long a frozen sequence number means the robot has stopped. Long
    #: enough to ride out a dropped request, short enough that pressing Start
    #: shows up almost at once.
    STALE_AFTER_S = 4.0

    def __init__(self, live, demo):
        self.live = live
        self.demo = demo
        self.using_demo = True
        self._sequence = None
        self._changed_at = 0.0

    def read(self):
        pose = None
        for source in self.live:
            pose = source.read()
            if pose:
                break

        now = time.monotonic()
        if pose:
            sequence = pose.get("sequence")
            if sequence != self._sequence:
                self._sequence = sequence
                self._changed_at = now

        moving = pose is not None and (now - self._changed_at) < self.STALE_AFTER_S

        if moving and self.using_demo:
            print("[room] the robot is running — following it")
            self.using_demo = False
        elif not moving and not self.using_demo:
            print("[room] the robot has stopped — showing the demo lap")
            self.using_demo = True

        return pose if moving else self.demo.read()


class DemoPose:
    """A scripted preview lap, for when nothing is publishing.

    NOT a simulation. It interpolates between waypoints and has no collision
    model, no sensors and no mapping — it exists so the scene is worth looking
    at before any of the stack is running.

    That distinction caused a real problem worth recording. The original lap
    was a simple rectangle inset 0.6 m from the walls, which in a *furnished*
    room drove straight through three things:

        bin      x 5.39-5.71, y 0.44-0.76   crossed by the bottom leg at y=0.6
        cabinet  x 5.38-5.83, y 1.85-3.35   crossed by the right leg at x=5.4
        sofa     x 0.10-2.10, y 3.28-4.13   crossed by the top leg at y=3.9

    A robot gliding through the sofa is exactly what a viewer reads as "the
    collision detection does not work", when in truth the collision model lives
    in the simulator and this path was never subject to it. The waypoints below
    keep to genuinely clear floor instead.

    For the real behaviour — the robot stopping against furniture, recording
    the contact and drawing it — run the mapper, which publishes to POSE_FILE
    and takes over from this.
    """

    def __init__(self):
        self.t = 0.0
        self.rng = random.Random(7)
        # Routed through floor that is provably clear, checked against every
        # footprint in `build_furniture` by
        # test_the_demo_lap_does_not_drive_through_the_furniture.
        #
        # It is not a full circuit of the room, and it cannot be. The table
        # with its four chairs occupies x 1.4-3.8, y 1.3-3.3 and the sofa
        # occupies the top left, leaving no continuous ring wide enough for the
        # robot. Pretending otherwise is what drove the old path through the
        # sofa. A preview that covers the open half honestly beats a lap that
        # looks complete by ignoring the furniture.
        self.waypoints = [
            (0.8, 0.9), (5.1, 0.9), (5.1, 3.9), (4.3, 3.9), (4.3, 0.9),
        ]
        self.index = 0
        self.x, self.y = self.waypoints[0]
        self.heading = 0.0

    def read(self):
        self.t += 1.0 / UPDATE_HZ
        target = self.waypoints[(self.index + 1) % len(self.waypoints)]

        dx, dy = target[0] - self.x, target[1] - self.y
        distance = math.hypot(dx, dy)
        if distance < 0.08:
            self.index = (self.index + 1) % len(self.waypoints)
        else:
            step = 0.35 / UPDATE_HZ
            self.x += dx / distance * step
            self.y += dy / distance * step
            self.heading = math.degrees(math.atan2(dy, dx)) % 360.0

        # Emitted in the POSE frame, like the real source, so the one transform
        # in RoomScene applies to both. The waypoints above stay written as
        # room positions because that is how anyone reads them against the
        # furniture, so the start offset comes back off here.
        return {
            "x_m": self.x - ROBOT_START_X_M,
            "y_m": self.y - ROBOT_START_Y_M,
            "heading_deg": self.heading,
            # Odometry tracks closely: 0.07 m measured over a circuit.
            "ideal_x_m": self.x - ROBOT_START_X_M + self.rng.gauss(0, 0.05),
            "ideal_y_m": self.y - ROBOT_START_Y_M + self.rng.gauss(0, 0.05),
            "ideal_heading_deg": self.heading,
        }


# ── Animation ───────────────────────────────────────────────────────────────


def _shortest_angle(target, current):
    diff = (target - current + 180.0) % 360.0 - 180.0
    return diff + 360.0 if diff <= -180.0 else diff


def build_camera(stage):
    """A camera framing both rooms, made the active viewport view.

    Without this the viewport keeps whatever default camera the app started
    with, which knows nothing about where this scene was built and generally
    opens inside the floor slab — a flat grey frame that reads as "nothing
    rendered".

    Framing. The scene spans x 0 to ROOM_W on the left and, with the measured
    room, out to about 14 m; the camera therefore sits back on -y, high enough
    to look down into both rooms at once rather than at the wall between the
    viewer and them.
    """
    # Centre of everything worth seeing.
    span_x = ROOM_W
    centre_x = span_x / 2.0
    centre_y = ROOM_H / 2.0

    eye_y = -14.0
    eye_z = 10.0

    camera = UsdGeom.Camera.Define(stage, Sdf.Path(f"{ROOT}/Camera"))

    # A USD camera looks down its own -Z with +Y up. Rotating +90° about X
    # aims it along world +Y — level, in a Z-up stage. Taking that back by the
    # pitch angle tilts it down onto the floor.
    pitch_deg = math.degrees(math.atan2(eye_z - 0.6, centre_y - eye_y))
    UsdGeom.XformCommonAPI(camera).SetTranslate(Gf.Vec3d(centre_x, eye_y, eye_z))
    UsdGeom.XformCommonAPI(camera).SetRotate(Gf.Vec3f(90.0 - pitch_deg, 0.0, 0.0))

    # Wider than the 50 mm default, which at this distance frames about a third
    # of the scene and crops the measured room off the side.
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))

    # Point the viewport at it. Wrapped because the scene is also loaded by the
    # test harness and by Kit apps with no viewport at all, and failing to
    # switch camera is not a reason to lose the room.
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = f"{ROOT}/Camera"
            print(f"[room] viewport camera set to {ROOT}/Camera")
    except Exception as exc:
        print(f"[room] could not set the viewport camera ({exc}).")
        print(f"       Select {ROOT}/Camera in the Stage tree and press F.")


# ── Physics ─────────────────────────────────────────────────────────────────


def add_collider(stage, path, approximation="convexHull"):
    """Make one prim solid.

    Purely a USD authoring step: it tags the geometry so a physics engine will
    collide against it. Nothing here simulates anything, and adding it does not
    make the robot stop — see `make_robot_physical` for why that is a separate
    question.
    """
    prim = stage.GetPrimAtPath(Sdf.Path(path))
    if not prim:
        return False
    try:
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr(approximation)
        return True
    except Exception:
        return False


#: Groups that are drawings rather than objects, and must NOT become solid.
#:
#: Every one of these would be a bug as a collider. The trail is a breadcrumb
#: per 0.15 m driven — nine hundred of them, lying on the floor, each one
#: something for a dynamic prop to catch on. The odometry ghost is where the
#: filter *thinks* the robot is, which is by definition not where anything is.
#: Lights and the camera are not in the room at all.
NOT_PHYSICAL = ("Trail", "RobotOdometry", "RobotRssi", "Lighting", "Camera",
                "Measured")


def make_solid(stage):
    """Give everything in the room a collider — walls, furniture, beacons.

    Walks the whole tree rather than the direct children of two groups. The
    old version did the latter and missed the beacons entirely: thirteen prims
    mounted on the walls, physically present, that anything dynamic passed
    straight through. It would also have silently skipped any furniture that
    was ever nested one level deeper, which is the kind of gap that appears
    later and looks like a physics bug.

    What this buys and what it does not
    -----------------------------------
    It makes the room *physically real* to PhysX: anything dynamic dropped into
    the scene rests on the floor and stops at the sofa, and Isaac Sim can drive
    the robot against it for real.

    It does NOT by itself stop the blue robot marker, because that marker is
    not simulated — it is placed each frame at the pose the mapper reports.
    Collision for it is decided in the mapper's own simulator, and the marker
    only mirrors the result. Making it dynamic instead would mean the twin no
    longer shows where the robot actually is, which is the one job it has.
    """
    root = stage.GetPrimAtPath(Sdf.Path(ROOT))
    if not root:
        return 0

    solid = skipped = 0

    def walk(prim):
        nonlocal solid, skipped
        for child in prim.GetChildren():
            path = child.GetPath().pathString
            name = path[len(ROOT) + 1:].split("/")[0]
            if name in NOT_PHYSICAL:
                skipped += 1
                continue
            # A grouping Xform has no geometry to collide with; its children do.
            if child.GetChildren():
                walk(child)
                continue
            if add_collider(stage, path):
                solid += 1

    walk(root)
    print(f"[room] {solid} colliders — the room is solid to physics "
          f"({skipped} drawing(s) left alone)")
    return solid


def make_robot_physical(stage, path=None):
    """Make the robot a KINEMATIC body: it pushes, and is not pushed.

    Kinematic rather than dynamic on purpose. A dynamic robot would be moved by
    PhysX, and would then disagree with the pose the mapper reports — the twin
    would be showing a robot that does not exist. Kinematic keeps the mapper
    authoritative while still letting the robot shove dynamic props around and
    generate contact events that Isaac Sim can report.
    """
    path = path or f"{ROOT}/RobotTrue"
    prim = stage.GetPrimAtPath(Sdf.Path(path))
    if not prim:
        return False
    try:
        body = UsdPhysics.RigidBodyAPI.Apply(prim)
        body.CreateKinematicEnabledAttr(True)
        for child in prim.GetChildren():
            add_collider(stage, child.GetPath().pathString)
        return True
    except Exception as exc:
        print(f"[room] could not make the robot physical: {exc}")
        return False


# ── The room the robot drew ─────────────────────────────────────────────────


class RoomScene:
    def __init__(self, stage, source, measured=None, geometry=None):
        self.stage = stage
        self.source = source
        self.measured = measured
        # Pushes the room's furniture to the simulator whenever it is moved,
        # so the robot drives in the room you can see. See GeometryPublisher.
        self.geometry = geometry
        self.subscription = None

        self.x = self.y = self.heading = 0.0
        self.ox = self.oy = 0.0
        self.rx = self.ry = 0.0
        self.trail_index = 0
        self._last_trail = (0.0, 0.0)
        self._frames = 0

    def start(self):
        app = omni.kit.app.get_app()
        self.subscription = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="roomscan_3d"
        )
        print("[room] running — call stop_room() to end")

    def _place(self, path, x, y, heading=None, z=None):
        prim = self.stage.GetPrimAtPath(path)
        if not prim:
            return
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate(Gf.Vec3d(x, y, z if z is not None else 0.0))
        if heading is not None:
            api.SetRotate(Gf.Vec3f(0.0, 0.0, heading))

    def _on_update(self, event):
        now = time.monotonic()

        # Polled before the early return below, so the measured room keeps
        # updating even when no pose is arriving — the map is still being
        # refined by a robot this scene cannot see.
        if self.measured is not None:
            self.measured.poll(now)

        # And the room the robot drives in follows the room on screen. Move a
        # table in the viewport and the simulated robot meets it in its new
        # place; this is what makes the 3D scene the physical world rather
        # than a picture of one. Also before the early return: rearranging the
        # furniture must work whether or not a scan is currently running.
        if self.geometry is not None:
            self.geometry.poll(now)

        pose = self.source.read()
        if not pose:
            return

        # Pose frame -> room frame. See ROBOT_START_X_M: pose coordinates are
        # relative to wherever the robot was standing when the run began, and
        # this room is laid out from a corner.
        # The solid robot is where it ACTUALLY is; the green ghost is where
        # odometry thinks it is. That is what this scene has always said it
        # drew, and for a long time it drew the estimate for both.
        #
        # It showed. This room is the true room, drawn at fixed coordinates, so
        # a robot placed by an estimate that has drifted a few hundred metres'
        # worth appears outside it — walking through a wall it was nowhere near.
        # That reads as collision being broken, when in fact the simulator stops
        # the robot dead at 4.39 m against a wall at 4.50, exactly its own
        # radius short.
        #
        # `true_*` is absent with real hardware, where no such number exists,
        # and the estimate is then the best the scene can do.
        has_truth = "true_x_m" in pose
        tx = float(pose.get("true_x_m", pose.get("x_m", 0.0))) + ROBOT_START_X_M
        ty = float(pose.get("true_y_m", pose.get("y_m", 0.0))) + ROBOT_START_Y_M
        th = float(pose.get("true_heading_deg", pose.get("heading_deg", 0.0)))

        # Chase rather than snap: telemetry arrives at ~10 Hz, Kit renders at
        # 60, and snapping shows as a visible stutter.
        self.x += (tx - self.x) * SMOOTHING
        self.y += (ty - self.y) * SMOOTHING
        self.heading = (self.heading + _shortest_angle(th, self.heading) * SMOOTHING) % 360.0
        self._place(f"{ROOT}/RobotTrue", self.x, self.y, self.heading)

        # The filter's own estimate, beside the truth. The gap between the two
        # IS the drift, and watching it open up over a long run is the single
        # most useful thing this scene shows — it is the reason the map comes
        # back a few per cent large. Nothing to draw when the two are the same
        # number, which is the case with real hardware.
        if SHOW_ODOMETRY_GHOST and has_truth:
            gx = float(pose.get("x_m", 0.0)) + ROBOT_START_X_M
            gy = float(pose.get("y_m", 0.0)) + ROBOT_START_Y_M
            self.ox += (gx - self.ox) * SMOOTHING
            self.oy += (gy - self.oy) * SMOOTHING
            self._place(f"{ROOT}/RobotOdometry", self.ox, self.oy,
                        float(pose.get("heading_deg", th)))


        self._drop_trail()

        self._frames += 1
        if self._frames % 300 == 0:
            rssi_gap = math.hypot(self.rx - self.x, self.ry - self.y)
            odom_gap = math.hypot(self.ox - self.x, self.oy - self.y)
            print(f"[room] robot ({self.x:.2f}, {self.y:.2f})  "
                  f"odometry off by {odom_gap:.2f} m  RSSI off by {rssi_gap:.2f} m")

    def _drop_trail(self, spacing=0.15):
        if math.hypot(self.x - self._last_trail[0], self.y - self._last_trail[1]) < spacing:
            return

        # Recycled by index, so the trail is a fixed number of prims that roll
        # rather than a list that grows for ever. See MAX_TRAIL_DOTS.
        path = f"{ROOT}/Trail/Dot_{self.trail_index % MAX_TRAIL_DOTS}"
        existing = self.stage.GetPrimAtPath(Sdf.Path(path))
        if existing:
            UsdGeom.XformCommonAPI(existing).SetTranslate(
                Gf.Vec3d(self.x, self.y, 0.008)
            )
        else:
            box(self.stage, path, (self.x, self.y, 0.008),
                (0.035, 0.035, 0.008), COL_TRAIL)

        self.trail_index += 1
        self._last_trail = (self.x, self.y)

    def clear_trail(self):
        """Wipe the breadcrumbs, for a run that has not happened yet.

        A trail from the previous run lying over a new one is the same kind of
        lie as furniture that is not there: it shows the robot having been
        somewhere it has not been this time. The mapper clears its own map on
        every fresh start, and this is the 3D half of that.
        """
        self.stage.RemovePrim(Sdf.Path(f"{ROOT}/Trail"))
        UsdGeom.Xform.Define(self.stage, Sdf.Path(f"{ROOT}/Trail"))
        self.trail_index = 0
        self._last_trail = (self.x, self.y)

    def stop(self):
        if self.subscription is not None:
            self.subscription.unsubscribe()
            self.subscription = None


# ── Entry points ────────────────────────────────────────────────────────────

_scene = None


def build_room():
    """Build the scene without animating it."""
    stage = omni.usd.get_context().get_stage()

    # Z is up.
    #
    # Everything below places height on Z — the floor at z≈0, walls rising to
    # z=2.4 — which is the robotics convention and matches the mapper's
    # coordinates. A Kit stage defaults to Y-up, and in a Y-up stage this room
    # is built lying on its side: the viewport opens inside a wall and shows a
    # flat grey field with the scene apparently missing. It is not missing, it
    # is edge-on.
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    UsdGeom.Xform.Define(stage, Sdf.Path(ROOT))

    build_shell(stage)

    # The furniture is always drawn, and the simulator is then told about it.
    #
    # This used to ask the mapper what was in the room and draw only that,
    # which was the right answer when the scene was a *view* of the world: a
    # view that decorates itself is a lie, and it was one — a table, four
    # chairs and a sofa rendered in a room the robot knew to be bare, which it
    # drove straight through.
    #
    # The direction is now the other way round. This scene IS the world, so
    # what is drawn here is what exists, and `GeometryPublisher` pushes it to
    # the simulator on the first frame. Both views agree as before, but they
    # agree on the room you can reach into and rearrange rather than on one
    # hardcoded somewhere else. Drawing nothing would leave nothing to drag.
    build_furniture(stage)

    build_beacons(stage)
    build_lighting(stage)
    build_all_robots(stage)
    build_camera(stage)

    if ENABLE_PHYSICS:
        make_solid(stage)
        make_robot_physical(stage)

    # What the two robots mean.
    #
    # "Why are there two?" is the first thing anyone asks, and the legend used
    # to answer a different question: it described an orange RSSI marker that
    # has been deleted, and called this room LEFT of a measured room that no
    # longer stands beside it. Naming the markers is not enough — the useful
    # part is what the distance between them is.
    print()
    print("  Two robots, and the distance between them is the point:")
    print("    BLUE   where the robot actually is")
    print("    GREEN  where its own dead reckoning thinks it is")
    print()
    print("  The gap between them IS the drift. It grows with distance driven,")
    print("  and it is why the measured room comes back larger than the real")
    print("  one — the 2D map is built from the green robot's idea of where it")
    print("  went. http://localhost:8080 prints the same gap in metres.")
    print()
    print("    MAGENTA  the four BLE beacons, on the walls at 2.1 m")
    print()
    return stage


def run_room():
    global _scene
    stop_room()

    stage = build_room()

    measured = None

    # The file first — cheaper, and needs no web server — then the mapper over
    # HTTP, which is what actually works when this file is pasted into the
    # Script Editor with nothing set. Only then the canned lap.
    # Both live sources, tried in order, with the demo lap only while neither
    # has a moving robot to show. Checked every frame rather than once at
    # startup — the robot is idle until somebody presses Start, and deciding at
    # launch meant the scene never noticed them pressing it.
    source = LiveOrDemo([FilePose(), HttpPose()], DemoPose())
    print(f"[room] watching for the robot — {POSE_FILE}, then {MAPPER_URL}/api/state")

    # The scene is the physical world: move the furniture here and the robot
    # meets it where you put it. Pushed on a timer rather than on an edit
    # notification, because dragging a prim in the viewport does not raise one
    # this script can hook without pulling in more of Kit than it should.
    def rebuild_furniture():
        """Delete the furniture and draw it again where it started."""
        stage.RemovePrim(Sdf.Path(f"{ROOT}/Furniture"))
        build_furniture(stage)
        if ENABLE_PHYSICS:
            make_solid(stage)

    scene_holder = {}

    def clear_trail():
        scene = scene_holder.get("scene")
        if scene is not None:
            scene.clear_trail()

    publisher = GeometryPublisher(
        stage, rebuild=rebuild_furniture, clear_trail=clear_trail
    )
    pieces = furniture_footprints(stage)
    print(
        f"[room] {len(pieces)} solid piece(s) of furniture — drag them in the "
        f"viewport and the robot will find them there"
    )

    _scene = RoomScene(stage, source, measured, geometry=publisher)
    scene_holder["scene"] = _scene
    _scene.start()


def stop_room():
    global _scene
    if _scene is not None:
        _scene.stop()
        _scene = None


run_room()
