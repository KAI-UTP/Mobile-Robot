"""A furnished 3D room in Omniverse, with the robot and its BLE beacons.

    Window > Script Editor, paste this whole file, Ctrl+Enter.

Builds a realistic room rather than an empty box: walls with a doorway and a
window, a table and chairs, a sofa, a cabinet, a rug — and the four Bluetooth
beacons in the corners, so the positioning setup is visible rather than
implied.

Three robots are drawn, and the difference between them is the point
-------------------------------------------------------------------
* **Solid blue** — where the robot actually is (ground truth).
* **Green outline** — where wheel odometry thinks it is. It tracks closely.
* **Orange sphere** — where Bluetooth trilateration thinks it is. It wanders
  metres away, all the time.

Measured over a full circuit of this room, odometry averages 0.07 m of error
and RSSI averages 2.71 m — about 60 % of the room's short dimension. Seeing
the orange marker drift across the sofa while the green one stays on the robot
explains, in one glance, why the map is built from odometry and Bluetooth is
used only for room-level presence.

Furniture is not decoration
---------------------------
It attenuates radio and blocks ultrasonic pulses, which is exactly why real
rooms are harder than empty ones. The layout below is deliberately awkward:
a table in the middle of the floor and a sofa across one corner.

Data source
-----------
Reads the same pose file `services/twin-control` writes, so no packages need
installing inside Kit. Set `USE_MQTT = True` if paho is available.
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
USE_MQTT = False
MQTT_HOST = "localhost"

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

# A pose file older than this is ignored in favour of the demo lap. Twin-control
# rewrites it several times a second while it is running, so anything this stale
# is left over from a previous session — and silently animating a robot from
# yesterday's file, frozen whereever it stopped, looks exactly like a bug in the
# scene.
POSE_STALE_AFTER_S = 15.0

SHOW_ODOMETRY_GHOST = True

# The orange ball showing where Bluetooth thinks the robot is, and the 2.71 m
# disc showing how sure it is.
#
# OFF, because BLE fusion is switched off. Measuring it showed it made the pose
# worse — 0.51 m of error became 1.42 m — so the filter ignores it, and drawing
# a marker for a sensor that contributes nothing implies it is being used.
#
# The disc was also the single most visually dominant thing in the room: 2.71 m
# of radius is 5.42 m across, which covers almost the whole floor of a 6 x 4.5 m
# room in translucent orange. Honest about BLE's uncertainty, and completely in
# the way.
SHOW_RSSI_MARKER = False
SHOW_BEACON_RANGES = False   # rings showing inferred distance; busy but instructive

# Give the walls and furniture colliders, and the robot a kinematic body.
#
# This is what makes the room solid to PhysX, so anything dynamic collides with
# it and Isaac Sim can drive the robot against it for real. It does NOT make
# the blue marker stop on its own — that marker mirrors the pose the mapper
# reports, and collision for it is decided in the mapper's simulator. See
# `make_robot_physical`.
ENABLE_PHYSICS = True

# Build the room the ROBOT drew, standing beside the real one — the two-screen
# comparison, in 3D, in Omniverse rather than in a browser canvas. Read live
# from the mapper over plain HTTP, so nothing needs installing inside Kit.
# Build the room the ROBOT drew, standing beside the real one.
#
# OFF. Omniverse's job is the physical room; the measured floor plan belongs on
# the 2D map at http://localhost:8080, which reads it far better — a flat plan
# is simply a better way to look at a floor plan than an extruded outline seen
# in perspective.
SHOW_MEASURED = False

# Which mapper the measured room is read from.
#
# This MUST be the same mapper that is writing POSE_FILE, or the two halves of
# the scene show two different robots in two different rooms and the comparison
# means nothing. It happened: the robot on the left followed a furnished-room
# mapper on port 8082 while the room on the right was read from a
# rectangular-room mapper on 8080, reporting 27.97 m2 and no obstacles beside a
# robot that had measured 23.37 m2 and found one.
#
# Settable from the environment so a mapper on a non-default port can be
# followed without editing this file.
MAPPER_URL = os.environ.get("MAPPER_URL", "http://localhost:8080")
MEASURED_GAP_M = 2.0         # clear floor between the two rooms
MEASURED_POLL_S = 2.0        # the outline changes far more slowly than the pose

# Measured mean error of RSSI trilateration in this room, from
# tests/test_rssi_accuracy.py. Used to animate the marker realistically when
# running without a live RSSI feed.
RSSI_TYPICAL_ERROR_M = 2.71

UPDATE_HZ = 60.0
SMOOTHING = 0.25

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
COL_RSSI       = (0.95, 0.55, 0.15)
COL_BEACON     = (0.85, 0.25, 0.75)
COL_TRAIL      = (0.95, 0.60, 0.20)

# The measured half. Deliberately cooler and flatter than the physical room:
# one side is a place, the other is a measurement, and they should not be
# mistaken for each other at a glance.
COL_MEAS_PAD   = (0.12, 0.14, 0.18)
COL_MEAS_WALL  = (0.25, 0.73, 0.31)   # green once the boundary closes
COL_MEAS_OPEN  = (0.82, 0.60, 0.13)   # amber while it is still open
COL_MEAS_BLOCK = (0.97, 0.32, 0.29)   # blocked floor: furniture in the way

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
    """Floor, walls, a doorway and a window.

    The doorway is built as two wall segments with a gap rather than a solid
    wall, because an opening is what makes the room realistic for both the
    range sensors and the radio model — and it is where a real robot would
    drive out and get lost.
    """
    UsdGeom.Xform.Define(stage, Sdf.Path(f"{ROOT}/Room"))

    box(stage, f"{ROOT}/Room/Floor",
        (ROOM_W / 2, ROOM_H / 2, -0.02), (ROOM_W, ROOM_H, 0.04), COL_FLOOR)

    h = WALL_HEIGHT / 2
    t = WALL_THICK

    # South wall, split around a 0.9 m doorway.
    door_x, door_w = 4.3, 0.9
    left_w = door_x - door_w / 2
    box(stage, f"{ROOT}/Room/WallS_left",
        (left_w / 2, 0, h), (left_w, t, WALL_HEIGHT), COL_WALL)
    right_start = door_x + door_w / 2
    right_w = ROOM_W - right_start
    box(stage, f"{ROOT}/Room/WallS_right",
        (right_start + right_w / 2, 0, h), (right_w, t, WALL_HEIGHT), COL_WALL)
    # Lintel above the opening.
    box(stage, f"{ROOT}/Room/WallS_lintel",
        (door_x, 0, 2.15), (door_w, t, 0.5), COL_WALL)
    # The door itself, swung open into the room.
    box(stage, f"{ROOT}/Room/Door",
        (door_x + 0.42, 0.42, 1.0), (0.05, 0.85, 2.0), COL_DOOR)

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

    print(f"[room] shell built: {ROOM_W} x {ROOM_H} x {WALL_HEIGHT} m, one doorway")


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

    if SHOW_RSSI_MARKER:
        UsdGeom.Xform.Define(stage, Sdf.Path(f"{ROOT}/RssiEstimate"))
        sphere(stage, f"{ROOT}/RssiEstimate/Marker", (0, 0, 0.35), 0.16, COL_RSSI, opacity=0.75)
        # A ring at the typical error radius, so the uncertainty is a size on
        # the floor rather than a number in a log.
        cylinder(stage, f"{ROOT}/RssiEstimate/ErrorRing",
                 (0, 0, 0.01), RSSI_TYPICAL_ERROR_M, 0.012, COL_RSSI, opacity=0.10)

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

        self.latest = {
            "x_m": pose["x_m"],
            "y_m": pose["y_m"],
            "heading_deg": pose["heading_deg"],
            "timestamp": pose.get("timestamp"),
        }
        return self.latest


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
            # RSSI wanders by metres. 2.71 m mean error, measured.
            "rssi_x_m": (
                self.x - ROBOT_START_X_M
                + self.rng.gauss(0, RSSI_TYPICAL_ERROR_M * 0.8)
            ),
            "rssi_y_m": (
                self.y - ROBOT_START_Y_M
                + self.rng.gauss(0, RSSI_TYPICAL_ERROR_M * 0.8)
            ),
        }


# ── Animation ───────────────────────────────────────────────────────────────


def _shortest_angle(target, current):
    diff = (target - current + 180.0) % 360.0 - 180.0
    return diff + 360.0 if diff <= -180.0 else diff


def room_has_furniture() -> bool:
    """Does the room the mapper is simulating actually contain furniture?

    Asked rather than assumed, because assuming is how this scene ended up
    showing a table, four chairs and a sofa in a room the robot knew to be
    empty. The robot drove straight through all of it, the 2D map came back a
    bare rectangle, and three views disagreed about the same room.

    Falls back to drawing the furniture when the mapper cannot be reached, so
    the scene is still worth looking at on its own — but says so, because a
    silent fallback is what made the original mismatch invisible.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(f"{MAPPER_URL}/api/world", timeout=1.5) as reply:
            world = json.loads(reply.read().decode("utf-8"))
    except Exception:
        print("[room] no mapper reachable — drawing the demo furniture")
        return True

    furniture = [b for b in world.get("boxes", []) if b.get("kind") == "furniture"]
    print(f"[room] mapper reports '{world.get('name')}' with {len(furniture)} furniture pieces")
    return bool(furniture)


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
    span_x = ROOM_W + MEASURED_GAP_M + ROOM_W if SHOW_MEASURED else ROOM_W
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


def make_solid(stage):
    """Give every wall and every piece of furniture a collider.

    What this buys and what it does not
    -----------------------------------
    It makes the room *physically real* to PhysX: anything dynamic dropped into
    the scene will rest on the floor and stop at the sofa, and Isaac Sim can
    drive the robot against it for real.

    It does NOT by itself stop the blue robot marker, because that marker is
    not simulated — it is placed each frame at the pose the mapper reports.
    Collision for it is decided in the mapper's own simulator, and the marker
    only ever mirrors the result. Making it a dynamic body instead would mean
    the twin no longer shows where the robot actually is, which is the one job
    it has.
    """
    solid = 0
    for parent in (f"{ROOT}/Room", f"{ROOT}/Furniture"):
        root = stage.GetPrimAtPath(Sdf.Path(parent))
        if not root:
            continue
        for prim in root.GetChildren():
            if add_collider(stage, prim.GetPath().pathString):
                solid += 1
    print(f"[room] {solid} colliders — walls and furniture are solid to physics")
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


class MeasuredRoom:
    """The robot's own map, built beside the real room.

    This is the second screen. The left room is the place; this one is what the
    robot worked out from range readings and dead reckoning, and putting them
    side by side in the same 3D view is the entire claim of a digital twin —
    you can see the error rather than read it off a number.

    Read over plain HTTP with `urllib`, which is in the standard library, so
    Kit needs no packages installed. The outline changes far more slowly than
    the pose, so it is polled every couple of seconds rather than every frame.

    Placed by its own bounding box, not by its coordinates
    -----------------------------------------------------
    The measured polygon lives in the pose estimate's frame, whose origin is
    wherever the robot happened to start and whose axes are however it happened
    to be facing. Those numbers are meaningless to place a model by. What is
    meaningful is the SHAPE, so the outline is translated to sit on its own pad
    and the two rooms are compared as shapes. `/compare` in the web UI scores
    that properly, with rotation and IoU; this is the version you can walk
    around.
    """

    def __init__(self, stage, origin_x, origin_y):
        self.stage = stage
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.root = f"{ROOT}/Measured"
        self._signature = None
        self._next_poll = 0.0
        self.summary = "waiting for the mapper"

        UsdGeom.Xform.Define(stage, Sdf.Path(self.root))
        self._build_pad()

    # ── Data ──────────────────────────────────────────────────────────────

    def poll(self, now):
        """Fetch the room if it is time. Returns True if the scene changed."""
        if now < self._next_poll:
            return False
        self._next_poll = now + MEASURED_POLL_S

        room = self._fetch()
        if room is None:
            return False

        # Rebuilding every poll would delete and recreate a hundred prims two
        # seconds apart for no reason, and the viewport flickers while it
        # happens. Only a room that actually changed is worth redrawing.
        signature = (
            round(room.get("area_m2", 0.0), 3),
            len(room.get("polygon", [])),
            len(room.get("obstacles", [])),
            round(room.get("blocked_area_m2", 0.0), 3),
            bool(room.get("is_closed")),
        )
        if signature == self._signature:
            return False
        self._signature = signature

        self.rebuild(room)
        return True

    def _fetch(self):
        import urllib.request

        try:
            with urllib.request.urlopen(f"{MAPPER_URL}/api/room", timeout=1.5) as reply:
                if reply.status != 200:
                    return None
                return json.loads(reply.read().decode("utf-8"))
        except Exception:
            # No mapper running, or no room measured yet. Neither is an error
            # worth printing every two seconds — the left room is still worth
            # looking at on its own.
            return None

    # ── Geometry ──────────────────────────────────────────────────────────

    def _build_pad(self):
        """A dark slab the measurement sits on, so it reads as a drawing."""
        box(
            self.stage,
            f"{self.root}/Pad",
            (self.origin_x + ROOM_W / 2, self.origin_y + ROOM_H / 2, -0.03),
            (ROOM_W + 0.6, ROOM_H + 0.6, 0.06),
            COL_MEAS_PAD,
        )

    def _clear(self):
        for name in ("Outline", "Blocked"):
            path = Sdf.Path(f"{self.root}/{name}")
            if self.stage.GetPrimAtPath(path):
                self.stage.RemovePrim(path)
        UsdGeom.Xform.Define(self.stage, Sdf.Path(f"{self.root}/Outline"))
        UsdGeom.Xform.Define(self.stage, Sdf.Path(f"{self.root}/Blocked"))

    def rebuild(self, room):
        polygon = room.get("polygon") or []
        if len(polygon) < 3:
            return

        self._clear()

        xs = [p["x_m"] for p in polygon]
        ys = [p["y_m"] for p in polygon]
        min_x, min_y = min(xs), min(ys)

        def place(x, y):
            """Pose frame -> this pad, centred on the pad rather than dumped in
            a corner."""
            width, height = max(xs) - min_x, max(ys) - min_y
            return (
                self.origin_x + (x - min_x) + (ROOM_W - width) / 2,
                self.origin_y + (y - min_y) + (ROOM_H - height) / 2,
            )

        closed = bool(room.get("is_closed"))
        colour = COL_MEAS_WALL if closed else COL_MEAS_OPEN

        # The outline, extruded. Drawn as a low wall rather than a flat line so
        # it reads as a room from the same viewing angle as the real one.
        for index in range(len(polygon)):
            ax, ay = place(polygon[index]["x_m"], polygon[index]["y_m"])
            nxt = polygon[(index + 1) % len(polygon)]
            bx, by = place(nxt["x_m"], nxt["y_m"])

            length = math.hypot(bx - ax, by - ay)
            if length < 1e-6:
                continue

            prim = box(
                self.stage,
                f"{self.root}/Outline/Wall_{index}",
                ((ax + bx) / 2, (ay + by) / 2, 0.30),
                (length, 0.06, 0.60),
                colour,
                opacity=0.85,
            )
            UsdGeom.XformCommonAPI(prim).SetRotate(
                Gf.Vec3f(0.0, 0.0, math.degrees(math.atan2(by - ay, bx - ax)))
            )

        # Blocked floor. Low translucent slabs, not furniture-shaped models:
        # the robot measured a footprint on the floor, not a table, and drawing
        # a table would claim knowledge it does not have. Compare them against
        # the real furniture in the left room by eye.
        for index, obstacle in enumerate(room.get("obstacles") or []):
            cx, cy = place(obstacle["centre_x_m"], obstacle["centre_y_m"])
            width = max(0.05, obstacle["max_x_m"] - obstacle["min_x_m"])
            depth = max(0.05, obstacle["max_y_m"] - obstacle["min_y_m"])
            box(
                self.stage,
                f"{self.root}/Blocked/Area_{index}",
                (cx, cy, 0.09),
                (width, depth, 0.18),
                COL_MEAS_BLOCK,
                opacity=0.55,
            )

        area = room.get("area_m2", 0.0)
        blocked = room.get("blocked_area_m2", 0.0)
        self.summary = (
            f"{area:.2f} m2 floor, {blocked:.2f} m2 blocked by "
            f"{len(room.get('obstacles') or [])} obstacle(s), "
            f"{max(0.0, area - blocked):.2f} m2 usable, "
            f"{'closed' if closed else 'OPEN'}"
        )
        print(f"[room] measured: {self.summary}")


class RoomScene:
    def __init__(self, stage, source, measured=None):
        self.stage = stage
        self.source = source
        self.measured = measured
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
        # Polled before the early return below, so the measured room keeps
        # updating even when no pose is arriving — the map is still being
        # refined by a robot this scene cannot see.
        if self.measured is not None:
            self.measured.poll(time.monotonic())

        pose = self.source.read()
        if not pose:
            return

        # Pose frame -> room frame. See ROBOT_START_X_M: pose coordinates are
        # relative to wherever the robot was standing when the run began, and
        # this room is laid out from a corner.
        tx = float(pose.get("x_m", 0.0)) + ROBOT_START_X_M
        ty = float(pose.get("y_m", 0.0)) + ROBOT_START_Y_M
        th = float(pose.get("heading_deg", 0.0))

        # Chase rather than snap: telemetry arrives at ~10 Hz, Kit renders at
        # 60, and snapping shows as a visible stutter.
        self.x += (tx - self.x) * SMOOTHING
        self.y += (ty - self.y) * SMOOTHING
        self.heading = (self.heading + _shortest_angle(th, self.heading) * SMOOTHING) % 360.0
        self._place(f"{ROOT}/RobotTrue", self.x, self.y, self.heading)

        if SHOW_ODOMETRY_GHOST:
            gx = float(pose.get("ideal_x_m", tx - ROBOT_START_X_M)) + ROBOT_START_X_M
            gy = float(pose.get("ideal_y_m", ty - ROBOT_START_Y_M)) + ROBOT_START_Y_M
            self.ox += (gx - self.ox) * SMOOTHING
            self.oy += (gy - self.oy) * SMOOTHING
            self._place(f"{ROOT}/RobotOdometry", self.ox, self.oy,
                        float(pose.get("ideal_heading_deg", th)))

        if SHOW_RSSI_MARKER and "rssi_x_m" in pose:
            # Slower smoothing: RSSI genuinely jumps, and showing that is the
            # point of the marker.
            self.rx += (float(pose["rssi_x_m"]) + ROBOT_START_X_M - self.rx) * 0.08
            self.ry += (float(pose["rssi_y_m"]) + ROBOT_START_Y_M - self.ry) * 0.08
            self._place(f"{ROOT}/RssiEstimate", self.rx, self.ry)

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
        box(self.stage, f"{ROOT}/Trail/Dot_{self.trail_index}",
            (self.x, self.y, 0.008), (0.035, 0.035, 0.008), COL_TRAIL)
        self.trail_index += 1
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
    if room_has_furniture():
        build_furniture(stage)
    else:
        print("[room] the mapper's room is empty — drawing no furniture")
    build_beacons(stage)
    build_lighting(stage)
    build_all_robots(stage)
    build_camera(stage)

    if ENABLE_PHYSICS:
        make_solid(stage)
        make_robot_physical(stage)

    print()
    print("  LEFT — the room that exists")
    print("    blue     = where the robot is")
    print("    green    = wheel odometry     (0.07 m mean error, measured)")
    print("    orange   = Bluetooth RSSI     (2.71 m mean error, measured)")
    print("    magenta  = BLE beacons")
    if SHOW_MEASURED:
        print()
        print("  RIGHT — the room the robot drew")
        print("    green    = outline, boundary closed")
        print("    amber    = outline, boundary still open")
        print("    red      = blocked floor: furniture it cannot drive over")
    print()
    return stage


def run_room():
    global _scene
    stop_room()

    stage = build_room()

    measured = None
    if SHOW_MEASURED:
        # Placed to the right of the real room, with clear floor between, so
        # one orbit of the viewport takes in both.
        measured = MeasuredRoom(stage, ROOM_W + MEASURED_GAP_M, 0.0)
        if not measured.poll(0.0):
            print(f"[room] no room from the mapper at {MAPPER_URL} yet")
            print("       start services/mapper/main.py --source sim")

    # The file first — cheaper, and needs no web server — then the mapper over
    # HTTP, which is what actually works when this file is pasted into the
    # Script Editor with nothing set. Only then the canned lap.
    source = FilePose()
    if source.read() is not None:
        print(f"[room] following the live robot from {POSE_FILE}")
    else:
        source = HttpPose()
        if source.read() is not None:
            print(f"[room] following the live robot from {MAPPER_URL}/api/state")
        else:
            print("[room] no live pose found — running the built-in demo lap")
            print("       start services/mapper/main.py --source sim")
            source = DemoPose()

    _scene = RoomScene(stage, source, measured)
    _scene.start()


def stop_room():
    global _scene
    if _scene is not None:
        _scene.stop()
        _scene = None


run_room()
