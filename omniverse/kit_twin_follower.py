"""Make the Omniverse robot follow the real one.

    Window > Script Editor, paste this whole file, Ctrl+Enter.

This is the display half of the digital twin: `services/twin-control` drives
the physical robot and publishes where it actually is; this script moves the
USD prim to match. Type an instruction, the real robot moves, and the
Omniverse robot moves with it.

Same idea as the SmartClean Twin project's `live_update.py`, with two changes:

* **MQTT instead of polling InfluxDB.** Pose arrives as it is produced rather
  than being re-queried on a timer, so the twin tracks in real time instead of
  lagging by the poll interval.
* **A ghost prim showing the commanded pose.** Two robots are drawn: a solid
  one where the machine actually is, and a translucent one where perfect,
  slip-free execution would have put it. The gap between them *is* the
  sim-to-real gap, visible rather than buried in a log.

Prerequisite — run once in the Script Editor:

    import omni.kit.pipapi
    omni.kit.pipapi.install("paho-mqtt", module="paho")

If that fails (some Kit builds sandbox pip), set USE_MQTT = False below and
the script will read the pose from a file the twin controller writes instead.
"""

import json
import math
import os
import tempfile

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, UsdGeom, Vt

# ── Configuration ───────────────────────────────────────────────────────────

USE_MQTT = True

MQTT_HOST = "localhost"
MQTT_PORT = 1883
ROBOT_ID = "MR3W01"
POSE_TOPIC = f"roommapper/{ROBOT_ID}/pose"

# Fallback when MQTT is unavailable in Kit's Python.
POSE_FILE = os.path.join(tempfile.gettempdir(), f"roommapper_{ROBOT_ID}_pose.json")

ROBOT_PATH = "/World/TwinRobot"
GHOST_PATH = "/World/TwinGhost"
TRAIL_PATH = "/World/TwinTrail"

SHOW_GHOST = True
SHOW_TRAIL = True

# How quickly the prim chases the reported pose, per frame. 1.0 snaps exactly.
# Slightly less smooths the visible jitter of a 10 Hz telemetry stream without
# hiding real motion.
SMOOTHING = 0.35

COL_ROBOT = (0.20, 0.60, 0.90)
COL_GHOST = (0.55, 0.55, 0.60)
COL_NOSE = (0.95, 0.75, 0.15)
COL_TRAIL = (0.95, 0.55, 0.20)


# ── Scene ───────────────────────────────────────────────────────────────────


def _cube(stage, path, translate, scale, color):
    prim = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    prim.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*translate))
    UsdGeom.XformCommonAPI(prim).SetScale(Gf.Vec3f(*scale))
    prim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return prim


def _cylinder(stage, path, translate, radius, height, color):
    prim = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    prim.CreateRadiusAttr(radius)
    prim.CreateHeightAttr(height)
    prim.CreateAxisAttr("Z")
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*translate))
    prim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return prim


def build_robot(stage, path, color, opacity=1.0):
    UsdGeom.Xform.Define(stage, Sdf.Path(path))
    body = _cylinder(stage, f"{path}/Body", (0, 0, 0.05), 0.14, 0.05, color)
    # A nose marker: without it, strafing and driving forward look identical.
    _cube(stage, f"{path}/Nose", (0.11, 0, 0.06), (0.06, 0.02, 0.02), COL_NOSE)

    if opacity < 1.0:
        UsdGeom.Gprim(body).CreateDisplayOpacityAttr(Vt.FloatArray([opacity]))
    return stage.GetPrimAtPath(path)


def build_scene():
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, Sdf.Path("/World"))

    build_robot(stage, ROBOT_PATH, COL_ROBOT)
    if SHOW_GHOST:
        build_robot(stage, GHOST_PATH, COL_GHOST, opacity=0.35)
    UsdGeom.Xform.Define(stage, Sdf.Path(TRAIL_PATH))

    print("[twin] scene ready")
    return stage


# ── Pose sources ────────────────────────────────────────────────────────────


class MqttPoseSource:
    """Subscribes to the pose topic and keeps the latest message."""

    def __init__(self):
        import paho.mqtt.client as mqtt

        self.latest = None
        self.connected = False
        self.messages = 0

        self.client = mqtt.Client(client_id="omniverse-twin-follower")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self.client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=60)
        # loop_start runs the network loop on its own thread, so it never
        # blocks Kit's update loop. Blocking that loop freezes the viewport.
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            client.subscribe(POSE_TOPIC)
            print(f"[twin] subscribed to {POSE_TOPIC}")
        else:
            print(f"[twin] MQTT connect failed rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        print("[twin] MQTT disconnected; paho will retry")

    def _on_message(self, client, userdata, message):
        try:
            self.latest = json.loads(message.payload)
            self.messages += 1
        except Exception as exc:
            print(f"[twin] bad pose message: {exc}")

    def read(self):
        return self.latest

    def shutdown(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


class FilePoseSource:
    """Reads the pose from a file. Fallback when Kit has no paho."""

    def __init__(self, path=POSE_FILE):
        self.path = path
        self.latest = None
        self.messages = 0
        self._last_mtime = 0.0
        print(f"[twin] reading pose from {path}")

    def read(self):
        try:
            mtime = os.path.getmtime(self.path)
            if mtime != self._last_mtime:
                self._last_mtime = mtime
                with open(self.path, encoding="utf-8") as handle:
                    self.latest = json.load(handle)
                self.messages += 1
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[twin] could not read pose file: {exc}")
        return self.latest

    def shutdown(self):
        pass


# ── Follower ────────────────────────────────────────────────────────────────


def _shortest_angle(target_deg, current_deg):
    """Signed difference in (-180, 180].

    Naive subtraction breaks at the 359 -> 0 wrap and sends the prim spinning
    the long way round, which looks dramatic and is entirely wrong.
    """
    diff = (target_deg - current_deg + 180.0) % 360.0 - 180.0
    return diff + 360.0 if diff <= -180.0 else diff


class TwinFollower:
    def __init__(self, stage, source):
        self.stage = stage
        self.source = source
        self.subscription = None

        self.x = self.y = self.heading = 0.0
        self.trail_index = 0
        self._last_trail = (0.0, 0.0)
        self._frames = 0

    def start(self):
        app = omni.kit.app.get_app()
        self.subscription = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="twin_follower"
        )
        print("[twin] following — call stop_follower() to end")

    def _on_update(self, event):
        pose = self.source.read()
        if not pose:
            return

        target_x = float(pose.get("x_m", 0.0))
        target_y = float(pose.get("y_m", 0.0))
        target_heading = float(pose.get("heading_deg", 0.0))

        # Chase the reported pose rather than snapping to it: telemetry
        # arrives at ~10 Hz while Kit renders at 60, so snapping shows as a
        # visible stutter.
        self.x += (target_x - self.x) * SMOOTHING
        self.y += (target_y - self.y) * SMOOTHING
        self.heading = (
            self.heading + _shortest_angle(target_heading, self.heading) * SMOOTHING
        ) % 360.0

        self._place(ROBOT_PATH, self.x, self.y, self.heading)

        if SHOW_GHOST:
            self._place(
                GHOST_PATH,
                float(pose.get("ideal_x_m", target_x)),
                float(pose.get("ideal_y_m", target_y)),
                float(pose.get("ideal_heading_deg", target_heading)),
            )

        if SHOW_TRAIL:
            self._maybe_trail()

        self._frames += 1
        if self._frames % 300 == 0:
            print(
                f"[twin] ({self.x:.2f}, {self.y:.2f}) hdg {self.heading:.0f} deg  "
                f"{self.source.messages} updates received"
            )

    def _place(self, path, x, y, heading_deg):
        prim = self.stage.GetPrimAtPath(path)
        if not prim:
            return
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate(Gf.Vec3d(x, y, 0.0))
        api.SetRotate(Gf.Vec3f(0.0, 0.0, heading_deg))

    def _maybe_trail(self, spacing_m=0.12):
        if math.hypot(self.x - self._last_trail[0], self.y - self._last_trail[1]) < spacing_m:
            return
        _cube(
            self.stage,
            f"{TRAIL_PATH}/Dot_{self.trail_index}",
            (self.x, self.y, 0.005),
            (0.03, 0.03, 0.01),
            COL_TRAIL,
        )
        self.trail_index += 1
        self._last_trail = (self.x, self.y)

    def stop(self):
        if self.subscription is not None:
            self.subscription.unsubscribe()
            self.subscription = None
        self.source.shutdown()


# ── Entry points ────────────────────────────────────────────────────────────

_follower = None


def start_follower():
    global _follower
    stop_follower()

    stage = build_scene()

    source = None
    if USE_MQTT:
        try:
            source = MqttPoseSource()
        except ImportError:
            print("[twin] paho-mqtt not available in Kit. Install it with:")
            print("    import omni.kit.pipapi")
            print("    omni.kit.pipapi.install('paho-mqtt', module='paho')")
            print("[twin] falling back to the pose file")

    if source is None:
        source = FilePoseSource()

    _follower = TwinFollower(stage, source)
    _follower.start()


def stop_follower():
    global _follower
    if _follower is not None:
        _follower.stop()
        _follower = None
        print("[twin] stopped")


start_follower()
