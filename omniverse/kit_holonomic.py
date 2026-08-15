"""Holonomic robot movement in plain Omniverse Kit — no Isaac Sim required.

WHEN TO USE THIS FILE
---------------------
Use it if you have Omniverse **Kit / Code / USD Composer** but not Isaac Sim.
It is the same approach the SmartClean Twin DT project already uses: the robot
is *moved* by writing its transform each frame, rather than *driven* by a
physics engine.

    Window > Script Editor, paste this whole file, press Ctrl+Enter.

Kinematic vs physics — the honest difference
--------------------------------------------
This script computes where the robot should be and puts it there. There is no
friction, no wheel slip, no mass, and the robot cannot be pushed by anything.
The motion it shows is exactly what the kinematics predict, which makes it
excellent for confirming the maths and the wall-following logic, and useless
for asking whether the real robot will slip on carpet.

Isaac Sim's physics answers the second question. Both matter, and the important
part is that **both run the same kinematics** — `robotmap_common.holonomic` —
so a result verified here still holds there.

Practically: start here to see the robot move correctly, then move to Isaac Sim
when you want sim-to-real fidelity.

Self-contained
--------------
Kit's Python cannot see this project's packages, so the kiwi-drive maths is
inlined below rather than imported. It is a direct copy of the tested
functions in `shared/robotmap_common/holonomic.py`; the module docstring there
explains the derivation. If you change one, change both — `test_holonomic.py`
guards the original.
"""

import math

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, UsdGeom, Vt

# ── Configuration ───────────────────────────────────────────────────────────

ROBOT_PATH = "/World/HoloRobot"
ROOM_WIDTH_M = 6.0
ROOM_HEIGHT_M = 4.5

WHEEL_RADIUS_M = 0.029
WHEEL_OFFSET_M = 0.100
WHEEL_ANGLES_DEG = (0.0, 120.0, 240.0)

UPDATE_HZ = 60.0
MAX_LINEAR_MPS = 0.35
MAX_ANGULAR_DPS = 90.0

COL_CHASSIS = (0.20, 0.55, 0.85)
COL_WHEEL = (0.15, 0.15, 0.18)
COL_NOSE = (0.95, 0.75, 0.15)
COL_FLOOR = (0.82, 0.78, 0.72)
COL_WALL = (0.90, 0.90, 0.90)
COL_TRAIL = (0.95, 0.60, 0.20)


# ── Kiwi-drive maths (mirror of robotmap_common.holonomic) ──────────────────


def inverse_kinematics(vx, vy, omega_dps):
    """Body twist -> the three wheel rim speeds that produce it."""
    omega_rad = math.radians(omega_dps)
    speeds = []
    for angle_deg in WHEEL_ANGLES_DEG:
        a = math.radians(angle_deg)
        speeds.append(-vx * math.sin(a) + vy * math.cos(a) + omega_rad * WHEEL_OFFSET_M)
    return speeds


def integrate_twist(vx, vy, omega_dps, heading_deg, dt_s):
    """Integrate a body twist into a world-frame pose change.

    Exact arc integration: a holonomic robot translating while rotating traces
    a curve, and treating it as a straight line accumulates error every time
    the robot does the thing this platform exists for.
    """
    omega_rad_s = math.radians(omega_dps)
    theta0 = math.radians(heading_deg)

    if abs(omega_rad_s) < 1e-9:
        cos_t, sin_t = math.cos(theta0), math.sin(theta0)
        dx = (vx * cos_t - vy * sin_t) * dt_s
        dy = (vx * sin_t + vy * cos_t) * dt_s
        return dx, dy, 0.0

    theta1 = theta0 + omega_rad_s * dt_s
    sin0, cos0 = math.sin(theta0), math.cos(theta0)
    sin1, cos1 = math.sin(theta1), math.cos(theta1)

    dx = (vx * (sin1 - sin0) + vy * (cos1 - cos0)) / omega_rad_s
    dy = (-vx * (cos1 - cos0) + vy * (sin1 - sin0)) / omega_rad_s
    return dx, dy, math.degrees(omega_rad_s * dt_s)


# ── Scene construction ──────────────────────────────────────────────────────


def _cube(stage, path, translate, scale, color):
    prim = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    prim.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*translate))
    UsdGeom.XformCommonAPI(prim).SetScale(Gf.Vec3f(*scale))
    prim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return prim


def _cylinder(stage, path, translate, radius, height, color, rotate=None):
    prim = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    prim.CreateRadiusAttr(radius)
    prim.CreateHeightAttr(height)
    prim.CreateAxisAttr("Z")
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*translate))
    if rotate:
        api.SetRotate(Gf.Vec3f(*rotate))
    prim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return prim


def build_scene():
    """Create the room and the robot. Safe to run more than once."""
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, Sdf.Path("/World"))

    # Room
    UsdGeom.Xform.Define(stage, Sdf.Path("/World/Room"))
    _cube(
        stage,
        "/World/Room/Floor",
        (ROOM_WIDTH_M / 2, ROOM_HEIGHT_M / 2, -0.01),
        (ROOM_WIDTH_M, ROOM_HEIGHT_M, 0.02),
        COL_FLOOR,
    )
    thickness, height = 0.1, 0.4
    for name, translate, scale in (
        ("WallSouth", (ROOM_WIDTH_M / 2, 0, height / 2), (ROOM_WIDTH_M, thickness, height)),
        ("WallNorth", (ROOM_WIDTH_M / 2, ROOM_HEIGHT_M, height / 2), (ROOM_WIDTH_M, thickness, height)),
        ("WallWest", (0, ROOM_HEIGHT_M / 2, height / 2), (thickness, ROOM_HEIGHT_M, height)),
        ("WallEast", (ROOM_WIDTH_M, ROOM_HEIGHT_M / 2, height / 2), (thickness, ROOM_HEIGHT_M, height)),
    ):
        _cube(stage, f"/World/Room/{name}", translate, scale, COL_WALL)

    # Robot
    UsdGeom.Xform.Define(stage, Sdf.Path(ROBOT_PATH))
    _cylinder(stage, f"{ROBOT_PATH}/Chassis", (0, 0, 0.05), 0.14, 0.05, COL_CHASSIS)

    # A nose marker, so the robot's facing is visible. Without it, pure
    # strafing and pure driving look identical and the demo loses its point.
    _cube(stage, f"{ROBOT_PATH}/Nose", (0.11, 0, 0.06), (0.06, 0.02, 0.02), COL_NOSE)

    for index, angle_deg in enumerate(WHEEL_ANGLES_DEG):
        a = math.radians(angle_deg)
        _cylinder(
            stage,
            f"{ROBOT_PATH}/Wheel_{index}",
            (WHEEL_OFFSET_M * math.cos(a), WHEEL_OFFSET_M * math.sin(a), 0.029),
            WHEEL_RADIUS_M,
            0.018,
            COL_WHEEL,
            rotate=(90.0, 0.0, angle_deg + 90.0),
        )

    UsdGeom.Xform.Define(stage, Sdf.Path("/World/Trail"))
    print(f"[scene] built room {ROOM_WIDTH_M} x {ROOM_HEIGHT_M} m and robot")
    return stage


# ── Driver ──────────────────────────────────────────────────────────────────


class KinematicHolonomicRobot:
    """Moves the robot prim according to the kiwi-drive kinematics."""

    def __init__(self, stage, x=1.0, y=1.0, heading_deg=0.0):
        self.stage = stage
        self.robot = stage.GetPrimAtPath(ROBOT_PATH)
        self.x = x
        self.y = y
        self.heading_deg = heading_deg

        self.wheel_angle_rad = [0.0, 0.0, 0.0]
        self.distance_travelled_m = 0.0
        self.trail_index = 0
        self._last_trail_xy = (x, y)

        self.apply_transform()

    def apply_transform(self):
        api = UsdGeom.XformCommonAPI(self.robot)
        api.SetTranslate(Gf.Vec3d(self.x, self.y, 0.0))
        api.SetRotate(Gf.Vec3f(0.0, 0.0, self.heading_deg))

    def step(self, vx, vy, omega_dps, dt_s):
        """Advance the robot by one frame of the given twist."""
        dx, dy, dheading = integrate_twist(vx, vy, omega_dps, self.heading_deg, dt_s)

        self.x += dx
        self.y += dy
        self.heading_deg = (self.heading_deg + dheading) % 360.0
        self.distance_travelled_m += math.hypot(dx, dy)

        # Spin the wheels visually at the rate the kinematics demand, so it is
        # obvious the three wheels really are doing different things.
        for index, speed in enumerate(inverse_kinematics(vx, vy, omega_dps)):
            self.wheel_angle_rad[index] += speed / WHEEL_RADIUS_M * dt_s
            wheel = self.stage.GetPrimAtPath(f"{ROBOT_PATH}/Wheel_{index}")
            if wheel:
                spin = math.degrees(self.wheel_angle_rad[index]) % 360.0
                UsdGeom.XformCommonAPI(wheel).SetRotate(
                    Gf.Vec3f(90.0, spin, WHEEL_ANGLES_DEG[index] + 90.0)
                )

        self.apply_transform()
        self._maybe_drop_trail()

    def _maybe_drop_trail(self, spacing_m=0.12):
        if math.hypot(self.x - self._last_trail_xy[0], self.y - self._last_trail_xy[1]) < spacing_m:
            return
        _cube(
            self.stage,
            f"/World/Trail/Dot_{self.trail_index}",
            (self.x, self.y, 0.005),
            (0.03, 0.03, 0.01),
            COL_TRAIL,
        )
        self.trail_index += 1
        self._last_trail_xy = (self.x, self.y)


# ── Demonstration routine ───────────────────────────────────────────────────

# Each entry: (label, vx, vy, omega_dps, seconds).
#
# The sequence is chosen to show what a holonomic base can do that a
# differential one cannot: it strafes without turning, then translates and
# rotates at the same time.
DEMO_SEQUENCE = [
    ("drive forward", MAX_LINEAR_MPS, 0.0, 0.0, 3.0),
    ("strafe LEFT (no turn)", 0.0, MAX_LINEAR_MPS, 0.0, 2.5),
    ("drive backward", -MAX_LINEAR_MPS, 0.0, 0.0, 3.0),
    ("strafe RIGHT (no turn)", 0.0, -MAX_LINEAR_MPS, 0.0, 2.5),
    ("rotate in place", 0.0, 0.0, MAX_ANGULAR_DPS, 4.0),
    ("diagonal", MAX_LINEAR_MPS * 0.7, MAX_LINEAR_MPS * 0.7, 0.0, 2.5),
    ("translate WHILE rotating", MAX_LINEAR_MPS, 0.0, MAX_ANGULAR_DPS * 0.6, 5.0),
    ("stop", 0.0, 0.0, 0.0, 1.0),
]


class DemoRunner:
    """Steps the robot through DEMO_SEQUENCE on Kit's update loop."""

    def __init__(self, robot):
        self.robot = robot
        self.index = 0
        self.elapsed = 0.0
        self.subscription = None
        self._announced = -1

    def start(self):
        app = omni.kit.app.get_app()
        self.subscription = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="holonomic_demo"
        )
        print("[demo] running — call stop_demo() to end early")

    def _on_update(self, event):
        dt = 1.0 / UPDATE_HZ

        if self.index >= len(DEMO_SEQUENCE):
            print(
                f"[demo] finished. travelled {self.robot.distance_travelled_m:.2f} m, "
                f"final heading {self.robot.heading_deg:.0f} deg"
            )
            self.stop()
            return

        label, vx, vy, omega, duration = DEMO_SEQUENCE[self.index]

        if self._announced != self.index:
            print(f"[demo] {label}")
            self._announced = self.index

        self.robot.step(vx, vy, omega, dt)
        self.elapsed += dt

        if self.elapsed >= duration:
            self.index += 1
            self.elapsed = 0.0

    def stop(self):
        if self.subscription is not None:
            self.subscription.unsubscribe()
            self.subscription = None


# ── Entry points ────────────────────────────────────────────────────────────

_runner = None
_robot = None


def run_demo():
    """Build the scene and run the movement demonstration."""
    global _runner, _robot
    stop_demo()

    stage = build_scene()
    _robot = KinematicHolonomicRobot(stage, x=1.0, y=1.0, heading_deg=0.0)
    _runner = DemoRunner(_robot)
    _runner.start()


def stop_demo():
    global _runner
    if _runner is not None:
        _runner.stop()
        _runner = None


def drive(vx=0.0, vy=0.0, omega_dps=0.0, seconds=2.0):
    """Drive one manual command. Useful for poking at it from the console.

        drive(vy=0.2, seconds=3)     # strafe left for three seconds
    """
    global _robot
    if _robot is None:
        stage = build_scene()
        _robot = KinematicHolonomicRobot(stage)

    steps = int(seconds * UPDATE_HZ)
    for _ in range(steps):
        _robot.step(vx, vy, omega_dps, 1.0 / UPDATE_HZ)
    print(
        f"[drive] now at ({_robot.x:.2f}, {_robot.y:.2f}) "
        f"heading {_robot.heading_deg:.0f} deg"
    )


# Running the file in the Script Editor starts the demo immediately.
run_demo()
