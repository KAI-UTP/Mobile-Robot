"""Drive the holonomic robot in Isaac Sim.

    isaac-sim/python.bat omniverse/run_isaac.py --mode teleop

Modes
-----
``teleop``   Keyboard control. Start here — it is how you confirm the robot
             moves the way you expect before anything else is built on top.
``verify``   Runs a self-check: each axis of motion in turn, comparing the
             commanded twist against what the physics engine actually did.
             Run this whenever the robot USD changes.
``mapping``  Autonomous wall-following, publishing sensor packets to MQTT so
             the existing mapper and web viewer work against the sim robot.

Teleop keys
-----------
    W / S     forward / back
    A / D     strafe LEFT / RIGHT   <- no turning involved; this is the whole
                                       point of a holonomic base
    Q / E     rotate left / right
    SPACE     stop
    R         reset the robot to its start pose

IMPORTANT: this file has not been executed. Isaac Sim is not installed on the
machine it was written on, so while the movement maths underneath it is covered
by 65 unit tests, the Isaac glue itself is unverified. Expect to adjust
`WHEEL_JOINT_NAMES` and the robot scale on first run; the `verify` mode exists
to make those adjustments quick.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# ── Isaac Sim must boot before anything else imports omni.* ─────────────────
# This ordering is not stylistic. SimulationApp starts the underlying Kit
# application; importing omni modules first fails with errors that point
# nowhere near the real cause.

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "shared", ROOT / "services", ROOT / "omniverse", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from isaac_compat import (  # noqa: E402
    IsaacNotAvailable,
    find_joint_indices,
    kaya_asset_paths,
    print_environment_report,
    require_isaac,
    resolve_existing_asset,
    start_simulation_app,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Holonomic robot in Isaac Sim")
    parser.add_argument(
        "--mode", choices=["teleop", "verify", "mapping"], default="teleop"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--robot",
        choices=["kaya", "custom"],
        default="kaya",
        help="kaya = the built-in NVIDIA 3-omni-wheel robot (recommended start)",
    )
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument(
        "--no-mqtt", action="store_true", help="mapping mode: skip publishing"
    )
    parser.add_argument("--room-width", type=float, default=6.0)
    parser.add_argument("--room-height", type=float, default=4.5)
    return parser.parse_args()


ARGS = _parse_args()

try:
    simulation_app = start_simulation_app(headless=ARGS.headless)
except IsaacNotAvailable as exc:
    # The message already explains how to fix this; a traceback would only
    # bury it, so suppress the chained exception.
    print(exc)
    raise SystemExit(1) from None

# Everything below is only importable once SimulationApp exists.
import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
from drive_controller import HolonomicDriveController, verify_wheel_order  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402
from robotmap_common.holonomic import (  # noqa: E402
    BodyTwist,
    DriveLimits,
    HolonomicGeometry,
    twist_from_direction,
)

isaac = require_isaac()
World = isaac["World"]
Articulation = isaac["Articulation"]

# ── Robot configuration ─────────────────────────────────────────────────────

ROBOT_PRIM_PATH = "/World/Robot"

# Kaya's three omni wheels. If you build a custom robot, change these to match
# and re-run `--mode verify`.
WHEEL_JOINT_NAMES = ["axle_0_joint", "axle_1_joint", "axle_2_joint"]

# NVIDIA Kaya's real dimensions. Replace with YOUR measurements when moving to
# the physical robot — `wheel_offset_m` especially, since an error there makes
# the robot rotate slightly whenever it is asked to translate.
KAYA_GEOMETRY = HolonomicGeometry(
    wheel_radius_m=0.04,
    wheel_offset_m=0.125,
    wheel_angles_deg=(0.0, 120.0, 240.0),
    ticks_per_revolution=4096,
)

LIMITS = DriveLimits(max_linear_mps=0.4, max_angular_dps=120.0, max_wheel_mps=0.8)

PHYSICS_DT = 1.0 / 60.0


# ── Scene ───────────────────────────────────────────────────────────────────


def build_room(stage, width_m: float, height_m: float) -> None:
    """A simple walled room for the robot to map.

    Walls are static colliders. Without a collider the ultrasonic raycasts pass
    straight through and the robot maps an empty plane.
    """
    UsdGeom.Xform.Define(stage, Sdf.Path("/World/Room"))

    thickness = 0.1
    wall_height = 0.5  # low enough to see over, tall enough for the sensors

    walls = [
        ("WallSouth", (width_m / 2, 0.0, wall_height / 2), (width_m, thickness, wall_height)),
        ("WallNorth", (width_m / 2, height_m, wall_height / 2), (width_m, thickness, wall_height)),
        ("WallWest", (0.0, height_m / 2, wall_height / 2), (thickness, height_m, wall_height)),
        ("WallEast", (width_m, height_m / 2, wall_height / 2), (thickness, height_m, wall_height)),
    ]

    for name, translate, scale in walls:
        path = f"/World/Room/{name}"
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
        cube.CreateSizeAttr(1.0)
        UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(*translate))
        UsdGeom.XformCommonAPI(cube).SetScale(Gf.Vec3f(*scale))
        # Static collider: participates in collisions but is never simulated
        # as a moving body.
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    print(f"[scene] room {width_m} x {height_m} m built")


def spawn_robot(world, stage, args) -> str:
    """Place the robot in the stage and return its prim path."""
    get_assets_root_path = isaac["get_assets_root_path"]
    add_reference_to_stage = isaac["add_reference_to_stage"]

    if args.robot == "kaya":
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError(
                "Could not reach the Isaac asset server.\n"
                "Isaac Sim needs to download its asset library on first run; "
                "check your network, or use --robot custom."
            )

        asset_path = resolve_existing_asset(kaya_asset_paths(assets_root))
        if asset_path is None:
            raise RuntimeError(
                "Kaya robot asset not found on the Nucleus server.\n"
                f"Tried: {kaya_asset_paths(assets_root)}\n"
                "Use --robot custom to build a simple stand-in instead."
            )

        add_reference_to_stage(usd_path=asset_path, prim_path=ROBOT_PRIM_PATH)
        print(f"[scene] loaded Kaya from {asset_path}")
    else:
        build_custom_robot(stage)

    return ROBOT_PRIM_PATH


def build_custom_robot(stage) -> None:
    """A minimal stand-in when the Kaya asset is unavailable.

    Honest warning: the wheels here are plain cylinders with no rollers, so
    they do NOT slide sideways and the robot will not behave holonomically
    under physics. It is useful for checking the scene loads and the joints are
    wired, not for studying motion. Modelling omni rollers properly means
    authoring a dozen small capsule bodies per wheel with their own joints,
    which is exactly the work Kaya already did.
    """
    carb.log_warn(
        "Custom robot has no omni rollers; motion will not be holonomic. "
        "Use --robot kaya for realistic behaviour."
    )

    UsdGeom.Xform.Define(stage, Sdf.Path(ROBOT_PRIM_PATH))

    chassis = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"{ROBOT_PRIM_PATH}/Chassis"))
    chassis.CreateRadiusAttr(0.15)
    chassis.CreateHeightAttr(0.06)
    chassis.CreateAxisAttr("Z")
    UsdGeom.XformCommonAPI(chassis).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.06))
    UsdPhysics.RigidBodyAPI.Apply(chassis.GetPrim())
    UsdPhysics.CollisionAPI.Apply(chassis.GetPrim())

    for index, angle_deg in enumerate(KAYA_GEOMETRY.wheel_angles_deg):
        angle = math.radians(angle_deg)
        offset = KAYA_GEOMETRY.wheel_offset_m
        position = Gf.Vec3d(
            offset * math.cos(angle), offset * math.sin(angle), 0.04
        )

        wheel_path = f"{ROBOT_PRIM_PATH}/Wheel_{index}"
        wheel = UsdGeom.Cylinder.Define(stage, Sdf.Path(wheel_path))
        wheel.CreateRadiusAttr(KAYA_GEOMETRY.wheel_radius_m)
        wheel.CreateHeightAttr(0.02)
        wheel.CreateAxisAttr("Z")
        UsdGeom.XformCommonAPI(wheel).SetTranslate(position)
        # Rotate so the wheel rolls tangentially, matching the kinematics.
        UsdGeom.XformCommonAPI(wheel).SetRotate(Gf.Vec3f(90.0, 0.0, angle_deg + 90.0))
        UsdPhysics.RigidBodyAPI.Apply(wheel.GetPrim())
        UsdPhysics.CollisionAPI.Apply(wheel.GetPrim())

        joint = UsdPhysics.RevoluteJoint.Define(
            stage, Sdf.Path(f"{ROBOT_PRIM_PATH}/axle_{index}_joint")
        )
        joint.CreateBody0Rel().SetTargets([chassis.GetPath()])
        joint.CreateBody1Rel().SetTargets([wheel.GetPath()])
        joint.CreateAxisAttr("Z")

    print("[scene] custom robot built (no rollers — motion will not be holonomic)")


# ── Body-frame velocity, for the wheel-order check ──────────────────────────


def make_body_velocity_reader(robot):
    """Return a function giving chassis velocity in the ROBOT frame.

    Read from the physics engine, deliberately independent of the controller's
    joint mapping — that independence is what lets the wheel-order check detect
    a permutation at all.
    """

    def read() -> tuple[float, float]:
        world_velocity = robot.get_linear_velocity()
        if world_velocity is None:
            return 0.0, 0.0

        _, orientation = robot.get_world_pose()
        # Quaternion (w, x, y, z) -> yaw about Z.
        w, x, y, z = orientation
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        vx_world, vy_world = float(world_velocity[0]), float(world_velocity[1])
        cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
        return (
            vx_world * cos_y - vy_world * sin_y,
            vx_world * sin_y + vy_world * cos_y,
        )

    return read


# ── Keyboard ────────────────────────────────────────────────────────────────


class KeyboardTeleop:
    """Maps key state to a body twist.

    Tracks keys held down rather than reacting to presses, so motion is smooth
    and diagonal combinations (forward + strafe) work naturally — which is the
    behaviour worth showing off on a holonomic base.
    """

    def __init__(self, limits: DriveLimits) -> None:
        self.limits = limits
        self.pressed: set[str] = set()
        self.reset_requested = False

        app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_window.get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_key
        )

    def _on_key(self, event, *args) -> bool:
        name = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.pressed.add(name)
            if name == "R":
                self.reset_requested = True
            if name == "SPACE":
                self.pressed.clear()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.pressed.discard(name)
        return True

    def twist(self) -> BodyTwist:
        vx = vy = omega = 0.0
        if "W" in self.pressed:
            vx += self.limits.max_linear_mps
        if "S" in self.pressed:
            vx -= self.limits.max_linear_mps
        # A and D strafe. On a differential robot these would have to steer.
        if "A" in self.pressed:
            vy += self.limits.max_linear_mps
        if "D" in self.pressed:
            vy -= self.limits.max_linear_mps
        if "Q" in self.pressed:
            omega += self.limits.max_angular_dps
        if "E" in self.pressed:
            omega -= self.limits.max_angular_dps
        return BodyTwist(vx_mps=vx, vy_mps=vy, omega_dps=omega)

    def shutdown(self) -> None:
        try:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard, self._subscription
            )
        except Exception:
            pass


# ── Modes ───────────────────────────────────────────────────────────────────


def run_verify(world, controller, read_body_velocity) -> None:
    """Check each axis of motion against what the physics engine reports."""
    print()
    print("=" * 62)
    print(" Motion verification")
    print("=" * 62)

    ok, message = verify_wheel_order(
        controller, lambda: world.step(render=False), read_body_velocity
    )
    print(f"  wheel order      : {'PASS' if ok else 'FAIL'} — {message}")

    checks = [
        ("forward   ", BodyTwist(vx_mps=0.2), (0.2, 0.0)),
        ("backward  ", BodyTwist(vx_mps=-0.2), (-0.2, 0.0)),
        ("strafe L  ", BodyTwist(vy_mps=0.2), (0.0, 0.2)),
        ("strafe R  ", BodyTwist(vy_mps=-0.2), (0.0, -0.2)),
        ("diagonal  ", twist_from_direction(45.0, 0.2), (0.141, 0.141)),
    ]

    print()
    print(f"  {'motion':<12}{'commanded':<20}{'measured':<20}error")
    for label, twist, expected in checks:
        controller.reset_odometry()
        # Let the wheels spin up before sampling; PhysX does not reach a
        # commanded velocity instantly.
        for _ in range(60):
            controller.drive(twist)
            world.step(render=False)

        vx, vy = read_body_velocity()
        error = math.hypot(vx - expected[0], vy - expected[1])
        print(
            f"  {label:<12}({expected[0]:+.2f}, {expected[1]:+.2f})      "
            f"({vx:+.2f}, {vy:+.2f})      {error:.3f} m/s"
        )

        controller.stop()
        for _ in range(30):
            world.step(render=False)

    print()
    print("  Rotation is checked separately because it has no linear component.")
    controller.reset_odometry()
    for _ in range(60):
        controller.drive(BodyTwist(omega_dps=60.0))
        world.step(render=False)
    measured = controller.read_state().measured
    print(f"  rotate      commanded +60 dps    wheels report {measured.omega_dps:+.1f} dps")
    controller.stop()

    print()
    print("  Large errors usually mean one of:")
    print("    - wheel_offset_m or wheel_radius_m do not match the robot")
    print("    - WHEEL_JOINT_NAMES are in the wrong order")
    print("    - the wheels have no rollers (custom robot), so cannot slide")
    print("=" * 62)


def run_teleop(world, controller, teleop, robot, start_pose) -> None:
    print()
    print("=" * 62)
    print(" Teleop — click the viewport first so it receives key events")
    print("   W/S forward/back   A/D STRAFE left/right   Q/E rotate")
    print("   SPACE stop         R reset")
    print("=" * 62)

    step = 0
    while simulation_app.is_running():
        if teleop.reset_requested:
            robot.set_world_pose(*start_pose)
            controller.reset_odometry()
            teleop.reset_requested = False
            print("[teleop] robot reset")

        controller.drive(teleop.twist())
        world.step(render=True)

        step += 1
        if step % 60 == 0:
            state = controller.read_state()
            position, _ = robot.get_world_pose()
            print(
                f"  pos ({position[0]:+.2f}, {position[1]:+.2f})  "
                f"cmd vx={state.commanded.vx_mps:+.2f} vy={state.commanded.vy_mps:+.2f} "
                f"w={state.commanded.omega_dps:+.0f}  "
                f"ticks {controller.encoder_ticks()}"
            )


def run_mapping(world, controller, robot, args) -> None:
    """Autonomous wall-following, feeding the existing mapping stack."""
    from isaac_bridge import IsaacSensorBridge

    bridge = IsaacSensorBridge(
        robot=robot,
        controller=controller,
        mqtt_host=args.mqtt_host,
        publish=not args.no_mqtt,
    )

    print()
    print("=" * 62)
    print(" Mapping mode — the robot follows walls and publishes telemetry")
    if not args.no_mqtt:
        print(f"   MQTT broker : {args.mqtt_host}:1883")
        print("   Start the mapper in another terminal:")
        print("     python services/mapper/main.py --source mqtt")
        print("   then open http://localhost:8080")
    print("=" * 62)

    while simulation_app.is_running():
        twist = bridge.step(PHYSICS_DT)
        controller.drive(twist)
        world.step(render=True)


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("[env] environment:")
    print_environment_report()

    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT)
    world.scene.add_default_ground_plane()

    stage = world.stage
    build_room(stage, ARGS.room_width, ARGS.room_height)
    spawn_robot(world, stage, ARGS)

    robot = Articulation(prim_path=ROBOT_PRIM_PATH, name="holonomic_robot")
    world.scene.add(robot)

    # reset() initialises physics; joint names are unavailable before it.
    world.reset()

    try:
        wheel_indices = find_joint_indices(robot, WHEEL_JOINT_NAMES)
    except ValueError as exc:
        print()
        print("[error] could not find the wheel joints:")
        print(exc)
        simulation_app.close()
        raise SystemExit(1) from None

    print(f"[robot] wheel joints {WHEEL_JOINT_NAMES} -> DOF indices {wheel_indices}")

    # Start clear of the wall so the first sensor readings are meaningful.
    start_position = np.array([1.0, 1.0, 0.05])
    start_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    robot.set_world_pose(start_position, start_orientation)

    controller = HolonomicDriveController(
        articulation=robot,
        wheel_joint_indices=wheel_indices,
        geometry=KAYA_GEOMETRY,
        limits=LIMITS,
    )
    read_body_velocity = make_body_velocity_reader(robot)

    try:
        if ARGS.mode == "verify":
            run_verify(world, controller, read_body_velocity)
        elif ARGS.mode == "mapping":
            run_mapping(world, controller, robot, ARGS)
        else:
            teleop = KeyboardTeleop(LIMITS)
            try:
                run_teleop(
                    world, controller, teleop, robot,
                    (start_position, start_orientation),
                )
            finally:
                teleop.shutdown()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
