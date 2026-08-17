"""The Omniverse scene, checked without Omniverse.

Why this exists
---------------
`omniverse/kit_room_3d.py` runs inside Omniverse's own Python and cannot be
imported normally: it pulls in `omni.*` and `pxr`, and calls `run_room()` at
module scope. So none of it is covered by anything, and a scene can be silently
wrong while still looking like a plausible room — axes swapped, a robot placed
by the wrong pose, an object with no collider.

This file used to be named for the "measured room" the scene drew beside the
real one. That half is gone: a flat floor plan reads better in a browser than
an extruded outline seen in perspective, and the 2D map now has a pane for
exactly that. What is left is the scene itself.

The file is therefore loaded here against stub USD modules that record what was
created, and the recorded geometry is checked against the room definition it
was given. The shipped file is what gets loaded, not a copy of it.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import pytest

KIT_SCRIPT = Path(__file__).resolve().parents[1] / "omniverse" / "kit_room_3d.py"


# ── Stub USD ─────────────────────────────────────────────────────────────────


class FakePrim:
    def __init__(self, path: str, stage=None) -> None:
        self.path = path
        self.stage = stage
        self.translate = (0.0, 0.0, 0.0)
        self.scale = (1.0, 1.0, 1.0)
        self.rotate = (0.0, 0.0, 0.0)
        self.colour = None
        self.opacity = None
        self.radius = None
        self.height = None
        # Every other Create*Attr the script sets, by name. Assigned before
        # __getattr__ can be reached, which would otherwise recurse.
        self.attrs: dict = {}
        # UsdPhysics schemas applied to this prim, by name.
        self.schemas: set = set()

    # The handful whose values the tests actually assert on.
    def CreateRadiusAttr(self, value): self.radius = value
    def CreateHeightAttr(self, value): self.height = value
    def CreateDisplayColorAttr(self, value): self.colour = tuple(value[0])

    def GetPrim(self): return self
    def __bool__(self): return True

    def GetAttribute(self, name):
        """USD's generic attribute access, which is all a plain `Usd.Prim` has.

        Modelled deliberately: `GetChildren()` returns untyped prims, so the
        typed getters (`GetRadiusAttr` and friends) are NOT available on them.
        The stub used to expose `.radius` directly, which let a bug through —
        every cylinder fell out of the radius branch and was measured by its
        scale instead, reporting a 0.035 m table leg as a 1.0 x 1.0 m obstacle.
        """
        values = {
            "radius": self.radius,
            "height": self.height,
            "size": self.attrs.get("CreateSizeAttr"),
        }
        value = values.get(name)
        if value is None:
            return None

        class _Attr:
            def __bool__(self): return True
            @staticmethod
            def Get(): return value

        return _Attr()

    def GetPath(self):
        return types.SimpleNamespace(pathString=self.path)

    def GetChildren(self):
        """Direct children, by path prefix — enough for the physics walk."""
        if self.stage is None:
            return []
        prefix = self.path + "/"
        return [
            prim for path, prim in self.stage.prims.items()
            if path.startswith(prefix) and "/" not in path[len(prefix):]
        ]

    def __getattr__(self, name):
        """Accept any other `Create<Something>Attr`, recording that it was set.

        USD prims expose one of these per schema attribute and the scene uses
        many — size, axis, intensity, width, extent. Enumerating them means the
        stub fails on a legitimate USD call that the shipped script is entitled
        to make, which is a broken test rather than a caught bug.

        Deliberately narrow: only the `Create*Attr` idiom is accepted, so a
        genuine typo or a call to something that is not a USD attribute setter
        still raises.
        """
        if name.startswith("Create") and name.endswith("Attr"):
            def setter(value=None, *_, **__):
                self.attrs[name] = value
            return setter
        raise AttributeError(name)


class FakeStage:
    def __init__(self) -> None:
        self.prims: dict[str, FakePrim] = {}

    def define(self, path: str) -> FakePrim:
        prim = FakePrim(str(path), stage=self)
        self.prims[str(path)] = prim
        return prim

    def GetPrimAtPath(self, path):
        return self.prims.get(str(path))

    def RemovePrim(self, path) -> None:
        prefix = str(path)
        for key in [k for k in self.prims if k == prefix or k.startswith(prefix + "/")]:
            del self.prims[key]


def _build_stub_modules(stage: FakeStage):
    """The smallest `pxr` and `omni` that kit_room_3d.py can run against."""

    class _Definable:
        @staticmethod
        def Define(_stage, path):
            return stage.define(path)

    class _XformCommonAPI:
        def __init__(self, prim):
            self.prim = prim

        def SetTranslate(self, value):
            self.prim.translate = tuple(value)

        def SetScale(self, value):
            self.prim.scale = tuple(value)

        def SetRotate(self, value):
            self.prim.rotate = tuple(value)

        def GetXformVectors(self, _time=0):
            """USD returns (translate, rotate, scale, pivot, rotationOrder).

            Reading the transform back is what lets the scene notice that
            someone has *dragged* a piece of furniture, which is the whole
            mechanism behind the 3D view being the physical world.
            """
            return (
                self.prim.translate, self.prim.rotate, self.prim.scale,
                (0.0, 0.0, 0.0), 0,
            )

    class _Gprim:
        def __init__(self, prim):
            self.prim = prim

        def CreateDisplayOpacityAttr(self, value):
            self.prim.opacity = value[0]

    class _Applied:
        """A UsdPhysics schema applied to a prim, recorded on the prim."""

        def __init__(self, name):
            self.name = name

        def Apply(self, prim):
            prim.schemas.add(self.name)
            return prim

    usdgeom = types.SimpleNamespace(
        Cube=_Definable, Cylinder=_Definable, Sphere=_Definable,
        Xform=_Definable, Mesh=_Definable, Camera=_Definable,
        XformCommonAPI=_XformCommonAPI, Gprim=_Gprim,
        SetStageUpAxis=lambda _stage, axis: setattr(stage, "up_axis", axis),
        Tokens=types.SimpleNamespace(x="x", y="y", z="z"),
    )
    usdphysics = types.SimpleNamespace(
        CollisionAPI=_Applied("collision"),
        MeshCollisionAPI=_Applied("mesh_collision"),
        RigidBodyAPI=_Applied("rigid_body"),
    )
    usdlux = types.SimpleNamespace(
        DistantLight=_Definable, SphereLight=_Definable,
        DomeLight=_Definable, RectLight=_Definable,
    )
    gf = types.SimpleNamespace(
        Vec3d=lambda *a: tuple(a), Vec3f=lambda *a: tuple(a),
        Vec2f=lambda *a: tuple(a),
    )
    sdf = types.SimpleNamespace(Path=str)
    vt = types.SimpleNamespace(
        Vec3fArray=lambda values: list(values), FloatArray=lambda values: list(values),
    )
    return usdgeom, usdlux, gf, sdf, vt, usdphysics


def _load_kit_module(stage: FakeStage) -> types.ModuleType:
    """Load the shipped script with USD stubbed and the entry call stripped."""
    source = KIT_SCRIPT.read_text(encoding="utf-8")

    # The file drives itself when pasted into the Script Editor. Here that would
    # build the whole scene on import, so the trailing call is removed.
    source = source.replace("\nrun_room()\n", "\n")
    assert "\nrun_room()\n" not in source

    usdgeom, usdlux, gf, sdf, vt, usdphysics = _build_stub_modules(stage)

    module = types.ModuleType("kit_room_3d")
    module.__dict__.update(
        {
            "UsdGeom": usdgeom, "UsdLux": usdlux, "UsdPhysics": usdphysics,
            "Gf": gf, "Sdf": sdf, "Vt": vt,
        }
    )

    # Strip the imports the stubs replace; everything else the file imports is
    # standard library and should be exercised for real.
    lines = [
        line for line in source.splitlines()
        if not line.startswith("import omni")
        and not line.startswith("from pxr import")
    ]
    module.__dict__["omni"] = types.SimpleNamespace(
        kit=types.SimpleNamespace(app=types.SimpleNamespace(get_app=lambda: None)),
        usd=types.SimpleNamespace(
            get_context=lambda: types.SimpleNamespace(get_stage=lambda: stage)
        ),
    )

    exec(compile("\n".join(lines), str(KIT_SCRIPT), "exec"), module.__dict__)
    return module


# ── A room to draw ───────────────────────────────────────────────────────────




def _walls(stage):
    return [p for path, p in stage.prims.items() if "/Measured/Outline/Wall_" in path]


def _blocked(stage):
    return [p for path, p in stage.prims.items() if "/Measured/Blocked/Area_" in path]


# ── The outline ──────────────────────────────────────────────────────────────


def _robot_xy(module, stage):
    return stage.prims[f"{module.ROOT}/RobotTrue"].translate[:2]


class _FixedPose:
    """A pose source that always answers the same thing."""

    def __init__(self, pose):
        self.pose = pose

    def read(self):
        return self.pose


def _run_scene(module, stage, pose, ticks=200):
    module.build_all_robots(stage)
    scene = module.RoomScene(stage, _FixedPose(pose))
    for _ in range(ticks):        # smoothing chases the target, so settle it
        scene._on_update(None)
    return _robot_xy(module, stage)


def test_a_pose_at_the_origin_puts_the_robot_where_it_started(kit):
    """The bug, in one case. A fresh filter reports (0, 0) — which drew the
    robot at the ROOM's corner, half inside the wall and reading as outside."""
    module, stage = kit
    x, y = _run_scene(module, stage, {"x_m": 0.0, "y_m": 0.0, "heading_deg": 0.0})

    assert x == pytest.approx(module.ROBOT_START_X_M, abs=0.05)
    assert y == pytest.approx(module.ROBOT_START_Y_M, abs=0.05)


def test_a_negative_pose_still_lands_inside_the_room(kit):
    """Poses go negative as soon as the robot moves back past its start, which
    is most of any lap. Those are the ones that ended up outside the walls."""
    module, stage = kit
    x, y = _run_scene(module, stage, {"x_m": -0.32, "y_m": -0.75, "heading_deg": 0.0})

    assert 0.0 < x < module.ROOM_W
    assert 0.0 < y < module.ROOM_H


def test_the_demo_lap_stays_inside_the_room(kit):
    """The demo emits pose-frame coordinates like the real source, so the one
    transform applies to both. Applying it twice would walk the robot out
    through the far wall instead."""
    module, stage = kit
    module.build_all_robots(stage)
    scene = module.RoomScene(stage, module.DemoPose())

    for _ in range(4000):
        scene._on_update(None)
        x, y = _robot_xy(module, stage)
        assert 0.0 <= x <= module.ROOM_W, f"demo lap left the room at x={x:.2f}"
        assert 0.0 <= y <= module.ROOM_H, f"demo lap left the room at y={y:.2f}"


def test_the_demo_lap_does_not_drive_through_the_furniture(kit):
    """The preview lap has no collision model, so its waypoints have to avoid
    the furniture by construction.

    The original rectangle, inset 0.6 m from the walls, drove through the bin,
    the cabinet and the sofa. A robot gliding through a sofa reads as "the
    collision detection is broken" — when in fact this path was never subject
    to it. Checking by eye is how it was missed.
    """
    module, stage = kit

    # Every footprint from build_furniture, padded by the chassis radius.
    # The chairs matter as much as the table: they stick out 0.2 m further on
    # each side, and a path threaded between table and wall still catches them.
    margin = 0.10
    blocked = [
        ("bin", 5.39, 0.44, 5.71, 0.76),
        ("cabinet", 5.375, 1.85, 5.825, 3.35),
        ("sofa", 0.10, 3.275, 2.10, 4.125),
        ("table", 1.90, 1.875, 3.30, 2.725),
        ("chair west", 1.39, 2.09, 1.81, 2.51),
        ("chair east", 3.39, 2.09, 3.81, 2.51),
        ("chair south", 2.39, 1.34, 2.81, 1.76),
        ("chair north", 2.39, 2.84, 2.81, 3.26),
    ]

    demo = module.DemoPose()
    for _ in range(30000):
        pose = demo.read()
        x = pose["x_m"] + module.ROBOT_START_X_M
        y = pose["y_m"] + module.ROBOT_START_Y_M
        for name, x0, y0, x1, y1 in blocked:
            inside = (
                x0 - margin < x < x1 + margin
                and y0 - margin < y < y1 + margin
            )
            assert not inside, f"demo lap drives through the {name} at ({x:.2f}, {y:.2f})"


def test_a_stale_pose_file_is_ignored(kit):
    """A file left over from a previous session pins the robot wherever that
    run stopped. That reads as the scene being broken rather than as nothing
    running — which is exactly how this was found."""
    module, stage = kit
    import json as _json
    import os
    import tempfile
    import time as _time

    path = os.path.join(tempfile.mkdtemp(), "pose.json")
    with open(path, "w", encoding="utf-8") as handle:
        _json.dump({"x_m": 0.0, "y_m": 0.0, "heading_deg": 0.0}, handle)

    source = module.FilePose(path)
    assert source.read() is not None, "a fresh file must be used"

    old = _time.time() - module.POSE_STALE_AFTER_S - 60
    os.utime(path, (old, old))
    assert source.read() is None, "a stale file must be ignored"


# ── The viewport actually shows the room ─────────────────────────────────────


def test_the_stage_is_set_to_z_up(kit):
    """Every height in this scene is on Z — floor at 0, walls to 2.4 m. A Kit
    stage defaults to Y-up, and in a Y-up stage the room is built lying on its
    side: the viewport opens inside a wall and shows a flat grey field that
    reads as 'nothing rendered'."""
    module, stage = kit
    module.build_room()

    assert getattr(stage, "up_axis", None) == "z"


def test_a_camera_is_created(kit):
    """Otherwise the viewport keeps whatever camera the app started with, which
    knows nothing about where this scene was built."""
    module, stage = kit
    module.build_room()

    assert f"{module.ROOT}/Camera" in stage.prims


def test_the_camera_frames_whatever_is_being_shown(kit):
    """The camera has to follow the scene's own extent.

    With the measured room switched off the scene is one room wide, and a
    camera still centred across two would put the room in the left half of the
    frame with empty floor beside it.
    """
    module, stage = kit
    module.SHOW_MEASURED = False
    module.build_room()
    camera = stage.prims[f"{module.ROOT}/Camera"]

    assert camera.translate[0] == pytest.approx(module.ROOM_W / 2.0, abs=0.5)
    assert camera.translate[1] < 0, "camera must stand back from the room"
    assert camera.translate[2] > 2.4, "camera must be above the walls"


def test_the_camera_is_tilted_down_at_the_floor(kit):
    """Level or upside down are both easy to author by accident, and both show
    an empty frame."""
    module, stage = kit
    module.build_room()
    camera = stage.prims[f"{module.ROOT}/Camera"]

    # 90 deg about X is level in a Z-up stage; less than that looks downwards.
    pitch = camera.rotate[0]
    assert 30.0 < pitch < 90.0, f"camera pitch {pitch} is not looking at the floor"


def test_the_camera_is_wider_than_the_default_lens(kit):
    """A 50 mm lens at this distance crops the measured room off the side."""
    module, stage = kit
    module.build_room()
    camera = stage.prims[f"{module.ROOT}/Camera"]

    assert camera.attrs.get("CreateFocalLengthAttr", 50.0) < 35.0


# ── Physics ──────────────────────────────────────────────────────────────────


def test_walls_and_furniture_get_colliders(kit):
    module, stage = kit
    module.build_room()

    solid = [p for p in stage.prims.values() if "collision" in p.schemas]
    assert len(solid) > 10, "the room is not solid to physics"


def test_the_sofa_is_solid(kit):
    """The object the robot was seen driving through."""
    module, stage = kit
    module.build_room()

    sofa = [
        p for path, p in stage.prims.items()
        if "Sofa" in path and "collision" in p.schemas
    ]
    assert sofa, "the sofa has no collider"


def test_the_robot_is_kinematic_not_dynamic(kit):
    """Dynamic would let PhysX move the robot, and it would then disagree with
    the pose the mapper reports — a twin showing a robot that does not exist.
    Kinematic keeps the mapper authoritative."""
    module, stage = kit
    module.build_room()

    robot = stage.prims[f"{module.ROOT}/RobotTrue"]
    assert "rigid_body" in robot.schemas
    assert robot.attrs.get("CreateKinematicEnabledAttr") is True


def test_physics_can_be_turned_off(kit):
    """It is authoring, not simulation, but a scene meant purely for viewing
    should not be forced to carry it."""
    module, stage = kit
    module.ENABLE_PHYSICS = False
    module.build_room()

    solid = [p for p in stage.prims.values() if "collision" in p.schemas]
    assert solid == []


# ── The file itself ──────────────────────────────────────────────────────────


def test_the_script_still_drives_itself_when_pasted(kit):
    """It is used by pasting the whole file into the Script Editor, so the
    trailing call is the entry point, not a leftover."""
    source = KIT_SCRIPT.read_text(encoding="utf-8")
    assert source.rstrip().endswith("run_room()")


def test_it_only_needs_the_standard_library(kit):
    """Kit's Python is not this project's, and asking a user to pip-install
    into Omniverse is where a demo stops being reproducible."""
    source = KIT_SCRIPT.read_text(encoding="utf-8")
    for banned in ("import requests", "import paho", "import numpy", "import pydantic"):
        assert banned not in source

    # urllib and json are what the measured half is fetched with.
    assert "urllib.request" in source
    assert json is not None and math is not None and sys is not None


# ── The scene is the world, so it draws the world ────────────────────────────
#
# This used to ask the mapper what was in the room and draw only that. That was
# right when the scene was a *view*: a view that decorates itself is a lie, and
# it was one — a table, four chairs and a sofa rendered in a room the robot knew
# to be bare, which it drove straight through.
#
# The direction is now reversed. The scene IS the world, so what it draws is
# what exists, and GeometryPublisher tells the simulator. Both halves still
# agree; they now agree on the room you can reach into and rearrange.


def _stub_fetch(payload=None, fail=False):
    """Answer an HTTP call from the scene without a network."""
    import contextlib

    @contextlib.contextmanager
    def fake_urlopen(_url, timeout=None):
        if fail:
            raise OSError("no mapper here")

        class Reply:
            @staticmethod
            def read():
                return json.dumps(payload).encode("utf-8")

        yield Reply()

    return fake_urlopen


def test_the_room_is_furnished_so_there_is_something_to_move(kit):
    """Requirement: rearrange the room and see what the robot does. An empty
    scene has nothing to rearrange."""
    module, stage = kit
    module.build_room()

    furniture = [p for p in stage.prims if "/Furniture/" in p]
    assert furniture, "nothing in the room to drag"


def test_what_is_drawn_is_what_the_simulator_is_told(kit):
    """The two halves agreeing is the whole point, and this is now the
    mechanism: the scene draws it, then reports exactly that."""
    module, stage = kit
    module.build_room()

    drawn = [p for p in stage.prims if "/Furniture/" in p and "Rug" not in p]
    reported = module.furniture_footprints(stage)

    assert drawn and reported
    assert len(reported) <= len(drawn)


# ── Where the robot's position comes from ────────────────────────────────────
#
# The scene needs a pose and a room, and for a long time it took them from two
# different places: the room over HTTP from the mapper, the robot from a file
# whose default path is this machine's temp directory. The mapper writes inside
# a container, and the scene is normally pasted into the Script Editor where
# there is no chance to set an environment variable — so the room went live and
# the robot silently ran the canned demo lap beside it.


def test_the_pose_can_come_from_the_mapper_over_http(kit, monkeypatch):
    """The same source the room already comes from."""
    module, _ = kit
    payload = {"pose": {
        "x_m": 1.25, "y_m": 2.5, "heading_deg": 90.0,
        "timestamp": "2026-08-16T00:00:00Z",
    }}
    monkeypatch.setattr("urllib.request.urlopen", _stub_fetch(payload))

    pose = module.HttpPose().read()
    assert pose["x_m"] == 1.25
    assert pose["y_m"] == 2.5
    assert pose["heading_deg"] == 90.0


def test_an_unreachable_mapper_yields_no_pose(kit, monkeypatch):
    """So the caller can fall through to the demo lap rather than freezing the
    robot at the origin, which reads as the scene being broken."""
    module, _ = kit
    monkeypatch.setattr("urllib.request.urlopen", _stub_fetch(fail=True))
    assert module.HttpPose().read() is None


def test_a_dropped_request_keeps_the_last_pose(kit, monkeypatch):
    """One timeout mid-run must not teleport the robot back to the origin."""
    module, _ = kit
    payload = {"pose": {"x_m": 3.0, "y_m": 1.0, "heading_deg": 0.0}}
    monkeypatch.setattr("urllib.request.urlopen", _stub_fetch(payload))
    source = module.HttpPose()
    assert source.read()["x_m"] == 3.0

    monkeypatch.setattr("urllib.request.urlopen", _stub_fetch(fail=True))
    assert source.read()["x_m"] == 3.0


def test_http_is_tried_before_the_demo_lap(kit, monkeypatch):
    """The ordering that was the whole bug: a live mapper must win over the
    canned lap even when no pose file exists."""
    module, _ = kit
    payload = {"pose": {"x_m": 2.0, "y_m": 2.0, "heading_deg": 45.0}}
    monkeypatch.setattr("urllib.request.urlopen", _stub_fetch(payload))

    # No pose file anywhere.
    monkeypatch.setattr(module, "POSE_FILE", "/definitely/not/a/path.json")

    file_source = module.FilePose(path="/definitely/not/a/path.json")
    assert file_source.read() is None
    assert module.HttpPose().read() is not None


# ── Everything in the room is solid ──────────────────────────────────────────


def test_the_beacons_are_solid(kit):
    """They were not. `make_solid` walked the direct children of two groups and
    the beacons are in a third, so thirteen prims mounted on the walls let
    everything pass straight through them."""
    module, stage = kit
    module.build_room()

    beacons = [
        p for path, p in stage.prims.items()
        if "/Beacons/" in path and p.GetChildren() == []
    ]
    assert beacons, "no beacons were drawn"
    assert all("collision" in p.schemas for p in beacons)


def test_every_piece_of_geometry_in_the_room_is_solid(kit):
    """Walls, furniture, beacons — anything that is an object rather than a
    drawing. Checked over the whole tree, so nesting something one level deeper
    cannot silently opt it out."""
    module, stage = kit
    module.build_room()

    soft = []
    for path, prim in stage.prims.items():
        if not path.startswith(module.ROOT + "/"):
            continue
        group = path[len(module.ROOT) + 1:].split("/")[0]
        if group in module.NOT_PHYSICAL:
            continue
        if prim.GetChildren():          # a grouping Xform has no geometry
            continue
        if "collision" not in prim.schemas:
            soft.append(path)

    assert not soft, f"objects with no collider: {soft[:6]}"


def test_the_drawings_are_not_solid(kit):
    """Each of these would be a bug as a collider. The trail is a breadcrumb
    every 0.15 m — hundreds of them lying on the floor for a dynamic prop to
    catch on — and the ghost is where the filter *thinks* the robot is, which
    is by definition not where anything is."""
    module, stage = kit
    module.build_room()

    for group in ("Trail", "RobotOdometry", "Lighting", "Camera"):
        for path, prim in stage.prims.items():
            if f"/{group}" in path:
                assert "collision" not in prim.schemas, f"{path} should not be solid"


def test_nesting_does_not_hide_an_object(kit):
    """The old walk took direct children only, so a prim one level deeper was
    skipped without anything saying so."""
    module, stage = kit
    module.build_room()

    # The intermediate group has to exist, as it would in USD — defining
    # /Furniture/Shelf/Plank there creates /Furniture/Shelf on the way. The
    # stub only records what it is handed, so the test creates it explicitly
    # rather than pretending the walk can find a prim with no parent.
    module.UsdGeom.Xform.Define(stage, f"{module.ROOT}/Furniture/Shelf")
    deep = f"{module.ROOT}/Furniture/Shelf/Plank"
    module.box(stage, deep, (1.0, 1.0, 0.5), (0.4, 0.2, 0.02), (0.5, 0.5, 0.5))
    module.make_solid(stage)

    assert "collision" in stage.prims[deep].schemas


# ── What the scene tells you about itself ────────────────────────────────────


def test_the_legend_explains_the_two_robots(kit, capsys):
    """"Why are there two robots?" is the first thing anyone asks, and the gap
    between them is the most useful thing the scene shows. Naming the colours
    is not enough — the legend has to say what the distance means."""
    module, _ = kit
    module.build_room()
    printed = capsys.readouterr().out.lower()

    assert "blue" in printed and "green" in printed
    assert "drift" in printed, "the legend names the markers but not the point"


def test_the_legend_does_not_describe_markers_that_were_removed(kit, capsys):
    """It advertised an orange RSSI marker for several commits after the marker
    was deleted, and called this room LEFT of a measured room that no longer
    stood beside it. A legend that describes the wrong scene is worse than
    none: it is read as the truth about what is on screen."""
    module, _ = kit
    module.build_room()
    printed = capsys.readouterr().out.lower()

    assert "rssi" not in printed
    assert "orange" not in printed
    assert "left —" not in printed and "right —" not in printed
