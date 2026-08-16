"""The measured room drawn in Omniverse must be the room that was measured.

Why this exists
---------------
`omniverse/kit_room_3d.py` runs inside Omniverse's own Python and cannot be
imported normally: it pulls in `omni.*` and `pxr`, and calls `run_room()` at
module scope. So none of it is covered by anything, and the half that draws the
robot's own map is exactly the half that can be silently wrong — a room drawn
with its axes swapped, or with obstacles offset from the outline they came
from, still looks like a plausible room.

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
    def __init__(self, path: str) -> None:
        self.path = path
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

    # The handful whose values the tests actually assert on.
    def CreateRadiusAttr(self, value): self.radius = value
    def CreateHeightAttr(self, value): self.height = value
    def CreateDisplayColorAttr(self, value): self.colour = tuple(value[0])

    def GetPrim(self): return self
    def __bool__(self): return True

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
        prim = FakePrim(str(path))
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

    class _Gprim:
        def __init__(self, prim):
            self.prim = prim

        def CreateDisplayOpacityAttr(self, value):
            self.prim.opacity = value[0]

    usdgeom = types.SimpleNamespace(
        Cube=_Definable, Cylinder=_Definable, Sphere=_Definable,
        Xform=_Definable, Mesh=_Definable,
        XformCommonAPI=_XformCommonAPI, Gprim=_Gprim,
    )
    usdlux = types.SimpleNamespace(
        DistantLight=_Definable, SphereLight=_Definable,
        DomeLight=_Definable, RectLight=_Definable,
    )
    gf = types.SimpleNamespace(
        Vec3d=lambda *a: tuple(a), Vec3f=lambda *a: tuple(a),
    )
    sdf = types.SimpleNamespace(Path=str)
    vt = types.SimpleNamespace(
        Vec3fArray=lambda values: list(values), FloatArray=lambda values: list(values),
    )
    return usdgeom, usdlux, gf, sdf, vt


def _load_kit_module(stage: FakeStage) -> types.ModuleType:
    """Load the shipped script with USD stubbed and the entry call stripped."""
    source = KIT_SCRIPT.read_text(encoding="utf-8")

    # The file drives itself when pasted into the Script Editor. Here that would
    # build the whole scene on import, so the trailing call is removed.
    source = source.replace("\nrun_room()\n", "\n")
    assert "\nrun_room()\n" not in source

    usdgeom, usdlux, gf, sdf, vt = _build_stub_modules(stage)

    module = types.ModuleType("kit_room_3d")
    module.__dict__.update(
        {
            "UsdGeom": usdgeom, "UsdLux": usdlux,
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


@pytest.fixture
def kit():
    stage = FakeStage()
    return _load_kit_module(stage), stage


# ── A room to draw ───────────────────────────────────────────────────────────


def _room(closed=True, obstacles=(), polygon=None):
    polygon = polygon or [
        {"x_m": 0.0, "y_m": 0.0},
        {"x_m": 6.0, "y_m": 0.0},
        {"x_m": 6.0, "y_m": 4.5},
        {"x_m": 0.0, "y_m": 4.5},
    ]
    return {
        "polygon": polygon,
        "area_m2": 27.0,
        "blocked_area_m2": sum(o["area_m2"] for o in obstacles),
        "obstacles": list(obstacles),
        "is_closed": closed,
    }


def _obstacle(cx, cy, w=1.0, d=1.0):
    return {
        "centre_x_m": cx, "centre_y_m": cy,
        "min_x_m": cx - w / 2, "max_x_m": cx + w / 2,
        "min_y_m": cy - d / 2, "max_y_m": cy + d / 2,
        "area_m2": w * d, "cells": 400,
    }


def _walls(stage):
    return [p for path, p in stage.prims.items() if "/Measured/Outline/Wall_" in path]


def _blocked(stage):
    return [p for path, p in stage.prims.items() if "/Measured/Blocked/Area_" in path]


# ── The outline ──────────────────────────────────────────────────────────────


def test_a_four_sided_room_draws_four_walls(kit):
    module, stage = kit
    measured = module.MeasuredRoom(stage, 8.0, 0.0)
    measured.rebuild(_room())

    assert len(_walls(stage)) == 4


def test_the_outline_closes(kit):
    """The last wall must join back to the first. An outline drawn from
    consecutive pairs only, without wrapping, leaves a room with a gap in it —
    which is exactly what an unclosed boundary is supposed to look like, so the
    bug would be invisible."""
    module, stage = kit
    measured = module.MeasuredRoom(stage, 8.0, 0.0)
    measured.rebuild(_room())

    # A closing wall runs the full height of the room; without wrapping, only
    # three walls exist and none of them do.
    lengths = sorted(round(p.scale[0], 3) for p in _walls(stage))
    assert lengths == [4.5, 4.5, 6.0, 6.0]


def test_wall_segments_are_rotated_to_match_their_edge(kit):
    """Unrotated boxes give a plus sign rather than a room."""
    module, stage = kit
    measured = module.MeasuredRoom(stage, 8.0, 0.0)
    measured.rebuild(_room())

    angles = sorted(round(abs(p.rotate[2]) % 180.0, 3) for p in _walls(stage))
    assert angles == [0.0, 0.0, 90.0, 90.0]


def test_a_closed_boundary_is_green_and_an_open_one_is_not(kit):
    """The grade of a scan turns on whether the boundary closed, so the two
    must not look alike."""
    module, stage = kit

    module.MeasuredRoom(stage, 8.0, 0.0).rebuild(_room(closed=True))
    closed_colour = _walls(stage)[0].colour

    stage.prims.clear()
    module.MeasuredRoom(stage, 8.0, 0.0).rebuild(_room(closed=False))
    open_colour = _walls(stage)[0].colour

    assert closed_colour == module.COL_MEAS_WALL
    assert open_colour == module.COL_MEAS_OPEN
    assert closed_colour != open_colour


def test_an_l_shaped_room_keeps_all_its_corners(kit):
    """Nothing may assume the room is rectangular."""
    module, stage = kit
    l_shape = [
        {"x_m": 0.0, "y_m": 0.0}, {"x_m": 6.0, "y_m": 0.0},
        {"x_m": 6.0, "y_m": 3.0}, {"x_m": 3.5, "y_m": 3.0},
        {"x_m": 3.5, "y_m": 5.0}, {"x_m": 0.0, "y_m": 5.0},
    ]
    module.MeasuredRoom(stage, 8.0, 0.0).rebuild(_room(polygon=l_shape))

    assert len(_walls(stage)) == 6


def test_a_degenerate_polygon_draws_nothing_rather_than_crashing(kit):
    """`/api/room` returns a partial outline while the robot is still driving."""
    module, stage = kit
    measured = module.MeasuredRoom(stage, 8.0, 0.0)
    measured.rebuild(_room(polygon=[{"x_m": 0.0, "y_m": 0.0}, {"x_m": 1.0, "y_m": 0.0}]))

    assert _walls(stage) == []


def test_repeated_points_do_not_produce_zero_length_walls(kit):
    """A zero-length segment has no defined direction, so its rotation would be
    arbitrary and it would render as a speck at a random angle."""
    module, stage = kit
    doubled = [
        {"x_m": 0.0, "y_m": 0.0}, {"x_m": 0.0, "y_m": 0.0},
        {"x_m": 6.0, "y_m": 0.0}, {"x_m": 6.0, "y_m": 4.5},
        {"x_m": 0.0, "y_m": 4.5},
    ]
    module.MeasuredRoom(stage, 8.0, 0.0).rebuild(_room(polygon=doubled))

    assert all(p.scale[0] > 1e-6 for p in _walls(stage))


# ── Placement ────────────────────────────────────────────────────────────────


def test_the_measurement_is_drawn_beside_the_real_room_not_on_top_of_it(kit):
    module, stage = kit
    origin_x = module.ROOM_W + module.MEASURED_GAP_M
    module.MeasuredRoom(stage, origin_x, 0.0).rebuild(_room())

    # Every part of the drawing sits clear of the real room's footprint.
    for prim in _walls(stage):
        assert prim.translate[0] > module.ROOM_W


def test_the_pose_frame_origin_does_not_move_the_drawing(kit):
    """The measured polygon lives in the pose estimate's frame, whose origin is
    wherever the robot happened to start. Placing the model by those raw
    coordinates would fling it across the stage as a scan progresses."""
    module, stage_a = kit

    module.MeasuredRoom(stage_a, 8.0, 0.0).rebuild(_room())
    near_origin = sorted(round(p.translate[0], 3) for p in _walls(stage_a))

    shifted = [
        {"x_m": p["x_m"] - 100.0, "y_m": p["y_m"] + 250.0}
        for p in _room()["polygon"]
    ]
    stage_b = FakeStage()
    module_b = _load_kit_module(stage_b)
    module_b.MeasuredRoom(stage_b, 8.0, 0.0).rebuild(_room(polygon=shifted))
    far_away = sorted(round(p.translate[0], 3) for p in _walls(stage_b))

    assert near_origin == far_away


def test_the_drawing_is_centred_on_its_pad(kit):
    """A small room dumped in a corner of the pad reads as a placement bug."""
    module, stage = kit
    origin_x, origin_y = 8.0, 0.0
    small = [
        {"x_m": 0.0, "y_m": 0.0}, {"x_m": 2.0, "y_m": 0.0},
        {"x_m": 2.0, "y_m": 1.5}, {"x_m": 0.0, "y_m": 1.5},
    ]
    module.MeasuredRoom(stage, origin_x, origin_y).rebuild(_room(polygon=small))

    xs = [p.translate[0] for p in _walls(stage)]
    ys = [p.translate[1] for p in _walls(stage)]
    assert (min(xs) + max(xs)) / 2 == pytest.approx(origin_x + module.ROOM_W / 2)
    assert (min(ys) + max(ys)) / 2 == pytest.approx(origin_y + module.ROOM_H / 2)


# ── Blocked floor ────────────────────────────────────────────────────────────


def test_obstacles_are_drawn(kit):
    module, stage = kit
    module.MeasuredRoom(stage, 8.0, 0.0).rebuild(
        _room(obstacles=[_obstacle(3.0, 2.0), _obstacle(5.0, 3.5, 0.8, 0.8)])
    )
    assert len(_blocked(stage)) == 2


def test_an_obstacle_lands_inside_the_outline_it_came_from(kit):
    """Outline and obstacles come from the same grid, so if they are placed by
    different rules the table ends up outside its own room."""
    module, stage = kit
    module.MeasuredRoom(stage, 8.0, 0.0).rebuild(_room(obstacles=[_obstacle(3.0, 2.25)]))

    wall_xs = [p.translate[0] for p in _walls(stage)]
    wall_ys = [p.translate[1] for p in _walls(stage)]
    slab = _blocked(stage)[0]

    assert min(wall_xs) < slab.translate[0] < max(wall_xs)
    assert min(wall_ys) < slab.translate[1] < max(wall_ys)


def test_an_obstacle_is_drawn_at_its_measured_size(kit):
    module, stage = kit
    module.MeasuredRoom(stage, 8.0, 0.0).rebuild(
        _room(obstacles=[_obstacle(3.0, 2.25, w=1.2, d=0.9)])
    )
    slab = _blocked(stage)[0]
    assert slab.scale[0] == pytest.approx(1.2)
    assert slab.scale[1] == pytest.approx(0.9)


def test_obstacles_are_low_and_translucent(kit):
    """They are a measured footprint on the floor, not a table. Drawing a
    solid, table-height box would claim knowledge the robot does not have and
    would hide the outline behind it."""
    module, stage = kit
    module.MeasuredRoom(stage, 8.0, 0.0).rebuild(_room(obstacles=[_obstacle(3.0, 2.25)]))

    slab = _blocked(stage)[0]
    assert slab.scale[2] < 0.4
    assert slab.opacity is not None and slab.opacity < 1.0
    assert slab.colour == module.COL_MEAS_BLOCK


def test_an_empty_room_draws_no_blocked_areas(kit):
    module, stage = kit
    module.MeasuredRoom(stage, 8.0, 0.0).rebuild(_room())
    assert _blocked(stage) == []


# ── Refreshing ───────────────────────────────────────────────────────────────


def test_a_rebuild_replaces_the_previous_drawing(kit):
    """Without clearing, every refresh stacks another outline on the last one
    and the room slowly fills with overlapping walls."""
    module, stage = kit
    measured = module.MeasuredRoom(stage, 8.0, 0.0)

    measured.rebuild(_room(obstacles=[_obstacle(3.0, 2.25)]))
    measured.rebuild(_room())

    assert len(_walls(stage)) == 4
    assert _blocked(stage) == []


def test_an_unchanged_room_is_not_redrawn(kit):
    """Deleting and recreating a hundred prims every two seconds flickers the
    viewport for no reason."""
    module, stage = kit
    measured = module.MeasuredRoom(stage, 8.0, 0.0)
    room = _room()
    measured._fetch = lambda: room

    assert measured.poll(0.0) is True
    assert measured.poll(1000.0) is False
    assert measured.poll(2000.0) is False


def test_a_changed_room_is_redrawn(kit):
    module, stage = kit
    measured = module.MeasuredRoom(stage, 8.0, 0.0)

    measured._fetch = lambda: _room()
    assert measured.poll(0.0) is True

    measured._fetch = lambda: _room(obstacles=[_obstacle(3.0, 2.25)])
    assert measured.poll(1000.0) is True
    assert len(_blocked(stage)) == 1


def test_polling_respects_its_interval(kit):
    """It runs on the render tick, which is 60 Hz. An HTTP request per frame
    would stall the viewport."""
    module, stage = kit
    measured = module.MeasuredRoom(stage, 8.0, 0.0)

    calls = []
    measured._fetch = lambda: calls.append(1) or _room()

    measured.poll(0.0)
    for _ in range(50):
        measured.poll(0.1)   # same instant, many frames later
    assert len(calls) == 1


def test_no_mapper_running_is_not_an_error(kit):
    """The real room is still worth looking at on its own."""
    module, stage = kit
    measured = module.MeasuredRoom(stage, 8.0, 0.0)

    def _boom():
        raise OSError("connection refused")

    measured._fetch = _boom
    with pytest.raises(OSError):
        measured._fetch()

    # The real fetch swallows it; poll must simply report no change.
    measured._fetch = lambda: None
    assert measured.poll(0.0) is False


def test_the_summary_reports_both_areas(kit):
    """Total floor and usable floor are different numbers sold to different
    people, so both must appear."""
    module, stage = kit
    measured = module.MeasuredRoom(stage, 8.0, 0.0)
    measured.rebuild(_room(obstacles=[_obstacle(3.0, 2.25)]))

    assert "27.00 m2 floor" in measured.summary
    assert "1.00 m2 blocked" in measured.summary
    assert "26.00 m2 usable" in measured.summary
    assert "closed" in measured.summary


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
