"""The room must contain the same furniture everywhere it is described.

Why this exists
---------------
The furnished demo room lived in three places:

* `simulator/virtual_robot.py` — what the robot can physically hit
* `services/mapper/world.py` — what the browser twin draws
* `omniverse/kit_room_3d.py` — what Omniverse draws

They drifted. The two renderers drew a sofa, four chairs and a bin; the
simulator knew only a table and a cabinet. So the robot drove clean through a
sofa that was plainly on screen, and looked like broken collision detection
when it was in fact obeying a world with no sofa in it.

Nothing catches that except a test. A disagreement about what is in the room
produces perfectly working code on both sides and a nonsense result in the
middle.

`room_layout.py` is now the single description. The simulator and the mapper
import it. `kit_room_3d.py` cannot — it runs inside Omniverse's own Python,
which cannot see this project — so it carries an inlined copy, and the copy is
compared here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from robotmap_common.room_layout import (
    FURNISHED_ROOM,
    ROOM_HEIGHT_M,
    ROOM_WIDTH_M,
    by_name,
    furnished_room_edges,
    total_blocked_area_m2,
)

KIT_SCRIPT = Path(__file__).resolve().parents[1] / "omniverse" / "kit_room_3d.py"


# ── The layout itself ────────────────────────────────────────────────────────


def test_every_piece_of_furniture_is_inside_the_room():
    for item in FURNISHED_ROOM:
        assert 0 <= item.min_x_m and item.max_x_m <= ROOM_WIDTH_M, item.name
        assert 0 <= item.min_y_m and item.max_y_m <= ROOM_HEIGHT_M, item.name


def test_the_furniture_does_not_overlap_itself():
    """Overlapping footprints would double-count blocked floor."""
    items = list(FURNISHED_ROOM)
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            apart = (
                a.max_x_m <= b.min_x_m or b.max_x_m <= a.min_x_m
                or a.max_y_m <= b.min_y_m or b.max_y_m <= a.min_y_m
            )
            assert apart, f"{a.name} overlaps {b.name}"


def test_the_room_is_not_mostly_furniture():
    """A room that is 40 % table tells you nothing about mapping accuracy."""
    assert total_blocked_area_m2() < 0.35 * ROOM_WIDTH_M * ROOM_HEIGHT_M


def test_each_footprint_becomes_four_wall_segments():
    assert len(furnished_room_edges()) == 4 * len(FURNISHED_ROOM)


def test_the_rug_is_not_an_obstacle():
    """It is 12 mm thick and the robot drives over it. Listing it would
    subtract nearly 4 m2 of perfectly usable floor from every measurement."""
    with pytest.raises(KeyError):
        by_name("rug")


# ── The simulator agrees ─────────────────────────────────────────────────────


def test_the_robot_can_hit_everything_that_is_drawn():
    """The bug this whole module exists for."""
    from simulator.virtual_robot import VirtualWorld

    world = VirtualWorld.room_with_furniture(ROOM_WIDTH_M, ROOM_HEIGHT_M)
    walls = {(round(w.x1, 3), round(w.y1, 3), round(w.x2, 3), round(w.y2, 3))
             for w in world.walls}

    for edge in furnished_room_edges():
        rounded = tuple(round(v, 3) for v in edge)
        assert rounded in walls, f"the simulator cannot collide with {rounded}"


def test_the_sofa_is_solid_to_the_robot():
    """Named explicitly because it is the one that was seen being driven
    through, in all three views."""
    from simulator.virtual_robot import VirtualWorld

    sofa = by_name("sofa")
    world = VirtualWorld.room_with_furniture(ROOM_WIDTH_M, ROOM_HEIGHT_M)

    # A ray fired across the middle of the sofa must stop at its near face.
    distance = world.raycast(
        sofa.centre_x_m, sofa.min_y_m - 1.0, bearing_deg=90.0, max_range=5.0
    )
    assert distance == pytest.approx(1.0, abs=0.05)


def test_the_robot_cannot_drive_through_a_chair():
    from simulator.virtual_robot import VirtualWorld

    chair = by_name("chair_west")
    world = VirtualWorld.room_with_furniture(ROOM_WIDTH_M, ROOM_HEIGHT_M)

    distance = world.raycast(
        chair.min_x_m - 0.5, chair.centre_y_m, bearing_deg=0.0, max_range=3.0
    )
    assert distance == pytest.approx(0.5, abs=0.05)


# ── The mapper's description agrees ──────────────────────────────────────────


def test_the_browser_twin_draws_the_same_furniture():
    from mapper.world import describe_world

    world = describe_world("furnished")
    drawn = {
        b.name: b for b in world.boxes
        if b.kind == "furniture" and not b.name.endswith("_back")
    }

    for item in FURNISHED_ROOM:
        # world.py names the chairs chair_0..3; match on position instead.
        match = [
            b for b in drawn.values()
            if abs(b.x_m - item.centre_x_m) < 0.01
            and abs(b.y_m - item.centre_y_m) < 0.01
        ]
        assert match, f"the browser twin does not draw {item.name}"
        assert match[0].width_m == pytest.approx(item.width_m, abs=0.01)
        assert match[0].depth_m == pytest.approx(item.depth_m, abs=0.01)


# ── The Omniverse copy agrees ────────────────────────────────────────────────


def _kit_numbers(pattern: str) -> list[tuple[float, ...]]:
    source = KIT_SCRIPT.read_text(encoding="utf-8")
    return [
        tuple(float(v) for v in match.groups())
        for match in re.finditer(pattern, source)
    ]


def test_the_omniverse_scene_uses_the_same_room_size():
    source = KIT_SCRIPT.read_text(encoding="utf-8")
    assert f"ROOM_W = {ROOM_WIDTH_M}" in source
    assert f"ROOM_H = {ROOM_HEIGHT_M}" in source


def test_the_omniverse_scene_puts_the_table_in_the_same_place():
    """The inlined copy cannot import the layout, so it is checked by reading
    the shipped file — the same arrangement used for the kiwi-drive maths."""
    table = by_name("table")
    found = _kit_numbers(r"table_x, table_y = ([\d.]+), ([\d.]+)")

    assert found, "could not find the table position in the Kit scene"
    assert found[0][0] == pytest.approx(table.centre_x_m)
    assert found[0][1] == pytest.approx(table.centre_y_m)


def test_the_omniverse_scene_puts_the_sofa_in_the_same_place():
    sofa = by_name("sofa")
    found = _kit_numbers(r"SofaBase\", \(([\d.]+), ([\d.]+), [\d.]+\), \(([\d.]+), ([\d.]+),")

    assert found, "could not find the sofa in the Kit scene"
    x, y, width, depth = found[0]
    assert x == pytest.approx(sofa.centre_x_m)
    assert y == pytest.approx(sofa.centre_y_m)
    assert width == pytest.approx(sofa.width_m)
    assert depth == pytest.approx(sofa.depth_m)


def test_the_omniverse_scene_puts_the_cabinet_in_the_same_place():
    cabinet = by_name("cabinet")
    found = _kit_numbers(r"Cabinet\", \(([\d.]+), ([\d.]+), [\d.]+\), \(([\d.]+), ([\d.]+),")

    assert found, "could not find the cabinet in the Kit scene"
    x, y, width, depth = found[0]
    assert x == pytest.approx(cabinet.centre_x_m)
    assert y == pytest.approx(cabinet.centre_y_m)
    assert width == pytest.approx(cabinet.width_m)
    assert depth == pytest.approx(cabinet.depth_m)
