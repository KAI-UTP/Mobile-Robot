"""The physical-world description that the 3D twin view renders.

The point of a digital twin is comparing the model against reality, so the
description of reality has to be right — and, more importantly, has to be
honest about when it *is not* reality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "mapper"))

from world import describe_world  # noqa: E402

# ── Shape ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("room", ["rectangular", "furnished", "l-shaped"])
def test_every_simulator_room_can_be_described(room):
    """A room the simulator can drive must be a room the twin view can draw."""
    world = describe_world(room)
    assert world.width_m > 0 and world.height_m > 0
    assert len(world.boxes) > 4


def test_unknown_room_falls_back_rather_than_failing():
    """An unrecognised name must not leave the twin view blank."""
    world = describe_world("something-else")
    assert world.boxes


def test_the_room_has_four_walls():
    boxes = describe_world("rectangular").boxes
    walls = [b for b in boxes if b.kind == "wall"]
    # Four sides, but the south wall is split around a doorway.
    assert len(walls) >= 4


def test_the_doorway_is_an_opening_not_a_solid_wall():
    """An opening is what makes the room realistic for the range sensors, and
    it is where a real robot drives out and gets lost."""
    boxes = describe_world("rectangular").boxes
    names = {b.name for b in boxes}
    assert "wall_s_left" in names and "wall_s_right" in names
    assert "wall_s" not in names
    assert any(b.kind == "door" for b in boxes)


def test_furniture_is_inside_the_room():
    """Furniture outside the walls would look broken and would misrepresent
    what the robot has to drive around."""
    world = describe_world("furnished")
    for box in world.boxes:
        if box.kind != "furniture":
            continue
        assert -0.5 <= box.x_m <= world.width_m + 0.5, box.name
        assert -0.5 <= box.y_m <= world.height_m + 0.5, box.name


def test_furniture_sits_on_or_above_the_floor():
    for box in describe_world("furnished").boxes:
        if box.kind == "furniture":
            assert box.z_m >= 0, box.name


def test_boxes_have_positive_extents():
    """A zero or negative dimension renders as an invisible or inverted box."""
    for room in ("rectangular", "furnished", "l-shaped"):
        for box in describe_world(room).boxes:
            assert box.width_m > 0 and box.depth_m > 0 and box.height_m > 0, box.name


# ── Beacons ──────────────────────────────────────────────────────────────────


def test_four_beacons_in_the_corners():
    """Corners maximise the bearing spread; one blocked beacon still leaves a
    solvable fix."""
    world = describe_world("rectangular")
    assert len(world.beacons) == 4

    xs = {round(b["x_m"], 1) for b in world.beacons}
    ys = {round(b["y_m"], 1) for b in world.beacons}
    assert len(xs) == 2 and len(ys) == 2


def test_beacons_are_not_collinear():
    """Four beacons in a line make trilateration mathematically impossible, so
    the drawn layout must not depict a broken install."""
    beacons = describe_world("rectangular").beacons
    assert len({round(b["y_m"], 2) for b in beacons}) > 1


def test_beacons_are_mounted_high():
    for beacon in describe_world("rectangular").beacons:
        assert beacon["z_m"] > 1.5


# ── Honesty about ground truth ───────────────────────────────────────────────


def test_simulator_mode_is_marked_ground_truth():
    world = describe_world("rectangular", is_ground_truth=True)
    assert world.is_ground_truth
    assert "ground truth" in world.source_note.lower()


def test_hardware_mode_does_not_claim_to_know_the_room():
    """The important one.

    With a real robot the true room is unknown. Presenting a reference layout
    as if it were measured would make the error figure look authoritative when
    it is really comparing the map against an assumption.
    """
    world = describe_world("rectangular", is_ground_truth=False)
    assert not world.is_ground_truth
    assert "not a measurement" in world.source_note.lower()


# ── Serialisation ────────────────────────────────────────────────────────────


def test_serialises_to_json_for_the_browser():
    import json

    payload = describe_world("furnished").to_dict()
    restored = json.loads(json.dumps(payload))

    assert restored["floor_area_m2"] == pytest.approx(27.0)
    assert len(restored["boxes"]) > 10
    assert len(restored["beacons"]) == 4


def test_floor_area_matches_the_dimensions():
    world = describe_world("rectangular")
    assert world.to_dict()["floor_area_m2"] == pytest.approx(
        world.width_m * world.height_m
    )


def test_rectangular_room_matches_the_simulator_truth():
    """The twin view's 'true area' must be the area the mapping tests compare
    against, or the two halves of the project disagree."""
    assert describe_world("rectangular").to_dict()["floor_area_m2"] == pytest.approx(27.0)


def test_every_box_has_a_colour_and_a_label():
    """The legend and tooltips are what make the view understandable."""
    for box in describe_world("furnished").boxes:
        assert box.colour.startswith("#")
        assert box.label
