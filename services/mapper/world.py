"""Describe the physical room, so the twin view can draw it.

A digital twin needs both halves: the thing, and the model of the thing. The
occupancy grid is the model — what the robot has worked out. This module
supplies the other half: what is actually there.

Where the truth comes from depends on the mode, and the view says which:

* **Simulator** — the world is known exactly, because the simulator built it.
  The 3D view is genuine ground truth and the comparison is meaningful.
* **Real robot** — nothing here is known. The reference room is a stated
  expectation, not a measurement, and the view labels it as such rather than
  implying the robot's map is being checked against reality.

Rooms are described as boxes rather than a mesh format. Everything in a room
at this level of detail is box-shaped, boxes are trivial to depth-sort and
shade, and the description stays readable in a JSON response.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Box:
    """An axis-aligned box, positioned by its centre.

    `label` is what the legend and tooltips show, so it is written for a
    person rather than as an identifier.
    """

    name: str
    x_m: float
    y_m: float
    z_m: float
    width_m: float
    depth_m: float
    height_m: float
    colour: str
    label: str = ""
    kind: str = "furniture"


@dataclass
class WorldDescription:
    """Everything needed to draw the physical room."""

    name: str
    width_m: float
    height_m: float
    wall_height_m: float = 2.4
    boxes: list[Box] = field(default_factory=list)
    beacons: list[dict] = field(default_factory=list)
    is_ground_truth: bool = True
    source_note: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "wall_height_m": self.wall_height_m,
            "boxes": [asdict(b) for b in self.boxes],
            "beacons": self.beacons,
            "is_ground_truth": self.is_ground_truth,
            "source_note": self.source_note,
            "floor_area_m2": round(self.width_m * self.height_m, 2),
        }


# ── Colours ──────────────────────────────────────────────────────────────────
# Kept in step with omniverse/kit_room_3d.py so the browser view and the
# Omniverse view are recognisably the same room.

WALL = "#dcd9d2"
FLOOR = "#bcb09a"
TABLE = "#73502f"
CHAIR = "#4d4744"
SOFA = "#526174"
CABINET = "#856044"
DOOR = "#a68159"
WINDOW = "#9ec7e0"
RUG = "#5a6b80"


def _walls(width: float, height: float, wall_height: float, door_x: float | None) -> list[Box]:
    """Four walls, optionally with a doorway gap in the south wall.

    The doorway is two segments and a lintel rather than a solid wall, because
    an opening is what makes the room realistic — and it is exactly where a
    real robot drives out and gets lost.
    """
    thickness = 0.12
    half = wall_height / 2
    boxes: list[Box] = []

    if door_x is None:
        boxes.append(Box("wall_s", width / 2, 0, half, width, thickness, wall_height, WALL, "Wall", "wall"))
    else:
        door_w = 0.9
        left_w = door_x - door_w / 2
        right_start = door_x + door_w / 2
        right_w = width - right_start

        boxes.append(Box("wall_s_left", left_w / 2, 0, half, left_w, thickness, wall_height, WALL, "Wall", "wall"))
        boxes.append(Box("wall_s_right", right_start + right_w / 2, 0, half, right_w, thickness, wall_height, WALL, "Wall", "wall"))
        boxes.append(Box("wall_s_lintel", door_x, 0, wall_height - 0.25, door_w, thickness, 0.5, WALL, "Above the door", "wall"))
        boxes.append(Box("door", door_x + 0.42, 0.42, 1.0, 0.06, 0.85, 2.0, DOOR, "Door", "door"))

    boxes.append(Box("wall_n", width / 2, height, half, width, thickness, wall_height, WALL, "Wall", "wall"))
    boxes.append(Box("wall_w", 0, height / 2, half, thickness, height, wall_height, WALL, "Wall", "wall"))
    boxes.append(Box("wall_e", width, height / 2, half, thickness, height, wall_height, WALL, "Wall", "wall"))
    boxes.append(Box("window", min(2.0, width / 2), height, 1.45, 1.6, 0.04, 1.1, WINDOW, "Window", "window"))

    return boxes


def _furniture(width: float, height: float) -> list[Box]:
    """A deliberately awkward layout.

    The table sits in open floor so the robot has to go round it, and the sofa
    cuts a corner — which is where wall-following most often loses the wall.
    An empty room would flatter the mapping.
    """
    table_x, table_y = min(2.6, width / 2), min(2.3, height / 2)
    boxes = [
        Box("rug", table_x, table_y, 0.006, 2.4, 1.6, 0.012, RUG, "Rug", "rug"),
        Box("table", table_x, table_y, 0.74, 1.4, 0.85, 0.06, TABLE, "Table", "furniture"),
    ]

    for index, (dx, dy) in enumerate(((-1.0, 0.0), (1.0, 0.0), (0.0, -0.75), (0.0, 0.75))):
        boxes.append(
            Box(f"chair_{index}", table_x + dx, table_y + dy, 0.45,
                0.42, 0.42, 0.05, CHAIR, "Chair", "furniture")
        )
        boxes.append(
            Box(f"chair_{index}_back", table_x + dx * 1.1, table_y + dy * 1.1, 0.68,
                0.42 if abs(dy) > 0 else 0.06, 0.06 if abs(dy) > 0 else 0.42,
                0.45, CHAIR, "Chair", "furniture")
        )

    boxes.append(Box("sofa", 1.1, height - 0.8, 0.22, 2.0, 0.85, 0.44, SOFA, "Sofa", "furniture"))
    boxes.append(Box("sofa_back", 1.1, height - 0.45, 0.58, 2.0, 0.18, 0.72, SOFA, "Sofa", "furniture"))
    boxes.append(Box("cabinet", width - 0.4, 2.6, 0.55, 0.45, 1.5, 1.10, CABINET, "Cabinet", "furniture"))
    boxes.append(Box("bin", width - 0.45, 0.6, 0.18, 0.32, 0.32, 0.36, CHAIR, "Bin", "furniture"))

    return boxes


def _beacons(width: float, height: float) -> list[dict]:
    """Four BLE beacons in the corners.

    Corners maximise the spread of bearings from anywhere inside. All four on
    one wall would make trilateration impossible, so showing where they are is
    worth doing.
    """
    return [
        {"id": f"B{index + 1}", "x_m": x, "y_m": y, "z_m": 2.1}
        for index, (x, y) in enumerate(
            ((0.15, 0.15), (width - 0.15, 0.15), (width - 0.15, height - 0.15), (0.15, height - 0.15))
        )
    ]


# ── Named worlds ─────────────────────────────────────────────────────────────


def describe_world(room_name: str, is_ground_truth: bool = True) -> WorldDescription:
    """Build the description matching a simulator room."""
    if room_name == "l-shaped":
        # The L is not expressible as one rectangle, so the twin view draws the
        # bounding room and the extra wall that forms the notch.
        world = WorldDescription(
            name="L-shaped room",
            width_m=6.0,
            height_m=5.0,
            boxes=_walls(6.0, 5.0, 2.4, door_x=None),
        )
        world.boxes.append(
            Box("wall_notch", 4.75, 3.0, 1.2, 2.5, 0.12, 2.4, WALL, "Wall", "wall")
        )
        world.boxes.append(
            Box("wall_notch_2", 3.5, 4.0, 1.2, 0.12, 2.0, 2.4, WALL, "Wall", "wall")
        )
        world.beacons = _beacons(6.0, 5.0)

    elif room_name == "furnished":
        world = WorldDescription(
            name="Furnished room",
            width_m=6.0,
            height_m=4.5,
            boxes=_walls(6.0, 4.5, 2.4, door_x=4.3) + _furniture(6.0, 4.5),
            beacons=_beacons(6.0, 4.5),
        )

    else:
        # The default rectangular room, drawn furnished so the twin view is
        # worth looking at even though the simulator's walls are bare.
        world = WorldDescription(
            name="Rectangular room",
            width_m=6.0,
            height_m=4.5,
            boxes=_walls(6.0, 4.5, 2.4, door_x=4.3) + _furniture(6.0, 4.5),
            beacons=_beacons(6.0, 4.5),
        )

    world.is_ground_truth = is_ground_truth
    world.source_note = (
        "Ground truth — the simulator built this room, so the comparison is exact."
        if is_ground_truth
        else "Reference layout, not a measurement. With a real robot the true "
             "room is unknown; this is what the room is expected to be."
    )
    return world
