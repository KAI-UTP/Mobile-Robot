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

from robotmap_common.room_layout import FURNISHED_ROOM, by_name


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

    # Where the pose estimate's origin sits in this room.
    #
    # The filter starts every run at (0, 0) wherever the robot happens to be
    # standing, so pose coordinates are relative to the start point and are
    # routinely negative. Anything drawing a pose into a room laid out from a
    # corner — which is every 3D view here — must add this first, or the robot
    # appears outside its own room. That is not a small offset either: it is
    # the whole distance from the corner to wherever the robot was set down.
    robot_start_x_m: float = 1.0
    robot_start_y_m: float = 1.0

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
            "robot_start_x_m": self.robot_start_x_m,
            "robot_start_y_m": self.robot_start_y_m,
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


# How each footprint is dressed for display: (z centre, height, colour, label).
# The layout says where the furniture is; this says what it looks like.
_STYLE = {
    "table": (0.74, 0.06, TABLE, "Table"),
    "chair_west": (0.45, 0.05, CHAIR, "Chair"),
    "chair_east": (0.45, 0.05, CHAIR, "Chair"),
    "chair_south": (0.45, 0.05, CHAIR, "Chair"),
    "chair_north": (0.45, 0.05, CHAIR, "Chair"),
    "sofa": (0.22, 0.44, SOFA, "Sofa"),
    "cabinet": (0.55, 1.10, CABINET, "Cabinet"),
    "bin": (0.18, 0.36, CHAIR, "Bin"),
}


def _furniture(width: float, height: float) -> list[Box]:
    """A deliberately awkward layout.

    The table sits in open floor so the robot has to go round it, and the sofa
    cuts a corner — which is where wall-following most often loses the wall.
    An empty room would flatter the mapping.

    Positions come from `robotmap_common.room_layout`, the single description
    of what is in this room, so the browser twin, the Omniverse scene and the
    robot's own world cannot disagree about it. They previously did: this
    function clamped the table to `min(2.3, height / 2)`, which in a 4.5 m room
    put it 5 cm from where both 3D scenes drew it.
    """
    boxes = [
        # Display only — 12 mm thick, and the robot drives straight over it, so
        # it is not in the layout and must not count as blocked floor.
        Box("rug", 2.6, 2.3, 0.006, 2.4, 1.6, 0.012, RUG, "Rug", "rug"),
    ]

    for item in FURNISHED_ROOM:
        z_m, height_m, colour, label = _STYLE[item.name]
        boxes.append(
            Box(
                item.name,
                item.centre_x_m, item.centre_y_m, z_m,
                item.width_m, item.depth_m, height_m,
                colour, label, "furniture",
            )
        )

    # Backs and cushions: they sit above the footprints already listed, so they
    # add nothing to the floor area and exist purely so the room reads as a
    # room rather than a set of slabs.
    for item in FURNISHED_ROOM:
        if not item.name.startswith("chair_"):
            continue
        along_x = item.name in ("chair_south", "chair_north")
        offset = 0.24 if item.name in ("chair_east", "chair_north") else -0.24
        boxes.append(
            Box(
                f"{item.name}_back",
                item.centre_x_m + (0.0 if along_x else offset),
                item.centre_y_m + (offset if along_x else 0.0),
                0.68,
                0.42 if along_x else 0.06,
                0.06 if along_x else 0.42,
                0.45, CHAIR, "Chair", "furniture",
            )
        )

    sofa = by_name("sofa")
    boxes.append(
        Box("sofa_back", sofa.centre_x_m, sofa.max_y_m - 0.09, 0.58,
            sofa.width_m, 0.18, 0.72, SOFA, "Sofa", "furniture")
    )

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


def describe_world(
    room_name: str,
    is_ground_truth: bool = True,
    robot_start: tuple[float, float] = (1.0, 1.0),
) -> WorldDescription:
    """Build the description matching a simulator room.

    `robot_start` is where in the room the robot began, which is the origin of
    the pose estimate's frame. The simulator knows it exactly; with real
    hardware it is where the operator set the robot down, and it has to be told.
    """
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
        # The default rectangular room. EMPTY, because the simulator's walls
        # are bare and this description has to be the truth about the room the
        # robot is in.
        #
        # It used to be drawn furnished "so the twin view is worth looking at",
        # and that made every 3D view a lie: the browser and Omniverse showed a
        # table, four chairs and a sofa that the robot's world did not contain,
        # so the robot drove straight through all of them and the 2D map came
        # back as a bare rectangle. Three views, three different rooms.
        #
        # A twin that decorates itself is not a twin. Use --room furnished for
        # a room with furniture in it, and the simulator will have the
        # furniture too.
        world = WorldDescription(
            name="Empty rectangular room",
            width_m=6.0,
            height_m=4.5,
            boxes=_walls(6.0, 4.5, 2.4, door_x=4.3),
            beacons=_beacons(6.0, 4.5),
        )

    world.robot_start_x_m, world.robot_start_y_m = robot_start
    world.is_ground_truth = is_ground_truth
    world.source_note = (
        "Ground truth — the simulator built this room, so the comparison is exact."
        if is_ground_truth
        else "Reference layout, not a measurement. With a real robot the true "
             "room is unknown; this is what the room is expected to be."
    )
    return world
