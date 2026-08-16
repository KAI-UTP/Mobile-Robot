"""Where the furniture is, stated once.

Why this exists
---------------
The furnished demo room was described in three places that had quietly drifted
apart:

* `simulator/virtual_robot.py` — what the robot can physically hit
* `services/mapper/world.py` — what the browser twin draws
* `omniverse/kit_room_3d.py` — what Omniverse draws

The last two agreed. The simulator did not: it knew only about a table and a
cabinet, and had never heard of the sofa, the four chairs or the bin. So the
robot drove straight through a sofa that both renderers were drawing, and it
was not misbehaving — it was obeying a world where that sofa does not exist.
No amount of collision work in the simulator or colliders in the scene can fix
a disagreement about what is in the room.

This module is the answer to "what is in the room", and everything that needs
to know reads it from here.

The exception, and how it is kept honest
---------------------------------------
`kit_room_3d.py` cannot import it. That file is pasted into Omniverse's Script
Editor and runs under Kit's own Python, which cannot see this project. It
therefore carries an inlined copy, and `tests/test_room_layout_parity.py`
compares the two so the copy cannot drift back out of step — the same
arrangement already used for the kiwi-drive maths in `kit_holonomic.py`.

Footprints only
---------------
These are floor footprints: what the robot bumps into and what shows as
blocked floor. Heights, colours and the decorative parts of each object belong
to the renderers, which have more to say about how a sofa looks than about
where it is.
"""

from __future__ import annotations

from dataclasses import dataclass

# The demo room these coordinates describe. Anything else needs its own layout;
# the positions below are absolute, not proportional, because furniture does
# not scale with the room.
ROOM_WIDTH_M = 6.0
ROOM_HEIGHT_M = 4.5


@dataclass(frozen=True)
class Footprint:
    """One obstacle's floor area, by centre and extent."""

    name: str
    centre_x_m: float
    centre_y_m: float
    width_m: float
    depth_m: float

    @property
    def min_x_m(self) -> float:
        return self.centre_x_m - self.width_m / 2.0

    @property
    def max_x_m(self) -> float:
        return self.centre_x_m + self.width_m / 2.0

    @property
    def min_y_m(self) -> float:
        return self.centre_y_m - self.depth_m / 2.0

    @property
    def max_y_m(self) -> float:
        return self.centre_y_m + self.depth_m / 2.0

    @property
    def area_m2(self) -> float:
        return self.width_m * self.depth_m

    def corners(self) -> list[tuple[float, float]]:
        """Anticlockwise, for building wall segments."""
        return [
            (self.min_x_m, self.min_y_m),
            (self.max_x_m, self.min_y_m),
            (self.max_x_m, self.max_y_m),
            (self.min_x_m, self.max_y_m),
        ]

    def edges(self) -> list[tuple[float, float, float, float]]:
        """The four sides as (x1, y1, x2, y2)."""
        points = self.corners()
        return [
            (*points[i], *points[(i + 1) % 4]) for i in range(4)
        ]


# The table sits in open floor so the robot has to go round it, and the sofa
# cuts a corner — which is where wall-following most often loses the wall. An
# empty room would flatter the mapping.
_TABLE_X, _TABLE_Y = 2.6, 2.3

FURNISHED_ROOM: tuple[Footprint, ...] = (
    Footprint("table", _TABLE_X, _TABLE_Y, 1.4, 0.85),
    # Four chairs, at the table's sides. The gap between a chair and the table
    # is under 10 cm — narrower than the robot — which is deliberate: a real
    # dining set is not a shape you can thread between.
    Footprint("chair_west", _TABLE_X - 1.0, _TABLE_Y, 0.42, 0.42),
    Footprint("chair_east", _TABLE_X + 1.0, _TABLE_Y, 0.42, 0.42),
    Footprint("chair_south", _TABLE_X, _TABLE_Y - 0.75, 0.42, 0.42),
    Footprint("chair_north", _TABLE_X, _TABLE_Y + 0.75, 0.42, 0.42),
    # Against the far wall. A robot following that wall meets it head-on, which
    # is exactly the case the contact detection exists for.
    Footprint("sofa", 1.1, ROOM_HEIGHT_M - 0.8, 2.0, 0.85),
    Footprint("cabinet", ROOM_WIDTH_M - 0.4, 2.6, 0.45, 1.5),
    Footprint("bin", ROOM_WIDTH_M - 0.45, 0.6, 0.32, 0.32),
)

# The rug is deliberately absent. It is 12 mm thick and the robot drives over
# it, so it is not an obstacle — listing it would subtract 3.8 m2 of perfectly
# usable floor from every measurement.


def furnished_room_edges() -> list[tuple[float, float, float, float]]:
    """Every furniture side, as wall segments the simulator can raycast."""
    return [edge for item in FURNISHED_ROOM for edge in item.edges()]


def total_blocked_area_m2() -> float:
    """Ground truth for how much floor the furniture stands on."""
    return sum(item.area_m2 for item in FURNISHED_ROOM)


def by_name(name: str) -> Footprint:
    for item in FURNISHED_ROOM:
        if item.name == name:
            return item
    raise KeyError(name)
