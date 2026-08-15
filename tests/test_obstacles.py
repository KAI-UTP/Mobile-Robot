"""Finding furniture in the map, separately from walls.

A wall and a table leg look identical to a range sensor — both are just
"occupied". What separates them is where they sit: a wall encloses the room,
furniture is surrounded by the room's own floor.

Getting that right is what lets the usable floor area exclude the space a
table stands on, which is a different number from the total floor area and
matters to a different customer.
"""

from __future__ import annotations

import math

import pytest
from mapping.occupancy_grid import OccupancyGrid
from mapping.room_extraction import RoomExtractor
from robotmap_common.models import PoseEstimate, RangeReading


def _pose(x, y, heading=0.0):
    return PoseEstimate(
        robot_id="TEST01", timestamp="2026-08-16T00:00:00Z", sequence=0,
        x_m=x, y_m=y, heading_deg=heading % 360, std_x_m=0.01, std_y_m=0.01,
    )


class Room:
    """A rectangular room with optional solid blocks standing in it."""

    def __init__(self, width, height, blocks=()):
        self.width = width
        self.height = height
        self.blocks = list(blocks)   # (min_x, min_y, max_x, max_y)

    def cast(self, x, y, bearing_deg, max_range):
        """Nearest hit along a bearing, walls and blocks alike."""
        angle = math.radians(bearing_deg)
        dx, dy = math.cos(angle), math.sin(angle)
        best = max_range

        step = 0.01
        distance = step
        while distance < max_range:
            px, py = x + dx * distance, y + dy * distance
            if px <= 0 or px >= self.width or py <= 0 or py >= self.height:
                return min(best, distance)
            for (x0, y0, x1, y1) in self.blocks:
                if x0 <= px <= x1 and y0 <= py <= y1:
                    return min(best, distance)
            distance += step
        return best


def _scan(room, x, y, max_range=6.0, step_deg=4):
    readings = []
    for bearing in range(0, 360, step_deg):
        d = room.cast(x, y, bearing, max_range)
        angle = bearing if bearing <= 180 else bearing - 360
        readings.append(RangeReading(angle_deg=angle, distance_m=min(d, max_range)))
    return readings


def _map_room(room, spacing=0.3, resolution=0.05):
    """Sweep the room on a grid of viewpoints, skipping inside furniture."""
    grid = OccupancyGrid(resolution_m=resolution, initial_size_m=4.0, max_range_m=6.0)

    y = spacing
    while y < room.height:
        x = spacing
        while x < room.width:
            inside_block = any(
                x0 - 0.15 <= x <= x1 + 0.15 and y0 - 0.15 <= y <= y1 + 0.15
                for (x0, y0, x1, y1) in room.blocks
            )
            if not inside_block:
                pose = _pose(x, y)
                grid.integrate_scan(pose, _scan(room, x, y))
                grid.mark_robot_footprint(pose)
            x += spacing
        y += spacing
    return grid


# ── Detection ────────────────────────────────────────────────────────────────


def test_an_empty_room_reports_no_obstacles():
    """A false obstacle in an empty room would wrongly reduce usable area."""
    grid = _map_room(Room(5.0, 4.0))
    outline = RoomExtractor().extract(grid, _pose(2.5, 2.0), "TEST01", "2026-08-16T00:00:00Z")

    assert outline.obstacles == []
    assert outline.blocked_area_m2 == 0.0
    assert outline.usable_area_m2 == pytest.approx(outline.area_m2)


def test_a_table_in_the_middle_is_found():
    """The case wall-following misses entirely."""
    table = (2.0, 1.5, 3.0, 2.5)
    grid = _map_room(Room(5.0, 4.0, [table]))
    outline = RoomExtractor().extract(grid, _pose(0.6, 0.6), "TEST01", "2026-08-16T00:00:00Z")

    assert len(outline.obstacles) >= 1
    biggest = max(outline.obstacles, key=lambda o: o.area_m2)
    assert biggest.centre_x_m == pytest.approx(2.5, abs=0.5)
    assert biggest.centre_y_m == pytest.approx(2.0, abs=0.5)


def test_the_obstacle_area_is_roughly_right():
    """A 1.0 x 1.0 m table is about 1 m2. The measured footprint is a little
    larger because the sensor sees the surface, not the outline."""
    grid = _map_room(Room(5.0, 4.0, [(2.0, 1.5, 3.0, 2.5)]))
    outline = RoomExtractor().extract(grid, _pose(0.6, 0.6), "TEST01", "2026-08-16T00:00:00Z")

    assert 0.4 < outline.blocked_area_m2 < 2.5


def test_two_separate_obstacles_stay_separate():
    grid = _map_room(Room(6.0, 4.0, [(1.2, 1.2, 1.9, 1.9), (4.0, 2.2, 4.7, 2.9)]))
    outline = RoomExtractor().extract(grid, _pose(0.5, 0.5), "TEST01", "2026-08-16T00:00:00Z")

    assert len(outline.obstacles) >= 2


def test_walls_are_not_reported_as_obstacles():
    """The distinction the whole module rests on. A wall touches the outside;
    furniture is enclosed by the room's own floor."""
    grid = _map_room(Room(5.0, 4.0, [(2.0, 1.5, 3.0, 2.5)]))
    outline = RoomExtractor().extract(grid, _pose(0.6, 0.6), "TEST01", "2026-08-16T00:00:00Z")

    # Every reported obstacle must sit inside the room, not on its edge.
    for obstacle in outline.obstacles:
        assert 0.2 < obstacle.centre_x_m < 4.8
        assert 0.2 < obstacle.centre_y_m < 3.8


def test_specks_of_noise_are_not_furniture():
    """Two stray cells from one noisy reading are not a table, and reporting
    them as such would bury the real obstacles."""
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=6.0, max_range_m=4.0)
    pose = _pose(2.0, 2.0)
    grid.integrate_scan(pose, [RangeReading(angle_deg=0.0, distance_m=1.0)])

    extractor = RoomExtractor()
    interior = extractor.flood_fill_interior(grid, *grid.world_to_cell(2.0, 2.0))
    obstacles = extractor.find_obstacles(grid, interior, min_area_m2=0.02)

    assert obstacles == []


# ── The numbers a customer sees ──────────────────────────────────────────────


def test_usable_area_excludes_what_furniture_stands_on():
    """Two different customers need two different numbers: a cleaning robot
    cannot reach under the table, but the floor there still needs flooring."""
    grid = _map_room(Room(5.0, 4.0, [(2.0, 1.5, 3.0, 2.5)]))
    outline = RoomExtractor().extract(grid, _pose(0.6, 0.6), "TEST01", "2026-08-16T00:00:00Z")

    assert outline.usable_area_m2 < outline.area_m2
    assert outline.usable_area_m2 == pytest.approx(
        outline.area_m2 - outline.blocked_area_m2
    )


def test_usable_area_never_goes_negative():
    """Defensive: a pathological map must not report negative floor."""
    grid = _map_room(Room(5.0, 4.0))
    outline = RoomExtractor().extract(grid, _pose(2.5, 2.0), "TEST01", "2026-08-16T00:00:00Z")
    outline.blocked_area_m2 = outline.area_m2 * 10
    assert outline.usable_area_m2 == 0.0


def test_the_room_area_still_includes_the_floor_under_furniture():
    """A table does not shrink the room. Total area must be unaffected by
    what is standing on it."""
    empty = RoomExtractor().extract(
        _map_room(Room(5.0, 4.0)), _pose(2.5, 2.0), "T", "2026-08-16T00:00:00Z"
    )
    furnished = RoomExtractor().extract(
        _map_room(Room(5.0, 4.0, [(2.0, 1.5, 3.0, 2.5)])),
        _pose(0.6, 0.6), "T", "2026-08-16T00:00:00Z",
    )

    assert furnished.area_m2 == pytest.approx(empty.area_m2, rel=0.15)


# ── Reporting shape ──────────────────────────────────────────────────────────


def test_each_obstacle_carries_a_bounding_box():
    """The 3D view draws boxes, so it needs extents rather than just a centre."""
    grid = _map_room(Room(5.0, 4.0, [(2.0, 1.5, 3.0, 2.5)]))
    outline = RoomExtractor().extract(grid, _pose(0.6, 0.6), "TEST01", "2026-08-16T00:00:00Z")

    for obstacle in outline.obstacles:
        assert obstacle.max_x_m > obstacle.min_x_m
        assert obstacle.max_y_m > obstacle.min_y_m
        assert obstacle.min_x_m <= obstacle.centre_x_m <= obstacle.max_x_m


def test_obstacles_are_reported_largest_first():
    """The big one is the one worth looking at."""
    grid = _map_room(Room(6.0, 4.0, [(1.2, 1.2, 2.4, 2.4), (4.2, 2.4, 4.6, 2.8)]))
    outline = RoomExtractor().extract(grid, _pose(0.5, 0.5), "TEST01", "2026-08-16T00:00:00Z")

    if len(outline.obstacles) >= 2:
        areas = [o.area_m2 for o in outline.obstacles]
        assert areas == sorted(areas, reverse=True)


def test_the_outline_serialises_with_obstacles():
    import json

    grid = _map_room(Room(5.0, 4.0, [(2.0, 1.5, 3.0, 2.5)]))
    outline = RoomExtractor().extract(grid, _pose(0.6, 0.6), "TEST01", "2026-08-16T00:00:00Z")

    restored = json.loads(outline.model_dump_json())
    assert "obstacles" in restored
    assert "blocked_area_m2" in restored
