"""A free seed cell can still lead nowhere.

The room is found by flood-filling free space from where the robot is standing.
When the robot's own cell is not free — it stopped against something, and a
contact is stamped 10 cm ahead of a drifting pose estimate — the extractor
looks for the nearest free cell and starts there instead.

"Nearest" is not the same as "leads somewhere". On a map built by contact
alone, single free cells get stranded between contact patches, and the nearest
one can be one of them. Filling from it returns one cell and the whole room
comes back as 0.00 m2 — which does not look like a seeding problem when you
are staring at it, because the seed IS free and the free space IS well
connected.

Measured on a furnished room mapped by contact alone: 5892 free cells, largest
connected region 5874 of them, fill returned ONE. 0.00 m2 of a 27 m2 room,
after the robot had driven 142 m and recorded 107 contacts.
"""

from __future__ import annotations

import numpy as np
from mapping.occupancy_grid import OccupancyGrid
from mapping.room_extraction import RoomExtractor
from robotmap_common.models import PoseEstimate


def _pose(x, y):
    return PoseEstimate(
        robot_id="TEST01", timestamp="2026-08-17T00:00:00Z", sequence=0,
        x_m=x, y_m=y, heading_deg=0.0, std_x_m=0.01, std_y_m=0.01,
    )


def _grid_with_a_room_and_a_speck():
    """A large connected floor, plus one isolated free cell near the robot."""
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=8.0, max_range_m=4.0)
    extractor = RoomExtractor()

    # A solid block of floor, well away from the robot.
    y = 1.0
    while y < 3.0:
        x = 1.0
        while x < 4.0:
            grid.mark_robot_footprint(_pose(x, y), radius_m=0.11)
            x += 0.05
        y += 0.05

    # The robot is standing somewhere else entirely, on a single free cell
    # surrounded by occupied ones. Well clear of the floor above, and inside
    # the grid — which starts at 8 m square and does not grow on its own.
    col, row = grid.world_to_cell(-2.5, -2.5)
    assert grid.in_bounds(col + 1, row + 1)
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            grid.grid[row + d_row, col + d_col] = 2.0      # occupied
    grid.grid[row, col] = -3.0                             # the speck

    return grid, extractor, (col, row)


def test_a_stranded_speck_does_not_become_the_whole_room():
    """The failure: one free cell, filled, reported as the room."""
    grid, extractor, (col, row) = _grid_with_a_room_and_a_speck()

    mask = extractor.flood_fill_interior(grid, col, row)

    assert mask.sum() > 100, (
        f"filled {int(mask.sum())} cells from a stranded seed — the room was lost"
    )


def test_it_falls_back_to_the_largest_connected_floor():
    """Not to any other region — to the biggest, because the room is the
    biggest piece of connected floor the robot has been on."""
    grid, extractor, (col, row) = _grid_with_a_room_and_a_speck()

    mask = extractor.flood_fill_interior(grid, col, row)
    free = grid.grid < extractor.free_threshold
    largest = extractor._largest_free_region(free, grid)

    assert int(mask.sum()) == int(largest.sum())


def test_a_good_seed_is_left_alone():
    """The fallback must not fire when seeding worked. A robot standing in its
    own room should fill that room, not have it swapped for another."""
    grid, extractor, _ = _grid_with_a_room_and_a_speck()
    col, row = grid.world_to_cell(2.5, 2.0)      # in the middle of the floor

    mask = extractor.flood_fill_interior(grid, col, row)
    free = grid.grid < extractor.free_threshold

    assert mask[row, col], "the seed's own cell is not in its own fill"
    assert int(mask.sum()) > 0.5 * int(free.sum())


def test_an_empty_map_stays_empty():
    """No floor observed means no room, not a crash."""
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=4.0, max_range_m=4.0)
    extractor = RoomExtractor()
    col, row = grid.world_to_cell(1.0, 1.0)

    mask = extractor.flood_fill_interior(grid, col, row)
    assert int(mask.sum()) <= 1


def test_the_fallback_is_not_reached_on_a_normal_map():
    """It walks every free cell, so it must stay on the rare path."""
    grid, extractor, _ = _grid_with_a_room_and_a_speck()
    col, row = grid.world_to_cell(2.5, 2.0)

    calls = []
    original = extractor._largest_free_region
    extractor._largest_free_region = lambda *a, **k: (
        calls.append(1) or original(*a, **k)
    )
    extractor.flood_fill_interior(grid, col, row)

    assert not calls, "the expensive fallback ran on a perfectly good seed"


def test_the_region_returned_is_actually_connected():
    """A mask stitched from several patches would report a room the robot
    cannot drive across."""
    grid, extractor, (col, row) = _grid_with_a_room_and_a_speck()
    mask = extractor.flood_fill_interior(grid, col, row)

    # Re-fill from any cell in the result; it must recover the whole of it.
    first = np.argwhere(mask)[0]
    again = extractor._fill_from(mask, grid, int(first[1]), int(first[0]))
    assert int(again.sum()) == int(mask.sum())
