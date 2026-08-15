"""End-to-end mapping tests against rooms of known size.

The point of these tests is accountability: a mapping system that produces a
pretty picture but the wrong area is worse than useless for a research
project, so every test here checks a *number* against ground truth.
"""

from __future__ import annotations

import math

import pytest
from mapping.occupancy_grid import OccupancyGrid
from mapping.room_extraction import RoomExtractor
from robotmap_common.models import PoseEstimate, RangeReading

# ── Synthetic environment ─────────────────────────────────────────────────────


class VirtualRoom:
    """A rectangular room used as ground truth for ray casting."""

    def __init__(self, width_m: float, height_m: float) -> None:
        self.width_m = width_m
        self.height_m = height_m

    @property
    def area_m2(self) -> float:
        return self.width_m * self.height_m

    def cast(self, x: float, y: float, bearing_deg: float, max_range: float) -> float:
        """Distance from (x, y) along a bearing to the nearest wall."""
        angle = math.radians(bearing_deg)
        dx, dy = math.cos(angle), math.sin(angle)

        best = max_range
        # Walls at x=0, x=width, y=0, y=height.
        for wall_pos, delta, origin in (
            (0.0, dx, x),
            (self.width_m, dx, x),
            (0.0, dy, y),
            (self.height_m, dy, y),
        ):
            if abs(delta) < 1e-9:
                continue
            t = (wall_pos - origin) / delta
            if 0 < t < best:
                hit_x, hit_y = x + dx * t, y + dy * t
                # Only count the hit if it lands within the wall's extent.
                on_wall = (
                    -1e-6 <= hit_x <= self.width_m + 1e-6
                    and -1e-6 <= hit_y <= self.height_m + 1e-6
                )
                if on_wall:
                    best = t
        return best


def _pose(x: float, y: float, heading: float = 0.0) -> PoseEstimate:
    return PoseEstimate(
        robot_id="TEST01",
        timestamp="2026-08-15T00:00:00Z",
        sequence=0,
        x_m=x,
        y_m=y,
        heading_deg=heading % 360,
        std_x_m=0.01,
        std_y_m=0.01,
    )


def _scan_360(room: VirtualRoom, x: float, y: float, max_range: float, step_deg: int = 6):
    """A full rotating scan from one spot."""
    readings = []
    for bearing in range(0, 360, step_deg):
        dist = room.cast(x, y, bearing, max_range)
        readings.append(
            RangeReading(angle_deg=_wrap180(bearing), distance_m=min(dist, max_range))
        )
    return readings


def _wrap180(deg: float) -> float:
    d = deg % 360.0
    return d - 360.0 if d > 180.0 else d


def _map_room(room: VirtualRoom, resolution=0.05, max_range=6.0, spacing=0.5):
    """Drive a lawnmower pattern through the room, scanning as we go."""
    grid = OccupancyGrid(
        resolution_m=resolution, initial_size_m=4.0, max_range_m=max_range
    )

    y = spacing
    while y < room.height_m:
        x = spacing
        while x < room.width_m:
            pose = _pose(x, y)
            grid.integrate_scan(pose, _scan_360(room, x, y, max_range))
            grid.mark_robot_footprint(pose)
            x += spacing
        y += spacing
    return grid


# ── Grid mechanics ────────────────────────────────────────────────────────────


def test_world_cell_roundtrip():
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=10.0)
    col, row = grid.world_to_cell(1.23, -2.34)
    x, y = grid.cell_to_world(col, row)
    # Recovered point must be within half a cell of the original.
    assert abs(x - 1.23) <= 0.025 + 1e-9
    assert abs(y - -2.34) <= 0.025 + 1e-9


def test_grid_grows_to_contain_far_point():
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=4.0)
    before = grid.width_cells
    grid.ensure_contains(20.0, 20.0)
    assert grid.width_cells > before
    col, row = grid.world_to_cell(20.0, 20.0)
    assert grid.in_bounds(col, row)


def test_growth_preserves_existing_data():
    """Padding must not move data relative to the world frame."""
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=4.0)
    pose = _pose(0.0, 0.0)
    grid.integrate_scan(pose, [RangeReading(angle_deg=0.0, distance_m=1.0)])

    marked_col, marked_row = grid.world_to_cell(1.0, 0.0)
    value_before = grid.grid[marked_row, marked_col]
    assert value_before > 0  # the wall we just recorded

    grid.ensure_contains(15.0, 15.0)

    new_col, new_row = grid.world_to_cell(1.0, 0.0)
    assert grid.grid[new_row, new_col] == pytest.approx(value_before)


def test_unknown_cells_report_minus_one():
    grid = OccupancyGrid(resolution_m=0.1, initial_size_m=2.0)
    assert (grid.occupancy_percent() == -1).all()


def test_max_range_reading_marks_free_not_wall():
    """A max-range return means 'nothing there' and must not create a wall."""
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=10.0, max_range_m=4.0)
    pose = _pose(0.0, 0.0)
    grid.integrate_scan(pose, [RangeReading(angle_deg=0.0, distance_m=4.0)])

    col, row = grid.world_to_cell(4.0, 0.0)
    assert grid.grid[row, col] <= 0.0


def test_repeated_observation_converges_to_occupied():
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=10.0, max_range_m=4.0)
    pose = _pose(0.0, 0.0)
    for _ in range(10):
        grid.integrate_scan(pose, [RangeReading(angle_deg=0.0, distance_m=1.0)])

    col, row = grid.world_to_cell(1.0, 0.0)
    prob = grid.probability_map()[row, col]
    assert prob > 0.9


def test_log_odds_are_clamped():
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=10.0, max_range_m=4.0)
    pose = _pose(0.0, 0.0)
    for _ in range(500):
        grid.integrate_scan(pose, [RangeReading(angle_deg=0.0, distance_m=1.0)])
    assert grid.grid.max() <= OccupancyGrid.LOG_ODDS_MAX + 1e-6


def test_free_evidence_can_revise_a_wall():
    """A passing obstacle must not leave a permanent wall behind."""
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=10.0, max_range_m=4.0)
    pose = _pose(0.0, 0.0)
    col, row = grid.world_to_cell(1.0, 0.0)

    for _ in range(5):
        grid.integrate_scan(pose, [RangeReading(angle_deg=0.0, distance_m=1.0)])
    assert grid.grid[row, col] > 0

    # The obstacle leaves; the sensor now sees straight past that cell.
    for _ in range(30):
        grid.integrate_scan(pose, [RangeReading(angle_deg=0.0, distance_m=3.0)])
    assert grid.grid[row, col] < 0


def test_scan_from_untrusted_pose_is_dropped():
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=10.0)
    bad_pose = PoseEstimate(
        robot_id="TEST01",
        timestamp="2026-08-15T00:00:00Z",
        sequence=0,
        x_m=0.0,
        y_m=0.0,
        heading_deg=0.0,
        std_x_m=5.0,  # hopelessly drifted
        std_y_m=5.0,
    )
    used = grid.integrate_scan(bad_pose, [RangeReading(angle_deg=0.0, distance_m=1.0)])
    assert used == 0
    assert grid.explored_cells() == 0


# ── Room extraction accuracy ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "width,height",
    [(5.0, 4.0), (3.0, 3.0), (6.0, 2.5)],
)
def test_measured_area_matches_ground_truth(width, height):
    """The headline claim: the reported area must match the real room."""
    room = VirtualRoom(width, height)
    grid = _map_room(room)

    extractor = RoomExtractor()
    outline = extractor.extract(grid, _pose(width / 2, height / 2), "TEST01", "2026-08-15T00:00:00Z")

    assert outline.polygon, "no polygon produced"
    # Within 10 % of truth — the grid resolution and wall thickness set the floor.
    assert outline.area_m2 == pytest.approx(room.area_m2, rel=0.10)


def test_measured_dimensions_match_ground_truth():
    room = VirtualRoom(5.0, 4.0)
    grid = _map_room(room)
    outline = RoomExtractor().extract(
        grid, _pose(2.5, 2.0), "TEST01", "2026-08-15T00:00:00Z"
    )

    assert outline.bounding_width_m == pytest.approx(5.0, abs=0.3)
    assert outline.bounding_height_m == pytest.approx(4.0, abs=0.3)


def test_perimeter_matches_ground_truth():
    room = VirtualRoom(5.0, 4.0)
    grid = _map_room(room)
    outline = RoomExtractor().extract(
        grid, _pose(2.5, 2.0), "TEST01", "2026-08-15T00:00:00Z"
    )
    assert outline.perimeter_m == pytest.approx(18.0, rel=0.12)


def test_rectangular_room_simplifies_to_few_corners():
    """A rectangle should come out as roughly four corners, not hundreds."""
    room = VirtualRoom(5.0, 4.0)
    grid = _map_room(room)
    outline = RoomExtractor().extract(
        grid, _pose(2.5, 2.0), "TEST01", "2026-08-15T00:00:00Z"
    )
    assert 4 <= len(outline.polygon) <= 12


def test_fully_mapped_room_is_reported_closed():
    room = VirtualRoom(5.0, 4.0)
    grid = _map_room(room)
    outline = RoomExtractor().extract(
        grid, _pose(2.5, 2.0), "TEST01", "2026-08-15T00:00:00Z"
    )
    assert outline.is_closed is True


def test_empty_grid_yields_empty_outline():
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=4.0)
    outline = RoomExtractor().extract(
        grid, _pose(0.0, 0.0), "TEST01", "2026-08-15T00:00:00Z"
    )
    assert outline.polygon == []
    assert outline.area_m2 == 0.0
    assert outline.is_closed is False


def test_flood_fill_does_not_leak_through_walls():
    """Free space beyond a wall must not be counted as part of the room."""
    # Two 2 m rooms separated by a wall at x = 2.0.
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=12.0, max_range_m=6.0)

    inner = VirtualRoom(2.0, 2.0)
    for y in (0.5, 1.0, 1.5):
        for x in (0.5, 1.0, 1.5):
            pose = _pose(x, y)
            grid.integrate_scan(pose, _scan_360(inner, x, y, 6.0))
            grid.mark_robot_footprint(pose)

    extractor = RoomExtractor()
    mask = extractor.flood_fill_interior(grid, *grid.world_to_cell(1.0, 1.0))

    # Nothing beyond the far wall should be included.
    outside_col, outside_row = grid.world_to_cell(4.0, 1.0)
    assert not mask[outside_row, outside_col]


def test_coverage_is_reported():
    room = VirtualRoom(5.0, 4.0)
    grid = _map_room(room)
    outline = RoomExtractor().extract(
        grid, _pose(2.5, 2.0), "TEST01", "2026-08-15T00:00:00Z"
    )
    assert 0.0 < outline.coverage_pct <= 100.0
