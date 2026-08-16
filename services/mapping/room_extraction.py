"""Turn an occupancy grid into a room outline with a measured floor area.

The pipeline
------------
1. **Flood fill** the *confidently* free space reachable from the robot, so
   furniture voids and anything beyond a wall are excluded. This is what
   makes the result "the room the robot is in" rather than "every free cell
   ever seen".
2. **Open** the region morphologically to erase thin tendrils (see below).
3. **Trace the boundary** with Moore-neighbour tracing, producing an ordered
   ring of cells.
4. **Simplify** the ring with Douglas-Peucker, collapsing the cell-sized
   staircase into the handful of corners that describe the room.
5. **Optionally square up** the corners, because real rooms are overwhelmingly
   rectilinear and the eye immediately reads a wobbly outline as wrong.
6. **Measure** area and perimeter from the final polygon.

Why steps 1 and 2 are strict
----------------------------
Ultrasonic pulses that strike a wall at a shallow angle reflect away instead
of returning, and the sensor reports maximum range. Before a wall has been
observed enough times to be treated as opaque, such a reading paints a thin
line of "free" cells straight through it. Those lines are the failure mode
that matters: a single one lets the flood fill escape the room and the
reported area balloons.

Two independent defences handle it, because either alone leaves a gap:

* **Evidence threshold.** A cell counts as floor only after several
  independent free observations. Genuine floor is seen hundreds of times and
  saturates; a leaked ray is seen once or twice.
* **Morphological opening.** Whatever still leaks is one or two cells wide,
  while the room is hundreds. Eroding then dilating deletes the former and
  restores the latter almost exactly.
"""

from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np
from robotmap_common.geometry import (
    douglas_peucker,
    min_area_rect,
    polygon_area_m2,
    polygon_perimeter_m,
)
from robotmap_common.models import (
    ObstacleFootprint,
    Point2D,
    PoseEstimate,
    RoomOutline,
)

from .occupancy_grid import OccupancyGrid

logger = logging.getLogger(__name__)

# Moore neighbourhood, clockwise from east. Order matters: boundary tracing
# relies on scanning neighbours in consistent rotational order.
_NEIGHBOURS_CW = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Even-odd ray cast. True when the point is inside the ring."""
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if crossing_x > x:
                inside = not inside
    return inside


def _shrink_polygon(
    polygon: list[tuple[float, float]], margin_m: float
) -> list[tuple[float, float]]:
    """Pull a polygon in towards its centroid by roughly `margin_m`.

    Used to keep a contact against a wall from registering as furniture. Exact
    polygon offsetting is not worth it here: the margin is a cell or two, the
    rooms are convex enough, and being slightly wrong costs a red patch one
    cell from the wall rather than a wrong measurement.
    """
    count = len(polygon)
    cx = sum(p[0] for p in polygon) / count
    cy = sum(p[1] for p in polygon) / count

    shrunk = []
    for x, y in polygon:
        dx, dy = cx - x, cy - y
        length = math.hypot(dx, dy)
        if length < 1e-9:
            shrunk.append((x, y))
            continue
        step = min(margin_m, length * 0.9)
        shrunk.append((x + dx / length * step, y + dy / length * step))
    return shrunk


class RoomExtractor:
    """Extracts a room polygon from an occupancy grid."""

    def __init__(
        self,
        free_threshold: float = -1.5,
        occupied_threshold: float = 0.4,
        simplify_epsilon_m: float = 0.08,
        squareness_tolerance_deg: float = 12.0,
        opening_radius_m: float = 0.15,
        boundary_dilation_cells: int = 1,
    ) -> None:
        # -1.5 log-odds is roughly two or three independent "free"
        # observations. It is deliberately not stricter: raising it also
        # erodes genuine floor in the middle of the room, which a
        # wall-following robot observes least, and the effect worsens as
        # resolution gets finer and each cell collects fewer rays. Leak
        # removal is the opening's job, not this threshold's.
        self.free_threshold = free_threshold
        self.occupied_threshold = occupied_threshold
        self.simplify_epsilon_m = simplify_epsilon_m
        self.squareness_tolerance_deg = squareness_tolerance_deg
        # Structuring-element radius for the opening. Necks narrower than
        # twice this are severed; a real doorway (~0.8 m) survives, which is
        # correct — a room with an open door is not a closed room.
        self.opening_radius_m = opening_radius_m
        # The traced ring follows the centres of the outermost *free* cells,
        # but the floor actually continues to the wall face one cell beyond.
        # Without this the reported room is systematically one cell small on
        # every side — about 6 % of a 3 m room.
        #
        # One cell is right for a range-sensor map, where a ray reaches the
        # wall itself. It is NOT right for a map built by driving: there the
        # outermost free cell is where the robot's CENTRE reached, and the
        # floor continues for a further chassis radius that the robot could
        # never stand on. Measured on a contact-only trace of a 6.0 x 4.5 room,
        # that inset alone accounted for the whole shortfall — 23.02 m2 against
        # 27.0, with a bounding box of 5.78 x 4.38 which is exactly the room
        # less one robot diameter. `robot_radius_m` corrects it; see
        # `for_contact_mapping`.
        self.boundary_dilation_cells = boundary_dilation_cells

    @classmethod
    def for_contact_mapping(
        cls, resolution_m: float = 0.05, robot_radius_m: float = 0.11, **kwargs
    ) -> RoomExtractor:
        """An extractor for a map built by driving rather than by ranging.

        Two differences, both following from how the evidence was gathered.

        The floor continues a chassis radius beyond the outermost cell the
        robot occupied, not the single cell a range reading justifies — the
        robot's centre cannot get closer to a wall than its own radius.

        And the boundary is simplified far more aggressively. A range scan
        supplies thousands of readings and can genuinely resolve an 8 cm
        feature; a contact trace supplies a few dozen touches. Keeping 8 cm
        detail there produced a 28-corner outline for a rectangular room —
        over-fitting the gaps between passes and reading 10 % low, because the
        polygon wound into every unswept pocket instead of around the room.
        """
        cells = max(1, int(round(robot_radius_m / resolution_m)))
        kwargs.setdefault("simplify_epsilon_m", 0.25)
        kwargs.setdefault("opening_radius_m", 0.10)
        return cls(boundary_dilation_cells=cells, **kwargs)

    # ── Step 1: reachable interior ────────────────────────────────────────

    def flood_fill_interior(
        self, grid: OccupancyGrid, seed_col: int, seed_row: int
    ) -> np.ndarray:
        """Return a mask of free cells connected to the seed.

        Four-connectivity, not eight: diagonal connectivity leaks the fill
        through the corner gap where two wall cells touch diagonally, which
        would merge the room with the corridor outside it.
        """
        free = grid.grid < self.free_threshold
        mask = np.zeros_like(free, dtype=bool)

        if not grid.in_bounds(seed_col, seed_row):
            logger.warning("Flood-fill seed outside grid bounds")
            return mask

        if not free[seed_row, seed_col]:
            # The seed cell is not free. Two quite different reasons:
            #
            # Unobserved — the robot's own cell may never have been seen. It is
            # standing there, so accept it.
            #
            # Occupied — the robot's REPORTED position can land inside an
            # obstacle it marked moments ago, because odometry drifts and a
            # bumper contact is stamped 10 cm ahead of an estimated pose.
            # Forcing that single cell free boxes the fill inside the patch: it
            # fills one cell, gives up, and the whole room comes back EMPTY.
            # Measured: a 20.45 m2 room became 0.00 m2 from one contact.
            #
            # So look for real floor nearby and start from there instead. The
            # robot drove in from somewhere, and that somewhere is adjacent.
            nearby = self._nearest_free(free, seed_col, seed_row, radius_cells=12)
            if nearby is not None:
                seed_col, seed_row = nearby
            else:
                free = free.copy()
                free[seed_row, seed_col] = True

        queue = deque([(seed_col, seed_row)])
        mask[seed_row, seed_col] = True

        while queue:
            col, row = queue.popleft()
            for d_col, d_row in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n_col, n_row = col + d_col, row + d_row
                if not grid.in_bounds(n_col, n_row):
                    continue
                if mask[n_row, n_col] or not free[n_row, n_col]:
                    continue
                mask[n_row, n_col] = True
                queue.append((n_col, n_row))

        return mask

    @staticmethod
    def _nearest_free(
        free: np.ndarray, col: int, row: int, radius_cells: int
    ) -> tuple[int, int] | None:
        """Closest free cell to (col, row), searched outward in rings."""
        height, width = free.shape
        for radius in range(1, radius_cells + 1):
            for d_row in range(-radius, radius + 1):
                for d_col in range(-radius, radius + 1):
                    # Only the rim of each square, so nearer cells win.
                    if max(abs(d_row), abs(d_col)) != radius:
                        continue
                    n_col, n_row = col + d_col, row + d_row
                    if 0 <= n_col < width and 0 <= n_row < height:
                        if free[n_row, n_col]:
                            return n_col, n_row
        return None

    # ── Step 2: morphological opening ─────────────────────────────────────

    @staticmethod
    def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
        """A cell survives only if its whole square neighbourhood is set."""
        if radius <= 0:
            return mask
        height, width = mask.shape
        padded = np.pad(mask, radius, mode="constant", constant_values=False)
        out = np.ones_like(mask, dtype=bool)
        for d_row in range(-radius, radius + 1):
            for d_col in range(-radius, radius + 1):
                out &= padded[
                    radius + d_row : radius + d_row + height,
                    radius + d_col : radius + d_col + width,
                ]
        return out

    @staticmethod
    def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
        """A cell is set if any cell in its square neighbourhood is set."""
        if radius <= 0:
            return mask
        height, width = mask.shape
        padded = np.pad(mask, radius, mode="constant", constant_values=False)
        out = np.zeros_like(mask, dtype=bool)
        for d_row in range(-radius, radius + 1):
            for d_col in range(-radius, radius + 1):
                out |= padded[
                    radius + d_row : radius + d_row + height,
                    radius + d_col : radius + d_col + width,
                ]
        return out

    def open_mask(self, mask: np.ndarray, resolution_m: float) -> np.ndarray:
        """Erode then dilate, deleting tendrils while preserving the room.

        Opening is the standard way to remove structures thinner than the
        structuring element. The erosion severs every leaked ray, and the
        dilation grows the surviving bulk back to very nearly its original
        extent — the room loses only the sub-centimetre roughness of its own
        edge, which the polygon simplification would have discarded anyway.
        """
        # At least three cells: a one- or two-cell element is too small to
        # sever the leaked rays reliably, so a coarse grid would otherwise
        # silently lose the protection a fine grid gets.
        radius = max(3, int(round(self.opening_radius_m / resolution_m)))
        return self._dilate(self._erode(mask, radius), radius)

    @staticmethod
    def largest_component(mask: np.ndarray, seed: tuple[int, int] | None = None) -> np.ndarray:
        """Keep one connected region: the seed's, or the biggest one.

        Opening can sever the room from stray blobs, so connectivity has to be
        re-established afterwards rather than trusted from before it.
        """
        if not mask.any():
            return mask

        visited = np.zeros_like(mask, dtype=bool)
        best_mask: np.ndarray | None = None
        best_size = 0

        seed_col, seed_row = seed if seed else (None, None)

        for start_row, start_col in np.argwhere(mask):
            if visited[start_row, start_col]:
                continue

            component = np.zeros_like(mask, dtype=bool)
            queue = deque([(int(start_col), int(start_row))])
            visited[start_row, start_col] = True
            component[start_row, start_col] = True
            size = 1

            while queue:
                col, row = queue.popleft()
                for d_col, d_row in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n_col, n_row = col + d_col, row + d_row
                    if not (
                        0 <= n_row < mask.shape[0] and 0 <= n_col < mask.shape[1]
                    ):
                        continue
                    if visited[n_row, n_col] or not mask[n_row, n_col]:
                        continue
                    visited[n_row, n_col] = True
                    component[n_row, n_col] = True
                    queue.append((n_col, n_row))
                    size += 1

            # The component holding the robot always wins, if it survived.
            if seed_row is not None and component[seed_row, seed_col]:
                return component
            if size > best_size:
                best_size, best_mask = size, component

        return best_mask if best_mask is not None else mask

    @staticmethod
    def fill_holes(mask: np.ndarray) -> np.ndarray:
        """Fill enclosed gaps in the region.

        A robot following the walls observes the middle of the room least, so
        the centre is riddled with cells that are genuinely floor but were
        never confidently seen. Anything fully surrounded by the room *is*
        part of the room, whether or not a sensor ray happened to cross it.

        Implemented by flooding the background inward from the grid border:
        background that the flood cannot reach is, by definition, enclosed.
        A room with a real gap in its boundary — an open doorway — lets the
        flood in and is correctly left unfilled.
        """
        height, width = mask.shape
        outside = np.zeros_like(mask, dtype=bool)
        queue: deque[tuple[int, int]] = deque()

        for col in range(width):
            for row in (0, height - 1):
                if not mask[row, col] and not outside[row, col]:
                    outside[row, col] = True
                    queue.append((col, row))
        for row in range(height):
            for col in (0, width - 1):
                if not mask[row, col] and not outside[row, col]:
                    outside[row, col] = True
                    queue.append((col, row))

        while queue:
            col, row = queue.popleft()
            for d_col, d_row in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n_col, n_row = col + d_col, row + d_row
                if not (0 <= n_row < height and 0 <= n_col < width):
                    continue
                if outside[n_row, n_col] or mask[n_row, n_col]:
                    continue
                outside[n_row, n_col] = True
                queue.append((n_col, n_row))

        return mask | ~(mask | outside)

    # ── Step 3: boundary tracing ──────────────────────────────────────────

    def trace_boundary(self, mask: np.ndarray) -> list[tuple[int, int]]:
        """Return the outer boundary of a filled region as ordered cells.

        Moore-neighbour tracing with Jacob's stopping criterion: the walk ends
        when the start cell is re-entered *from the same direction*, not merely
        revisited. Stopping on revisit alone truncates rooms that pinch to a
        one-cell-wide doorway, which is exactly where the walk passes twice.
        """
        filled = np.argwhere(mask)
        if filled.size == 0:
            return []

        # Start at the lowest row, then lowest column — guaranteed on the hull.
        start_row, start_col = filled[0]
        start = (int(start_col), int(start_row))

        boundary: list[tuple[int, int]] = [start]
        current = start
        # Entry direction into the start cell: came from the west.
        backtrack_index = 4

        max_steps = int(mask.sum() * 8) + 64  # generous, but always terminates

        for _ in range(max_steps):
            found = False
            # Scan clockwise starting just past where we came from.
            for offset in range(1, 9):
                idx = (backtrack_index + offset) % 8
                d_col, d_row = _NEIGHBOURS_CW[idx]
                n_col, n_row = current[0] + d_col, current[1] + d_row

                if not (0 <= n_row < mask.shape[0] and 0 <= n_col < mask.shape[1]):
                    continue
                if not mask[n_row, n_col]:
                    continue

                # The new backtrack direction points back at the current cell.
                backtrack_index = (idx + 4) % 8
                current = (n_col, n_row)
                found = True
                break

            if not found:
                break  # isolated single cell

            if current == start and len(boundary) > 2:
                break

            boundary.append(current)

        return boundary

    # ── Step 3-4: simplify and square up ──────────────────────────────────

    def _to_world(
        self, grid: OccupancyGrid, cells: list[tuple[int, int]]
    ) -> list[tuple[float, float]]:
        return [grid.cell_to_world(col, row) for col, row in cells]

    def _square_up(
        self, points: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Snap near-axis-aligned edges onto the axes.

        Only edges already within tolerance are moved, so a genuinely angled
        wall survives intact. Purely cosmetic, but it is the difference
        between output that reads as a floor plan and output that reads as
        a scribble.
        """
        if len(points) < 3:
            return points

        result = list(points)
        tolerance = math.radians(self.squareness_tolerance_deg)

        for i in range(len(result)):
            j = (i + 1) % len(result)
            x1, y1 = result[i]
            x2, y2 = result[j]

            angle = math.atan2(y2 - y1, x2 - x1)
            # Distance to the nearest multiple of 90 degrees.
            snapped = round(angle / (math.pi / 2)) * (math.pi / 2)

            if abs(angle - snapped) > tolerance:
                continue

            if abs(math.cos(snapped)) < 1e-9:
                # Vertical edge: share one x.
                mean_x = (x1 + x2) / 2
                result[i] = (mean_x, y1)
                result[j] = (mean_x, y2)
            else:
                # Horizontal edge: share one y.
                mean_y = (y1 + y2) / 2
                result[i] = (x1, mean_y)
                result[j] = (x2, mean_y)

        return result

    # ── Full pipeline ─────────────────────────────────────────────────────

    def extract(
        self,
        grid: OccupancyGrid,
        pose: PoseEstimate,
        robot_id: str,
        timestamp: str,
        square_up: bool = True,
    ) -> RoomOutline:
        """Run the full extraction and return the measured room."""
        seed_col, seed_row = grid.world_to_cell(pose.x_m, pose.y_m)
        interior = self.flood_fill_interior(grid, seed_col, seed_row)

        if interior.sum() < 4:
            return self._empty_outline(robot_id, timestamp)

        # Erase leaked tendrils, then re-establish connectivity: the opening
        # may have detached the room from blobs it was only hair-connected to.
        interior = self.open_mask(interior, grid.resolution_m)
        seed = (seed_col, seed_row) if grid.in_bounds(seed_col, seed_row) else None
        if seed is not None and not interior[seed_row, seed_col]:
            # Erosion can consume the robot's own cell when it finishes close
            # to a wall; fall back to the largest surviving region.
            seed = None
        interior = self.largest_component(interior, seed)

        # Observed-free area, before enclosed gaps are filled in. This is the
        # honest measure of how much floor the robot actually saw, and is what
        # the coverage figure is reported against.
        observed_cells = int(interior.sum())

        # Keep the un-filled mask: obstacle detection needs the difference
        # between the room's filled footprint and its actual free floor, and
        # once the holes are filled that difference is gone.
        observed_free = interior.copy()

        interior = self.fill_holes(interior)

        # Grow out to the wall face so the outline sits on the wall rather
        # than one cell inside it.
        enclosed = self._dilate(interior, self.boundary_dilation_cells)

        interior_cells = int(enclosed.sum())
        if interior_cells < 4:
            return self._empty_outline(robot_id, timestamp)

        boundary_cells = self.trace_boundary(enclosed)
        if len(boundary_cells) < 4:
            return self._empty_outline(robot_id, timestamp)

        world_points = self._to_world(grid, boundary_cells)

        # Close the ring before simplifying so the final edge is considered.
        closed = world_points + [world_points[0]]
        simplified = douglas_peucker(closed, self.simplify_epsilon_m)
        if len(simplified) > 1 and simplified[0] == simplified[-1]:
            simplified = simplified[:-1]

        if square_up:
            simplified = self._square_up(simplified)

        if len(simplified) < 3:
            return self._empty_outline(robot_id, timestamp)

        area = polygon_area_m2(simplified)
        perimeter = polygon_perimeter_m(simplified)

        # Measured against the room's own axes, not the map's arbitrary ones.
        long_side_m, short_side_m, _ = min_area_rect(simplified)

        # Coverage: how much of the enclosed room the robot actually observed,
        # as opposed to inferred by filling holes. A low value means the area
        # figure rests more on inference than on measurement.
        cell_area = grid.resolution_m**2
        observed_area = observed_cells * cell_area
        coverage = min(100.0, (observed_area / area * 100.0) if area > 0 else 0.0)

        obstacles = self.find_obstacles(grid, observed_free, room_polygon=simplified)
        blocked = sum(o["area_m2"] for o in obstacles)

        return RoomOutline(
            robot_id=robot_id,
            timestamp=timestamp,
            polygon=[Point2D(x_m=x, y_m=y) for x, y in simplified],
            area_m2=area,
            obstacles=[ObstacleFootprint(**o) for o in obstacles],
            blocked_area_m2=blocked,
            perimeter_m=perimeter,
            bounding_width_m=long_side_m,
            bounding_height_m=short_side_m,
            coverage_pct=coverage,
            # Closure is judged on the observed free region, not the dilated
            # one: dilation deliberately pushes the boundary into the wall
            # cells, so the dilated region's rim sits in unknown space beyond
            # the wall and would always look unexplored.
            is_closed=self._is_enclosed(grid, interior),
        )

    def _is_enclosed(self, grid: OccupancyGrid, interior: np.ndarray) -> bool:
        """True when the interior does not touch the edge of the known grid.

        If the free region runs off the edge of what has been mapped, the
        robot has not yet found the whole boundary and the area figure is a
        lower bound, not a measurement.
        """
        if interior[0, :].any() or interior[-1, :].any():
            return False
        if interior[:, 0].any() or interior[:, -1].any():
            return False

        # Also require the ring around the interior to be mostly wall rather
        # than unexplored, otherwise an open doorway reads as enclosed.
        unknown = grid.grid == 0.0
        dilated = np.zeros_like(interior)
        for d_col, d_row in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            dilated |= np.roll(np.roll(interior, d_row, axis=0), d_col, axis=1)
        border = dilated & ~interior

        border_count = int(border.sum())
        if border_count == 0:
            return False
        unknown_border = int((border & unknown).sum())
        return (unknown_border / border_count) < 0.15

    # ── Obstacles inside the room ─────────────────────────────────────────

    @staticmethod
    def _contacts_inside(
        grid: OccupancyGrid,
        contacts: np.ndarray,
        room_polygon: list[tuple[float, float]] | None,
        fallback: np.ndarray,
    ) -> np.ndarray:
        """Contact cells that lie strictly inside the room outline.

        Without a polygon there is nothing to be inside of, so it falls back to
        the filled footprint — conservative, and the case where the room is not
        yet closed enough to have an outline at all.
        """
        if not room_polygon or len(room_polygon) < 3:
            return contacts & fallback

        inset = _shrink_polygon(room_polygon, grid.resolution_m * 1.5)

        keep = np.zeros_like(contacts, dtype=bool)
        for row, col in np.argwhere(contacts):
            x, y = grid.cell_to_world(int(col), int(row))
            if _point_in_polygon(x, y, inset):
                keep[row, col] = True
        return keep

    def find_obstacles(
        self,
        grid: OccupancyGrid,
        interior: np.ndarray,
        min_area_m2: float = 0.02,
        room_polygon: list[tuple[float, float]] | None = None,
    ) -> list[dict]:
        """Occupied islands sitting inside the room — furniture, not walls.

        A boundary wall and a table leg look identical to a range sensor: both
        are simply "occupied". What separates them is *where* they are. A wall
        encloses the room; furniture is surrounded by the room's own floor.

        So an occupied region is furniture when it is fully enclosed by the
        interior free space, and wall when it touches the outside. That
        distinction is what lets the usable floor area exclude the space a
        table stands on, which is the number a flooring installer actually
        needs.

        `min_area_m2` discards specks: a couple of stray cells from one noisy
        reading are not a piece of furniture, and reporting them as such would
        bury the real obstacles in noise.
        """
        occupied = grid.occupied_mask(self.occupied_threshold)
        if not occupied.any():
            return []

        # The room's filled footprint: its free space plus every gap enclosed
        # by it. Furniture sits in exactly those gaps, because the flood fill
        # flowed around it; the walls do not, because they are reachable from
        # outside the room.
        #
        # Dilating the interior instead — the obvious first attempt — reaches
        # into the wall cells and reports the entire wall ring as one giant
        # obstacle spanning the room. An empty room then appears to contain a
        # 0.9 m2 obstacle, and usable area is wrong everywhere.
        #
        # This also handles a cabinet pushed flat against a wall: the sliver
        # behind it is unreachable from outside, so it fills, and the cabinet
        # is still found.
        room_footprint = self.fill_holes(interior)

        # Blocked floor is the room's filled footprint minus its free floor.
        #
        # NOT simply the occupied cells: a range sensor only ever sees an
        # object's outer faces, never its middle, so a 1 m table appears as a
        # hollow 5 cm outline. Measuring that gives 0.19 m2 for a table that
        # actually stands on about 1 m2 — the perimeter, not the footprint.
        #
        # Subtracting free floor from the filled room recovers the solid
        # region, because the hole-fill already flowed around the object and
        # filled its unobserved middle.
        candidates = room_footprint & ~interior

        # Anything the robot has physically run into counts, enclosed or not.
        #
        # The enclosure test above needs free space observed all the way around
        # an object before it registers, which is right for something seen at a
        # distance — a gap in the sweep is not furniture. But a bumper contact
        # is not an inference. The robot touched something, so something is
        # there, and requiring it to drive a full circuit before drawing it
        # means a single touch-and-retreat reports an empty room. Measured on a
        # slim pillar the sonar kept missing: 2 contacts recorded, 0.00 m2
        # blocked reported.
        #
        # Tested against the room OUTLINE rather than the filled footprint. The
        # footprint is the observed free space plus its enclosed gaps, and a
        # robot that has driven one corridor past an object has not enclosed
        # it — so masking by the footprint discards exactly the contacts this
        # is meant to keep. The outline is the room, and inside the room is
        # where furniture is.
        #
        # A contact on the boundary itself is a wall, not furniture, and is
        # excluded by the same test.
        contacts = getattr(grid, "contact_mask", None)
        if contacts is not None and contacts.any():
            candidates = candidates | self._contacts_inside(
                grid, contacts, room_polygon, room_footprint
            )

        if not candidates.any():
            return []

        cell_area = grid.resolution_m**2
        obstacles: list[dict] = []
        visited = np.zeros_like(candidates, dtype=bool)

        for start_row, start_col in np.argwhere(candidates):
            if visited[start_row, start_col]:
                continue

            queue = deque([(int(start_col), int(start_row))])
            visited[start_row, start_col] = True
            cells: list[tuple[int, int]] = []
            touches_edge = False
            # Require evidence that something is actually there. A pocket the
            # robot merely never looked into is unexplored floor, not a table,
            # and calling it an obstacle would invent furniture from a gap in
            # the sweep.
            has_occupied_evidence = False

            while queue:
                col, row = queue.popleft()
                cells.append((col, row))
                if occupied[row, col]:
                    has_occupied_evidence = True

                if (
                    col <= 0 or row <= 0
                    or col >= candidates.shape[1] - 1
                    or row >= candidates.shape[0] - 1
                ):
                    touches_edge = True

                for d_col, d_row in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                     (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    n_col, n_row = col + d_col, row + d_row
                    if not grid.in_bounds(n_col, n_row):
                        continue
                    if visited[n_row, n_col] or not candidates[n_row, n_col]:
                        continue
                    visited[n_row, n_col] = True
                    queue.append((n_col, n_row))

            area = len(cells) * cell_area
            if area < min_area_m2 or touches_edge or not has_occupied_evidence:
                continue

            xs = [grid.cell_to_world(c, r)[0] for c, r in cells]
            ys = [grid.cell_to_world(c, r)[1] for c, r in cells]

            obstacles.append(
                {
                    "centre_x_m": sum(xs) / len(xs),
                    "centre_y_m": sum(ys) / len(ys),
                    "min_x_m": min(xs),
                    "min_y_m": min(ys),
                    "max_x_m": max(xs),
                    "max_y_m": max(ys),
                    "area_m2": round(area, 3),
                    "cells": len(cells),
                }
            )

        obstacles.sort(key=lambda o: o["area_m2"], reverse=True)
        return obstacles

    def _empty_outline(self, robot_id: str, timestamp: str) -> RoomOutline:
        return RoomOutline(
            robot_id=robot_id,
            timestamp=timestamp,
            polygon=[],
            area_m2=0.0,
            perimeter_m=0.0,
            bounding_width_m=0.0,
            bounding_height_m=0.0,
            coverage_pct=0.0,
            is_closed=False,
        )
