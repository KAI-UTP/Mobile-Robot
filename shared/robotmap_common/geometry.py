"""Geometry and kinematics helpers.

Everything here is pure and side-effect free so it can be unit tested without
a broker, a robot, or a running stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Mean Earth radius (WGS-84 authalic), metres.
EARTH_RADIUS_M = 6_371_008.8


# ── Robot physical configuration ──────────────────────────────────────────────


@dataclass(frozen=True)
class RobotGeometry:
    """Physical constants of the differential-drive base.

    `wheel_base_m` is the distance between the two *driven* wheel contact
    patches.  The caster is ignored — it carries no odometry information.

    Getting these two numbers right matters more than any filter tuning:
    a 2 % error in `wheel_diameter_m` is a 2 % error in every distance the
    robot ever reports.  Calibrate them, do not trust the datasheet.
    """

    wheel_diameter_m: float = 0.065
    wheel_base_m: float = 0.150

    # Encoder counts per full turn of the *wheel*, after any gearbox.
    #
    # This is the single most important hardware choice in the robot, and the
    # easiest to get wrong. The 20-slot slotted-disc encoders sold with hobby
    # chassis resolve about 10 mm of travel, and — worse — a half-count
    # difference between the two wheels reads as several degrees of rotation.
    # Measured on the simulator, dropping from 360 to 20 counts per
    # revolution moved end-of-circuit position error from 0.11 m to 0.69 m,
    # and it stayed at 0.56 m even with every other noise source switched
    # off. No amount of filtering recovers information the encoder never
    # captured.
    #
    # Use a quadrature encoder on the motor shaft, before the gearbox, where
    # the reduction multiplies its resolution: an 11 PPR magnetic encoder
    # behind a 34:1 gearbox yields 11 x 4 x 34 = 1496 counts per wheel turn.
    ticks_per_revolution: int = 360

    # Multiplicative correction applied to the left wheel only. Compensates
    # for the two wheels never being exactly the same diameter, which is what
    # makes an uncorrected robot curve when commanded to drive straight.
    left_wheel_trim: float = 1.0

    @property
    def wheel_circumference_m(self) -> float:
        return math.pi * self.wheel_diameter_m

    @property
    def metres_per_tick(self) -> float:
        return self.wheel_circumference_m / self.ticks_per_revolution

    def validate(self) -> None:
        if self.wheel_diameter_m <= 0:
            raise ValueError("wheel_diameter_m must be positive")
        if self.wheel_base_m <= 0:
            raise ValueError("wheel_base_m must be positive")
        if self.ticks_per_revolution <= 0:
            raise ValueError("ticks_per_revolution must be positive")


# ── Angle helpers ─────────────────────────────────────────────────────────────


def normalize_deg(angle: float) -> float:
    """Wrap an angle into [0, 360)."""
    return angle % 360.0


def angle_difference_deg(a: float, b: float) -> float:
    """Shortest signed difference a - b, in (-180, 180].

    Used everywhere a heading error is computed; naive subtraction breaks at
    the 359 degrees -> 0 degrees wrap and sends the filter the wrong way.
    """
    diff = (a - b + 180.0) % 360.0 - 180.0
    return diff + 360.0 if diff <= -180.0 else diff


# ── Differential-drive odometry ───────────────────────────────────────────────


@dataclass
class OdometryDelta:
    """Motion increment produced by one pair of encoder readings."""

    delta_x_m: float
    delta_y_m: float
    delta_heading_deg: float
    distance_m: float
    left_distance_m: float
    right_distance_m: float


def differential_drive_delta(
    left_ticks_delta: int,
    right_ticks_delta: int,
    heading_deg: float,
    geometry: RobotGeometry,
) -> OdometryDelta:
    """Integrate one encoder increment into a pose change.

    Uses exact arc integration rather than the straight-line approximation.
    For a robot turning while driving, the straight-line form places the
    robot on the chord instead of the arc, and that error accumulates into a
    map whose walls bow inwards on every curve.
    """
    m_per_tick = geometry.metres_per_tick
    d_left = left_ticks_delta * m_per_tick * geometry.left_wheel_trim
    d_right = right_ticks_delta * m_per_tick

    d_center = (d_left + d_right) / 2.0
    d_theta = (d_right - d_left) / geometry.wheel_base_m  # radians

    heading_rad = math.radians(heading_deg)

    if abs(d_theta) < 1e-9:
        # Straight line — the arc formula degenerates to 0/0 here.
        dx = d_center * math.cos(heading_rad)
        dy = d_center * math.sin(heading_rad)
    else:
        # Travel along the arc about the instantaneous centre of curvature.
        radius = d_center / d_theta
        dx = radius * (math.sin(heading_rad + d_theta) - math.sin(heading_rad))
        dy = -radius * (math.cos(heading_rad + d_theta) - math.cos(heading_rad))

    return OdometryDelta(
        delta_x_m=dx,
        delta_y_m=dy,
        delta_heading_deg=math.degrees(d_theta),
        distance_m=abs(d_center),
        left_distance_m=d_left,
        right_distance_m=d_right,
    )


def wrap_tick_delta(current: int, previous: int, bits: int = 32) -> int:
    """Return current - previous, correcting for counter overflow.

    Encoder counters on a microcontroller are fixed width and wrap around.
    Without this, one wrap injects a multi-kilometre jump into the map.
    """
    span = 1 << bits
    delta = (current - previous) % span
    if delta >= span // 2:
        delta -= span
    return delta


# ── Geodetic conversion ───────────────────────────────────────────────────────


def gps_to_local_xy(
    latitude: float,
    longitude: float,
    anchor_lat: float,
    anchor_lon: float,
) -> tuple[float, float]:
    """Project a GPS fix into local east/north metres about an anchor.

    Equirectangular projection.  Over the few hundred metres a room-mapping
    robot covers, its error against a proper geodesic is well under a
    millimetre — far below GNSS noise — and it is cheap and invertible.

    Returns (east_m, north_m), which map directly onto (x, y).
    """
    lat_rad = math.radians(latitude)
    anchor_lat_rad = math.radians(anchor_lat)
    mean_lat = (lat_rad + anchor_lat_rad) / 2.0

    d_lat = lat_rad - anchor_lat_rad
    d_lon = math.radians(longitude - anchor_lon)

    east = EARTH_RADIUS_M * d_lon * math.cos(mean_lat)
    north = EARTH_RADIUS_M * d_lat
    return east, north


def local_xy_to_gps(
    x_m: float,
    y_m: float,
    anchor_lat: float,
    anchor_lon: float,
) -> tuple[float, float]:
    """Inverse of `gps_to_local_xy`. Returns (latitude, longitude)."""
    lat = anchor_lat + math.degrees(y_m / EARTH_RADIUS_M)
    mean_lat = math.radians((lat + anchor_lat) / 2.0)

    # Guard the pole singularity, where a metre of easting spans every longitude.
    cos_lat = math.cos(mean_lat)
    if abs(cos_lat) < 1e-12:
        return lat, anchor_lon

    lon = anchor_lon + math.degrees(x_m / (EARTH_RADIUS_M * cos_lat))
    return lat, lon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two fixes, in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_lat = p2 - p1
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


# ── Ray projection and grid traversal ─────────────────────────────────────────


def project_ray(
    x_m: float,
    y_m: float,
    heading_deg: float,
    sensor_angle_deg: float,
    distance_m: float,
) -> tuple[float, float]:
    """Return the world coordinate a range reading hit.

    `sensor_angle_deg` is the sensor's mounting angle relative to the robot's
    forward axis; it is added to the robot heading to get the world bearing.
    """
    world_angle = math.radians(heading_deg + sensor_angle_deg)
    return (
        x_m + distance_m * math.cos(world_angle),
        y_m + distance_m * math.sin(world_angle),
    )


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Integer grid cells along a line, endpoints included.

    Used to mark every cell between the robot and a detected obstacle as
    free — the "the sensor saw through here" half of an occupancy update,
    which matters as much as marking the hit itself.
    """
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


# ── Polygon measurement ───────────────────────────────────────────────────────


def polygon_area_m2(points: list[tuple[float, float]]) -> float:
    """Absolute area of a simple polygon via the shoelace formula."""
    if len(points) < 3:
        return 0.0
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def polygon_perimeter_m(points: list[tuple[float, float]]) -> float:
    """Closed-loop perimeter length."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Convex hull by Andrew's monotone chain, counter-clockwise."""
    unique = sorted(set(points))
    if len(unique) < 3:
        return unique

    def build(sequence):
        chain: list[tuple[float, float]] = []
        for point in sequence:
            while len(chain) >= 2 and _cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        return chain

    lower = build(unique)
    upper = build(reversed(unique))
    return lower[:-1] + upper[:-1]


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def min_area_rect(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Smallest-area enclosing rectangle. Returns (long_side, short_side, angle_deg).

    Why not an axis-aligned bounding box
    ------------------------------------
    The map's axes point wherever the robot happened to be facing when it was
    switched on, which is arbitrary. Measuring a room against those axes
    reports the diagonal extent of a rotated rectangle rather than its sides:
    a 6.0 x 4.5 m room mapped 15 degrees askew measures 6.9 x 5.9 m
    axis-aligned, and both numbers are wrong.

    The minimum-area rectangle recovers the room's own axes instead, so the
    reported dimensions mean what a person with a tape measure would get.

    Uses the rotating-calipers result that the optimal rectangle always shares
    an edge with the convex hull, so it suffices to test each hull edge.
    """
    hull = convex_hull(points)
    if len(hull) < 3:
        if not points:
            return 0.0, 0.0, 0.0
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return max(xs) - min(xs), max(ys) - min(ys), 0.0

    best = (float("inf"), 0.0, 0.0, 0.0)  # area, width, height, angle

    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        edge_angle = math.atan2(y2 - y1, x2 - x1)

        cos_a, sin_a = math.cos(-edge_angle), math.sin(-edge_angle)
        rotated = [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in hull]

        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        area = width * height

        if area < best[0]:
            best = (area, width, height, math.degrees(edge_angle))

    _, width, height, angle = best
    long_side, short_side = max(width, height), min(width, height)
    return long_side, short_side, angle % 180.0


def douglas_peucker(
    points: list[tuple[float, float]], epsilon_m: float
) -> list[tuple[float, float]]:
    """Simplify a polyline, keeping points that carry the shape.

    A traced grid contour is a staircase of thousands of cell-sized steps.
    This collapses it to the handful of corners that actually describe the
    room, which is what makes the output look like a floor plan.
    """
    if len(points) < 3:
        return list(points)

    start, end = points[0], points[-1]
    max_dist = 0.0
    index = 0

    for i in range(1, len(points) - 1):
        dist = _perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            index = i

    if max_dist <= epsilon_m:
        return [start, end]

    left = douglas_peucker(points[: index + 1], epsilon_m)
    right = douglas_peucker(points[index:], epsilon_m)
    return left[:-1] + right


def _perpendicular_distance(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
) -> float:
    """Distance from `point` to the segment through the two line points."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    # Cross-product magnitude divided by segment length.
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / math.hypot(dx, dy)
