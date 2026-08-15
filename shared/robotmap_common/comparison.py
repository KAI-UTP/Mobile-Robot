"""Compare the room the robot drew against the room that actually exists.

This is what turns the two-screen display into a result. Showing a reference
floor plan beside the robot's map is a picture; putting a number on the
difference is twin fidelity, and it is the number a research project is
actually assessed on.

Four measures, because each one hides a different kind of error
---------------------------------------------------------------
* **Area error** is the headline, and the easiest to fake. A map that is too
  wide and too short can land on exactly the right area while being the wrong
  shape entirely.
* **Dimension error** catches that, but not position: a correctly sized room
  drawn two metres to the left scores perfectly.
* **IoU** (intersection over union) catches both. It is the honest single
  number — it only approaches 1.0 when the two rooms are the same size, the
  same shape, *and* in the same place.
* **Centroid offset** separates "wrong shape" from "right shape, wrong
  place", which usually means odometry drift rather than a mapping fault.

Alignment
---------
The robot's map is in its own frame, whose origin is wherever it was switched
on and whose axes point wherever it happened to be facing. Comparing raw
coordinates would measure that arbitrary offset rather than any real error, so
the polygons are aligned first — see `align_polygons`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import min_area_rect, polygon_area_m2, polygon_perimeter_m

Point = tuple[float, float]


@dataclass
class ComparisonResult:
    """How closely the robot's map matches the real room."""

    truth_area_m2: float
    measured_area_m2: float
    area_error_pct: float

    truth_dimensions_m: tuple[float, float]
    measured_dimensions_m: tuple[float, float]
    long_side_error_pct: float
    short_side_error_pct: float

    iou: float
    centroid_offset_m: float
    perimeter_error_pct: float

    @property
    def grade(self) -> str:
        """A one-word summary, for the dashboard.

        Banded on IoU rather than area, because area alone is the measure most
        easily satisfied by a wrong map.
        """
        if self.iou >= 0.90:
            return "EXCELLENT"
        if self.iou >= 0.80:
            return "GOOD"
        if self.iou >= 0.65:
            return "FAIR"
        return "POOR"

    def summary(self) -> str:
        return (
            f"{self.measured_area_m2:.2f} m2 vs {self.truth_area_m2:.2f} m2 truth "
            f"({self.area_error_pct:+.1f}%), IoU {self.iou:.2f} — {self.grade}"
        )


# ── Alignment ────────────────────────────────────────────────────────────────


def centroid(polygon: list[Point]) -> Point:
    """Area-weighted centroid of a simple polygon.

    Not the mean of the vertices: that is pulled toward whichever edge happens
    to have the most points, and a traced map has far more vertices along the
    walls it saw closely.
    """
    if len(polygon) < 3:
        if not polygon:
            return (0.0, 0.0)
        return (
            sum(p[0] for p in polygon) / len(polygon),
            sum(p[1] for p in polygon) / len(polygon),
        )

    cx = cy = signed_area = 0.0
    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % len(polygon)]
        cross = x0 * y1 - x1 * y0
        signed_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    signed_area *= 0.5
    if abs(signed_area) < 1e-12:
        return (
            sum(p[0] for p in polygon) / len(polygon),
            sum(p[1] for p in polygon) / len(polygon),
        )

    return (cx / (6.0 * signed_area), cy / (6.0 * signed_area))


def rotate_polygon(polygon: list[Point], angle_deg: float, about: Point) -> list[Point]:
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    ox, oy = about
    return [
        (
            ox + (x - ox) * cos_a - (y - oy) * sin_a,
            oy + (x - ox) * sin_a + (y - oy) * cos_a,
        )
        for x, y in polygon
    ]


def translate_polygon(polygon: list[Point], dx: float, dy: float) -> list[Point]:
    return [(x + dx, y + dy) for x, y in polygon]


def align_polygons(
    measured: list[Point], truth: list[Point], align_rotation: bool = True
) -> list[Point]:
    """Move the measured polygon onto the truth polygon for comparison.

    The robot's map frame has an arbitrary origin and an arbitrary orientation
    — both set by where it happened to be standing and facing at startup.
    Comparing raw coordinates would mostly measure that, so the measured
    polygon is first rotated to match the truth's principal axis, then
    translated so the centroids coincide.

    Only rigid motion is applied: no scaling and no shearing. Scaling would
    hide exactly the error worth finding, since a map that is uniformly 10 %
    too small is 10 % wrong and must score as such.
    """
    if len(measured) < 3 or len(truth) < 3:
        return list(measured)

    aligned = list(measured)

    if align_rotation:
        _, _, truth_angle = min_area_rect(truth)
        _, _, measured_angle = min_area_rect(aligned)
        # A rectangle's orientation only means anything modulo 90 degrees.
        delta = (truth_angle - measured_angle) % 90.0
        if delta > 45.0:
            delta -= 90.0
        aligned = rotate_polygon(aligned, delta, centroid(aligned))

    truth_centre = centroid(truth)
    measured_centre = centroid(aligned)
    return translate_polygon(
        aligned,
        truth_centre[0] - measured_centre[0],
        truth_centre[1] - measured_centre[1],
    )


# ── Overlap ──────────────────────────────────────────────────────────────────


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Even-odd ray casting."""
    x, y = point
    inside = False
    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % len(polygon)]
        if (y0 > y) != (y1 > y):
            t = (y - y0) / (y1 - y0)
            if x0 + t * (x1 - x0) > x:
                inside = not inside
    return inside


def intersection_over_union(
    polygon_a: list[Point], polygon_b: list[Point], resolution_m: float = 0.02
) -> float:
    """Overlap of two polygons, from 0 (disjoint) to 1 (identical).

    Computed by sampling a grid rather than by exact polygon clipping.
    Clipping is precise but fiddly to get right for concave shapes, and a
    traced room map is reliably concave; a 2 cm sample grid is well inside the
    accuracy of the map being judged, and it cannot produce a wrong answer on
    a self-touching polygon the way a naive clipper can.
    """
    if len(polygon_a) < 3 or len(polygon_b) < 3:
        return 0.0

    xs = [p[0] for p in polygon_a] + [p[0] for p in polygon_b]
    ys = [p[1] for p in polygon_a] + [p[1] for p in polygon_b]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if max_x - min_x < resolution_m or max_y - min_y < resolution_m:
        return 0.0

    # Cap the sample count so a huge or very fine input cannot stall the UI.
    columns = int((max_x - min_x) / resolution_m) + 1
    rows = int((max_y - min_y) / resolution_m) + 1
    max_samples = 500_000
    if columns * rows > max_samples:
        scale = math.sqrt(columns * rows / max_samples)
        resolution_m *= scale
        columns = int((max_x - min_x) / resolution_m) + 1
        rows = int((max_y - min_y) / resolution_m) + 1

    intersection = 0
    union = 0

    for row in range(rows):
        y = min_y + (row + 0.5) * resolution_m
        for col in range(columns):
            x = min_x + (col + 0.5) * resolution_m
            in_a = point_in_polygon((x, y), polygon_a)
            in_b = point_in_polygon((x, y), polygon_b)
            if in_a and in_b:
                intersection += 1
            if in_a or in_b:
                union += 1

    return intersection / union if union else 0.0


# ── The comparison ───────────────────────────────────────────────────────────


def _percent_error(measured: float, truth: float) -> float:
    if abs(truth) < 1e-9:
        return 0.0
    return (measured - truth) / truth * 100.0


def compare_rooms(
    measured: list[Point],
    truth: list[Point],
    align: bool = True,
    iou_resolution_m: float = 0.02,
) -> ComparisonResult:
    """Measure how closely the robot's map matches the real room."""
    if len(measured) < 3 or len(truth) < 3:
        return ComparisonResult(
            truth_area_m2=polygon_area_m2(truth) if len(truth) >= 3 else 0.0,
            measured_area_m2=polygon_area_m2(measured) if len(measured) >= 3 else 0.0,
            area_error_pct=0.0,
            truth_dimensions_m=(0.0, 0.0),
            measured_dimensions_m=(0.0, 0.0),
            long_side_error_pct=0.0,
            short_side_error_pct=0.0,
            iou=0.0,
            centroid_offset_m=0.0,
            perimeter_error_pct=0.0,
        )

    # Offset is measured BEFORE alignment — aligning is what removes it, so
    # taking it afterwards would always report zero.
    raw_offset = math.dist(centroid(measured), centroid(truth))

    compared = align_polygons(measured, truth) if align else list(measured)

    truth_area = polygon_area_m2(truth)
    measured_area = polygon_area_m2(compared)

    truth_long, truth_short, _ = min_area_rect(truth)
    measured_long, measured_short, _ = min_area_rect(compared)

    return ComparisonResult(
        truth_area_m2=truth_area,
        measured_area_m2=measured_area,
        area_error_pct=_percent_error(measured_area, truth_area),
        truth_dimensions_m=(truth_long, truth_short),
        measured_dimensions_m=(measured_long, measured_short),
        long_side_error_pct=_percent_error(measured_long, truth_long),
        short_side_error_pct=_percent_error(measured_short, truth_short),
        iou=intersection_over_union(compared, truth, iou_resolution_m),
        centroid_offset_m=raw_offset,
        perimeter_error_pct=_percent_error(
            polygon_perimeter_m(compared), polygon_perimeter_m(truth)
        ),
    )


# ── Reference rooms ──────────────────────────────────────────────────────────

# The ground-truth shapes the simulator drives inside. Having them here means
# the comparison view can score a simulated run with no measuring tape, which
# is what makes the two-screen demo work before the real robot exists.
REFERENCE_ROOMS: dict[str, list[Point]] = {
    "rectangular": [(0.0, 0.0), (6.0, 0.0), (6.0, 4.5), (0.0, 4.5)],
    "l-shaped": [
        (0.0, 0.0),
        (6.0, 0.0),
        (6.0, 3.0),
        (3.5, 3.0),
        (3.5, 5.0),
        (0.0, 5.0),
    ],
    "furnished": [(0.0, 0.0), (6.0, 0.0), (6.0, 4.5), (0.0, 4.5)],
}


def reference_room(name: str) -> list[Point] | None:
    return REFERENCE_ROOMS.get(name)
