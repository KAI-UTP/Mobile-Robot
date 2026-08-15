"""Turn a saved scan into something a person can use elsewhere.

A measurement that only exists inside the app is not a deliverable. A flooring
installer needs a number in a quote; a facilities manager needs a plan in a
report. These produce files that open in other software.

Generated server-side rather than in the browser so the same output is
available from the API, from a script, and from a phone that cannot run a
canvas export.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict


def _bounds(polygon: list[dict]) -> tuple[float, float, float, float]:
    xs = [p["x_m"] for p in polygon]
    ys = [p["y_m"] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def scan_to_svg(scan, px_per_m: float = 100.0, margin_m: float = 0.8) -> str:
    """A dimensioned floor plan.

    Drawn to scale with the dimensions written on, because the first thing
    anyone does with a floor plan is check a wall length against it. An
    unlabelled outline forces them back into the app.
    """
    polygon = scan.polygon
    if len(polygon) < 3:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="120">'
            '<text x="20" y="60" font-family="sans-serif" font-size="14">'
            "No room outline in this scan</text></svg>"
        )

    min_x, min_y, max_x, max_y = _bounds(polygon)
    width_m = max_x - min_x
    height_m = max_y - min_y

    svg_w = (width_m + margin_m * 2) * px_per_m
    svg_h = (height_m + margin_m * 2) * px_per_m + 70  # room for the caption

    def sx(x: float) -> float:
        return (x - min_x + margin_m) * px_per_m

    def sy(y: float) -> float:
        # SVG y grows downward; the world's does not.
        return (max_y - y + margin_m) * px_per_m

    points = " ".join(f"{sx(p['x_m']):.1f},{sy(p['y_m']):.1f}" for p in polygon)

    # Colour the outline by trustworthiness, so a bad scan cannot be printed
    # and mistaken for a good one.
    stroke = {
        "GOOD": "#1a7f37",
        "ACCEPTABLE": "#9a6700",
        "POOR": "#bc4c00",
        "UNUSABLE": "#cf222e",
    }.get(scan.quality.grade, "#57606a")

    caption_y = svg_h - 40
    warning = ""
    if not scan.quality.is_usable:
        reason = scan.quality.reasons[0] if scan.quality.reasons else "low quality scan"
        warning = (
            f'<text x="20" y="{svg_h - 14:.0f}" font-family="sans-serif" '
            f'font-size="13" fill="#cf222e">NOT RELIABLE — {_escape(reason)}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w:.0f}" height="{svg_h:.0f}" viewBox="0 0 {svg_w:.0f} {svg_h:.0f}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <polygon points="{points}" fill="#eef4fa" stroke="{stroke}" stroke-width="3" stroke-linejoin="round"/>
  <text x="{sx((min_x + max_x) / 2):.0f}" y="{sy(min_y) + 26:.0f}" font-family="sans-serif" font-size="14" text-anchor="middle" fill="#57606a">{width_m:.2f} m</text>
  <text x="{sx(min_x) - 14:.0f}" y="{sy((min_y + max_y) / 2):.0f}" font-family="sans-serif" font-size="14" text-anchor="middle" fill="#57606a" transform="rotate(-90 {sx(min_x) - 14:.0f} {sy((min_y + max_y) / 2):.0f})">{height_m:.2f} m</text>
  <text x="20" y="{caption_y:.0f}" font-family="sans-serif" font-size="18" font-weight="600" fill="#24292f">{_escape(scan.name)}</text>
  <text x="20" y="{caption_y + 20:.0f}" font-family="sans-serif" font-size="14" fill="#57606a">{scan.area_m2:.2f} m2 floor area &#183; {scan.long_side_m:.2f} &#215; {scan.short_side_m:.2f} m &#183; {scan.quality.grade.lower()}</text>
  {warning}
</svg>"""


def _escape(text: str) -> str:
    """Escape for XML text content.

    Scan names come from users and go straight into the SVG; an unescaped
    ampersand alone makes the file fail to open.
    """
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def scan_to_json(scan) -> str:
    """The full scan, minus the occupancy grid.

    The grid is excluded deliberately: this export is meant to be read by
    another program or a human, and a megabyte of base64 helps neither. The
    grid stays available in the stored file.
    """
    payload = asdict(scan)
    payload.pop("grid_b64", None)
    return json.dumps(payload, indent=2)


def scans_to_csv(summaries: list[dict]) -> str:
    """Every scan as a row — the format a quote gets built from.

    A spreadsheet is where a flooring or cleaning quote actually gets written,
    so this is the export most likely to be used in anger.
    """
    output = io.StringIO()
    columns = [
        "name",
        "created_at",
        "area_m2",
        "long_side_m",
        "short_side_m",
        "perimeter_m",
        "grade",
        "is_usable",
        "coverage_pct",
        "scan_id",
    ]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for summary in summaries:
        writer.writerow(summary)

    # A trailing total, because summing a column by hand is where mistakes get
    # made. Only usable scans count toward it.
    usable = [s for s in summaries if s.get("is_usable")]
    if usable:
        writer.writerow(
            {
                "name": f"TOTAL ({len(usable)} usable of {len(summaries)})",
                "area_m2": round(sum(s["area_m2"] for s in usable), 2),
            }
        )

    return output.getvalue()
