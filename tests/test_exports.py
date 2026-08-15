"""Exports — the files that leave the app.

A measurement nobody can get out is not a deliverable, and an export that
opens broken is worse than none. These check the output is well-formed and
that a bad scan cannot be exported looking like a good one.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "mapper"))

from exports import scan_to_json, scan_to_svg, scans_to_csv  # noqa: E402
from storage import Scan, ScanStore, assess_quality  # noqa: E402


def _scan(name="Living room", area=27.0, closed=True, coverage=95.0, confidence=0.8):
    return Scan(
        scan_id="20260816-120000-abc123",
        name=name,
        created_at="2026-08-16T12:00:00+00:00",
        robot_id="MR3W01",
        area_m2=area,
        perimeter_m=21.0,
        long_side_m=6.0,
        short_side_m=4.5,
        polygon=[{"x_m": 0.0, "y_m": 0.0}, {"x_m": 6.0, "y_m": 0.0},
                 {"x_m": 6.0, "y_m": 4.5}, {"x_m": 0.0, "y_m": 4.5}],
        quality=assess_quality(closed, coverage, confidence),
        distance_travelled_m=23.5,
    )


# ── SVG ──────────────────────────────────────────────────────────────────────


def test_svg_is_well_formed_xml():
    """It has to open in a browser and in Illustrator."""
    ElementTree.fromstring(scan_to_svg(_scan()))


def test_svg_is_drawn_to_scale():
    """A floor plan that is not to scale is a picture, not a plan."""
    svg = scan_to_svg(_scan(), px_per_m=100.0, margin_m=0.5)
    root = ElementTree.fromstring(svg)
    # 6 m wide plus 0.5 m margin each side, at 100 px/m.
    assert float(root.get("width")) == pytest.approx(700.0, abs=1.0)


def test_svg_carries_the_measurements():
    """The first thing anyone does is check a wall length against the plan."""
    svg = scan_to_svg(_scan())
    assert "27.00 m2" in svg
    assert "6.00" in svg and "4.50" in svg
    assert "Living room" in svg


def test_an_unreliable_scan_is_marked_on_the_plan():
    """A printed plan outlives the app. Without this, a bad scan becomes a
    piece of paper that looks authoritative."""
    svg = scan_to_svg(_scan(closed=False))
    assert "NOT RELIABLE" in svg
    ElementTree.fromstring(svg)


def test_a_good_scan_carries_no_warning():
    assert "NOT RELIABLE" not in scan_to_svg(_scan())


def test_grade_changes_the_outline_colour():
    good = scan_to_svg(_scan())
    bad = scan_to_svg(_scan(closed=False))
    assert good != bad


def test_special_characters_in_a_name_do_not_break_the_file():
    """Names come from users; a bare ampersand alone makes SVG fail to parse."""
    svg = scan_to_svg(_scan(name="Tom & Jerry's <Room>"))
    ElementTree.fromstring(svg)
    assert "&amp;" in svg


def test_an_empty_polygon_produces_a_valid_placeholder():
    scan = _scan()
    scan.polygon = []
    svg = scan_to_svg(scan)
    ElementTree.fromstring(svg)
    assert "No room outline" in svg


# ── JSON ─────────────────────────────────────────────────────────────────────


def test_json_export_round_trips():
    import json

    data = json.loads(scan_to_json(_scan()))
    assert data["area_m2"] == pytest.approx(27.0)
    assert data["quality"]["grade"] == "GOOD"
    assert len(data["polygon"]) == 4


def test_json_export_omits_the_grid():
    """This export is for humans and other programs; a megabyte of base64
    helps neither, and the grid stays in the stored file."""
    import json

    scan = _scan()
    scan.grid_b64 = ScanStore.encode_grid(b"\x00" * 50_000)

    data = json.loads(scan_to_json(scan))
    assert "grid_b64" not in data
    assert len(scan_to_json(scan)) < 5_000


# ── CSV ──────────────────────────────────────────────────────────────────────


def test_csv_has_a_header_and_a_row_per_scan():
    rows = scans_to_csv([_scan(name="A").summary(), _scan(name="B").summary()])
    lines = rows.strip().splitlines()
    assert lines[0].startswith("name,created_at,area_m2")
    assert len(lines) == 4  # header + 2 scans + total


def test_csv_totals_only_usable_scans():
    """Summing areas the robot flagged as unreliable would produce a
    confident-looking total made of bad parts."""
    summaries = [
        _scan(name="Good", area=27.0).summary(),
        _scan(name="Bad", area=99.0, closed=False).summary(),
    ]
    text = scans_to_csv(summaries)
    assert "27.0" in text
    assert "TOTAL (1 usable of 2)" in text


def test_csv_parses_as_csv():
    import csv
    import io

    summaries = [_scan(name="Room, with comma").summary()]
    parsed = list(csv.DictReader(io.StringIO(scans_to_csv(summaries))))
    assert parsed[0]["name"] == "Room, with comma"


def test_empty_csv_still_has_a_header():
    assert scans_to_csv([]).strip().startswith("name,")
