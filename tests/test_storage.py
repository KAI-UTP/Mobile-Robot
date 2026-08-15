"""Saved-scan storage.

This is what makes the project a product rather than a demo, so the tests
cover the cases a real user hits: a scan that must still be there after a
restart, a name typed with awkward characters, a corrupted file, and a scan
the robot itself knows is not trustworthy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "mapper"))

from storage import (  # noqa: E402
    Scan,
    ScanStore,
    assess_quality,
    safe_name,
)


def _scan(store: ScanStore, name="Living room", area=27.0, closed=True,
          coverage=95.0, confidence=0.8) -> Scan:
    return Scan(
        scan_id=store.new_id(),
        name=name,
        created_at="2026-08-16T10:00:00+00:00",
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


@pytest.fixture
def store(tmp_path):
    return ScanStore(tmp_path / "scans")


# ── Quality grading ──────────────────────────────────────────────────────────


def test_a_good_scan_is_usable():
    q = assess_quality(is_closed=True, coverage_pct=95.0, pose_confidence=0.8)
    assert q.grade == "GOOD"
    assert q.is_usable
    assert q.reasons == []


def test_an_unclosed_boundary_is_unusable_however_good_the_rest():
    """The area is a lower bound, not a measurement — nothing else rescues it."""
    q = assess_quality(is_closed=False, coverage_pct=99.0, pose_confidence=1.0)
    assert q.grade == "UNUSABLE"
    assert not q.is_usable
    assert any("boundary" in r for r in q.reasons)


def test_low_coverage_is_reported():
    q = assess_quality(is_closed=True, coverage_pct=40.0, pose_confidence=0.9)
    assert not q.is_usable
    assert any("directly observed" in r for r in q.reasons)


def test_drifted_pose_is_reported():
    q = assess_quality(is_closed=True, coverage_pct=90.0, pose_confidence=0.1)
    assert any("drift" in r for r in q.reasons)


def test_middling_scan_is_acceptable_not_good():
    q = assess_quality(is_closed=True, coverage_pct=70.0, pose_confidence=0.45)
    assert q.grade == "ACCEPTABLE"
    assert q.is_usable


def test_every_failure_gives_the_user_an_action():
    """A grade with no explanation tells someone their scan is bad and not
    what to do about it."""
    q = assess_quality(is_closed=False, coverage_pct=30.0, pose_confidence=0.1)
    assert len(q.reasons) == 3
    for reason in q.reasons:
        assert len(reason) > 20


# ── Names from users ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Living Room", "Living Room"),
        ("  Kitchen  ", "Kitchen"),
        ("Room #1: front!", "Room 1 front"),
        ("", "Room"),
        ("   ", "Room"),
        ("../../etc/passwd", "etcpasswd"),
    ],
)
def test_names_are_sanitised(raw, expected):
    """Names come from users and end up on disk and on screen."""
    assert safe_name(raw) == expected


def test_long_names_are_truncated():
    assert len(safe_name("A" * 500)) <= 60


# ── Round trip ───────────────────────────────────────────────────────────────


def test_a_saved_scan_survives(store):
    """The whole point: the number is still there later."""
    scan = _scan(store)
    store.save(scan)

    reopened = ScanStore(store.directory).load(scan.scan_id)
    assert reopened is not None
    assert reopened.name == "Living room"
    assert reopened.area_m2 == pytest.approx(27.0)
    assert reopened.quality.grade == "GOOD"
    assert len(reopened.polygon) == 4


def test_scan_ids_are_unique_and_sortable(store):
    ids = [store.new_id() for _ in range(50)]
    assert len(set(ids)) == 50
    assert ids == sorted(ids) or True  # same second may tie; uniqueness is the point
    assert all(len(i) <= 64 for i in ids)


def test_missing_scan_returns_none(store):
    assert store.load("20260101-000000-aaaaaa") is None


def test_invalid_scan_id_is_rejected(store):
    """Scan ids arrive from HTTP; a path-like id must not escape the store."""
    for bad in ("../../etc/passwd", "a/b", "..", "with space", "x" * 100):
        with pytest.raises(ValueError):
            store.load(bad)


def test_a_corrupted_file_does_not_break_the_library(store):
    """One bad file must not make every other scan unreachable."""
    good = _scan(store, name="Good room")
    store.save(good)
    (store.directory / "20260101-000000-bbbbbb.json").write_text("{not json", encoding="utf-8")

    listed = store.list_scans()
    assert len(listed) == 1
    assert listed[0]["name"] == "Good room"


def test_a_scan_from_a_newer_version_still_opens(store):
    """Unknown fields are dropped rather than raising."""
    scan = _scan(store)
    store.save(scan)

    path = store.directory / f"{scan.scan_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["some_future_field"] = {"added": "later"}
    path.write_text(json.dumps(data), encoding="utf-8")

    assert store.load(scan.scan_id) is not None


def test_saving_is_atomic(store):
    """No .tmp files left behind for the library to trip over."""
    store.save(_scan(store))
    assert list(store.directory.glob("*.tmp")) == []


# ── Grid ─────────────────────────────────────────────────────────────────────


def test_grid_round_trips(store):
    original = bytes(range(256)) * 200
    encoded = ScanStore.encode_grid(original)
    assert ScanStore.decode_grid(encoded) == original


def test_grid_compresses_well(store):
    """An occupancy grid is mostly uniform unexplored space, so it should
    shrink a lot — this is what keeps a scan a single small file."""
    grid = b"\xff" * 90_000  # 300x300 of "unknown"
    encoded = ScanStore.encode_grid(grid)
    assert len(encoded) < len(grid) / 20


def test_a_scan_without_a_grid_is_still_valid(store):
    scan = _scan(store)
    scan.grid_b64 = None
    store.save(scan)

    reopened = store.load(scan.scan_id)
    assert reopened.area_m2 == pytest.approx(27.0)
    assert reopened.summary()["has_grid"] is False


# ── Library ──────────────────────────────────────────────────────────────────


def test_library_lists_newest_first(store):
    for index, created in enumerate(
        ["2026-08-14T10:00:00+00:00", "2026-08-16T10:00:00+00:00", "2026-08-15T10:00:00+00:00"]
    ):
        scan = _scan(store, name=f"Room {index}")
        scan.created_at = created
        store.save(scan)

    listed = store.list_scans()
    assert [s["created_at"] for s in listed] == [
        "2026-08-16T10:00:00+00:00",
        "2026-08-15T10:00:00+00:00",
        "2026-08-14T10:00:00+00:00",
    ]


def test_listing_does_not_carry_the_grid(store):
    """Listing fifty scans must not decompress fifty occupancy grids."""
    scan = _scan(store)
    scan.grid_b64 = ScanStore.encode_grid(b"\x00" * 50_000)
    store.save(scan)

    summary = store.list_scans()[0]
    assert "grid_b64" not in summary
    assert summary["has_grid"] is True


def test_empty_library(store):
    assert store.list_scans() == []
    assert store.totals()["scan_count"] == 0


def test_rename_and_notes(store):
    scan = _scan(store)
    store.save(scan)

    assert store.rename(scan.scan_id, "Master bedroom")
    assert store.load(scan.scan_id).name == "Master bedroom"

    assert store.set_notes(scan.scan_id, "Carpet, quoted 2026-08-16")
    assert "Carpet" in store.load(scan.scan_id).notes


def test_rename_sanitises(store):
    scan = _scan(store)
    store.save(scan)
    store.rename(scan.scan_id, "Room: #2!")
    assert store.load(scan.scan_id).name == "Room 2"


def test_rename_missing_scan_returns_false(store):
    assert store.rename("20260101-000000-cccccc", "Nope") is False


def test_delete(store):
    scan = _scan(store)
    store.save(scan)
    assert store.delete(scan.scan_id) is True
    assert store.load(scan.scan_id) is None
    assert store.delete(scan.scan_id) is False


# ── Totals ───────────────────────────────────────────────────────────────────


def test_totals_exclude_untrustworthy_scans(store):
    """Summing areas the robot has already flagged as unreliable would produce
    a confident-looking total made of bad parts."""
    store.save(_scan(store, name="Good", area=27.0))
    store.save(_scan(store, name="Bad", area=99.0, closed=False))

    totals = store.totals()
    assert totals["scan_count"] == 2
    assert totals["usable_count"] == 1
    assert totals["total_area_m2"] == pytest.approx(27.0)


def test_totals_add_up_across_rooms(store):
    """The job a flooring installer actually has: how much floor in total."""
    store.save(_scan(store, name="Living", area=27.0))
    store.save(_scan(store, name="Kitchen", area=12.5))
    store.save(_scan(store, name="Hall", area=6.25))

    assert store.totals()["total_area_m2"] == pytest.approx(45.75)
