"""Saved scans — the difference between a demo and a product.

Until now a completed scan was logged and discarded. That is fine for proving
the mapping works and useless for anything else: a flooring installer who
measures a room wants the number tomorrow, and wants the next room to be as
easy as the first.

A scan is stored as one JSON file per scan in a directory. Not a database,
deliberately — one file per scan can be copied, emailed, diffed, backed up and
inspected in a text editor, and the whole store survives the application being
uninstalled. At the scale this operates (tens of rooms, not millions) a
database would add operational burden and buy nothing.

The occupancy grid is stored gzipped and base64-encoded inside the same file
so a scan stays a single self-contained artefact. A 300x300 grid is 90 kB raw
and around 3 kB compressed, because most of it is uniform unexplored space.
"""

from __future__ import annotations

import base64
import gzip
import json
import logging
import re
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "shared", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

logger = logging.getLogger("storage")

SCHEMA_VERSION = 1

# Anything outside this is rejected from a filename. Scan names come from users
# and end up on disk, so a name like "../../etc/passwd" or "Room: 1?" must not
# become a path.
_UNSAFE = re.compile(r"[^A-Za-z0-9 _-]")


@dataclass
class ScanQuality:
    """Whether this scan can be trusted enough to quote from.

    A measurement without a confidence statement invites someone to rely on a
    bad number. The robot knows when a scan went poorly — the boundary never
    closed, half the floor was never seen, the pose drifted — so it says so
    rather than reporting an area with false precision.
    """

    is_closed: bool = False
    coverage_pct: float = 0.0
    pose_confidence: float = 0.0
    # Observed floor lying outside the reported outline. Non-zero means the
    # boundary closed somewhere it should not have; see `assess_quality`.
    floor_outside_pct: float = 0.0
    # Distance driven on the perimeter lap, over the perimeter of the outline
    # that lap produced. Below ~0.7 the robot did not go round this room.
    lap_coverage: float | None = None
    grade: str = "UNUSABLE"
    reasons: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.grade in ("GOOD", "ACCEPTABLE")


@dataclass
class Scan:
    """One completed room measurement."""

    scan_id: str
    name: str
    created_at: str
    robot_id: str

    area_m2: float
    perimeter_m: float
    long_side_m: float
    short_side_m: float
    polygon: list[dict]

    quality: ScanQuality

    # What was standing on the floor, and how much of it that covers. Kept
    # separate from `area_m2` rather than subtracted from it because the two
    # numbers are sold to different people: a flooring quote needs the floor
    # under the table, a cleaning quote does not.
    obstacles: list[dict] = field(default_factory=list)
    blocked_area_m2: float = 0.0

    distance_travelled_m: float = 0.0
    duration_s: float = 0.0
    notes: str = ""

    # Occupancy grid, gzipped + base64. Optional: a scan is still meaningful
    # without it, just not redrawable.
    grid_b64: str | None = None
    grid_meta: dict | None = None

    schema_version: int = SCHEMA_VERSION

    @property
    def usable_area_m2(self) -> float:
        """Floor the robot could actually reach, after furniture."""
        return max(0.0, self.area_m2 - self.blocked_area_m2)

    def summary(self) -> dict:
        """The subset the library list needs — deliberately without the grid.

        Listing fifty scans should not mean decompressing fifty occupancy
        grids.
        """
        return {
            "scan_id": self.scan_id,
            "name": self.name,
            "created_at": self.created_at,
            "area_m2": round(self.area_m2, 2),
            "blocked_area_m2": round(self.blocked_area_m2, 2),
            "usable_area_m2": round(self.usable_area_m2, 2),
            "obstacle_count": len(self.obstacles),
            "long_side_m": round(self.long_side_m, 2),
            "short_side_m": round(self.short_side_m, 2),
            "perimeter_m": round(self.perimeter_m, 1),
            "grade": self.quality.grade,
            "is_usable": self.quality.is_usable,
            "coverage_pct": round(self.quality.coverage_pct, 0),
            "has_grid": self.grid_b64 is not None,
        }


#: How much observed floor may lie outside the outline before the outline is
#: judged to be in the wrong place. Some overshoot is normal and harmless: the
#: robot's own footprint paints free cells a little past the wall it is hugging,
#: and the outline is simplified. A room measured properly runs a few per cent.
#: A boundary that closed early runs far higher — the furnished-room failure
#: that prompted this had most of the floor it had seen sitting outside.
FLOOR_OUTSIDE_LIMIT_PCT = 15.0

#: How much of its reported boundary the robot must actually have driven.
#:
#: A perimeter lap goes round the room, so the distance driven should be close
#: to the perimeter of the outline it produces. Measured at the moment each lap
#: closed:
#:
#:     rectangular  18.7 m driven / 20.9 m perimeter = 0.90   26.63 m2  right
#:     l-shaped     19.6 m / 21.8 m                  = 0.90   24.74 m2  right
#:     furnished    10.5 m / 21.0 m                  = 0.50   13.18 m2  WRONG
#:
#: The furnished lap wove among the furniture and curled back to its start
#: having driven half the boundary it went on to report. Under 0.7 the robot
#: did not lap that room, whatever the outline looks like.
#:
#: Slack matters: the lap runs about 0.35 m inside the wall, so a real lap is
#: always a little shorter than the perimeter it measures. 0.90 is what that
#: costs; 0.70 leaves room for a lap that had to detour around something.
LAP_COVERAGE_LIMIT = 0.70


def assess_quality(
    is_closed: bool,
    coverage_pct: float,
    pose_confidence: float,
    floor_outside_pct: float = 0.0,
    lap_coverage: float | None = None,
) -> ScanQuality:
    """Grade a scan, and say why.

    The thresholds are what the honest engineering says, not what flatters the
    result:

    * An unclosed boundary means the robot never got all the way round, so the
      area is a lower bound rather than a measurement. That alone disqualifies
      a scan no matter how good everything else looks.
    * Below 60 % observed coverage most of the "room" is inference from
      hole-filling rather than anything a sensor saw.
    * Below 0.3 pose confidence the map is not reliably self-consistent, so
      the outline may not close on the truth even if it closes on itself.
    * Floor observed OUTSIDE the outline means the outline is in the wrong
      place, however tidy it looks.
    * A robot that drove far less than the perimeter it is reporting did not
      go round that room.

    That last one exists because the first three are all measures of *internal
    consistency*, and a robot that maps half a room very tidily scores well on
    every one of them. The robot cannot drive through walls, so a path that
    leaves the outline proves the outline is not the wall — evidence the robot
    collects itself, needing no ground truth, which matters because with real
    hardware there is none.

    The last two are separate failures and need separate evidence. A lap that
    closes EARLY leaves an outline that is too small AND a path that never
    leaves it, because both were cut short together — the furnished room reads
    0.3 % of its path outside, and looks perfect. What gives it away is that
    the robot drove 10.5 m and reported a boundary 21.0 m round.
    """
    reasons: list[str] = []

    if not is_closed:
        reasons.append(
            "boundary never closed — drive the full perimeter; the area is a "
            "lower bound"
        )
    if coverage_pct < 60.0:
        reasons.append(
            f"only {coverage_pct:.0f}% of the floor was directly observed"
        )
    if pose_confidence < 0.3:
        reasons.append(
            "position estimate drifted; re-scan from the starting point"
        )
    if floor_outside_pct >= FLOOR_OUTSIDE_LIMIT_PCT:
        reasons.append(
            f"{floor_outside_pct:.0f}% of the floor the robot saw lies OUTSIDE "
            "this outline — the boundary closed early, so the area is too small"
        )
    lap_short = lap_coverage is not None and lap_coverage < LAP_COVERAGE_LIMIT
    if lap_short:
        reasons.append(
            f"the robot drove only {lap_coverage * 100:.0f}% of the boundary it "
            "is reporting — it did not go round this room, so the area is too "
            "small"
        )

    if not is_closed:
        grade = "UNUSABLE"
    elif floor_outside_pct >= FLOOR_OUTSIDE_LIMIT_PCT or lap_short:
        # However tidy the outline is, it is not the room.
        grade = "POOR"
    elif coverage_pct >= 85.0 and pose_confidence >= 0.6:
        grade = "GOOD"
    elif coverage_pct >= 60.0 and pose_confidence >= 0.3:
        grade = "ACCEPTABLE"
    else:
        grade = "POOR"

    return ScanQuality(
        is_closed=is_closed,
        coverage_pct=coverage_pct,
        pose_confidence=pose_confidence,
        floor_outside_pct=floor_outside_pct,
        lap_coverage=lap_coverage,
        grade=grade,
        reasons=reasons,
    )


def safe_name(name: str, fallback: str = "Room") -> str:
    """Make a user-supplied name safe to display and to log."""
    cleaned = _UNSAFE.sub("", (name or "").strip())[:60].strip()
    return cleaned or fallback


class ScanStore:
    """A directory of saved scans."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory or ROOT / "scans")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logger.info("Scan store at %s", self.directory)

    # ── Paths ─────────────────────────────────────────────────────────────

    def _path(self, scan_id: str) -> Path:
        """Resolve a scan id to a file, refusing anything path-like.

        Scan ids reach this from HTTP requests. Without this check a request
        for `../../secrets` would read outside the store.
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", scan_id):
            raise ValueError(f"invalid scan id: {scan_id!r}")
        return self.directory / f"{scan_id}.json"

    @staticmethod
    def new_id() -> str:
        """Sortable, unique, and readable: 20260816-142530-a1b2c3."""
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:6]}"

    # ── Grid encoding ─────────────────────────────────────────────────────

    @staticmethod
    def encode_grid(grid_bytes: bytes) -> str:
        return base64.b64encode(gzip.compress(grid_bytes, compresslevel=6)).decode()

    @staticmethod
    def decode_grid(encoded: str) -> bytes:
        return gzip.decompress(base64.b64decode(encoded))

    # ── Write ─────────────────────────────────────────────────────────────

    def save(self, scan: Scan) -> Path:
        """Write a scan atomically.

        Written to a temporary file and renamed, so a crash mid-write leaves
        the previous version intact rather than a truncated file that fails to
        parse forever after.
        """
        path = self._path(scan.scan_id)
        payload = asdict(scan)

        with self._lock:
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp.replace(path)

        logger.info(
            "Saved scan %s (%s): %.2f m2, %s",
            scan.scan_id, scan.name, scan.area_m2, scan.quality.grade,
        )
        return path

    # ── Read ──────────────────────────────────────────────────────────────

    def load(self, scan_id: str) -> Scan | None:
        path = self._path(scan_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Could not read scan %s: %s", scan_id, exc)
            return None

        quality = ScanQuality(**data.pop("quality", {}))
        # Drop unknown keys rather than raising: a scan written by a newer
        # version should still open here, minus whatever is new.
        known = {f for f in Scan.__dataclass_fields__ if f != "quality"}
        filtered = {k: v for k, v in data.items() if k in known}
        return Scan(quality=quality, **filtered)

    def list_scans(self) -> list[dict]:
        """Summaries of every saved scan, newest first."""
        summaries = []
        for path in self.directory.glob("*.json"):
            scan = self.load(path.stem)
            if scan is not None:
                summaries.append(scan.summary())
        return sorted(summaries, key=lambda s: s["created_at"], reverse=True)

    # ── Modify ────────────────────────────────────────────────────────────

    def rename(self, scan_id: str, name: str) -> bool:
        scan = self.load(scan_id)
        if scan is None:
            return False
        scan.name = safe_name(name, fallback=scan.name)
        self.save(scan)
        return True

    def set_notes(self, scan_id: str, notes: str) -> bool:
        scan = self.load(scan_id)
        if scan is None:
            return False
        scan.notes = (notes or "")[:2000]
        self.save(scan)
        return True

    def delete(self, scan_id: str) -> bool:
        path = self._path(scan_id)
        if not path.exists():
            return False
        with self._lock:
            path.unlink()
        logger.info("Deleted scan %s", scan_id)
        return True

    # ── Aggregate ─────────────────────────────────────────────────────────

    def totals(self) -> dict:
        """Headline numbers for the library page.

        Total area counts only usable scans: adding up figures the robot has
        already flagged as untrustworthy would produce a confident-looking sum
        of unreliable parts.
        """
        scans = self.list_scans()
        usable = [s for s in scans if s["is_usable"]]
        return {
            "scan_count": len(scans),
            "usable_count": len(usable),
            "total_area_m2": round(sum(s["area_m2"] for s in usable), 2),
        }
