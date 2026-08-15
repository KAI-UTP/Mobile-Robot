"""Guards on the source files themselves.

Why this exists
---------------
The entry-point scripts (`services/*/main.py`, the Omniverse scripts) are not
imported by any test — they open serial ports, start Isaac Sim, or block on
input. So when a bulk edit corrupted three of them, the whole suite stayed
green and the damage went unnoticed.

Specifically: a PowerShell text replacement mangled UTF-8 into mojibake, and a
subsequent repair left bare CR characters as line separators. Python's
universal newlines accepts a lone CR, so `compile()` succeeded and nothing
complained — the files were valid and unreadable at the same time.

These tests check every file in the project for that class of damage. They are
fast, need nothing installed, and would have caught it immediately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".venv", "__pycache__", ".git", "node_modules", ".ruff_cache"}

# Sequences produced when UTF-8 is decoded as cp1252 and re-encoded. Each is a
# real example seen in this project: an em dash, a byte-order mark, and an
# accented character.
#
# Written as escapes rather than literals on purpose. Spelling them out would
# put the very characters being searched for into this file, and the check
# would fail on itself — which it did on the first run.
MOJIBAKE_MARKERS = (
    "\u00e2\u20ac\u201d",  # em dash U+2014, mangled
    "\u00ef\u00bb\u00bf",  # byte-order mark, mangled
    "\u00c3\u00a9",  # e-acute, mangled
)


def _project_files(suffix: str) -> list[Path]:
    return [
        path
        for path in sorted(ROOT.rglob(f"*{suffix}"))
        if not any(part in SKIP_PARTS for part in path.parts)
    ]


PYTHON_FILES = _project_files(".py")


def test_the_project_has_python_files():
    """Guards the guard: a broken glob would make everything below vacuous."""
    assert len(PYTHON_FILES) > 30


@pytest.mark.parametrize("path", PYTHON_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_python_file_compiles(path: Path):
    """Covers the entry points that no other test imports."""
    source = path.read_bytes().decode("utf-8-sig")
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        pytest.fail(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")


@pytest.mark.parametrize("path", PYTHON_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_carriage_returns(path: Path):
    """LF only.

    A lone CR is a valid line separator to Python but not to most tools, and
    mixed endings make every diff unreadable. This is the check that would
    have caught the corruption directly.
    """
    raw = path.read_bytes()
    assert b"\r" not in raw, (
        f"{path.relative_to(ROOT)} contains CR. Normalise to LF: "
        "read_bytes(), replace b'\\r\\n' then b'\\r' with b'\\n', write_bytes()."
    )


@pytest.mark.parametrize("path", PYTHON_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_mojibake(path: Path):
    """Catches UTF-8 that has been round-tripped through cp1252."""
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    found = [marker for marker in MOJIBAKE_MARKERS if marker in text]
    assert not found, (
        f"{path.relative_to(ROOT)} contains mojibake {found}. "
        "A tool has re-encoded UTF-8 as cp1252 — check any bulk edit."
    )


@pytest.mark.parametrize("path", PYTHON_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_byte_order_mark(path: Path):
    """A BOM on a .py file breaks some tooling and shows up in diffs."""
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf"), (
        f"{path.relative_to(ROOT)} starts with a UTF-8 BOM. "
        "PowerShell's Set-Content -Encoding utf8 adds one; write bytes instead."
    )


@pytest.mark.parametrize("path", PYTHON_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_files_are_valid_utf8(path: Path):
    raw = path.read_bytes()
    try:
        raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        pytest.fail(f"{path.relative_to(ROOT)} is not valid UTF-8: {exc}")
