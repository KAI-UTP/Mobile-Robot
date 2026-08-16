"""Measuring the same room a second time.

A scan is a one-shot: the robot drives its lap, sweeps, saves, and the thread
ends. Measuring again meant restarting the container, which is a strange thing
to ask of someone comparing two runs of the same room.

The hazard is not the button, it is the handover. Two simulator threads
interleaving packets into one pipeline draw a room from two robots standing in
different places, and the result looks like a mapping bug rather than like two
scans fighting over one map. These tests are mostly about that.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "shared", ROOT / "services", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mapper import main as m  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet_state(tmp_path, monkeypatch):
    """Isolate global mapper state between tests."""
    m.app.state.mqtt_publisher = None
    m.app.state.coverage = None
    m.app.state.sim_settings = None
    m._scan_thread = None
    m._scan_stop.clear()
    monkeypatch.setattr(m.store, "directory", tmp_path, raising=False)
    yield
    m._scan_stop.set()
    if m._scan_thread is not None:
        m._scan_thread.join(timeout=5.0)
    m._scan_thread = None
    m._scan_stop.clear()


# ── Hardware mode ────────────────────────────────────────────────────────────


def test_with_no_simulator_it_clears_the_map_and_says_so():
    """There is nothing to restart when a real robot is the source. Silently
    doing nothing, or claiming a scan had restarted, would both be lies."""
    result = m.rescan()

    assert result["status"] == "cleared"
    assert "drive the robot" in result["detail"]


def test_clearing_actually_empties_the_map():
    m.rescan()
    assert m.pipeline.room is None


# ── Restarting a simulated scan ──────────────────────────────────────────────


def _settings(**overrides):
    base = {"room": "rectangular", "indoor": True, "speed": 200.0, "sweep": False}
    base.update(overrides)
    return base


def test_it_restarts_the_scan_it_was_actually_running():
    """Repeating a default run rather than the configured one would make two
    scans of 'the same room' incomparable, which is the whole point of running
    it twice."""
    m.app.state.sim_settings = _settings(room="furnished")
    result = m.rescan()

    assert result["status"] == "restarted"
    assert result["room"] == "furnished"


def test_a_restart_leaves_exactly_one_simulator_running():
    """The hazard this whole mechanism exists to avoid."""
    m.app.state.sim_settings = _settings()

    m.rescan()
    time.sleep(0.5)
    m.rescan()
    time.sleep(0.5)

    alive = [t for t in threading.enumerate() if t.name == "simulator" and t.is_alive()]
    assert len(alive) <= 1, f"{len(alive)} simulators running at once"


def test_the_previous_scan_is_stopped_before_the_map_is_cleared():
    """Clearing while the old robot is still writing means it immediately
    starts painting the fresh map from wherever it had drifted to."""
    m.app.state.sim_settings = _settings()
    m.rescan()
    time.sleep(0.5)

    first = m._scan_thread
    assert first is not None and first.is_alive()

    m.rescan()
    assert not first.is_alive(), "the previous simulator was still running"


def test_a_restart_wipes_the_previous_measurement():
    m.app.state.sim_settings = _settings()
    m.rescan()
    time.sleep(1.0)

    m.pipeline.refresh_room()
    m.rescan()

    # Straight after a restart the new robot has driven almost nothing, so
    # there cannot be a completed room yet.
    room = m.pipeline.room
    assert room is None or not room.is_closed


def test_a_restart_clears_the_coverage_report():
    """Otherwise /api/coverage keeps serving the last run's collisions and
    obstacles against a map they did not come from."""
    m.app.state.sim_settings = _settings()
    m.app.state.coverage = object()

    m.rescan()
    assert m.app.state.coverage is None


# ── Concurrency ──────────────────────────────────────────────────────────────


def test_two_simultaneous_restarts_do_not_both_proceed():
    """Two people watching the same dashboard, both pressing the button."""
    m.app.state.sim_settings = _settings()
    results: list[dict] = []
    barrier = threading.Barrier(2)

    def press():
        barrier.wait()
        results.append(m.rescan())

    threads = [threading.Thread(target=press) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)

    assert len(results) == 2
    assert sum(r["status"] == "restarted" for r in results) <= 1
    assert any(r["status"] in ("restarted", "busy") for r in results)


def test_a_refused_restart_says_why():
    """A button that silently does nothing is worse than one that refuses."""
    m.app.state.sim_settings = _settings()
    m._scan_lock.acquire()
    try:
        result = m.rescan()
    finally:
        m._scan_lock.release()

    assert result["status"] == "busy"
    assert result["detail"]


def test_a_busy_restart_does_not_wipe_the_map():
    """Refusing must be inert. Clearing the map and then declining to start a
    new scan would leave the user with nothing at all.

    Measured by how much has been explored, not by comparing the grid byte for
    byte: the simulator is still running and legitimately adds cells between
    the two reads, so an exact match fails for the wrong reason.
    """
    m.app.state.sim_settings = _settings()
    m.rescan()
    time.sleep(1.2)
    before = m.pipeline.grid.explored_cells()
    assert before > 0, "nothing was mapped, so the test proves nothing"

    m._scan_lock.acquire()
    try:
        assert m.rescan()["status"] == "busy"
    finally:
        m._scan_lock.release()

    assert m.pipeline.grid.explored_cells() >= before


# ── Stopping a run part-way ──────────────────────────────────────────────────


def test_a_scan_can_be_abandoned_mid_lap():
    """`stop` is what makes the handover possible at all: without it the old
    robot drives its full lap and sweep no matter what the user asked for."""
    stop = threading.Event()
    thread = m.start_sim_source("rectangular", indoor=True, speed=200.0, stop=stop)
    time.sleep(0.5)
    assert thread.is_alive()

    stop.set()
    thread.join(timeout=10.0)
    assert not thread.is_alive()


def test_start_sim_source_hands_back_its_thread():
    """The caller has to be able to join it; assuming it stopped is exactly the
    bug this mechanism is meant to prevent."""
    stop = threading.Event()
    thread = m.start_sim_source("rectangular", indoor=True, speed=200.0, stop=stop)
    try:
        assert isinstance(thread, threading.Thread)
        assert thread.daemon, "must not hold the process open on shutdown"
    finally:
        stop.set()
        thread.join(timeout=10.0)
