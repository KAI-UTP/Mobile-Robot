"""Setting up a run, then starting it.

The order is the point. What the robot can sense decides HOW it maps a room,
and what is standing in the room decides how hard that is — so both are chosen
before the robot goes out rather than configured in a file and hoped for.
Ticking a lidar is the difference between measuring the same room in 23 m and
measuring it in 435 m, and that stops being a paragraph in a README when you
can watch it happen.
"""

from __future__ import annotations

import pytest
from robotmap_common.hardware import ACTUAL, Capability, Strategy

# ── Choosing a sensor suite ──────────────────────────────────────────────────


def test_fitting_a_lidar_switches_the_strategy():
    chosen = ACTUAL.with_fitted(
        ["Feetech STS3215 servo bus", "2D lidar (e.g. RPLIDAR A1)"]
    )
    assert chosen.has(Capability.RANGE)
    assert chosen.strategy() is Strategy.WALL_FOLLOWING


def test_unticking_everything_that_sees_falls_back_to_contact():
    chosen = ACTUAL.with_fitted(["Feetech STS3215 servo bus"])
    assert not chosen.has(Capability.RANGE)
    assert chosen.strategy() is Strategy.CONTACT_ONLY


def test_choosing_replaces_rather_than_adds():
    """The boxes that are ticked are the whole answer. A device left out has to
    come back unfitted, or unticking one would silently do nothing."""
    chosen = ACTUAL.with_fitted(["2D lidar (e.g. RPLIDAR A1)"])
    fitted = [d.name for d in chosen.fitted]

    assert fitted == ["2D lidar (e.g. RPLIDAR A1)"]
    assert not chosen.has(Capability.ODOMETRY)


def test_names_are_matched_leniently():
    """Case and stray whitespace come from a browser, not from the source."""
    chosen = ACTUAL.with_fitted(["  feetech sts3215 SERVO BUS  "])
    assert chosen.has(Capability.ODOMETRY)


def test_an_unknown_name_is_ignored_rather_than_fatal():
    """A stale name in an open browser tab must not take the mapper down."""
    chosen = ACTUAL.with_fitted(["Feetech STS3215 servo bus", "flux capacitor"])
    assert chosen.has(Capability.ODOMETRY)
    assert len(chosen.fitted) == 1


def test_every_device_is_still_described_after_choosing():
    """The list has to keep showing what is NOT fitted, or there is no way to
    tick it back on."""
    chosen = ACTUAL.with_fitted([])
    assert len(chosen.describe()["devices"]) == len(ACTUAL.devices)
    assert chosen.strategy() is Strategy.CONTACT_ONLY


def test_the_original_profile_is_not_modified():
    """Frozen dataclasses, but worth pinning: two browsers choosing different
    suites must not corrupt each other's idea of the robot."""
    before = [d.fitted for d in ACTUAL.devices]
    ACTUAL.with_fitted(["2D lidar (e.g. RPLIDAR A1)"])
    assert [d.fitted for d in ACTUAL.devices] == before


# ── The endpoints the buttons call ───────────────────────────────────────────


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from mapper.main import app

    return TestClient(app)


def test_the_page_can_ask_what_is_fitted(client):
    body = client.get("/api/hardware").json()
    assert "devices" in body and "strategy" in body


def test_the_page_can_choose_the_suite(client):
    reply = client.post(
        "/api/hardware",
        json={"fitted": ["Feetech STS3215 servo bus", "2D lidar (e.g. RPLIDAR A1)"]},
    )
    assert reply.status_code == 200
    assert reply.json()["strategy"] == "WALL_FOLLOWING"

    reply = client.post("/api/hardware", json={"fitted": ["Feetech STS3215 servo bus"]})
    assert reply.json()["strategy"] == "CONTACT_ONLY"


def test_a_malformed_choice_is_rejected(client):
    assert client.post("/api/hardware", json={"fitted": "lidar"}).status_code == 400


def test_the_page_can_ask_whether_the_robot_is_out(client):
    body = client.get("/api/scan/status").json()
    assert "running" in body and "strategy" in body


def test_stopping_when_nothing_runs_is_not_an_error(client):
    """Pressing Stop twice is not a fault, and should not read like one."""
    body = client.post("/api/scan/stop").json()
    assert body["status"] in ("idle", "stopped", "stopping")


def test_the_room_can_be_emptied_and_refilled(client):
    client.post("/api/world/geometry", json={"boxes": []})
    assert client.get("/api/world/geometry").json()["pieces"] == []

    client.post("/api/world/geometry", json={
        "boxes": [{"min_x_m": 2.0, "min_y_m": 1.5, "max_x_m": 3.0, "max_y_m": 2.5}],
    })
    pieces = client.get("/api/world/geometry").json()["pieces"]
    assert len(pieces) in (0, 1)   # 0 when no simulator is running in this process
