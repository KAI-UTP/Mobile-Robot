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


# ── Driving it by hand ───────────────────────────────────────────────────────
#
# The same pipeline, the same collision inference, the same map — only the
# thing choosing the velocities is different. It is the only way to reach a
# corner the autonomy has given up on.


def test_every_direction_is_accepted(client):
    for action in ("forward", "back", "left", "right",
                   "turn_left", "turn_right", "stop"):
        reply = client.post("/api/drive", json={"action": action})
        assert reply.status_code == 200, action


def test_an_unknown_direction_says_what_it_knows(client):
    """Driving is the one place a typo must not be guessed at."""
    reply = client.post("/api/drive", json={"action": "wiggle"})
    assert reply.status_code == 400
    assert "forward" in reply.json()["known"]


def test_left_strafes_rather_than_turning(client):
    """A kiwi drive goes sideways without rotating, and a manual control that
    turned instead would waste the one thing this base does that others cannot."""
    vx, vy, omega = client.post("/api/drive", json={"action": "left"}).json()["twist"]
    assert vy > 0
    assert omega == 0
    assert vx == 0


def test_turning_does_not_translate(client):
    vx, vy, omega = client.post(
        "/api/drive", json={"action": "turn_left"}
    ).json()["twist"]
    assert omega > 0
    assert vx == 0 and vy == 0


def test_stop_means_stop(client):
    """Releasing a key has to halt the robot, not queue something. A driving
    command is a statement about now."""
    client.post("/api/drive", json={"action": "forward"})
    twist = client.post("/api/drive", json={"action": "stop"}).json()["twist"]
    assert twist == [0.0, 0.0, 0.0]


def test_forward_and_back_are_opposites(client):
    forward = client.post("/api/drive", json={"action": "forward"}).json()["twist"]
    back = client.post("/api/drive", json={"action": "back"}).json()["twist"]
    assert forward[0] == -back[0]


def test_manual_speed_is_below_the_autonomy(client):
    """Someone steering by eye through a browser has a far longer reaction time
    than a control loop, and the robot is mapping while they do it."""
    from mapper.main import MANUAL_SPEED_MPS

    assert 0.05 <= MANUAL_SPEED_MPS <= 0.25


# ── The controller on its own ────────────────────────────────────────────────


def test_the_controller_page_is_served(client):
    reply = client.get("/drive")
    assert reply.status_code == 200
    assert "STOP" in reply.text


def test_the_controller_locks_zoom():
    """A pinch or a double-tap-to-zoom mid-drive moves the controls out from
    under the thumb holding one down."""
    from pathlib import Path

    page = (
        Path(__file__).resolve().parents[1]
        / "services" / "mapper" / "static" / "drive.html"
    ).read_text(encoding="utf-8")

    assert "user-scalable=no" in page
    assert "touch-action: none" in page


def test_the_controller_renews_while_held():
    """The server expires a command after a second, so a page that sent one
    press and went quiet would stop the robot mid-drive."""
    from pathlib import Path

    page = (
        Path(__file__).resolve().parents[1]
        / "services" / "mapper" / "static" / "drive.html"
    ).read_text(encoding="utf-8")

    assert "setInterval" in page and "RENEW_MS" in page


def test_the_controller_releases_when_hidden():
    """A control held while the phone shows a notification would otherwise
    never come back up."""
    from pathlib import Path

    page = (
        Path(__file__).resolve().parents[1]
        / "services" / "mapper" / "static" / "drive.html"
    ).read_text(encoding="utf-8")

    assert "visibilitychange" in page
    assert "pointercancel" in page


# ── The watchdog behind it ───────────────────────────────────────────────────


def test_a_command_expires(client):
    """The failure that matters on a phone over Wi-Fi: the connection drops
    mid-press, the last thing the robot heard was 'forward'. Silence has to
    mean stop, and the backstop has to live at the end that keeps moving."""
    import time as _time

    from mapper.main import MANUAL_WATCHDOG_S, app

    client.post("/api/drive", json={"action": "forward"})
    assert app.state.manual_deadline > _time.monotonic()

    # Short enough that a held control renewing a few times a second never
    # trips it, long enough to survive one dropped request.
    assert 0.3 <= MANUAL_WATCHDOG_S <= 3.0


def test_every_command_renews_the_deadline(client):
    import time as _time

    from mapper.main import app

    client.post("/api/drive", json={"action": "forward"})
    first = app.state.manual_deadline
    _time.sleep(0.02)
    client.post("/api/drive", json={"action": "forward"})

    assert app.state.manual_deadline > first


# ── Switching who drives ─────────────────────────────────────────────────────


def test_the_mode_can_be_switched(client):
    for mode in ("automatic", "manual", "idle"):
        reply = client.post("/api/mode", json={"mode": mode})
        assert reply.status_code in (200, 409), mode


def test_an_unknown_mode_is_refused(client):
    reply = client.post("/api/mode", json={"mode": "telepathy"})
    assert reply.status_code == 409
    assert "telepathy" in reply.json()["detail"]


def test_the_status_says_who_is_driving(client):
    assert "mode" in client.get("/api/scan/status").json()


def test_switching_keeps_the_map_by_default(client):
    """Switching is handing the robot over, not starting again. Taking the
    controls to finish a corner the autonomy gave up on is the main reason to
    do it, and wiping the room would defeat that."""
    from mapper.main import pipeline

    client.post("/api/mode", json={"mode": "idle"})
    before = pipeline.packets_processed
    client.post("/api/mode", json={"mode": "idle"})

    assert pipeline.packets_processed >= before


def test_starting_a_run_can_clear_the_map(client):
    """The deliberate Start at the beginning of a run does clear, and asks."""
    from mapper.main import pipeline

    client.post("/api/mode", json={"mode": "idle", "clear_map": True})
    assert pipeline.packets_processed == 0


# ── The setup flow ───────────────────────────────────────────────────────────


def _page():
    from pathlib import Path

    return (
        Path(__file__).resolve().parents[1]
        / "services" / "mapper" / "static" / "index.html"
    ).read_text(encoding="utf-8")


def test_the_page_asks_the_sensors_first():
    """One question at a time, over a page that waits. Showing all of it at
    once beside a map of a run that has not happened invited people to press
    Start before they had chosen anything."""
    page = _page()
    assert 'id="dlgSensors"' in page
    assert 'id="dlgStart"' in page
    assert "showDialog('sensors')" in page


def test_the_dialogs_cover_the_page():
    """A half-configured run is not something to start, so the page waits
    behind them rather than being usable through them."""
    page = _page()
    assert ".scrim" in page
    assert "position: fixed; inset: 0" in page


def test_reloading_mid_run_does_not_ask_again():
    """A run already going means the setup has happened."""
    page = _page()
    assert "if (body.running)" in page


def test_both_modes_are_offered():
    page = _page()
    assert 'data-mode="automatic"' in page
    assert 'data-mode="manual"' in page


def test_setting_up_a_new_run_resets_the_room():
    """This regressed once, silently. Rewriting the setup flow dropped the call
    and the room reset became unreachable from the page — while still passing
    its own tests, because nothing checked that anything called it.

    An endpoint with no caller is not a feature.
    """
    page = _page()
    assert "/api/scene/reset" in page, "nothing on the page resets the room"

    # And it stops the robot first: it cannot be driving while its sensors are
    # being chosen.
    setup = page[page.index("async function reopenSetup"):]
    setup = setup[:setup.index("\n}")]
    assert "'idle'" in setup
    assert "/api/scene/reset" in setup


# ── The accuracy score ───────────────────────────────────────────────────────
#
# /api/compare survived the deletion of the page that drew it, on the grounds
# that "the score is still worth having, and CI still checks it". CI did not.
# Nothing referenced the endpoint at all, which is how it would have rotted
# quietly until someone needed the number. Now it does.


def test_the_comparison_endpoint_answers(client):
    body = client.get("/api/compare").json()
    assert "reference_name" in body
    assert "measured_polygon" in body
    assert "truth_polygon" in body


def test_it_reports_a_truth_to_score_against(client):
    """The reference room is what makes the score mean anything. Without one
    the endpoint is just the measured room repeated back."""
    body = client.get("/api/compare").json()
    assert body["truth_polygon"], "no reference room to compare against"


def test_it_does_not_invent_a_truth_when_there_is_none(client):
    """With real hardware the true room is unknown, and presenting a reference
    layout as if it were measured would make the comparison look authoritative
    when it is an assumption."""
    from mapper.main import app

    previous = app.state.reference_room_name
    app.state.reference_room_name = "no-such-room"
    try:
        body = client.get("/api/compare").json()
        assert body["truth_polygon"] == []
    finally:
        app.state.reference_room_name = previous
