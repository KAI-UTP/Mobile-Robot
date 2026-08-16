"""Noticing a crash on a robot with no bumper.

The robot's whole sensor list is a servo bus, BLE beacons and GNSS. There is no
contact switch, so a collision cannot be read — it has to be inferred.

The tempting inference, "the reported position stopped changing", does not work
on its own and these tests pin down why: BLE is 2.71 m and GNSS indoors ~25 m,
while a collision stops the robot moving by centimetres. Worse, wheel odometry
keeps advancing the pose straight through the obstacle, because the wheels are
still turning. What actually sees it is the servo bus.
"""

from __future__ import annotations

import pytest
from robotmap_common.collision import (
    CollisionConfig,
    CollisionDetector,
)

DT = 0.1
CRUISE = 0.18


def _cycle(detector, *, commanded=CRUISE, delivering=1.0, load=0.3,
           x=0.0, y=0.0, heading=0.0, dt=DT):
    """One control cycle, three wheels all behaving the same way."""
    wheels = [commanded, commanded, commanded]
    return detector.update(
        commanded_speed_mps=commanded,
        commanded_wheel_speeds=wheels,
        measured_wheel_speeds=[w * delivering for w in wheels],
        wheel_loads=[load, load, load],
        x_m=x, y_m=y, heading_deg=heading, dt_s=dt,
    )


def _run(detector, cycles, **kwargs):
    event = None
    for _ in range(cycles):
        event = _cycle(detector, **kwargs) or event
    return event


# ── Driving normally ─────────────────────────────────────────────────────────


def test_free_running_is_not_a_collision():
    detector = CollisionDetector()
    assert _run(detector, 50, delivering=1.0, load=0.3) is None
    assert detector.count == 0


def test_a_little_slip_is_not_a_collision():
    """Omni wheels slip by design. Calling that contact would put red patches
    all over open floor."""
    detector = CollisionDetector()
    assert _run(detector, 50, delivering=0.85, load=0.4) is None


def test_stopping_on_purpose_is_not_a_collision():
    """A robot easing to a halt legitimately reports almost no wheel speed. If
    that counted, every turn would leave a red patch behind it."""
    detector = CollisionDetector()
    assert _run(detector, 40, commanded=0.0, delivering=0.0, load=0.1) is None


# ── Hitting something ────────────────────────────────────────────────────────


def test_wheels_not_delivering_is_a_collision():
    """The signal that actually works: told 0.18 m/s, giving almost nothing."""
    detector = CollisionDetector()
    event = _run(detector, 10, delivering=0.05, load=0.9)

    assert event is not None
    assert event.detector == "wheels"


def test_it_reports_which_wheel_and_why():
    """A red patch the robot cannot justify is one nobody can trust."""
    detector = CollisionDetector()
    event = _run(detector, 10, delivering=0.05, load=0.9)

    assert event.wheel_index is not None
    assert "stalled" in event.reason or "delivering" in event.reason


def test_one_bad_read_is_not_a_collision():
    """A dropped frame on a serial bus looks exactly like a stalled wheel for
    one cycle. Three in a row is contact."""
    detector = CollisionDetector()
    assert _cycle(detector, delivering=0.0, load=0.9) is None
    assert detector.count == 0


def test_it_confirms_over_several_cycles_not_instantly():
    config = CollisionConfig(confirm_cycles=3)
    detector = CollisionDetector(config)

    assert _cycle(detector, delivering=0.0, load=0.9) is None
    assert _cycle(detector, delivering=0.0, load=0.9) is None
    assert _cycle(detector, delivering=0.0, load=0.9) is not None


def test_it_detects_quickly_enough_to_matter():
    """At 10 Hz and cruise speed, three cycles is 0.3 s and about 5 cm."""
    config = CollisionConfig(confirm_cycles=3)
    detector = CollisionDetector(config)
    cycles = 0
    while _cycle(detector, delivering=0.0, load=0.9) is None:
        cycles += 1
        assert cycles < 20
    travelled = (cycles + 1) * DT * CRUISE
    assert travelled < 0.10, f"took {travelled:.2f} m to notice"


def test_one_obstacle_is_reported_once():
    """The robot leans on it for many cycles while backing away. Reporting
    every cycle would smear a single touch into a wall across the map."""
    detector = CollisionDetector()
    _run(detector, 60, delivering=0.02, load=0.9)
    assert detector.count == 1


def test_touching_again_after_getting_free_is_a_new_collision():
    detector = CollisionDetector()
    _run(detector, 10, delivering=0.02, load=0.9)
    _run(detector, 10, delivering=1.0, load=0.3)
    _run(detector, 10, delivering=0.02, load=0.9)

    assert detector.count == 2


def test_a_high_load_alone_is_not_a_collision():
    """A robot climbing a rug draws load and still moves. Load only confirms
    what the speed shortfall already showed."""
    detector = CollisionDetector()
    assert _run(detector, 40, delivering=1.0, load=0.95) is None


# ── The slow backstop ────────────────────────────────────────────────────────


def test_wheels_spinning_freely_are_caught_by_lack_of_progress():
    """The case the fast detector cannot see: the wheel delivers its commanded
    speed and normal load while the robot goes nowhere, because it is spinning
    on a smooth floor or jammed against something it can slide along."""
    config = CollisionConfig(stall_window_s=5.0, min_commanded_travel_m=0.5)
    detector = CollisionDetector(config)

    event = None
    for _ in range(200):
        # Perfect wheels, perfect load, and the robot never actually moves.
        event = _cycle(detector, delivering=1.0, load=0.3, x=2.0, y=2.0) or event
        if event:
            break

    assert event is not None
    assert event.detector == "no-progress"


def test_real_progress_is_not_a_stall():
    config = CollisionConfig(stall_window_s=5.0, min_commanded_travel_m=0.5)
    detector = CollisionDetector(config)

    x = 0.0
    for _ in range(200):
        x += CRUISE * DT
        assert _cycle(detector, delivering=1.0, load=0.3, x=x, y=0.0) is None


def test_the_slow_window_needs_enough_commanded_travel_to_beat_the_noise():
    """This is the whole reason the position check cannot be fast. BLE has
    2.71 m of error, so a verdict drawn over less commanded travel than that
    is reading noise. The window must not fire on a robot that was barely
    asked to move."""
    config = CollisionConfig(stall_window_s=1.0, min_commanded_travel_m=3.0)
    detector = CollisionDetector(config)

    # Crawling: 0.01 m/s for 40 s commands only 0.4 m, far inside the error.
    for _ in range(400):
        assert _cycle(
            detector, commanded=0.01, delivering=1.0, load=0.3, x=1.0, y=1.0
        ) is None


def test_the_threshold_is_wider_than_ble_error():
    """Guards the constant itself against being quietly tightened into
    something BLE cannot support."""
    assert CollisionConfig().min_commanded_travel_m > 2.71


# ── Reporting ────────────────────────────────────────────────────────────────


def test_the_summary_splits_the_two_detectors():
    """They fail in different ways, so a red patch on the map should say which
    one put it there."""
    detector = CollisionDetector()
    _run(detector, 10, delivering=0.02, load=0.9)

    summary = detector.summary()
    assert summary["collisions"] == 1
    assert summary["by_wheels"] == 1
    assert summary["by_no_progress"] == 0


def test_reset_clears_everything():
    detector = CollisionDetector()
    _run(detector, 10, delivering=0.02, load=0.9)
    detector.reset()

    assert detector.count == 0
    assert detector.in_contact is False


def test_missing_servo_feedback_is_not_a_collision():
    """A bus read that failed returns nothing. Inventing an obstacle from an
    absent reading would scatter furniture across the map on a flaky cable."""
    detector = CollisionDetector()
    for _ in range(20):
        event = detector.update(
            commanded_speed_mps=CRUISE,
            commanded_wheel_speeds=[CRUISE] * 3,
            measured_wheel_speeds=None,
            wheel_loads=None,
            x_m=0.0, y_m=0.0, heading_deg=0.0, dt_s=DT,
        )
        assert event is None


def test_missing_load_still_detects_a_stalled_wheel():
    """Load is a confirmation, not a requirement — a servo that has given up
    may draw very little. The wheel is stopped either way."""
    detector = CollisionDetector()
    event = None
    for _ in range(10):
        event = detector.update(
            commanded_speed_mps=CRUISE,
            commanded_wheel_speeds=[CRUISE] * 3,
            measured_wheel_speeds=[0.0] * 3,
            wheel_loads=None,
            x_m=0.0, y_m=0.0, heading_deg=0.0, dt_s=DT,
        ) or event

    assert event is not None


@pytest.mark.parametrize("delivering", [0.0, 0.1, 0.2])
def test_various_degrees_of_stall_are_all_caught(delivering):
    detector = CollisionDetector()
    assert _run(detector, 10, delivering=delivering, load=0.9) is not None
