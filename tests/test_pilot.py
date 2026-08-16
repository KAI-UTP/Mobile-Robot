"""The autonomous scan on real hardware.

These tests are mostly about the robot NOT moving. Every other part of this
project can be wrong on screen and corrected later; this one drives 12 V
servos across a room, so the failure modes are physical and the tests that
matter are the ones about stopping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "pilot"))

from pilot import Pilot, PilotConfig, ScanPhase  # noqa: E402
from robotmap_common.models import (  # noqa: E402
    EncoderData,
    PoseEstimate,
    RangeReading,
    SensorPacket,
)

DT = 0.1


def _packet(front=3.0, right=0.35, bumper=False) -> SensorPacket:
    return SensorPacket(
        robot_id="MR3W01",
        timestamp="2026-08-16T00:00:00Z",
        sequence=1,
        encoders=EncoderData(left_ticks=0, right_ticks=0, dt_ms=100),
        ranges=[
            RangeReading(angle_deg=0.0, distance_m=front),
            RangeReading(angle_deg=-90.0, distance_m=right),
            RangeReading(angle_deg=90.0, distance_m=3.0),
        ],
        bumper_active=bumper,
    )


def _pose(x=1.0, y=1.0, heading=0.0) -> PoseEstimate:
    return PoseEstimate(
        robot_id="MR3W01",
        timestamp="2026-08-16T00:00:00Z",
        sequence=1,
        x_m=x, y_m=y, heading_deg=heading,
        std_x_m=0.02, std_y_m=0.02,
    )


def _stationary(twist) -> bool:
    return twist.is_stationary(1e-9)


# ── Not moving when it must not ──────────────────────────────────────────────


def test_it_does_not_move_before_any_sensor_data_arrives():
    """Otherwise the robot drives off the instant it is switched on, using a
    world model it does not have yet."""
    pilot = Pilot()
    assert _stationary(pilot.step(None, None, DT))


def test_it_does_not_move_with_a_pose_but_no_ranges():
    pilot = Pilot()
    assert _stationary(pilot.step(None, _pose(), DT))


def test_stale_sensor_data_stops_the_robot():
    """A frozen world model while the robot keeps moving through the real
    world is exactly when it hits something."""
    pilot = Pilot(PilotConfig(stale_after_s=0.5))
    moving = pilot.step(_packet(), _pose(), DT, age_s=0.1)
    assert not _stationary(moving)

    assert _stationary(pilot.step(_packet(), _pose(), DT, age_s=2.0))


def test_it_recovers_when_sensor_data_comes_back():
    """A dropped packet must not end the scan — telemetry over Bluetooth drops
    packets routinely."""
    pilot = Pilot(PilotConfig(stale_after_s=0.5))
    pilot.step(_packet(), _pose(), DT, age_s=0.1)
    pilot.step(_packet(), _pose(), DT, age_s=2.0)

    assert not _stationary(pilot.step(_packet(), _pose(), DT, age_s=0.1))


def test_a_single_bump_backs_off_rather_than_abandoning_the_scan():
    """Halting on the first contact was the original behaviour and it made the
    robot useless: run against a furnished room it clipped a cabinet standing
    against a wall 43 s in and gave up having measured nothing. Furniture
    against a wall is the normal case."""
    pilot = Pilot()
    pilot.step(_packet(), _pose(), DT)

    twist = pilot.step(_packet(bumper=True), _pose(), DT)
    assert pilot.status.phase == ScanPhase.PERIMETER
    assert twist.vx_mps < 0, "must reverse away from what it hit"
    assert pilot.status.bumps == 1


def test_backing_off_covers_a_real_distance():
    """A single reversed control cycle moves under two centimetres, which does
    not clear anything."""
    pilot = Pilot()
    pilot.step(_packet(), _pose(), DT)
    pilot.step(_packet(bumper=True), _pose(), DT)

    reversing = 0
    for _ in range(40):
        twist = pilot.step(_packet(), _pose(), DT)
        if twist.vx_mps < 0:
            reversing += 1
        else:
            break
    assert reversing >= 3


def test_it_returns_to_following_the_wall_after_backing_off():
    pilot = Pilot()
    pilot.step(_packet(), _pose(), DT)
    pilot.step(_packet(bumper=True), _pose(), DT)

    for _ in range(60):
        twist = pilot.step(_packet(), _pose(), DT)
        if twist.vx_mps > 0:
            break
    assert pilot.status.phase == ScanPhase.PERIMETER
    assert twist.vx_mps > 0


def test_a_contact_that_clears_is_counted_once():
    """The bumper stays closed for the whole time the robot is against the
    object. Counting per packet would burn the entire budget in under a second
    on a single graze."""
    pilot = Pilot()
    pilot.step(_packet(), _pose(), DT)

    # Held for the first few cycles, then clear as the robot reverses off it.
    for _ in range(3):
        pilot.step(_packet(bumper=True), _pose(), DT)
    for _ in range(40):
        pilot.step(_packet(), _pose(), DT)

    assert pilot.status.bumps == 1


def test_a_bumper_that_never_clears_counts_again_and_eventually_halts():
    """Backing off is only a recovery if it recovers. A bumper still shut after
    a completed reverse is a robot that has not got free, and re-counting is
    what turns that into the halt — while still being far cheaper than
    counting every packet."""
    pilot = Pilot()
    pilot.step(_packet(), _pose(), DT)

    packets = 0
    while pilot.status.phase == ScanPhase.PERIMETER and packets < 400:
        pilot.step(_packet(bumper=True), _pose(), DT)
        packets += 1

    assert pilot.status.phase == ScanPhase.STOPPED
    assert pilot.status.bumps <= packets // 4, "counting far too eagerly"


def test_repeated_contacts_do_halt_the_scan():
    """Backing off does not fix a robot wedged somewhere and shoving."""
    pilot = Pilot(PilotConfig(max_bumps=3, bump_backoff_m=0.01))
    pilot.step(_packet(), _pose(), DT)

    for _ in range(400):
        pilot.step(_packet(bumper=True), _pose(), DT)
        pilot.step(_packet(), _pose(), DT)
        if pilot.status.phase == ScanPhase.STOPPED:
            break

    assert pilot.status.phase == ScanPhase.STOPPED
    assert "stuck" in pilot.status.stopped_reason


def test_a_halted_scan_stays_halted():
    """It must not resume by itself once it has decided something is wrong."""
    pilot = Pilot(PilotConfig(max_bumps=0))
    pilot.step(_packet(), _pose(), DT)
    pilot.step(_packet(bumper=True), _pose(), DT)
    assert pilot.status.phase == ScanPhase.STOPPED

    for _ in range(20):
        assert _stationary(pilot.step(_packet(), _pose(), DT))
    assert pilot.status.phase == ScanPhase.STOPPED


def test_a_perimeter_that_never_closes_gives_up():
    """Otherwise a robot that has lost the wall drives until its battery is
    flat."""
    pilot = Pilot(PilotConfig(perimeter_timeout_s=0.0))
    pilot.step(_packet(), _pose(), DT)

    assert _stationary(pilot.step(_packet(), _pose(), DT))
    assert pilot.status.phase == ScanPhase.STOPPED
    assert "timed out" in pilot.status.stopped_reason


# ── Speed limits ─────────────────────────────────────────────────────────────


def test_speed_is_limited():
    pilot = Pilot(PilotConfig(max_linear_mps=0.05))
    twist = pilot.step(_packet(), _pose(), DT)
    assert abs(twist.vx_mps) <= 0.05 + 1e-9


def test_limiting_a_diagonal_keeps_its_direction():
    """Clamping vx and vy separately turns a fast diagonal into a different
    heading — on a holonomic base the robot then quietly drives somewhere
    other than where it was told."""
    from robotmap_common.holonomic import BodyTwist

    pilot = Pilot(PilotConfig(max_linear_mps=0.1))
    limited = pilot._limit(BodyTwist(vx_mps=3.0, vy_mps=4.0))

    assert limited.vx_mps == pytest.approx(0.06, abs=1e-6)
    assert limited.vy_mps == pytest.approx(0.08, abs=1e-6)
    # Same direction, allowed magnitude.
    assert (limited.vx_mps / limited.vy_mps) == pytest.approx(3.0 / 4.0)


def test_rotation_is_limited_too():
    from robotmap_common.holonomic import BodyTwist

    pilot = Pilot(PilotConfig(max_angular_dps=30.0))
    assert pilot._limit(BodyTwist(omega_dps=500.0)).omega_dps == 30.0
    assert pilot._limit(BodyTwist(omega_dps=-500.0)).omega_dps == -30.0


# ── Getting through the phases ───────────────────────────────────────────────


def test_it_starts_the_perimeter_lap_on_the_first_good_packet():
    pilot = Pilot()
    twist = pilot.step(_packet(), _pose(), DT)
    assert pilot.status.phase == ScanPhase.PERIMETER
    assert not _stationary(twist)


def test_the_perimeter_lap_does_not_strafe_when_it_has_room():
    """Hugging a wall means holding a distance to one side while driving along
    it, which is a differential motion. Lateral velocity here would be the
    controller being misused — it is reserved for escaping a squeeze."""
    pilot = Pilot()
    for _ in range(50):
        twist = pilot.step(_packet(), _pose(), DT)
        assert twist.vy_mps == 0.0


# ── Lateral clearance ────────────────────────────────────────────────────────


def _squeezed(left: float, right: float) -> SensorPacket:
    packet = _packet()
    packet.ranges = [
        RangeReading(angle_deg=0.0, distance_m=4.0),
        RangeReading(angle_deg=90.0, distance_m=left),
        RangeReading(angle_deg=45.0, distance_m=left),
        RangeReading(angle_deg=-90.0, distance_m=right),
        RangeReading(angle_deg=-45.0, distance_m=right),
    ]
    return packet


def test_it_sidesteps_away_from_something_crowding_its_flank():
    """Wall-following regulates ONE wall and watches ahead. Nothing watches the
    other side, so the robot drove into a 0.40 m gap between a cabinet and the
    wall reporting 4 m of clear space ahead the whole way in."""
    pilot = Pilot()
    pilot.step(_packet(), _pose(), DT)

    twist = pilot.step(_squeezed(left=0.10, right=3.0), _pose(), DT)
    assert twist.vy_mps < 0, "must move away from the near left"


def test_it_sidesteps_the_other_way_too():
    pilot = Pilot()
    pilot.step(_packet(), _pose(), DT)

    twist = pilot.step(_squeezed(left=3.0, right=0.10), _pose(), DT)
    assert twist.vy_mps > 0, "must move away from the near right"


def test_the_squeeze_escape_holds_its_heading():
    """The whole reason to strafe rather than turn: the wall-following loop is
    left undisturbed and still sees its wall at the angle it expects. A
    differential robot would have to turn away and re-acquire.

    Tested on the escape layer alone. Comparing two whole control cycles would
    not isolate it — the follower's own steering legitimately differs between
    two different sensor readings.
    """
    from robotmap_common.holonomic import BodyTwist

    pilot = Pilot()
    commanded = BodyTwist(vx_mps=0.18, omega_dps=17.0)
    escaped = pilot._keep_clear(_squeezed(left=0.10, right=3.0).ranges, commanded)

    assert escaped.omega_dps == commanded.omega_dps
    assert escaped.vy_mps < 0


def test_it_slows_down_when_squeezed():
    """A squeeze taken at cruise speed is a scrape along whatever caused it."""
    pilot = Pilot()
    pilot.step(_packet(), _pose(), DT)

    clear = pilot.step(_packet(), _pose(), DT)
    tight = pilot.step(_squeezed(left=0.10, right=3.0), _pose(), DT)
    assert 0 < tight.vx_mps < clear.vx_mps


def test_the_wall_being_followed_does_not_trigger_an_escape():
    """The follower deliberately holds 0.35 m against its wall. If that read as
    a squeeze the robot would sidestep away from the very wall it is measuring
    and lose the outline."""
    pilot = Pilot()
    pilot.step(_packet(), _pose(), DT)

    for _ in range(30):
        twist = pilot.step(_packet(right=0.35), _pose(), DT)
        assert twist.vy_mps == 0.0


def test_the_escape_is_stronger_the_closer_the_obstacle():
    pilot = Pilot()
    pilot.step(_packet(), _pose(), DT)

    near = pilot.step(_squeezed(left=0.05, right=3.0), _pose(), DT)
    far = pilot.step(_squeezed(left=0.19, right=3.0), _pose(), DT)
    assert abs(near.vy_mps) > abs(far.vy_mps)


def test_something_dead_ahead_is_not_treated_as_a_flank():
    """That is the follower's job, and reacting to it here would fight it."""
    pilot = Pilot()
    pilot.step(_packet(), _pose(), DT)

    twist = pilot.step(_packet(front=0.10), _pose(), DT)
    assert twist.vy_mps == 0.0


def test_the_sweep_can_be_skipped():
    """Some customers only want the outline, and the sweep doubles the time."""
    pilot = Pilot(PilotConfig(sweep=False))
    pilot.step(_packet(), _pose(), DT)
    pilot.follower.loop_closed = True

    pilot.step(_packet(), _pose(), DT)
    assert pilot.status.phase == ScanPhase.DONE


def test_the_sweep_is_handed_the_measured_outline():
    """Without it the sweep discovers each row's end by driving into the wall,
    which on real hardware is a collision per row."""
    pilot = Pilot(PilotConfig())
    pilot.set_bounds((0.0, 0.0, 6.0, 4.5))
    pilot.step(_packet(), _pose(), DT)
    pilot.follower.loop_closed = True
    pilot.step(_packet(), _pose(), DT)

    assert pilot.status.phase == ScanPhase.SWEEPING
    assert pilot.planner is not None
    assert pilot.planner.bounds == (0.0, 0.0, 6.0, 4.5)


def test_a_room_measured_late_still_reaches_the_sweep():
    """The mapper publishes the outline when it has one, which may be after
    the sweep has already begun."""
    pilot = Pilot()
    pilot.step(_packet(), _pose(), DT)
    pilot.follower.loop_closed = True
    pilot.step(_packet(), _pose(), DT)
    assert pilot.planner.bounds is None

    pilot.set_bounds((0.0, 0.0, 6.0, 4.5))
    assert pilot.planner.bounds == (0.0, 0.0, 6.0, 4.5)


def test_a_finished_scan_commands_nothing():
    pilot = Pilot(PilotConfig(sweep=False))
    pilot.step(_packet(), _pose(), DT)
    pilot.follower.loop_closed = True
    pilot.step(_packet(), _pose(), DT)

    for _ in range(10):
        assert _stationary(pilot.step(_packet(), _pose(), DT))


def test_the_status_says_what_it_is_doing():
    """An autonomous robot that cannot explain itself is one nobody will let
    out of the lab."""
    pilot = Pilot()
    pilot.step(_packet(), _pose(), DT)

    status = pilot.status.as_dict()
    assert status["phase"] == "PERIMETER"
    assert status["note"]
    assert status["packets"] == 1
