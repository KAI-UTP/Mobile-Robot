"""Tests for the Isaac Sim drive controller, using a fake articulation.

Isaac Sim cannot be installed on every machine, so the controller was written
to depend only on three articulation methods. This file substitutes a simple
physics-free stand-in for them, which means the control logic — clamping,
scaling, joint indexing, encoder accumulation — is verified everywhere, and the
only unverified code left is the handful of lines that fetch Isaac's real
articulation object.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "omniverse"))

from drive_controller import (  # noqa: E402
    HolonomicDriveController,
    verify_wheel_order,
)
from robotmap_common.holonomic import (  # noqa: E402
    BodyTwist,
    DriveLimits,
    HolonomicGeometry,
    WheelSpeeds,
    forward_kinematics,
)

GEOM = HolonomicGeometry(
    wheel_radius_m=0.029,
    wheel_offset_m=0.100,
    wheel_angles_deg=(0.0, 120.0, 240.0),
    ticks_per_revolution=4096,
)


class FakeArticulation:
    """A perfect, instantaneous articulation.

    Joints reach their velocity target immediately and integrate position at
    that rate. Real physics does neither, but this isolates the controller's
    own logic from PhysX behaviour — which is the point of the test.

    `extra_dofs` pads the joint vector so the tests exercise the case where the
    wheels are not the only joints, which is true of any real robot.
    """

    def __init__(self, wheel_indices=(0, 1, 2), extra_dofs: int = 0, dt: float = 0.01):
        self.dof_count = max(wheel_indices) + 1 + extra_dofs
        self.velocities = [0.0] * self.dof_count
        self.positions = [0.0] * self.dof_count
        self.dt = dt
        self.last_targets = None
        # The physically true wheel-to-joint mapping. The controller may be
        # given a different one; the difference is exactly what the wheel-order
        # check has to detect, so body velocity is always computed from this.
        self.true_wheel_indices = tuple(wheel_indices)

    def body_velocity(self):
        """Chassis velocity in the body frame, as PhysX would report it.

        Independent of whatever mapping the controller happens to be using —
        that independence is the whole point.
        """
        wheel_linear = [
            self.velocities[i] * GEOM.wheel_radius_m for i in self.true_wheel_indices
        ]
        twist = forward_kinematics(WheelSpeeds(tuple(wheel_linear)), GEOM)
        return twist.vx_mps, twist.vy_mps

    def get_joint_positions(self):
        return list(self.positions)

    def get_joint_velocities(self):
        return list(self.velocities)

    def set_joint_velocity_targets(self, targets):
        self.last_targets = list(targets)
        for i, target in enumerate(targets):
            if i < self.dof_count:
                self.velocities[i] = target

    def step(self):
        """Advance, wrapping joint angles the way a revolute joint does."""
        for i in range(self.dof_count):
            self.positions[i] += self.velocities[i] * self.dt
            while self.positions[i] > math.pi:
                self.positions[i] -= 2.0 * math.pi
            while self.positions[i] < -math.pi:
                self.positions[i] += 2.0 * math.pi


def _controller(articulation=None, indices=(0, 1, 2), limits=None):
    art = articulation or FakeArticulation(indices)
    return HolonomicDriveController(art, indices, GEOM, limits), art


# ── Construction ─────────────────────────────────────────────────────────────


def test_requires_exactly_three_wheels():
    art = FakeArticulation()
    with pytest.raises(ValueError, match="three wheel joints"):
        HolonomicDriveController(art, [0, 1], GEOM)


def test_rejects_singular_geometry():
    art = FakeArticulation()
    bad = HolonomicGeometry(wheel_angles_deg=(0.0, 3.0, 240.0))
    with pytest.raises(ValueError):
        HolonomicDriveController(art, [0, 1, 2], bad)


# ── Commanding ───────────────────────────────────────────────────────────────


def test_stop_zeroes_every_wheel():
    ctrl, art = _controller()
    ctrl.drive(BodyTwist(vx_mps=0.2))
    ctrl.stop()
    assert art.last_targets[:3] == pytest.approx([0.0, 0.0, 0.0])


def test_joint_targets_are_angular_not_linear():
    """A common and silent error: sending rim speed in m/s to a joint that
    expects rad/s. At a 29 mm wheel radius that is a 34x error."""
    ctrl, art = _controller()
    speeds = ctrl.drive(BodyTwist(omega_dps=90.0))

    expected_angular = speeds.values[0] / GEOM.wheel_radius_m
    assert art.last_targets[0] == pytest.approx(expected_angular)
    # And confirm it is not simply the linear speed.
    assert abs(art.last_targets[0] - speeds.values[0]) > 1e-3


def test_targets_are_written_to_the_right_joint_indices():
    """The wheels are rarely joints 0-2 on a real robot."""
    art = FakeArticulation(wheel_indices=(2, 5, 7))
    ctrl = HolonomicDriveController(art, [2, 5, 7], GEOM)
    ctrl.drive(BodyTwist(omega_dps=45.0))

    assert len(art.last_targets) >= 8
    for driven in (2, 5, 7):
        assert art.last_targets[driven] != 0.0
    for untouched in (0, 1, 3, 4, 6):
        assert art.last_targets[untouched] == 0.0


def test_target_vector_covers_all_dofs():
    art = FakeArticulation(wheel_indices=(0, 1, 2), extra_dofs=4)
    ctrl = HolonomicDriveController(art, [0, 1, 2], GEOM)
    ctrl.drive(BodyTwist(vx_mps=0.1))
    assert len(art.last_targets) == art.dof_count


# ── Limits ───────────────────────────────────────────────────────────────────


def test_linear_speed_is_clamped():
    limits = DriveLimits(max_linear_mps=0.2, max_angular_dps=90.0, max_wheel_mps=10.0)
    ctrl, _ = _controller(limits=limits)
    ctrl.drive(BodyTwist(vx_mps=5.0))
    assert ctrl.state.commanded.vx_mps == pytest.approx(0.2)


def test_diagonal_clamping_preserves_heading():
    """Clamping vx and vy independently would rotate the requested direction
    toward 45 degrees. Magnitude clamping must not."""
    limits = DriveLimits(max_linear_mps=0.2, max_angular_dps=90.0, max_wheel_mps=10.0)
    ctrl, _ = _controller(limits=limits)
    ctrl.drive(BodyTwist(vx_mps=3.0, vy_mps=1.0))

    commanded = ctrl.state.commanded
    original = math.atan2(1.0, 3.0)
    clamped = math.atan2(commanded.vy_mps, commanded.vx_mps)
    assert clamped == pytest.approx(original, abs=1e-9)
    assert math.hypot(commanded.vx_mps, commanded.vy_mps) == pytest.approx(0.2)


def test_angular_speed_is_clamped_both_directions():
    limits = DriveLimits(max_linear_mps=1.0, max_angular_dps=45.0, max_wheel_mps=10.0)
    ctrl, _ = _controller(limits=limits)

    ctrl.drive(BodyTwist(omega_dps=500.0))
    assert ctrl.state.commanded.omega_dps == pytest.approx(45.0)

    ctrl.drive(BodyTwist(omega_dps=-500.0))
    assert ctrl.state.commanded.omega_dps == pytest.approx(-45.0)


def test_wheel_limit_flag_is_reported():
    limits = DriveLimits(max_linear_mps=10.0, max_angular_dps=1000.0, max_wheel_mps=0.05)
    ctrl, _ = _controller(limits=limits)
    ctrl.drive(BodyTwist(vx_mps=1.0))
    assert ctrl.state.was_speed_limited is True
    assert max(abs(v) for v in ctrl.state.wheel_speeds_mps) == pytest.approx(0.05)


# ── Reading back ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "twist",
    [
        BodyTwist(vx_mps=0.2),
        BodyTwist(vy_mps=0.15),
        BodyTwist(omega_dps=60.0),
        BodyTwist(vx_mps=0.1, vy_mps=-0.05, omega_dps=20.0),
    ],
)
def test_measured_twist_matches_command_on_ideal_joints(twist):
    """With instantaneous joints and no slip, what comes back must equal what
    went in. Any mismatch here is a bug in the controller, not in physics."""
    ctrl, art = _controller()
    ctrl.drive(twist)
    art.step()

    measured = ctrl.read_state().measured
    assert measured.vx_mps == pytest.approx(twist.vx_mps, abs=1e-9)
    assert measured.vy_mps == pytest.approx(twist.vy_mps, abs=1e-9)
    assert measured.omega_dps == pytest.approx(twist.omega_dps, abs=1e-7)


# ── Encoder accumulation ─────────────────────────────────────────────────────


def test_encoder_ticks_start_at_zero():
    ctrl, _ = _controller()
    assert ctrl.encoder_ticks() == (0, 0, 0)


def test_encoder_ticks_accumulate_with_rotation():
    ctrl, art = _controller()
    ctrl.drive(BodyTwist(omega_dps=90.0))
    for _ in range(50):
        art.step()
        ctrl.read_state()

    ticks = ctrl.encoder_ticks()
    assert all(t > 0 for t in ticks), "spinning should advance every wheel"


def test_encoder_accumulation_survives_joint_wraparound():
    """A continuously spinning revolute joint wraps its reported angle. Naive
    differencing would inject a whole revolution of phantom travel each time.

    Spin fast enough and long enough to wrap many times, then check the total
    against the analytically expected rotation.
    """
    art = FakeArticulation(dt=0.01)
    ctrl = HolonomicDriveController(art, [0, 1, 2], GEOM)

    ctrl.drive(BodyTwist(omega_dps=180.0))
    commanded_angular = art.last_targets[0]  # rad/s

    steps = 400
    for _ in range(steps):
        art.step()
        ctrl.read_state()

    expected_rad = commanded_angular * art.dt * steps
    # Confirm the joint really did wrap, so the test is meaningful.
    assert abs(expected_rad) > 4 * math.pi

    counts_per_rad = GEOM.ticks_per_revolution / (2 * math.pi)
    expected_ticks = expected_rad * counts_per_rad
    assert ctrl.encoder_ticks()[0] == pytest.approx(expected_ticks, rel=0.02)


def test_reset_odometry_clears_ticks():
    ctrl, art = _controller()
    ctrl.drive(BodyTwist(omega_dps=90.0))
    for _ in range(20):
        art.step()
        ctrl.read_state()
    assert ctrl.encoder_ticks() != (0, 0, 0)

    ctrl.reset_odometry()
    assert ctrl.encoder_ticks() == (0, 0, 0)


def test_encoder_ticks_feed_odometry_consistently():
    """Ticks produced here must be interpretable by the same odometry function
    the real servo bus will use."""
    from robotmap_common.holonomic import odometry_from_ticks

    ctrl, art = _controller()
    ctrl.drive(BodyTwist(vx_mps=0.2))

    steps = 100
    for _ in range(steps):
        art.step()
        ctrl.read_state()

    ticks = ctrl.encoder_ticks()
    elapsed = art.dt * steps
    delta = odometry_from_ticks(ticks, heading_deg=0.0, dt_s=elapsed, geometry=GEOM)

    # Driving forward for `elapsed` seconds at 0.2 m/s.
    assert delta.delta_x_m == pytest.approx(0.2 * elapsed, rel=0.02)
    assert delta.delta_y_m == pytest.approx(0.0, abs=1e-3)


# ── Wheel-order verification ─────────────────────────────────────────────────


def test_wheel_order_check_passes_when_correct():
    art = FakeArticulation()
    ctrl = HolonomicDriveController(art, [0, 1, 2], GEOM)
    ok, message = verify_wheel_order(ctrl, art.step, art.body_velocity)
    assert ok, message


@pytest.mark.parametrize(
    "wrong_order",
    [
        [0, 2, 1],  # last two swapped
        [1, 0, 2],  # first two swapped
        [1, 2, 0],  # rotated
        [2, 0, 1],  # rotated the other way
    ],
)
def test_wheel_order_check_catches_permuted_joints(wrong_order):
    """Every wrong permutation must be caught.

    This failure is nearly invisible by eye: all three wheels turn, the robot
    moves smoothly, and it simply travels the wrong way.
    """
    art = FakeArticulation()
    ctrl = HolonomicDriveController(art, wrong_order, GEOM)
    ok, message = verify_wheel_order(ctrl, art.step, art.body_velocity)
    assert not ok, f"permutation {wrong_order} went undetected"
    assert "degrees off" in message


def test_wheel_order_check_cannot_rely_on_controller_readback():
    """Guards the design flaw itself.

    The controller's own readback shares the joint mapping with its writes, so
    a permutation cancels out and the readback always agrees with the command.
    Any future refactor that goes back to comparing against `read_state()`
    would silently make the check useless; this test pins why.
    """
    art = FakeArticulation()
    ctrl = HolonomicDriveController(art, [0, 2, 1], GEOM)  # deliberately wrong

    ctrl.drive(BodyTwist(vx_mps=0.15))
    art.step()

    # Readback agrees with the command despite the wrong mapping...
    measured = ctrl.read_state().measured
    assert measured.vx_mps == pytest.approx(0.15, abs=1e-9)
    assert measured.vy_mps == pytest.approx(0.0, abs=1e-9)

    # ...while the chassis is demonstrably moving somewhere else.
    vx, vy = art.body_velocity()
    assert abs(math.degrees(math.atan2(vy, vx))) > 25.0


def test_wheel_order_check_reports_a_dead_robot():
    class DeadArticulation(FakeArticulation):
        def set_joint_velocity_targets(self, targets):
            self.last_targets = list(targets)  # accepted, but nothing moves

    art = DeadArticulation()
    ctrl = HolonomicDriveController(art, [0, 1, 2], GEOM)
    ok, message = verify_wheel_order(ctrl, art.step, art.body_velocity)
    assert not ok
    assert "did not move" in message
