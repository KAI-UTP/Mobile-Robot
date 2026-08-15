"""Holonomic drive controller for Isaac Sim.

Design
------
This module contains **no Isaac Sim imports**. It takes an object satisfying
`ArticulationLike` — three methods — and drives it. That is deliberate:

* the kinematics live in `robotmap_common.holonomic`, which is unit tested;
* the control logic here is tested against a fake articulation;
* the only untested code is the dozen lines in `run_isaac.py` that hand Isaac's
  real articulation to this class.

So a bug in the movement maths shows up on any laptop, not only on a machine
with Isaac Sim installed.

Wheel ordering
--------------
`wheel_joint_names` must be listed in the same order as
`HolonomicGeometry.wheel_angles_deg`. Getting this wrong is the single most
likely mistake when adapting to a different robot, and it is close to
undebuggable by eye — the robot simply drives off at the wrong angle. The
`verify_wheel_order` helper below exists to catch it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from robotmap_common.holonomic import (
    BodyTwist,
    DriveLimits,
    HolonomicGeometry,
    WheelSpeeds,
    forward_kinematics,
    inverse_kinematics,
    scale_to_limit,
)


class ArticulationLike(Protocol):
    """The minimum an Isaac Sim articulation must provide.

    Isaac's own `Articulation` satisfies this; so does the test double.
    """

    def get_joint_positions(self) -> Sequence[float]:
        ...

    def get_joint_velocities(self) -> Sequence[float]:
        ...

    def set_joint_velocity_targets(self, targets: Sequence[float]) -> None:
        ...


@dataclass
class DriveState:
    """What the controller last commanded and last observed."""

    commanded: BodyTwist = field(default_factory=BodyTwist)
    measured: BodyTwist = field(default_factory=BodyTwist)
    wheel_speeds_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wheel_positions_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    was_speed_limited: bool = False


class HolonomicDriveController:
    """Turns body-twist commands into wheel joint velocity targets."""

    def __init__(
        self,
        articulation: ArticulationLike,
        wheel_joint_indices: Sequence[int],
        geometry: HolonomicGeometry | None = None,
        limits: DriveLimits | None = None,
    ) -> None:
        if len(wheel_joint_indices) != 3:
            raise ValueError("a kiwi drive needs exactly three wheel joints")

        self.articulation = articulation
        self.wheel_joint_indices = list(wheel_joint_indices)
        self.geometry = geometry or HolonomicGeometry()
        self.geometry.validate()
        self.limits = limits or DriveLimits()

        self.state = DriveState()

        # Wheel rotation accumulated since reset, used to synthesise encoder
        # counts for the mapping stack.
        self._total_joint_rad = [0.0, 0.0, 0.0]
        self._last_joint_rad: list[float] | None = None

    # ── Commanding ────────────────────────────────────────────────────────

    def drive(self, twist: BodyTwist) -> WheelSpeeds:
        """Apply a body twist. Returns the wheel speeds actually commanded."""
        clamped = self._clamp_twist(twist)
        speeds = inverse_kinematics(clamped, self.geometry)
        speeds, limited = scale_to_limit(speeds, self.limits.max_wheel_mps)

        # Joint targets are angular. A wheel of radius r moving its rim at
        # v m/s turns at v/r rad/s.
        angular = [v / self.geometry.wheel_radius_m for v in speeds.values]

        targets = self._build_target_vector(angular)
        self.articulation.set_joint_velocity_targets(targets)

        self.state.commanded = clamped
        self.state.wheel_speeds_mps = speeds.values
        self.state.was_speed_limited = limited
        return speeds

    def stop(self) -> None:
        self.drive(BodyTwist())

    def _clamp_twist(self, twist: BodyTwist) -> BodyTwist:
        """Limit translation and rotation before the kinematics see them.

        Translation is clamped by magnitude rather than per-axis, so a
        diagonal command keeps its direction — clamping vx and vy separately
        would rotate the requested heading toward 45 degrees.
        """
        speed = math.hypot(twist.vx_mps, twist.vy_mps)
        vx, vy = twist.vx_mps, twist.vy_mps

        if speed > self.limits.max_linear_mps and speed > 0:
            factor = self.limits.max_linear_mps / speed
            vx *= factor
            vy *= factor

        omega = max(
            -self.limits.max_angular_dps,
            min(self.limits.max_angular_dps, twist.omega_dps),
        )
        return BodyTwist(vx_mps=vx, vy_mps=vy, omega_dps=omega)

    def _build_target_vector(self, angular: Sequence[float]) -> list[float]:
        """Place wheel targets at their joint indices, leaving others at zero.

        Isaac expects a target for every DOF in the articulation, not just the
        three being driven, so the vector must be full length.
        """
        dof_count = max(self.wheel_joint_indices) + 1
        try:
            dof_count = max(dof_count, len(self.articulation.get_joint_velocities()))
        except Exception:
            pass

        targets = [0.0] * dof_count
        for slot, joint_index in enumerate(self.wheel_joint_indices):
            targets[joint_index] = angular[slot]
        return targets

    # ── Reading back ──────────────────────────────────────────────────────

    def read_state(self) -> DriveState:
        """Sample the joints and recover what the robot actually did.

        The measured twist will not equal the commanded one: the wheels take
        time to reach their targets, and omni rollers slip. That difference is
        the sim-to-real gap this project sets out to measure, so it is exposed
        rather than hidden.
        """
        velocities = list(self.articulation.get_joint_velocities())
        positions = list(self.articulation.get_joint_positions())

        wheel_angular = [velocities[i] for i in self.wheel_joint_indices]
        wheel_linear = [w * self.geometry.wheel_radius_m for w in wheel_angular]

        self.state.measured = forward_kinematics(
            WheelSpeeds(tuple(wheel_linear)), self.geometry  # type: ignore[arg-type]
        )

        wheel_positions = [positions[i] for i in self.wheel_joint_indices]
        self._accumulate_rotation(wheel_positions)
        self.state.wheel_positions_rad = tuple(self._total_joint_rad)  # type: ignore[assignment]

        return self.state

    def _accumulate_rotation(self, positions: Sequence[float]) -> None:
        """Track total wheel rotation across joint-angle wraparound.

        A continuously spinning revolute joint reports an angle that wraps.
        Differencing raw angles would inject a full revolution of phantom
        travel at every wrap, so each step is unwrapped to the shortest
        equivalent rotation before being accumulated.
        """
        if self._last_joint_rad is None:
            self._last_joint_rad = list(positions)
            return

        for i, current in enumerate(positions):
            delta = current - self._last_joint_rad[i]
            # Shortest way round: a real step is far less than half a turn at
            # any sane control rate.
            while delta > math.pi:
                delta -= 2.0 * math.pi
            while delta < -math.pi:
                delta += 2.0 * math.pi
            self._total_joint_rad[i] += delta

        self._last_joint_rad = list(positions)

    def encoder_ticks(self) -> tuple[int, int, int]:
        """Accumulated wheel rotation as encoder counts.

        Lets the Isaac robot feed the same `SensorPacket` schema the real
        servo bus will, so the mapping stack cannot tell them apart.
        """
        counts_per_rad = self.geometry.ticks_per_revolution / (2.0 * math.pi)
        return tuple(  # type: ignore[return-value]
            int(round(rad * counts_per_rad)) for rad in self._total_joint_rad
        )

    def reset_odometry(self) -> None:
        self._total_joint_rad = [0.0, 0.0, 0.0]
        self._last_joint_rad = None


# ── Wheel-order verification ─────────────────────────────────────────────────


def verify_wheel_order(
    controller: HolonomicDriveController,
    step_fn,
    measure_body_velocity_fn,
    tolerance_deg: float = 25.0,
) -> tuple[bool, str]:
    """Check that the joint list matches the geometry's wheel order.

    Commands a pure forward motion, lets the simulation settle, then asks the
    **physics engine** where the chassis actually went. If the joints are
    listed in the wrong order the robot travels off at the wrong angle —
    typically rotated by 120 or 240 degrees, or mirrored.

    Why the ground truth has to come from outside the controller
    ------------------------------------------------------------
    An earlier version of this function compared the command against
    `controller.read_state().measured`, and could never fail. The controller
    writes wheel *i*'s target to ``wheel_joint_indices[i]`` and reads that same
    slot back, so any permutation cancels out exactly and the readback always
    agrees with the command. The mistake is only visible against a measurement
    that does not share the mapping — which is what the chassis body velocity
    from PhysX provides.

    `step_fn` advances the simulation one physics step.
    `measure_body_velocity_fn` returns the chassis velocity ``(vx, vy)`` in the
    ROBOT body frame, in m/s. In Isaac Sim that is the chassis rigid body's
    linear velocity rotated into body coordinates.
    """
    controller.reset_odometry()
    test_twist = BodyTwist(vx_mps=0.15)

    for _ in range(30):
        controller.drive(test_twist)
        step_fn()

    vx, vy = measure_body_velocity_fn()
    controller.stop()

    speed = math.hypot(vx, vy)
    if speed < 1e-3:
        return False, "robot did not move; check joint indices and drive gains"

    heading_error = math.degrees(math.atan2(vy, vx))
    if abs(heading_error) > tolerance_deg:
        return False, (
            f"commanded forward but the chassis moved {heading_error:+.0f} "
            "degrees off. wheel_joint_names order probably does not match "
            "wheel_angles_deg"
        )

    return True, f"wheel order OK (heading error {heading_error:+.1f} degrees)"
