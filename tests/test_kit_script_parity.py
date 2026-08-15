"""The Omniverse Kit script must agree with the tested kinematics.

Why this exists
---------------
`omniverse/kit_holonomic.py` runs inside Omniverse's own Python, which cannot
see this project's packages. Its kiwi-drive maths is therefore an inlined copy
of `robotmap_common.holonomic`, and inlined copies drift: someone fixes a sign
error in one and not the other, and the simulation quietly stops matching the
robot.

Nothing else catches that. The Kit script cannot be imported normally — it
calls `run_demo()` at module scope and imports `omni.*` — so it is loaded here
with those parts stripped, and its functions are compared against the tested
originals over a wide sweep of inputs.

If this fails, the two implementations have diverged. Fix the copy in
kit_holonomic.py; `holonomic.py` is the source of truth.
"""

from __future__ import annotations

import math
import types
from pathlib import Path

import pytest
from robotmap_common.holonomic import (
    BodyTwist,
    HolonomicGeometry,
    integrate_twist,
    inverse_kinematics,
)

KIT_SCRIPT = Path(__file__).resolve().parents[1] / "omniverse" / "kit_holonomic.py"

# The Kit script's own constants, which the comparison geometry must match.
KIT_WHEEL_RADIUS_M = 0.029
KIT_WHEEL_OFFSET_M = 0.100
KIT_WHEEL_ANGLES_DEG = (0.0, 120.0, 240.0)

GEOM = HolonomicGeometry(
    wheel_radius_m=KIT_WHEEL_RADIUS_M,
    wheel_offset_m=KIT_WHEEL_OFFSET_M,
    wheel_angles_deg=KIT_WHEEL_ANGLES_DEG,
)


def _load_kit_maths() -> types.ModuleType:
    """Load only the maths from the Kit script.

    The file imports omni.* and runs a demo at import time, so it is read as
    text and the pure functions are executed in isolation. Crude, but it means
    the actual shipped file is checked rather than a duplicate of it.
    """
    source = KIT_SCRIPT.read_text(encoding="utf-8")

    module = types.ModuleType("kit_maths")
    module.__dict__["math"] = math
    module.__dict__["WHEEL_RADIUS_M"] = KIT_WHEEL_RADIUS_M
    module.__dict__["WHEEL_OFFSET_M"] = KIT_WHEEL_OFFSET_M
    module.__dict__["WHEEL_ANGLES_DEG"] = KIT_WHEEL_ANGLES_DEG

    # Pull out just the two function definitions by name.
    wanted = ("def inverse_kinematics(", "def integrate_twist(")
    lines = source.splitlines()
    collected: list[str] = []

    for start_marker in wanted:
        try:
            start = next(i for i, line in enumerate(lines) if line.startswith(start_marker))
        except StopIteration:
            raise AssertionError(
                f"{KIT_SCRIPT.name} no longer defines {start_marker!r}. "
                "If it was renamed, update this test; the two implementations "
                "must stay comparable."
            ) from None

        end = start + 1
        while end < len(lines) and (
            lines[end].startswith((" ", "\t")) or not lines[end].strip()
        ):
            end += 1
        collected.extend(lines[start:end])
        collected.append("")

    exec("\n".join(collected), module.__dict__)  # noqa: S102
    return module


KIT = _load_kit_maths()


# ── The script still exposes what is expected ────────────────────────────────


def test_kit_script_exists():
    assert KIT_SCRIPT.exists(), f"{KIT_SCRIPT} is missing"


def test_kit_constants_match_the_geometry_used_here():
    """If the Kit script's dimensions change, this test's premise breaks."""
    source = KIT_SCRIPT.read_text(encoding="utf-8")
    assert f"WHEEL_RADIUS_M = {KIT_WHEEL_RADIUS_M}" in source
    assert f"WHEEL_OFFSET_M = {KIT_WHEEL_OFFSET_M}" in source


# ── Inverse kinematics parity ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "vx,vy,omega",
    [
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.0, 0.2, 0.0),
        (0.0, 0.0, 45.0),
        (0.15, -0.1, 30.0),
        (-0.2, 0.05, -60.0),
        (0.3, 0.3, 90.0),
        (-0.05, -0.25, -15.0),
    ],
)
def test_inverse_kinematics_matches(vx, vy, omega):
    """Every wheel speed must agree to full float precision."""
    expected = inverse_kinematics(
        BodyTwist(vx_mps=vx, vy_mps=vy, omega_dps=omega), GEOM
    ).values
    actual = KIT.inverse_kinematics(vx, vy, omega)

    assert len(actual) == 3
    for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
        assert got == pytest.approx(want, abs=1e-12), (
            f"wheel {index} differs: kit_holonomic.py gives {got}, "
            f"holonomic.py gives {want}"
        )


def test_inverse_kinematics_matches_across_a_sweep():
    """A parameter sweep, in case the sign error hides in an untried octant."""
    for vx in (-0.3, -0.1, 0.0, 0.1, 0.3):
        for vy in (-0.3, 0.0, 0.3):
            for omega in (-90.0, -20.0, 0.0, 20.0, 90.0):
                expected = inverse_kinematics(
                    BodyTwist(vx_mps=vx, vy_mps=vy, omega_dps=omega), GEOM
                ).values
                actual = KIT.inverse_kinematics(vx, vy, omega)
                for want, got in zip(expected, actual, strict=True):
                    assert got == pytest.approx(want, abs=1e-12)


# ── Odometry integration parity ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "vx,vy,omega,heading,dt",
    [
        (0.2, 0.0, 0.0, 0.0, 0.1),         # straight, no rotation
        (0.2, 0.0, 0.0, 90.0, 0.1),        # straight, rotated frame
        (0.0, 0.2, 0.0, 45.0, 0.1),        # strafe
        (0.5, 0.0, 90.0, 0.0, 1.0),        # the arc case
        (0.3, 0.2, -45.0, 210.0, 0.25),    # everything at once
        (0.0, 0.0, 60.0, 0.0, 0.5),        # rotation only
        (0.1, 0.0, 1e-12, 33.0, 0.1),      # near-zero omega, straight-line branch
    ],
)
def test_integrate_twist_matches(vx, vy, omega, heading, dt):
    """Including the arc branch, which is where a copy is most likely to rot."""
    expected = integrate_twist(
        BodyTwist(vx_mps=vx, vy_mps=vy, omega_dps=omega), heading, dt
    )
    dx, dy, dheading = KIT.integrate_twist(vx, vy, omega, heading, dt)

    assert dx == pytest.approx(expected.delta_x_m, abs=1e-12)
    assert dy == pytest.approx(expected.delta_y_m, abs=1e-12)
    assert dheading == pytest.approx(expected.delta_heading_deg, abs=1e-12)


def test_both_take_the_same_branch_at_the_omega_threshold():
    """The straight-line and arc branches must switch over at the same point.

    A mismatched threshold produces a small discontinuity that only shows up
    as a map that does not quite close, which is painful to trace back here.
    """
    for omega in (0.0, 1e-10, 1e-9, 1e-8, 1e-6):
        expected = integrate_twist(BodyTwist(vx_mps=0.2, omega_dps=omega), 0.0, 0.1)
        dx, dy, _ = KIT.integrate_twist(0.2, 0.0, omega, 0.0, 0.1)
        assert dx == pytest.approx(expected.delta_x_m, abs=1e-12)
        assert dy == pytest.approx(expected.delta_y_m, abs=1e-12)


def test_kit_maths_closes_a_strafing_square():
    """The demo the Kit script actually performs must come out right.

    Drives a square by strafing without ever rotating; it must return to the
    origin. This is the behaviour shown on screen, so it is worth asserting
    against the shipped file rather than only against the library.
    """
    x = y = heading = 0.0
    for direction in (0.0, 90.0, 180.0, 270.0):
        angle = math.radians(direction)
        vx, vy = math.cos(angle), math.sin(angle)
        for _ in range(20):
            dx, dy, dheading = KIT.integrate_twist(vx, vy, 0.0, heading, 0.1)
            x += dx
            y += dy
            heading = (heading + dheading) % 360.0

    assert math.hypot(x, y) < 1e-9
    assert heading == pytest.approx(0.0)
