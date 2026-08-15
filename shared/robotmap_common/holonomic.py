"""Kiwi-drive kinematics: three omni wheels at 120 degrees.

What a kiwi drive is
--------------------
Three omni wheels, all driven, spaced evenly around the chassis. Each wheel has
free-spinning rollers around its rim, so it drives along its own axis and slides
freely sideways. Combining three of them lets the robot translate in **any**
direction while independently rotating — it never has to turn before it can
move sideways.

That is the property that makes this platform different from the differential
drive in `geometry.py`, and it is why that module's `differential_drive_delta`
must not be used here. A differential robot has two degrees of freedom
(forward, turn); this one has three (x, y, turn).

The convention used throughout
------------------------------
Robot body frame:

* **+x** points forward
* **+y** points left
* **omega** is counter-clockwise positive

Wheel *i* is mounted at angle ``alpha_i`` measured counter-clockwise from +x,
at distance ``wheel_offset_m`` from the chassis centre. Its rollers let it slide
sideways, so it only ever measures or drives along its tangential direction,
which is perpendicular to its mounting radius.

Deriving the wheel equation
---------------------------
The contact point of wheel *i* sits at ``R * (cos a, sin a)``. Its velocity is
the body velocity plus the rotational term::

    v_contact = (vx - omega*R*sin a,  vy + omega*R*cos a)

The wheel can only drive along its rolling direction ``(-sin a, cos a)``, so
project onto it::

    v_i = -vx*sin a + vy*cos a + omega*R

which is the equation every function below is built on. The sideways component
is absorbed by the rollers and is exactly the information omni wheels throw
away — which is why their odometry is worse than a differential drive's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Three wheels evenly spaced. 0/120/240 puts wheel 1 straight ahead.
DEFAULT_WHEEL_ANGLES_DEG = (0.0, 120.0, 240.0)


@dataclass(frozen=True)
class HolonomicGeometry:
    """Physical configuration of the kiwi-drive base.

    All three numbers must be MEASURED. `wheel_offset_m` in particular is the
    distance from the chassis centre to each wheel's contact patch, not to the
    motor body or the chassis edge — an error here shows up as the robot
    rotating slightly whenever it is commanded to translate.
    """

    wheel_radius_m: float = 0.029
    wheel_offset_m: float = 0.100
    wheel_angles_deg: tuple[float, ...] = DEFAULT_WHEEL_ANGLES_DEG

    # Encoder counts per full revolution of the wheel. Bus servos usually
    # report absolute position from a 12-bit magnetic encoder (4096 counts),
    # which is far finer than the 360 counts the differential build required.
    ticks_per_revolution: int = 4096

    # Per-wheel multiplicative correction, in wheel order. Compensates for the
    # three wheels never being identical; uncorrected, the robot drifts along a
    # slow curve when commanded to drive straight.
    wheel_trim: tuple[float, ...] = (1.0, 1.0, 1.0)

    def validate(self) -> None:
        if self.wheel_radius_m <= 0:
            raise ValueError("wheel_radius_m must be positive")
        if self.wheel_offset_m <= 0:
            raise ValueError("wheel_offset_m must be positive")
        if len(self.wheel_angles_deg) != 3:
            raise ValueError("a kiwi drive has exactly three wheels")
        if len(self.wheel_trim) != 3:
            raise ValueError("wheel_trim must have one entry per wheel")

        # Three wheels whose angles are too close together cannot span the
        # plane: the robot would be unable to move in some direction at all,
        # and the forward-kinematics matrix would be singular.
        angles = sorted(a % 360.0 for a in self.wheel_angles_deg)
        for i in range(3):
            gap = (angles[(i + 1) % 3] - angles[i]) % 360.0
            if gap < 15.0:
                raise ValueError(
                    f"wheel angles {self.wheel_angles_deg} are nearly collinear; "
                    "the drive would be singular"
                )

    @property
    def wheel_circumference_m(self) -> float:
        return 2.0 * math.pi * self.wheel_radius_m

    @property
    def metres_per_tick(self) -> float:
        return self.wheel_circumference_m / self.ticks_per_revolution


@dataclass
class BodyTwist:
    """Robot-frame velocity command: translate and rotate at once."""

    vx_mps: float = 0.0
    vy_mps: float = 0.0
    omega_dps: float = 0.0

    def is_stationary(self, epsilon: float = 1e-9) -> bool:
        return (
            abs(self.vx_mps) < epsilon
            and abs(self.vy_mps) < epsilon
            and abs(self.omega_dps) < epsilon
        )


@dataclass
class WheelSpeeds:
    """Linear speed at each wheel's rim, metres per second."""

    values: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def as_rad_per_s(self, geometry: HolonomicGeometry) -> tuple[float, float, float]:
        r = geometry.wheel_radius_m
        return tuple(v / r for v in self.values)  # type: ignore[return-value]

    def as_rpm(self, geometry: HolonomicGeometry) -> tuple[float, float, float]:
        circumference = geometry.wheel_circumference_m
        return tuple(v / circumference * 60.0 for v in self.values)  # type: ignore[return-value]

    @property
    def max_abs(self) -> float:
        return max(abs(v) for v in self.values)


# ── Inverse kinematics: what the robot should do -> what each wheel must do ───


def inverse_kinematics(
    twist: BodyTwist, geometry: HolonomicGeometry
) -> WheelSpeeds:
    """Body twist -> the three wheel rim speeds that produce it.

    This is the function called on every control cycle, in Isaac Sim and on the
    real servo bus alike.
    """
    omega_rad = math.radians(twist.omega_dps)
    speeds = []

    for index, angle_deg in enumerate(geometry.wheel_angles_deg):
        a = math.radians(angle_deg)
        v = (
            -twist.vx_mps * math.sin(a)
            + twist.vy_mps * math.cos(a)
            + omega_rad * geometry.wheel_offset_m
        )
        speeds.append(v * geometry.wheel_trim[index])

    return WheelSpeeds(tuple(speeds))  # type: ignore[arg-type]


def scale_to_limit(
    speeds: WheelSpeeds, max_wheel_speed_mps: float
) -> tuple[WheelSpeeds, bool]:
    """Scale all three wheels down together if any exceeds the limit.

    Scaling *proportionally* rather than clipping each wheel independently is
    the point. Clipping changes the ratio between the three speeds, and on a
    holonomic base the ratio is the direction of travel — so a clipped robot
    asked to drive diagonally at full speed would set off at the wrong angle
    instead of simply going slower. Uniform scaling preserves the direction and
    sacrifices only speed.

    Returns the scaled speeds and whether any scaling was applied.
    """
    if max_wheel_speed_mps <= 0:
        raise ValueError("max_wheel_speed_mps must be positive")

    peak = speeds.max_abs
    if peak <= max_wheel_speed_mps:
        return speeds, False

    factor = max_wheel_speed_mps / peak
    return WheelSpeeds(tuple(v * factor for v in speeds.values)), True  # type: ignore[arg-type]


# ── Forward kinematics: what the wheels did -> what the robot did ────────────


def _forward_matrix(geometry: HolonomicGeometry) -> list[list[float]]:
    """The 3x3 matrix M with v = M @ (vx, vy, omega_rad)."""
    rows = []
    for angle_deg in geometry.wheel_angles_deg:
        a = math.radians(angle_deg)
        rows.append([-math.sin(a), math.cos(a), geometry.wheel_offset_m])
    return rows


def _invert_3x3(m: list[list[float]]) -> list[list[float]]:
    """Explicit 3x3 inverse by cofactors.

    Written out rather than pulled from numpy so this module stays importable
    inside Isaac Sim's Python environment without extra packages, and so it can
    run on a microcontroller-class interpreter later if that ever matters.
    """
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise ValueError("wheel configuration is singular; check the wheel angles")

    inv_det = 1.0 / det
    return [
        [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
        [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
        [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
    ]


def forward_kinematics(
    speeds: WheelSpeeds, geometry: HolonomicGeometry
) -> BodyTwist:
    """Three wheel rim speeds -> the body twist they imply.

    This is how odometry is recovered from the servos' own encoders. Note it is
    an *exact* inverse of `inverse_kinematics` only when the wheels do not slip;
    omni-wheel rollers slip by design, so the real robot's recovered twist is
    always a little wrong. That is a property of the platform, not a bug, and it
    is why the mapping stack leans on range sensing rather than odometry alone.
    """
    inverse = _invert_3x3(_forward_matrix(geometry))

    # Undo the per-wheel trim before solving, so calibration does not leak into
    # the estimate as a phantom body motion.
    corrected = [
        speeds.values[i] / geometry.wheel_trim[i] for i in range(3)
    ]

    vx = sum(inverse[0][i] * corrected[i] for i in range(3))
    vy = sum(inverse[1][i] * corrected[i] for i in range(3))
    omega_rad = sum(inverse[2][i] * corrected[i] for i in range(3))

    return BodyTwist(vx_mps=vx, vy_mps=vy, omega_dps=math.degrees(omega_rad))


# ── Odometry ─────────────────────────────────────────────────────────────────


@dataclass
class HolonomicDelta:
    """Pose change in the WORLD frame produced by one control interval."""

    delta_x_m: float
    delta_y_m: float
    delta_heading_deg: float
    distance_m: float


def integrate_twist(
    twist: BodyTwist, heading_deg: float, dt_s: float
) -> HolonomicDelta:
    """Integrate a body twist into a world-frame pose change.

    Uses exact arc integration when the robot is rotating. A holonomic robot
    translating while it rotates traces a curve, and treating that as a straight
    line in the world frame accumulates error every time the robot does the one
    thing this platform is built for: moving and turning at once.

    The closed form comes from rotating the body velocity through the heading as
    it sweeps, integrated over the interval::

        integral of R(theta0 + omega*t) dt  for t in [0, dt]

    which evaluates to the (sin, cos) differences below.
    """
    omega_rad_s = math.radians(twist.omega_dps)
    theta0 = math.radians(heading_deg)

    if abs(omega_rad_s) < 1e-9:
        # Straight line: the heading is constant, so this is a plain rotation
        # of the body velocity into the world frame.
        cos_t, sin_t = math.cos(theta0), math.sin(theta0)
        dx = (twist.vx_mps * cos_t - twist.vy_mps * sin_t) * dt_s
        dy = (twist.vx_mps * sin_t + twist.vy_mps * cos_t) * dt_s
        d_heading = 0.0
    else:
        theta1 = theta0 + omega_rad_s * dt_s
        sin0, cos0 = math.sin(theta0), math.cos(theta0)
        sin1, cos1 = math.sin(theta1), math.cos(theta1)

        dx = (
            twist.vx_mps * (sin1 - sin0) + twist.vy_mps * (cos1 - cos0)
        ) / omega_rad_s
        dy = (
            -twist.vx_mps * (cos1 - cos0) + twist.vy_mps * (sin1 - sin0)
        ) / omega_rad_s
        d_heading = math.degrees(omega_rad_s * dt_s)

    return HolonomicDelta(
        delta_x_m=dx,
        delta_y_m=dy,
        delta_heading_deg=d_heading,
        distance_m=math.hypot(dx, dy),
    )


def wheel_deltas_to_speeds(
    tick_deltas: tuple[int, int, int],
    dt_s: float,
    geometry: HolonomicGeometry,
) -> WheelSpeeds:
    """Encoder tick increments -> wheel rim speeds."""
    if dt_s <= 0:
        return WheelSpeeds()
    metres_per_tick = geometry.metres_per_tick
    return WheelSpeeds(
        tuple(delta * metres_per_tick / dt_s for delta in tick_deltas)  # type: ignore[arg-type]
    )


def wrap_tick_delta_holonomic(current: int, previous: int, span: int) -> int:
    """Return current - previous, correcting for a counter that wraps at `span`.

    Distinct from `geometry.wrap_tick_delta`, which assumes a power-of-two
    microcontroller counter. A bus servo reports *absolute* position within one
    revolution, so its counter wraps at `counts_per_revolution` — 4096 on a
    12-bit encoder, which is not a power of two boundary the other function
    would recognise.

    Getting this wrong is not subtle in its effect but is easy to miss in code:
    every time a wheel passes its zero point the robot appears to jump most of
    a wheel revolution, and the map tears.
    """
    if span <= 0:
        raise ValueError("span must be positive")

    delta = (current - previous) % span
    # A real step is far less than half a revolution at any sane control rate,
    # so anything larger is the counter having wrapped the other way.
    if delta > span // 2:
        delta -= span
    return delta


def odometry_from_ticks(
    tick_deltas: tuple[int, int, int],
    heading_deg: float,
    dt_s: float,
    geometry: HolonomicGeometry,
) -> HolonomicDelta:
    """Full odometry step: encoder ticks in, world-frame pose change out."""
    speeds = wheel_deltas_to_speeds(tick_deltas, dt_s, geometry)
    twist = forward_kinematics(speeds, geometry)
    return integrate_twist(twist, heading_deg, dt_s)


# ── Convenience: named motions ───────────────────────────────────────────────


@dataclass
class DriveLimits:
    """Speed ceilings, used by teleop and by the explorer."""

    max_linear_mps: float = 0.30
    max_angular_dps: float = 90.0
    max_wheel_mps: float = 0.45


def twist_from_direction(
    direction_deg: float, speed_mps: float, omega_dps: float = 0.0
) -> BodyTwist:
    """Build a twist that translates along a bearing in the ROBOT frame.

    0 degrees is straight ahead, 90 is directly left. This is the call that
    makes holonomic motion obvious in code: strafing right is simply
    ``twist_from_direction(-90, 0.2)``, with no turn involved.
    """
    a = math.radians(direction_deg)
    return BodyTwist(
        vx_mps=speed_mps * math.cos(a),
        vy_mps=speed_mps * math.sin(a),
        omega_dps=omega_dps,
    )
