"""Bluetooth positioning from signal strength: beacons, path loss, multilateration.

How it works
------------
Radio power falls off predictably with distance, so a measured RSSI can be
inverted into an estimated range. With three or more beacons at known
positions, three ranges intersect at a point — trilateration. More than three
over-determines the system and is solved by least squares, which is what this
module does.

Why it is hard, stated up front
-------------------------------
The inversion is exponential, so RSSI error becomes distance error very
quickly. With a path-loss exponent of 2.5, distance goes as
``10^((TxPower - RSSI) / 25)``:

    RSSI error   distance error at 5 m
    ---------    ---------------------
      1 dB              ~10 %  (0.5 m)
      3 dB              ~32 %  (1.6 m)
      6 dB              ~74 %  (3.7 m)

Indoor shadowing is routinely 4-8 dB one-sigma, and worse when a person stands
between the tag and a beacon. So metre-level error is the *expected* outcome
of a correct implementation, not a sign of a bug.

That is why this module reports an uncertainty alongside every fix, and why
`localization/fusion.py` gates RSSI the same way it gates GPS: a position is
only allowed to correct the map if it is more precise than the map needs.

What it is genuinely good for
-----------------------------
* **Room-level presence** — which room the robot is in, reliably.
* **Bounding unbounded odometry drift** — odometry drifts without limit; RSSI
  is wrong by metres but never *more* wrong the longer you drive.
* **Recovering from being picked up and moved**, which odometry cannot detect
  at all.

Whether it earns a place in the map is an empirical question, and
`tests/test_rssi_accuracy.py` answers it with numbers rather than opinion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Reference distance for the path-loss model. One metre is the convention, and
# it is where beacon TxPower is normally specified.
REFERENCE_DISTANCE_M = 1.0

# Typical path-loss exponents. Free space is 2.0; walls and furniture raise it.
PATH_LOSS_FREE_SPACE = 2.0
PATH_LOSS_OPEN_ROOM = 2.2
PATH_LOSS_FURNISHED_ROOM = 2.7
PATH_LOSS_THROUGH_WALLS = 3.5


@dataclass(frozen=True)
class Beacon:
    """A BLE beacon at a surveyed position.

    `tx_power_dbm` is the RSSI a receiver sees at exactly one metre — a
    calibration constant, not the transmit power. It varies by several dB
    between supposedly identical beacons, so it must be measured per beacon;
    assuming a datasheet value is one of the largest error sources here.
    """

    beacon_id: str
    x_m: float
    y_m: float
    tx_power_dbm: float = -59.0

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(x - self.x_m, y - self.y_m)


@dataclass
class BeaconReading:
    """One RSSI sample from one beacon."""

    beacon_id: str
    rssi_dbm: float
    # Beacons advertise several times a second; averaging a short window before
    # inverting cuts the shadowing noise considerably.
    sample_count: int = 1


@dataclass
class RssiFix:
    """An estimated position from trilateration."""

    x_m: float
    y_m: float
    beacons_used: int
    residual_m: float
    estimated_error_m: float
    converged: bool = True
    reason: str = ""

    @property
    def is_usable(self) -> bool:
        """Whether this fix is worth anything at all.

        Deliberately generous — this only rejects nonsense. Whether the fix is
        good enough to touch the map is a separate and much stricter decision,
        made by the pose filter.
        """
        return self.converged and self.beacons_used >= 3 and self.estimated_error_m < 15.0


# ── Path loss ────────────────────────────────────────────────────────────────


def rssi_to_distance(
    rssi_dbm: float,
    tx_power_dbm: float = -59.0,
    path_loss_exponent: float = PATH_LOSS_FURNISHED_ROOM,
) -> float:
    """Invert the log-distance path-loss model to a range in metres.

        RSSI = TxPower - 10 * n * log10(d / d0)
        =>  d = d0 * 10 ^ ((TxPower - RSSI) / (10 * n))

    Note how steeply this amplifies error: the exponent means a few dB of
    shadowing becomes a large fraction of the distance. See the module
    docstring for the numbers.
    """
    if path_loss_exponent <= 0:
        raise ValueError("path_loss_exponent must be positive")

    exponent = (tx_power_dbm - rssi_dbm) / (10.0 * path_loss_exponent)
    distance = REFERENCE_DISTANCE_M * (10.0**exponent)

    # Clamp to something physically plausible for an indoor beacon. A wildly
    # low RSSI otherwise produces a kilometre-scale range that dominates the
    # least-squares fit and drags the whole solution away.
    return max(0.05, min(distance, 60.0))


def distance_to_rssi(
    distance_m: float,
    tx_power_dbm: float = -59.0,
    path_loss_exponent: float = PATH_LOSS_FURNISHED_ROOM,
) -> float:
    """Forward model — used by the simulator to synthesise readings."""
    distance = max(distance_m, 0.01)
    return tx_power_dbm - 10.0 * path_loss_exponent * math.log10(
        distance / REFERENCE_DISTANCE_M
    )


def distance_error_from_rssi_error(
    rssi_error_db: float,
    distance_m: float,
    path_loss_exponent: float = PATH_LOSS_FURNISHED_ROOM,
) -> float:
    """How much distance error a given RSSI error implies at some range.

    Exists to make the amplification explicit and testable rather than a claim
    in a comment.
    """
    factor = 10.0 ** (rssi_error_db / (10.0 * path_loss_exponent))
    return abs(distance_m * (factor - 1.0))


# ── Trilateration ────────────────────────────────────────────────────────────


def _linear_least_squares(
    beacons: list[Beacon], distances: list[float]
) -> tuple[float, float] | None:
    """Closed-form starting estimate.

    Squaring the circle equations and subtracting the first from the rest
    cancels the quadratic terms and leaves a linear system. It is not the
    optimal answer — it weights beacons unevenly — but it is a good starting
    point for the iterative refinement, and unlike a guess it cannot start in
    a local minimum on the wrong side of the room.
    """
    if len(beacons) < 3:
        return None

    x0, y0, r0 = beacons[0].x_m, beacons[0].y_m, distances[0]

    a11 = a12 = a21 = a22 = b1 = b2 = 0.0
    for beacon, radius in zip(beacons[1:], distances[1:], strict=True):
        ax = 2.0 * (beacon.x_m - x0)
        ay = 2.0 * (beacon.y_m - y0)
        b = (
            r0**2 - radius**2
            - x0**2 + beacon.x_m**2
            - y0**2 + beacon.y_m**2
        )
        a11 += ax * ax
        a12 += ax * ay
        a21 += ay * ax
        a22 += ay * ay
        b1 += ax * b
        b2 += ay * b

    determinant = a11 * a22 - a12 * a21
    if abs(determinant) < 1e-12:
        # Beacons are collinear: the geometry cannot resolve a position.
        return None

    return ((b1 * a22 - b2 * a12) / determinant, (a11 * b2 - a21 * b1) / determinant)


def _beacon_span(beacons: list[Beacon]) -> float:
    """Diagonal of the box the beacons occupy."""
    xs = [b.x_m for b in beacons]
    ys = [b.y_m for b in beacons]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _rms_residual(
    beacons: list[Beacon], distances: list[float], x: float, y: float
) -> float:
    residuals = [
        math.hypot(x - b.x_m, y - b.y_m) - d
        for b, d in zip(beacons, distances, strict=True)
    ]
    return math.sqrt(sum(r * r for r in residuals) / len(residuals))


def _refine(
    beacons: list[Beacon],
    distances: list[float],
    start: tuple[float, float],
    iterations: int = 40,
    max_step_m: float = 5.0,
) -> tuple[float, float, float]:
    """Levenberg-Marquardt refinement minimising the range residuals.

    Returns (x, y, rms_residual).

    Damped rather than plain Gauss-Newton, and step-limited, because plain
    Gauss-Newton *diverges* on this problem. When noisy ranges are mutually
    inconsistent the circles have no common intersection, the Jacobian goes
    near-singular, and an undamped step is enormous — an earlier version
    reported positions tens of kilometres away, which is worse than useless
    because a wildly wrong fix still looks like a fix.
    """
    x, y = start
    best_x, best_y = x, y
    best_cost = _rms_residual(beacons, distances, x, y)

    # Damping: raised when a step makes things worse, lowered when it helps.
    # High damping degrades gracefully toward a short gradient-descent step
    # instead of an unbounded Newton one.
    damping = 1e-3

    for _ in range(iterations):
        jtj_xx = jtj_xy = jtj_yy = 0.0
        jtr_x = jtr_y = 0.0

        for beacon, measured in zip(beacons, distances, strict=True):
            dx = x - beacon.x_m
            dy = y - beacon.y_m
            predicted = math.hypot(dx, dy)
            if predicted < 1e-9:
                continue

            residual = predicted - measured
            gx = dx / predicted
            gy = dy / predicted

            jtj_xx += gx * gx
            jtj_xy += gx * gy
            jtj_yy += gy * gy
            jtr_x += gx * residual
            jtr_y += gy * residual

        # LM: inflate the diagonal. As damping grows the step shortens and
        # turns toward steepest descent, which cannot run away.
        a11 = jtj_xx * (1.0 + damping)
        a22 = jtj_yy * (1.0 + damping)
        determinant = a11 * a22 - jtj_xy * jtj_xy
        if abs(determinant) < 1e-12:
            break

        step_x = (jtr_x * a22 - jtr_y * jtj_xy) / determinant
        step_y = (jtr_y * a11 - jtr_x * jtj_xy) / determinant

        # Hard cap as well as damping: one pathological iteration must not be
        # able to throw the estimate across the map.
        step_length = math.hypot(step_x, step_y)
        if step_length > max_step_m:
            scale = max_step_m / step_length
            step_x *= scale
            step_y *= scale

        candidate_x = x - step_x
        candidate_y = y - step_y
        candidate_cost = _rms_residual(beacons, distances, candidate_x, candidate_y)

        if candidate_cost < best_cost:
            x, y = candidate_x, candidate_y
            best_x, best_y, best_cost = x, y, candidate_cost
            damping = max(damping * 0.5, 1e-6)
        else:
            # Rejected: damp harder and try a shorter, safer step.
            damping = min(damping * 4.0, 1e6)
            if damping >= 1e6:
                break

        if step_length < 1e-6:
            break

    return best_x, best_y, best_cost


def trilaterate(
    readings: list[BeaconReading],
    beacons: dict[str, Beacon],
    path_loss_exponent: float = PATH_LOSS_FURNISHED_ROOM,
    shadowing_sigma_db: float = 6.0,
) -> RssiFix:
    """Estimate a position from beacon RSSI readings.

    `shadowing_sigma_db` is how noisy the environment is believed to be; it
    does not change the position, only the honesty of the reported
    uncertainty.
    """
    known = [
        (reading, beacons[reading.beacon_id])
        for reading in readings
        if reading.beacon_id in beacons
    ]

    if len(known) < 3:
        return RssiFix(
            x_m=0.0, y_m=0.0, beacons_used=len(known),
            residual_m=0.0, estimated_error_m=999.0, converged=False,
            reason=f"need 3 beacons for a 2-D fix, heard {len(known)}",
        )

    used_beacons = [beacon for _, beacon in known]
    distances = [
        rssi_to_distance(reading.rssi_dbm, beacon.tx_power_dbm, path_loss_exponent)
        for reading, beacon in known
    ]

    start = _linear_least_squares(used_beacons, distances)
    if start is None:
        return RssiFix(
            x_m=0.0, y_m=0.0, beacons_used=len(known),
            residual_m=0.0, estimated_error_m=999.0, converged=False,
            reason="beacons are collinear; no unique solution",
        )

    x, y, residual = _refine(used_beacons, distances, start)

    if not (math.isfinite(x) and math.isfinite(y)):
        return RssiFix(
            x_m=0.0, y_m=0.0, beacons_used=len(known),
            residual_m=0.0, estimated_error_m=999.0, converged=False,
            reason="solver diverged",
        )

    # Sanity bound. A receiver hearing these beacons at all is within radio
    # range, so a fix far outside the beacon field is not a position — it is
    # the solver having failed on mutually inconsistent ranges. Reporting it
    # as a valid fix would be worse than reporting nothing.
    span = _beacon_span(used_beacons)
    centre_x = sum(b.x_m for b in used_beacons) / len(used_beacons)
    centre_y = sum(b.y_m for b in used_beacons) / len(used_beacons)
    distance_from_centre = math.hypot(x - centre_x, y - centre_y)
    limit = span + max(distances) + 5.0

    if distance_from_centre > limit:
        return RssiFix(
            x_m=x, y_m=y, beacons_used=len(known),
            residual_m=residual, estimated_error_m=999.0, converged=False,
            reason=(
                f"solution {distance_from_centre:.0f} m from the beacons; "
                "ranges are mutually inconsistent"
            ),
        )

    error = estimate_position_error(
        used_beacons, distances, (x, y), shadowing_sigma_db, path_loss_exponent
    )

    return RssiFix(
        x_m=x, y_m=y,
        beacons_used=len(known),
        residual_m=residual,
        estimated_error_m=error,
        converged=True,
        reason="ok",
    )


def estimate_position_error(
    beacons: list[Beacon],
    distances: list[float],
    position: tuple[float, float],
    shadowing_sigma_db: float,
    path_loss_exponent: float,
) -> float:
    """Predict the position error, from the ranging error and the geometry.

    Two things drive it, and both matter:

    * **Ranging error** — shadowing translated into metres, which grows with
      distance because the inversion is exponential.
    * **Geometric dilution** — beacons bunched in one direction leave the
      perpendicular direction poorly constrained. Three beacons in a line give
      an excellent residual and a position that can be metres out sideways, so
      residual alone is not a usable confidence measure.
    """
    range_errors = [
        distance_error_from_rssi_error(shadowing_sigma_db, d, path_loss_exponent)
        for d in distances
    ]
    mean_range_error = sum(range_errors) / len(range_errors)

    dilution = geometric_dilution(beacons, position)
    return mean_range_error * dilution


def geometric_dilution(
    beacons: list[Beacon], position: tuple[float, float]
) -> float:
    """How well the beacon layout constrains this point. 1.0 is ideal.

    Computed from the spread of bearings to the beacons. Bearings spanning all
    directions constrain both axes; bearings clustered in an arc leave the
    perpendicular direction weak.
    """
    x, y = position
    bearings = [
        math.atan2(beacon.y_m - y, beacon.x_m - x)
        for beacon in beacons
        if math.hypot(beacon.x_m - x, beacon.y_m - y) > 1e-9
    ]
    if len(bearings) < 3:
        return 5.0

    # Sum the unit bearing vectors: cancellation means good spread, alignment
    # means they all pull the same way and the fit is weak across it.
    sum_x = sum(math.cos(b) for b in bearings)
    sum_y = sum(math.sin(b) for b in bearings)
    alignment = math.hypot(sum_x, sum_y) / len(bearings)  # 0 = ideal, 1 = worst

    # Maps 0 -> 1.0 and approaches 6.0 as the beacons align.
    return 1.0 + 5.0 * alignment**2


# ── Beacon layouts ───────────────────────────────────────────────────────────


@dataclass
class BeaconLayout:
    """A set of beacons covering a room."""

    beacons: dict[str, Beacon] = field(default_factory=dict)

    def add(self, beacon: Beacon) -> BeaconLayout:
        self.beacons[beacon.beacon_id] = beacon
        return self

    @classmethod
    def room_corners(
        cls, width_m: float, height_m: float, tx_power_dbm: float = -59.0
    ) -> BeaconLayout:
        """One beacon per corner — the practical minimum for a room.

        Corners maximise the bearing spread from anywhere inside, which is
        what keeps the geometric dilution near 1. Four rather than three also
        means one blocked beacon still leaves a solvable fix.
        """
        layout = cls()
        for index, (x, y) in enumerate(
            [(0.0, 0.0), (width_m, 0.0), (width_m, height_m), (0.0, height_m)]
        ):
            layout.add(Beacon(f"B{index + 1}", x, y, tx_power_dbm))
        return layout

    def as_list(self) -> list[Beacon]:
        return list(self.beacons.values())
