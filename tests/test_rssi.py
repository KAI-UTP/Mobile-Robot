"""Bluetooth RSSI trilateration.

Two jobs. First, prove the maths is right — with no noise, trilateration must
recover the exact position. Second, prove the *uncertainty* is honest, because
the whole question about RSSI is whether it is precise enough to trust, and a
technique that reports confident wrong answers is worse than one that admits
its error.
"""

from __future__ import annotations

import math
import random

import pytest
from robotmap_common.rssi import (
    Beacon,
    BeaconLayout,
    BeaconReading,
    distance_error_from_rssi_error,
    distance_to_rssi,
    geometric_dilution,
    rssi_to_distance,
    trilaterate,
)

ROOM_W, ROOM_H = 6.0, 4.5
LAYOUT = BeaconLayout.room_corners(ROOM_W, ROOM_H)


def _perfect_readings(x: float, y: float, path_loss: float = 2.7):
    """RSSI a receiver at (x, y) would see with no noise at all."""
    return [
        BeaconReading(
            beacon_id=b.beacon_id,
            rssi_dbm=distance_to_rssi(b.distance_to(x, y), b.tx_power_dbm, path_loss),
        )
        for b in LAYOUT.as_list()
    ]


# ── Path loss model ──────────────────────────────────────────────────────────


def test_rssi_at_one_metre_is_the_tx_power():
    """TxPower is defined as the RSSI at one metre; the model must honour it."""
    assert distance_to_rssi(1.0, tx_power_dbm=-59.0) == pytest.approx(-59.0)
    assert rssi_to_distance(-59.0, tx_power_dbm=-59.0) == pytest.approx(1.0)


def test_distance_and_rssi_round_trip():
    for distance in (0.5, 1.0, 2.5, 5.0, 10.0):
        rssi = distance_to_rssi(distance)
        assert rssi_to_distance(rssi) == pytest.approx(distance, rel=1e-9)


def test_signal_weakens_with_distance():
    assert distance_to_rssi(1.0) > distance_to_rssi(5.0) > distance_to_rssi(10.0)


def test_higher_path_loss_means_shorter_inferred_range():
    """A cluttered room attenuates faster, so the same RSSI implies less
    distance. Using a free-space exponent indoors over-estimates range."""
    rssi = -75.0
    assert rssi_to_distance(rssi, path_loss_exponent=3.5) < rssi_to_distance(
        rssi, path_loss_exponent=2.0
    )


def test_absurd_rssi_is_clamped():
    """One wild reading must not produce a kilometre-scale range that
    dominates the least-squares fit."""
    assert rssi_to_distance(-120.0) <= 60.0
    assert rssi_to_distance(10.0) >= 0.05


def test_invalid_path_loss_rejected():
    with pytest.raises(ValueError):
        rssi_to_distance(-70.0, path_loss_exponent=0.0)


# ── The amplification, quantified ────────────────────────────────────────────


@pytest.mark.parametrize(
    "rssi_error_db,minimum_pct",
    [(1.0, 5.0), (3.0, 25.0), (6.0, 60.0)],
)
def test_small_rssi_errors_become_large_distance_errors(rssi_error_db, minimum_pct):
    """The central difficulty with RSSI, as a test rather than a claim.

    At a 2.5 path-loss exponent, 6 dB of shadowing — utterly ordinary indoors —
    is more than half the distance.
    """
    distance = 5.0
    error = distance_error_from_rssi_error(rssi_error_db, distance, 2.5)
    assert error / distance * 100 >= minimum_pct


def test_error_amplification_grows_with_range():
    """Which is why a beacon on the far side of the room contributes least."""
    near = distance_error_from_rssi_error(6.0, 2.0, 2.5)
    far = distance_error_from_rssi_error(6.0, 10.0, 2.5)
    assert far > near * 3


# ── Trilateration with no noise ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "x,y",
    [(3.0, 2.25), (1.0, 1.0), (5.0, 3.5), (0.5, 4.0), (3.0, 0.5)],
)
def test_noiseless_trilateration_is_exact(x, y):
    """With perfect readings the solver must land on the exact point.

    Any error here is a bug in the maths, not a property of RSSI.
    """
    fix = trilaterate(_perfect_readings(x, y), LAYOUT.beacons)

    assert fix.converged
    assert fix.x_m == pytest.approx(x, abs=1e-4)
    assert fix.y_m == pytest.approx(y, abs=1e-4)
    assert fix.residual_m < 1e-4


def test_fewer_than_three_beacons_cannot_fix():
    """Two circles intersect at two points; a plane needs three."""
    readings = _perfect_readings(3.0, 2.25)[:2]
    fix = trilaterate(readings, LAYOUT.beacons)

    assert not fix.converged
    assert not fix.is_usable
    assert "3 beacons" in fix.reason


def test_exactly_three_beacons_works():
    readings = _perfect_readings(3.0, 2.25)[:3]
    fix = trilaterate(readings, LAYOUT.beacons)
    assert fix.converged
    assert fix.x_m == pytest.approx(3.0, abs=1e-3)


def test_unknown_beacons_are_ignored():
    """A stray beacon from a neighbouring room must not be trilaterated
    against, since its position is unknown."""
    readings = _perfect_readings(3.0, 2.25)
    readings.append(BeaconReading("STRANGER", -70.0))

    fix = trilaterate(readings, LAYOUT.beacons)
    assert fix.beacons_used == 4
    assert fix.x_m == pytest.approx(3.0, abs=1e-3)


def test_collinear_beacons_are_rejected():
    """Three beacons on a line leave the perpendicular direction unconstrained;
    there is no unique solution and the solver must say so rather than return
    an arbitrary point."""
    line = {
        "L1": Beacon("L1", 0.0, 2.0),
        "L2": Beacon("L2", 3.0, 2.0),
        "L3": Beacon("L3", 6.0, 2.0),
    }
    readings = [
        BeaconReading(b.beacon_id, distance_to_rssi(b.distance_to(3.0, 3.0)))
        for b in line.values()
    ]
    fix = trilaterate(readings, line)
    assert not fix.converged or fix.estimated_error_m > 5.0


# ── Geometry ─────────────────────────────────────────────────────────────────


def test_corner_beacons_give_good_geometry_from_the_middle():
    dilution = geometric_dilution(LAYOUT.as_list(), (3.0, 2.25))
    assert dilution < 1.5


def test_clustered_beacons_give_poor_geometry():
    """All beacons in one corner: the residual can look excellent while the
    position is metres out along the unconstrained direction."""
    clustered = [
        Beacon("C1", 0.0, 0.0),
        Beacon("C2", 0.4, 0.0),
        Beacon("C3", 0.0, 0.4),
    ]
    assert geometric_dilution(clustered, (5.0, 4.0)) > geometric_dilution(
        LAYOUT.as_list(), (3.0, 2.25)
    )


# ── Honesty of the reported uncertainty ──────────────────────────────────────


def test_reported_error_grows_with_a_noisier_environment():
    readings = _perfect_readings(3.0, 2.25)
    quiet = trilaterate(readings, LAYOUT.beacons, shadowing_sigma_db=2.0)
    noisy = trilaterate(readings, LAYOUT.beacons, shadowing_sigma_db=8.0)
    assert noisy.estimated_error_m > quiet.estimated_error_m * 2


def test_reported_error_is_metre_scale_in_a_realistic_room():
    """The headline finding, pinned as a test.

    With ordinary indoor shadowing, a correct implementation reports metre-
    scale uncertainty. That is the technique, not a defect — and it is why the
    pose filter will not let RSSI redraw a 6 m room.
    """
    fix = trilaterate(
        _perfect_readings(3.0, 2.25), LAYOUT.beacons, shadowing_sigma_db=6.0
    )
    assert fix.estimated_error_m > 0.5
    assert fix.estimated_error_m < 15.0


def test_a_usable_fix_is_still_flagged_usable():
    """Metre-scale error is not the same as worthless; it is enough to know
    which room you are in."""
    fix = trilaterate(_perfect_readings(3.0, 2.25), LAYOUT.beacons)
    assert fix.is_usable


# ── Behaviour under realistic noise ──────────────────────────────────────────


def _noisy_readings(x, y, sigma_db, rng, path_loss=2.7):
    return [
        BeaconReading(
            beacon_id=b.beacon_id,
            rssi_dbm=distance_to_rssi(b.distance_to(x, y), b.tx_power_dbm, path_loss)
            + rng.gauss(0.0, sigma_db),
        )
        for b in LAYOUT.as_list()
    ]


def test_accuracy_degrades_as_shadowing_rises():
    """More noise must mean worse positions — monotonically, over many trials."""
    rng = random.Random(42)
    truth = (3.0, 2.25)

    def mean_error(sigma):
        errors = []
        for _ in range(120):
            fix = trilaterate(_noisy_readings(*truth, sigma, rng), LAYOUT.beacons)
            if fix.converged:
                errors.append(math.dist((fix.x_m, fix.y_m), truth))
        return sum(errors) / len(errors)

    assert mean_error(2.0) < mean_error(6.0) < mean_error(10.0)


def test_realistic_indoor_noise_gives_metre_scale_error():
    """The number this whole exercise exists to establish.

    Six dB shadowing is ordinary indoors. If this ever drops below 0.3 m the
    simulation has become unrealistically kind and every conclusion drawn from
    it should be re-examined.
    """
    rng = random.Random(7)
    truth = (3.0, 2.25)

    errors = []
    for _ in range(300):
        fix = trilaterate(_noisy_readings(*truth, 6.0, rng), LAYOUT.beacons)
        if fix.converged:
            errors.append(math.dist((fix.x_m, fix.y_m), truth))

    mean_error = sum(errors) / len(errors)
    assert 0.3 < mean_error < 5.0, f"mean RSSI error {mean_error:.2f} m"


def test_averaging_samples_helps():
    """Beacons advertise several times a second, and shadowing is largely
    zero-mean, so averaging before inverting is nearly free accuracy."""
    rng = random.Random(11)
    truth = (2.0, 3.0)

    def error_with(samples):
        totals = []
        for _ in range(100):
            averaged = []
            for beacon in LAYOUT.as_list():
                true_rssi = distance_to_rssi(beacon.distance_to(*truth), beacon.tx_power_dbm, 2.7)
                mean = sum(true_rssi + rng.gauss(0, 6.0) for _ in range(samples)) / samples
                averaged.append(BeaconReading(beacon.beacon_id, mean, samples))
            fix = trilaterate(averaged, LAYOUT.beacons)
            if fix.converged:
                totals.append(math.dist((fix.x_m, fix.y_m), truth))
        return sum(totals) / len(totals)

    assert error_with(10) < error_with(1)


def test_the_solver_never_runs_away():
    """Guards a bug that made the technique look catastrophically worse than
    it is.

    An earlier version used undamped Gauss-Newton. When noisy ranges are
    mutually inconsistent the circles have no common intersection, the
    Jacobian goes near-singular, and the step is enormous — it reported
    positions tens of kilometres away, at a mean error of 34 km, while still
    marking them converged. A wildly wrong fix that presents as valid is worse
    than no fix.

    Now damped, step-limited, and bounded to the beacon field.
    """
    rng = random.Random(99)
    worst = 0.0

    for _ in range(500):
        # Punishing noise, well beyond anything realistic, to provoke it.
        fix = trilaterate(_noisy_readings(3.0, 2.25, 15.0, rng), LAYOUT.beacons)
        if fix.converged:
            worst = max(worst, math.dist((fix.x_m, fix.y_m), (3.0, 2.25)))

    assert worst < 60.0, f"a converged fix landed {worst:.0f} m away"


def test_wildly_inconsistent_ranges_are_reported_not_returned():
    """When the ranges cannot describe any single point, say so."""
    impossible = [
        BeaconReading("B1", -95.0),  # implies very far
        BeaconReading("B2", -40.0),  # implies very near
        BeaconReading("B3", -95.0),
        BeaconReading("B4", -40.0),
    ]
    fix = trilaterate(impossible, LAYOUT.beacons)
    # Either it refuses, or it stays somewhere physically plausible.
    assert not fix.converged or math.hypot(fix.x_m - 3.0, fix.y_m - 2.25) < 60.0


def test_the_fix_stays_roughly_inside_the_room():
    """Even noisy, it should not place the robot in the next building — which
    is what makes it useful for bounding drift."""
    rng = random.Random(3)
    outside = 0
    for _ in range(200):
        fix = trilaterate(_noisy_readings(3.0, 2.25, 6.0, rng), LAYOUT.beacons)
        if fix.converged and not (
            -3 < fix.x_m < ROOM_W + 3 and -3 < fix.y_m < ROOM_H + 3
        ):
            outside += 1
    assert outside < 20


# ── Layout ───────────────────────────────────────────────────────────────────


def test_corner_layout_has_four_beacons_at_the_corners():
    layout = BeaconLayout.room_corners(6.0, 4.5)
    assert len(layout.beacons) == 4
    positions = {(b.x_m, b.y_m) for b in layout.as_list()}
    assert positions == {(0.0, 0.0), (6.0, 0.0), (6.0, 4.5), (0.0, 4.5)}


def test_four_beacons_tolerate_one_being_blocked():
    """A person standing in front of a beacon is routine; the fix must
    survive it, which is why four are placed rather than three."""
    readings = _perfect_readings(3.0, 2.25)[1:]  # drop one entirely
    fix = trilaterate(readings, LAYOUT.beacons)
    assert fix.converged
    assert fix.beacons_used == 3
