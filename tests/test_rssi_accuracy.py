"""Does Bluetooth RSSI positioning earn a place in the map?

The project was asked to add Bluetooth trilateration. Rather than assume it
works or refuse to try, this measures it: the robot drives a real circuit,
RSSI and odometry both estimate position, and both are scored against ground
truth the simulator knows exactly.

The answer is a number, and the number decides how the fusion treats it.
"""

from __future__ import annotations

import math

import pytest
from localization.fusion import PoseFilter
from robotmap_common.holonomic import HolonomicGeometry
from robotmap_common.rssi import BeaconLayout, trilaterate

from autonomy.explorer import ExploreState, WallFollower
from simulator.virtual_robot import NoiseProfile, VirtualRobot, VirtualWorld

ROOM_W, ROOM_H = 6.0, 4.5
GEOM = HolonomicGeometry(ticks_per_revolution=4096)
DT_S = 0.1


class Run:
    """One circuit with both position sources recorded against truth."""

    def __init__(self) -> None:
        self.rssi_errors: list[float] = []
        self.odometry_errors: list[float] = []
        self.rssi_fixes = 0
        self.rssi_failures = 0
        self.samples = 0


def drive_and_measure(
    shadowing_db: float = 6.0,
    seed: int = 42,
    max_steps: int = 2500,
    layout: BeaconLayout | None = None,
) -> Run:
    """Drive a lap, comparing both estimates against ground truth every step."""
    world = VirtualWorld.rectangular_room(ROOM_W, ROOM_H)
    world.indoor = True

    noise = NoiseProfile(ble_shadowing_sigma_db=shadowing_db)
    robot = VirtualRobot(world=world, noise=noise, seed=seed)
    robot.true_x, robot.true_y = 1.0, 1.0

    beacons = layout or BeaconLayout.room_corners(ROOM_W, ROOM_H)
    robot.attach_beacons(beacons)

    pose_filter = PoseFilter("MR3W01", holonomic_geometry=GEOM)
    follower = WallFollower()
    run = Run()

    start_x, start_y = 1.0, 1.0

    for _ in range(max_steps):
        packet = robot.build_packet(dt_ms=int(DT_S * 1000))
        pose = pose_filter.update(packet)

        truth_x, truth_y, _ = robot.true_pose()

        # Odometry works in a map frame whose origin is the start point.
        odometry_error = math.hypot(
            pose.x_m - (truth_x - start_x), pose.y_m - (truth_y - start_y)
        )

        fix = trilaterate(robot.read_beacons(), beacons.beacons)
        run.samples += 1

        if fix.converged:
            run.rssi_fixes += 1
            run.rssi_errors.append(math.hypot(fix.x_m - truth_x, fix.y_m - truth_y))
            run.odometry_errors.append(odometry_error)
        else:
            run.rssi_failures += 1

        command = follower.step(packet.ranges, pose.x_m, pose.y_m, DT_S)
        if command.state == ExploreState.FINISHED:
            break
        robot.drive(command.linear_mps, command.angular_dps, DT_S)

    return run


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


# ── The headline measurement ─────────────────────────────────────────────────


def test_rssi_produces_fixes_throughout_the_circuit():
    """It works — a fix nearly every step from four corner beacons."""
    run = drive_and_measure()
    assert run.samples > 500
    assert run.rssi_fixes / run.samples > 0.95


def test_rssi_error_is_metre_scale():
    """The number the whole question turns on.

    A correct RSSI implementation in a realistically noisy room lands around a
    metre. That is not a bug to be tuned away — it is the technique.
    """
    run = drive_and_measure()
    mean_error = _mean(run.rssi_errors)
    assert 0.3 < mean_error < 4.0, f"mean RSSI error {mean_error:.2f} m"


def test_odometry_beats_rssi_over_one_circuit():
    """The comparison that decides the architecture.

    Over a single lap odometry is far more accurate, which is why the map is
    built from odometry and RSSI is not allowed to redraw it.
    """
    run = drive_and_measure()
    rssi = _mean(run.rssi_errors)
    odometry = _mean(run.odometry_errors)

    assert odometry < rssi, (
        f"odometry {odometry:.2f} m vs RSSI {rssi:.2f} m — "
        "if RSSI ever wins here, revisit the fusion"
    )
    assert rssi > odometry * 2


def test_rssi_error_does_not_grow_with_distance_driven():
    """RSSI's one real advantage, and the reason to keep it.

    Odometry drifts without bound; RSSI is wrong by metres but no *more* wrong
    after an hour than after a minute. Comparing the first and last thirds of
    the circuit shows the error is stationary.
    """
    run = drive_and_measure()
    third = len(run.rssi_errors) // 3
    early = _mean(run.rssi_errors[:third])
    late = _mean(run.rssi_errors[-third:])

    assert late < early * 2.0, (
        f"RSSI error grew from {early:.2f} m to {late:.2f} m; it should be "
        "bounded, not drifting"
    )


# ── Sensitivity ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("sigma_db", [2.0, 6.0, 10.0])
def test_accuracy_tracks_the_environment(sigma_db):
    """A quiet corridor and a cluttered office are different problems."""
    run = drive_and_measure(shadowing_db=sigma_db)
    assert run.rssi_fixes > 100
    assert _mean(run.rssi_errors) < 10.0


def test_a_quiet_environment_is_measurably_better():
    quiet = _mean(drive_and_measure(shadowing_db=2.0).rssi_errors)
    noisy = _mean(drive_and_measure(shadowing_db=10.0).rssi_errors)
    assert quiet < noisy


def test_beacons_on_one_wall_cannot_fix_a_position_at_all():
    """Not merely worse — impossible.

    Four beacons in a straight line leave the perpendicular direction
    completely unconstrained: any point and its mirror image across that line
    fit the ranges equally well. The solver must refuse rather than pick one.

    This is the single most likely installation mistake, so it is worth
    knowing it fails loudly instead of silently halving the accuracy.
    """
    from robotmap_common.rssi import Beacon

    one_wall = BeaconLayout()
    for index, x in enumerate((0.5, 2.0, 3.5, 5.0)):
        one_wall.add(Beacon(f"W{index}", x, 0.0))

    run = drive_and_measure(layout=one_wall)

    assert run.rssi_fixes == 0, "collinear beacons must not produce a fix"
    assert run.rssi_failures > 100


def test_beacon_placement_matters_as_much_as_noise():
    """Nearly-collinear beacons are solvable but much worse.

    A realistic bad install: beacons roughly along one wall with small
    variations. It fixes, and it is measurably worse than corners — which is
    the argument for spending five minutes on placement.
    """
    from robotmap_common.rssi import Beacon

    corners = BeaconLayout.room_corners(ROOM_W, ROOM_H)

    near_wall = BeaconLayout()
    for index, (x, y) in enumerate(((0.5, 0.0), (2.0, 0.4), (3.5, 0.0), (5.0, 0.4))):
        near_wall.add(Beacon(f"W{index}", x, y))

    good = _mean(drive_and_measure(layout=corners).rssi_errors)
    bad = _mean(drive_and_measure(layout=near_wall).rssi_errors)

    assert bad > good, f"corner {good:.2f} m vs near-wall {bad:.2f} m"


@pytest.mark.parametrize("seed", [1, 42, 2024])
def test_the_result_holds_across_noise_draws(seed):
    """Any single run could be luck."""
    run = drive_and_measure(seed=seed)
    assert _mean(run.rssi_errors) < 5.0
    assert _mean(run.odometry_errors) < _mean(run.rssi_errors)


# ── The conclusion, as an executable statement ───────────────────────────────


def test_rssi_is_too_coarse_to_draw_a_room():
    """Why RSSI is gated out of the map, stated as a test.

    The room is 6.0 x 4.5 m. If the position feeding the occupancy grid is
    uncertain by a metre or more, walls land a metre from where they belong
    and the floor plan is worthless — the same argument already established
    for GPS, at a smaller scale.
    """
    run = drive_and_measure()
    rssi_error = _mean(run.rssi_errors)
    smallest_room_dimension = ROOM_H

    assert rssi_error > smallest_room_dimension * 0.05, (
        "RSSI error is under 5% of the room's smallest dimension — if this "
        "ever holds on real hardware, RSSI could contribute to the map and "
        "the gate should be revisited"
    )
