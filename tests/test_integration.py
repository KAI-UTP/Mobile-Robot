"""Full-pipeline tests: virtual robot -> fusion -> occupancy grid -> room area.

These run the entire system exactly as the deployed stack does, differing only
in that the sensor packets come from the simulator instead of a radio link.
They are the tests that justify the project's central claim: that this robot
can measure a room it has never seen.
"""

from __future__ import annotations

import math

import pytest
from localization.fusion import PoseFilter
from mapping.occupancy_grid import OccupancyGrid
from mapping.room_extraction import RoomExtractor
from robotmap_common.geometry import RobotGeometry
from robotmap_common.models import PoseSource

from autonomy.explorer import ExploreState, WallFollower
from simulator.virtual_robot import VirtualRobot, VirtualWorld

# 360 counts per wheel revolution — the resolution the BOM specifies.
# `test_encoder_resolution_dominates_drift` shows what happens below it.
GEOM = RobotGeometry(wheel_diameter_m=0.065, wheel_base_m=0.150, ticks_per_revolution=360)
DT_S = 0.1


class PipelineResult:
    def __init__(self) -> None:
        self.outline = None
        self.final_pose = None
        self.true_pose = None
        self.filter = None
        self.grid = None
        self.loop_closed = False
        self.steps = 0
        self.start_x = 0.0
        self.start_y = 0.0

    def pose_error_m(self) -> float:
        """Distance between estimate and ground truth, in a common frame.

        The filter works in a map frame whose origin is wherever the robot
        started; ground truth is in world coordinates. Comparing them
        directly would measure the start offset, not the drift.
        """
        true_x, true_y, _ = self.true_pose
        return math.hypot(
            self.final_pose.x_m - (true_x - self.start_x),
            self.final_pose.y_m - (true_y - self.start_y),
        )


def run_pipeline(
    world: VirtualWorld,
    start_x: float = 1.0,
    start_y: float = 1.0,
    max_steps: int = 4000,
    indoor: bool = True,
    seed: int = 42,
    geometry: RobotGeometry | None = None,
) -> PipelineResult:
    """Drive the virtual robot around the room and map it.

    `geometry` configures both the simulated hardware and the filter, so
    passing one models building the robot differently.
    """
    world.indoor = indoor

    robot = VirtualRobot(world=world, geometry=geometry or GEOM, seed=seed)
    robot.true_x, robot.true_y = start_x, start_y

    pose_filter = PoseFilter("MR3W01", geometry or GEOM)
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=6.0, max_range_m=4.0)
    follower = WallFollower()
    extractor = RoomExtractor()

    result = PipelineResult()
    result.start_x, result.start_y = start_x, start_y
    pose = None

    for step in range(max_steps):
        packet = robot.build_packet(dt_ms=int(DT_S * 1000))
        pose = pose_filter.update(packet)

        grid.integrate_scan(pose, packet.ranges)
        grid.mark_robot_footprint(pose)

        command = follower.step(packet.ranges, pose.x_m, pose.y_m, DT_S)
        if command.state == ExploreState.FINISHED:
            result.loop_closed = True
            result.steps = step
            break

        robot.drive(command.linear_mps, command.angular_dps, DT_S)

    result.steps = result.steps or max_steps
    result.outline = extractor.extract(
        grid, pose, "MR3W01", "2026-08-15T00:00:00Z"
    )
    result.final_pose = pose
    result.true_pose = robot.true_pose()
    result.filter = pose_filter
    result.grid = grid
    return result


# ── The headline claims ───────────────────────────────────────────────────────


def test_robot_maps_a_rectangular_room_to_correct_area():
    """A 6.0 x 4.5 m room is 27 m2. The robot must find that, having been told
    nothing about the room in advance."""
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    result = run_pipeline(world)

    assert result.outline.polygon, "no room outline produced"
    # 15 % tolerance: the robot only ever sees the walls from a standoff
    # distance, through noisy sonar, from a drifting pose estimate.
    assert result.outline.area_m2 == pytest.approx(27.0, rel=0.15)


def test_robot_measures_room_dimensions():
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    result = run_pipeline(world)

    assert result.outline.bounding_width_m == pytest.approx(6.0, abs=0.6)
    assert result.outline.bounding_height_m == pytest.approx(4.5, abs=0.6)


def test_wall_follower_closes_the_loop():
    """The robot must recognise it has been all the way round."""
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    result = run_pipeline(world)
    assert result.loop_closed, "robot never completed a circuit"


def test_l_shaped_room_is_not_reported_as_its_bounding_box():
    """The real test of the extractor: an L-shaped room's area is much less
    than the rectangle that contains it."""
    world = VirtualWorld.l_shaped_room()
    result = run_pipeline(world, start_x=1.0, start_y=1.0)

    # Ground truth: 6x3 plus 3.5x2 = 18 + 7 = 25 m2. Bounding box is 6x5 = 30.
    assert result.outline.area_m2 == pytest.approx(25.0, rel=0.20)
    assert result.outline.area_m2 < 28.0, "reported the bounding box, not the room"


def test_odometry_drift_stays_bounded_over_a_circuit():
    """Drift over one lap must stay small enough for the map to line up.

    With the specified 360-count encoders, an 18 m circuit around this room
    ends within a couple of decimetres of truth — comfortably enough for the
    grid to stay self-consistent.
    """
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    result = run_pipeline(world)

    error = result.pose_error_m()
    assert error < 0.30, f"pose drifted {error:.2f} m over one circuit"


def test_encoder_resolution_dominates_drift():
    """Justifies the encoder line in the bill of materials.

    A 20-slot disc — the kind bundled with hobby chassis — resolves about
    10 mm of wheel travel, and a single count of difference between the two
    wheels reads as several degrees of turn. The information is lost at the
    sensor, so no filter can recover it. This test pins the difference so the
    requirement cannot be quietly downgraded later.
    """
    coarse = RobotGeometry(
        wheel_diameter_m=0.065, wheel_base_m=0.150, ticks_per_revolution=20
    )
    coarse_result = run_pipeline(VirtualWorld.rectangular_room(6.0, 4.5), geometry=coarse)
    fine_result = run_pipeline(VirtualWorld.rectangular_room(6.0, 4.5))

    assert fine_result.pose_error_m() < coarse_result.pose_error_m() / 2.0


def test_filter_uncertainty_brackets_the_real_error():
    """A filter that is confidently wrong is worse than one that admits doubt.
    Reported sigma must not badly understate the true error."""
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    result = run_pipeline(world)

    error = result.pose_error_m()
    reported = math.hypot(result.final_pose.std_x_m, result.final_pose.std_y_m)

    # Allow the estimate to be optimistic by a factor, but not wildly so.
    assert error < reported * 6.0 + 0.3


# ── GPS behaviour in the real pipeline ────────────────────────────────────────


def test_indoor_run_never_accepts_gps():
    """The whole point: indoors, GPS contributes nothing to the map."""
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    result = run_pipeline(world, indoor=True)

    assert result.filter.gps_acceptances == 0
    assert result.filter.gps_rejections > 0
    assert result.filter.anchor_lat is None
    assert result.final_pose.source != PoseSource.ODOMETRY_IMU_GPS


def test_indoor_map_survives_bad_gps():
    """Even while being fed garbage fixes, the map must still be correct.

    This is the test that would fail if GPS were naively trusted: the indoor
    fixes wander by tens of metres, and folding them in would scatter the
    occupancy grid across a wide area and destroy the area measurement.
    """
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    result = run_pipeline(world, indoor=True)

    assert result.outline.area_m2 == pytest.approx(27.0, rel=0.15)


def test_outdoor_run_anchors_the_map():
    """Outdoors the fixes become good enough to georeference the map."""
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    result = run_pipeline(world, indoor=False)

    assert result.filter.anchor_lat is not None
    assert result.filter.gps_acceptances > 0
    assert result.final_pose.anchor_latitude is not None


def test_outdoor_consumer_gps_never_steers_the_map():
    """Even a clean outdoor fix is ~4 m accurate — too coarse for a 6 m room."""
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    result = run_pipeline(world, indoor=False)
    assert result.filter.gps_corrections == 0


def test_outdoor_gps_does_not_wreck_the_map():
    """The map built outdoors must be as good as the one built indoors.

    Before the accuracy gate existed, this test failed hard: consumer fixes
    dragged the pose by metres, the grid smeared, loop closure never fired,
    and the reported area of this 27 m2 room ranged from 26 to 54 m2 purely
    on the noise draw.
    """
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    result = run_pipeline(world, indoor=False)
    assert result.outline.area_m2 == pytest.approx(27.0, rel=0.15)
    assert result.loop_closed


# ── Robustness ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [1, 7, 42, 123, 2024])
def test_result_is_stable_across_noise_seeds(seed):
    """Any single good run could be luck. Repeat with different noise draws."""
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    result = run_pipeline(world, seed=seed)
    assert result.outline.area_m2 == pytest.approx(27.0, rel=0.20)


def test_furnished_room_still_measures_the_floor():
    """Furniture creates interior obstacles; the room area should still be
    recovered rather than the robot mapping the furniture as the boundary."""
    world = VirtualWorld.room_with_furniture(6.0, 4.5)
    result = run_pipeline(world, start_x=0.6, start_y=0.6)

    assert result.outline.polygon
    # Furniture blocks some floor from view, so accept a wider band.
    assert result.outline.area_m2 == pytest.approx(27.0, rel=0.30)


def test_small_room_is_mapped():
    world = VirtualWorld.rectangular_room(3.0, 2.5)
    result = run_pipeline(world, start_x=0.6, start_y=0.6)
    assert result.outline.area_m2 == pytest.approx(7.5, rel=0.25)


def test_map_reports_coverage_and_closure():
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    result = run_pipeline(world)
    assert result.outline.coverage_pct > 0.0
    assert result.outline.is_closed is True
