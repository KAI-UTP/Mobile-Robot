"""Drawing what the robot bumps into.

Range sensors miss a great deal of real furniture: a chair leg narrower than
the ultrasonic beam, a sofa that absorbs the pulse, anything angled enough to
reflect the echo away, and everything mounted below the sensor. The bumper is
what catches those — but until a contact is written into the occupancy grid it
exists only in the controller's memory, so the robot steers around an obstacle
it never draws.

These tests are about the contact reaching the map, staying there, and not
stopping the scan.
"""

from __future__ import annotations

import math

import pytest
from mapper.pipeline import MappingPipeline
from mapping.occupancy_grid import OccupancyGrid
from robotmap_common.models import (
    DriveKind,
    EncoderData,
    PoseEstimate,
    RangeReading,
    SensorPacket,
)


def _pose(x=1.0, y=1.0, heading=0.0):
    return PoseEstimate(
        robot_id="TEST01", timestamp="2026-08-16T00:00:00Z", sequence=1,
        x_m=x, y_m=y, heading_deg=heading % 360, std_x_m=0.01, std_y_m=0.01,
    )


def _grid():
    return OccupancyGrid(resolution_m=0.05, initial_size_m=8.0, max_range_m=4.0)


def _occupied_at(grid, x, y):
    col, row = grid.world_to_cell(x, y)
    return grid.grid[row, col] > 0.4


# ── The contact reaches the map ──────────────────────────────────────────────


def test_a_contact_marks_the_map():
    grid = _grid()
    assert grid.mark_contact(_pose(1.0, 1.0, heading=0.0)) > 0


def test_it_marks_in_front_of_the_robot_not_under_it():
    """The switch is on the nose. Marking the centre would put the obstacle
    exactly where the robot is standing — the one place it is not."""
    grid = _grid()
    grid.mark_contact(_pose(1.0, 1.0, heading=0.0), bumper_offset_m=0.10)

    assert _occupied_at(grid, 1.10, 1.00), "nothing marked ahead of the robot"
    assert not _occupied_at(grid, 0.85, 1.00), "marked behind the robot"


@pytest.mark.parametrize("heading", [0.0, 90.0, 180.0, 270.0, 45.0])
def test_it_marks_whichever_way_the_robot_is_facing(heading):
    """A contact fixed to one axis would draw furniture in the wrong place for
    three quarters of a sweep — and a boustrophedon sweep spends half its time
    facing backwards."""
    grid = _grid()
    grid.mark_contact(_pose(2.0, 2.0, heading=heading), bumper_offset_m=0.12)

    angle = math.radians(heading)
    ahead_x = 2.0 + 0.12 * math.cos(angle)
    ahead_y = 2.0 + 0.12 * math.sin(angle)
    assert _occupied_at(grid, ahead_x, ahead_y)


def test_a_contact_is_immediately_solid():
    """One touch has to be enough. Requiring several would mean the robot must
    hit the same chair three times before it appears, and the avoidance logic
    exists precisely to stop that happening."""
    grid = _grid()
    grid.mark_contact(_pose(1.0, 1.0))

    col, row = grid.world_to_cell(1.10, 1.0)
    assert grid.grid[row, col] >= grid.LOG_ODDS_BLOCKING


def test_a_contact_outweighs_a_range_reading():
    """A bumper is a switch closed by the object itself. An echo is an
    inference, and a wrong one often enough."""
    assert _log_odds_of(OccupancyGrid.P_CONTACT) > _log_odds_of(
        OccupancyGrid.P_OCCUPIED
    )


def _log_odds_of(p):
    return math.log(p / (1.0 - p))


def test_the_grid_grows_to_hold_a_contact_at_its_edge():
    """A contact just past the current bounds must not be silently dropped."""
    grid = OccupancyGrid(resolution_m=0.05, initial_size_m=2.0, max_range_m=4.0)
    assert grid.mark_contact(_pose(0.95, 0.0, heading=0.0)) > 0


# ── It has to survive ────────────────────────────────────────────────────────


def test_the_robot_footprint_does_not_erase_a_contact():
    """The free circle under the robot is a little wider than the chassis and
    the bumper sits just outside it. Left alone, a robot stopped against an
    obstacle scrubs out the contact it just recorded — at ten packets a second,
    within about half a second."""
    grid = _grid()
    pose = _pose(1.0, 1.0, heading=0.0)
    grid.mark_contact(pose)

    for _ in range(50):
        grid.mark_robot_footprint(pose)

    assert _occupied_at(grid, 1.10, 1.0), "the contact was rubbed out"


def test_the_footprint_still_clears_ordinary_unknown_floor():
    """The skip must be narrow: closing the gaps under the robot is the whole
    reason that marking exists."""
    grid = _grid()
    pose = _pose(3.0, 3.0)
    grid.mark_robot_footprint(pose)

    col, row = grid.world_to_cell(3.0, 3.0)
    assert grid.grid[row, col] < 0


def test_a_few_passing_rays_do_not_erase_a_contact():
    """Sonar skimming past an obstacle reports the wall behind it and carves
    free space through the object. A contact has to outlast that."""
    grid = _grid()
    pose = _pose(1.0, 1.0, heading=0.0)
    grid.mark_contact(pose)

    for _ in range(5):
        grid.integrate_scan(pose, [RangeReading(angle_deg=0.0, distance_m=3.0)])

    assert _occupied_at(grid, 1.10, 1.0)


# ── Through the pipeline ─────────────────────────────────────────────────────


def _packet(sequence, bumper=False, ticks=(0, 0, 0)):
    return SensorPacket(
        robot_id="TEST01",
        timestamp="2026-08-16T00:00:00Z",
        sequence=sequence,
        drive=DriveKind.HOLONOMIC_3WHEEL,
        wheel_ticks=list(ticks),
        bumper_active=bumper,
        encoders=EncoderData(left_ticks=ticks[0], right_ticks=ticks[1], dt_ms=100),
        ranges=[RangeReading(angle_deg=0.0, distance_m=2.0)],
    )


def test_the_pipeline_records_a_contact():
    pipeline = MappingPipeline(robot_id="TEST01")
    pipeline.process(_packet(1))
    pipeline.process(_packet(2, bumper=True))

    assert pipeline.contacts == 1


def test_one_touch_counts_once_however_long_the_switch_stays_shut():
    """The switch stays closed for as long as the robot is against the object.
    Re-marking every packet would smear a single touch into a wall as the
    robot backs away from it."""
    pipeline = MappingPipeline(robot_id="TEST01")
    pipeline.process(_packet(1))
    for seq in range(2, 22):
        pipeline.process(_packet(seq, bumper=True))

    assert pipeline.contacts == 1


def test_touching_again_after_letting_go_counts_again():
    pipeline = MappingPipeline(robot_id="TEST01")
    pipeline.process(_packet(1))
    pipeline.process(_packet(2, bumper=True))
    pipeline.process(_packet(3, bumper=False))
    pipeline.process(_packet(4, bumper=True))

    assert pipeline.contacts == 2


def test_a_contact_does_not_stop_the_scan():
    """The whole point. The map gains a blocked patch and mapping carries on —
    an obstacle is information, not a failure."""
    pipeline = MappingPipeline(robot_id="TEST01")
    pipeline.process(_packet(1, bumper=True))

    before = pipeline.packets_processed
    for seq in range(2, 12):
        pipeline.process(_packet(seq))

    assert pipeline.packets_processed == before + 10
    assert pipeline.pose is not None


def test_contacts_are_reported():
    pipeline = MappingPipeline(robot_id="TEST01")
    pipeline.process(_packet(1))
    pipeline.process(_packet(2, bumper=True))

    assert pipeline.state()["diagnostics"]["contacts"] == 1


def test_a_reset_forgets_the_contacts():
    """Otherwise a rescan reports the previous run's collisions against a map
    they did not come from."""
    pipeline = MappingPipeline(robot_id="TEST01")
    pipeline.process(_packet(1))
    pipeline.process(_packet(2, bumper=True))
    pipeline.reset(clear_map=True)

    assert pipeline.contacts == 0



# ── It becomes a red area on the map ─────────────────────────────────────────


def _fully_mapped_grid(width=5.0, height=4.0):
    """A rectangular room swept thoroughly enough to extract cleanly."""
    grid = _grid()
    step = 0.3
    y = step
    while y < height:
        x = step
        while x < width:
            pose = _pose(x, y)
            readings = []
            for bearing in range(0, 360, 6):
                angle = math.radians(bearing)
                dx, dy = math.cos(angle), math.sin(angle)
                distance, limit = 0.01, 6.0
                while distance < limit:
                    px, py = x + dx * distance, y + dy * distance
                    if px <= 0 or px >= width or py <= 0 or py >= height:
                        break
                    distance += 0.01
                readings.append(
                    RangeReading(
                        angle_deg=bearing if bearing <= 180 else bearing - 360,
                        distance_m=min(distance, limit),
                    )
                )
            grid.integrate_scan(pose, readings)
            grid.mark_robot_footprint(pose)
            x += step
        y += step
    return grid


def _mapped_room(contacts=(), width=5.0, height=4.0):
    """That room, plus zero or more bumper contacts, extracted."""
    from mapping.room_extraction import RoomExtractor

    grid = _fully_mapped_grid(width, height)
    for cx, cy, heading in contacts:
        grid.mark_contact(_pose(cx, cy, heading))

    # Extracted from a pose clear of every contact. A real robot reverses off
    # whatever it touched before the map is re-extracted; the case where it has
    # not is covered separately below.
    return RoomExtractor().extract(
        grid, _pose(1.0, 1.0), "TEST01", "2026-08-16T00:00:00Z"
    )


def test_a_touched_object_becomes_blocked_floor_without_circling_it():
    """The behaviour the whole feature exists for. Hole-filling needs free
    space observed all the way AROUND an object before it registers, and a
    single touch-and-retreat never encloses anything — measured on a slim
    pillar the sonar kept missing: 2 contacts recorded, 0.00 m2 reported."""
    empty = _mapped_room()
    touched = _mapped_room(contacts=[(2.5, 2.0, 0.0)])

    assert empty.blocked_area_m2 == 0.0
    assert touched.blocked_area_m2 > 0.0
    assert len(touched.obstacles) >= 1


def test_the_red_area_is_where_the_robot_touched():
    touched = _mapped_room(contacts=[(2.5, 2.0, 0.0)])
    obstacle = max(touched.obstacles, key=lambda o: o.area_m2)

    # Bumper is ahead of centre, so the mark sits just beyond 2.5 in x.
    assert obstacle.centre_x_m == pytest.approx(2.6, abs=0.25)
    assert obstacle.centre_y_m == pytest.approx(2.0, abs=0.25)


def test_bumping_a_wall_does_not_invent_furniture():
    """Walls are the room, not something standing in it. A red patch on the
    boundary would double-count the wall and shrink the usable floor for no
    reason."""
    against_wall = _mapped_room(contacts=[(4.85, 2.0, 0.0)])

    for obstacle in against_wall.obstacles:
        assert obstacle.centre_x_m < 4.6, "a wall contact was drawn as furniture"


def test_a_contact_under_the_reported_pose_does_not_destroy_the_map():
    """Odometry drift can put the robot's reported position inside a patch it
    marked moments ago — the mark goes 10 cm ahead of an *estimated* pose.

    The flood fill seeds from that pose. Boxed inside an occupied patch it
    filled one cell, gave up, and returned an EMPTY room: a 20.45 m2 room
    became 0.00 m2 from a single contact. Found while writing these tests.
    """
    from mapping.room_extraction import RoomExtractor

    grid = _fully_mapped_grid()
    seed = _pose(2.5, 2.0)
    grid.mark_contact(seed)

    room = RoomExtractor().extract(grid, seed, "TEST01", "2026-08-16T00:00:00Z")
    assert room.area_m2 > 15.0, "an occupied seed emptied the whole room"


def test_the_room_area_is_unchanged_by_a_contact():
    """A table does not shrink the room, and neither does touching one. Total
    floor still has to be laid; only usable floor drops."""
    empty = _mapped_room()
    touched = _mapped_room(contacts=[(2.5, 2.0, 0.0)])

    assert touched.area_m2 == pytest.approx(empty.area_m2, rel=0.05)
    assert touched.usable_area_m2 < touched.area_m2


def test_a_contact_while_still_touching_after_a_reset_is_counted_again():
    """The edge tracker must reset with everything else, or a rescan that
    begins with the robot already against something never records it."""
    pipeline = MappingPipeline(robot_id="TEST01")
    pipeline.process(_packet(1, bumper=True))
    pipeline.reset(clear_map=True)
    pipeline.process(_packet(2, bumper=True))

    assert pipeline.contacts == 1
