"""The 3D scene is the physical world, not a picture of one.

Why this matters
----------------
For most of this project the renderer and the world the robot drove in were two
separate descriptions that merely looked alike, and they drifted apart over and
over: furniture drawn in a room the simulator knew to be bare, a robot gliding
through a sofa that existed only on screen, a 2D map that came back a plain
rectangle while both 3D views showed a furnished room.

The fix is that one of them has to be in charge. The Omniverse scene is, because
that is the one a person can reach into and rearrange. Drag a table across the
viewport and the simulated robot has to meet it where you put it — collide with
it, infer the contact from the servo bus exactly as the hardware would, and draw
the obstacle on the 2D map in its new place.

These tests cover both halves of that loop: the scene reading its own stage, and
the simulator accepting what it is told.
"""

from __future__ import annotations

import math

import pytest

from simulator.virtual_robot import VirtualWorld

# ── The simulator accepts a new layout ───────────────────────────────────────


def test_furniture_can_be_replaced_without_touching_the_room():
    """The room shell is not ours to rearrange; the furniture is."""
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    shell_before = [w for w in world.walls if w.kind == "shell"]

    world.set_furniture([(2.0, 2.0, 3.0, 3.0)])

    assert [w for w in world.walls if w.kind == "shell"] == shell_before
    assert len(world.furniture_footprints) == 1


def test_a_new_layout_replaces_the_old_one():
    """Not appends. Dragging a table across the room must not leave a ghost of
    it behind for the robot to bump into."""
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    world.set_furniture([(1.0, 1.0, 2.0, 2.0)])
    world.set_furniture([(4.0, 3.0, 5.0, 3.8)])

    assert world.furniture_footprints == [(4.0, 3.0, 5.0, 3.8)]


def test_the_robot_can_no_longer_drive_where_the_furniture_now_is():
    """The point of the whole exercise."""
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    # Clear floor to begin with: a ray across the middle reaches the far wall.
    assert world.raycast(1.0, 2.25, 0.0, 8.0) == pytest.approx(5.0, abs=0.05)

    world.set_furniture([(3.0, 1.75, 3.5, 2.75)])

    # Now it stops at the table.
    assert world.raycast(1.0, 2.25, 0.0, 8.0) == pytest.approx(2.0, abs=0.05)


def test_moving_the_furniture_moves_where_the_robot_is_blocked():
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    world.set_furniture([(2.0, 1.75, 2.5, 2.75)])
    near = world.raycast(1.0, 2.25, 0.0, 8.0)

    world.set_furniture([(4.0, 1.75, 4.5, 2.75)])
    far = world.raycast(1.0, 2.25, 0.0, 8.0)

    assert far > near
    assert far == pytest.approx(3.0, abs=0.05)


def test_clearing_the_room_leaves_it_empty():
    world = VirtualWorld.room_with_furniture(6.0, 4.5)
    assert world.furniture_footprints

    world.set_furniture([])

    assert world.furniture_footprints == []
    # And the room is still a room.
    assert world.raycast(1.0, 2.25, 0.0, 8.0) == pytest.approx(5.0, abs=0.05)


def test_a_flat_footprint_is_discarded():
    """A zero-width box is invisible to the raycaster but would still register
    as contact at a point, which reads on the map as a phantom obstacle."""
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    world.set_furniture([(2.0, 2.0, 2.0, 3.0), (1.0, 1.0, 2.0, 2.0)])

    assert len(world.furniture_footprints) == 1


def test_contact_is_reported_against_the_new_layout():
    """Contact is what the real robot has instead of range readings, so it is
    the path that actually matters for this hardware."""
    world = VirtualWorld.rectangular_room(6.0, 4.5)
    world.set_furniture([(2.9, 2.15, 3.1, 2.35)])

    # Standing right beside the new table.
    assert world.nearest_wall_distance(2.8, 2.25) < 0.15
    # And nowhere near where it used to be.
    assert world.nearest_wall_distance(1.0, 2.25) > 0.5


# ── The scene reads its own stage ────────────────────────────────────────────


def test_the_scene_reports_the_furniture_it_has_drawn(kit):
    module, _ = kit
    module.build_room()

    boxes = module.furniture_footprints(module.omni.usd.get_context().get_stage())
    assert boxes, "the scene drew furniture but reported none of it"
    for min_x, min_y, max_x, max_y in boxes:
        assert max_x > min_x and max_y > min_y


def test_a_dragged_prim_is_reported_where_it_now_is(kit):
    """The mechanism. USD keeps the transform on the prim, so moving one in the
    viewport changes what this reads — which is why it reads the stage rather
    than the constants the stage was built from."""
    module, stage = kit
    module.build_room()

    table = next(p for path, p in stage.prims.items() if "TableTop" in path)
    original = module.furniture_footprints(stage)

    table.translate = (table.translate[0] + 1.5, table.translate[1], table.translate[2])
    moved = module.furniture_footprints(stage)

    assert moved != original
    shifted = [b for b in moved if b not in original]
    assert shifted, "moving a table changed nothing"


def test_the_rug_is_not_reported_as_an_obstacle(kit):
    """A robot does not bump into a rug, and drawing one as blocked floor puts
    a red patch on the map where the floor is perfectly clear."""
    module, stage = kit
    module.build_room()

    rug = next(
        (p for path, p in stage.prims.items() if "Rug" in path), None
    )
    if rug is None:
        pytest.skip("this room has no rug")

    rug_x, rug_y = rug.translate[0], rug.translate[1]
    for min_x, min_y, max_x, max_y in module.furniture_footprints(stage):
        covers = min_x <= rug_x <= max_x and min_y <= rug_y <= max_y
        # Something else may legitimately stand on the rug; what must not
        # happen is a footprint the exact size of the rug.
        if covers:
            assert not (
                math.isclose(max_x - min_x, abs(rug.scale[0]), abs_tol=0.01)
                and math.isclose(max_y - min_y, abs(rug.scale[1]), abs_tol=0.01)
            ), "the rug itself was reported as an obstacle"


# ── Only speaking up when something changed ──────────────────────────────────


def test_geometry_is_not_pushed_when_nothing_moved(kit, monkeypatch):
    """This runs off the render loop at 60 Hz. The mapper does not need sixty
    identical messages a second."""
    module, stage = kit
    module.build_room()

    posts = []
    monkeypatch.setattr(
        module.GeometryPublisher, "_post",
        lambda self, boxes: posts.append(boxes) or True,
    )

    publisher = module.GeometryPublisher(stage, interval_s=0.0)
    assert publisher.poll(1.0) is True          # first look: always news
    assert publisher.poll(2.0) is False         # nothing moved
    assert publisher.poll(3.0) is False
    assert len(posts) == 1


def test_moving_something_does_push(kit, monkeypatch):
    module, stage = kit
    module.build_room()

    posts = []
    monkeypatch.setattr(
        module.GeometryPublisher, "_post",
        lambda self, boxes: posts.append(boxes) or True,
    )

    publisher = module.GeometryPublisher(stage, interval_s=0.0)
    publisher.poll(1.0)

    table = next(p for path, p in stage.prims.items() if "TableTop" in path)
    table.translate = (table.translate[0] + 0.8, table.translate[1], table.translate[2])

    assert publisher.poll(2.0) is True
    assert len(posts) == 2


def test_a_table_leg_is_not_a_metre_wide_obstacle(kit):
    """The bug that reached the live stack: 35 pieces pushed to the simulator,
    and every table and chair leg among them was a 1.0 x 1.0 m box.

    `GetChildren()` hands back plain `Usd.Prim` objects, which have no
    `GetRadiusAttr` — only the typed `UsdGeom.Cylinder` wrapper does. So every
    cylinder fell out of the radius branch into the box branch and was measured
    by its scale, which a cylinder never sets, leaving the default of 1. A room
    furnished with metre-wide legs has almost no floor left to drive on.
    """
    module, stage = kit
    module.build_room()

    for min_x, min_y, max_x, max_y in module.furniture_footprints(stage):
        width, depth = max_x - min_x, max_y - min_y
        assert width < 2.6 and depth < 2.6, (
            f"a {width:.2f} x {depth:.2f} m piece of furniture in a 6.0 x 4.5 m room"
        )


def test_the_legs_are_leg_sized(kit):
    """A 0.035 m radius leg is 0.07 m across, not 1.0."""
    module, stage = kit
    module.build_room()

    legs = [p for path, p in stage.prims.items() if "Leg" in path or "leg" in path]
    assert legs, "this room has no legs to check"

    boxes = module.furniture_footprints(stage)
    smallest = min((max_x - min_x) for min_x, _, max_x, _ in boxes)
    assert smallest < 0.2, f"smallest piece is {smallest:.2f} m — legs were mismeasured"


def test_the_room_still_has_floor_to_drive_on(kit):
    """The consequence worth stating in its own right: whatever the scene
    pushes has to leave a room the robot can actually move around in."""
    module, stage = kit
    module.build_room()

    covered = sum(
        (max_x - min_x) * (max_y - min_y)
        for min_x, min_y, max_x, max_y in module.furniture_footprints(stage)
    )
    # Overlapping footprints make this an overestimate, which is the safe
    # direction: if even the overestimate leaves most of the floor clear, the
    # room is drivable.
    assert covered < 0.5 * (6.0 * 4.5), f"{covered:.1f} m2 of a 27 m2 room is furniture"


def test_the_layout_is_repeated_periodically(kit, monkeypatch):
    """The mapper holds the room in memory, so restarting its container throws
    the furniture away. A publisher that only speaks when something moves never
    mentions it again: the room silently empties and the robot drives through
    where the table is on screen — the exact failure this exists to prevent.
    """
    module, stage = kit
    module.build_room()

    posts = []
    monkeypatch.setattr(
        module.GeometryPublisher, "_post",
        lambda self, boxes: posts.append(boxes) or True,
    )

    publisher = module.GeometryPublisher(stage, interval_s=0.0)
    publisher.poll(0.0)
    assert len(posts) == 1

    # Quiet for a while, then the heartbeat comes round.
    publisher.poll(1.0)
    assert len(posts) == 1
    publisher.poll(module.GeometryPublisher.HEARTBEAT_S + 1.0)
    assert len(posts) == 2


def test_a_failed_push_is_retried(kit, monkeypatch):
    """Otherwise the scene believes a layout the mapper never received."""
    module, stage = kit
    module.build_room()

    attempts = []
    monkeypatch.setattr(
        module.GeometryPublisher, "_post",
        lambda self, boxes: attempts.append(boxes) and False,
    )

    publisher = module.GeometryPublisher(stage, interval_s=0.0)
    publisher.poll(0.0)
    publisher.poll(1.0)

    assert len(attempts) == 2, "gave up after one failure"


# ── The trail belongs to one run ─────────────────────────────────────────────


def test_the_trail_is_wiped_for_a_new_run(kit):
    """A trail from the previous run lying over a new one is the same lie as
    furniture that is not there: it shows the robot having been somewhere it
    has not been this time. The mapper clears its map on every fresh start;
    this is the 3D half of that."""
    module, stage = kit
    module.build_room()
    scene = module.RoomScene(stage, source=None)

    for i in range(12):
        scene.x, scene.y = i * 0.3, 1.0
        scene._drop_trail()
    assert [p for p in stage.prims if "/Trail/" in p], "no trail was drawn"

    scene.clear_trail()
    assert not [p for p in stage.prims if "/Trail/" in p], "the trail survived"
    assert scene.trail_index == 0


def test_the_trail_stops_growing(kit):
    """One dot every 0.15 m and a contact run drives 300 m, so an uncapped
    trail is thousands of prims burying the room it is drawn on."""
    module, stage = kit
    module.build_room()
    scene = module.RoomScene(stage, source=None)

    cap = module.MAX_TRAIL_DOTS
    for i in range(cap + 200):
        scene.x, scene.y = (i % 30) * 0.2, (i // 30) * 0.2
        scene._drop_trail()

    dots = [p for p in stage.prims if "/Trail/Dot_" in p]
    assert len(dots) <= cap, f"{len(dots)} dots for a cap of {cap}"


def test_a_recycled_dot_moves_to_where_the_robot_now_is(kit):
    """Recycling must reposition the prim, not leave it where it was — a stale
    dot is a breadcrumb for a place the robot is not."""
    module, stage = kit
    module.build_room()
    scene = module.RoomScene(stage, source=None)

    cap = module.MAX_TRAIL_DOTS
    # The first drop has to be a real move: a dot is only left once the robot
    # has travelled the spacing, so dropping one where it already is does
    # nothing.
    scene.x, scene.y = 0.2, 0.0
    scene._drop_trail()
    first = stage.prims[f"{module.ROOT}/Trail/Dot_0"]
    assert first.translate[0] == pytest.approx(0.2)

    # Drive far enough to come all the way round to Dot_0 again.
    for i in range(1, cap + 1):
        scene.x, scene.y = (i + 1) * 0.2, 0.0
        scene._drop_trail()

    assert first.translate[0] == pytest.approx((cap + 1) * 0.2)


# ── The 3D robot is the physical one ─────────────────────────────────────────
#
# This scene draws the true room at fixed coordinates, so it has to draw the
# true robot. Placed by an estimate that has drifted, the robot appears outside
# a room it is nowhere near the edge of — which reads as collision being broken
# when the simulator is in fact stopping it dead at its own radius from the wall.


def _scene_with(module, stage, pose):
    scene = module.RoomScene(stage, source=type("S", (), {"read": lambda s: pose})())
    scene._on_update(None)
    return scene


def test_the_solid_robot_follows_the_true_pose(kit):
    module, stage = kit
    module.build_room()

    # An estimate that has drifted well outside the room, and the truth.
    pose = {
        "x_m": 9.0, "y_m": 9.0, "heading_deg": 0.0,
        "true_x_m": 2.0, "true_y_m": 1.5, "true_heading_deg": 0.0,
    }
    scene = _scene_with(module, stage, pose)

    # Smoothing chases rather than snaps, so it should have moved towards the
    # truth and away from the drifted estimate.
    assert scene.x < 6.0, "the solid robot followed the drifted estimate"
    assert scene.y < 6.0


def test_the_ghost_follows_the_estimate(kit):
    """The gap between the two IS the drift, and it is the most useful thing
    this scene shows."""
    module, stage = kit
    module.build_room()

    pose = {
        "x_m": 4.0, "y_m": 3.0, "heading_deg": 0.0,
        "true_x_m": 1.0, "true_y_m": 1.0, "true_heading_deg": 0.0,
    }
    scene = _scene_with(module, stage, pose)

    assert scene.ox > scene.x, "the ghost is not tracking the estimate"


def test_hardware_falls_back_to_the_estimate(kit):
    """There is no true pose on a real robot, and the scene still has to draw
    something."""
    module, stage = kit
    module.build_room()

    scene = _scene_with(module, stage, {"x_m": 2.0, "y_m": 1.5, "heading_deg": 0.0})
    assert scene.x > 0.0 and scene.y > 0.0


# ── Following the robot, not a recording of one ──────────────────────────────
#
# The source used to be chosen once at startup, which was survivable while the
# mapper drove off the moment it booted. It stopped being survivable when the
# page gained a Start button: the robot is idle when Kit launches, so the scene
# fell back to its canned demo lap permanently and never noticed Start being
# pressed. The web app showed the real robot; Omniverse showed a scripted
# rectangle. Not two views drifting apart — two different robots.


class _Fixed:
    def __init__(self, pose): self.pose = pose
    def read(self): return self.pose


def test_it_shows_the_demo_lap_while_the_robot_is_idle(kit):
    module, _ = kit
    demo = _Fixed({"x_m": 99.0, "y_m": 99.0, "heading_deg": 0.0})
    source = module.LiveOrDemo([_Fixed(None)], demo)

    assert source.read() is demo.pose
    assert source.using_demo


def test_it_switches_to_the_robot_when_it_starts_moving(kit):
    """The whole point: pressing Start has to be noticed."""
    module, _ = kit
    live = _Fixed({"x_m": 1.0, "y_m": 1.0, "heading_deg": 0.0, "sequence": 1})
    source = module.LiveOrDemo([live], _Fixed({"x_m": 99.0, "y_m": 99.0}))

    source.read()                       # first look establishes the sequence
    live.pose = dict(live.pose, sequence=2, x_m=1.2)
    pose = source.read()

    assert pose["x_m"] == 1.2, "did not follow the robot once it started"
    assert not source.using_demo


def test_a_frozen_pose_is_not_a_running_robot(kit):
    """The mapper holds its last pose indefinitely, so "is there a pose?" is
    answered yes for ever once anything has run. A sequence number that stops
    advancing is what says the robot has stopped."""
    import time as _time

    module, _ = kit
    live = _Fixed({"x_m": 1.0, "y_m": 1.0, "heading_deg": 0.0, "sequence": 7})
    demo = _Fixed({"x_m": 99.0, "y_m": 99.0})
    source = module.LiveOrDemo([live], demo)
    source.STALE_AFTER_S = 0.05

    source.read()
    live.pose = dict(live.pose, sequence=8)
    source.read()
    assert not source.using_demo

    _time.sleep(0.1)                    # nothing new arrives
    assert source.read() is demo.pose
    assert source.using_demo


def test_the_http_source_forwards_the_whole_pose(kit, monkeypatch):
    """It used to copy three fields, which silently dropped `sequence` — the
    only way to tell a running robot from a stopped one — and `true_x_m`,
    without which the scene draws the drifted estimate and the robot walks
    through walls again."""
    from test_kit_scene import _stub_fetch

    module, _ = kit
    payload = {"pose": {
        "x_m": 1.0, "y_m": 2.0, "heading_deg": 0.0,
        "sequence": 42, "true_x_m": 1.1, "true_y_m": 2.1,
    }}
    monkeypatch.setattr("urllib.request.urlopen", _stub_fetch(payload))

    pose = module.HttpPose().read()
    assert pose["sequence"] == 42
    assert pose["true_x_m"] == 1.1
