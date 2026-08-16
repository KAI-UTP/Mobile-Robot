"""Growing obstacles by the robot's own size.

The idea is Nav2's inflation layer — the coloured halos around every obstacle
in an RViz costmap — implemented here without ROS.

It exists because a planner reasons about the robot as a *point*. That is only
safe if obstacles are first grown by the robot's radius: a point that stays out
of the grown region is a robot whose whole body stays out of the real one.
Without it the robot clips corners it planned straight through.
"""

from __future__ import annotations

import numpy as np
import pytest
from mapping.costmap import (
    FREE,
    INSCRIBED,
    LETHAL,
    Costmap,
)

RES = 0.05


def _grid(width=60, height=60):
    return np.zeros((height, width), dtype=bool)


def _with_obstacle_at(row=30, col=30):
    grid = _grid()
    grid[row, col] = True
    return grid


# ── The three bands ──────────────────────────────────────────────────────────


def test_the_obstacle_itself_is_lethal():
    cost = Costmap().build(_with_obstacle_at(), RES)
    assert cost[30, 30] == LETHAL


def test_within_a_robot_radius_is_forbidden_not_merely_expensive():
    """A cell within one radius of an obstacle is in collision at EVERY
    heading. No orientation rescues it, so it is as forbidden as the obstacle."""
    costmap = Costmap(robot_radius_m=0.11)
    cost = costmap.build(_with_obstacle_at(), RES)

    # Two cells away is 0.10 m — inside the 0.11 m radius.
    assert cost[30, 32] == INSCRIBED
    assert costmap.is_forbidden(cost[30, 32])


def test_beyond_the_radius_is_discouraged_but_allowed():
    """This is what makes a robot prefer the middle of a doorway to scraping
    the frame — a preference, not a prohibition."""
    costmap = Costmap(robot_radius_m=0.11, inflation_radius_m=0.35)
    cost = costmap.build(_with_obstacle_at(), RES)

    # 0.25 m away: inside the inflation band, outside the inscribed one.
    value = cost[30, 35]
    assert FREE < value < INSCRIBED
    assert not costmap.is_forbidden(value)


def test_far_from_everything_is_free():
    cost = Costmap(inflation_radius_m=0.35).build(_with_obstacle_at(), RES)
    assert cost[0, 0] == FREE


# ── The shape of the decay ───────────────────────────────────────────────────


def test_cost_falls_with_distance():
    """A flat halo would give a planner no reason to prefer the middle of a
    gap over its edge."""
    cost = Costmap().build(_with_obstacle_at(), RES)

    near = int(cost[30, 34])
    far = int(cost[30, 37])
    assert near > far > FREE


def test_a_higher_scaling_factor_hugs_obstacles_more_tightly():
    """The tuning knob, kept with Nav2's name and meaning so advice written
    for Nav2 applies here."""
    loose = Costmap(cost_scaling_factor=1.0).build(_with_obstacle_at(), RES)
    tight = Costmap(cost_scaling_factor=6.0).build(_with_obstacle_at(), RES)

    assert int(tight[30, 36]) < int(loose[30, 36])


def test_the_halo_reaches_about_the_inflation_radius():
    costmap = Costmap(robot_radius_m=0.10, inflation_radius_m=0.30)
    cost = costmap.build(_with_obstacle_at(), RES)

    inflated = np.argwhere(cost > FREE)
    rows, cols = inflated[:, 0], inflated[:, 1]
    reach_cells = max(
        int(np.abs(rows - 30).max()), int(np.abs(cols - 30).max())
    )
    # 0.30 m at 0.05 m cells is 6 cells; chamfer rounding allows one over.
    assert 5 <= reach_cells <= 7


# ── Geometry ─────────────────────────────────────────────────────────────────


def test_the_halo_surrounds_the_obstacle_on_all_sides():
    """A one-sided inflation would leave the robot free to approach from the
    direction it usually does."""
    cost = Costmap().build(_with_obstacle_at(), RES)

    assert cost[28, 30] > FREE     # above
    assert cost[32, 30] > FREE     # below
    assert cost[30, 28] > FREE     # left
    assert cost[30, 32] > FREE     # right


def test_two_obstacles_close_together_leave_no_gap():
    """The case that matters most: a gap narrower than the robot must not read
    as passable.

    The arithmetic is worth stating, because it is easy to get backwards. The
    robot fits between two obstacles when its CENTRE can sit at least one
    radius from each — so a 0.11 m radius needs them more than 0.22 m apart.
    At 0.15 m apart the midpoint is only 0.075 m from each, and every pose in
    the gap is in collision.
    """
    grid = _grid()
    grid[30, 27] = True
    grid[30, 30] = True           # 3 cells = 0.15 m apart, under one diameter

    costmap = Costmap(robot_radius_m=0.11)
    cost = costmap.build(grid, RES)

    assert costmap.is_forbidden(
        cost[30, 28]
    ), "a gap too narrow to enter reads as open"
    assert costmap.is_forbidden(cost[30, 29])


def test_a_gap_wider_than_the_robot_stays_open():
    """The boundary case, from the other side. 0.30 m apart leaves 0.15 m of
    clearance for a 0.11 m radius, so the robot fits and the costmap must not
    pretend otherwise — inflation that seals passable gaps is as wrong as
    inflation that misses impassable ones."""
    grid = _grid()
    grid[30, 27] = True
    grid[30, 33] = True           # 6 cells = 0.30 m apart

    costmap = Costmap(robot_radius_m=0.11)
    assert not costmap.is_forbidden(costmap.build(grid, RES)[30, 30])


def test_a_wide_gap_stays_passable():
    """The complement: inflation must not seal every doorway."""
    grid = _grid()
    grid[30, 15] = True
    grid[30, 45] = True           # 1.5 m apart

    costmap = Costmap(robot_radius_m=0.11)
    cost = costmap.build(grid, RES)

    assert not costmap.is_forbidden(cost[30, 30])


# ── Edges and degenerate input ───────────────────────────────────────────────


def test_an_empty_map_inflates_to_nothing():
    cost = Costmap().build(_grid(), RES)
    assert cost.max() == FREE


def test_an_obstacle_at_the_edge_does_not_crash():
    grid = _grid()
    grid[0, 0] = True
    cost = Costmap().build(grid, RES)

    assert cost[0, 0] == LETHAL
    assert cost[1, 1] > FREE


def test_the_inflation_radius_cannot_be_smaller_than_the_robot():
    """Otherwise the inscribed band would extend past the inflated one and the
    bands would contradict each other."""
    with pytest.raises(ValueError):
        Costmap(robot_radius_m=0.20, inflation_radius_m=0.10)


def test_resolution_is_respected():
    """The radius is in metres, so a finer grid must inflate over more cells,
    not the same number."""
    fine = Costmap().build(_with_obstacle_at(), 0.025)
    coarse = Costmap().build(_with_obstacle_at(), 0.10)

    assert (fine > FREE).sum() > (coarse > FREE).sum()


# ── Reporting ────────────────────────────────────────────────────────────────


def test_the_summary_separates_the_bands():
    """They mean different things to a planner and should not be added up."""
    costmap = Costmap()
    summary = costmap.summarise(costmap.build(_with_obstacle_at(), RES), RES)

    assert summary["lethal_m2"] > 0
    assert summary["inscribed_m2"] > 0
    assert summary["inflated_m2"] > 0
    assert summary["robot_radius_m"] == costmap.robot_radius_m
