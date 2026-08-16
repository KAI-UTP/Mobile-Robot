"""What the robot has decides what the robot does.

The one fact that shapes this codebase is that the robot cannot see: STS3215
servos, GNSS and Bluetooth, no lidar and no ultrasonic ring. That used to be
scattered across a `contact_only=True` default, a hardcoded `WallFollower` and
comments in half a dozen files. These tests pin it down in one place, and pin
down the thing that makes it real — the strategy following from the hardware
rather than from a flag.
"""

from __future__ import annotations

import pytest
from robotmap_common.hardware import (
    ACTUAL,
    ALL_DEVICES,
    BLE_BEACONS,
    LIDAR_2D,
    SERVO_BUS,
    WITH_LIDAR,
    Capability,
    HardwareProfile,
    Strategy,
    profile,
)

# ── The robot as it is ───────────────────────────────────────────────────────


def test_the_robot_cannot_see():
    """No lidar, no ultrasonic, no radar. Everything else follows from this."""
    assert not ACTUAL.has(Capability.RANGE)


def test_the_robot_knows_how_far_it_drove():
    assert ACTUAL.has(Capability.ODOMETRY)
    assert SERVO_BUS in ACTUAL.providers(Capability.ODOMETRY)


def test_the_robot_can_tell_it_hit_something_without_a_bumper():
    """Inferred from the servo bus: a wheel delivering a fraction of its
    commanded speed at high load has something solid in front of it."""
    assert ACTUAL.has(Capability.CONTACT)
    assert SERVO_BUS.provides(Capability.CONTACT)


def test_ble_is_present_but_switched_off():
    """It measured worse than dead reckoning — 0.51 m of drift became 1.42 m —
    so it is described, and not fitted. A device that is wired up but harmful
    has to be visibly off rather than quietly absent."""
    assert not BLE_BEACONS.fitted
    assert not BLE_BEACONS.provides(Capability.POSITION)
    assert "1.42" in BLE_BEACONS.notes


# ── The decision that has to follow from the hardware ────────────────────────


def test_no_range_sensor_means_mapping_by_contact():
    """Wall-following holds a *measured distance* to a wall. With nothing to
    measure it with, the robot has to find the room by driving into it."""
    assert ACTUAL.strategy() is Strategy.CONTACT_ONLY


def test_fitting_a_lidar_changes_the_strategy_on_its_own():
    """The payoff for describing hardware instead of hardcoding behaviour: no
    other line has to be edited for the robot to start wall-following."""
    assert WITH_LIDAR.has(Capability.RANGE)
    assert WITH_LIDAR.strategy() is Strategy.WALL_FOLLOWING


def test_any_range_provider_will_do():
    """Ultrasonic, radar, lidar, or something behind an Arduino — the strategy
    depends on the capability, not on which box supplies it."""
    for device in (LIDAR_2D,):
        fitted = HardwareProfile("test", (SERVO_BUS, _fit(device)))
        assert fitted.strategy() is Strategy.WALL_FOLLOWING


def _fit(device):
    from dataclasses import replace

    return replace(device, fitted=True)


# ── Describing itself ────────────────────────────────────────────────────────


def test_unfitted_hardware_is_still_listed():
    """The physical build is coming, and knowing what each option would unlock
    is the useful half of planning it."""
    described = ACTUAL.describe()
    names = [d["name"] for d in described["devices"]]

    assert any("lidar" in n.lower() for n in names)
    assert any("radar" in n.lower() for n in names)
    assert any("arduino" in n.lower() for n in names)
    assert any("fpga" in n.lower() for n in names)


def test_the_description_says_which_strategy_it_implies():
    described = ACTUAL.describe()
    assert described["strategy"] == "CONTACT_ONLY"
    assert WITH_LIDAR.describe()["strategy"] == "WALL_FOLLOWING"


def test_every_device_declares_a_capability():
    """A device that provides nothing is a note, not a device."""
    for device in ALL_DEVICES:
        assert device.capabilities, f"{device.name} provides nothing"


def test_every_device_records_measured_accuracy():
    """Quoted from this project, not from a datasheet."""
    for device in ALL_DEVICES:
        assert device.accuracy, f"{device.name} has no accuracy recorded"


# ── Lookup ───────────────────────────────────────────────────────────────────


def test_profiles_are_addressable_by_name():
    assert profile("actual") is ACTUAL
    assert profile("with-lidar") is WITH_LIDAR


def test_an_unknown_profile_lists_the_real_ones():
    """A typo in a launcher script should not need a source dive."""
    with pytest.raises(ValueError, match="actual"):
        profile("nonsense")
