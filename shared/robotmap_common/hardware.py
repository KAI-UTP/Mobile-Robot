"""What sensors this robot actually has, and what that means it can do.

Why this exists
---------------
The robot is a three-wheel holonomic base with Feetech STS3215 servos, a GNSS
receiver and Bluetooth. No lidar, no ultrasonic ring, no bumper. That single
fact decides more of this codebase than anything else: it rules out the entire
family of mapping strategies that hold a *measured distance* to a wall, and it
means a collision has to be inferred from what the servos report rather than
read from a switch.

Until now that fact was scattered — a `contact_only=True` default in the pilot,
a hardcoded `WallFollower` in the mapper, comments in half a dozen files
explaining that the range readings the simulator produces do not exist on the
real robot. Adding a lidar later would have meant finding every one of them.

So the hardware is described once, here, and the code asks. `strategy()` is the
payoff: it is the actual decision the stack makes, derived from what is fitted
rather than hardcoded, which is what makes the rest of this a description of
reality instead of documentation that can rot.

Adding hardware
---------------
Set `fitted=True` on a device below — or build a profile with it — and the
stack changes behaviour without another line being edited. The devices that are
not fitted are listed anyway, because the physical build is coming and knowing
exactly what each one would unlock is the useful half of planning it.

Honesty rule
------------
`accuracy` is measured on this project, not quoted from a datasheet. Where
something was measured and turned out worse than the alternative it says so,
and `fitted` is False. BLE is the example: it is physically present and wired
up, and it is switched off because it made the pose worse (0.51 m -> 1.42 m).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Capability(str, Enum):
    """What a device tells the robot. Strategies are chosen from these."""

    ODOMETRY = "ODOMETRY"      # how far the wheels turned
    CONTACT = "CONTACT"        # it hit something
    RANGE = "RANGE"            # distance to a surface, without touching it
    POSITION = "POSITION"      # an absolute fix, in some frame
    HEADING = "HEADING"        # absolute heading
    ORIENTATION = "ORIENTATION"  # roll/pitch/yaw rates


class Strategy(str, Enum):
    """How the room gets mapped, given what is fitted."""

    WALL_FOLLOWING = "WALL_FOLLOWING"
    CONTACT_ONLY = "CONTACT_ONLY"


@dataclass(frozen=True)
class Device:
    """One piece of hardware, fitted or not.

    `fitted` is the only field the code branches on. The rest is for the human
    deciding what to buy next, and for the dashboard that shows what this robot
    is actually running on.
    """

    name: str
    capabilities: frozenset[Capability]
    fitted: bool
    interface: str = ""
    accuracy: str = ""
    notes: str = ""

    def provides(self, capability: Capability) -> bool:
        return self.fitted and capability in self.capabilities


# ── What is on the robot today ───────────────────────────────────────────────

SERVO_BUS = Device(
    name="Feetech STS3215 servo bus",
    capabilities=frozenset({Capability.ODOMETRY, Capability.CONTACT}),
    fitted=True,
    interface="USB-C serial, half-duplex TTL",
    accuracy="0.51 m of drift over 200 m driven, measured",
    notes=(
        "Provides CONTACT without a bumper. A wheel delivering a fraction of "
        "its commanded speed at high load has something solid in front of it, "
        "which is the only collision evidence this robot produces. See "
        "robotmap_common.collision."
    ),
)

GNSS = Device(
    name="GNSS receiver",
    capabilities=frozenset({Capability.POSITION}),
    fitted=True,
    interface="USB serial, NMEA",
    accuracy="2.5 m CEP outdoors; no usable fix indoors",
    notes=(
        "Anchors the map to the world outdoors. Indoors it is the reason the "
        "pose filter has to survive with no absolute reference at all, which "
        "is the case that actually matters for room mapping."
    ),
)

BLE_BEACONS = Device(
    name="BLE beacons (RSSI trilateration)",
    capabilities=frozenset({Capability.POSITION}),
    fitted=False,
    interface="Bluetooth LE, 4 fixed beacons",
    accuracy="2.71 m mean error, measured",
    notes=(
        "Present and wired up, deliberately switched off. Folding it into the "
        "filter made the pose WORSE — 0.51 m -> 1.42 m — because correlated "
        "shadowing was being treated as independent noise, so 6812 corrections "
        "at 10 Hz shrank the covariance until the filter believed sigma=0.18 m "
        "while sitting 1.42 m from the truth. Set fitted=True to re-enable and "
        "watch the same thing happen. See localization/fusion.py."
    ),
)

# ── Not fitted: what the physical build could add ────────────────────────────

LIDAR_2D = Device(
    name="2D lidar (e.g. RPLIDAR A1)",
    capabilities=frozenset({Capability.RANGE}),
    fitted=False,
    interface="USB serial",
    accuracy="~0.02 m at 6 m, typical for the class",
    notes=(
        "The single biggest upgrade available. RANGE switches the stack from "
        "CONTACT_ONLY to WALL_FOLLOWING, which maps a room in about 23 m of "
        "driving instead of 435 m, and measures it to 1.4 % instead of 7.9 %."
    ),
)

ULTRASONIC_RING = Device(
    name="Ultrasonic ring (HC-SR04 x N)",
    capabilities=frozenset({Capability.RANGE}),
    fitted=False,
    interface="GPIO, via a microcontroller",
    accuracy="~0.03 m, but a +-15 deg cone and blind to soft furnishings",
    notes=(
        "The cheap route to RANGE. The wide cone is why the wall-follower "
        "needs contact recovery even when range sensors are fitted: a low bin "
        "sits entirely inside the cone's blind spot."
    ),
)

RADAR = Device(
    name="mmWave radar",
    capabilities=frozenset({Capability.RANGE}),
    fitted=False,
    interface="UART or SPI",
    accuracy="coarse in angle, excellent in range",
    notes=(
        "Sees through smoke and dust and does not care about surface colour, "
        "which ultrasonic and lidar both do. Poor angular resolution makes it "
        "a poor sole mapper but a good obstacle backstop."
    ),
)

IMU = Device(
    name="IMU (accelerometer + gyro)",
    capabilities=frozenset({Capability.ORIENTATION, Capability.HEADING}),
    fitted=False,
    interface="I2C",
    accuracy="heading drift of a few degrees per minute, uncorrected",
    notes=(
        "The pose filter already accepts IMU input and the simulator already "
        "produces it. Fitting one mainly improves heading during the turns, "
        "which is where wheel odometry is weakest."
    ),
)

ARDUINO_BRIDGE = Device(
    name="Arduino sensor bridge",
    capabilities=frozenset({Capability.RANGE, Capability.CONTACT}),
    fitted=False,
    interface="USB serial, line protocol",
    accuracy="depends entirely on what is wired to it",
    notes=(
        "Not a sensor — a way to attach sensors that need hard real-time "
        "pin handling, which a Windows PC cannot do. Ultrasonic triggering "
        "and bumper switches belong behind this."
    ),
)

FPGA = Device(
    name="FPGA quadrature/encoder front end",
    capabilities=frozenset({Capability.ODOMETRY}),
    fitted=False,
    interface="USB or PCIe",
    accuracy="loses no counts at any speed",
    notes=(
        "Worth it only if the servo bus turns out to drop counts under load. "
        "Measure that first: the current 0.51 m over 200 m is 0.26 %, and an "
        "FPGA cannot improve on error that comes from wheel slip rather than "
        "from counting."
    ),
)


ALL_DEVICES: tuple[Device, ...] = (
    SERVO_BUS, GNSS, BLE_BEACONS,
    LIDAR_2D, ULTRASONIC_RING, RADAR, IMU, ARDUINO_BRIDGE, FPGA,
)


# ── Profiles ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HardwareProfile:
    """A named set of devices, and what the stack can do with them."""

    name: str
    devices: tuple[Device, ...]

    def has(self, capability: Capability) -> bool:
        return any(d.provides(capability) for d in self.devices)

    def providers(self, capability: Capability) -> tuple[Device, ...]:
        return tuple(d for d in self.devices if d.provides(capability))

    @property
    def fitted(self) -> tuple[Device, ...]:
        return tuple(d for d in self.devices if d.fitted)

    def strategy(self) -> Strategy:
        """How to map a room with this hardware.

        The one decision that has to follow from what is fitted rather than
        from a flag someone set. Wall-following holds a measured distance to a
        wall; with no RANGE there is nothing to measure, so the robot has to
        find the room by driving into it.
        """
        if self.has(Capability.RANGE):
            return Strategy.WALL_FOLLOWING
        return Strategy.CONTACT_ONLY

    def describe(self) -> dict:
        """For /api/hardware, so the dashboards can show what this is."""
        return {
            "profile": self.name,
            "strategy": self.strategy().value,
            "devices": [
                {
                    "name": d.name,
                    "fitted": d.fitted,
                    "capabilities": sorted(c.value for c in d.capabilities),
                    "interface": d.interface,
                    "accuracy": d.accuracy,
                    "notes": d.notes,
                }
                for d in self.devices
            ],
        }


#: The robot as it stands: servos, GNSS, Bluetooth. No range sensing.
ACTUAL = HardwareProfile("actual", ALL_DEVICES)

#: The same robot with a lidar bolted on — the first upgrade worth costing.
WITH_LIDAR = HardwareProfile(
    "with-lidar",
    tuple(
        Device(**{**d.__dict__, "fitted": True}) if d is LIDAR_2D else d
        for d in ALL_DEVICES
    ),
)

#: What the simulator has always pretended to be: a robot that can see.
#: Useful for comparing the two strategies on the same room, and it is what
#: the demo runs, so the difference is never hidden.
SIMULATED = WITH_LIDAR

PROFILES: dict[str, HardwareProfile] = {
    "actual": ACTUAL,
    "with-lidar": WITH_LIDAR,
    "simulated": SIMULATED,
}


def profile(name: str) -> HardwareProfile:
    """Look up a profile by name, with a message that lists the options."""
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown hardware profile {name!r}; try one of {known}") from None
