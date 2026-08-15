"""Wire protocols for daisy-chained bus servos.

The situation this solves
-------------------------
The robot has no microcontroller. A servo bus board takes power from an
adapter, daisy-chains the three wheel servos, and presents itself to the PC as
a USB serial port. The PC sends velocity commands directly.

Three families dominate this market and they are physically interchangeable —
same three-pin daisy chain, same half-duplex TTL — but their packet formats are
incompatible. The brand has not been confirmed yet, so rather than guess, this
module implements all three behind one interface and `scan.py` works out which
one is actually answering.

Common ground between them
--------------------------
All three use a header, an ID byte, a length, an instruction, parameters and a
checksum. They differ in the header bytes, the checksum, the byte order, and
the register addresses. Those differences are exactly what makes a wrong guess
produce silence rather than a helpful error, which is why probing is worth
more than assuming.
"""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ServoReading:
    """Feedback from one servo. Fields absent on a family are left None."""

    servo_id: int
    position_raw: int | None = None
    speed_raw: int | None = None
    load_raw: int | None = None
    voltage_v: float | None = None
    temperature_c: float | None = None


class ServoProtocol(ABC):
    """One servo bus wire format."""

    name: str = "abstract"
    default_baud: int = 1_000_000
    # Encoder counts per full output revolution. Set the wheel geometry's
    # ticks_per_revolution to match whichever family is confirmed.
    counts_per_revolution: int = 4096

    @abstractmethod
    def ping(self, servo_id: int) -> bytes:
        """Packet asking a servo to identify itself."""

    @abstractmethod
    def parse_ping_reply(self, data: bytes, servo_id: int) -> bool:
        """True if `data` is a valid reply from `servo_id`."""

    @abstractmethod
    def set_wheel_mode(self, servo_id: int) -> list[bytes]:
        """Packets that put a servo into continuous rotation.

        A servo shipped for an arm joint is in position mode and will refuse to
        turn past its angle limits — the most common reason a newly wired wheel
        twitches and stops. Wheel mode is normally set by zeroing the angle
        limit registers.
        """

    @abstractmethod
    def set_velocity(self, servo_id: int, value: int) -> bytes:
        """Packet commanding a signed velocity, in raw units."""

    @abstractmethod
    def read_state(self, servo_id: int) -> bytes:
        """Packet requesting position and speed."""

    @abstractmethod
    def parse_state_reply(self, data: bytes, servo_id: int) -> ServoReading | None:
        """Decode a state reply, or None if malformed."""

    @abstractmethod
    def velocity_units_per_rad_s(self) -> float:
        """Raw velocity units per radian/second at the output shaft."""

    def expected_reply_length(self) -> int:
        """Bytes to wait for when reading a state reply."""
        return 16


# ── Feetech STS / SCS ────────────────────────────────────────────────────────


class FeetechProtocol(ServoProtocol):
    """Feetech STS3215 and relatives.

    Very common in the LeRobot SO-100/SO-101 arms and Waveshare's servo kits,
    and the most likely candidate for a cheap daisy-chained wheel setup. The
    framing is Dynamixel protocol 1.0's, but the multi-byte registers are
    LITTLE-endian where Dynamixel's are big-endian — a subtle difference that
    makes a wrong guess produce plausible-looking nonsense rather than silence.
    """

    name = "feetech"
    default_baud = 1_000_000
    counts_per_revolution = 4096

    HEADER = b"\xff\xff"
    INST_PING = 0x01
    INST_READ = 0x02
    INST_WRITE = 0x03

    ADDR_MIN_ANGLE = 9
    ADDR_MAX_ANGLE = 11
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56

    @staticmethod
    def _checksum(payload: bytes) -> int:
        """Inverted sum of everything after the header."""
        return (~sum(payload)) & 0xFF

    def _packet(self, servo_id: int, instruction: int, params: bytes = b"") -> bytes:
        length = len(params) + 2
        body = bytes([servo_id, length, instruction]) + params
        return self.HEADER + body + bytes([self._checksum(body)])

    def ping(self, servo_id: int) -> bytes:
        return self._packet(servo_id, self.INST_PING)

    def parse_ping_reply(self, data: bytes, servo_id: int) -> bool:
        idx = data.find(self.HEADER)
        if idx < 0 or len(data) < idx + 6:
            return False
        return data[idx + 2] == servo_id

    def set_wheel_mode(self, servo_id: int) -> list[bytes]:
        # Zeroing both angle limits switches the servo to continuous rotation.
        return [
            self._packet(
                servo_id,
                self.INST_WRITE,
                bytes([self.ADDR_MIN_ANGLE]) + struct.pack("<H", 0),
            ),
            self._packet(
                servo_id,
                self.INST_WRITE,
                bytes([self.ADDR_MAX_ANGLE]) + struct.pack("<H", 0),
            ),
        ]

    def set_velocity(self, servo_id: int, value: int) -> bytes:
        # Sign lives in bit 15, magnitude in the low 15 bits — not two's
        # complement, which is the usual trip hazard here.
        magnitude = min(abs(int(value)), 0x7FFF)
        encoded = magnitude | (0x8000 if value < 0 else 0x0000)
        return self._packet(
            servo_id,
            self.INST_WRITE,
            bytes([self.ADDR_GOAL_SPEED]) + struct.pack("<H", encoded),
        )

    def read_state(self, servo_id: int) -> bytes:
        return self._packet(
            servo_id,
            self.INST_READ,
            bytes([self.ADDR_PRESENT_POSITION, 4]),
        )

    def parse_state_reply(self, data: bytes, servo_id: int) -> ServoReading | None:
        idx = data.find(self.HEADER)
        if idx < 0 or len(data) < idx + 9:
            return None
        if data[idx + 2] != servo_id:
            return None

        params = data[idx + 5 : idx + 9]
        if len(params) < 4:
            return None

        position = struct.unpack("<H", params[0:2])[0]
        raw_speed = struct.unpack("<H", params[2:4])[0]
        speed = -(raw_speed & 0x7FFF) if raw_speed & 0x8000 else raw_speed

        return ServoReading(servo_id=servo_id, position_raw=position, speed_raw=speed)

    def velocity_units_per_rad_s(self) -> float:
        # One unit is approximately 0.732 RPM for the STS series.
        rpm_per_unit = 0.732
        rad_s_per_unit = rpm_per_unit * 2.0 * 3.141592653589793 / 60.0
        return 1.0 / rad_s_per_unit


# ── Dynamixel protocol 1.0 ───────────────────────────────────────────────────


class DynamixelProtocol(ServoProtocol):
    """ROBOTIS Dynamixel AX/MX, protocol 1.0.

    Same framing as Feetech but BIG-endian registers and a different register
    map. Better built and considerably more expensive; if the lab bought these
    the servos will say "DYNAMIXEL" on the case.
    """

    name = "dynamixel"
    default_baud = 1_000_000
    counts_per_revolution = 4096

    HEADER = b"\xff\xff"
    INST_PING = 0x01
    INST_READ = 0x02
    INST_WRITE = 0x03

    ADDR_CW_ANGLE_LIMIT = 6
    ADDR_CCW_ANGLE_LIMIT = 8
    ADDR_MOVING_SPEED = 32
    ADDR_PRESENT_POSITION = 36

    @staticmethod
    def _checksum(payload: bytes) -> int:
        return (~sum(payload)) & 0xFF

    def _packet(self, servo_id: int, instruction: int, params: bytes = b"") -> bytes:
        length = len(params) + 2
        body = bytes([servo_id, length, instruction]) + params
        return self.HEADER + body + bytes([self._checksum(body)])

    def ping(self, servo_id: int) -> bytes:
        return self._packet(servo_id, self.INST_PING)

    def parse_ping_reply(self, data: bytes, servo_id: int) -> bool:
        idx = data.find(self.HEADER)
        if idx < 0 or len(data) < idx + 6:
            return False
        return data[idx + 2] == servo_id

    def set_wheel_mode(self, servo_id: int) -> list[bytes]:
        return [
            self._packet(
                servo_id,
                self.INST_WRITE,
                bytes([self.ADDR_CW_ANGLE_LIMIT]) + struct.pack(">H", 0),
            ),
            self._packet(
                servo_id,
                self.INST_WRITE,
                bytes([self.ADDR_CCW_ANGLE_LIMIT]) + struct.pack(">H", 0),
            ),
        ]

    def set_velocity(self, servo_id: int, value: int) -> bytes:
        magnitude = min(abs(int(value)), 0x3FF)  # 10-bit magnitude
        encoded = magnitude | (0x400 if value < 0 else 0x000)
        return self._packet(
            servo_id,
            self.INST_WRITE,
            bytes([self.ADDR_MOVING_SPEED]) + struct.pack(">H", encoded),
        )

    def read_state(self, servo_id: int) -> bytes:
        return self._packet(
            servo_id, self.INST_READ, bytes([self.ADDR_PRESENT_POSITION, 4])
        )

    def parse_state_reply(self, data: bytes, servo_id: int) -> ServoReading | None:
        idx = data.find(self.HEADER)
        if idx < 0 or len(data) < idx + 9:
            return None
        if data[idx + 2] != servo_id:
            return None

        params = data[idx + 5 : idx + 9]
        if len(params) < 4:
            return None

        position = struct.unpack(">H", params[0:2])[0]
        raw_speed = struct.unpack(">H", params[2:4])[0]
        speed = -(raw_speed & 0x3FF) if raw_speed & 0x400 else raw_speed

        return ServoReading(servo_id=servo_id, position_raw=position, speed_raw=speed)

    def velocity_units_per_rad_s(self) -> float:
        rpm_per_unit = 0.111  # AX-12 units
        rad_s_per_unit = rpm_per_unit * 2.0 * 3.141592653589793 / 60.0
        return 1.0 / rad_s_per_unit


# ── LewanSoul / HiWonder LX-16A ──────────────────────────────────────────────


class LewanSoulProtocol(ServoProtocol):
    """LewanSoul / HiWonder LX-16A.

    Cheapest of the three and common from Malaysian hobby suppliers. Uses a
    completely different header (0x55 0x55), a plain additive checksum, and a
    much lower default baud — so it is easy to distinguish from the other two
    once you probe for it.
    """

    name = "lewansoul"
    default_baud = 115_200
    counts_per_revolution = 1000  # 0-1000 over 240 degrees of travel

    HEADER = b"\x55\x55"
    CMD_SERVO_MOVE_TIME_WRITE = 1
    CMD_SERVO_OR_MOTOR_MODE_WRITE = 29
    CMD_SERVO_POS_READ = 28

    @staticmethod
    def _checksum(payload: bytes) -> int:
        return (~sum(payload)) & 0xFF

    def _packet(self, servo_id: int, command: int, params: bytes = b"") -> bytes:
        length = len(params) + 3
        body = bytes([servo_id, length, command]) + params
        return self.HEADER + body + bytes([self._checksum(body)])

    def ping(self, servo_id: int) -> bytes:
        return self._packet(servo_id, self.CMD_SERVO_POS_READ)

    def parse_ping_reply(self, data: bytes, servo_id: int) -> bool:
        idx = data.find(self.HEADER)
        if idx < 0 or len(data) < idx + 6:
            return False
        return data[idx + 2] == servo_id

    def set_wheel_mode(self, servo_id: int) -> list[bytes]:
        # mode 1 = continuous rotation, speed 0 to begin with.
        params = bytes([1, 0]) + struct.pack("<h", 0)
        return [self._packet(servo_id, self.CMD_SERVO_OR_MOTOR_MODE_WRITE, params)]

    def set_velocity(self, servo_id: int, value: int) -> bytes:
        speed = max(-1000, min(1000, int(value)))
        params = bytes([1, 0]) + struct.pack("<h", speed)
        return self._packet(servo_id, self.CMD_SERVO_OR_MOTOR_MODE_WRITE, params)

    def read_state(self, servo_id: int) -> bytes:
        return self._packet(servo_id, self.CMD_SERVO_POS_READ)

    def parse_state_reply(self, data: bytes, servo_id: int) -> ServoReading | None:
        idx = data.find(self.HEADER)
        if idx < 0 or len(data) < idx + 8:
            return None
        if data[idx + 2] != servo_id:
            return None
        position = struct.unpack("<h", data[idx + 5 : idx + 7])[0]
        return ServoReading(servo_id=servo_id, position_raw=position)

    def velocity_units_per_rad_s(self) -> float:
        # The LX-16A's continuous mode is open loop and only roughly
        # proportional; treat this as a starting point to calibrate, not a spec.
        return 1000.0 / (2.0 * 3.141592653589793 * 1.5)


# ── Feetech STS series — the confirmed hardware ──────────────────────────────


class FeetechStsProtocol(FeetechProtocol):
    """Feetech **STS3215** and the rest of the STS series. CONFIRMED HARDWARE.

    Shares its framing with the older SCS series above — same header, same
    checksum, same little-endian registers — but differs in three ways that
    each produce a distinct and confusing failure if the SCS behaviour is used
    instead. All three were wrong here until the model was confirmed.

    **1. Wheel mode lives in a Mode register, not the angle limits.**
    SCS servos enter continuous rotation when both angle limits are zeroed.
    STS servos have an explicit Mode register (33): 0 position, 1 wheel,
    2 open-loop PWM, 3 step. Zeroing the angle limits on an STS does not
    enable wheel mode, so the servo stays in position mode and simply refuses
    to turn past its end stops.

    **2. Speed is in steps per second, not SCS speed units.**
    One unit advances the output by one encoder step per second, and there are
    4096 steps per revolution. The SCS figure of 0.732 RPM per unit is roughly
    fifty times larger; using it would send a speed command fifty times too
    high, which the servo clamps to its maximum. The wheels would run flat out
    for any requested speed, and the robot would be uncontrollable in a way
    that looks mechanical rather than numerical.

    Sanity check against the datasheet: max speed is about 3400 units, and
    3400 / 4096 rev/s = 0.83 rev/s = 50 RPM, which matches the quoted no-load
    speed of the 12 V part.

    **3. Torque must be explicitly enabled.**
    An STS servo powers up with its output free. Without writing 1 to
    Torque_Enable (40) it accepts every command, acknowledges every packet,
    reports its position correctly, and does not move — indistinguishable from
    a wiring fault at a glance.

    The 12 V part is the higher-speed, higher-torque variant; there is also a
    7.4 V one. The protocol is identical, but the supply must match or the
    servo will either be sluggish or be damaged.
    """

    name = "sts3215"
    default_baud = 1_000_000
    counts_per_revolution = 4096

    ADDR_MODE = 33
    ADDR_TORQUE_ENABLE = 40
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56
    ADDR_PRESENT_SPEED = 58

    MODE_WHEEL = 1

    # Datasheet maximum for the 12 V part, in steps/second.
    MAX_SPEED_UNITS = 3400

    def set_wheel_mode(self, servo_id: int) -> list[bytes]:
        """Mode to wheel, then enable torque.

        Order matters: changing mode with torque already enabled can make the
        servo lurch as it reinterprets its current goal.
        """
        return [
            self._packet(
                servo_id,
                self.INST_WRITE,
                bytes([self.ADDR_MODE, self.MODE_WHEEL]),
            ),
            self._packet(
                servo_id,
                self.INST_WRITE,
                bytes([self.ADDR_TORQUE_ENABLE, 1]),
            ),
        ]

    def disable_torque(self, servo_id: int) -> bytes:
        """Free the output shaft, so the robot can be pushed by hand."""
        return self._packet(
            servo_id, self.INST_WRITE, bytes([self.ADDR_TORQUE_ENABLE, 0])
        )

    def set_velocity(self, servo_id: int, value: int) -> bytes:
        """Sign-magnitude speed in steps/second, clamped to the datasheet max."""
        magnitude = min(abs(int(value)), self.MAX_SPEED_UNITS)
        encoded = magnitude | (0x8000 if value < 0 else 0x0000)
        return self._packet(
            servo_id,
            self.INST_WRITE,
            bytes([self.ADDR_GOAL_SPEED]) + struct.pack("<H", encoded),
        )

    def velocity_units_per_rad_s(self) -> float:
        """Steps/second per radian/second.

        One revolution is 2*pi radians and 4096 steps, so a wheel turning at
        1 rad/s advances 4096 / (2*pi) = 651.9 steps each second.
        """
        return self.counts_per_revolution / (2.0 * 3.141592653589793)

    @property
    def max_wheel_rad_s(self) -> float:
        """Fastest the servo can turn, radians/second. Useful for limits."""
        return self.MAX_SPEED_UNITS / self.velocity_units_per_rad_s()

    def read_state(self, servo_id: int) -> bytes:
        """Read position and speed in one transaction.

        Present_Position (56) and Present_Speed (58) are adjacent, so four
        bytes from 56 fetches both. One round trip instead of two matters:
        at 10 Hz across three servos that is 30 exchanges a second on a
        half-duplex bus, and halving them halves the latency budget.
        """
        return self._packet(
            servo_id, self.INST_READ, bytes([self.ADDR_PRESENT_POSITION, 4])
        )

    def parse_state_reply(self, data: bytes, servo_id: int) -> ServoReading | None:
        """Decode position and speed.

        Both are sign-magnitude with the sign in bit 15, not two's complement.
        Reading speed as two's complement turns a slow reverse into a very
        fast forward — the reported motion would be wrong in both magnitude
        and direction, and the odometry would follow it.
        """
        idx = data.find(self.HEADER)
        if idx < 0 or len(data) < idx + 9:
            return None
        if data[idx + 2] != servo_id:
            return None

        params = data[idx + 5 : idx + 9]
        if len(params) < 4:
            return None

        raw_position = struct.unpack("<H", params[0:2])[0]
        # Position is an unsigned 0-4095 absolute angle within one turn.
        position = raw_position & 0x0FFF

        raw_speed = struct.unpack("<H", params[2:4])[0]
        speed = -(raw_speed & 0x7FFF) if raw_speed & 0x8000 else raw_speed

        return ServoReading(servo_id=servo_id, position_raw=position, speed_raw=speed)


# ── Registry ─────────────────────────────────────────────────────────────────

ALL_PROTOCOLS: dict[str, type[ServoProtocol]] = {
    # The confirmed hardware, and therefore the default everywhere.
    "sts3215": FeetechStsProtocol,
    "feetech": FeetechProtocol,  # older SCS series
    "dynamixel": DynamixelProtocol,
    "lewansoul": LewanSoulProtocol,
}

# Baud rates worth probing, most likely first.
COMMON_BAUDS = [1_000_000, 115_200, 500_000, 57_600, 250_000, 9_600]


def get_protocol(name: str) -> ServoProtocol:
    if name not in ALL_PROTOCOLS:
        raise ValueError(
            f"unknown protocol {name!r}; choose from {sorted(ALL_PROTOCOLS)}"
        )
    return ALL_PROTOCOLS[name]()
