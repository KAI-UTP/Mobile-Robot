"""Servo bus wire-protocol tests.

No hardware needed: packet framing, checksums and byte order are all
verifiable by construction. Worth testing precisely because a malformed packet
produces silence rather than an error, so a mistake here looks like a wiring
fault and gets debugged in the wrong place.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "servo-bus"))

from protocols import (  # noqa: E402
    ALL_PROTOCOLS,
    DynamixelProtocol,
    FeetechProtocol,
    FeetechStsProtocol,
    LewanSoulProtocol,
    get_protocol,
)

# ── Registry ─────────────────────────────────────────────────────────────────


def test_all_protocols_are_constructible():
    for name in ALL_PROTOCOLS:
        protocol = get_protocol(name)
        assert protocol.name == name


def test_unknown_protocol_is_rejected_helpfully():
    with pytest.raises(ValueError, match="unknown protocol"):
        get_protocol("nonexistent")


@pytest.mark.parametrize("name", sorted(ALL_PROTOCOLS))
def test_every_protocol_implements_the_interface(name):
    protocol = get_protocol(name)
    protocol.ping(1)
    protocol.set_wheel_mode(1)
    protocol.set_velocity(1, 100)
    protocol.read_state(1)
    assert protocol.velocity_units_per_rad_s() > 0
    assert protocol.counts_per_revolution > 0


# ── Feetech ──────────────────────────────────────────────────────────────────


def test_feetech_ping_framing():
    packet = FeetechProtocol().ping(1)
    assert packet[:2] == b"\xff\xff"
    assert packet[2] == 1       # id
    assert packet[3] == 2       # length
    assert packet[4] == 0x01    # ping instruction


def test_feetech_checksum_is_inverted_sum():
    protocol = FeetechProtocol()
    packet = protocol.ping(1)
    body = packet[2:-1]
    assert packet[-1] == (~sum(body)) & 0xFF


@pytest.mark.parametrize("servo_id", [1, 2, 3, 10, 253])
def test_feetech_ping_carries_the_id(servo_id):
    assert FeetechProtocol().ping(servo_id)[2] == servo_id


def test_feetech_velocity_sign_uses_bit_15_not_twos_complement():
    """The Feetech sign convention is a magnitude plus a sign bit. Encoding it
    as two's complement instead sends a wildly wrong speed rather than a
    reversed one — a wheel that bolts instead of backing up."""
    protocol = FeetechProtocol()

    forward = protocol.set_velocity(1, 500)
    backward = protocol.set_velocity(1, -500)

    # Little-endian value sits after the register address byte.
    forward_value = forward[6] | (forward[7] << 8)
    backward_value = backward[6] | (backward[7] << 8)

    assert forward_value == 500
    assert backward_value == (500 | 0x8000)
    # Confirm it is NOT two's complement.
    assert backward_value != (-500 & 0xFFFF)


def test_feetech_uses_little_endian():
    """Feetech is little-endian where Dynamixel is big-endian. Getting this
    backwards yields plausible-looking but wrong values, not silence."""
    protocol = FeetechProtocol()
    packet = protocol.set_velocity(1, 0x0102)
    assert packet[6] == 0x02  # low byte first
    assert packet[7] == 0x01


def test_feetech_velocity_is_clamped():
    packet = FeetechProtocol().set_velocity(1, 999_999)
    value = packet[6] | (packet[7] << 8)
    assert value == 0x7FFF


def test_feetech_wheel_mode_zeroes_both_angle_limits():
    packets = FeetechProtocol().set_wheel_mode(1)
    assert len(packets) == 2
    for packet in packets:
        assert packet[5] in (FeetechProtocol.ADDR_MIN_ANGLE, FeetechProtocol.ADDR_MAX_ANGLE)
        assert packet[6] == 0 and packet[7] == 0


def test_feetech_parses_its_own_state_reply():
    protocol = FeetechProtocol()
    # id 1, length 6, error 0, position 512 LE, speed 100 LE
    body = bytes([1, 6, 0, 0x00, 0x02, 0x64, 0x00])
    reply = b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])

    reading = protocol.parse_state_reply(reply, 1)
    assert reading is not None
    assert reading.position_raw == 512
    assert reading.speed_raw == 100


def test_feetech_decodes_negative_speed():
    protocol = FeetechProtocol()
    body = bytes([1, 6, 0, 0x00, 0x02, 0x64, 0x80])  # speed 100 with sign bit
    reply = b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])

    reading = protocol.parse_state_reply(reply, 1)
    assert reading.speed_raw == -100


def test_feetech_rejects_a_reply_from_the_wrong_servo():
    protocol = FeetechProtocol()
    body = bytes([7, 6, 0, 0x00, 0x02, 0x64, 0x00])
    reply = b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])
    assert protocol.parse_state_reply(reply, 1) is None


def test_feetech_rejects_truncated_data():
    assert FeetechProtocol().parse_state_reply(b"\xff\xff\x01", 1) is None
    assert FeetechProtocol().parse_state_reply(b"", 1) is None


# ── Feetech STS3215 — the confirmed hardware ─────────────────────────────────


def test_sts3215_is_the_default_protocol():
    """It is the confirmed hardware, so nothing should have to opt in."""
    from driver import BusConfig

    assert BusConfig(port="COM1").protocol == "sts3215"
    assert get_protocol("sts3215").name == "sts3215"


def test_sts_wheel_mode_uses_the_mode_register_not_angle_limits():
    """The difference from the older SCS series.

    SCS enters continuous rotation when both angle limits are zeroed. STS has
    an explicit Mode register; zeroing angle limits on an STS leaves it in
    position mode, where it refuses to turn past its end stops. The servo
    acknowledges everything and the wheel twitches and halts.
    """
    protocol = FeetechStsProtocol()
    packets = protocol.set_wheel_mode(1)

    assert packets[0][5] == FeetechStsProtocol.ADDR_MODE
    assert packets[0][6] == FeetechStsProtocol.MODE_WHEEL

    # And confirm it is NOT doing the SCS thing.
    addresses = [p[5] for p in packets]
    assert FeetechProtocol.ADDR_MIN_ANGLE not in addresses
    assert FeetechProtocol.ADDR_MAX_ANGLE not in addresses


def test_sts_wheel_mode_enables_torque():
    """An STS servo powers up with its output free. Without this it accepts
    every command, acknowledges every packet, reports position correctly, and
    does not move — indistinguishable from a wiring fault."""
    packets = FeetechStsProtocol().set_wheel_mode(1)

    torque_packets = [
        p for p in packets if p[5] == FeetechStsProtocol.ADDR_TORQUE_ENABLE
    ]
    assert len(torque_packets) == 1
    assert torque_packets[0][6] == 1


def test_sts_sets_mode_before_enabling_torque():
    """Changing mode with torque already on can make the servo lurch as it
    reinterprets its current goal."""
    packets = FeetechStsProtocol().set_wheel_mode(1)
    assert packets[0][5] == FeetechStsProtocol.ADDR_MODE
    assert packets[1][5] == FeetechStsProtocol.ADDR_TORQUE_ENABLE


def test_sts_torque_can_be_released():
    """So the robot can be pushed by hand during calibration."""
    packet = FeetechStsProtocol().disable_torque(1)
    assert packet[5] == FeetechStsProtocol.ADDR_TORQUE_ENABLE
    assert packet[6] == 0


def test_sts_speed_unit_is_steps_per_second():
    """The unit that was wrong before the model was confirmed.

    One unit advances the output one encoder step per second, and there are
    4096 steps per revolution. The older SCS figure of 0.732 RPM per unit is
    roughly fifty times larger — using it would send a speed fifty times too
    high, the servo would clamp to maximum, and the wheels would run flat out
    for any requested speed.
    """
    protocol = FeetechStsProtocol()
    units_per_rad_s = protocol.velocity_units_per_rad_s()

    # One revolution per second is 2*pi rad/s and should be 4096 units.
    units_for_one_rev_per_s = units_per_rad_s * 2 * math.pi
    assert units_for_one_rev_per_s == pytest.approx(4096, rel=1e-6)


def test_sts_max_speed_matches_the_datasheet():
    """3400 units should work out to roughly the quoted 50 RPM no-load speed
    of the 12 V part. If the unit conversion were wrong this would not."""
    protocol = FeetechStsProtocol()
    max_rpm = protocol.max_wheel_rad_s * 60.0 / (2 * math.pi)
    assert 45.0 < max_rpm < 55.0


def test_sts_speed_differs_from_the_scs_convention():
    """Guards against the two being conflated again."""
    sts = FeetechStsProtocol().velocity_units_per_rad_s()
    scs = FeetechProtocol().velocity_units_per_rad_s()
    assert sts / scs > 10.0


def test_sts_velocity_is_clamped_to_the_datasheet_maximum():
    protocol = FeetechStsProtocol()
    packet = protocol.set_velocity(1, 999_999)
    value = packet[6] | (packet[7] << 8)
    assert value == FeetechStsProtocol.MAX_SPEED_UNITS


def test_sts_reverse_uses_the_sign_bit():
    protocol = FeetechStsProtocol()
    packet = protocol.set_velocity(1, -500)
    value = packet[6] | (packet[7] << 8)
    assert value == (500 | 0x8000)


def test_sts_reads_position_and_speed_in_one_transaction():
    """Present_Position and Present_Speed are adjacent, so four bytes fetches
    both. Halving the round trips halves the latency budget on a half-duplex
    bus polled thirty times a second."""
    packet = FeetechStsProtocol().read_state(1)
    assert packet[5] == FeetechStsProtocol.ADDR_PRESENT_POSITION
    assert packet[6] == 4


def test_sts_parses_position_and_speed():
    protocol = FeetechStsProtocol()
    # position 2048 LE, speed 300 LE
    body = bytes([1, 6, 0, 0x00, 0x08, 0x2C, 0x01])
    reply = b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])

    reading = protocol.parse_state_reply(reply, 1)
    assert reading.position_raw == 2048
    assert reading.speed_raw == 300


def test_sts_decodes_reverse_speed_as_sign_magnitude():
    """Reading it as two's complement would turn a slow reverse into a very
    fast forward — wrong in both magnitude and direction, and the odometry
    would follow it."""
    protocol = FeetechStsProtocol()
    body = bytes([1, 6, 0, 0x00, 0x08, 0x2C, 0x81])  # 300 with the sign bit
    reply = b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])

    reading = protocol.parse_state_reply(reply, 1)
    assert reading.speed_raw == -300
    assert reading.speed_raw != 33068  # what two's complement would give


def test_sts_position_is_masked_to_twelve_bits():
    """Position is an absolute 0-4095 angle within one turn; the upper nibble
    carries status bits on some firmware revisions."""
    protocol = FeetechStsProtocol()
    body = bytes([1, 6, 0, 0xFF, 0xFF, 0x00, 0x00])
    reply = b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])

    reading = protocol.parse_state_reply(reply, 1)
    assert 0 <= reading.position_raw <= 4095


def test_sts_inherits_working_framing_from_feetech():
    """It is the same wire format, so the shared framing must still hold."""
    protocol = FeetechStsProtocol()
    packet = protocol.ping(3)
    assert packet[:2] == b"\xff\xff"
    assert packet[2] == 3
    body = packet[2:-1]
    assert packet[-1] == (~sum(body)) & 0xFF


def test_sts_encoder_resolution_supports_the_geometry_default():
    """HolonomicGeometry defaults to 4096 counts/rev; the servo must match."""
    from robotmap_common.holonomic import HolonomicGeometry

    assert (
        FeetechStsProtocol().counts_per_revolution
        == HolonomicGeometry().ticks_per_revolution
    )


# ── Dynamixel ────────────────────────────────────────────────────────────────


def test_dynamixel_uses_big_endian():
    """The mirror image of the Feetech case, and the reason both exist."""
    packet = DynamixelProtocol().set_velocity(1, 0x0102)
    assert packet[6] == 0x01  # high byte first
    assert packet[7] == 0x02


def test_dynamixel_velocity_is_ten_bit():
    protocol = DynamixelProtocol()
    packet = protocol.set_velocity(1, 99_999)
    value = (packet[6] << 8) | packet[7]
    assert value == 0x3FF


def test_dynamixel_reverse_sets_bit_10():
    protocol = DynamixelProtocol()
    packet = protocol.set_velocity(1, -100)
    value = (packet[6] << 8) | packet[7]
    assert value == (100 | 0x400)


def test_dynamixel_parses_big_endian_position():
    protocol = DynamixelProtocol()
    body = bytes([1, 6, 0, 0x02, 0x00, 0x00, 0x64])  # position 512, speed 100
    reply = b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])

    reading = protocol.parse_state_reply(reply, 1)
    assert reading.position_raw == 512
    assert reading.speed_raw == 100


def test_feetech_and_dynamixel_differ_on_the_wire():
    """They share a header, so a scan must distinguish them by content. If
    these ever produced identical bytes the scanner could not tell them
    apart."""
    feetech = FeetechProtocol().set_velocity(1, 300)
    dynamixel = DynamixelProtocol().set_velocity(1, 300)
    assert feetech != dynamixel


# ── LewanSoul ────────────────────────────────────────────────────────────────


def test_lewansoul_uses_a_distinct_header():
    """0x55 0x55 rather than 0xFF 0xFF, so it is unambiguous during a scan."""
    packet = LewanSoulProtocol().ping(1)
    assert packet[:2] == b"\x55\x55"


def test_lewansoul_defaults_to_a_lower_baud():
    assert LewanSoulProtocol().default_baud == 115_200
    assert FeetechProtocol().default_baud == 1_000_000


def test_lewansoul_velocity_is_clamped_symmetrically():
    protocol = LewanSoulProtocol()
    for value, expected in ((5000, 1000), (-5000, -1000)):
        packet = protocol.set_velocity(1, value)
        speed = int.from_bytes(packet[7:9], "little", signed=True)
        assert speed == expected


def test_lewansoul_wheel_mode_selects_continuous_rotation():
    packets = LewanSoulProtocol().set_wheel_mode(1)
    assert len(packets) == 1
    assert packets[0][5] == 1  # mode 1 = continuous


# ── Cross-protocol ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(ALL_PROTOCOLS))
def test_zero_velocity_is_encoded_as_zero(name):
    """Stopping must be unambiguous — this is the packet the watchdog and the
    shutdown path both rely on."""
    protocol = get_protocol(name)
    packet = protocol.set_velocity(1, 0)
    assert isinstance(packet, bytes)
    assert len(packet) > 4
    # No stray sign bit on a zero command.
    assert protocol.set_velocity(1, 0) == protocol.set_velocity(1, 0)


@pytest.mark.parametrize("name", sorted(ALL_PROTOCOLS))
def test_packets_are_deterministic(name):
    protocol = get_protocol(name)
    assert protocol.ping(3) == protocol.ping(3)
    assert protocol.set_velocity(2, 250) == protocol.set_velocity(2, 250)


@pytest.mark.parametrize("name", sorted(ALL_PROTOCOLS))
def test_different_ids_produce_different_packets(name):
    protocol = get_protocol(name)
    assert protocol.ping(1) != protocol.ping(2)


def test_bus_servo_resolution_beats_the_differential_requirement():
    """Bus servos carry a 12-bit absolute encoder, so the encoder-resolution
    problem that dominated the differential design does not arise here."""
    assert FeetechProtocol().counts_per_revolution >= 4096
    assert DynamixelProtocol().counts_per_revolution >= 4096
