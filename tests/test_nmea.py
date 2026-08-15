"""NMEA parsing tests, using real sentence formats.

No receiver needed. These matter because a partially-parsed sentence yields a
position that looks entirely valid and is simply wrong — the failure this
project's GPS gate exists to catch, arriving from the wrong direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "gps-reader"))

from nmea import (  # noqa: E402
    NmeaState,
    nmea_to_decimal,
    parse_sentence,
    verify_checksum,
)
from robotmap_common.models import GpsFixQuality  # noqa: E402


def _checksummed(body: str) -> str:
    """Append a correct NMEA checksum to a sentence body."""
    checksum = 0
    for character in body.lstrip("$"):
        checksum ^= ord(character)
    return f"{body}*{checksum:02X}"


# ── Checksum ─────────────────────────────────────────────────────────────────


def test_valid_checksum_accepted():
    assert verify_checksum(_checksummed("$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M"))


def test_corrupted_sentence_rejected():
    """A truncated or garbled sentence parses into a perfectly plausible but
    wrong position, so it must be rejected before parsing."""
    good = _checksummed("$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M")
    corrupted = good.replace("4807.038", "4907.038")
    assert not verify_checksum(corrupted)


def test_missing_checksum_rejected():
    assert not verify_checksum("$GPGGA,123519,4807.038,N")


def test_malformed_checksum_rejected():
    assert not verify_checksum("$GPGGA,123519*ZZ")


# ── Coordinate conversion ────────────────────────────────────────────────────


def test_ddmm_is_not_read_as_decimal_degrees():
    """4807.038 means 48 degrees 7.038 minutes, not 4807 degrees.

    Reading it as a plain decimal is the classic NMEA mistake and puts the
    robot hundreds of kilometres from where it is.
    """
    result = nmea_to_decimal("4807.038", "N")
    assert result == pytest.approx(48.1173, abs=0.0001)
    assert result != pytest.approx(4807.038)


def test_southern_and_western_hemispheres_are_negative():
    assert nmea_to_decimal("4807.038", "S") == pytest.approx(-48.1173, abs=0.0001)
    assert nmea_to_decimal("01131.000", "W") == pytest.approx(-11.5167, abs=0.0001)


def test_utp_coordinates_round_trip():
    """UTP sits at roughly 4.3852 N, 100.9739 E."""
    latitude = nmea_to_decimal("0423.112", "N")
    longitude = nmea_to_decimal("10058.434", "E")
    assert latitude == pytest.approx(4.3852, abs=0.001)
    assert longitude == pytest.approx(100.9739, abs=0.001)


def test_empty_coordinate_rejected():
    with pytest.raises(ValueError):
        nmea_to_decimal("", "N")


# ── GGA ──────────────────────────────────────────────────────────────────────


def test_gga_extracts_the_quality_fields():
    """The fields that decide whether the fix is usable at all."""
    state = NmeaState()
    sentence = _checksummed(
        "$GPGGA,123519,0423.112,N,10058.434,E,1,09,0.9,45.4,M,46.9,M,,"
    )
    assert parse_sentence(sentence, state)

    assert state.fix_quality == GpsFixQuality.GPS
    assert state.satellites == 9
    assert state.hdop == pytest.approx(0.9)
    assert state.altitude_m == pytest.approx(45.4)


def test_gga_with_no_fix_does_not_invent_a_position():
    """A receiver indoors emits GGA with quality 0. It must not be read as a
    position at the equator."""
    state = NmeaState()
    sentence = _checksummed("$GPGGA,123519,,,,,0,00,99.9,,M,,M,,")
    parse_sentence(sentence, state)

    assert state.fix_quality == GpsFixQuality.NO_FIX
    assert state.satellites == 0
    assert state.to_gps_data() is None


def test_no_fix_marks_a_previous_position_untrustworthy():
    """Losing the fix must not leave the last good position looking current.

    This is the indoor case: the robot drives inside, the receiver stops
    fixing, and a stale position would otherwise keep being treated as live.
    """
    state = NmeaState()
    parse_sentence(
        _checksummed("$GPGGA,123519,0423.112,N,10058.434,E,1,09,0.9,45.4,M,,,,"),
        state,
    )
    assert state.to_gps_data().is_usable_for_position

    parse_sentence(_checksummed("$GPGGA,123619,,,,,0,00,99.9,,M,,M,,"), state)
    assert state.fix_quality == GpsFixQuality.NO_FIX
    assert not state.to_gps_data().is_usable_for_position


@pytest.mark.parametrize(
    "code,expected",
    [
        (0, GpsFixQuality.NO_FIX),
        (1, GpsFixQuality.GPS),
        (2, GpsFixQuality.DGPS),
        (4, GpsFixQuality.RTK_FIXED),
        (5, GpsFixQuality.RTK_FLOAT),
    ],
)
def test_every_fix_quality_code_maps(code, expected):
    state = NmeaState()
    sentence = _checksummed(
        f"$GPGGA,123519,0423.112,N,10058.434,E,{code},09,0.9,45.4,M,,,,"
    )
    parse_sentence(sentence, state)
    assert state.fix_quality == expected


def test_indoor_fix_fails_the_usability_gate():
    """Few satellites and high HDOP is what a receiver reports indoors — a
    position that exists and is meaningless."""
    state = NmeaState()
    sentence = _checksummed(
        "$GPGGA,123519,0423.112,N,10058.434,E,1,03,12.5,45.4,M,,,,"
    )
    parse_sentence(sentence, state)

    gps = state.to_gps_data()
    assert gps.is_usable_for_position is False
    assert gps.estimated_accuracy_m > 20.0


# ── GSA and RMC ──────────────────────────────────────────────────────────────


def test_gsa_supplies_hdop_when_gga_omits_it():
    """Some receivers leave the GGA HDOP field blank."""
    state = NmeaState()
    parse_sentence(
        _checksummed("$GPGGA,123519,0423.112,N,10058.434,E,1,09,,45.4,M,,,,"), state
    )
    assert state.hdop == 99.9  # untouched by GGA

    parse_sentence(
        _checksummed("$GPGSA,A,3,04,05,,09,12,,,24,,,,,2.5,1.3,2.1"), state
    )
    assert state.hdop == pytest.approx(1.3)


def test_rmc_supplies_speed_and_course():
    state = NmeaState()
    sentence = _checksummed(
        "$GPRMC,123519,A,0423.112,N,10058.434,E,022.4,084.4,230394,003.1,W"
    )
    parse_sentence(sentence, state)

    assert state.speed_mps == pytest.approx(22.4 * 0.514444, abs=0.01)
    assert state.course_deg == pytest.approx(84.4)


def test_invalid_rmc_is_ignored():
    state = NmeaState()
    sentence = _checksummed("$GPRMC,123519,V,,,,,,,230394,,")
    parse_sentence(sentence, state)
    assert state.latitude is None


def test_rmc_alone_cannot_establish_trust():
    """RMC says 'valid' with no satellite count or HDOP, so a position from it
    alone must still fail the gate. This is why GGA drives quality."""
    state = NmeaState()
    parse_sentence(
        _checksummed(
            "$GPRMC,123519,A,0423.112,N,10058.434,E,000.0,084.4,230394,003.1,W"
        ),
        state,
    )
    gps = state.to_gps_data()
    assert gps is not None
    assert gps.is_usable_for_position is False


# ── Talker IDs and robustness ────────────────────────────────────────────────


@pytest.mark.parametrize("talker", ["GP", "GN", "GL", "GA", "GB"])
def test_all_constellation_talker_ids_accepted(talker):
    """Multi-GNSS receivers emit GN; matching only GP would ignore them."""
    state = NmeaState()
    sentence = _checksummed(
        f"${talker}GGA,123519,0423.112,N,10058.434,E,1,09,0.9,45.4,M,,,,"
    )
    assert parse_sentence(sentence, state)
    assert state.satellites == 9


def test_unknown_sentence_types_are_skipped_quietly():
    state = NmeaState()
    assert not parse_sentence(_checksummed("$GPVTG,054.7,T,034.4,M,005.5,N"), state)
    assert state.sentences_bad == 0  # skipped, not counted as an error


def test_garbage_does_not_raise():
    """Serial lines get corrupted; the reader must not die on one."""
    state = NmeaState()
    for junk in ("", "not a sentence", "$", "$GPGGA", "\x00\xff garbage"):
        parse_sentence(junk, state)
    assert state.latitude is None


def test_truncated_gga_is_counted_as_bad():
    state = NmeaState()
    parse_sentence(_checksummed("$GPGGA,123519,0423.112"), state)
    assert state.sentences_bad == 1


def test_state_tracks_checksum_failures():
    """A rising failure count usually means the wrong baud rate."""
    state = NmeaState()
    good = _checksummed("$GPGGA,123519,0423.112,N,10058.434,E,1,09,0.9,45.4,M,,,,")
    parse_sentence(good.replace("0423", "0424"), state)
    assert state.checksum_failures == 1


# ── Whole-burst behaviour ────────────────────────────────────────────────────


def test_fields_merge_across_a_sentence_burst():
    """One fix is spread over several sentence types; treating any single one
    as the whole fix loses information."""
    state = NmeaState()
    for sentence in (
        "$GPGGA,123519,0423.112,N,10058.434,E,1,09,,45.4,M,,,,",
        "$GPGSA,A,3,04,05,,09,12,,,24,,,,,2.5,0.9,2.1",
        "$GPRMC,123519,A,0423.112,N,10058.434,E,002.4,084.4,230394,003.1,W",
    ):
        parse_sentence(_checksummed(sentence), state)

    gps = state.to_gps_data()
    assert gps.satellites == 9          # from GGA
    assert gps.hdop == pytest.approx(0.9)  # from GSA
    assert gps.speed_mps is not None    # from RMC
    assert gps.is_usable_for_position is True
