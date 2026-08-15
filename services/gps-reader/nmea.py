"""NMEA 0183 parsing for a USB GNSS receiver.

Why this moved to the PC
------------------------
The ESP32 firmware used to parse NMEA on the robot. With no microcontroller
there is nothing on the robot to do it, so the receiver plugs into the PC as a
USB serial device and the parsing happens here. The output is the same
`GpsData` the accuracy gate in `localization/fusion.py` already expects, so
nothing downstream changes.

What matters most in these sentences
------------------------------------
Not the latitude and longitude — those are always present and always look
plausible, even indoors where they are meaningless. The fields that decide
whether a fix can be trusted are **fix quality**, **satellite count** and
**HDOP**, and they only appear in specific sentence types. A parser that reads
only RMC (the most commonly quoted sentence) throws away exactly the
information this project depends on, which is why GGA is parsed here and GSA
is used to fill in HDOP when GGA's field is blank.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "shared", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robotmap_common.models import GpsData, GpsFixQuality


class NmeaError(ValueError):
    """A sentence that could not be parsed."""


@dataclass
class NmeaState:
    """Accumulated state across sentences.

    A single GNSS fix is spread across several sentence types arriving in a
    burst, so fields are merged rather than replaced: GGA carries quality and
    satellite count, GSA carries the dilution figures, RMC carries speed and
    course. Treating any one sentence as the whole fix loses information.
    """

    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    fix_quality: GpsFixQuality = GpsFixQuality.NO_FIX
    satellites: int = 0
    hdop: float = 99.9
    speed_mps: float | None = None
    course_deg: float | None = None

    sentences_seen: int = 0
    sentences_bad: int = 0
    checksum_failures: int = 0

    def to_gps_data(self) -> GpsData | None:
        """Return a GpsData, or None if no position has been seen yet."""
        if self.latitude is None or self.longitude is None:
            return None
        return GpsData(
            latitude=self.latitude,
            longitude=self.longitude,
            altitude_m=self.altitude_m,
            fix_quality=self.fix_quality,
            satellites=self.satellites,
            hdop=self.hdop,
            speed_mps=self.speed_mps,
            course_deg=self.course_deg,
        )


def verify_checksum(sentence: str) -> bool:
    """Check the NMEA XOR checksum.

    Worth doing rather than trusting the line. A partially received sentence
    parses into a perfectly well-formed position that is simply wrong, and a
    wrong position that looks valid is the failure mode this whole project is
    built to avoid.
    """
    if "*" not in sentence:
        return False

    body, _, checksum_text = sentence.partition("*")
    body = body.lstrip("$")

    try:
        expected = int(checksum_text[:2], 16)
    except (ValueError, IndexError):
        return False

    actual = 0
    for character in body:
        actual ^= ord(character)
    return actual == expected


def nmea_to_decimal(value: str, hemisphere: str) -> float:
    """Convert ddmm.mmmm to decimal degrees.

    NMEA packs degrees and minutes into one number, so 4523.4567 means
    45 degrees 23.4567 minutes. Reading it as a plain decimal — which it very
    much looks like — puts the robot hundreds of kilometres away.
    """
    if not value:
        raise NmeaError("empty coordinate")

    raw = float(value)
    degrees = int(raw / 100)
    minutes = raw - degrees * 100
    result = degrees + minutes / 60.0

    if hemisphere in ("S", "W"):
        result = -result
    return result


def parse_gga(fields: list[str], state: NmeaState) -> None:
    """Global Positioning System Fix Data.

    The important one: carries fix quality, satellite count and HDOP.
    """
    if len(fields) < 9:
        raise NmeaError("GGA too short")

    try:
        quality_code = int(fields[6]) if fields[6] else 0
    except ValueError:
        quality_code = 0

    state.fix_quality = GpsFixQuality.from_nmea(quality_code)

    try:
        state.satellites = int(fields[7]) if fields[7] else 0
    except ValueError:
        state.satellites = 0

    if quality_code == 0:
        # No fix. Leave the last known position alone but make sure the
        # quality fields say so, otherwise a stale position would keep being
        # treated as current.
        return

    if fields[2] and fields[4]:
        state.latitude = nmea_to_decimal(fields[2], fields[3])
        state.longitude = nmea_to_decimal(fields[4], fields[5])

    if fields[8]:
        try:
            state.hdop = float(fields[8])
        except ValueError:
            pass

    if len(fields) > 9 and fields[9]:
        try:
            state.altitude_m = float(fields[9])
        except ValueError:
            pass


def parse_gsa(fields: list[str], state: NmeaState) -> None:
    """Fills in HDOP when GGA leaves it blank, as some receivers do."""
    if len(fields) < 17:
        return
    if fields[16]:
        try:
            state.hdop = float(fields[16])
        except ValueError:
            pass


def parse_rmc(fields: list[str], state: NmeaState) -> None:
    """Recommended Minimum: position, speed and course.

    Note its status field is only 'A' or 'V' — valid or not. It gives no
    satellite count and no HDOP, so a receiver indoors emits RMC sentences
    marked valid that are wrong by tens of metres. That is why GGA drives the
    quality assessment and RMC only contributes speed and course.
    """
    if len(fields) < 8:
        raise NmeaError("RMC too short")

    if fields[2] != "A":
        return

    if fields[3] and fields[5]:
        state.latitude = nmea_to_decimal(fields[3], fields[4])
        state.longitude = nmea_to_decimal(fields[5], fields[6])

    if fields[7]:
        try:
            # Knots to metres per second.
            state.speed_mps = float(fields[7]) * 0.514444
        except ValueError:
            pass

    if len(fields) > 8 and fields[8]:
        try:
            state.course_deg = float(fields[8]) % 360.0
        except ValueError:
            pass


# Talker IDs vary by constellation — GP for GPS, GN for multi-GNSS, GL for
# GLONASS, GA for Galileo, BD/GB for BeiDou. Matching on the last three
# characters accepts all of them.
_PARSERS = {
    "GGA": parse_gga,
    "GSA": parse_gsa,
    "RMC": parse_rmc,
}


def parse_sentence(sentence: str, state: NmeaState) -> bool:
    """Fold one NMEA sentence into the state. Returns True if it was used."""
    sentence = sentence.strip()
    state.sentences_seen += 1

    if not sentence.startswith("$"):
        state.sentences_bad += 1
        return False

    if "*" in sentence and not verify_checksum(sentence):
        state.checksum_failures += 1
        state.sentences_bad += 1
        return False

    body = sentence.split("*")[0]
    fields = body.split(",")
    if not fields:
        state.sentences_bad += 1
        return False

    sentence_type = fields[0][-3:].upper()
    parser = _PARSERS.get(sentence_type)
    if parser is None:
        return False

    try:
        parser(fields, state)
    except (NmeaError, ValueError, IndexError):
        state.sentences_bad += 1
        return False

    return True
