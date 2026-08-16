"""Pydantic data models — single source of truth for every message schema.

Design notes
------------
The robot is a *differential drive* platform: two independently driven
wheels plus one passive caster.  That is what "3-wheel robot" almost always
means in practice, and it is the geometry every model here assumes.

Position is expressed in a **local map frame** in metres, with the origin at
wherever the robot was switched on.  This is deliberate: GPS is not accurate
enough to define the frame (see docs/LOCALIZATION.md), so the map frame is
odometry-defined and GPS is only used to *anchor* that frame to the world
when a trustworthy fix is available.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

# ── Enumerations ──────────────────────────────────────────────────────────────


class MotionState(str, Enum):
    STOPPED = "STOPPED"
    DRIVING = "DRIVING"
    TURNING = "TURNING"
    AVOIDING = "AVOIDING"


class MappingMode(str, Enum):
    IDLE = "IDLE"
    MAPPING = "MAPPING"  # actively exploring and building the grid
    LOCALIZING = "LOCALIZING"  # map is frozen, only tracking position
    RETURNING = "RETURNING"
    MANUAL = "MANUAL"  # operator teleoperation


class GpsFixQuality(str, Enum):
    """Maps to the NMEA GGA fix-quality field."""

    NO_FIX = "NO_FIX"  # 0
    GPS = "GPS"  # 1 — standard positioning
    DGPS = "DGPS"  # 2 — differential, sub-metre possible
    RTK_FIXED = "RTK_FIXED"  # 4 — centimetre grade
    RTK_FLOAT = "RTK_FLOAT"  # 5

    @classmethod
    def from_nmea(cls, value: int) -> GpsFixQuality:
        return {
            0: cls.NO_FIX,
            1: cls.GPS,
            2: cls.DGPS,
            4: cls.RTK_FIXED,
            5: cls.RTK_FLOAT,
        }.get(value, cls.NO_FIX)


class PoseSource(str, Enum):
    """Which sensors actually contributed to a fused pose."""

    ODOMETRY_ONLY = "ODOMETRY_ONLY"
    ODOMETRY_IMU = "ODOMETRY_IMU"
    ODOMETRY_IMU_GPS = "ODOMETRY_IMU_GPS"
    DEAD_RECKONING_DEGRADED = "DEAD_RECKONING_DEGRADED"


class DriveKind(str, Enum):
    """Which drive geometry produced a packet.

    The robot is a three-wheel holonomic (kiwi) base: three omni wheels at
    120 degrees, all driven. It can translate in any direction without
    turning first, which a differential drive cannot do, and its odometry
    maths is entirely different — see `holonomic.py` rather than the
    differential helpers in `geometry.py`.
    """

    DIFFERENTIAL = "DIFFERENTIAL"
    HOLONOMIC_3WHEEL = "HOLONOMIC_3WHEEL"


class LinkType(str, Enum):
    WIFI_MQTT = "WIFI_MQTT"
    BLUETOOTH_SERIAL = "BLUETOOTH_SERIAL"
    BLE_GATT = "BLE_GATT"
    SIMULATED = "SIMULATED"


class RobotCommand(str, Enum):
    START_MAPPING = "START_MAPPING"
    STOP_MAPPING = "STOP_MAPPING"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    RESET_ODOMETRY = "RESET_ODOMETRY"
    CLEAR_MAP = "CLEAR_MAP"
    SAVE_MAP = "SAVE_MAP"
    RETURN_HOME = "RETURN_HOME"
    SET_ANCHOR = "SET_ANCHOR"  # bind current pose to current GPS fix
    MANUAL_MODE = "MANUAL_MODE"
    AUTO_MODE = "AUTO_MODE"
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"


# ── Raw sensor sub-models ─────────────────────────────────────────────────────


class EncoderData(BaseModel):
    """Cumulative wheel-encoder counts.

    Cumulative rather than per-tick deltas so that a dropped packet costs
    accuracy for one interval instead of corrupting the pose permanently.
    """

    left_ticks: int
    right_ticks: int
    left_rpm: float = Field(default=0.0, ge=-500.0, le=500.0)
    right_rpm: float = Field(default=0.0, ge=-500.0, le=500.0)
    dt_ms: int = Field(..., gt=0, le=10_000, description="Interval since previous sample")


class ImuData(BaseModel):
    """Inertial data. `heading_deg` is the yaw estimate after fusion on-board."""

    heading_deg: float = Field(..., ge=0.0, lt=360.0)
    gyro_z_dps: float = Field(..., ge=-2000.0, le=2000.0)
    accel_x_ms2: float = Field(default=0.0, ge=-160.0, le=160.0)
    accel_y_ms2: float = Field(default=0.0, ge=-160.0, le=160.0)
    temperature_c: float | None = Field(default=None, ge=-40.0, le=125.0)
    calibrated: bool = Field(default=False)


class RangeReading(BaseModel):
    """One distance measurement from a range sensor.

    `angle_deg` is relative to the robot's forward axis, counter-clockwise
    positive.  A fixed forward-facing HC-SR04 always reports angle 0.
    """

    angle_deg: float = Field(..., ge=-180.0, le=180.0)
    distance_m: float = Field(..., ge=0.0, le=12.0)
    valid: bool = Field(default=True)
    sensor_id: str = Field(default="front")


class GpsData(BaseModel):
    """A GNSS fix as reported by the receiver.

    Nothing here is trusted for indoor positioning; the localization service
    gates it on `fix_quality`, `satellites` and `hdop`.
    """

    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    altitude_m: float | None = Field(default=None, ge=-500.0, le=10_000.0)
    fix_quality: GpsFixQuality = GpsFixQuality.NO_FIX
    satellites: int = Field(default=0, ge=0, le=64)
    hdop: float = Field(default=99.9, ge=0.0, le=99.9)
    speed_mps: float | None = Field(default=None, ge=0.0, le=100.0)
    course_deg: float | None = Field(default=None, ge=0.0, lt=360.0)

    @property
    def estimated_accuracy_m(self) -> float:
        """Rough horizontal accuracy estimate, in metres.

        HDOP scaled by the user-equivalent range error for the correction
        method in use. The spread across these tiers is the whole reason a
        room-mapping robot can use some GNSS receivers and not others:
        uncorrected consumer GPS lands around 4 m, which is comparable to the
        width of the room being measured, while RTK reaches centimetres.
        """
        uere_m = {
            GpsFixQuality.NO_FIX: 100.0,
            GpsFixQuality.GPS: 4.0,
            GpsFixQuality.DGPS: 1.0,
            GpsFixQuality.RTK_FLOAT: 0.3,
            GpsFixQuality.RTK_FIXED: 0.02,
        }[self.fix_quality]
        return self.hdop * uere_m

    @property
    def is_usable_for_position(self) -> bool:
        """True only for fixes good enough to correct a metre-scale map."""
        return (
            self.fix_quality != GpsFixQuality.NO_FIX
            and self.satellites >= 5
            and self.hdop <= 2.0
        )


class PowerData(BaseModel):
    battery_v: float = Field(..., ge=0.0, le=30.0)
    battery_soc: float = Field(..., ge=0.0, le=100.0)
    current_a: float = Field(default=0.0, ge=-20.0, le=20.0)


# ── Raw sensor packet (what the robot actually transmits) ─────────────────────


class SensorPacket(BaseModel):
    """One complete sample from the robot.

    This is the only message the robot produces.  Everything downstream is
    derived from it, which keeps the robot side simple and cheap to debug.
    """

    schema_version: str = Field(default="1.0")
    robot_id: str
    timestamp: str  # ISO 8601
    sequence: int = Field(..., ge=0)
    link: LinkType = LinkType.WIFI_MQTT
    drive: DriveKind = DriveKind.DIFFERENTIAL

    encoders: EncoderData
    # Encoder counts for every wheel, in the order given by the drive geometry.
    #
    # Why this exists alongside `encoders`: the two-wheel `EncoderData` came
    # first, when the platform was believed to be a differential drive. The
    # actual robot is a three-wheel holonomic base, and a third wheel does not
    # fit that shape. Rather than break every existing message and test, the
    # third wheel travels here and `drive` says which field to trust.
    #
    # A holonomic packet fills both: `wheel_ticks` is authoritative, and
    # `encoders` carries the first two wheels so older readers still parse.
    wheel_ticks: list[int] | None = Field(default=None)

    imu: ImuData | None = None
    ranges: list[RangeReading] = Field(default_factory=list)
    gps: GpsData | None = None
    power: PowerData | None = None

    bumper_active: bool = Field(default=False)

    # What the servos report they are ACTUALLY doing, as opposed to what they
    # were told to do. On a robot with no bumper and no range sensors this is
    # the only evidence a collision ever produces: a wheel that is delivering a
    # fraction of its commanded speed, at high load, is a wheel with something
    # solid in front of it. See `robotmap_common.collision`.
    #
    # Rim speed in metres per second, and load normalised to 0..1.
    wheel_speeds_mps: list[float] | None = Field(default=None)
    wheel_loads: list[float] | None = Field(default=None)

    @field_validator("wheel_ticks")
    @classmethod
    def validate_wheel_ticks(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and len(v) not in (2, 3, 4):
            raise ValueError("wheel_ticks must hold 2, 3 or 4 wheels")
        return v

    @property
    def is_holonomic(self) -> bool:
        return self.drive == DriveKind.HOLONOMIC_3WHEEL and self.wheel_ticks is not None

    @field_validator("robot_id")
    @classmethod
    def validate_robot_id(cls, v: str) -> str:
        if not v or len(v) > 20:
            raise ValueError("robot_id must be 1-20 characters")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}") from exc
        return v


# ── Fused pose ────────────────────────────────────────────────────────────────


class PoseEstimate(BaseModel):
    """Robot position in the local map frame, in metres.

    `std_x_m` / `std_y_m` / `std_heading_deg` are one-sigma uncertainties from
    the filter.  They grow without bound under pure dead reckoning, which is
    exactly the signal the UI uses to warn that the map is drifting.
    """

    robot_id: str
    timestamp: str
    sequence: int

    x_m: float
    y_m: float
    heading_deg: float = Field(..., ge=0.0, lt=360.0)

    linear_velocity_mps: float = Field(default=0.0, ge=-3.0, le=3.0)
    angular_velocity_dps: float = Field(default=0.0, ge=-360.0, le=360.0)

    std_x_m: float = Field(default=0.0, ge=0.0)
    std_y_m: float = Field(default=0.0, ge=0.0)
    std_heading_deg: float = Field(default=0.0, ge=0.0)

    source: PoseSource = PoseSource.ODOMETRY_ONLY
    distance_travelled_m: float = Field(default=0.0, ge=0.0)

    # Populated only when a trustworthy GPS fix has anchored the map frame.
    anchor_latitude: float | None = None
    anchor_longitude: float | None = None

    @property
    def position_confidence(self) -> float:
        """Collapse the covariance into a 0-1 score for display.

        Half a metre of one-sigma error is treated as the point where the
        estimate stops being useful for a room-scale map.
        """
        worst = max(self.std_x_m, self.std_y_m)
        return max(0.0, min(1.0, 1.0 - worst / 0.5))


# ── Scans and mapping ─────────────────────────────────────────────────────────


class PosedScan(BaseModel):
    """Range readings tagged with the pose they were taken from.

    Pairing them in one message removes any ambiguity about which pose a ray
    should be projected from — the classic source of smeared occupancy grids.
    """

    robot_id: str
    timestamp: str
    pose: PoseEstimate
    ranges: list[RangeReading]


class GridMetadata(BaseModel):
    resolution_m: float = Field(..., gt=0.0, le=1.0)
    width_cells: int = Field(..., gt=0)
    height_cells: int = Field(..., gt=0)
    origin_x_m: float  # world coordinate of cell (0,0)
    origin_y_m: float


class MapSnapshot(BaseModel):
    """Full occupancy grid.

    `data` is row-major, one byte per cell: 0-100 = occupancy probability
    percent, -1 = unknown.  Same convention as the ROS OccupancyGrid message,
    so the output can be fed straight into standard tooling later.
    """

    robot_id: str
    timestamp: str
    metadata: GridMetadata
    data: list[int]
    explored_cells: int = Field(default=0, ge=0)

    @field_validator("data")
    @classmethod
    def validate_cells(cls, v: list[int]) -> list[int]:
        for cell in v:
            if cell < -1 or cell > 100:
                raise ValueError("Occupancy values must be -1 (unknown) or 0-100")
        return v


class Point2D(BaseModel):
    x_m: float
    y_m: float


class ObstacleFootprint(BaseModel):
    """Something standing on the floor inside the room.

    Furniture rather than wall: an occupied region fully enclosed by the
    room's own free space. The distinction matters commercially — the floor
    under a table still needs flooring, but a cleaning robot cannot reach it,
    so the two areas are reported separately rather than merged.
    """

    centre_x_m: float
    centre_y_m: float
    min_x_m: float
    min_y_m: float
    max_x_m: float
    max_y_m: float
    area_m2: float = Field(..., ge=0.0)
    cells: int = Field(default=0, ge=0)


class RoomOutline(BaseModel):
    """The extracted room boundary and its measurements."""

    robot_id: str
    timestamp: str
    polygon: list[Point2D]
    area_m2: float = Field(..., ge=0.0)

    # Obstacles found standing on the floor, and how much floor they cover.
    obstacles: list[ObstacleFootprint] = Field(default_factory=list)
    blocked_area_m2: float = Field(default=0.0, ge=0.0)
    perimeter_m: float = Field(..., ge=0.0)
    bounding_width_m: float = Field(..., ge=0.0)
    bounding_height_m: float = Field(..., ge=0.0)
    coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Share of the room the robot has driven"
    )
    is_closed: bool = Field(
        default=False, description="True once the boundary forms a complete loop"
    )

    @property
    def usable_area_m2(self) -> float:
        """Floor the robot could actually drive on.

        Total area minus what furniture stands on. For a cleaning contract
        this is the number that matters; for flooring, `area_m2` is, since the
        floor under a table still has to be laid.
        """
        return max(0.0, self.area_m2 - self.blocked_area_m2)


# ── Control ───────────────────────────────────────────────────────────────────


class CommandRequest(BaseModel):
    robot_id: str
    command: RobotCommand
    value: float | None = None


class CommandMessage(BaseModel):
    command_id: str
    robot_id: str
    command: RobotCommand
    value: float | None = None
    timestamp: str
    issued_by: str = "web-map"


class AckMessage(BaseModel):
    robot_id: str
    command_id: str
    command: str
    accepted: bool
    timestamp: str
    reason: str | None = None


# ── Operations ────────────────────────────────────────────────────────────────


class AlertMessage(BaseModel):
    robot_id: str
    timestamp: str
    alarm_id: str
    alarm_type: str
    severity: str  # INFO | WARNING | CRITICAL
    description: str
    value: float | None = None
    threshold: float | None = None


class ServiceHealth(BaseModel):
    service: str
    status: str  # healthy | degraded | unhealthy
    timestamp: str
    uptime_s: float
    details: dict = Field(default_factory=dict)
