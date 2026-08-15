"""A virtual 3-wheel robot that emits realistic, *imperfect* sensor data.

Purpose
-------
The hardware is not finalised, but the whole pipeline downstream of the robot
can be built and demonstrated today. This simulator stands in for the real
base and produces the same `SensorPacket` the firmware will.

The noise here is deliberate and physically motivated. A simulator that emits
perfect data proves nothing — the filter and the mapper would look flawless
and then fall apart on contact with a real robot. Each error source below is
one that genuinely exists on this class of hardware:

* quantised encoder ticks (a 20-slot wheel encoder resolves ~5 mm)
* systematic wheel-diameter mismatch, which makes "straight" curve
* random wheel slip
* gyro bias drift
* ultrasonic beam width, dropouts, and specular reflection off angled walls
* GPS that degrades indoors exactly the way a real receiver does
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime

from robotmap_common.geometry import RobotGeometry, local_xy_to_gps, normalize_deg
from robotmap_common.models import (
    EncoderData,
    GpsData,
    GpsFixQuality,
    ImuData,
    LinkType,
    PowerData,
    RangeReading,
    SensorPacket,
)

# Anchor the virtual world at Universiti Teknologi PETRONAS.
DEFAULT_LAT, DEFAULT_LON = 4.3852, 100.9739


# ── World ─────────────────────────────────────────────────────────────────────


@dataclass
class Wall:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class VirtualWorld:
    """The ground-truth room the virtual robot drives around in."""

    walls: list[Wall] = field(default_factory=list)
    indoor: bool = True

    @classmethod
    def rectangular_room(cls, width_m: float = 6.0, height_m: float = 4.5) -> VirtualWorld:
        return cls(
            walls=[
                Wall(0, 0, width_m, 0),
                Wall(width_m, 0, width_m, height_m),
                Wall(width_m, height_m, 0, height_m),
                Wall(0, height_m, 0, 0),
            ]
        )

    @classmethod
    def l_shaped_room(cls) -> VirtualWorld:
        """An L-shaped room — a better test than a rectangle, because a naive
        mapper will happily report the bounding box instead of the true area."""
        pts = [(0, 0), (6, 0), (6, 3), (3.5, 3), (3.5, 5), (0, 5)]
        walls = [
            Wall(*pts[i], *pts[(i + 1) % len(pts)]) for i in range(len(pts))
        ]
        return cls(walls=walls)

    @classmethod
    def room_with_furniture(cls, width_m: float = 6.0, height_m: float = 4.5) -> VirtualWorld:
        world = cls.rectangular_room(width_m, height_m)
        # A table in the middle and a cabinet against one wall.
        for x1, y1, x2, y2 in (
            (2.0, 1.5, 3.2, 1.5),
            (3.2, 1.5, 3.2, 2.6),
            (3.2, 2.6, 2.0, 2.6),
            (2.0, 2.6, 2.0, 1.5),
            (4.8, 3.4, 5.6, 3.4),
            (5.6, 3.4, 5.6, 4.2),
        ):
            world.walls.append(Wall(x1, y1, x2, y2))
        return world

    def raycast(self, x: float, y: float, bearing_deg: float, max_range: float) -> float:
        """Exact distance to the nearest wall along a bearing."""
        angle = math.radians(bearing_deg)
        dx, dy = math.cos(angle), math.sin(angle)
        best = max_range

        for wall in self.walls:
            hit = _ray_segment_intersection(x, y, dx, dy, wall)
            if hit is not None and 0 < hit < best:
                best = hit
        return best

    def contains(self, x: float, y: float) -> bool:
        """Even-odd point-in-polygon test by casting one ray to infinity."""
        crossings = 0
        for wall in self.walls:
            if (wall.y1 > y) != (wall.y2 > y):
                t = (y - wall.y1) / (wall.y2 - wall.y1)
                if wall.x1 + t * (wall.x2 - wall.x1) > x:
                    crossings += 1
        return crossings % 2 == 1


def _ray_segment_intersection(
    ox: float, oy: float, dx: float, dy: float, wall: Wall
) -> float | None:
    """Distance along a ray to a segment, or None if they miss."""
    sx, sy = wall.x2 - wall.x1, wall.y2 - wall.y1
    denom = dx * sy - dy * sx
    if abs(denom) < 1e-12:
        return None  # parallel

    t = ((wall.x1 - ox) * sy - (wall.y1 - oy) * sx) / denom
    u = ((wall.x1 - ox) * dy - (wall.y1 - oy) * dx) / denom

    if t >= 0 and 0 <= u <= 1:
        return t
    return None


# ── Sensor imperfection model ─────────────────────────────────────────────────


@dataclass(frozen=True)
class NoiseProfile:
    """Every value here corresponds to a real defect of real hardware."""

    # The left wheel is 1.5 % smaller than nominal. This is the dominant
    # systematic error on cheap robots and is why they curve when told to
    # drive straight.
    left_wheel_scale: float = 0.985
    # Fraction of commanded motion randomly lost to slip.
    wheel_slip_stddev: float = 0.01

    # Residual gyro bias, degrees per second, *after* the stationary
    # calibration the firmware performs at boot. An uncalibrated MPU6050 sits
    # around 0.15 deg/s, which integrates to roughly 19 degrees over a
    # two-minute mapping run and shears the finished map badly enough that a
    # rectangular room stops measuring as a rectangle. Averaging a few hundred
    # stationary samples at startup and subtracting the mean brings it to the
    # value below — which is why that calibration is not optional.
    #
    # Raise this to 0.15 to reproduce the uncalibrated behaviour and see the
    # map deform.
    gyro_bias_dps: float = 0.02
    imu_noise_deg: float = 0.4

    # Ultrasonic characteristics.
    ultrasonic_noise_m: float = 0.01
    ultrasonic_dropout_rate: float = 0.03
    # Beyond this incidence angle the pulse reflects away and never returns —
    # the reason sonar maps have gaps at angled walls.
    ultrasonic_max_incidence_deg: float = 60.0

    gps_outdoor_noise_m: float = 2.5
    gps_indoor_noise_m: float = 25.0

    # ── BLE beacons ─────────────────────────────────────────────────────
    # Log-normal shadowing, one sigma in dB. Four to eight is the range
    # normally measured in furnished indoor spaces; 6 is a fair middle.
    # This single number dominates RSSI positioning accuracy.
    ble_shadowing_sigma_db: float = 6.0
    # Beacons are not identical. Their true TxPower varies by several dB from
    # the nominal figure, and unless each is individually calibrated that
    # error is indistinguishable from distance.
    ble_tx_power_spread_db: float = 2.5
    # A body between tag and beacon costs this much. Common enough in a room
    # with people in it to be worth modelling.
    ble_body_blocking_db: float = 8.0
    ble_body_blocking_rate: float = 0.08
    # Beyond this the beacon is not heard at all.
    ble_max_range_m: float = 25.0


# ── The robot ─────────────────────────────────────────────────────────────────


class VirtualRobot:
    """Simulates the physical robot and its sensors."""

    def __init__(
        self,
        robot_id: str = "MR3W01",
        world: VirtualWorld | None = None,
        geometry: RobotGeometry | None = None,
        noise: NoiseProfile | None = None,
        sensor_angles_deg: tuple[float, ...] = (0.0, -45.0, 45.0, -90.0, 90.0),
        max_range_m: float = 4.0,
        seed: int = 42,
    ) -> None:
        self.robot_id = robot_id
        self.world = world or VirtualWorld.rectangular_room()
        self.geometry = geometry or RobotGeometry()
        self.noise = noise or NoiseProfile()
        self.sensor_angles_deg = sensor_angles_deg
        self.max_range_m = max_range_m
        self.rng = random.Random(seed)

        # Ground truth — never exposed in telemetry, only used for scoring.
        self.true_x = 1.0
        self.true_y = 1.0
        self.true_heading = 0.0

        self.left_ticks = 0
        self.right_ticks = 0
        self.sequence = 0
        self.battery_soc = 100.0

        self._gyro_accumulated_bias = 0.0
        self._imu_heading = 0.0

    # ── Motion ────────────────────────────────────────────────────────────

    def drive(self, linear_mps: float, angular_dps: float, dt_s: float) -> None:
        """Advance ground truth, then work out what the encoders would read.

        Note the direction of causation: the encoders are *derived from* the
        motion that actually happened, including slip. That is what makes the
        odometry error realistic rather than injected after the fact.
        """
        angular_rps = math.radians(angular_dps)

        # Wheel speeds from the differential-drive inverse kinematics.
        half_base = self.geometry.wheel_base_m / 2.0
        v_left = linear_mps - angular_rps * half_base
        v_right = linear_mps + angular_rps * half_base

        d_left_true = v_left * dt_s
        d_right_true = v_right * dt_s

        # Slip: the wheel turns but the robot does not move quite that far.
        slip_l = 1.0 + self.rng.gauss(0.0, self.noise.wheel_slip_stddev)
        slip_r = 1.0 + self.rng.gauss(0.0, self.noise.wheel_slip_stddev)

        d_left_actual = d_left_true * slip_l
        d_right_actual = d_right_true * slip_r

        d_center = (d_left_actual + d_right_actual) / 2.0
        d_theta = (d_right_actual - d_left_actual) / self.geometry.wheel_base_m

        heading_rad = math.radians(self.true_heading)
        if abs(d_theta) < 1e-9:
            self.true_x += d_center * math.cos(heading_rad)
            self.true_y += d_center * math.sin(heading_rad)
        else:
            radius = d_center / d_theta
            self.true_x += radius * (
                math.sin(heading_rad + d_theta) - math.sin(heading_rad)
            )
            self.true_y -= radius * (
                math.cos(heading_rad + d_theta) - math.cos(heading_rad)
            )
        self.true_heading = normalize_deg(self.true_heading + math.degrees(d_theta))

        # Encoders count the wheel turning, and cannot observe slip. The left
        # wheel's diameter error means its ticks systematically misreport.
        m_per_tick = self.geometry.metres_per_tick
        self.left_ticks += round(
            d_left_actual / (m_per_tick * self.noise.left_wheel_scale)
        )
        self.right_ticks += round(d_right_actual / m_per_tick)

        # IMU integrates true rotation plus a slowly accumulating bias.
        self._gyro_accumulated_bias += self.noise.gyro_bias_dps * dt_s
        self._imu_heading = normalize_deg(
            self.true_heading + self._gyro_accumulated_bias
        )

        self.battery_soc = max(0.0, self.battery_soc - 0.002 * dt_s)

    # ── Sensors ───────────────────────────────────────────────────────────

    def read_ranges(self) -> list[RangeReading]:
        readings: list[RangeReading] = []
        for angle in self.sensor_angles_deg:
            readings.append(self._read_one_range(angle))
        return readings

    def _read_one_range(self, sensor_angle_deg: float) -> RangeReading:
        bearing = self.true_heading + sensor_angle_deg

        if self.rng.random() < self.noise.ultrasonic_dropout_rate:
            return RangeReading(
                angle_deg=sensor_angle_deg,
                distance_m=0.0,
                valid=False,
                sensor_id=f"s{int(sensor_angle_deg)}",
            )

        true_dist = self.world.raycast(
            self.true_x, self.true_y, bearing, self.max_range_m
        )

        # Specular reflection: a pulse striking a wall at a shallow angle
        # bounces away instead of returning, and the sensor reports max range.
        if true_dist < self.max_range_m:
            incidence = self._incidence_angle(bearing, true_dist)
            if incidence > self.noise.ultrasonic_max_incidence_deg:
                true_dist = self.max_range_m

        measured = true_dist + self.rng.gauss(0.0, self.noise.ultrasonic_noise_m)
        measured = max(0.02, min(self.max_range_m, measured))

        return RangeReading(
            angle_deg=sensor_angle_deg,
            distance_m=measured,
            valid=True,
            sensor_id=f"s{int(sensor_angle_deg)}",
        )

    def _incidence_angle(self, bearing_deg: float, distance: float) -> float:
        """Angle between the ray and the wall normal at the hit point."""
        angle = math.radians(bearing_deg)
        hit_x = self.true_x + distance * math.cos(angle)
        hit_y = self.true_y + distance * math.sin(angle)

        nearest, best_d = None, float("inf")
        for wall in self.world.walls:
            d = _point_segment_distance(hit_x, hit_y, wall)
            if d < best_d:
                best_d, nearest = d, wall

        if nearest is None:
            return 0.0

        wall_angle = math.atan2(nearest.y2 - nearest.y1, nearest.x2 - nearest.x1)
        normal_angle = wall_angle + math.pi / 2
        diff = abs(math.degrees(angle - normal_angle)) % 180.0
        return min(diff, 180.0 - diff)

    def read_imu(self) -> ImuData:
        return ImuData(
            heading_deg=normalize_deg(
                self._imu_heading + self.rng.gauss(0.0, self.noise.imu_noise_deg)
            ),
            gyro_z_dps=self.rng.gauss(0.0, 1.0),
            temperature_c=32.0,
            calibrated=True,
        )

    def read_gps(self) -> GpsData:
        """Emit a fix whose quality depends on being indoors.

        Indoors the receiver still reports a position — that is exactly the
        trap this project has to defend against — but with few satellites and
        a large HDOP, which is what makes the fix rejectable.
        """
        indoor = self.world.indoor

        noise_m = (
            self.noise.gps_indoor_noise_m if indoor else self.noise.gps_outdoor_noise_m
        )
        err_x = self.rng.gauss(0.0, noise_m)
        err_y = self.rng.gauss(0.0, noise_m)

        lat, lon = local_xy_to_gps(
            self.true_x + err_x, self.true_y + err_y, DEFAULT_LAT, DEFAULT_LON
        )

        if indoor:
            satellites = self.rng.randint(0, 4)
            hdop = self.rng.uniform(6.0, 25.0)
            quality = GpsFixQuality.NO_FIX if satellites < 3 else GpsFixQuality.GPS
        else:
            satellites = self.rng.randint(7, 12)
            hdop = self.rng.uniform(0.7, 1.6)
            quality = GpsFixQuality.GPS

        return GpsData(
            latitude=lat,
            longitude=lon,
            altitude_m=45.0,
            fix_quality=quality,
            satellites=satellites,
            hdop=round(hdop, 1),
        )

    # ── BLE beacons ───────────────────────────────────────────────────────

    def attach_beacons(self, layout) -> None:
        """Install a beacon layout for the robot to hear.

        Each beacon gets a fixed per-beacon TxPower offset, drawn once. That
        offset is *systematic*: it does not average away over samples, which
        is why per-beacon calibration matters more than sample count.
        """
        self.beacon_layout = layout
        self._beacon_tx_offsets = {
            beacon.beacon_id: self.rng.gauss(0.0, self.noise.ble_tx_power_spread_db)
            for beacon in layout.as_list()
        }

    def read_beacons(self):
        """RSSI from every beacon in range, with realistic impairments.

        Three error sources, all of which exist in a real room:

        * **Log-normal shadowing** — the dominant term, and zero-mean, so
          averaging several advertisements does help.
        * **Per-beacon TxPower error** — systematic, so averaging does NOT
          help. Only calibration does.
        * **Body blocking** — occasional deep fades from someone standing in
          the path. These are one-sided and produce outliers, not noise.
        """
        from robotmap_common.rssi import BeaconReading, distance_to_rssi

        layout = getattr(self, "beacon_layout", None)
        if layout is None:
            return []

        readings = []
        for beacon in layout.as_list():
            distance = beacon.distance_to(self.true_x, self.true_y)
            if distance > self.noise.ble_max_range_m:
                continue

            true_rssi = distance_to_rssi(
                distance,
                beacon.tx_power_dbm + self._beacon_tx_offsets.get(beacon.beacon_id, 0.0),
                # The simulator attenuates as a furnished room does; the solver
                # is not told this and uses its own assumed exponent, exactly
                # as it would on real hardware.
                path_loss_exponent=2.7,
            )

            rssi = true_rssi + self.rng.gauss(0.0, self.noise.ble_shadowing_sigma_db)

            if self.rng.random() < self.noise.ble_body_blocking_rate:
                rssi -= abs(self.rng.gauss(self.noise.ble_body_blocking_db, 3.0))

            readings.append(BeaconReading(beacon.beacon_id, rssi))

        return readings

    # ── Packet assembly ───────────────────────────────────────────────────

    def build_packet(self, dt_ms: int = 100, include_gps: bool = True) -> SensorPacket:
        self.sequence += 1
        return SensorPacket(
            robot_id=self.robot_id,
            timestamp=datetime.now(UTC).isoformat(),
            sequence=self.sequence,
            link=LinkType.SIMULATED,
            encoders=EncoderData(
                left_ticks=self.left_ticks,
                right_ticks=self.right_ticks,
                dt_ms=dt_ms,
            ),
            imu=self.read_imu(),
            ranges=self.read_ranges(),
            gps=self.read_gps() if include_gps else None,
            power=PowerData(
                battery_v=7.4 * (0.85 + 0.15 * self.battery_soc / 100),
                battery_soc=self.battery_soc,
                current_a=0.6,
            ),
        )

    # ── Scoring ───────────────────────────────────────────────────────────

    def true_pose(self) -> tuple[float, float, float]:
        """Ground truth, for measuring how well the filter is doing."""
        return self.true_x, self.true_y, self.true_heading


def _point_segment_distance(px: float, py: float, wall: Wall) -> float:
    dx, dy = wall.x2 - wall.x1, wall.y2 - wall.y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - wall.x1, py - wall.y1)
    t = max(0.0, min(1.0, ((px - wall.x1) * dx + (py - wall.y1) * dy) / length_sq))
    return math.hypot(px - (wall.x1 + t * dx), py - (wall.y1 + t * dy))
