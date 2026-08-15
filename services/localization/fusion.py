"""Sensor fusion: wheel odometry + IMU heading + gated GPS -> one pose.

Why a filter at all
-------------------
Each sensor is individually inadequate:

* **Wheel encoders** give excellent *short-term* position but drift without
  bound. Wheel slip and diameter error accumulate; after 20 m of driving a
  typical hobby base is off by 0.5-1 m and there is no way for it to notice.
* **IMU** gives good short-term heading but the gyro bias integrates into
  yaw drift of degrees per minute.
* **GPS** does not drift — its error is bounded — but the bound is 3-5 m
  outdoors and effectively unbounded indoors.

The filter keeps odometry as the backbone, corrects its heading with the IMU,
and lets GPS pull out the slow position drift *only when the fix is good
enough to be an improvement*. That last condition is the whole game, and is
handled by `_gps_is_trustworthy`.

State vector: [x, y, theta] in metres and radians, local map frame.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from robotmap_common.geometry import (
    RobotGeometry,
    angle_difference_deg,
    differential_drive_delta,
    gps_to_local_xy,
    normalize_deg,
    wrap_tick_delta,
)
from robotmap_common.holonomic import HolonomicGeometry, odometry_from_ticks
from robotmap_common.models import (
    GpsData,
    ImuData,
    PoseEstimate,
    PoseSource,
    SensorPacket,
)

logger = logging.getLogger(__name__)


# ── Tuning ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FilterConfig:
    """Noise parameters. These are the knobs worth tuning on real hardware."""

    # Process noise, expressed as the one-sigma error accumulated after one
    # metre of driving. Odometry error is a random walk, so variance grows
    # linearly with distance and sigma grows with its square root: after
    # 100 m the default below predicts 0.03 * sqrt(100) = 30 cm of drift.
    #
    # 3 cm/m suits a small hobby base on a hard floor. Raise it for carpet,
    # which slips considerably more.
    odometry_position_noise_per_m: float = 0.03
    # Heading error accumulated per metre driven, degrees. This is what
    # unequal wheel diameters look like to the filter.
    odometry_heading_noise_per_m: float = 1.5
    # Heading error per degree turned — turning in place slips much more than
    # driving straight, so it gets its own term.
    odometry_heading_noise_per_deg: float = 0.05

    # One-sigma position error, metres, beyond which the pose is reported as
    # degraded and scans stop being written into the map.
    degraded_position_std_m: float = 0.5

    # Odometry noise is multiplied by this on a holonomic base.
    #
    # Omni wheels are built around rollers that slide sideways — that sliding
    # is what makes the robot holonomic, and it is invisible to the encoders.
    # A differential wheel only slips when traction is lost; an omni wheel
    # slips a little all the time, by design. Roughly double the noise is a
    # reasonable starting point; tune it against a measured drive once the
    # real robot runs.
    holonomic_noise_multiplier: float = 2.0

    # IMU heading measurement noise, degrees. An MPU6050 with a settled bias
    # holds about this once complementary-filtered on-board.
    imu_heading_noise_deg: float = 2.0
    # Above this rotation rate the accelerometer reference is useless and the
    # IMU is briefly distrusted.
    imu_max_trusted_rate_dps: float = 200.0

    # GPS acceptance gates.
    gps_max_hdop: float = 2.0
    gps_min_satellites: int = 5
    # Reject a fix that disagrees with the current estimate by more than this
    # many sigma — the standard innovation gate that stops one wild fix from
    # destroying an otherwise good map.
    gps_outlier_sigma: float = 3.0
    # Never let GPS correct the map by more than this in one step, however
    # confident it claims to be.
    gps_max_correction_m: float = 5.0

    # Consecutive good fixes required before GPS is allowed to anchor the
    # map frame. Guards against a single fluke fix defining the world.
    gps_anchor_required_fixes: int = 5

    # A fix may only *correct* the pose if its estimated accuracy is better
    # than this. Anchoring is unaffected — a coarse fix still georeferences
    # the map perfectly well.
    #
    # This is the single most important rule in the file. A consumer GPS fix
    # is accurate to roughly 4 m, which is the width of the room being
    # mapped, so folding it into the pose does not refine the map, it
    # scrambles it: the estimate is dragged around by metres of GNSS noise,
    # the occupancy grid smears, and loop closure never triggers. Measured on
    # the simulator, enabling such corrections moved the reported area of a
    # 27 m2 room to between 26 and 54 m2 depending only on the noise draw,
    # while rejecting them held it at 25.8-27.2 m2.
    #
    # Only differential or RTK receivers clear this bar. For everything else
    # the map is built from odometry and the GPS supplies georeferencing.
    gps_max_accuracy_for_correction_m: float = 1.0


# ── Filter ────────────────────────────────────────────────────────────────────


@dataclass
class _Covariance:
    """Diagonal-plus-heading covariance.

    A full 3x3 matrix is the textbook answer, but the cross terms contribute
    little for a differential-drive robot at this scale and the diagonal form
    is far easier to reason about and debug in a student project.
    """

    var_x: float = 0.0
    var_y: float = 0.0
    var_heading: float = 0.0  # degrees squared


class PoseFilter:
    """Fuses one robot's sensors into a running pose estimate."""

    def __init__(
        self,
        robot_id: str,
        geometry: RobotGeometry | None = None,
        config: FilterConfig | None = None,
        holonomic_geometry: HolonomicGeometry | None = None,
    ) -> None:
        self.robot_id = robot_id
        self.geometry = geometry or RobotGeometry()
        self.geometry.validate()
        # Used only for packets that declare themselves holonomic. Both are
        # held so one filter can serve either platform — useful while the
        # differential simulator and the real kiwi-drive robot coexist.
        self.holonomic_geometry = holonomic_geometry or HolonomicGeometry()
        self.holonomic_geometry.validate()
        self.config = config or FilterConfig()

        # Pose in the local map frame.
        self.x_m = 0.0
        self.y_m = 0.0
        self.heading_deg = 0.0

        self.covariance = _Covariance()

        self.linear_velocity_mps = 0.0
        self.angular_velocity_dps = 0.0
        self.distance_travelled_m = 0.0

        self._prev_left_ticks: int | None = None
        self._prev_right_ticks: int | None = None
        self._prev_wheel_ticks: list[int] | None = None
        self._prev_imu_heading: float | None = None
        # Offset between the IMU's arbitrary zero and the map frame's zero.
        self._imu_heading_offset: float | None = None

        self.anchor_lat: float | None = None
        self.anchor_lon: float | None = None
        # Map-frame position occupied at the instant the anchor was set. The
        # anchor is only established after several consecutive good fixes, by
        # which time the robot has usually moved; without recording where it
        # was, every GPS position would be offset by that distance.
        self.anchor_map_x = 0.0
        self.anchor_map_y = 0.0
        self._consecutive_good_fixes = 0

        self._used_imu = False
        self._used_gps = False
        self.sequence = 0

        # Diagnostics surfaced to the UI and logs. Acceptance and correction
        # are counted separately: a fix can be perfectly valid, and used to
        # georeference the map, without being precise enough to move the robot
        # on it.
        self.gps_rejections = 0
        self.gps_acceptances = 0
        self.gps_corrections = 0
        self.last_gps_reason = "no GPS data received"

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Return to the origin and forget all history."""
        self.x_m = self.y_m = self.heading_deg = 0.0
        self.covariance = _Covariance()
        self.distance_travelled_m = 0.0
        self._prev_left_ticks = self._prev_right_ticks = None
        self._prev_wheel_ticks = None
        self._prev_imu_heading = None
        self._imu_heading_offset = None
        self._consecutive_good_fixes = 0
        logger.info("Pose filter reset to origin")

    # ── Main entry point ──────────────────────────────────────────────────

    def update(self, packet: SensorPacket) -> PoseEstimate:
        """Fold one sensor packet into the estimate and return the new pose."""
        self._used_imu = False
        self._used_gps = False

        self._predict_from_encoders(packet)

        if packet.imu is not None:
            self._correct_heading_from_imu(packet.imu)

        if packet.gps is not None:
            self._correct_position_from_gps(packet.gps)

        self.sequence = packet.sequence
        return self._build_estimate(packet.timestamp)

    # ── Prediction step ───────────────────────────────────────────────────

    def _predict_from_encoders(self, packet: SensorPacket) -> None:
        """Advance the pose using the wheel encoders and grow the covariance."""
        if packet.is_holonomic:
            self._predict_holonomic(packet)
            return

        enc = packet.encoders

        if self._prev_left_ticks is None or self._prev_right_ticks is None:
            # First packet only establishes the baseline; no motion implied.
            self._prev_left_ticks = enc.left_ticks
            self._prev_right_ticks = enc.right_ticks
            return

        left_delta = wrap_tick_delta(enc.left_ticks, self._prev_left_ticks)
        right_delta = wrap_tick_delta(enc.right_ticks, self._prev_right_ticks)
        self._prev_left_ticks = enc.left_ticks
        self._prev_right_ticks = enc.right_ticks

        delta = differential_drive_delta(
            left_delta, right_delta, self.heading_deg, self.geometry
        )

        self.x_m += delta.delta_x_m
        self.y_m += delta.delta_y_m
        self.heading_deg = normalize_deg(self.heading_deg + delta.delta_heading_deg)
        self.distance_travelled_m += delta.distance_m

        dt_s = enc.dt_ms / 1000.0
        self.linear_velocity_mps = delta.distance_m / dt_s if dt_s > 0 else 0.0
        self.angular_velocity_dps = delta.delta_heading_deg / dt_s if dt_s > 0 else 0.0

        self._grow_covariance(delta.distance_m, abs(delta.delta_heading_deg))

    def _predict_holonomic(self, packet: SensorPacket) -> None:
        """Advance the pose from three omni-wheel encoders.

        Kept separate from the differential path rather than generalised into
        it, because the two share almost nothing: this one solves a 3x3 system
        for a full planar twist, the other integrates an arc from two wheels.

        Omni-wheel odometry is materially worse than differential. The rollers
        that let the robot strafe also slide during ordinary driving, and that
        sliding is invisible to the encoders — so the noise terms below are
        inflated accordingly. This is why the mapping stack leans on range
        sensing rather than trusting dead reckoning.
        """
        ticks = packet.wheel_ticks
        if ticks is None or len(ticks) < 3:
            return

        dt_s = packet.encoders.dt_ms / 1000.0

        if self._prev_wheel_ticks is None:
            self._prev_wheel_ticks = list(ticks[:3])
            return

        deltas = tuple(
            wrap_tick_delta(ticks[i], self._prev_wheel_ticks[i]) for i in range(3)
        )
        self._prev_wheel_ticks = list(ticks[:3])

        delta = odometry_from_ticks(
            deltas, self.heading_deg, dt_s, self.holonomic_geometry
        )

        self.x_m += delta.delta_x_m
        self.y_m += delta.delta_y_m
        self.heading_deg = normalize_deg(self.heading_deg + delta.delta_heading_deg)
        self.distance_travelled_m += delta.distance_m

        self.linear_velocity_mps = delta.distance_m / dt_s if dt_s > 0 else 0.0
        self.angular_velocity_dps = (
            delta.delta_heading_deg / dt_s if dt_s > 0 else 0.0
        )

        self._grow_covariance(
            delta.distance_m,
            abs(delta.delta_heading_deg),
            noise_scale=self.config.holonomic_noise_multiplier,
        )

    def _grow_covariance(
        self, distance_m: float, turn_deg: float, noise_scale: float = 1.0
    ) -> None:
        """Inflate uncertainty to reflect that dead reckoning just got worse.

        Odometry error is a random walk, so *variance* accumulates linearly
        with distance travelled and sigma therefore grows with its square
        root.  Writing it this way matters: the intuitive-looking alternative,
        adding (sigma_per_m * distance)^2 each step, makes the total depend on
        how finely the motion happens to be sampled — the same metre driven
        would report a hundred times less uncertainty when reported in a
        hundred packets instead of one.
        """
        cfg = self.config
        scale_sq = noise_scale**2
        pos_var = cfg.odometry_position_noise_per_m**2 * distance_m * scale_sq
        self.covariance.var_x += pos_var
        self.covariance.var_y += pos_var

        self.covariance.var_heading += (
            cfg.odometry_heading_noise_per_m**2 * distance_m
            + cfg.odometry_heading_noise_per_deg**2 * turn_deg
        ) * scale_sq

    # ── IMU heading correction ────────────────────────────────────────────

    def _correct_heading_from_imu(self, imu: ImuData) -> None:
        """Blend the IMU yaw into the odometry heading.

        The IMU's zero is arbitrary, so the first reading defines an offset
        that converts IMU yaw into map-frame heading. Thereafter the two are
        combined by their relative confidence, which is a scalar Kalman gain.
        """
        if not imu.calibrated:
            return

        # Mid-turn the accelerometer tilt reference is swamped; skip.
        if abs(imu.gyro_z_dps) > self.config.imu_max_trusted_rate_dps:
            return

        if self._imu_heading_offset is None:
            self._imu_heading_offset = angle_difference_deg(
                self.heading_deg, imu.heading_deg
            )
            self._prev_imu_heading = imu.heading_deg
            return

        imu_map_heading = normalize_deg(imu.heading_deg + self._imu_heading_offset)

        odom_var = self.covariance.var_heading
        imu_var = self.config.imu_heading_noise_deg**2

        # Kalman gain: weight the measurement by how much better it is.
        gain = odom_var / (odom_var + imu_var) if (odom_var + imu_var) > 0 else 0.0

        innovation = angle_difference_deg(imu_map_heading, self.heading_deg)
        self.heading_deg = normalize_deg(self.heading_deg + gain * innovation)
        self.covariance.var_heading = (1.0 - gain) * odom_var

        self._prev_imu_heading = imu.heading_deg
        self._used_imu = True

    # ── GPS position correction ───────────────────────────────────────────

    def _gps_is_trustworthy(self, gps: GpsData) -> tuple[bool, str]:
        """Decide whether a fix is good enough to touch the map.

        This is the function that stops GPS from ruining an indoor map. A
        receiver sitting inside a building still emits fixes; they are simply
        wrong, by tens of metres, and they wander. The only defence is to
        refuse them, and the NMEA quality fields are what make that possible.
        """
        cfg = self.config

        if gps.fix_quality.value == "NO_FIX":
            return False, "no satellite fix"
        if gps.satellites < cfg.gps_min_satellites:
            return False, f"only {gps.satellites} satellites (need {cfg.gps_min_satellites})"
        if gps.hdop > cfg.gps_max_hdop:
            return False, f"HDOP {gps.hdop:.1f} too high (need <= {cfg.gps_max_hdop})"
        return True, "fix accepted"

    def _correct_position_from_gps(self, gps: GpsData) -> None:
        accepted, reason = self._gps_is_trustworthy(gps)
        self.last_gps_reason = reason

        if not accepted:
            self._consecutive_good_fixes = 0
            self.gps_rejections += 1
            return

        self._consecutive_good_fixes += 1
        self.gps_acceptances += 1

        # The first sustained run of good fixes defines where the map is on
        # Earth. It does not move the robot — the origin is wherever the
        # robot started, and that stays true.
        if self.anchor_lat is None:
            if self._consecutive_good_fixes >= self.config.gps_anchor_required_fixes:
                self.anchor_lat = gps.latitude
                self.anchor_lon = gps.longitude
                self.anchor_map_x = self.x_m
                self.anchor_map_y = self.y_m
                logger.info(
                    "Map frame anchored at %.6f, %.6f (map %.2f, %.2f)",
                    gps.latitude,
                    gps.longitude,
                    self.x_m,
                    self.y_m,
                )
            return

        # Refuse to correct the map with a fix coarser than the map itself.
        accuracy = gps.estimated_accuracy_m
        if accuracy > self.config.gps_max_accuracy_for_correction_m:
            self.last_gps_reason = (
                f"anchor only: {accuracy:.1f} m accuracy is too coarse to "
                f"correct a room-scale map"
            )
            return

        east, north = gps_to_local_xy(
            gps.latitude, gps.longitude, self.anchor_lat, self.anchor_lon
        )
        # Offsets are relative to the anchor, so shift them into the map frame.
        gps_x = east + self.anchor_map_x
        gps_y = north + self.anchor_map_y

        # Measurement variance from the receiver's own accuracy estimate.
        gps_var = max(accuracy**2, 0.0004)

        innovation_x = gps_x - self.x_m
        innovation_y = gps_y - self.y_m
        innovation_dist = math.hypot(innovation_x, innovation_y)

        # Innovation gate: how far off could this plausibly be, given both
        # our uncertainty and the receiver's?
        combined_sigma = math.sqrt(
            max(self.covariance.var_x, self.covariance.var_y) + gps_var
        )
        if innovation_dist > self.config.gps_outlier_sigma * combined_sigma:
            self.last_gps_reason = (
                f"rejected outlier: {innovation_dist:.1f} m from estimate"
            )
            self.gps_rejections += 1
            return

        gain_x = self.covariance.var_x / (self.covariance.var_x + gps_var)
        gain_y = self.covariance.var_y / (self.covariance.var_y + gps_var)

        correction_x = gain_x * innovation_x
        correction_y = gain_y * innovation_y

        # Hard clamp. Even a formally valid correction should never teleport
        # the robot across the map in one step.
        correction_dist = math.hypot(correction_x, correction_y)
        max_corr = self.config.gps_max_correction_m
        if correction_dist > max_corr:
            scale = max_corr / correction_dist
            correction_x *= scale
            correction_y *= scale

        self.x_m += correction_x
        self.y_m += correction_y
        self.covariance.var_x *= 1.0 - gain_x
        self.covariance.var_y *= 1.0 - gain_y

        self.gps_corrections += 1
        self._used_gps = True

    # ── Output ────────────────────────────────────────────────────────────

    def _classify_source(self) -> PoseSource:
        if self._used_gps:
            return PoseSource.ODOMETRY_IMU_GPS
        # Past the degraded threshold the map can no longer be trusted to line
        # up with itself; report that rather than pretend otherwise.
        if (
            max(self.covariance.var_x, self.covariance.var_y)
            > self.config.degraded_position_std_m**2
        ):
            return PoseSource.DEAD_RECKONING_DEGRADED
        if self._used_imu:
            return PoseSource.ODOMETRY_IMU
        return PoseSource.ODOMETRY_ONLY

    def _build_estimate(self, timestamp: str) -> PoseEstimate:
        return PoseEstimate(
            robot_id=self.robot_id,
            timestamp=timestamp,
            sequence=self.sequence,
            x_m=self.x_m,
            y_m=self.y_m,
            heading_deg=normalize_deg(self.heading_deg),
            linear_velocity_mps=max(-3.0, min(3.0, self.linear_velocity_mps)),
            angular_velocity_dps=max(-360.0, min(360.0, self.angular_velocity_dps)),
            std_x_m=math.sqrt(self.covariance.var_x),
            std_y_m=math.sqrt(self.covariance.var_y),
            std_heading_deg=math.sqrt(self.covariance.var_heading),
            source=self._classify_source(),
            distance_travelled_m=self.distance_travelled_m,
            anchor_latitude=self.anchor_lat,
            anchor_longitude=self.anchor_lon,
        )

    # ── Diagnostics ───────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        return {
            "gps_acceptances": self.gps_acceptances,
            "gps_corrections": self.gps_corrections,
            "gps_rejections": self.gps_rejections,
            "last_gps_reason": self.last_gps_reason,
            "anchored": self.anchor_lat is not None,
            "imu_locked": self._imu_heading_offset is not None,
            "distance_travelled_m": round(self.distance_travelled_m, 2),
            "std_x_m": round(math.sqrt(self.covariance.var_x), 3),
            "std_y_m": round(math.sqrt(self.covariance.var_y), 3),
            "std_heading_deg": round(math.sqrt(self.covariance.var_heading), 2),
        }
