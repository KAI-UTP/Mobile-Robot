"""Autonomous room scan: the same two controllers CI tests, driving hardware.

    python services/pilot/main.py --servo-port COM5

What this closes
----------------
Before this existed the robot could stream telemetry and be driven by hand, and
the autonomy could drive a virtual robot in CI — but nothing joined the two.
`Topics.COMMAND` was declared and had no consumer, so a real scan meant a human
with a keyboard. The measurement pipeline was a product; the robot was a demo.

The loop
--------
Sensor packets in (MQTT or a driver poll), one controller step, a body twist
out. The controllers are the ones from `autonomy/`, unchanged and unaware of
which robot they are driving:

1. `WallFollower` until the perimeter closes.
2. `CoveragePlanner` over the interior, bounded by the outline phase 1 measured.

Safety
------
Real hardware, so the failure modes are physical:

* **The wheels stop when this process does.** Every exit path, including an
  exception and a Ctrl-C, goes through `stop()`. A crash that leaves a robot
  driving into a wall at full speed is not an acceptable failure.
* **Stale sensor data stops the robot.** If packets stop arriving the world
  model is frozen while the robot keeps moving through the real world, which is
  precisely when it hits something. Older than `stale_after_s` means stop.
* **The driver's own watchdog is the backstop.** It halts the wheels if no
  command arrives, so even this process being SIGKILLed stops the robot within
  `watchdog_s`.
* **Speed is limited here as well as in the driver.** Two independent limits,
  because the one that matters is whichever is lower and a scan does not need
  to be fast.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum

from robotmap_common.holonomic import BodyTwist
from robotmap_common.models import PoseEstimate, SensorPacket

from autonomy.coverage import CoveragePlanner, CoverageState
from autonomy.explorer import ExploreState, WallFollower

logger = logging.getLogger("pilot")


class ScanPhase(str, Enum):
    WAITING = "WAITING"          # no sensor data yet
    PERIMETER = "PERIMETER"      # wall-following the boundary
    SWEEPING = "SWEEPING"        # row-by-row over the interior
    DONE = "DONE"
    STOPPED = "STOPPED"          # halted on a fault


@dataclass
class PilotConfig:
    # Scans are slow on purpose. Every metre driven is odometry drift, and a
    # room measured badly quickly is worth less than one measured well.
    max_linear_mps: float = 0.20
    max_angular_dps: float = 70.0

    # Sensor data older than this means the robot is driving blind.
    stale_after_s: float = 0.6

    # Skip the interior sweep and measure the outline only.
    sweep: bool = True

    # Give up rather than drive for ever if the perimeter never closes.
    perimeter_timeout_s: float = 600.0
    sweep_timeout_s: float = 900.0


@dataclass
class PilotStatus:
    phase: ScanPhase = ScanPhase.WAITING
    note: str = ""
    packets: int = 0
    stopped_reason: str = ""
    started_at: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "note": self.note,
            "packets": self.packets,
            "stopped_reason": self.stopped_reason,
            "elapsed_s": round(time.monotonic() - self.started_at, 1),
        }


class Pilot:
    """Runs an autonomous scan against whatever supplies poses and ranges.

    Deliberately given the pose rather than estimating one: the localisation
    filter already exists and runs in the mapper. Duplicating it here would
    mean two pose estimates that disagree, and the robot would be steering by
    the one nobody is looking at.
    """

    def __init__(
        self,
        config: PilotConfig | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.config = config or PilotConfig()
        self.status = PilotStatus()

        self.follower = WallFollower()
        self.planner: CoveragePlanner | None = None
        self._bounds = bounds
        self._phase_started_at = time.monotonic()

    # ── The loop ──────────────────────────────────────────────────────────

    def step(
        self,
        packet: SensorPacket | None,
        pose: PoseEstimate | None,
        dt_s: float,
        age_s: float = 0.0,
    ) -> BodyTwist:
        """One control cycle. Returns the twist to send to the wheels."""
        if self.status.phase in (ScanPhase.DONE, ScanPhase.STOPPED):
            return BodyTwist()

        if packet is None or pose is None:
            self.status.note = "waiting for sensor data"
            return BodyTwist()

        # Driving on a frozen world model is how a robot hits things.
        if age_s > self.config.stale_after_s:
            self.status.note = f"sensor data {age_s:.1f}s old"
            return BodyTwist()

        self.status.packets += 1

        if packet.bumper_active and self.status.phase == ScanPhase.PERIMETER:
            # Wall-following has no recovery behaviour — it assumes the range
            # sensors saw the wall. A contact means they did not, and pressing
            # on grinds the wheels against it.
            return self._halt("bumper triggered during the perimeter lap")

        if self.status.phase == ScanPhase.WAITING:
            self.status.phase = ScanPhase.PERIMETER
            self._phase_started_at = time.monotonic()

        if self.status.phase == ScanPhase.PERIMETER:
            return self._perimeter(packet, pose, dt_s)

        return self._sweep(packet, pose, dt_s)

    def _perimeter(self, packet, pose, dt_s: float) -> BodyTwist:
        if time.monotonic() - self._phase_started_at > self.config.perimeter_timeout_s:
            return self._halt("perimeter lap timed out")

        command = self.follower.step(packet.ranges, pose.x_m, pose.y_m, dt_s)
        self.status.note = command.note

        if command.state == ExploreState.FINISHED:
            logger.info(
                "Perimeter closed after %.1f m", self.follower.distance_travelled_m
            )
            if not self.config.sweep:
                return self._finish("outline measured; sweep disabled")
            self.status.phase = ScanPhase.SWEEPING
            self._phase_started_at = time.monotonic()
            self.planner = CoveragePlanner(bounds=self._bounds)
            return BodyTwist()

        # Wall-following predates the holonomic base and speaks in forward and
        # turn only. That is not a gap worth closing: hugging a wall means
        # holding a distance to one side while driving along it, which is
        # exactly a differential motion. Strafing would not make the lap
        # shorter or the outline better.
        return self._limit(BodyTwist(vx_mps=command.linear_mps, omega_dps=command.angular_dps))

    def _sweep(self, packet, pose, dt_s: float) -> BodyTwist:
        assert self.planner is not None

        if time.monotonic() - self._phase_started_at > self.config.sweep_timeout_s:
            return self._finish("sweep timed out; outline already measured")

        command = self.planner.step(
            packet.ranges,
            pose.x_m,
            pose.y_m,
            pose.heading_deg,
            packet.bumper_active,
            dt_s,
        )
        self.status.note = command.note

        if command.state == CoverageState.FINISHED:
            summary = self.planner.summary()
            logger.info("Sweep complete: %s", summary)
            return self._finish(
                f"{summary['rows_completed']} rows, "
                f"{summary['obstacles_found']} obstacles, "
                f"{summary['collisions']} collisions"
            )

        return self._limit(
            BodyTwist(
                vx_mps=command.linear_mps,
                vy_mps=command.lateral_mps,
                omega_dps=command.angular_dps,
            )
        )

    # ── Limits and endings ────────────────────────────────────────────────

    def _limit(self, twist: BodyTwist) -> BodyTwist:
        """Clamp translation by magnitude so a diagonal keeps its direction.

        Clamping the components separately would turn a fast diagonal into a
        different heading, which on a holonomic base means the robot quietly
        drives somewhere other than where it was told.
        """
        speed = math.hypot(twist.vx_mps, twist.vy_mps)
        vx, vy = twist.vx_mps, twist.vy_mps
        if speed > self.config.max_linear_mps and speed > 0:
            factor = self.config.max_linear_mps / speed
            vx, vy = vx * factor, vy * factor

        omega = max(
            -self.config.max_angular_dps,
            min(self.config.max_angular_dps, twist.omega_dps),
        )
        return BodyTwist(vx_mps=vx, vy_mps=vy, omega_dps=omega)

    def _halt(self, reason: str) -> BodyTwist:
        logger.error("Scan stopped: %s", reason)
        self.status.phase = ScanPhase.STOPPED
        self.status.stopped_reason = reason
        self.status.note = reason
        return BodyTwist()

    def _finish(self, note: str) -> BodyTwist:
        logger.info("Scan finished: %s", note)
        self.status.phase = ScanPhase.DONE
        self.status.note = note
        return BodyTwist()

    @property
    def is_running(self) -> bool:
        return self.status.phase not in (ScanPhase.DONE, ScanPhase.STOPPED)

    def set_bounds(self, bounds: tuple[float, float, float, float] | None) -> None:
        """Hand the sweep the outline the perimeter lap measured.

        Called by the runner once the mapper has published a room. Without it
        the sweep has to discover each row's end by driving into the wall,
        which on real hardware is a collision per row.
        """
        self._bounds = bounds
        if self.planner is not None and self.planner.bounds is None:
            self.planner.bounds = bounds
