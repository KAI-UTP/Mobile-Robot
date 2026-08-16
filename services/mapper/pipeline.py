"""The mapping pipeline: sensor packets in, pose and room outline out.

Deliberately transport-agnostic. The same object is driven by MQTT packets
from a real robot, by the Bluetooth bridge, or by the simulator running
in-process, which is what makes the offline demo and the live system share
one code path instead of drifting apart.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

from localization.fusion import FilterConfig, PoseFilter
from mapping.occupancy_grid import OccupancyGrid
from mapping.room_extraction import RoomExtractor
from robotmap_common.geometry import RobotGeometry
from robotmap_common.models import (
    MapSnapshot,
    PoseEstimate,
    RoomOutline,
    SensorPacket,
)

logger = logging.getLogger(__name__)


class MappingPipeline:
    """Fuses sensors, maintains the occupancy grid, extracts the room."""

    def __init__(
        self,
        robot_id: str = "MR3W01",
        geometry: RobotGeometry | None = None,
        filter_config: FilterConfig | None = None,
        resolution_m: float = 0.05,
        max_range_m: float = 4.0,
        room_refresh_every: int = 20,
    ) -> None:
        self.robot_id = robot_id
        self.filter = PoseFilter(robot_id, geometry, filter_config)
        self.grid = OccupancyGrid(
            resolution_m=resolution_m, initial_size_m=8.0, max_range_m=max_range_m
        )
        self.extractor = RoomExtractor()

        # Re-extracting the outline on every packet is wasted work: the room
        # cannot change meaningfully in 100 ms, and the trace is the most
        # expensive step in the pipeline.
        self.room_refresh_every = room_refresh_every

        self.pose: PoseEstimate | None = None
        self.room: RoomOutline | None = None
        self.trail: list[tuple[float, float]] = []
        self.packets_processed = 0
        self.packets_rejected = 0

        # Bumper contacts folded into the map. Edge-triggered, so this counts
        # objects touched rather than packets spent touching one.
        self.contacts = 0
        self._bumper_was_active = False

        # The pipeline is driven from a network callback and read by the web
        # server, so every mutation is guarded.
        self._lock = threading.Lock()

    # ── Ingestion ─────────────────────────────────────────────────────────

    def process(self, packet: SensorPacket) -> PoseEstimate:
        """Fold one sensor packet into the map."""
        with self._lock:
            pose = self.filter.update(packet)
            self.pose = pose

            self.grid.integrate_scan(pose, packet.ranges)
            self.grid.mark_robot_footprint(pose)

            # Anything the robot ran into goes on the map. The range sensors
            # miss a lot of real furniture — chair legs narrower than the
            # beam, soft sofas, anything below the sensor's mounting height —
            # and the bumper is what catches those. Recorded after the
            # footprint so a robot stopped against an obstacle marks it rather
            # than erasing it.
            #
            # Only on the closing edge. The switch stays shut for as long as
            # the robot is against the object, and re-marking every packet
            # would smear one touch into a wall as the robot backs away.
            if packet.bumper_active and not self._bumper_was_active:
                self.grid.mark_contact(pose)
                self.contacts += 1
                logger.info(
                    "Contact %d recorded at (%.2f, %.2f) — drawn as blocked "
                    "floor; scan continues",
                    self.contacts, pose.x_m, pose.y_m,
                )
            self._bumper_was_active = packet.bumper_active

            # Record the trail sparsely — one point per 5 cm of travel keeps
            # the browser payload small over a long run.
            if not self.trail:
                self.trail.append((pose.x_m, pose.y_m))
            else:
                last_x, last_y = self.trail[-1]
                if (pose.x_m - last_x) ** 2 + (pose.y_m - last_y) ** 2 > 0.0025:
                    self.trail.append((pose.x_m, pose.y_m))

            self.packets_processed += 1

            if self.packets_processed % self.room_refresh_every == 0:
                self._refresh_room_locked()

            return pose

    def _refresh_room_locked(self) -> None:
        if self.pose is None:
            return
        try:
            self.room = self.extractor.extract(
                self.grid,
                self.pose,
                self.robot_id,
                datetime.now(UTC).isoformat(),
            )
        except Exception:
            # A malformed grid must not take the whole service down; the next
            # refresh will try again with more data.
            logger.exception("Room extraction failed")

    def refresh_room(self) -> RoomOutline | None:
        with self._lock:
            self._refresh_room_locked()
            return self.room

    # ── Control ───────────────────────────────────────────────────────────

    def reset(self, clear_map: bool = True) -> None:
        with self._lock:
            self.filter.reset()
            if clear_map:
                self.grid.clear()
                self.room = None
            self.trail.clear()
            self.packets_processed = 0
            self.contacts = 0
            self._bumper_was_active = False
            logger.info("Pipeline reset (clear_map=%s)", clear_map)

    # ── Output ────────────────────────────────────────────────────────────

    def snapshot(self) -> MapSnapshot | None:
        with self._lock:
            if self.pose is None:
                return None
            return self.grid.snapshot(
                self.robot_id, datetime.now(UTC).isoformat()
            )

    def state(self) -> dict:
        """Everything the web UI needs, in one JSON-serialisable dict."""
        with self._lock:
            metadata = self.grid.metadata()
            return {
                "robot_id": self.robot_id,
                "pose": self.pose.model_dump() if self.pose else None,
                "confidence": self.pose.position_confidence if self.pose else 0.0,
                "room": self.room.model_dump() if self.room else None,
                "trail": self.trail[-2000:],
                "grid": {
                    "resolution_m": metadata.resolution_m,
                    "width": metadata.width_cells,
                    "height": metadata.height_cells,
                    "origin_x_m": metadata.origin_x_m,
                    "origin_y_m": metadata.origin_y_m,
                    "explored_cells": self.grid.explored_cells(),
                },
                "diagnostics": {
                    **self.filter.diagnostics(),
                    "packets_processed": self.packets_processed,
                    "packets_rejected": self.packets_rejected,
                    "contacts": self.contacts,
                },
            }

    def grid_bytes(self) -> bytes:
        """Occupancy as one signed byte per cell, row-major.

        Sent as a binary WebSocket frame rather than JSON: a 200x200 grid is
        40 kB raw but well over 150 kB as a JSON array of numbers, and this
        goes out several times a second.
        """
        with self._lock:
            return self.grid.occupancy_percent().astype("int8").tobytes()
