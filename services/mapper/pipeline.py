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
from mapping.costmap import Costmap
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

        # Obstacles grown by the robot's own size. Kept apart from the grid
        # because the two answer different questions: the grid says what is
        # there, this says where the robot may go. Change the chassis and this
        # changes while the evidence does not.
        self.costmap = Costmap()

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

        # Set once the boundary is trusted and should stop being revised.
        # See `freeze_outline`.
        self._frozen_outline: RoomOutline | None = None

        # The pipeline is driven from a network callback and read by the web
        # server, so every mutation is guarded.
        # Re-entrant because `record_contact` is called both from inside
        # `process`, which already holds it, and from the control loop, which
        # does not. A plain Lock deadlocks the mapper on the first collision.
        self._lock = threading.RLock()

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
                self.record_contact(pose, "bumper switch")
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

    def record_contact(self, pose: PoseEstimate, reason: str = "contact") -> None:
        """Draw something the robot could not get through, and carry on.

        Called both by the optional bumper path above and by the controller's
        collision detector, which infers contact from the servo bus because
        this robot has no contact switch. Either way the map gains a blocked
        patch and mapping continues — an obstacle is information, not a
        failure, and stopping the scan would throw away the rest of the room.
        """
        with self._lock:
            self.grid.mark_contact(pose)
            self.contacts += 1
        logger.info(
            "Contact %d at (%.2f, %.2f): %s — drawn as blocked floor, "
            "scan continues",
            self.contacts, pose.x_m, pose.y_m, reason,
        )

    def costmap_bytes(self) -> bytes:
        """The map inflated by the robot's radius, for planning and for display.

        Nav2's inflation layer, without ROS: obstacles grown by the chassis so
        a planner can treat the robot as a point, plus a decaying penalty
        beyond that so it prefers the middle of a gap to its edge.

        Computed on demand rather than kept in step with every packet. It is a
        derived view — the evidence is the occupancy grid — and a full-map
        distance transform per packet at 10 Hz would cost far more than the
        handful of times anyone actually looks at it.
        """
        with self._lock:
            occupied = self.grid.occupied_mask()
            cost = self.costmap.build(occupied, self.grid.resolution_m)
            return cost.tobytes()

    def costmap_summary(self) -> dict:
        with self._lock:
            occupied = self.grid.occupied_mask()
            cost = self.costmap.build(occupied, self.grid.resolution_m)
            return self.costmap.summarise(cost, self.grid.resolution_m)

    def freeze_outline(self) -> None:
        """Keep the outline measured so far, and stop revising it.

        The perimeter lap and the interior sweep are good at different things,
        and the sweep is actively bad at one of them. The lap observes the
        boundary from close range and at good incidence angles: measured, it
        returns 26.63 m2 against a true 27.0. The sweep then drives 41 m across
        the middle of the room, accumulating dead-reckoning error the whole
        way, and drags the same outline out to about 30 m2 with a bounding box
        of 7.0 x 6.7 for a 6.0 x 4.5 room.

        Nothing about crossing the middle of a room tells you where its walls
        are, so there is no reason to let it try. After this call the boundary
        is fixed and only the obstacle findings are updated — which is the one
        thing the sweep genuinely contributes.
        """
        with self._lock:
            if self.room is not None:
                self._frozen_outline = self.room
                logger.info(
                    "Outline frozen at %.2f m2; the sweep will only add "
                    "obstacles from here",
                    self.room.area_m2,
                )

    def _refresh_room_locked(self) -> None:
        if self.pose is None:
            return
        # Once the boundary is frozen it is also the boundary the obstacle
        # search should judge against. The sweep's freshly traced outline has
        # drifted outwards by then — 7.03 x 6.65 m against the frozen
        # 5.95 x 4.48 m on the empty room — and a wall that far inside the
        # supposed room reads as furniture. See `extract`.
        trusted = None
        if self._frozen_outline is not None:
            trusted = [(p.x_m, p.y_m) for p in self._frozen_outline.polygon]

        try:
            room = self.extractor.extract(
                self.grid,
                self.pose,
                self.robot_id,
                datetime.now(UTC).isoformat(),
                boundary_polygon=trusted,
            )
        except Exception:
            # A malformed grid must not take the whole service down; the next
            # refresh will try again with more data.
            logger.exception("Room extraction failed")
            return

        if self._frozen_outline is not None:
            # Keep the boundary the perimeter lap measured, and take only the
            # obstacles from the fresh extraction. Coverage still moves,
            # because the sweep genuinely does observe more of the floor.
            frozen = self._frozen_outline
            room.polygon = frozen.polygon
            room.area_m2 = frozen.area_m2
            room.perimeter_m = frozen.perimeter_m
            room.bounding_width_m = frozen.bounding_width_m
            room.bounding_height_m = frozen.bounding_height_m
            room.is_closed = frozen.is_closed

        self.room = room

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
            self._frozen_outline = None
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
