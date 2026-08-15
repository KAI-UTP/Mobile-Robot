"""MQTT topic constants for the RoomMapper robot.

Topic layout mirrors SmartClean Twin: a single prefix, one robot id, then
one subtree per data class.  Raw topics carry whatever the robot sent;
validated topics carry data that has passed Pydantic validation.
"""


class Topics:
    PREFIX = "roommapper"
    ROBOT_ID = "MR3W01"  # Mobile Robot, 3-Wheel, unit 01
    _BASE = f"{PREFIX}/{ROBOT_ID}"

    # ── Raw sensor streams (published by firmware or the Bluetooth bridge) ──
    SENSORS_RAW = f"{_BASE}/sensors/raw"
    SENSORS_VALIDATED = f"{_BASE}/sensors/validated"

    # ── Derived estimates ──────────────────────────────────────────────────
    POSE = f"{_BASE}/pose"  # fused pose from the localization service
    SCAN = f"{_BASE}/scan"  # range readings tagged with the pose they were taken at
    MAP_UPDATE = f"{_BASE}/map/update"  # incremental occupancy grid changes
    MAP_SNAPSHOT = f"{_BASE}/map/snapshot"  # full grid, published periodically
    ROOM = f"{_BASE}/room"  # extracted room polygon + area

    # ── Control ────────────────────────────────────────────────────────────
    COMMAND = f"{_BASE}/command"
    COMMAND_ALL = f"{_BASE}/command/#"
    ACK = f"{_BASE}/ack"

    # ── Operations ─────────────────────────────────────────────────────────
    ALERT = f"{_BASE}/alert"
    SERVICE_HEALTH = f"{_BASE}/service/health"

    @classmethod
    def for_robot(cls, robot_id: str) -> dict[str, str]:
        """Return the full topic set for a specific robot id.

        Useful when running more than one robot against the same broker.
        """
        base = f"{cls.PREFIX}/{robot_id}"
        return {
            "sensors_raw": f"{base}/sensors/raw",
            "sensors_validated": f"{base}/sensors/validated",
            "pose": f"{base}/pose",
            "scan": f"{base}/scan",
            "map_update": f"{base}/map/update",
            "map_snapshot": f"{base}/map/snapshot",
            "room": f"{base}/room",
            "command": f"{base}/command",
            "ack": f"{base}/ack",
            "alert": f"{base}/alert",
            "service_health": f"{base}/service/health",
        }
