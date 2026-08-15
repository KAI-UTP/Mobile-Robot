"""Shared MQTT helper — reconnect-capable wrapper around paho.

Kept deliberately close to the SmartClean Twin helper so the two projects
read the same way.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


def build_client(
    client_id: str,
    on_message: Callable | None = None,
    subscriptions: list[str] | None = None,
    clean_session: bool = True,
) -> mqtt.Client:
    """Return a connected paho Client with retry logic."""
    host = os.environ.get("MQTT_HOST", "mosquitto")
    port = int(os.environ.get("MQTT_PORT", "1883"))

    client = mqtt.Client(client_id=client_id, clean_session=clean_session)

    if on_message:
        client.on_message = on_message

    def _on_connect(c: mqtt.Client, userdata, flags, rc: int) -> None:
        if rc == 0:
            logger.info("MQTT connected (host=%s port=%d)", host, port)
            for topic in subscriptions or []:
                c.subscribe(topic)
                logger.info("Subscribed to %s", topic)
        else:
            logger.error("MQTT connect failed rc=%d", rc)

    client.on_connect = _on_connect
    client.on_disconnect = lambda c, u, rc: logger.warning("MQTT disconnected rc=%d", rc)

    for attempt in range(1, 11):
        try:
            client.connect(host, port, keepalive=60)
            client.loop_start()
            return client
        except Exception as exc:
            logger.warning("MQTT connect attempt %d failed: %s", attempt, exc)
            time.sleep(min(2**attempt, 30))

    raise RuntimeError("Could not connect to MQTT broker after 10 attempts")
