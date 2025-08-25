"""Simple message broker abstraction for camera nodes."""
from __future__ import annotations

from typing import Any, Dict, Iterable

import zmq


class CommunicationBroker:
    """ZeroMQ based publish/subscribe broker.

    A single *PUB* socket is bound during initialisation. Subscribers
    connect via their own *SUB* sockets created by :meth:`subscribe`.
    Messages are sent as JSON dictionaries.
    """

    def __init__(self, address: str = "tcp://127.0.0.1:5556") -> None:
        self._context = zmq.Context.instance()
        self._address = address
        self._pub_socket = self._context.socket(zmq.PUB)
        self._pub_socket.bind(address)

    # ------------------------------------------------------------------
    def subscribe(self) -> zmq.Socket:
        """Connect a new subscriber socket to the broker."""
        sub = self._context.socket(zmq.SUB)
        sub.connect(self._address)
        # Subscribe to all topics
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        return sub

    # ------------------------------------------------------------------
    def publish_threat_alert(self, camera_id: str, signature: Any, timestamp: float) -> None:
        """Broadcast a new threat alert to all subscribers."""
        alert = {
            "type": "NEW_THREAT",
            "camera_id": camera_id,
            "signature": signature,
            "timestamp": timestamp,
        }
        self._broadcast(alert)

    # ------------------------------------------------------------------
    def publish_id_assignment(self, global_id: str, camera_id: str, signature: Any, timestamp: float) -> None:
        """Broadcast assignment of a global ID for a new threat."""
        alert = {
            "type": "GLOBAL_ID_ASSIGN",
            "global_id": global_id,
            "camera_id": camera_id,
            "signature": signature,
            "timestamp": timestamp,
        }
        self._broadcast(alert)

    # ------------------------------------------------------------------
    def publish_reid_match(self, global_id: str, camera_id: str, location: Any, timestamp: float) -> None:
        """Broadcast a re-identification match from a camera node."""
        alert = {
            "type": "REID_MATCH",
            "global_id": global_id,
            "camera_id": camera_id,
            "location": location,
            "timestamp": timestamp,
        }
        self._broadcast(alert)

    # ------------------------------------------------------------------
    def _broadcast(self, alert: Dict[str, Any]) -> None:
        """Send an alert to all subscribers over the network."""
        self._pub_socket.send_json(alert)

    # ------------------------------------------------------------------
    @staticmethod
    def drain(socket: zmq.Socket) -> Iterable[Dict[str, Any]]:
        """Yield all currently queued alerts from a subscriber socket."""
        alerts = []
        while True:
            try:
                alerts.append(socket.recv_json(flags=zmq.NOBLOCK))
            except zmq.Again:
                break
        return alerts
