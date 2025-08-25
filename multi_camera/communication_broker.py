"""Simple message broker abstraction for camera nodes."""
from __future__ import annotations

import queue
from typing import Any, Dict, Iterable, List


class CommunicationBroker:
    """In-memory message broker supporting basic publish/subscribe.

    This implementation keeps a separate queue for each subscriber to
    emulate broadcast semantics.
    """

    def __init__(self) -> None:
        self._subscribers: List["queue.Queue[Dict[str, Any]]"] = []

    # ------------------------------------------------------------------
    def subscribe(self) -> "queue.Queue[Dict[str, Any]]":
        """Register a new subscriber and return its queue."""
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._subscribers.append(q)
        return q

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
        """Push an alert to all subscriber queues."""
        for q in self._subscribers:
            q.put(alert)

    # ------------------------------------------------------------------
    @staticmethod
    def drain(queue_: "queue.Queue[Dict[str, Any]]") -> Iterable[Dict[str, Any]]:
        """Yield all currently queued alerts from a subscriber queue."""
        alerts = []
        while True:
            try:
                alerts.append(queue_.get_nowait())
            except queue.Empty:
                break
        return alerts
