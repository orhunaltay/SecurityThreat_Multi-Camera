"""System orchestrator for multi-camera security tracking."""
from __future__ import annotations

import threading
import time
from typing import List

import numpy as np

from multi_camera.camera_node import CameraNode
from multi_camera.communication_broker import CommunicationBroker
from multi_camera.feature_extractor import FeatureExtractor
from multi_camera.global_tracker import GlobalTracker


def dummy_frame_source() -> np.ndarray:
    """Mock function returning an empty frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def start_tracker_loop(broker: CommunicationBroker, tracker: GlobalTracker) -> threading.Thread:
    """Start a background thread to consume alerts for the global tracker."""
    queue = broker.subscribe()

    def loop() -> None:
        while True:
            for alert in CommunicationBroker.drain(queue):
                if alert["type"] == "NEW_THREAT":
                    gid = tracker.register_new_threat(
                        alert["camera_id"], alert["signature"], alert["timestamp"]
                    )
                    broker.publish_id_assignment(gid, alert["camera_id"], alert["signature"], alert["timestamp"])
                elif alert["type"] == "REID_MATCH":
                    tracker.handle_reid_match(
                        alert["global_id"], alert["camera_id"], alert.get("location"), alert["timestamp"]
                    )
            time.sleep(0.1)

    thread = threading.Thread(target=loop, daemon=True, name="GlobalTrackerLoop")
    thread.start()
    return thread


def main() -> None:
    """Initialize system components and start camera nodes."""
    broker = CommunicationBroker()
    tracker = GlobalTracker()
    feature_extractor = FeatureExtractor()

    tracker_thread = start_tracker_loop(broker, tracker)

    cameras: List[CameraNode] = []
    for i in range(2):  # example with two cameras
        node = CameraNode(f"camera_{i}", broker, feature_extractor, frame_source=dummy_frame_source)
        node.start()
        cameras.append(node)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for cam in cameras:
            cam.stop()
        tracker_thread.join(timeout=1)


if __name__ == "__main__":
    main()
