"""System orchestrator for multi-camera security tracking."""
from __future__ import annotations

import logging
import threading
import time
from typing import List, Tuple

import numpy as np

from multi_camera.camera_node import CameraNode
from multi_camera.communication_broker import CommunicationBroker
from multi_camera.feature_extractor import FeatureExtractor
from multi_camera.global_tracker import GlobalTracker


def dummy_frame_source() -> np.ndarray:
    """Mock function returning an empty frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def start_tracker_loop(
    broker: CommunicationBroker, tracker: GlobalTracker
) -> Tuple[threading.Thread, threading.Event]:
    """Start a background thread to consume alerts for the global tracker."""

    queue = broker.subscribe()
    stop_event = threading.Event()

    def loop() -> None:
        logger = logging.getLogger("tracker")
        try:
            while not stop_event.is_set():
                try:
                    for alert in CommunicationBroker.drain(queue):
                        if alert["type"] == "NEW_THREAT":
                            gid = tracker.register_new_threat(
                                alert["camera_id"], alert["signature"], alert["timestamp"]
                            )
                            broker.publish_id_assignment(
                                gid, alert["camera_id"], alert["signature"], alert["timestamp"]
                            )
                        elif alert["type"] == "REID_MATCH":
                            tracker.handle_reid_match(
                                alert["global_id"],
                                alert["camera_id"],
                                alert.get("location"),
                                alert["timestamp"],
                            )
                except Exception:
                    # Continue processing even if a single alert fails
                    logger.exception("Error processing alert")
                time.sleep(0.1)
        except Exception:
            logger.exception("Tracker loop crashed")
        finally:
            logger.info("Tracker loop stopped")

    thread = threading.Thread(target=loop, daemon=False, name="GlobalTrackerLoop")
    thread.start()
    return thread, stop_event


def main() -> None:
    """Initialize system components and start camera nodes."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger = logging.getLogger("orchestrator")

    broker = CommunicationBroker()
    tracker = GlobalTracker()
    feature_extractor = FeatureExtractor()

    tracker_thread, tracker_stop = start_tracker_loop(broker, tracker)

    cameras: List[CameraNode] = []
    for i in range(2):  # example with two cameras
        node = CameraNode(
            f"camera_{i}",
            broker,
            feature_extractor,
            frame_source=dummy_frame_source,
        )
        node.start()
        logger.info("Camera node started", extra={"camera_id": node.camera_id})
        cameras.append(node)

    try:
        while True:
            time.sleep(1)
            for cam in cameras:
                if not cam.is_alive():
                    logger.error(
                        "Camera thread stopped unexpectedly",
                        extra={"camera_id": cam.camera_id},
                    )
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        for cam in cameras:
            cam.stop()
            cam.join(timeout=2)
            if cam.is_alive():
                logger.warning(
                    "Camera thread did not terminate within timeout",
                    extra={"camera_id": cam.camera_id},
                )
        tracker_stop.set()
        tracker_thread.join(timeout=2)
        if tracker_thread.is_alive():
            logger.warning("Tracker thread did not terminate within timeout")


if __name__ == "__main__":
    main()
