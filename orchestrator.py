"""System orchestrator for multi-camera security tracking."""
from __future__ import annotations

import threading
import time
from typing import Generator, List, Union

import cv2
import numpy as np

from multi_camera.camera_node import CameraNode
from multi_camera.communication_broker import CommunicationBroker
from multi_camera.feature_extractor import FeatureExtractor
from multi_camera.global_tracker import GlobalTracker


def frame_source_generator(source: Union[int, str]) -> Generator[np.ndarray, None, None]:
    """Yield frames from an RTSP stream or webcam index.

    Args:
        source: Integer webcam index or RTSP URL understood by ``cv2.VideoCapture``.

    Yields:
        Consecutive frames read from the video source.
    """

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source {source}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


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

    # Example sources: replace with actual RTSP URLs or device indices
    sources: List[Union[int, str]] = [0, 1]

    cameras: List[CameraNode] = []
    for i, src in enumerate(sources):
        gen = frame_source_generator(src)
        node = CameraNode(
            f"camera_{i}",
            broker,
            feature_extractor,
            frame_source=lambda gen=gen: next(gen, None),
        )
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
