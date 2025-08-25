"""System orchestrator for multi-camera security tracking."""
from __future__ import annotations

import argparse
import threading
import time
from typing import Generator, List, Union

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None

from multi_camera.camera_node import CameraNode
from multi_camera.communication_broker import CommunicationBroker
from multi_camera.feature_extractor import FeatureExtractor
from multi_camera.global_tracker import GlobalTracker


def frame_source_generator(source: Union[int, str]) -> Generator[np.ndarray, None, None]:
    """Yield frames from an RTSP stream or webcam index.

    If :mod:`cv2` is unavailable the generator emits blank frames so the rest
    of the system can still be exercised.

    Args:
        source: Integer webcam index or RTSP URL understood by ``cv2.VideoCapture``.

    Yields:
        Consecutive frames read from the video source.
    """

    if cv2 is None:
        while True:
            yield np.zeros((480, 640, 3), dtype=np.uint8)

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


def main(argv: List[str] | None = None) -> None:
    """Initialize system components and start camera nodes."""
    parser = argparse.ArgumentParser(description="Run the multi-camera demo")
    parser.add_argument(
        "sources",
        nargs="*",
        default=["0"],
        help="Camera indices or RTSP URLs (default: 0)",
    )
    args = parser.parse_args(argv)

    sources: List[Union[int, str]] = []
    for src in args.sources:
        try:
            sources.append(int(src))
        except ValueError:
            sources.append(src)

    broker = CommunicationBroker()
    tracker = GlobalTracker()
    feature_extractor = FeatureExtractor()

    tracker_thread = start_tracker_loop(broker, tracker)

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
