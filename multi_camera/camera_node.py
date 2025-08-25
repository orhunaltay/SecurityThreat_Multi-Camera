"""Camera node module for multi-camera security system."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .communication_broker import CommunicationBroker
from .feature_extractor import FeatureExtractor


@dataclass
class Detection:
    """Simple structure holding detection information."""
    bbox: tuple[int, int, int, int]
    confidence: float


class CameraNode(threading.Thread):
    """Processes frames from a single camera and shares threat signatures.

    Each camera node runs in its own thread. It monitors the video feed,
    performs detection and feature extraction, and communicates with
    other nodes through the communication broker.
    """

    def __init__(
        self,
        camera_id: str,
        broker: CommunicationBroker,
        feature_extractor: FeatureExtractor,
        *,
        frame_source: Optional[Any] = None,
        poll_interval: float = 0.05,
    ) -> None:
        super().__init__(name=f"CameraNode-{camera_id}", daemon=False)
        self.camera_id = camera_id
        self.broker = broker
        self.feature_extractor = feature_extractor
        self.frame_source = frame_source
        self.poll_interval = poll_interval
        self.alert_queue = broker.subscribe()
        self._stop_event = threading.Event()
        self._logger = logging.getLogger(f"camera.{camera_id}")

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main thread loop processing incoming frames."""
        self._logger.info("Camera node started")
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self._get_frame()
                    if frame is not None:
                        self.process_frame(frame)
                except Exception:
                    self._logger.exception("Error during frame processing")
                time.sleep(self.poll_interval)
        except Exception:
            self._logger.exception("Camera thread crashed")
        finally:
            self._logger.info("Camera node stopped")

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> None:
        """Process a single frame for threats and re-identification.

        Args:
            frame: Raw image array from the camera.
        """
        detection = self.detect_threat(frame)
        if detection:
            bbox = detection.bbox
            cropped = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            signature = self.extract_features(cropped)
            self.broker.publish_threat_alert(self.camera_id, signature, time.time())

        # Check for matches of existing targets
        for alert in CommunicationBroker.drain(self.alert_queue):
            self.search_for_target(frame, alert)

    # ------------------------------------------------------------------
    def detect_threat(self, frame: np.ndarray) -> Optional[Detection]:
        """Detect threats in the frame.

        Uses a placeholder detector to return a detection object or None.
        """
        return None  # TODO: integrate detector such as YOLO

    # ------------------------------------------------------------------
    def extract_features(self, cropped: np.ndarray) -> np.ndarray:
        """Generate a feature vector for the detected threat."""
        return self.feature_extractor.extract(cropped)

    # ------------------------------------------------------------------
    def search_for_target(self, frame: np.ndarray, alert: Dict[str, Any]) -> None:
        """Search the current frame for a target described by the alert."""
        # Placeholder: in real implementation compare features for re-id
        pass

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Stop the node processing loop."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    def _get_frame(self) -> Optional[np.ndarray]:
        """Retrieve the next frame from the camera source."""
        if callable(self.frame_source):
            try:
                return self.frame_source()
            except Exception:
                self._logger.exception("Frame source failed")
                self._stop_event.set()
        return None
