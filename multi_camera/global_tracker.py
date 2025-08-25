"""Global tracking module maintaining targets across cameras."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TargetState:
    """State information for a single tracked target."""

    global_id: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_camera: Optional[str] = None


class GlobalTracker:
    """Central tracker fusing detections from multiple cameras."""

    def __init__(self) -> None:
        self.targets: Dict[str, TargetState] = {}
        self._next_id = 0

    # ------------------------------------------------------------------
    def register_new_threat(self, camera_id: str, signature: Any, timestamp: float) -> str:
        """Register a new threat and return the global ID."""
        global_id = f"T{self._next_id}"; self._next_id += 1
        state = TargetState(global_id=global_id, current_camera=camera_id)
        state.history.append({"camera_id": camera_id, "timestamp": timestamp})
        self.targets[global_id] = state
        return global_id

    # ------------------------------------------------------------------
    def handle_reid_match(self, global_id: str, camera_id: str, location: Any, timestamp: float) -> None:
        """Update existing target state with new camera location."""
        state = self.targets.setdefault(global_id, TargetState(global_id=global_id))
        state.current_camera = camera_id
        state.history.append({
            "camera_id": camera_id,
            "location": location,
            "timestamp": timestamp,
        })

    # ------------------------------------------------------------------
    def get_target_trajectory(self, global_id: str) -> List[Dict[str, Any]]:
        """Return trajectory history for a given global target ID."""
        state = self.targets.get(global_id)
        return state.history if state else []
