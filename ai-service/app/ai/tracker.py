"""Per-camera tracking bookkeeping.

Ultralytics/ByteTrack maintains the actual track association; this module keeps
the per-camera view of it and guarantees that tracking IDs are treated as
temporary session labels only. No identity, face or biometric matching exists
anywhere in this service.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..domain.models import Detection, FrameDetections


@dataclass
class TrackState:
    tracking_id: str
    last_seen: float
    frames: int = 1


@dataclass
class CameraTracks:
    """Temporary, per-session track table for one camera."""

    camera_id: str
    ttl_seconds: float = 5.0
    tracks: dict[str, TrackState] = field(default_factory=dict)

    def update(self, detections: FrameDetections, now: Optional[float] = None) -> None:
        moment = time.monotonic() if now is None else now
        for person in detections.persons:
            if not person.tracking_id:
                continue
            state = self.tracks.get(person.tracking_id)
            if state is None:
                self.tracks[person.tracking_id] = TrackState(person.tracking_id, moment)
            else:
                state.last_seen = moment
                state.frames += 1
        self._expire(moment)

    def _expire(self, now: float) -> None:
        for tracking_id in [
            key for key, state in self.tracks.items() if (now - state.last_seen) > self.ttl_seconds
        ]:
            del self.tracks[tracking_id]

    def is_stable(self, tracking_id: Optional[str], min_frames: int = 2) -> bool:
        """A brand-new flickering track should not immediately drive alerts."""
        if not tracking_id:
            return False
        state = self.tracks.get(tracking_id)
        return bool(state and state.frames >= min_frames)

    @property
    def active_count(self) -> int:
        return len(self.tracks)


class TrackRegistry:
    """Keeps tracking state strictly separate per camera."""

    def __init__(self) -> None:
        self._cameras: dict[str, CameraTracks] = {}

    def for_camera(self, camera_id: str) -> CameraTracks:
        if camera_id not in self._cameras:
            self._cameras[camera_id] = CameraTracks(camera_id)
        return self._cameras[camera_id]

    def drop(self, camera_id: str) -> None:
        self._cameras.pop(camera_id, None)

    def person_by_id(self, camera_id: str, detections: FrameDetections, tracking_id: str) -> Optional[Detection]:
        for person in detections.persons:
            if person.tracking_id == tracking_id:
                return person
        return None