"""In-memory latest annotated frame per camera.

All MJPEG viewers share the single AI inference result. A browser viewer never
starts a second inference loop, and no placeholder frame is ever produced in
live mode.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class AnnotatedFrame:
    jpeg: bytes
    created_at: float


class StreamHub:
    """Thread-safe holder for the newest annotated JPEG of each camera."""

    def __init__(self, max_age_seconds: float = 5.0) -> None:
        self._frames: dict[str, AnnotatedFrame] = {}
        self._lock = threading.Lock()
        self._max_age = max_age_seconds

    def publish(self, camera_id: str, jpeg: bytes) -> None:
        with self._lock:
            self._frames[camera_id] = AnnotatedFrame(jpeg, time.monotonic())

    def latest(self, camera_id: str) -> Optional[bytes]:
        with self._lock:
            frame = self._frames.get(camera_id)
        if frame is None:
            return None
        if (time.monotonic() - frame.created_at) > self._max_age:
            return None
        return frame.jpeg

    def has(self, camera_id: str) -> bool:
        return self.latest(camera_id) is not None

    def drop(self, camera_id: str) -> None:
        with self._lock:
            self._frames.pop(camera_id, None)