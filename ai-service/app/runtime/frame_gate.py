"""Distinct-frame gate.

Guarantees that each *captured* frame is analysed at most once, so a frozen or
repeated image can never inflate `detection_frame_count` or satisfy
`min_matching_frames`. Pure logic: no OpenCV, no threads, no queue — only the
newest frame is ever considered, so latency stays bounded.
"""

from __future__ import annotations


class FrameGate:
    """Accepts a capture sequence number only when it is new for that camera."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def accept(self, camera_id: str, sequence: int) -> bool:
        if self._seen.get(camera_id) == sequence:
            return False
        self._seen[camera_id] = sequence
        return True

    def reset(self, camera_id: str) -> None:
        self._seen.pop(camera_id, None)


__all__ = ["FrameGate"]
