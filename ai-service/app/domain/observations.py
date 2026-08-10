"""Immutable, derived per-frame observation view.

`FrameDetections` remains the detector's source of truth. The structures here
are a *derived*, read-only projection of a single analysed frame, carrying only
metadata that is already known from that frame. They exist so future behavioural
engines can consume a stable view without touching the detector result or the
frozen mobile-phone pipeline.

Deliberately absent (and out of scope here): pose, keypoints, regions, desk
state, behavioural scores and any phone-use conclusion. No history is stored:
these objects describe one frame and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .geometry import BBox
from .models import SourceMode


@dataclass(frozen=True, slots=True)
class PersonObservation:
    """One person as observed in the current analysed frame."""

    #: Tracker identity when the tracker provided one, otherwise ``None``.
    person_tracking_id: Optional[str]
    #: Normalized (0..1) person bounding box of this frame.
    person_bbox: BBox
    #: Detector confidence for this person.
    confidence: float


@dataclass(frozen=True, slots=True)
class FrameObservations:
    """Immutable metadata view of one analysed frame."""

    camera_id: str
    persons: tuple[PersonObservation, ...] = ()
    #: Accepted capture sequence of the frame (from CaptureWorker/FrameGate).
    frame_sequence: Optional[int] = None
    observed_at: Optional[datetime] = None
    source_mode: Optional[SourceMode] = None

    @property
    def person_count(self) -> int:
        return len(self.persons)
