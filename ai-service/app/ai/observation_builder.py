"""Pure translation of detector output into an immutable observation view.

Stateless and deterministic: metadata only, no inference, no database access, no
event creation, no rule decisions, and no mutation of `FrameDetections` or any
`Detection`. Building observations must stay effectively free in the inference
hot path.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..domain.models import FrameDetections, SourceMode
from ..domain.observations import FrameObservations, PersonObservation


def build_person_observation(detection) -> PersonObservation:  # noqa: ANN001 - Detection
    """Projects one person detection into an immutable observation."""
    return PersonObservation(
        person_tracking_id=detection.tracking_id,
        person_bbox=detection.bbox,
        confidence=float(detection.confidence),
    )


def build_frame_observations(
    *,
    camera_id: str,
    detections: FrameDetections,
    frame_sequence: Optional[int] = None,
    observed_at: Optional[datetime] = None,
    source_mode: Optional[SourceMode] = None,
) -> FrameObservations:
    """`FrameDetections` + frame metadata -> derived `FrameObservations`."""
    return FrameObservations(
        camera_id=camera_id,
        persons=tuple(build_person_observation(person) for person in detections.persons),
        frame_sequence=frame_sequence,
        observed_at=observed_at,
        source_mode=source_mode,
    )
