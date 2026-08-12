"""Pure builder: ONE TrackedPoseFrameResult -> ONE TrackedBodyFeatureFrame.

Same-frame provenance is structural: the whole frame's body-feature subjects are
derived inside this single call from a single authoritative tracked-pose frame,
so independently supplied body features from arbitrary frames can never be
assumed to belong together.

Geometry only. No behaviour, no thresholds, no temporal state, no events. Task
3B types are consumed unmodified.
"""

from __future__ import annotations

from app.ai.wrist_arm_feature_builder import build_wrist_arm_features
from app.domain.pair_geometry import (
    PairFrameStatus,
    TrackedBodyFeatureFrame,
)
from app.domain.tracked_pose_observations import (
    TrackedPoseFrameResult,
    TrackedPoseFrameStatus,
)


def build_body_feature_frame(
    frame: TrackedPoseFrameResult,
) -> TrackedBodyFeatureFrame:
    """Derives wrist/arm features for every safely associated tracked person.

    A degraded tracked-pose frame yields a degraded body-feature frame with zero
    subjects; no body subject is ever fabricated.
    """
    if not isinstance(frame, TrackedPoseFrameResult):
        raise TypeError("frame must be a TrackedPoseFrameResult")

    common = {
        "camera_id": frame.camera_id,
        "frame_sequence": frame.frame_sequence,
        "observed_at": frame.observed_at,
        "source_mode": frame.source_mode,
        "source_pose_status": frame.source_pose_status,
        "source_frame_status": frame.status,
        "unresolved_pose_count": len(frame.unresolved),
        "reason": frame.reason,
    }

    if frame.status is not TrackedPoseFrameStatus.OK:
        return TrackedBodyFeatureFrame(status=PairFrameStatus(frame.status.value), **common)

    subjects = tuple(
        build_wrist_arm_features(observation) for observation in frame.observations
    )
    return TrackedBodyFeatureFrame(
        status=PairFrameStatus.OK, subjects=subjects, **common
    )
