"""Pure, stateless builder: TrackedPoseObservation -> TrackedBodyFeatures.

This module contains GEOMETRY ONLY. It has no clock, no global state, no
runtime, no camera manager and no model inference, and it never mutates its
input. Behaviour, thresholds, pair/inter-person geometry, temporal history and
events are all deliberately out of scope.

Architectural note for a later phase (NOT implemented here): future
``document_exchange`` events MUST require an explicit armed exam-monitoring
state; starting a camera must never arm it by itself.

Keypoints are read through the semantic lookup
(``observation.available_keypoint(PoseKeypointName.X)``) — raw COCO indices are
never used here.
"""

from __future__ import annotations

import math
from typing import Optional

from app.domain.body_features import (
    ArmFeatures,
    BodySide,
    SideAvailability,
    TrackedBodyFeatures,
    WristFeatures,
)
from app.domain.tracked_pose_observations import (
    TrackedPoseKeypoint,
    TrackedPoseObservation,
)


def _point(keypoint: Optional[TrackedPoseKeypoint]) -> Optional[tuple[float, float]]:
    """Frame coordinates of a genuinely available keypoint, else ``None``."""
    if keypoint is None or not keypoint.available:
        return None
    if keypoint.x is None or keypoint.y is None:
        return None
    return (float(keypoint.x), float(keypoint.y))


def _distance(
    a: Optional[tuple[float, float]], b: Optional[tuple[float, float]]
) -> Optional[float]:
    if a is None or b is None:
        return None
    value = math.hypot(a[0] - b[0], a[1] - b[1])
    return value if math.isfinite(value) else None


def _relative_to_person(value: Optional[float], person_diagonal: float) -> Optional[float]:
    if value is None or person_diagonal <= 0.0:
        return None
    ratio = value / person_diagonal
    return ratio if math.isfinite(ratio) else None


def _angle_degrees(
    shoulder: Optional[tuple[float, float]],
    elbow: Optional[tuple[float, float]],
    wrist: Optional[tuple[float, float]],
) -> Optional[float]:
    """Interior angle at the elbow, or ``None`` for missing/degenerate input."""
    if shoulder is None or elbow is None or wrist is None:
        return None
    ux, uy = shoulder[0] - elbow[0], shoulder[1] - elbow[1]
    vx, vy = wrist[0] - elbow[0], wrist[1] - elbow[1]
    u_len = math.hypot(ux, uy)
    v_len = math.hypot(vx, vy)
    if u_len <= 0.0 or v_len <= 0.0:
        return None
    cosine = (ux * vx + uy * vy) / (u_len * v_len)
    cosine = max(-1.0, min(1.0, cosine))
    degrees = math.degrees(math.acos(cosine))
    return degrees if math.isfinite(degrees) else None


def _wrist_features(
    side: BodySide, keypoint: Optional[TrackedPoseKeypoint]
) -> WristFeatures:
    if keypoint is None or not keypoint.available:
        return WristFeatures(side=side, available=False)
    return WristFeatures(
        side=side,
        available=True,
        x=keypoint.x,
        y=keypoint.y,
        # Person-relative coordinates are passed through unclamped on purpose.
        relative_position=keypoint.relative_position,
        inside_person=keypoint.inside_person,
        keypoint_confidence=keypoint.confidence,
    )


def build_arm_features(
    observation: TrackedPoseObservation, side: BodySide
) -> ArmFeatures:
    """Geometry of ONE arm, computed independently of the other side."""
    shoulder_kp = observation.available_keypoint(side.shoulder)
    elbow_kp = observation.available_keypoint(side.elbow)
    wrist_kp = observation.available_keypoint(side.wrist)

    shoulder = _point(shoulder_kp)
    elbow = _point(elbow_kp)
    wrist = _point(wrist_kp)

    diagonal = observation.person_bbox.diagonal

    wrist_to_elbow = _distance(wrist, elbow)
    elbow_to_shoulder = _distance(elbow, shoulder)
    shoulder_to_wrist = _distance(shoulder, wrist)

    ratio: Optional[float] = None
    if (
        wrist_to_elbow is not None
        and elbow_to_shoulder is not None
        and shoulder_to_wrist is not None
    ):
        segment_sum = wrist_to_elbow + elbow_to_shoulder
        if segment_sum > 0.0:
            candidate = shoulder_to_wrist / segment_sum
            if math.isfinite(candidate):
                ratio = candidate

    return ArmFeatures(
        side=side,
        availability=SideAvailability(
            shoulder_available=shoulder is not None,
            elbow_available=elbow is not None,
            wrist_available=wrist is not None,
        ),
        wrist=_wrist_features(side, wrist_kp),
        wrist_to_elbow_distance=wrist_to_elbow,
        elbow_to_shoulder_distance=elbow_to_shoulder,
        shoulder_to_wrist_distance=shoulder_to_wrist,
        wrist_to_elbow_distance_relative_to_person=_relative_to_person(
            wrist_to_elbow, diagonal
        ),
        elbow_to_shoulder_distance_relative_to_person=_relative_to_person(
            elbow_to_shoulder, diagonal
        ),
        shoulder_to_wrist_distance_relative_to_person=_relative_to_person(
            shoulder_to_wrist, diagonal
        ),
        shoulder_wrist_to_segment_sum_ratio=ratio,
        elbow_angle_degrees=_angle_degrees(shoulder, elbow, wrist),
        shoulder_confidence=shoulder_kp.confidence if shoulder_kp else None,
        elbow_confidence=elbow_kp.confidence if elbow_kp else None,
    )


def build_wrist_arm_features(
    observation: TrackedPoseObservation,
) -> TrackedBodyFeatures:
    """Pure derivation of wrist/arm geometry for ONE tracked person, ONE frame."""
    if not isinstance(observation, TrackedPoseObservation):
        raise TypeError("observation must be a TrackedPoseObservation")
    return TrackedBodyFeatures(
        person_tracking_id=observation.person_tracking_id,
        person_index=observation.person_index,
        person_bbox=observation.person_bbox,
        person_confidence=observation.person_confidence,
        pose_confidence=observation.pose_confidence,
        left_arm=build_arm_features(observation, BodySide.LEFT),
        right_arm=build_arm_features(observation, BodySide.RIGHT),
    )
