"""Immutable pose domain contract (pose output only).

This module is pure: no Ultralytics, OpenCV, network or database access. It
describes ONE analysed frame of human pose estimation in normalized 0..1 frame
coordinates and nothing else.

Deliberately absent (and out of scope here): person tracking identity, region
facts, head-down / wrist-low / lean features, behavioural scores and any
concealed-device conclusion. Behavioural thresholds are NOT defined here;
Task 2G will calibrate them from real footage.

Truthfulness rules
------------------
* A keypoint is either available (finite normalized coordinates inside the
  frame plus, when the source supplies it, a finite confidence in 0..1) or it
  is explicitly unavailable with ``x``/``y`` set to ``None``.
* ``(0.0, 0.0)`` is never assumed to be an observed joint: the pinned
  Ultralytics 8.3.55 ``Keypoints`` implementation masks low-confidence
  keypoints by zeroing their coordinates, so availability is decided from the
  confidence channel, never from coordinates alone.
* When confidence is genuinely absent, it stays ``None`` rather than being
  invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .geometry import BBox


class PoseKeypointName(str, Enum):
    """Semantic COCO human-pose keypoint names (no magic integers elsewhere)."""

    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


#: Canonical COCO-17 ordering: index == position in this tuple.
COCO_17_KEYPOINTS: tuple[PoseKeypointName, ...] = (
    PoseKeypointName.NOSE,
    PoseKeypointName.LEFT_EYE,
    PoseKeypointName.RIGHT_EYE,
    PoseKeypointName.LEFT_EAR,
    PoseKeypointName.RIGHT_EAR,
    PoseKeypointName.LEFT_SHOULDER,
    PoseKeypointName.RIGHT_SHOULDER,
    PoseKeypointName.LEFT_ELBOW,
    PoseKeypointName.RIGHT_ELBOW,
    PoseKeypointName.LEFT_WRIST,
    PoseKeypointName.RIGHT_WRIST,
    PoseKeypointName.LEFT_HIP,
    PoseKeypointName.RIGHT_HIP,
    PoseKeypointName.LEFT_KNEE,
    PoseKeypointName.RIGHT_KNEE,
    PoseKeypointName.LEFT_ANKLE,
    PoseKeypointName.RIGHT_ANKLE,
)

#: Expected keypoint count of the default MVP human pose schema.
COCO_17_KEYPOINT_COUNT = len(COCO_17_KEYPOINTS)

#: Name -> canonical COCO index.
COCO_17_INDEX_BY_NAME: dict[PoseKeypointName, int] = {
    name: index for index, name in enumerate(COCO_17_KEYPOINTS)
}


def coco_17_index(name: PoseKeypointName) -> int:
    """Canonical COCO index of a semantic keypoint name."""
    return COCO_17_INDEX_BY_NAME[name]


class PoseStatus(str, Enum):
    """Outcome of one pose inference attempt.

    ``OK`` covers a successful inference, including a successful inference that
    found zero people. Every other member means the frame produced NO usable
    pose evidence; such results always carry an empty ``instances`` tuple.
    """

    OK = "ok"
    MODEL_UNAVAILABLE = "model_unavailable"
    INFERENCE_FAILED = "inference_failed"
    MALFORMED_RESULT = "malformed_result"
    UNSUPPORTED_POSE_SCHEMA = "unsupported_pose_schema"
    KEYPOINTS_ABSENT = "keypoints_absent"


@dataclass(frozen=True, slots=True)
class PoseKeypoint:
    """One semantic joint of one pose instance."""

    name: PoseKeypointName
    index: int
    available: bool
    x: Optional[float] = None
    y: Optional[float] = None
    confidence: Optional[float] = None

    @classmethod
    def unavailable(
        cls,
        name: PoseKeypointName,
        index: int,
        confidence: Optional[float] = None,
    ) -> "PoseKeypoint":
        """An explicitly unobserved joint: never carries coordinates."""
        return cls(name=name, index=index, available=False, confidence=confidence)


@dataclass(frozen=True, slots=True)
class PoseInstance:
    """One detected human pose in normalized frame coordinates.

    ``bbox`` is pose-model instance geometry only. It is NOT a tracking
    identity: this contract intentionally carries no persistent subject id.
    """

    bbox: BBox
    keypoints: tuple[PoseKeypoint, ...]
    confidence: Optional[float] = None

    def keypoint(self, name: PoseKeypointName) -> Optional[PoseKeypoint]:
        """Safe lookup by semantic name; ``None`` when the schema lacks it."""
        for keypoint in self.keypoints:
            if keypoint.name == name:
                return keypoint
        return None

    def available_keypoint(self, name: PoseKeypointName) -> Optional[PoseKeypoint]:
        """Returns the keypoint only when it was genuinely observed."""
        keypoint = self.keypoint(name)
        if keypoint is None or not keypoint.available:
            return None
        return keypoint

    @property
    def available_keypoint_count(self) -> int:
        return sum(1 for keypoint in self.keypoints if keypoint.available)


@dataclass(frozen=True, slots=True)
class PoseFrameResult:
    """Immutable pose result of exactly one analysed frame."""

    status: PoseStatus
    instances: tuple[PoseInstance, ...] = ()
    reason: Optional[str] = None
    model_name: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status is PoseStatus.OK

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    @classmethod
    def failure(
        cls,
        status: PoseStatus,
        reason: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> "PoseFrameResult":
        """Degraded result: never carries pose instances."""
        return cls(status=status, instances=(), reason=reason, model_name=model_name)
