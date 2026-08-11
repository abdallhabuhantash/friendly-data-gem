"""Immutable pose domain contract (pose output only).

This module is pure: no Ultralytics, OpenCV, network or database access. It
describes ONE analysed frame of human pose estimation in normalized 0..1 frame
coordinates and nothing else.

Deliberately absent (and out of scope here): person tracking identity, region
facts, head-down / wrist-low / lean features, behavioural scores and any
concealed-device conclusion. Behavioural thresholds are NOT defined here;
Task 2G will calibrate them from real footage.

Truthfulness invariants (enforced by the dataclasses themselves)
---------------------------------------------------------------
* A ``PoseKeypoint`` exposed as ``available=True`` MUST carry usable normalized
  coordinates (finite, inside 0..1) AND an explicit finite confidence in 0..1.
* ``confidence=None`` can never coexist with ``available=True``. The supported
  MVP schema is confidence-bearing COCO-17; a coordinate-only pose result is
  NOT behavioural evidence and is reported as
  :attr:`PoseStatus.KEYPOINT_CONFIDENCE_ABSENT` instead. Visibility is never
  inferred from coordinates alone, so ``(0.0, 0.0)`` is never treated as an
  observed joint.
* ``available=False`` never carries coordinates.
* A ``PoseInstance`` carries exactly one supported COCO-17 keypoint schema in
  canonical order, with a positive normalized bbox fully inside the frame.
* A degraded (non-``OK``) ``PoseFrameResult`` never carries instances.
"""

from __future__ import annotations

import math
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
    #: Coordinates only: the source exposed no keypoint confidence channel, so
    #: no keypoint can be trusted as a behavioural observation.
    KEYPOINT_CONFIDENCE_ABSENT = "keypoint_confidence_absent"


class PoseContractError(ValueError):
    """Raised when a pose domain object would violate its own invariants."""


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def _in_unit_range(value: float) -> bool:
    return 0.0 <= float(value) <= 1.0


@dataclass(frozen=True, slots=True)
class PoseKeypoint:
    """One semantic joint of one pose instance.

    ``available=True`` requires finite normalized coordinates inside 0..1 and an
    explicit finite confidence in 0..1. ``available=False`` carries no
    coordinates.
    """

    name: PoseKeypointName
    index: int
    available: bool
    x: Optional[float] = None
    y: Optional[float] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, PoseKeypointName):
            raise PoseContractError(f"unknown pose keypoint name: {self.name!r}")
        canonical = COCO_17_INDEX_BY_NAME[self.name]
        if self.index != canonical:
            raise PoseContractError(
                f"{self.name.value} must use canonical COCO index {canonical}, got {self.index}"
            )
        if self.confidence is not None:
            if not _finite(self.confidence) or not _in_unit_range(self.confidence):
                raise PoseContractError(
                    f"{self.name.value} confidence must be finite in 0..1, got {self.confidence!r}"
                )
        if self.available:
            if self.x is None or self.y is None:
                raise PoseContractError(
                    f"{self.name.value} is available but has no coordinates"
                )
            for label, value in (("x", self.x), ("y", self.y)):
                if not _finite(value) or not _in_unit_range(value):
                    raise PoseContractError(
                        f"{self.name.value} {label} must be finite normalized 0..1, got {value!r}"
                    )
            if self.confidence is None:
                raise PoseContractError(
                    f"{self.name.value} is available but carries no confidence channel"
                )
        else:
            if self.x is not None or self.y is not None:
                raise PoseContractError(
                    f"{self.name.value} is unavailable and must not carry coordinates"
                )

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
    The keypoint tuple is exactly the supported COCO-17 schema in canonical
    order; alternate schemas must get their own explicit contract instead of
    silently reusing COCO indices.
    """

    bbox: BBox
    keypoints: tuple[PoseKeypoint, ...]
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        box = self.bbox
        if not isinstance(box, BBox):
            raise PoseContractError("pose instance bbox must be a BBox")
        for label, value in (
            ("x", box.x),
            ("y", box.y),
            ("width", box.width),
            ("height", box.height),
        ):
            if not _finite(value):
                raise PoseContractError(f"pose bbox {label} must be finite, got {value!r}")
        if box.width <= 0.0 or box.height <= 0.0:
            raise PoseContractError("pose bbox must have positive width and height")
        if not (
            _in_unit_range(box.x)
            and _in_unit_range(box.y)
            and _in_unit_range(box.x2)
            and _in_unit_range(box.y2)
        ):
            raise PoseContractError("pose bbox must lie fully inside the normalized frame")
        if self.confidence is not None and (
            not _finite(self.confidence) or not _in_unit_range(self.confidence)
        ):
            raise PoseContractError(
                f"pose instance confidence must be finite in 0..1, got {self.confidence!r}"
            )
        if not isinstance(self.keypoints, tuple):
            raise PoseContractError("pose keypoints must be an immutable tuple")
        if len(self.keypoints) != COCO_17_KEYPOINT_COUNT:
            raise PoseContractError(
                f"expected {COCO_17_KEYPOINT_COUNT} COCO keypoints, got {len(self.keypoints)}"
            )
        for expected_index, (expected_name, keypoint) in enumerate(
            zip(COCO_17_KEYPOINTS, self.keypoints)
        ):
            if not isinstance(keypoint, PoseKeypoint):
                raise PoseContractError("pose keypoints must be PoseKeypoint values")
            if keypoint.name is not expected_name or keypoint.index != expected_index:
                raise PoseContractError(
                    "pose keypoints must follow the canonical COCO-17 order without "
                    f"duplicates or gaps (position {expected_index} carries "
                    f"{keypoint.name.value})"
                )

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
    """Immutable pose result of exactly one analysed frame.

    Only ``OK`` may carry instances (zero or more). Any degraded status means
    the frame produced no usable pose evidence at all.
    """

    status: PoseStatus
    instances: tuple[PoseInstance, ...] = ()
    reason: Optional[str] = None
    model_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PoseStatus):
            raise PoseContractError(f"unknown pose status: {self.status!r}")
        if not isinstance(self.instances, tuple):
            raise PoseContractError("pose instances must be an immutable tuple")
        for instance in self.instances:
            if not isinstance(instance, PoseInstance):
                raise PoseContractError("pose instances must be PoseInstance values")
        if self.status is not PoseStatus.OK and self.instances:
            raise PoseContractError(
                f"degraded pose result ({self.status.value}) must carry zero instances"
            )

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
        if status is PoseStatus.OK:
            raise PoseContractError("failure() requires a degraded pose status")
        return cls(status=status, instances=(), reason=reason, model_name=model_name)
