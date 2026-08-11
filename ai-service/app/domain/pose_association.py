"""Immutable pose <-> tracked-person association contract (identity only).

Pure domain types for ONE analysed frame. No temporal state, no regions, no
behaviour, no database, no model objects. Association answers only "which pose
instance corresponds to which detector-tracked person", never what the person's
joints are doing.

Invariants enforced here:

* ``PoseAssociationSpec`` gates are explicit caller-supplied values; there are
  NO production defaults and no silent clamping.
* Only :attr:`PoseMatchStatus.ASSOCIATED` may carry a ``person_tracking_id``.
* A degraded frame status never carries an ``ASSOCIATED`` match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .pose import COCO_17_KEYPOINT_COUNT, PoseStatus


class PoseAssociationError(ValueError):
    """Raised when an association domain object would violate its invariants."""


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _unit(value: object) -> bool:
    return _finite(value) and 0.0 <= float(value) <= 1.0


class PoseMatchStatus(str, Enum):
    """Outcome for ONE pose instance in one frame."""

    ASSOCIATED = "associated"
    AMBIGUOUS = "ambiguous"
    UNASSOCIATED = "unassociated"
    UNTRACKED_BLOCKED = "untracked_blocked"
    INSUFFICIENT_KEYPOINTS = "insufficient_keypoints"


class PoseAssociationFrameStatus(str, Enum):
    """Outcome of the whole association attempt for one frame."""

    OK = "ok"
    POSE_UNAVAILABLE = "pose_unavailable"
    INVALID_PERSON_OBSERVATIONS = "invalid_person_observations"


@dataclass(frozen=True, slots=True)
class PoseAssociationSpec:
    """Explicit, validated candidate gates. No invented production defaults."""

    min_bbox_iou: float
    min_pose_bbox_containment: float
    min_available_keypoints: int
    min_keypoint_inside_ratio: float

    def __post_init__(self) -> None:
        for label, value in (
            ("min_bbox_iou", self.min_bbox_iou),
            ("min_pose_bbox_containment", self.min_pose_bbox_containment),
            ("min_keypoint_inside_ratio", self.min_keypoint_inside_ratio),
        ):
            if not _unit(value):
                raise PoseAssociationError(
                    f"{label} must be finite in 0..1, got {value!r}"
                )
        if isinstance(self.min_available_keypoints, bool) or not isinstance(
            self.min_available_keypoints, int
        ):
            raise PoseAssociationError(
                f"min_available_keypoints must be an int, got {self.min_available_keypoints!r}"
            )
        if not (1 <= self.min_available_keypoints <= COCO_17_KEYPOINT_COUNT):
            raise PoseAssociationError(
                "min_available_keypoints must be within "
                f"1..{COCO_17_KEYPOINT_COUNT}, got {self.min_available_keypoints}"
            )


@dataclass(frozen=True, slots=True)
class PosePersonPairFacts:
    """Raw explainable geometry for one (pose, person) pair. No conclusions."""

    pose_index: int
    person_index: int
    person_tracking_id: Optional[str]
    bbox_iou: float
    pose_bbox_containment_in_person: float
    person_bbox_containment_in_pose: float
    pose_center_inside_person: bool
    available_keypoint_count: int
    keypoints_inside_person_count: int
    keypoint_inside_person_ratio: float
    eligible: bool


@dataclass(frozen=True, slots=True)
class PoseMatch:
    """Association outcome for one pose instance."""

    pose_index: int
    status: PoseMatchStatus
    person_tracking_id: Optional[str] = None
    person_index: Optional[int] = None
    reason: Optional[str] = None
    candidates: tuple[PosePersonPairFacts, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, PoseMatchStatus):
            raise PoseAssociationError(f"unknown pose match status: {self.status!r}")
        if self.status is PoseMatchStatus.ASSOCIATED:
            if not self.person_tracking_id:
                raise PoseAssociationError(
                    "an ASSOCIATED match requires a non-empty person_tracking_id"
                )
            if self.person_index is None:
                raise PoseAssociationError(
                    "an ASSOCIATED match requires the matched person_index"
                )
        elif self.person_tracking_id is not None:
            raise PoseAssociationError(
                f"status {self.status.value} must not carry a person_tracking_id"
            )
        if not isinstance(self.candidates, tuple):
            raise PoseAssociationError("candidates must be an immutable tuple")


@dataclass(frozen=True, slots=True)
class PoseAssociationFrameResult:
    """Immutable association result for exactly one analysed frame."""

    status: PoseAssociationFrameStatus
    matches: tuple[PoseMatch, ...] = ()
    source_pose_status: Optional[PoseStatus] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PoseAssociationFrameStatus):
            raise PoseAssociationError(f"unknown frame status: {self.status!r}")
        if not isinstance(self.matches, tuple):
            raise PoseAssociationError("matches must be an immutable tuple")
        for match in self.matches:
            if not isinstance(match, PoseMatch):
                raise PoseAssociationError("matches must be PoseMatch values")
        if self.status is not PoseAssociationFrameStatus.OK and any(
            match.status is PoseMatchStatus.ASSOCIATED for match in self.matches
        ):
            raise PoseAssociationError(
                f"degraded association frame ({self.status.value}) must carry no "
                "ASSOCIATED match"
            )

    @property
    def ok(self) -> bool:
        return self.status is PoseAssociationFrameStatus.OK

    @property
    def associated_matches(self) -> tuple[PoseMatch, ...]:
        return tuple(
            match for match in self.matches if match.status is PoseMatchStatus.ASSOCIATED
        )
