"""Immutable DERIVED tracked-pose observation contract (one frame only).

This module joins three already-validated source contracts — a
``PoseFrameResult``, the same-frame ``FrameObservations`` and the
``PoseAssociationFrameResult`` — into a truthful, behaviour-free view:

    "for tracked person <id>, on THIS frame, these canonical COCO-17 keypoints
    were available, and these were their normalized frame and person-relative
    positions."

Deliberately absent (and out of scope): lower-person-region membership,
``PersonRegionSpec`` thresholds, wrist/head/posture features, movement,
temporal history, scores, events and any concealed-device conclusion. Nothing
here says what a joint position MEANS.

Invariants enforced by the dataclasses themselves:

* An available ``TrackedPoseKeypoint`` carries finite normalized frame
  coordinates plus a relative position and an ``inside_person`` fact; an
  unavailable one carries ``x=y=None``, ``relative_position=None`` and
  ``inside_person=None``. ``(0, 0)`` is never invented.
* Relative coordinates are NEVER clamped: a valid frame keypoint outside the
  detector person box legitimately yields values outside 0..1 with
  ``inside_person=False``.
* A ``TrackedPoseObservation`` always carries a non-blank tracking id, the
  canonical 17-slot keypoint tuple and positive normalized boxes.
* Confidences stay separate: person detector confidence, pose instance
  confidence and per-keypoint confidence are never combined.
* A degraded/inconsistent ``TrackedPoseFrameResult`` carries zero observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from .geometry import BBox
from .models import SourceMode
from .pose import (
    COCO_17_INDEX_BY_NAME,
    COCO_17_KEYPOINT_COUNT,
    COCO_17_KEYPOINTS,
    PoseKeypointName,
    PoseStatus,
)
from .pose_association import PoseMatch, PoseMatchStatus
from .regions import RelativePoint


class TrackedPoseContractError(ValueError):
    """Raised when a tracked-pose domain object would violate its invariants."""


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _unit(value: object) -> bool:
    return _finite(value) and 0.0 <= float(value) <= 1.0


def strict_index(value: object) -> bool:
    """True only for a REAL non-negative ``int`` (``bool`` is never an index)."""
    return type(value) is int and value >= 0


class TrackedPoseFrameStatus(str, Enum):
    """Outcome of ONE derived tracked-pose frame build."""

    OK = "ok"
    POSE_UNAVAILABLE = "pose_unavailable"
    ASSOCIATION_UNAVAILABLE = "association_unavailable"
    INCONSISTENT_INPUT = "inconsistent_input"


@dataclass(frozen=True, slots=True)
class TrackedPoseKeypoint:
    """One canonical COCO joint of one tracked pose subject."""

    name: PoseKeypointName
    index: int
    available: bool
    confidence: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    relative_position: Optional[RelativePoint] = None
    inside_person: Optional[bool] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, PoseKeypointName):
            raise TrackedPoseContractError(f"unknown pose keypoint name: {self.name!r}")
        canonical = COCO_17_INDEX_BY_NAME[self.name]
        if self.index != canonical:
            raise TrackedPoseContractError(
                f"{self.name.value} must use canonical COCO index {canonical}, "
                f"got {self.index}"
            )
        if self.confidence is not None and not _unit(self.confidence):
            raise TrackedPoseContractError(
                f"{self.name.value} confidence must be finite in 0..1, got {self.confidence!r}"
            )
        if self.available:
            if self.x is None or self.y is None:
                raise TrackedPoseContractError(
                    f"{self.name.value} is available but has no frame coordinates"
                )
            for label, value in (("x", self.x), ("y", self.y)):
                if not _unit(value):
                    raise TrackedPoseContractError(
                        f"{self.name.value} {label} must be finite normalized 0..1, "
                        f"got {value!r}"
                    )
            if self.confidence is None:
                raise TrackedPoseContractError(
                    f"{self.name.value} is available but carries no confidence"
                )
            if self.relative_position is None or self.inside_person is None:
                raise TrackedPoseContractError(
                    f"{self.name.value} is available but carries no resolved "
                    "person-relative position"
                )
            if type(self.inside_person) is not bool:
                raise TrackedPoseContractError(
                    f"{self.name.value} inside_person must be a real bool"
                )
            if not isinstance(self.relative_position, RelativePoint):
                raise TrackedPoseContractError("relative_position must be a RelativePoint")
            if self.inside_person is not self.relative_position.inside_person:
                raise TrackedPoseContractError(
                    f"{self.name.value} inside_person contradicts its relative position"
                )
        else:
            if self.x is not None or self.y is not None:
                raise TrackedPoseContractError(
                    f"{self.name.value} is unavailable and must not carry coordinates"
                )
            if self.relative_position is not None or self.inside_person is not None:
                raise TrackedPoseContractError(
                    f"{self.name.value} is unavailable and must not carry relative geometry"
                )
        if self.relative_position is not None:
            if not isinstance(self.relative_position, RelativePoint):
                raise TrackedPoseContractError("relative_position must be a RelativePoint")
            if not (
                _finite(self.relative_position.relative_x)
                and _finite(self.relative_position.relative_y)
            ):
                raise TrackedPoseContractError("relative_position must be finite")


@dataclass(frozen=True, slots=True)
class TrackedPoseObservation:
    """ONE pose instance safely ASSOCIATED to ONE tracked person, ONE frame."""

    person_tracking_id: str
    person_index: int
    pose_index: int
    person_bbox: BBox
    person_confidence: float
    pose_bbox: BBox
    keypoints: tuple[TrackedPoseKeypoint, ...]
    pose_confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.person_tracking_id, str) or not self.person_tracking_id.strip():
            raise TrackedPoseContractError(
                "a tracked pose observation requires a non-blank person_tracking_id"
            )
        for label, value in (
            ("person_index", self.person_index),
            ("pose_index", self.pose_index),
        ):
            if not strict_index(value):
                raise TrackedPoseContractError(f"{label} must be a non-negative int")
        for label, box in (("person_bbox", self.person_bbox), ("pose_bbox", self.pose_bbox)):
            if not isinstance(box, BBox):
                raise TrackedPoseContractError(f"{label} must be a BBox")
            if not all(_finite(v) for v in (box.x, box.y, box.width, box.height)):
                raise TrackedPoseContractError(f"{label} must be finite")
            if box.width <= 0.0 or box.height <= 0.0:
                raise TrackedPoseContractError(f"{label} must have positive extent")
            if not (_unit(box.x) and _unit(box.y) and _unit(box.x2) and _unit(box.y2)):
                raise TrackedPoseContractError(f"{label} must lie inside the normalized frame")
        if not _unit(self.person_confidence):
            raise TrackedPoseContractError("person_confidence must be finite in 0..1")
        if self.pose_confidence is not None and not _unit(self.pose_confidence):
            raise TrackedPoseContractError("pose_confidence must be finite in 0..1")
        if not isinstance(self.keypoints, tuple):
            raise TrackedPoseContractError("keypoints must be an immutable tuple")
        if len(self.keypoints) != COCO_17_KEYPOINT_COUNT:
            raise TrackedPoseContractError(
                f"expected {COCO_17_KEYPOINT_COUNT} tracked keypoints, got {len(self.keypoints)}"
            )
        for expected_index, (expected_name, keypoint) in enumerate(
            zip(COCO_17_KEYPOINTS, self.keypoints)
        ):
            if not isinstance(keypoint, TrackedPoseKeypoint):
                raise TrackedPoseContractError("keypoints must be TrackedPoseKeypoint values")
            if keypoint.name is not expected_name or keypoint.index != expected_index:
                raise TrackedPoseContractError(
                    "tracked keypoints must follow canonical COCO-17 order without gaps"
                )

    @property
    def available_keypoint_count(self) -> int:
        return sum(1 for keypoint in self.keypoints if keypoint.available)

    def keypoint(self, name: PoseKeypointName) -> Optional[TrackedPoseKeypoint]:
        """Safe semantic lookup; never raises on unknown/unavailable joints."""
        if not isinstance(name, PoseKeypointName):
            return None
        for keypoint in self.keypoints:
            if keypoint.name is name:
                return keypoint
        return None

    def available_keypoint(self, name: PoseKeypointName) -> Optional[TrackedPoseKeypoint]:
        """Returns the keypoint only when it was genuinely observed."""
        keypoint = self.keypoint(name)
        if keypoint is None or not keypoint.available:
            return None
        return keypoint


@dataclass(frozen=True, slots=True)
class UnresolvedPoseDiagnostic:
    """A pose instance that existed but could NOT be assigned an identity."""

    pose_index: int
    match_status: PoseMatchStatus
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not strict_index(self.pose_index):
            raise TrackedPoseContractError("pose_index must be a non-negative int")
        if not isinstance(self.match_status, PoseMatchStatus):
            raise TrackedPoseContractError(f"unknown match status: {self.match_status!r}")
        if self.match_status is PoseMatchStatus.ASSOCIATED:
            raise TrackedPoseContractError(
                "an ASSOCIATED match is resolved and is not an unresolved diagnostic"
            )

    @classmethod
    def from_match(cls, match: PoseMatch) -> "UnresolvedPoseDiagnostic":
        return cls(
            pose_index=match.pose_index,
            match_status=match.status,
            reason=match.reason,
        )


@dataclass(frozen=True, slots=True)
class TrackedPoseFrameResult:
    """Immutable derived tracked-pose result for exactly ONE analysed frame."""

    status: TrackedPoseFrameStatus
    observations: tuple[TrackedPoseObservation, ...] = ()
    unresolved: tuple[UnresolvedPoseDiagnostic, ...] = ()
    camera_id: Optional[str] = None
    frame_sequence: Optional[int] = None
    observed_at: Optional[datetime] = None
    source_mode: Optional[SourceMode] = None
    source_pose_status: Optional[PoseStatus] = None
    pose_instance_count: int = 0
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, TrackedPoseFrameStatus):
            raise TrackedPoseContractError(f"unknown frame status: {self.status!r}")
        if not isinstance(self.observations, tuple):
            raise TrackedPoseContractError("observations must be an immutable tuple")
        if not isinstance(self.unresolved, tuple):
            raise TrackedPoseContractError("unresolved must be an immutable tuple")
        for observation in self.observations:
            if not isinstance(observation, TrackedPoseObservation):
                raise TrackedPoseContractError(
                    "observations must be TrackedPoseObservation values"
                )
        for diagnostic in self.unresolved:
            if not isinstance(diagnostic, UnresolvedPoseDiagnostic):
                raise TrackedPoseContractError(
                    "unresolved must be UnresolvedPoseDiagnostic values"
                )
        if self.status is not TrackedPoseFrameStatus.OK and (
            self.observations or self.unresolved
        ):
            raise TrackedPoseContractError(
                f"degraded tracked-pose frame ({self.status.value}) must carry no "
                "observations or unresolved diagnostics"
            )
        if not strict_index(self.pose_instance_count):
            raise TrackedPoseContractError("pose_instance_count must be a non-negative int")
        if self.camera_id is not None and (
            not isinstance(self.camera_id, str) or not self.camera_id.strip()
        ):
            raise TrackedPoseContractError("camera_id must be a non-blank string")
        if self.frame_sequence is not None and not strict_index(self.frame_sequence):
            raise TrackedPoseContractError("frame_sequence must be a non-negative int")
        if self.observed_at is not None and not isinstance(self.observed_at, datetime):
            raise TrackedPoseContractError("observed_at must be a datetime")
        if self.source_mode is not None and not isinstance(self.source_mode, SourceMode):
            raise TrackedPoseContractError("source_mode must be a valid SourceMode")
        if self.source_pose_status is not None and not isinstance(
            self.source_pose_status, PoseStatus
        ):
            raise TrackedPoseContractError("source_pose_status must be a valid PoseStatus")
        seen_persons: set[int] = set()
        seen_tracks: set[str] = set()
        seen_poses: set[int] = set()
        for observation in self.observations:
            if observation.person_index in seen_persons:
                raise TrackedPoseContractError("duplicate person_index in tracked observations")
            if observation.person_tracking_id in seen_tracks:
                raise TrackedPoseContractError("duplicate tracking id in tracked observations")
            if observation.pose_index in seen_poses:
                raise TrackedPoseContractError("duplicate pose_index in tracked observations")
            seen_persons.add(observation.person_index)
            seen_tracks.add(observation.person_tracking_id)
            seen_poses.add(observation.pose_index)

        if self.status is TrackedPoseFrameStatus.OK:
            unresolved_poses: set[int] = set()
            for diagnostic in self.unresolved:
                if diagnostic.pose_index in unresolved_poses:
                    raise TrackedPoseContractError(
                        "duplicate pose_index in unresolved diagnostics"
                    )
                if diagnostic.pose_index in seen_poses:
                    raise TrackedPoseContractError(
                        "a pose cannot be both resolved and unresolved"
                    )
                unresolved_poses.add(diagnostic.pose_index)
            if seen_poses | unresolved_poses != set(range(self.pose_instance_count)):
                raise TrackedPoseContractError(
                    "an ok tracked-pose frame must account for every source pose exactly once"
                )

    @property
    def ok(self) -> bool:
        return self.status is TrackedPoseFrameStatus.OK

    @property
    def tracked_count(self) -> int:
        return len(self.observations)

    def observation_for(self, person_tracking_id: str) -> Optional[TrackedPoseObservation]:
        """Safe lookup by temporary tracking identity."""
        for observation in self.observations:
            if observation.person_tracking_id == person_tracking_id:
                return observation
        return None
