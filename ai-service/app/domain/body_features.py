"""Immutable DERIVED wrist/arm geometry facts for ONE tracked person, ONE frame.

This layer is PURE GEOMETRY. It answers only:

    "for this tracked person, on THIS frame, which of the shoulder/elbow/wrist
    joints were actually available per side, where were the wrists, and what
    were the plain Euclidean relationships between the available joints."

Deliberately absent (and out of scope here): behaviour, intent, exchange or
concealment conclusions, pair/inter-person geometry, temporal history, scores,
thresholds and events. Nothing in this module says what a measurement MEANS.

Architectural note for a later phase (NOT implemented here): any future
``document_exchange`` event MUST require an explicit armed exam-monitoring
state. Starting a camera alone must never arm document-exchange monitoring, and
paper distribution before that state is armed must not raise any
document-exchange alert.

Invariants enforced by the dataclasses themselves:

* An available wrist carries finite normalized frame coordinates, a
  person-relative position and an ``inside_person`` fact; an unavailable one
  carries ``None`` everywhere. ``(0, 0)`` is never invented.
* Person-relative coordinates are NEVER clamped: an available wrist outside the
  detector person box legitimately yields values outside 0..1 with
  ``inside_person=False``.
* Every distance/ratio/angle is ``None`` unless ALL of its source keypoints were
  genuinely available. No joint is ever inferred.
* Left and right sides are separate values; neither borrows from the other.
* Confidences stay separate: detector person confidence, pose instance
  confidence and per-keypoint confidence are never combined into one score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .geometry import BBox
from .pose import PoseKeypointName
from .regions import RelativePoint


class BodyFeatureContractError(ValueError):
    """Raised when a derived body-feature object would violate its invariants."""


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _unit(value: object) -> bool:
    return _finite(value) and 0.0 <= float(value) <= 1.0


class BodySide(str, Enum):
    """Anatomical side, as reported by the pose model itself."""

    LEFT = "left"
    RIGHT = "right"

    @property
    def shoulder(self) -> PoseKeypointName:
        return (
            PoseKeypointName.LEFT_SHOULDER
            if self is BodySide.LEFT
            else PoseKeypointName.RIGHT_SHOULDER
        )

    @property
    def elbow(self) -> PoseKeypointName:
        return (
            PoseKeypointName.LEFT_ELBOW
            if self is BodySide.LEFT
            else PoseKeypointName.RIGHT_ELBOW
        )

    @property
    def wrist(self) -> PoseKeypointName:
        return (
            PoseKeypointName.LEFT_WRIST
            if self is BodySide.LEFT
            else PoseKeypointName.RIGHT_WRIST
        )


@dataclass(frozen=True, slots=True)
class SideAvailability:
    """Which of the three arm joints of ONE side were genuinely available."""

    shoulder_available: bool
    elbow_available: bool
    wrist_available: bool

    def __post_init__(self) -> None:
        for label, value in (
            ("shoulder_available", self.shoulder_available),
            ("elbow_available", self.elbow_available),
            ("wrist_available", self.wrist_available),
        ):
            if type(value) is not bool:
                raise BodyFeatureContractError(f"{label} must be a real bool")

    @property
    def available_joint_count(self) -> int:
        return sum(
            (self.shoulder_available, self.elbow_available, self.wrist_available)
        )

    @property
    def full_chain_available(self) -> bool:
        return self.shoulder_available and self.elbow_available and self.wrist_available


@dataclass(frozen=True, slots=True)
class WristFeatures:
    """Truthful facts about ONE wrist keypoint (never a hand, palm or finger).

    COCO-17 provides a wrist only; no hand centre, grasp or contact state is
    derivable and none is represented here.
    """

    side: BodySide
    available: bool
    x: Optional[float] = None
    y: Optional[float] = None
    relative_position: Optional[RelativePoint] = None
    inside_person: Optional[bool] = None
    #: Per-keypoint source confidence, kept raw and never fused with others.
    keypoint_confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.side, BodySide):
            raise BodyFeatureContractError(f"unknown body side: {self.side!r}")
        if type(self.available) is not bool:
            raise BodyFeatureContractError("available must be a real bool")
        if self.keypoint_confidence is not None and not _unit(self.keypoint_confidence):
            raise BodyFeatureContractError(
                "keypoint_confidence must be finite in 0..1"
            )
        if self.available:
            if self.x is None or self.y is None:
                raise BodyFeatureContractError(
                    "an available wrist requires frame coordinates"
                )
            for label, value in (("x", self.x), ("y", self.y)):
                if not _unit(value):
                    raise BodyFeatureContractError(
                        f"wrist {label} must be finite normalized 0..1, got {value!r}"
                    )
            if self.relative_position is None or self.inside_person is None:
                raise BodyFeatureContractError(
                    "an available wrist requires a resolved person-relative position"
                )
            if not isinstance(self.relative_position, RelativePoint):
                raise BodyFeatureContractError(
                    "relative_position must be a RelativePoint"
                )
            if type(self.inside_person) is not bool:
                raise BodyFeatureContractError("inside_person must be a real bool")
            if self.inside_person is not self.relative_position.inside_person:
                raise BodyFeatureContractError(
                    "inside_person contradicts the person-relative position"
                )
            if not (
                _finite(self.relative_position.relative_x)
                and _finite(self.relative_position.relative_y)
            ):
                raise BodyFeatureContractError("relative_position must be finite")
        else:
            if self.x is not None or self.y is not None:
                raise BodyFeatureContractError(
                    "an unavailable wrist must not carry coordinates"
                )
            if self.relative_position is not None or self.inside_person is not None:
                raise BodyFeatureContractError(
                    "an unavailable wrist must not carry relative geometry"
                )


@dataclass(frozen=True, slots=True)
class ArmFeatures:
    """Plain Euclidean geometry of ONE arm, derived only from available joints.

    All measurements are ``None`` unless every joint they depend on was
    genuinely available. Distances are in normalized frame units; the
    ``*_relative_to_person`` values divide by the person box diagonal so they are
    comparable across person sizes.
    """

    side: BodySide
    availability: SideAvailability
    wrist: WristFeatures
    wrist_to_elbow_distance: Optional[float] = None
    elbow_to_shoulder_distance: Optional[float] = None
    shoulder_to_wrist_distance: Optional[float] = None
    wrist_to_elbow_distance_relative_to_person: Optional[float] = None
    elbow_to_shoulder_distance_relative_to_person: Optional[float] = None
    shoulder_to_wrist_distance_relative_to_person: Optional[float] = None
    #: shoulder->wrist straight distance divided by (shoulder->elbow + elbow->wrist).
    #: 1.0 == perfectly collinear chain, smaller == more folded. ``None`` when any
    #: joint is missing or the summed segment length is zero.
    shoulder_wrist_to_segment_sum_ratio: Optional[float] = None
    #: Interior angle at the elbow, degrees in 0..180, between elbow->shoulder and
    #: elbow->wrist. ``None`` when any joint is missing or a segment has zero length.
    elbow_angle_degrees: Optional[float] = None
    #: Raw source confidences, kept separate on purpose.
    shoulder_confidence: Optional[float] = None
    elbow_confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.side, BodySide):
            raise BodyFeatureContractError(f"unknown body side: {self.side!r}")
        if not isinstance(self.availability, SideAvailability):
            raise BodyFeatureContractError("availability must be a SideAvailability")
        if not isinstance(self.wrist, WristFeatures):
            raise BodyFeatureContractError("wrist must be WristFeatures")
        if self.wrist.side is not self.side:
            raise BodyFeatureContractError("wrist side contradicts the arm side")
        if self.wrist.available is not self.availability.wrist_available:
            raise BodyFeatureContractError(
                "wrist availability contradicts the side availability facts"
            )
        for label, value in (
            ("wrist_to_elbow_distance", self.wrist_to_elbow_distance),
            ("elbow_to_shoulder_distance", self.elbow_to_shoulder_distance),
            ("shoulder_to_wrist_distance", self.shoulder_to_wrist_distance),
            (
                "wrist_to_elbow_distance_relative_to_person",
                self.wrist_to_elbow_distance_relative_to_person,
            ),
            (
                "elbow_to_shoulder_distance_relative_to_person",
                self.elbow_to_shoulder_distance_relative_to_person,
            ),
            (
                "shoulder_to_wrist_distance_relative_to_person",
                self.shoulder_to_wrist_distance_relative_to_person,
            ),
            (
                "shoulder_wrist_to_segment_sum_ratio",
                self.shoulder_wrist_to_segment_sum_ratio,
            ),
        ):
            if value is not None and (not _finite(value) or float(value) < 0.0):
                raise BodyFeatureContractError(
                    f"{label} must be a finite non-negative number, got {value!r}"
                )
        if self.elbow_angle_degrees is not None and not (
            _finite(self.elbow_angle_degrees) and 0.0 <= self.elbow_angle_degrees <= 180.0
        ):
            raise BodyFeatureContractError(
                "elbow_angle_degrees must be finite within 0..180"
            )
        for label, value in (
            ("shoulder_confidence", self.shoulder_confidence),
            ("elbow_confidence", self.elbow_confidence),
        ):
            if value is not None and not _unit(value):
                raise BodyFeatureContractError(f"{label} must be finite in 0..1")


@dataclass(frozen=True, slots=True)
class TrackedBodyFeatures:
    """Derived wrist/arm geometry of ONE tracked person on ONE frame."""

    person_tracking_id: str
    person_index: int
    person_bbox: BBox
    person_confidence: float
    left_arm: ArmFeatures
    right_arm: ArmFeatures
    #: Pose instance confidence, kept raw and never fused with the others.
    pose_confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.person_tracking_id, str)
            or not self.person_tracking_id.strip()
        ):
            raise BodyFeatureContractError(
                "body features require a non-blank person_tracking_id"
            )
        if type(self.person_index) is not int or self.person_index < 0:
            raise BodyFeatureContractError("person_index must be a non-negative int")
        if not isinstance(self.person_bbox, BBox):
            raise BodyFeatureContractError("person_bbox must be a BBox")
        if self.person_bbox.width <= 0.0 or self.person_bbox.height <= 0.0:
            raise BodyFeatureContractError("person_bbox must have positive extent")
        if not _unit(self.person_confidence):
            raise BodyFeatureContractError("person_confidence must be finite in 0..1")
        if self.pose_confidence is not None and not _unit(self.pose_confidence):
            raise BodyFeatureContractError("pose_confidence must be finite in 0..1")
        for label, arm, side in (
            ("left_arm", self.left_arm, BodySide.LEFT),
            ("right_arm", self.right_arm, BodySide.RIGHT),
        ):
            if not isinstance(arm, ArmFeatures):
                raise BodyFeatureContractError(f"{label} must be ArmFeatures")
            if arm.side is not side:
                raise BodyFeatureContractError(f"{label} must carry the {side.value} side")

    def arm(self, side: BodySide) -> Optional[ArmFeatures]:
        """Safe lookup; never raises on an unknown side value."""
        if side is BodySide.LEFT:
            return self.left_arm
        if side is BodySide.RIGHT:
            return self.right_arm
        return None
