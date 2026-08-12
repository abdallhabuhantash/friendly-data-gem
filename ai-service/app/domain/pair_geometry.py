"""Immutable SAME-FRAME person-pair and wrist-pair geometry facts.

This layer is PURE GEOMETRY for exactly ONE analysed frame. It answers only:

    "where are these two tracked people, and their genuinely available wrists,
    relative to each other on THIS same frame?"

It deliberately does NOT answer "did they exchange paper?". There is no paper /
document detector in the system, no proximity or contact threshold, no
directionality (giver/receiver), no temporal history and no event.

Architectural note for a later phase (NOT implemented here): any future
``document_exchange`` engine MUST require an explicit armed exam-monitoring
state after legitimate paper distribution. Nothing here arms anything.

Invariants enforced by the dataclasses themselves:

* Same-frame provenance is structural: pairs only exist inside a frame-scoped
  result built from ONE tracked-pose frame.
* A pair holds exactly TWO DISTINCT non-blank tracking ids, canonically ordered,
  so ``pair(A, B) == pair(B, A)``. Tracking ids are never rewritten.
* Wrist facts exist only for genuinely available wrists; a missing wrist is
  ``None`` everywhere and never becomes ``(0, 0)``.
* Person-relative coordinates are NEVER clamped, and the centre-line projection
  parameter ``t`` may legitimately fall outside 0..1.
* Degraded input is preserved as a degraded status, never as a valid empty frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .body_features import BodySide, TrackedBodyFeatures
from .geometry import BBox
from .pose import PoseStatus
from .regions import RelativePoint
from .tracked_pose_observations import (
    SOURCE_MODES,
    SourceMode,
    TrackedPoseFrameStatus,
    strict_index,
)
from datetime import datetime


class PairGeometryContractError(ValueError):
    """Raised when a pair-geometry object would violate its invariants."""


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


#: Frame status is reused verbatim from the tracked-pose layer so a degraded
#: source can never be laundered into a different vocabulary.
PairFrameStatus = TrackedPoseFrameStatus


def _validate_frame_metadata(
    camera_id: Optional[str],
    frame_sequence: Optional[int],
    observed_at: Optional[datetime],
    source_mode: Optional[SourceMode],
    source_pose_status: Optional[PoseStatus],
) -> None:
    if camera_id is not None and (
        not isinstance(camera_id, str) or not camera_id.strip()
    ):
        raise PairGeometryContractError("camera_id must be a non-blank string")
    if frame_sequence is not None and not strict_index(frame_sequence):
        raise PairGeometryContractError("frame_sequence must be a non-negative int")
    if observed_at is not None and not isinstance(observed_at, datetime):
        raise PairGeometryContractError("observed_at must be a datetime")
    if source_mode is not None and source_mode not in SOURCE_MODES:
        raise PairGeometryContractError("source_mode must be a valid SourceMode")
    if source_pose_status is not None and not isinstance(source_pose_status, PoseStatus):
        raise PairGeometryContractError("source_pose_status must be a valid PoseStatus")


@dataclass(frozen=True, slots=True)
class TrackedBodyFeatureFrame:
    """Frame-level body-feature container for exactly ONE analysed frame.

    ``subjects`` are derived in one call from a single ``TrackedPoseFrameResult``;
    body features from arbitrary frames are never accepted and assumed to belong
    together.
    """

    status: PairFrameStatus
    subjects: tuple[TrackedBodyFeatures, ...] = ()
    camera_id: Optional[str] = None
    frame_sequence: Optional[int] = None
    observed_at: Optional[datetime] = None
    source_mode: Optional[SourceMode] = None
    source_pose_status: Optional[PoseStatus] = None
    source_frame_status: Optional[TrackedPoseFrameStatus] = None
    unresolved_pose_count: int = 0
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PairFrameStatus):
            raise PairGeometryContractError(f"unknown frame status: {self.status!r}")
        if not isinstance(self.subjects, tuple):
            raise PairGeometryContractError("subjects must be an immutable tuple")
        for subject in self.subjects:
            if not isinstance(subject, TrackedBodyFeatures):
                raise PairGeometryContractError(
                    "subjects must be TrackedBodyFeatures values"
                )
        if self.status is not PairFrameStatus.OK and self.subjects:
            raise PairGeometryContractError(
                f"degraded body-feature frame ({self.status.value}) must carry no subjects"
            )
        seen: set[str] = set()
        for subject in self.subjects:
            if subject.person_tracking_id in seen:
                raise PairGeometryContractError(
                    "duplicate tracking id in body-feature subjects"
                )
            seen.add(subject.person_tracking_id)
        if not strict_index(self.unresolved_pose_count):
            raise PairGeometryContractError(
                "unresolved_pose_count must be a non-negative int"
            )
        if self.source_frame_status is not None and not isinstance(
            self.source_frame_status, TrackedPoseFrameStatus
        ):
            raise PairGeometryContractError(
                "source_frame_status must be a TrackedPoseFrameStatus"
            )
        _validate_frame_metadata(
            self.camera_id,
            self.frame_sequence,
            self.observed_at,
            self.source_mode,
            self.source_pose_status,
        )

    @property
    def subject_count(self) -> int:
        return len(self.subjects)

    def subject(self, tracking_id: str) -> Optional[TrackedBodyFeatures]:
        for subject in self.subjects:
            if subject.person_tracking_id == tracking_id:
                return subject
        return None


@dataclass(frozen=True, slots=True)
class PersonPairKey:
    """Symmetric identity of ONE unordered same-frame person pair.

    The two exact tracking ids are stored in a deterministic canonical order, so
    ``PersonPairKey.of(a, b) == PersonPairKey.of(b, a)``. No id is normalised or
    rewritten, no role (giver/receiver) is implied and the key carries no
    permanent identity beyond this frame.
    """

    first_tracking_id: str
    second_tracking_id: str

    def __post_init__(self) -> None:
        for label, value in (
            ("first_tracking_id", self.first_tracking_id),
            ("second_tracking_id", self.second_tracking_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise PairGeometryContractError(f"{label} must be a non-blank string")
        if self.first_tracking_id == self.second_tracking_id:
            raise PairGeometryContractError(
                "a pair requires two DISTINCT tracking ids"
            )
        if (self.first_tracking_id, self.second_tracking_id) != tuple(
            sorted((self.first_tracking_id, self.second_tracking_id))
        ):
            raise PairGeometryContractError(
                "pair tracking ids must be in canonical order; use PersonPairKey.of()"
            )

    @classmethod
    def of(cls, a: str, b: str) -> "PersonPairKey":
        """Canonical, order-independent construction from two exact tracking ids."""
        for value in (a, b):
            if not isinstance(value, str) or not value.strip():
                raise PairGeometryContractError(
                    "pair tracking ids must be non-blank strings"
                )
        first, second = sorted((a, b))
        return cls(first_tracking_id=first, second_tracking_id=second)

    @property
    def tracking_ids(self) -> tuple[str, str]:
        return (self.first_tracking_id, self.second_tracking_id)


@dataclass(frozen=True, slots=True)
class WristPairGeometry:
    """Raw geometry between ONE available wrist of A and ONE available wrist of B.

    Distances are in normalized frame units only. No pixels, no metres, no
    centimetres, no depth and no contact/touch claim.
    """

    side_a: BodySide
    side_b: BodySide
    a_x: float
    a_y: float
    b_x: float
    b_y: float
    distance: float
    distance_relative_to_person_a_diagonal: Optional[float] = None
    distance_relative_to_person_b_diagonal: Optional[float] = None
    distance_relative_to_mean_person_diagonal: Optional[float] = None

    def __post_init__(self) -> None:
        for label, value in (("side_a", self.side_a), ("side_b", self.side_b)):
            if not isinstance(value, BodySide):
                raise PairGeometryContractError(f"{label} must be a BodySide")
        for label, value in (
            ("a_x", self.a_x),
            ("a_y", self.a_y),
            ("b_x", self.b_x),
            ("b_y", self.b_y),
            ("distance", self.distance),
        ):
            if not _finite(value):
                raise PairGeometryContractError(f"{label} must be a finite number")
        if self.distance < 0.0:
            raise PairGeometryContractError("distance must be non-negative")
        for label, value in (
            (
                "distance_relative_to_person_a_diagonal",
                self.distance_relative_to_person_a_diagonal,
            ),
            (
                "distance_relative_to_person_b_diagonal",
                self.distance_relative_to_person_b_diagonal,
            ),
            (
                "distance_relative_to_mean_person_diagonal",
                self.distance_relative_to_mean_person_diagonal,
            ),
        ):
            if value is not None and (not _finite(value) or float(value) < 0.0):
                raise PairGeometryContractError(
                    f"{label} must be a finite non-negative number"
                )


@dataclass(frozen=True, slots=True)
class WristRelativeToOtherPerson:
    """One available wrist expressed relative to the OTHER person's detector box.

    ``inside_other_person_bbox`` is a 2D projection fact ONLY. It does not mean
    the wrist touches, contacts or reaches that person.
    """

    #: Tracking id of the person the wrist belongs to.
    wrist_owner_tracking_id: str
    #: Tracking id of the person whose bbox is the reference frame.
    other_tracking_id: str
    side: BodySide
    relative_position: RelativePoint
    inside_other_person_bbox: bool

    def __post_init__(self) -> None:
        for label, value in (
            ("wrist_owner_tracking_id", self.wrist_owner_tracking_id),
            ("other_tracking_id", self.other_tracking_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise PairGeometryContractError(f"{label} must be a non-blank string")
        if self.wrist_owner_tracking_id == self.other_tracking_id:
            raise PairGeometryContractError(
                "a wrist cannot be expressed relative to its own owner"
            )
        if not isinstance(self.side, BodySide):
            raise PairGeometryContractError("side must be a BodySide")
        if not isinstance(self.relative_position, RelativePoint):
            raise PairGeometryContractError("relative_position must be a RelativePoint")
        if not (
            _finite(self.relative_position.relative_x)
            and _finite(self.relative_position.relative_y)
        ):
            raise PairGeometryContractError("relative_position must be finite")
        if type(self.inside_other_person_bbox) is not bool:
            raise PairGeometryContractError(
                "inside_other_person_bbox must be a real bool"
            )
        if self.inside_other_person_bbox is not self.relative_position.inside_person:
            raise PairGeometryContractError(
                "inside_other_person_bbox contradicts the relative position"
            )

    @property
    def relative_x(self) -> float:
        return self.relative_position.relative_x

    @property
    def relative_y(self) -> float:
        return self.relative_position.relative_y


@dataclass(frozen=True, slots=True)
class WristAxisProjection:
    """Projection of ONE available wrist onto the A-centre -> B-centre segment.

    ``t = 0`` is person A's bbox centre and ``t = 1`` is person B's bbox centre.
    Values outside 0..1 are legitimate and are NEVER clamped. No interaction
    zone, band width or threshold is defined here: that requires calibration.
    """

    wrist_owner_tracking_id: str
    side: BodySide
    available: bool
    t: Optional[float] = None
    perpendicular_distance: Optional[float] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.wrist_owner_tracking_id, str)
            or not self.wrist_owner_tracking_id.strip()
        ):
            raise PairGeometryContractError(
                "wrist_owner_tracking_id must be a non-blank string"
            )
        if not isinstance(self.side, BodySide):
            raise PairGeometryContractError("side must be a BodySide")
        if type(self.available) is not bool:
            raise PairGeometryContractError("available must be a real bool")
        if self.available:
            if not _finite(self.t):
                raise PairGeometryContractError(
                    "an available projection requires a finite t"
                )
            if not _finite(self.perpendicular_distance) or float(
                self.perpendicular_distance
            ) < 0.0:
                raise PairGeometryContractError(
                    "perpendicular_distance must be finite and non-negative"
                )
        else:
            if self.t is not None or self.perpendicular_distance is not None:
                raise PairGeometryContractError(
                    "an unavailable projection must not carry values"
                )


@dataclass(frozen=True, slots=True)
class PersonPairGeometry:
    """Neutral same-frame geometry of ONE unordered person pair."""

    key: PersonPairKey
    person_a: TrackedBodyFeatures
    person_b: TrackedBodyFeatures
    center_distance: float
    bbox_min_separation: float
    bbox_intersection_area: float
    bbox_iou: float
    center_distance_relative_to_person_a_diagonal: Optional[float] = None
    center_distance_relative_to_person_b_diagonal: Optional[float] = None
    center_distance_relative_to_mean_person_diagonal: Optional[float] = None
    wrist_pairs: tuple[WristPairGeometry, ...] = ()
    nearest_available_wrist_pair: Optional[WristPairGeometry] = None
    wrists_relative_to_other_person: tuple[WristRelativeToOtherPerson, ...] = ()
    wrist_axis_projections: tuple[WristAxisProjection, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, PersonPairKey):
            raise PairGeometryContractError("key must be a PersonPairKey")
        for label, subject in (("person_a", self.person_a), ("person_b", self.person_b)):
            if not isinstance(subject, TrackedBodyFeatures):
                raise PairGeometryContractError(f"{label} must be TrackedBodyFeatures")
        if self.person_a.person_tracking_id == self.person_b.person_tracking_id:
            raise PairGeometryContractError("a person can never be paired with itself")
        if set(self.key.tracking_ids) != {
            self.person_a.person_tracking_id,
            self.person_b.person_tracking_id,
        }:
            raise PairGeometryContractError("pair key contradicts its members")
        for label, value in (
            ("center_distance", self.center_distance),
            ("bbox_min_separation", self.bbox_min_separation),
            ("bbox_intersection_area", self.bbox_intersection_area),
            ("bbox_iou", self.bbox_iou),
        ):
            if not _finite(value) or float(value) < 0.0:
                raise PairGeometryContractError(
                    f"{label} must be a finite non-negative number"
                )
        for label, value in (
            (
                "center_distance_relative_to_person_a_diagonal",
                self.center_distance_relative_to_person_a_diagonal,
            ),
            (
                "center_distance_relative_to_person_b_diagonal",
                self.center_distance_relative_to_person_b_diagonal,
            ),
            (
                "center_distance_relative_to_mean_person_diagonal",
                self.center_distance_relative_to_mean_person_diagonal,
            ),
        ):
            if value is not None and (not _finite(value) or float(value) < 0.0):
                raise PairGeometryContractError(
                    f"{label} must be a finite non-negative number"
                )
        for label, items, kind in (
            ("wrist_pairs", self.wrist_pairs, WristPairGeometry),
            (
                "wrists_relative_to_other_person",
                self.wrists_relative_to_other_person,
                WristRelativeToOtherPerson,
            ),
            ("wrist_axis_projections", self.wrist_axis_projections, WristAxisProjection),
        ):
            if not isinstance(items, tuple):
                raise PairGeometryContractError(f"{label} must be an immutable tuple")
            for item in items:
                if not isinstance(item, kind):
                    raise PairGeometryContractError(
                        f"{label} must contain {kind.__name__} values"
                    )
        if self.nearest_available_wrist_pair is not None:
            if not isinstance(self.nearest_available_wrist_pair, WristPairGeometry):
                raise PairGeometryContractError(
                    "nearest_available_wrist_pair must be a WristPairGeometry"
                )
            if self.nearest_available_wrist_pair not in self.wrist_pairs:
                raise PairGeometryContractError(
                    "nearest_available_wrist_pair must be one of wrist_pairs"
                )
        elif self.wrist_pairs:
            raise PairGeometryContractError(
                "nearest_available_wrist_pair is required when wrist pairs exist"
            )

    @property
    def person_a_bbox(self) -> BBox:
        return self.person_a.person_bbox

    @property
    def person_b_bbox(self) -> BBox:
        return self.person_b.person_bbox

    @property
    def person_a_center(self) -> tuple[float, float]:
        return self.person_a.person_bbox.center

    @property
    def person_b_center(self) -> tuple[float, float]:
        return self.person_b.person_bbox.center

    @property
    def has_available_wrist_pair(self) -> bool:
        return bool(self.wrist_pairs)


@dataclass(frozen=True, slots=True)
class PersonPairFrameResult:
    """All unordered person pairs of exactly ONE analysed frame.

    A valid frame with fewer than two subjects legitimately carries zero pairs;
    that is *no evidence*, never a failure. A degraded source frame is reported
    with its degraded status and carries no pairs at all.
    """

    status: PairFrameStatus
    pairs: tuple[PersonPairGeometry, ...] = ()
    subject_count: int = 0
    camera_id: Optional[str] = None
    frame_sequence: Optional[int] = None
    observed_at: Optional[datetime] = None
    source_mode: Optional[SourceMode] = None
    source_pose_status: Optional[PoseStatus] = None
    source_frame_status: Optional[TrackedPoseFrameStatus] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PairFrameStatus):
            raise PairGeometryContractError(f"unknown frame status: {self.status!r}")
        if not isinstance(self.pairs, tuple):
            raise PairGeometryContractError("pairs must be an immutable tuple")
        for pair in self.pairs:
            if not isinstance(pair, PersonPairGeometry):
                raise PairGeometryContractError(
                    "pairs must be PersonPairGeometry values"
                )
        if self.status is not PairFrameStatus.OK and self.pairs:
            raise PairGeometryContractError(
                f"degraded pair frame ({self.status.value}) must carry no pairs"
            )
        if not strict_index(self.subject_count):
            raise PairGeometryContractError("subject_count must be a non-negative int")
        seen: set[tuple[str, str]] = set()
        for pair in self.pairs:
            if pair.key.tracking_ids in seen:
                raise PairGeometryContractError("duplicate pair in one frame result")
            seen.add(pair.key.tracking_ids)
        expected = self.subject_count * (self.subject_count - 1) // 2
        if self.status is PairFrameStatus.OK and len(self.pairs) != expected:
            raise PairGeometryContractError(
                f"{self.subject_count} subjects require exactly {expected} pairs"
            )
        if self.source_frame_status is not None and not isinstance(
            self.source_frame_status, TrackedPoseFrameStatus
        ):
            raise PairGeometryContractError(
                "source_frame_status must be a TrackedPoseFrameStatus"
            )
        _validate_frame_metadata(
            self.camera_id,
            self.frame_sequence,
            self.observed_at,
            self.source_mode,
            self.source_pose_status,
        )

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    def pair(self, a: str, b: str) -> Optional[PersonPairGeometry]:
        """Order-independent lookup of one pair by its two exact tracking ids."""
        key = PersonPairKey.of(a, b)
        for pair in self.pairs:
            if pair.key == key:
                return pair
        return None
