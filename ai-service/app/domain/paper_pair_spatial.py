"""Immutable SAME-FRAME paper <-> person-pair spatial geometry facts.

Scope
-----
This layer answers exactly ONE kind of question, for exactly ONE analysed frame:

    "given the person-pair geometry of this frame (Task 3C) and the paper
    evidence of the SAME frame (Task 3E/3E-B), where does each detected paper
    lie, purely as 2D image-plane geometry, relative to each tracked person and
    each genuinely available wrist?"

Everything here is raw derived geometry. There is deliberately NO:

* ownership: no ``paper_owner``, ``person_holding_paper``, ``holder``,
  ``giver``, ``receiver`` or ``transferred_to``;
* contact claim: COCO pose exposes a WRIST keypoint, never a palm, finger or
  grasp state, so "paper near wrist" only ever means "paper centre near the
  wrist point in 2D image space";
* depth claim: overlapping 2D projections may sit at very different depths, so
  no centimetres, metres, physical contact or true 3D proximity is expressed;
* threshold decision: no ``paper_between_people``, ``paper_in_interaction_zone``,
  ``transfer_zone`` or ``handoff_zone`` boolean, because each would require
  calibration this layer does not have;
* temporal state: no previous location, velocity, direction, dwell, history or
  state machine, and nothing from the Task 3D temporal handoff layer is
  imported or consumed;
* fused confidence: the paper detector's confidence is preserved verbatim and is
  never combined with person, pose, wrist-keypoint or temporal confidence.

Object identity
---------------
Paper detections are frame-local facts. There is no paper tracking id and a
``paper_detection_index`` is a position inside ONE frame's detection tuple only;
it is never stable across frames and never a person identity. Overlapping paper
detections are never merged.

Truthfulness of statuses
------------------------
* ``OK`` with zero facts on a valid frame means "no paper evidence was detected
  on this valid frame" — never "no paper existed".
* A degraded paper frame or a degraded pair frame yields a degraded spatial
  status with ZERO trusted spatial facts; detector failure is never laundered
  into valid zero-paper evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from .body_features import BodySide
from .geometry import BBox
from .paper_evidence import CANONICAL_PAPER_CLASS, PaperEvidenceStatus
from .pair_geometry import PairFrameStatus, PersonPairKey
from .regions import RelativePoint
from .tracked_pose_observations import strict_index


class PaperPairSpatialContractError(ValueError):
    """Raised when a paper/pair spatial object would violate its invariants."""


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _non_blank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PaperPairSpatialContractError(f"{label} must be a non-blank string")
    return value


def _non_negative(value: object, label: str) -> float:
    if not _finite(value) or float(value) < 0.0:
        raise PaperPairSpatialContractError(
            f"{label} must be a finite non-negative number"
        )
    return float(value)


def _optional_non_negative(value: object, label: str) -> Optional[float]:
    if value is None:
        return None
    return _non_negative(value, label)


class PaperPairSpatialStatus(str, Enum):
    """Outcome of ONE same-frame paper <-> person-pair spatial derivation."""

    OK = "ok"
    #: The paper-evidence frame itself was not ``OK``; no paper evidence exists
    #: to place geometrically. This is NOT "zero paper detected".
    PAPER_EVIDENCE_DEGRADED = "paper_evidence_degraded"
    #: The Task 3C person-pair frame was degraded; no person geometry is trusted.
    PAIR_GEOMETRY_DEGRADED = "pair_geometry_degraded"
    #: The two inputs could not be proven to describe the SAME source frame.
    INCONSISTENT_INPUT = "inconsistent_input"


@dataclass(frozen=True, slots=True)
class SameFrameJoin:
    """EXPLICIT same-frame provenance declared by the caller.

    The Task 3C pair frame and the Task 3E paper frame are produced by two
    INDEPENDENT pipelines with INDEPENDENT counters and INDEPENDENT clocks:

    * the pair frame carries ``frame_sequence`` (a runtime frame counter) and an
      optional ABSOLUTE ``observed_at`` datetime;
    * the paper frame carries ``frame_index`` (e.g. an evaluator/decode index)
      and an optional MEDIA-RELATIVE ``timestamp_seconds``.

    Those numbers are NEVER required to be equal, and equal numbers NEVER imply
    correspondence. This object IS the caller's explicit declaration that a
    specific ``pair_frame_sequence`` and a specific ``paper_frame_index`` are
    the same source image. Each side is validated ONLY against its own declared
    field, and no offset or conversion between the two clocks is invented.
    """

    camera_id: str
    pair_frame_sequence: int
    paper_frame_index: int
    #: Expected ABSOLUTE datetime of the pair-side observation (pair side only).
    pair_observed_at: Optional[datetime] = None
    #: Expected MEDIA-RELATIVE seconds of the paper-side frame (paper side only).
    paper_timestamp_seconds: Optional[float] = None
    #: Allowed absolute disagreement for the PAPER-side relative timestamp.
    paper_timestamp_tolerance_seconds: float = 0.0

    def __post_init__(self) -> None:
        _non_blank(self.camera_id, "camera_id")
        if not strict_index(self.pair_frame_sequence):
            raise PaperPairSpatialContractError(
                "pair_frame_sequence must be a non-negative int"
            )
        if not strict_index(self.paper_frame_index):
            raise PaperPairSpatialContractError(
                "paper_frame_index must be a non-negative int"
            )
        if self.paper_timestamp_seconds is not None and not _finite(
            self.paper_timestamp_seconds
        ):
            raise PaperPairSpatialContractError(
                "paper_timestamp_seconds must be a finite number when supplied"
            )
        if self.pair_observed_at is not None and not isinstance(
            self.pair_observed_at, datetime
        ):
            raise PaperPairSpatialContractError("pair_observed_at must be a datetime")
        _non_negative(
            self.paper_timestamp_tolerance_seconds,
            "paper_timestamp_tolerance_seconds",
        )



@dataclass(frozen=True, slots=True)
class PaperGeometryFacts:
    """Frame-local geometry of ONE paper detection.

    No identity across frames: there is no paper tracking id, and
    ``detection_index`` is only a position inside this frame's detection tuple.
    ``confidence`` is the paper detector's own confidence, never fused.
    """

    detection_index: int
    class_name: str
    confidence: float
    bbox: BBox
    center_x: float
    center_y: float
    width: float
    height: float
    diagonal: float
    raw_prompt: Optional[str] = None

    def __post_init__(self) -> None:
        if not strict_index(self.detection_index):
            raise PaperPairSpatialContractError(
                "detection_index must be a non-negative int"
            )
        if self.class_name != CANONICAL_PAPER_CLASS:
            raise PaperPairSpatialContractError(
                f"class_name must be the canonical {CANONICAL_PAPER_CLASS!r} class"
            )
        if not _finite(self.confidence) or not 0.0 <= float(self.confidence) <= 1.0:
            raise PaperPairSpatialContractError("confidence must be finite within 0..1")
        if not isinstance(self.bbox, BBox):
            raise PaperPairSpatialContractError("bbox must be a BBox")
        for label, value in (("center_x", self.center_x), ("center_y", self.center_y)):
            if not _finite(value):
                raise PaperPairSpatialContractError(f"{label} must be finite")
        for label, value in (
            ("width", self.width),
            ("height", self.height),
            ("diagonal", self.diagonal),
        ):
            if not _finite(value) or float(value) <= 0.0:
                raise PaperPairSpatialContractError(f"{label} must be finite positive")
        if self.raw_prompt is not None:
            _non_blank(self.raw_prompt, "raw_prompt")


@dataclass(frozen=True, slots=True)
class PaperPersonSpatialFact:
    """Where ONE paper detection lies relative to ONE tracked person's bbox.

    ``center_inside_person_bbox`` and ``bbox_iou`` are 2D projection facts only.
    Overlap does NOT mean the person holds, touches or owns the paper.
    Relative coordinates are NEVER clamped.
    """

    paper_detection_index: int
    person_tracking_id: str
    relative_position: RelativePoint
    center_inside_person_bbox: bool
    bbox_intersection_area: float
    bbox_iou: Optional[float]
    center_distance_to_person_center: float
    center_distance_relative_to_person_diagonal: Optional[float] = None

    def __post_init__(self) -> None:
        if not strict_index(self.paper_detection_index):
            raise PaperPairSpatialContractError(
                "paper_detection_index must be a non-negative int"
            )
        _non_blank(self.person_tracking_id, "person_tracking_id")
        if not isinstance(self.relative_position, RelativePoint):
            raise PaperPairSpatialContractError(
                "relative_position must be a RelativePoint"
            )
        if not (
            _finite(self.relative_position.relative_x)
            and _finite(self.relative_position.relative_y)
        ):
            raise PaperPairSpatialContractError("relative_position must be finite")
        if type(self.center_inside_person_bbox) is not bool:
            raise PaperPairSpatialContractError(
                "center_inside_person_bbox must be a real bool"
            )
        if self.center_inside_person_bbox is not self.relative_position.inside_person:
            raise PaperPairSpatialContractError(
                "center_inside_person_bbox contradicts the relative position"
            )
        _non_negative(self.bbox_intersection_area, "bbox_intersection_area")
        if self.bbox_iou is not None:
            iou_value = _non_negative(self.bbox_iou, "bbox_iou")
            if iou_value > 1.0:
                raise PaperPairSpatialContractError("bbox_iou must be within 0..1")
        _non_negative(
            self.center_distance_to_person_center, "center_distance_to_person_center"
        )
        _optional_non_negative(
            self.center_distance_relative_to_person_diagonal,
            "center_distance_relative_to_person_diagonal",
        )

    @property
    def relative_x(self) -> float:
        return self.relative_position.relative_x

    @property
    def relative_y(self) -> float:
        return self.relative_position.relative_y


@dataclass(frozen=True, slots=True)
class PaperWristSpatialFact:
    """2D distance between ONE paper centre and ONE genuinely available wrist.

    A wrist that was not genuinely available never produces a fact here, and is
    never substituted by ``(0, 0)`` or any other placeholder. Distance is in
    normalized frame units: no pixels, centimetres, metres or depth. Nearness
    NEVER means holding, grasping or contact.
    """

    paper_detection_index: int
    wrist_owner_tracking_id: str
    side: BodySide
    wrist_x: float
    wrist_y: float
    distance: float
    distance_relative_to_owner_person_diagonal: Optional[float] = None
    distance_relative_to_mean_pair_diagonal: Optional[float] = None

    def __post_init__(self) -> None:
        if not strict_index(self.paper_detection_index):
            raise PaperPairSpatialContractError(
                "paper_detection_index must be a non-negative int"
            )
        _non_blank(self.wrist_owner_tracking_id, "wrist_owner_tracking_id")
        if not isinstance(self.side, BodySide):
            raise PaperPairSpatialContractError("side must be a BodySide")
        for label, value in (("wrist_x", self.wrist_x), ("wrist_y", self.wrist_y)):
            if not _finite(value):
                raise PaperPairSpatialContractError(f"{label} must be finite")
        _non_negative(self.distance, "distance")
        _optional_non_negative(
            self.distance_relative_to_owner_person_diagonal,
            "distance_relative_to_owner_person_diagonal",
        )
        _optional_non_negative(
            self.distance_relative_to_mean_pair_diagonal,
            "distance_relative_to_mean_pair_diagonal",
        )


@dataclass(frozen=True, slots=True)
class PaperPairAxisProjection:
    """Projection of ONE paper centre onto the A-centre -> B-centre segment.

    ``t = 0`` is person A's bbox centre, ``t = 1`` is person B's bbox centre.
    Values outside 0..1 are legitimate geometric facts and are NEVER clamped.
    Coincident person centres make the projection unavailable. No band width,
    zone or threshold is defined: that would require calibration.
    """

    paper_detection_index: int
    available: bool
    t: Optional[float] = None
    perpendicular_distance: Optional[float] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not strict_index(self.paper_detection_index):
            raise PaperPairSpatialContractError(
                "paper_detection_index must be a non-negative int"
            )
        if type(self.available) is not bool:
            raise PaperPairSpatialContractError("available must be a real bool")
        if self.available:
            if not _finite(self.t):
                raise PaperPairSpatialContractError(
                    "an available projection requires a finite t"
                )
            _non_negative(self.perpendicular_distance, "perpendicular_distance")
        elif self.t is not None or self.perpendicular_distance is not None:
            raise PaperPairSpatialContractError(
                "an unavailable projection must not carry values"
            )


@dataclass(frozen=True, slots=True)
class PaperPairSpatialFact:
    """All geometry of ONE paper detection relative to ONE unordered person pair.

    ``nearest_available_wrist`` means EXACTLY ONE thing: the mathematically
    nearest genuinely available wrist to the paper centre in 2D image geometry,
    with deterministic tie-breaking. It does NOT mean holder, owner, giver,
    receiver, touching or grasping.
    """

    pair_key: PersonPairKey
    paper: PaperGeometryFacts
    person_facts: tuple[PaperPersonSpatialFact, ...]
    wrist_facts: tuple[PaperWristSpatialFact, ...] = ()
    nearest_available_wrist: Optional[PaperWristSpatialFact] = None
    axis_projection: Optional[PaperPairAxisProjection] = None

    def __post_init__(self) -> None:
        if not isinstance(self.pair_key, PersonPairKey):
            raise PaperPairSpatialContractError("pair_key must be a PersonPairKey")
        if not isinstance(self.paper, PaperGeometryFacts):
            raise PaperPairSpatialContractError("paper must be PaperGeometryFacts")
        for label, items, kind in (
            ("person_facts", self.person_facts, PaperPersonSpatialFact),
            ("wrist_facts", self.wrist_facts, PaperWristSpatialFact),
        ):
            if not isinstance(items, tuple):
                raise PaperPairSpatialContractError(
                    f"{label} must be an immutable tuple"
                )
            for item in items:
                if not isinstance(item, kind):
                    raise PaperPairSpatialContractError(
                        f"{label} must contain {kind.__name__} values"
                    )
                if item.paper_detection_index != self.paper.detection_index:
                    raise PaperPairSpatialContractError(
                        f"{label} must reference this paper detection index"
                    )
        members = set(self.pair_key.tracking_ids)
        if {fact.person_tracking_id for fact in self.person_facts} - members:
            raise PaperPairSpatialContractError(
                "person_facts must only describe members of this pair"
            )
        if len(self.person_facts) != len(
            {fact.person_tracking_id for fact in self.person_facts}
        ):
            raise PaperPairSpatialContractError("duplicate person fact in a pair")
        if {fact.wrist_owner_tracking_id for fact in self.wrist_facts} - members:
            raise PaperPairSpatialContractError(
                "wrist_facts must only describe members of this pair"
            )
        if self.nearest_available_wrist is not None:
            if not isinstance(self.nearest_available_wrist, PaperWristSpatialFact):
                raise PaperPairSpatialContractError(
                    "nearest_available_wrist must be a PaperWristSpatialFact"
                )
            if self.nearest_available_wrist not in self.wrist_facts:
                raise PaperPairSpatialContractError(
                    "nearest_available_wrist must be one of wrist_facts"
                )
        elif self.wrist_facts:
            raise PaperPairSpatialContractError(
                "nearest_available_wrist is required when wrist facts exist"
            )
        if self.axis_projection is not None:
            if not isinstance(self.axis_projection, PaperPairAxisProjection):
                raise PaperPairSpatialContractError(
                    "axis_projection must be a PaperPairAxisProjection"
                )
            if self.axis_projection.paper_detection_index != self.paper.detection_index:
                raise PaperPairSpatialContractError(
                    "axis_projection must reference this paper detection index"
                )


@dataclass(frozen=True, slots=True)
class PaperPairSpatialFrame:
    """Every paper x person-pair spatial fact of exactly ONE analysed frame.

    ``OK`` with zero facts is legitimate: it means either that the frame carried
    no paper evidence, or that fewer than two tracked people were available to
    form a pair. It is never a claim that no paper existed.

    Any degraded status carries ZERO facts.
    """

    status: PaperPairSpatialStatus
    facts: tuple[PaperPairSpatialFact, ...] = ()
    paper_detection_count: int = 0
    pair_count: int = 0
    camera_id: Optional[str] = None
    #: Pair-pipeline frame counter, as explicitly declared in the join.
    pair_frame_sequence: Optional[int] = None
    #: Paper-pipeline frame index, as explicitly declared in the join. It is a
    #: SEPARATE counter and is never required to equal ``pair_frame_sequence``.
    paper_frame_index: Optional[int] = None
    #: Media-relative seconds from the PAPER pipeline only.
    paper_timestamp_seconds: Optional[float] = None
    #: Absolute datetime from the PAIR pipeline only.
    pair_observed_at: Optional[datetime] = None

    source_paper_status: Optional[PaperEvidenceStatus] = None
    source_pair_status: Optional[PairFrameStatus] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PaperPairSpatialStatus):
            raise PaperPairSpatialContractError(f"unknown status: {self.status!r}")
        if not isinstance(self.facts, tuple):
            raise PaperPairSpatialContractError("facts must be an immutable tuple")
        for fact in self.facts:
            if not isinstance(fact, PaperPairSpatialFact):
                raise PaperPairSpatialContractError(
                    "facts must be PaperPairSpatialFact values"
                )
        if self.status is not PaperPairSpatialStatus.OK and self.facts:
            raise PaperPairSpatialContractError(
                f"degraded spatial frame ({self.status.value}) must carry no facts"
            )
        for label, value in (
            ("paper_detection_count", self.paper_detection_count),
            ("pair_count", self.pair_count),
        ):
            if not strict_index(value):
                raise PaperPairSpatialContractError(
                    f"{label} must be a non-negative int"
                )
        if self.camera_id is not None:
            _non_blank(self.camera_id, "camera_id")
        for label, value in (
            ("pair_frame_sequence", self.pair_frame_sequence),
            ("paper_frame_index", self.paper_frame_index),
        ):
            if value is not None and not strict_index(value):
                raise PaperPairSpatialContractError(
                    f"{label} must be a non-negative int"
                )
        if self.paper_timestamp_seconds is not None and not _finite(
            self.paper_timestamp_seconds
        ):
            raise PaperPairSpatialContractError(
                "paper_timestamp_seconds must be finite"
            )
        if self.pair_observed_at is not None and not isinstance(
            self.pair_observed_at, datetime
        ):
            raise PaperPairSpatialContractError("pair_observed_at must be a datetime")

        if self.source_paper_status is not None and not isinstance(
            self.source_paper_status, PaperEvidenceStatus
        ):
            raise PaperPairSpatialContractError(
                "source_paper_status must be a PaperEvidenceStatus"
            )
        if self.source_pair_status is not None and not isinstance(
            self.source_pair_status, PairFrameStatus
        ):
            raise PaperPairSpatialContractError(
                "source_pair_status must be a PairFrameStatus"
            )

    @property
    def fact_count(self) -> int:
        return len(self.facts)

    def facts_for_pair(
        self, pair_key: PersonPairKey
    ) -> tuple[PaperPairSpatialFact, ...]:
        return tuple(fact for fact in self.facts if fact.pair_key == pair_key)

    def facts_for_paper(
        self, detection_index: int
    ) -> tuple[PaperPairSpatialFact, ...]:
        return tuple(
            fact for fact in self.facts if fact.paper.detection_index == detection_index
        )
