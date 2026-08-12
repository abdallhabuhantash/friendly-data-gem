"""Pure builder: ONE pair frame + ONE SAME-frame paper frame -> spatial facts.

This module is a pure function library. It performs no model inference, opens no
camera, reads no clock, touches no global state and is imported by NOTHING on
the live path (Orchestrator, PoseRuntime, EngineRegistry, CameraManager and
PhoneRuleEngine all remain unaware of it).

It reuses the already-validated geometry helpers (``relative_point``,
``intersection_area``, ``iou``) so there is exactly ONE relative-coordinate
policy in the system.

Explicitly NOT implemented here, by contract:

* no ownership, holder, giver or receiver inference;
* no grasp/contact claim (COCO pose gives a wrist point, not a hand);
* no depth or physical-distance claim;
* no threshold, interaction zone or "paper between people" boolean;
* no temporal fusion: Task 3D (``exchange_temporal_state`` /
  ``HandoffTemporalResult``) is neither imported nor consumed.
"""

from __future__ import annotations

import math
from typing import Optional

from app.ai.region_resolver import relative_point
from app.domain.body_features import BodySide, TrackedBodyFeatures
from app.domain.geometry import BBox, intersection_area, iou
from app.domain.paper_evidence import (
    PaperDetection,
    PaperEvidenceFrame,
    PaperEvidenceStatus,
)
from app.domain.pair_geometry import (
    PairFrameStatus,
    PersonPairFrameResult,
    PersonPairGeometry,
)
from app.domain.paper_pair_spatial import (
    PaperGeometryFacts,
    PaperPairAxisProjection,
    PaperPairSpatialFact,
    PaperPairSpatialFrame,
    PaperPairSpatialStatus,
    PaperPersonSpatialFact,
    PaperWristSpatialFact,
    SameFrameJoin,
)

_SIDES: tuple[BodySide, ...] = (BodySide.LEFT, BodySide.RIGHT)


def _ratio(value: float, denominator: float) -> Optional[float]:
    if denominator <= 0.0:
        return None
    result = value / denominator
    return result if math.isfinite(result) else None


def _provenance_mismatch(
    pair_frame: PersonPairFrameResult,
    paper_frame: PaperEvidenceFrame,
    join: SameFrameJoin,
) -> Optional[str]:
    """Returns a reason string when the two inputs cannot be proven same-frame."""
    if pair_frame.camera_id is not None and pair_frame.camera_id != join.camera_id:
        return "camera_identity_mismatch"
    if (
        pair_frame.frame_sequence is not None
        and pair_frame.frame_sequence != join.frame_sequence
    ):
        return "frame_sequence_mismatch"
    if (
        paper_frame.frame_index is not None
        and paper_frame.frame_index != join.frame_sequence
    ):
        return "paper_frame_index_mismatch"
    if paper_frame.timestamp_seconds is not None:
        if join.timestamp_seconds is None:
            return "paper_timestamp_without_declared_join_timestamp"
        if (
            abs(float(paper_frame.timestamp_seconds) - float(join.timestamp_seconds))
            > join.timestamp_tolerance_seconds
        ):
            return "timestamp_disagreement"
    if (
        pair_frame.observed_at is not None
        and join.observed_at is not None
        and pair_frame.observed_at != join.observed_at
    ):
        return "observed_at_mismatch"
    return None


def _paper_facts(index: int, detection: PaperDetection) -> PaperGeometryFacts:
    box = detection.bbox
    center_x, center_y = box.center
    return PaperGeometryFacts(
        detection_index=index,
        class_name=detection.class_name,
        confidence=detection.confidence,
        bbox=box,
        center_x=center_x,
        center_y=center_y,
        width=box.width,
        height=box.height,
        diagonal=box.diagonal,
        raw_prompt=detection.raw_prompt,
    )


def _person_fact(
    paper: PaperGeometryFacts,
    person: TrackedBodyFeatures,
) -> Optional[PaperPersonSpatialFact]:
    person_box: BBox = person.person_bbox
    relative = relative_point(person_box, (paper.center_x, paper.center_y))
    if relative is None:
        return None
    overlap = intersection_area(paper.bbox, person_box)
    iou_value = iou(paper.bbox, person_box)
    if not math.isfinite(iou_value):
        iou_value = None
    person_center = person_box.center
    distance = math.hypot(
        paper.center_x - person_center[0], paper.center_y - person_center[1]
    )
    return PaperPersonSpatialFact(
        paper_detection_index=paper.detection_index,
        person_tracking_id=person.person_tracking_id,
        relative_position=relative,
        center_inside_person_bbox=relative.inside_person,
        bbox_intersection_area=overlap,
        bbox_iou=iou_value,
        center_distance_to_person_center=distance,
        center_distance_relative_to_person_diagonal=_ratio(
            distance, person_box.diagonal
        ),
    )


def _available_wrist_point(
    person: TrackedBodyFeatures, side: BodySide
) -> Optional[tuple[float, float]]:
    arm = person.arm(side)
    if arm is None:
        return None
    wrist = arm.wrist
    if not wrist.available or wrist.x is None or wrist.y is None:
        return None
    return (float(wrist.x), float(wrist.y))


def _axis_projection(
    paper: PaperGeometryFacts,
    center_a: tuple[float, float],
    center_b: tuple[float, float],
) -> PaperPairAxisProjection:
    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 0.0:
        return PaperPairAxisProjection(
            paper_detection_index=paper.detection_index,
            available=False,
            reason="degenerate_axis_coincident_centers",
        )
    px = paper.center_x - center_a[0]
    py = paper.center_y - center_a[1]
    # t is NEVER clamped: outside 0..1 is a legitimate geometric fact.
    t = (px * dx + py * dy) / length_squared
    perpendicular = abs(px * dy - py * dx) / math.sqrt(length_squared)
    if not (math.isfinite(t) and math.isfinite(perpendicular)):
        return PaperPairAxisProjection(
            paper_detection_index=paper.detection_index,
            available=False,
            reason="non_finite_projection",
        )
    return PaperPairAxisProjection(
        paper_detection_index=paper.detection_index,
        available=True,
        t=t,
        perpendicular_distance=perpendicular,
    )


def _pair_fact(
    pair: PersonPairGeometry, paper: PaperGeometryFacts
) -> PaperPairSpatialFact:
    person_a = pair.person_a
    person_b = pair.person_b
    diagonal_a = person_a.person_bbox.diagonal
    diagonal_b = person_b.person_bbox.diagonal
    mean_diagonal = (diagonal_a + diagonal_b) / 2.0

    person_facts = tuple(
        fact
        for fact in (
            _person_fact(paper, person_a),
            _person_fact(paper, person_b),
        )
        if fact is not None
    )

    wrist_facts: list[PaperWristSpatialFact] = []
    for person, own_diagonal in ((person_a, diagonal_a), (person_b, diagonal_b)):
        for side in _SIDES:
            point = _available_wrist_point(person, side)
            if point is None:
                # A missing wrist NEVER becomes (0, 0) or any placeholder fact.
                continue
            distance = math.hypot(paper.center_x - point[0], paper.center_y - point[1])
            if not math.isfinite(distance):
                continue
            wrist_facts.append(
                PaperWristSpatialFact(
                    paper_detection_index=paper.detection_index,
                    wrist_owner_tracking_id=person.person_tracking_id,
                    side=side,
                    wrist_x=point[0],
                    wrist_y=point[1],
                    distance=distance,
                    distance_relative_to_owner_person_diagonal=_ratio(
                        distance, own_diagonal
                    ),
                    distance_relative_to_mean_pair_diagonal=_ratio(
                        distance, mean_diagonal
                    ),
                )
            )

    # Purely mathematical nearest wrist in 2D image geometry. NOT a holder,
    # owner, giver, receiver, touch or grasp claim. Deterministic tie-break on
    # (distance, tracking id, anatomical side) — never on input order.
    nearest = (
        min(
            wrist_facts,
            key=lambda item: (
                item.distance,
                item.wrist_owner_tracking_id,
                item.side.value,
            ),
        )
        if wrist_facts
        else None
    )

    return PaperPairSpatialFact(
        pair_key=pair.key,
        paper=paper,
        person_facts=person_facts,
        wrist_facts=tuple(wrist_facts),
        nearest_available_wrist=nearest,
        axis_projection=_axis_projection(
            paper, person_a.person_bbox.center, person_b.person_bbox.center
        ),
    )


def build_paper_pair_spatial_frame(
    pair_frame: PersonPairFrameResult,
    paper_frame: PaperEvidenceFrame,
    join: SameFrameJoin,
) -> PaperPairSpatialFrame:
    """Derives every paper x person-pair spatial fact for ONE analysed frame.

    Pure: same inputs always give the same output. The inputs are never mutated.
    """
    if not isinstance(pair_frame, PersonPairFrameResult):
        raise TypeError("pair_frame must be a PersonPairFrameResult")
    if not isinstance(paper_frame, PaperEvidenceFrame):
        raise TypeError("paper_frame must be a PaperEvidenceFrame")
    if not isinstance(join, SameFrameJoin):
        raise TypeError("join must be an explicit SameFrameJoin")

    common = {
        "camera_id": join.camera_id,
        "frame_sequence": join.frame_sequence,
        "timestamp_seconds": join.timestamp_seconds,
        "observed_at": join.observed_at or pair_frame.observed_at,
        "source_paper_status": paper_frame.status,
        "source_pair_status": pair_frame.status,
    }

    mismatch = _provenance_mismatch(pair_frame, paper_frame, join)
    if mismatch is not None:
        return PaperPairSpatialFrame(
            status=PaperPairSpatialStatus.INCONSISTENT_INPUT,
            reason=mismatch,
            **common,
        )

    if pair_frame.status is not PairFrameStatus.OK:
        return PaperPairSpatialFrame(
            status=PaperPairSpatialStatus.PAIR_GEOMETRY_DEGRADED,
            reason=pair_frame.reason or pair_frame.status.value,
            **common,
        )

    if paper_frame.status is not PaperEvidenceStatus.OK:
        # Detector failure is NEVER valid zero-paper evidence.
        return PaperPairSpatialFrame(
            status=PaperPairSpatialStatus.PAPER_EVIDENCE_DEGRADED,
            reason=paper_frame.reason or paper_frame.status.value,
            **common,
        )

    papers = tuple(
        _paper_facts(index, detection)
        for index, detection in enumerate(paper_frame.detections)
    )
    facts = tuple(
        _pair_fact(pair, paper) for pair in pair_frame.pairs for paper in papers
    )
    return PaperPairSpatialFrame(
        status=PaperPairSpatialStatus.OK,
        facts=facts,
        paper_detection_count=len(papers),
        pair_count=len(pair_frame.pairs),
        **common,
    )
