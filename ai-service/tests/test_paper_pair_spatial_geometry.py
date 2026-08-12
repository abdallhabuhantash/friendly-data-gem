"""Deterministic pure tests for paper <-> person-pair spatial geometry (Task 3F).

Pure geometry only: no temporal fusion, no ownership, no thresholds, no runtime.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.ai.paper_pair_spatial_builder import build_paper_pair_spatial_frame
from app.ai.person_pair_geometry_builder import build_person_pair_frame_from_tracked_pose
from app.domain.body_features import BodySide
from app.domain.geometry import BBox
from app.domain.pair_geometry import PairFrameStatus, PersonPairFrameResult, PersonPairKey
from app.domain.paper_evidence import (
    PaperDetection,
    PaperEvidenceFrame,
    PaperEvidenceStatus,
)
from app.domain.paper_pair_spatial import (
    PaperPairSpatialContractError,
    PaperPairSpatialFrame,
    PaperPairSpatialStatus,
    PaperWristSpatialFact,
    SameFrameJoin,
)
from app.domain.pose import COCO_17_KEYPOINTS, PoseKeypointName, PoseStatus, coco_17_index
from app.domain.regions import RelativePoint
from app.domain.tracked_pose_observations import (
    TrackedPoseFrameResult,
    TrackedPoseFrameStatus,
    TrackedPoseKeypoint,
    TrackedPoseObservation,
)

from tests._source_scan import code_text

OBSERVED_AT = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
CAMERA_ID = "cam-1"
FRAME_SEQUENCE = 3

JOIN = SameFrameJoin(
    camera_id=CAMERA_ID,
    frame_sequence=FRAME_SEQUENCE,
    timestamp_seconds=1.5,
    observed_at=OBSERVED_AT,
)


# ------------------------------------------------------------------ fixtures


def relative(point: tuple[float, float], box: BBox) -> RelativePoint:
    return RelativePoint(
        relative_x=(point[0] - box.x) / box.width,
        relative_y=(point[1] - box.y) / box.height,
    )


def observation(
    tracking_id: str,
    box: BBox,
    points: dict[PoseKeypointName, tuple[float, float]],
    index: int,
) -> TrackedPoseObservation:
    keypoints = []
    for name in COCO_17_KEYPOINTS:
        kp_index = coco_17_index(name)
        if name in points:
            x, y = points[name]
            rel = relative((x, y), box)
            keypoints.append(
                TrackedPoseKeypoint(
                    name=name,
                    index=kp_index,
                    available=True,
                    confidence=0.8,
                    x=x,
                    y=y,
                    relative_position=rel,
                    inside_person=rel.inside_person,
                )
            )
        else:
            keypoints.append(
                TrackedPoseKeypoint(name=name, index=kp_index, available=False)
            )
    return TrackedPoseObservation(
        person_tracking_id=tracking_id,
        person_index=index,
        pose_index=index,
        person_bbox=box,
        person_confidence=0.7,
        pose_bbox=box,
        pose_confidence=0.9,
        keypoints=tuple(keypoints),
    )


def pair_frame(people: list[tuple[str, BBox, dict]]) -> PersonPairFrameResult:
    observations = tuple(
        observation(tid, box, points, index)
        for index, (tid, box, points) in enumerate(people)
    )
    return build_person_pair_frame_from_tracked_pose(
        TrackedPoseFrameResult(
            status=TrackedPoseFrameStatus.OK,
            observations=observations,
            camera_id=CAMERA_ID,
            frame_sequence=FRAME_SEQUENCE,
            observed_at=OBSERVED_AT,
            source_mode="live",
            source_pose_status=PoseStatus.OK,
            pose_instance_count=len(observations),
        )
    )


def person(
    tid: str,
    x: float,
    wrists: dict[BodySide, tuple[float, float]] | None = None,
    box: BBox | None = None,
):
    person_box = box if box is not None else BBox(x, 0.2, 0.2, 0.4)
    points: dict[PoseKeypointName, tuple[float, float]] = {}
    for side, point in (wrists or {}).items():
        name = (
            PoseKeypointName.LEFT_WRIST
            if side is BodySide.LEFT
            else PoseKeypointName.RIGHT_WRIST
        )
        points[name] = point
    return (tid, person_box, points)


def paper(bbox: BBox, confidence: float = 0.6, raw_prompt: str = "sheet of paper"):
    return PaperDetection(bbox=bbox, confidence=confidence, raw_prompt=raw_prompt)


def paper_frame(
    detections: tuple[PaperDetection, ...] = (),
    status: PaperEvidenceStatus = PaperEvidenceStatus.OK,
    timestamp_seconds: float | None = 1.5,
) -> PaperEvidenceFrame:
    return PaperEvidenceFrame(
        status=status,
        detections=detections,
        model_name="yolo-world",
        backend="open_vocab",
        reason=None if status is PaperEvidenceStatus.OK else "detector reported failure",
        frame_index=FRAME_SEQUENCE,
        timestamp_seconds=timestamp_seconds,
    )


def build(people, detections=(), *, paper_status=PaperEvidenceStatus.OK, join=JOIN):
    return build_paper_pair_spatial_frame(
        pair_frame(people), paper_frame(detections, status=paper_status), join
    )


# ------------------------------------------------------------------- basics


def test_zero_paper_detections_is_valid_empty() -> None:
    result = build([person("a", 0.1), person("b", 0.6)])
    assert result.status is PaperPairSpatialStatus.OK
    assert result.facts == ()
    assert result.paper_detection_count == 0
    assert result.pair_count == 1
    assert result.source_paper_status is PaperEvidenceStatus.OK


def test_one_paper_and_one_pair() -> None:
    result = build(
        [person("a", 0.1), person("b", 0.6)], (paper(BBox(0.4, 0.3, 0.05, 0.05)),)
    )
    assert result.status is PaperPairSpatialStatus.OK
    assert result.fact_count == 1
    fact = result.facts[0]
    assert fact.pair_key == PersonPairKey.of("a", "b")
    assert fact.paper.detection_index == 0
    assert fact.paper.class_name == "paper"
    assert fact.paper.raw_prompt == "sheet of paper"
    assert fact.paper.confidence == pytest.approx(0.6)
    assert fact.paper.center_x == pytest.approx(0.425)
    assert fact.paper.diagonal == pytest.approx(math.hypot(0.05, 0.05))
    assert len(fact.person_facts) == 2


def test_multiple_papers_remain_independent() -> None:
    result = build(
        [person("a", 0.1), person("b", 0.6)],
        (
            paper(BBox(0.15, 0.3, 0.04, 0.04), confidence=0.4),
            paper(BBox(0.65, 0.3, 0.04, 0.04), confidence=0.9),
        ),
    )
    assert result.paper_detection_count == 2
    assert result.fact_count == 2
    indexes = sorted(fact.paper.detection_index for fact in result.facts)
    assert indexes == [0, 1]
    first, second = result.facts_for_paper(0)[0], result.facts_for_paper(1)[0]
    assert first.paper.confidence == pytest.approx(0.4)
    assert second.paper.confidence == pytest.approx(0.9)
    assert first.paper.bbox != second.paper.bbox


def test_overlapping_papers_are_not_merged() -> None:
    box = BBox(0.4, 0.3, 0.05, 0.05)
    result = build([person("a", 0.1), person("b", 0.6)], (paper(box), paper(box)))
    assert result.paper_detection_count == 2
    assert result.fact_count == 2


def test_three_people_three_pairs_each_paper_independent() -> None:
    result = build(
        [person("a", 0.05), person("b", 0.4), person("c", 0.75)],
        (paper(BBox(0.4, 0.3, 0.04, 0.04)),),
    )
    assert result.pair_count == 3
    assert result.fact_count == 3
    keys = {fact.pair_key for fact in result.facts}
    assert keys == {
        PersonPairKey.of("a", "b"),
        PersonPairKey.of("a", "c"),
        PersonPairKey.of("b", "c"),
    }
    # No pair may alter another pair's facts: geometry differs per pair.
    axis_ts = {fact.pair_key: fact.axis_projection.t for fact in result.facts}
    assert len(set(axis_ts.values())) == 3


def test_inputs_unchanged_and_output_immutable() -> None:
    people = [person("a", 0.1), person("b", 0.6)]
    pairs = pair_frame(people)
    papers = paper_frame((paper(BBox(0.4, 0.3, 0.05, 0.05)),))
    pairs_before = copy.deepcopy(pairs)
    papers_before = copy.deepcopy(papers)
    result = build_paper_pair_spatial_frame(pairs, papers, JOIN)
    assert pairs == pairs_before
    assert papers == papers_before
    assert isinstance(result, PaperPairSpatialFrame)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.facts[0].paper.confidence = 0.1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = PaperPairSpatialStatus.INCONSISTENT_INPUT  # type: ignore[misc]


def test_builder_is_deterministic() -> None:
    args = ([person("a", 0.1), person("b", 0.6)], (paper(BBox(0.4, 0.3, 0.05, 0.05)),))
    assert build(*args) == build(*args)


# ---------------------------------------------------------- person geometry


def _person_fact(result, tracking_id: str, detection_index: int = 0):
    fact = result.facts_for_paper(detection_index)[0]
    return next(
        item for item in fact.person_facts if item.person_tracking_id == tracking_id
    )


def test_paper_center_inside_person_a_bbox() -> None:
    result = build(
        [person("a", 0.1), person("b", 0.6)], (paper(BBox(0.16, 0.36, 0.04, 0.04)),)
    )
    a = _person_fact(result, "a")
    assert a.center_inside_person_bbox is True
    assert 0.0 <= a.relative_x <= 1.0
    assert 0.0 <= a.relative_y <= 1.0


def test_paper_center_outside_person_a_bbox() -> None:
    result = build(
        [person("a", 0.1), person("b", 0.6)], (paper(BBox(0.62, 0.36, 0.04, 0.04)),)
    )
    a = _person_fact(result, "a")
    assert a.center_inside_person_bbox is False


def test_relative_coordinates_are_never_clamped() -> None:
    result = build(
        [person("a", 0.4), person("b", 0.7)], (paper(BBox(0.02, 0.02, 0.04, 0.04)),)
    )
    a = _person_fact(result, "a")
    assert a.relative_x < 0.0
    assert a.relative_y < 0.0


def test_paper_person_overlap_and_iou() -> None:
    # person a box: x 0.1..0.3, y 0.2..0.6
    result = build(
        [person("a", 0.1), person("b", 0.6)], (paper(BBox(0.2, 0.3, 0.05, 0.05)),)
    )
    a = _person_fact(result, "a")
    assert a.bbox_intersection_area == pytest.approx(0.05 * 0.05)
    assert a.bbox_iou is not None and a.bbox_iou > 0.0


def test_paper_person_no_overlap() -> None:
    result = build(
        [person("a", 0.1), person("b", 0.6)], (paper(BBox(0.4, 0.05, 0.05, 0.05)),)
    )
    a = _person_fact(result, "a")
    assert a.bbox_intersection_area == pytest.approx(0.0)
    assert a.bbox_iou == pytest.approx(0.0)


def test_boundary_touching_geometry_has_zero_overlap() -> None:
    # paper starts exactly where person a's box ends (x2 == 0.3).
    result = build(
        [person("a", 0.1), person("b", 0.6)], (paper(BBox(0.3, 0.3, 0.05, 0.05)),)
    )
    a = _person_fact(result, "a")
    assert a.bbox_intersection_area == pytest.approx(0.0)


def test_scale_equivalent_normalized_geometry_is_deterministic() -> None:
    small = build(
        [
            person("a", 0.0, box=BBox(0.10, 0.10, 0.10, 0.20)),
            person("b", 0.0, box=BBox(0.30, 0.10, 0.10, 0.20)),
        ],
        (paper(BBox(0.20, 0.15, 0.02, 0.04)),),
    )
    large = build(
        [
            person("a", 0.0, box=BBox(0.20, 0.20, 0.20, 0.40)),
            person("b", 0.0, box=BBox(0.60, 0.20, 0.20, 0.40)),
        ],
        (paper(BBox(0.40, 0.30, 0.04, 0.08)),),
    )
    a_small = _person_fact(small, "a")
    a_large = _person_fact(large, "a")
    assert a_small.relative_x == pytest.approx(a_large.relative_x)
    assert a_small.relative_y == pytest.approx(a_large.relative_y)
    assert a_small.center_distance_relative_to_person_diagonal == pytest.approx(
        a_large.center_distance_relative_to_person_diagonal
    )
    assert small.facts[0].axis_projection.t == pytest.approx(
        large.facts[0].axis_projection.t
    )


def test_center_distance_relative_to_person_diagonal() -> None:
    result = build(
        [person("a", 0.1), person("b", 0.6)], (paper(BBox(0.4, 0.38, 0.04, 0.04)),)
    )
    a = _person_fact(result, "a")
    diagonal = BBox(0.1, 0.2, 0.2, 0.4).diagonal
    assert a.center_distance_relative_to_person_diagonal == pytest.approx(
        a.center_distance_to_person_center / diagonal
    )


# ------------------------------------------------------------------- wrists


def _wrist(result, tracking_id: str, side: BodySide):
    return next(
        item
        for item in result.facts[0].wrist_facts
        if item.wrist_owner_tracking_id == tracking_id and item.side is side
    )


def test_paper_near_each_wrist_of_both_people() -> None:
    people = [
        person(
            "a",
            0.1,
            {BodySide.LEFT: (0.12, 0.30), BodySide.RIGHT: (0.28, 0.30)},
        ),
        person(
            "b",
            0.6,
            {BodySide.LEFT: (0.62, 0.30), BodySide.RIGHT: (0.78, 0.30)},
        ),
    ]
    cases = {
        ("a", BodySide.LEFT): BBox(0.11, 0.29, 0.02, 0.02),
        ("a", BodySide.RIGHT): BBox(0.27, 0.29, 0.02, 0.02),
        ("b", BodySide.LEFT): BBox(0.61, 0.29, 0.02, 0.02),
        ("b", BodySide.RIGHT): BBox(0.77, 0.29, 0.02, 0.02),
    }
    for (tid, side), box in cases.items():
        result = build(people, (paper(box),))
        fact = result.facts[0]
        assert len(fact.wrist_facts) == 4
        nearest = fact.nearest_available_wrist
        assert nearest is not None
        assert (nearest.wrist_owner_tracking_id, nearest.side) == (tid, side)


def test_wrist_distance_and_normalizations() -> None:
    people = [
        person("a", 0.1, {BodySide.LEFT: (0.20, 0.30)}),
        person("b", 0.6),
    ]
    result = build(people, (paper(BBox(0.20, 0.34, 0.04, 0.04)),))
    wrist = _wrist(result, "a", BodySide.LEFT)
    expected = math.hypot(0.22 - 0.20, 0.36 - 0.30)
    assert wrist.distance == pytest.approx(expected)
    diagonal = BBox(0.1, 0.2, 0.2, 0.4).diagonal
    assert wrist.distance_relative_to_owner_person_diagonal == pytest.approx(
        expected / diagonal
    )
    assert wrist.distance_relative_to_mean_pair_diagonal == pytest.approx(
        expected / diagonal
    )


def test_missing_wrist_produces_no_fake_spatial_fact() -> None:
    people = [
        person("a", 0.1, {BodySide.LEFT: (0.20, 0.30)}),
        person("b", 0.6),
    ]
    result = build(people, (paper(BBox(0.40, 0.30, 0.04, 0.04)),))
    sides = {
        (item.wrist_owner_tracking_id, item.side)
        for item in result.facts[0].wrist_facts
    }
    assert sides == {("a", BodySide.LEFT)}
    for item in result.facts[0].wrist_facts:
        assert (item.wrist_x, item.wrist_y) != (0.0, 0.0)


def test_no_wrists_available_yields_no_nearest_wrist() -> None:
    result = build(
        [person("a", 0.1), person("b", 0.6)], (paper(BBox(0.4, 0.3, 0.04, 0.04)),)
    )
    fact = result.facts[0]
    assert fact.wrist_facts == ()
    assert fact.nearest_available_wrist is None


def test_equal_distance_wrist_tie_is_deterministic() -> None:
    people = [
        person("a", 0.1, {BodySide.LEFT: (0.20, 0.28), BodySide.RIGHT: (0.20, 0.44)}),
        person("b", 0.6),
    ]
    result = build(people, (paper(BBox(0.19, 0.35, 0.02, 0.02)),))
    fact = result.facts[0]
    left = _wrist(result, "a", BodySide.LEFT)
    right = _wrist(result, "a", BodySide.RIGHT)
    assert left.distance == pytest.approx(right.distance)
    assert fact.nearest_available_wrist is left
    assert fact.nearest_available_wrist.side is BodySide.LEFT


def test_equal_distance_across_people_tie_breaks_on_tracking_id() -> None:
    people = [
        person(
            "a",
            0.0,
            {BodySide.LEFT: (0.25, 0.30)},
            box=BBox(0.20, 0.20, 0.20, 0.40),
        ),
        person(
            "b",
            0.0,
            {BodySide.LEFT: (0.75, 0.30)},
            box=BBox(0.60, 0.20, 0.20, 0.40),
        ),
    ]
    result = build(people, (paper(BBox(0.48, 0.29, 0.04, 0.02)),))
    nearest = result.facts[0].nearest_available_wrist
    assert nearest is not None
    assert nearest.wrist_owner_tracking_id == "a"


def test_reversed_subject_order_gives_canonical_result() -> None:
    forward = [
        person("a", 0.1, {BodySide.LEFT: (0.20, 0.30)}),
        person("b", 0.6, {BodySide.RIGHT: (0.70, 0.30)}),
    ]
    reversed_people = list(reversed(forward))
    detections = (paper(BBox(0.4, 0.3, 0.04, 0.04)),)
    first = build(forward, detections)
    second = build(reversed_people, detections)
    assert first.facts[0].pair_key == second.facts[0].pair_key
    assert first.facts[0].wrist_facts == second.facts[0].wrist_facts
    assert first.facts[0].axis_projection == second.facts[0].axis_projection


# ---------------------------------------------------------------- pair axis


def _axis(paper_box: BBox, people=None):
    people = people or [
        person("a", 0.0, box=BBox(0.20, 0.20, 0.20, 0.40)),
        person("b", 0.0, box=BBox(0.60, 0.20, 0.20, 0.40)),
    ]
    return build(people, (paper(paper_box),)).facts[0].axis_projection


def test_paper_axis_t_zero_at_person_a_center() -> None:
    axis = _axis(BBox(0.29, 0.39, 0.02, 0.02))
    assert axis.available is True
    assert axis.t == pytest.approx(0.0)
    assert axis.perpendicular_distance == pytest.approx(0.0)


def test_paper_axis_t_one_at_person_b_center() -> None:
    axis = _axis(BBox(0.69, 0.39, 0.02, 0.02))
    assert axis.t == pytest.approx(1.0)


def test_paper_axis_t_between_zero_and_one() -> None:
    axis = _axis(BBox(0.49, 0.39, 0.02, 0.02))
    assert axis.t == pytest.approx(0.5)


def test_paper_axis_t_below_zero_is_not_clamped() -> None:
    axis = _axis(BBox(0.10, 0.39, 0.02, 0.02))
    assert axis.t < 0.0


def test_paper_axis_t_above_one_is_not_clamped() -> None:
    axis = _axis(BBox(0.90, 0.39, 0.02, 0.02))
    assert axis.t > 1.0


def test_paper_axis_perpendicular_distance() -> None:
    axis = _axis(BBox(0.49, 0.19, 0.02, 0.02))
    # paper centre (0.50, 0.20); axis is the horizontal line y = 0.40.
    assert axis.t == pytest.approx(0.5)
    assert axis.perpendicular_distance == pytest.approx(0.20)


def test_coincident_person_centers_make_axis_unavailable() -> None:
    people = [
        person("a", 0.0, box=BBox(0.40, 0.30, 0.20, 0.20)),
        person("b", 0.0, box=BBox(0.40, 0.30, 0.20, 0.20)),
    ]
    axis = _axis(BBox(0.10, 0.10, 0.02, 0.02), people)
    assert axis.available is False
    assert axis.t is None
    assert axis.perpendicular_distance is None
    assert axis.reason == "degenerate_axis_coincident_centers"


# --------------------------------------------------------------- provenance


def test_same_frame_inputs_succeed() -> None:
    result = build([person("a", 0.1), person("b", 0.6)])
    assert result.status is PaperPairSpatialStatus.OK
    assert result.camera_id == CAMERA_ID
    assert result.frame_sequence == FRAME_SEQUENCE
    assert result.timestamp_seconds == pytest.approx(1.5)


def test_mismatched_frame_sequence_rejected() -> None:
    result = build_paper_pair_spatial_frame(
        pair_frame([person("a", 0.1), person("b", 0.6)]),
        paper_frame((paper(BBox(0.4, 0.3, 0.04, 0.04)),)),
        dataclasses.replace(JOIN, frame_sequence=99),
    )
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.facts == ()
    assert result.reason == "frame_sequence_mismatch"


def test_mismatched_paper_frame_index_rejected() -> None:
    papers = PaperEvidenceFrame(
        status=PaperEvidenceStatus.OK,
        detections=(paper(BBox(0.4, 0.3, 0.04, 0.04)),),
        frame_index=FRAME_SEQUENCE + 1,
        timestamp_seconds=1.5,
    )
    result = build_paper_pair_spatial_frame(
        pair_frame([person("a", 0.1), person("b", 0.6)]), papers, JOIN
    )
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.reason == "paper_frame_index_mismatch"


def test_mismatched_camera_identity_rejected() -> None:
    result = build_paper_pair_spatial_frame(
        pair_frame([person("a", 0.1), person("b", 0.6)]),
        paper_frame(),
        dataclasses.replace(JOIN, camera_id="cam-other"),
    )
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.reason == "camera_identity_mismatch"


def test_timestamp_disagreement_rejected() -> None:
    result = build_paper_pair_spatial_frame(
        pair_frame([person("a", 0.1), person("b", 0.6)]),
        paper_frame(timestamp_seconds=9.75),
        JOIN,
    )
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.reason == "timestamp_disagreement"


def test_paper_timestamp_without_declared_join_timestamp_rejected() -> None:
    result = build_paper_pair_spatial_frame(
        pair_frame([person("a", 0.1), person("b", 0.6)]),
        paper_frame(timestamp_seconds=1.5),
        SameFrameJoin(camera_id=CAMERA_ID, frame_sequence=FRAME_SEQUENCE),
    )
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.reason == "paper_timestamp_without_declared_join_timestamp"


def test_timestamp_within_explicit_tolerance_is_accepted() -> None:
    result = build_paper_pair_spatial_frame(
        pair_frame([person("a", 0.1), person("b", 0.6)]),
        paper_frame(timestamp_seconds=1.52),
        dataclasses.replace(JOIN, timestamp_tolerance_seconds=0.05),
    )
    assert result.status is PaperPairSpatialStatus.OK


def test_observed_at_disagreement_rejected() -> None:
    result = build_paper_pair_spatial_frame(
        pair_frame([person("a", 0.1), person("b", 0.6)]),
        paper_frame(),
        dataclasses.replace(
            JOIN, observed_at=datetime(2026, 1, 2, tzinfo=timezone.utc)
        ),
    )
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.reason == "observed_at_mismatch"


def test_join_must_be_explicit() -> None:
    with pytest.raises(TypeError):
        build_paper_pair_spatial_frame(
            pair_frame([person("a", 0.1)]), paper_frame(), None  # type: ignore[arg-type]
        )


# ----------------------------------------------------------- degraded input


@pytest.mark.parametrize(
    "status",
    [
        PaperEvidenceStatus.MODEL_UNAVAILABLE,
        PaperEvidenceStatus.INFERENCE_FAILED,
        PaperEvidenceStatus.MALFORMED_RESULT,
        PaperEvidenceStatus.MODEL_SCHEMA_MISMATCH,
        PaperEvidenceStatus.PROMPT_CONFIGURATION_INVALID,
    ],
)
def test_degraded_paper_frame_yields_degraded_spatial_result(status) -> None:
    result = build([person("a", 0.1), person("b", 0.6)], paper_status=status)
    assert result.status is PaperPairSpatialStatus.PAPER_EVIDENCE_DEGRADED
    assert result.facts == ()
    assert result.paper_detection_count == 0
    assert result.source_paper_status is status


@pytest.mark.parametrize(
    "status",
    [
        PairFrameStatus.POSE_UNAVAILABLE,
        PairFrameStatus.ASSOCIATION_UNAVAILABLE,
        PairFrameStatus.INCONSISTENT_INPUT,
    ],
)
def test_degraded_pair_frame_yields_degraded_spatial_result(status) -> None:
    degraded = PersonPairFrameResult(
        status=status,
        camera_id=CAMERA_ID,
        frame_sequence=FRAME_SEQUENCE,
        observed_at=OBSERVED_AT,
    )
    result = build_paper_pair_spatial_frame(
        degraded, paper_frame((paper(BBox(0.4, 0.3, 0.04, 0.04)),)), JOIN
    )
    assert result.status is PaperPairSpatialStatus.PAIR_GEOMETRY_DEGRADED
    assert result.facts == ()
    assert result.pair_count == 0
    assert result.source_pair_status is status


def test_both_degraded_never_yields_partial_facts() -> None:
    degraded = PersonPairFrameResult(
        status=PairFrameStatus.POSE_UNAVAILABLE,
        camera_id=CAMERA_ID,
        frame_sequence=FRAME_SEQUENCE,
    )
    result = build_paper_pair_spatial_frame(
        degraded,
        paper_frame(status=PaperEvidenceStatus.INFERENCE_FAILED),
        JOIN,
    )
    assert result.status is not PaperPairSpatialStatus.OK
    assert result.facts == ()


def test_degraded_spatial_frame_cannot_carry_facts() -> None:
    with pytest.raises(PaperPairSpatialContractError):
        PaperPairSpatialFrame(
            status=PaperPairSpatialStatus.PAPER_EVIDENCE_DEGRADED,
            facts=build(
                [person("a", 0.1), person("b", 0.6)],
                (paper(BBox(0.4, 0.3, 0.04, 0.04)),),
            ).facts,
        )


def test_missing_wrist_fact_cannot_be_faked_by_contract() -> None:
    with pytest.raises(PaperPairSpatialContractError):
        PaperWristSpatialFact(
            paper_detection_index=-1,
            wrist_owner_tracking_id="a",
            side=BodySide.LEFT,
            wrist_x=0.0,
            wrist_y=0.0,
            distance=0.0,
        )
    with pytest.raises(PaperPairSpatialContractError):
        PaperWristSpatialFact(
            paper_detection_index=0,
            wrist_owner_tracking_id="   ",
            side=BodySide.LEFT,
            wrist_x=0.0,
            wrist_y=0.0,
            distance=0.0,
        )


# ----------------------------------------------------- static / vocabulary


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "app" / "domain" / "paper_pair_spatial.py"
BUILDER = ROOT / "app" / "ai" / "paper_pair_spatial_builder.py"

FORBIDDEN_TOKENS = (
    "cheating",
    "suspicious",
    "transferred",
    "giver",
    "receiver",
    "holder",
    "owner",
    "grasp",
    "contact",
    "exchange_completed",
    "paper_handoff",
    "paper_between",
    "interaction_zone",
    "transfer_zone",
    "handoff_zone",
)


@pytest.mark.parametrize("path", [DOMAIN, BUILDER])
def test_no_semantic_decision_vocabulary_in_code(path: Path) -> None:
    code = code_text(path).lower()
    for token in FORBIDDEN_TOKENS:
        # Word-boundary match: ``wrist_owner_tracking_id`` is neutral wrist
        # provenance, never a claim of paper ownership.
        assert re.search(rf"\b{token}\b", code) is None, (
            f"{path.name} must not express {token!r}"
        )


@pytest.mark.parametrize("path", [DOMAIN, BUILDER])
def test_no_temporal_or_task_3d_fusion(path: Path) -> None:
    code = code_text(path).lower()
    for token in (
        "exchange_temporal_state",
        "handofftemporalresult",
        "handoff_temporal",
        "velocity",
        "dwell",
        "previous_frame",
        "history",
        "state_machine",
    ):
        assert token not in code, f"{path.name} must stay purely same-frame ({token})"


@pytest.mark.parametrize("path", [DOMAIN, BUILDER])
def test_no_depth_or_physical_unit_claims(path: Path) -> None:
    code = code_text(path).lower()
    for token in ("centimet", "metres", "meters", "depth_", "millimet"):
        assert token not in code


def test_live_runtime_modules_do_not_import_this_layer() -> None:
    runtime_modules = (
        ROOT / "app" / "runtime" / "orchestrator.py",
        ROOT / "app" / "ai" / "pose_runtime.py",
        ROOT / "app" / "ai" / "engine_registry.py",
        ROOT / "app" / "runtime" / "camera_manager.py",
        ROOT / "app" / "ai" / "phone_rule_engine.py",
    )
    for module in runtime_modules:
        if not module.exists():
            continue
        text = module.read_text(encoding="utf-8")
        assert "paper_pair_spatial" not in text, f"{module.name} must stay unaware"


def test_no_ownership_fields_on_the_output_contract() -> None:
    result = build(
        [
            person("a", 0.1, {BodySide.LEFT: (0.20, 0.30)}),
            person("b", 0.6, {BodySide.RIGHT: (0.70, 0.30)}),
        ],
        (paper(BBox(0.4, 0.3, 0.04, 0.04)),),
    )
    fact = result.facts[0]
    for obj in (fact, fact.paper, fact.axis_projection, *fact.person_facts, *fact.wrist_facts):
        names = {f.name for f in dataclasses.fields(obj)}
        assert not {
            "paper_owner",
            "person_holding_paper",
            "holder",
            "owner",
            "giver",
            "receiver",
            "transferred_to",
            "paper_between_people",
            "paper_in_interaction_zone",
        } & names
    # There is no paper identity across frames.
    assert "paper_tracking_id" not in {
        f.name for f in dataclasses.fields(fact.paper)
    }


def test_paper_confidence_is_never_fused() -> None:
    result = build(
        [
            person("a", 0.1, {BodySide.LEFT: (0.20, 0.30)}),
            person("b", 0.6, {BodySide.RIGHT: (0.70, 0.30)}),
        ],
        (paper(BBox(0.4, 0.3, 0.04, 0.04), confidence=0.42),),
    )
    fact = result.facts[0]
    assert fact.paper.confidence == pytest.approx(0.42)
    all_names = set()
    for obj in (fact, fact.paper, *fact.person_facts, *fact.wrist_facts):
        all_names |= {f.name for f in dataclasses.fields(obj)}
    assert not any("fused" in name or "combined" in name or "score" in name for name in all_names)
