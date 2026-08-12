"""Deterministic pure tests for same-frame person-pair geometry (Task 3C)."""

from __future__ import annotations

import ast
import dataclasses
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.ai.body_feature_frame_builder import build_body_feature_frame
from app.ai.person_pair_geometry_builder import (
    build_person_pair_frame,
    build_person_pair_frame_from_tracked_pose,
)
from app.domain.body_features import BodySide
from app.domain.geometry import BBox
from app.domain.pair_geometry import (
    PairFrameStatus,
    PairGeometryContractError,
    PersonPairFrameResult,
    PersonPairKey,
)
from app.domain.pose import COCO_17_KEYPOINTS, PoseKeypointName, PoseStatus, coco_17_index
from app.domain.regions import RelativePoint
from app.domain.tracked_pose_observations import (
    TrackedPoseFrameResult,
    TrackedPoseFrameStatus,
    TrackedPoseKeypoint,
    TrackedPoseObservation,
)

OBSERVED_AT = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def relative(point: tuple[float, float], box: BBox) -> RelativePoint:
    return RelativePoint(
        relative_x=(point[0] - box.x) / box.width,
        relative_y=(point[1] - box.y) / box.height,
    )


def observation(
    tracking_id: str,
    box: BBox,
    points: dict[PoseKeypointName, tuple[float, float]],
    person_index: int,
    pose_index: int,
) -> TrackedPoseObservation:
    keypoints = []
    for name in COCO_17_KEYPOINTS:
        index = coco_17_index(name)
        if name in points:
            x, y = points[name]
            rel = relative((x, y), box)
            keypoints.append(
                TrackedPoseKeypoint(
                    name=name,
                    index=index,
                    available=True,
                    confidence=0.8,
                    x=x,
                    y=y,
                    relative_position=rel,
                    inside_person=rel.inside_person,
                )
            )
        else:
            keypoints.append(TrackedPoseKeypoint(name=name, index=index, available=False))
    return TrackedPoseObservation(
        person_tracking_id=tracking_id,
        person_index=person_index,
        pose_index=pose_index,
        person_bbox=box,
        person_confidence=0.7,
        pose_bbox=box,
        pose_confidence=0.9,
        keypoints=tuple(keypoints),
    )


def frame(people: list[tuple[str, BBox, dict]]) -> TrackedPoseFrameResult:
    observations = tuple(
        observation(tid, box, points, index, index)
        for index, (tid, box, points) in enumerate(people)
    )
    return TrackedPoseFrameResult(
        status=TrackedPoseFrameStatus.OK,
        observations=observations,
        camera_id="cam-1",
        frame_sequence=3,
        observed_at=OBSERVED_AT,
        source_mode="live",
        source_pose_status=PoseStatus.OK,
        pose_instance_count=len(observations),
    )


def pair_frame(people: list[tuple[str, BBox, dict]]) -> PersonPairFrameResult:
    return build_person_pair_frame_from_tracked_pose(frame(people))


def person(tid: str, x: float, wrists: dict[BodySide, tuple[float, float]] | None = None):
    box = BBox(x, 0.2, 0.2, 0.4)
    points: dict[PoseKeypointName, tuple[float, float]] = {}
    for side, point in (wrists or {}).items():
        points[
            PoseKeypointName.LEFT_WRIST if side is BodySide.LEFT else PoseKeypointName.RIGHT_WRIST
        ] = point
    return (tid, box, points)


# ---------------------------------------------------------------- enumeration


def test_zero_people_yields_zero_pairs() -> None:
    result = pair_frame([])
    assert result.status is PairFrameStatus.OK
    assert result.pair_count == 0
    assert result.subject_count == 0


def test_one_person_yields_zero_pairs() -> None:
    result = pair_frame([person("a", 0.1)])
    assert result.pair_count == 0
    assert result.subject_count == 1


def test_two_people_yield_one_pair() -> None:
    result = pair_frame([person("a", 0.1), person("b", 0.5)])
    assert result.pair_count == 1
    assert result.pairs[0].key == PersonPairKey.of("b", "a")


def test_three_people_yield_three_unique_pairs() -> None:
    result = pair_frame([person("a", 0.05), person("b", 0.35), person("c", 0.65)])
    keys = {pair.key.tracking_ids for pair in result.pairs}
    assert result.pair_count == 3
    assert keys == {("a", "b"), ("a", "c"), ("b", "c")}


def test_four_people_yield_six_unique_pairs() -> None:
    result = pair_frame(
        [person("a", 0.02), person("b", 0.24), person("c", 0.46), person("d", 0.68)]
    )
    keys = {pair.key.tracking_ids for pair in result.pairs}
    assert result.pair_count == 6
    assert len(keys) == 6


def test_reversed_subject_order_gives_identical_pair_identities() -> None:
    forward = pair_frame([person("a", 0.05), person("b", 0.35), person("c", 0.65)])
    reversed_frame = pair_frame(
        [person("c", 0.65), person("b", 0.35), person("a", 0.05)]
    )
    assert [p.key for p in forward.pairs] == [p.key for p in reversed_frame.pairs]
    assert [p.center_distance for p in forward.pairs] == pytest.approx(
        [p.center_distance for p in reversed_frame.pairs]
    )


def test_no_person_is_ever_paired_with_themselves() -> None:
    result = pair_frame([person("a", 0.05), person("b", 0.35), person("c", 0.65)])
    for pair in result.pairs:
        assert pair.person_a.person_tracking_id != pair.person_b.person_tracking_id


def test_duplicate_tracking_ids_are_rejected_not_dropped() -> None:
    box = BBox(0.1, 0.2, 0.2, 0.4)
    with pytest.raises(Exception):
        frame([("a", box, {}), ("a", BBox(0.5, 0.2, 0.2, 0.4), {})])


def test_pair_key_is_symmetric_and_rejects_self_pairs() -> None:
    assert PersonPairKey.of("z", "a") == PersonPairKey.of("a", "z")
    assert PersonPairKey.of("z", "a").tracking_ids == ("a", "z")
    with pytest.raises(PairGeometryContractError):
        PersonPairKey.of("a", "a")
    with pytest.raises(PairGeometryContractError):
        PersonPairKey.of("a", "  ")
    with pytest.raises(PairGeometryContractError):
        PersonPairKey(first_tracking_id="z", second_tracking_id="a")


def test_pair_lookup_is_order_independent() -> None:
    result = pair_frame([person("a", 0.1), person("b", 0.5)])
    assert result.pair("b", "a") is result.pairs[0]
    assert result.pair("a", "zzz") is None


# ------------------------------------------------------------------- wrists


def test_all_four_wrist_combinations_are_preserved() -> None:
    result = pair_frame(
        [
            person("a", 0.1, {BodySide.LEFT: (0.15, 0.4), BodySide.RIGHT: (0.25, 0.4)}),
            person("b", 0.5, {BodySide.LEFT: (0.55, 0.4), BodySide.RIGHT: (0.65, 0.4)}),
        ]
    )
    pair = result.pairs[0]
    combos = {(w.side_a, w.side_b) for w in pair.wrist_pairs}
    assert len(pair.wrist_pairs) == 4
    assert combos == {
        (BodySide.LEFT, BodySide.LEFT),
        (BodySide.LEFT, BodySide.RIGHT),
        (BodySide.RIGHT, BodySide.LEFT),
        (BodySide.RIGHT, BodySide.RIGHT),
    }


def test_one_wrist_each_gives_exactly_one_wrist_pair() -> None:
    result = pair_frame(
        [
            person("a", 0.1, {BodySide.RIGHT: (0.25, 0.4)}),
            person("b", 0.5, {BodySide.LEFT: (0.55, 0.4)}),
        ]
    )
    pair = result.pairs[0]
    assert len(pair.wrist_pairs) == 1
    assert pair.wrist_pairs[0].side_a is BodySide.RIGHT
    assert pair.wrist_pairs[0].side_b is BodySide.LEFT
    assert pair.wrist_pairs[0].distance == pytest.approx(0.3)


def test_one_person_without_wrists_gives_no_wrist_pairs() -> None:
    result = pair_frame(
        [person("a", 0.1, {BodySide.LEFT: (0.15, 0.4)}), person("b", 0.5)]
    )
    pair = result.pairs[0]
    assert pair.wrist_pairs == ()
    assert pair.nearest_available_wrist_pair is None
    assert pair.has_available_wrist_pair is False


def test_both_people_without_wrists_gives_no_wrist_facts() -> None:
    result = pair_frame([person("a", 0.1), person("b", 0.5)])
    pair = result.pairs[0]
    assert pair.wrist_pairs == ()
    assert pair.nearest_available_wrist_pair is None
    assert pair.wrists_relative_to_other_person == ()
    assert pair.wrist_axis_projections == ()


def test_nearest_available_wrist_pair_is_the_mathematical_minimum() -> None:
    result = pair_frame(
        [
            person("a", 0.1, {BodySide.LEFT: (0.15, 0.4), BodySide.RIGHT: (0.28, 0.4)}),
            person("b", 0.5, {BodySide.LEFT: (0.52, 0.4), BodySide.RIGHT: (0.68, 0.4)}),
        ]
    )
    nearest = result.pairs[0].nearest_available_wrist_pair
    assert nearest is not None
    assert (nearest.side_a, nearest.side_b) == (BodySide.RIGHT, BodySide.LEFT)
    assert nearest.distance == pytest.approx(0.24)


def test_equal_distance_tie_is_deterministic_and_order_independent() -> None:
    a_wrists = {BodySide.LEFT: (0.2, 0.4), BodySide.RIGHT: (0.2, 0.4)}
    b_wrists = {BodySide.LEFT: (0.6, 0.4), BodySide.RIGHT: (0.6, 0.4)}
    forward = pair_frame([person("a", 0.1, a_wrists), person("b", 0.5, b_wrists)])
    backward = pair_frame([person("b", 0.5, b_wrists), person("a", 0.1, a_wrists)])
    first = forward.pairs[0].nearest_available_wrist_pair
    second = backward.pairs[0].nearest_available_wrist_pair
    assert first is not None and second is not None
    assert (first.side_a, first.side_b) == (BodySide.LEFT, BodySide.LEFT)
    assert (first.side_a, first.side_b) == (second.side_a, second.side_b)


def test_missing_wrist_never_becomes_zero_zero() -> None:
    result = pair_frame([person("a", 0.1, {BodySide.LEFT: (0.15, 0.4)}), person("b", 0.5)])
    subject_b = result.pairs[0].person_b
    assert subject_b.left_arm.wrist.x is None
    assert subject_b.right_arm.wrist.x is None
    for wrist_fact in result.pairs[0].wrists_relative_to_other_person:
        assert wrist_fact.wrist_owner_tracking_id == "a"


def test_wrist_inside_other_person_bbox_is_a_raw_fact_only() -> None:
    result = pair_frame(
        [
            person("a", 0.1, {BodySide.RIGHT: (0.55, 0.4)}),
            person("b", 0.5),
        ]
    )
    facts = result.pairs[0].wrists_relative_to_other_person
    assert len(facts) == 1
    fact = facts[0]
    assert fact.inside_other_person_bbox is True
    assert 0.0 <= fact.relative_x <= 1.0
    assert not hasattr(fact, "touching")
    assert not hasattr(fact, "contact")


def test_wrist_outside_other_bbox_has_unclamped_relative_coordinates() -> None:
    result = pair_frame(
        [
            person("a", 0.1, {BodySide.LEFT: (0.05, 0.1)}),
            person("b", 0.5),
        ]
    )
    fact = result.pairs[0].wrists_relative_to_other_person[0]
    assert fact.inside_other_person_bbox is False
    assert fact.relative_x < 0.0
    assert fact.relative_y < 0.0


def test_wrist_pair_normalizations_are_present_and_finite() -> None:
    result = pair_frame(
        [
            person("a", 0.1, {BodySide.LEFT: (0.15, 0.4)}),
            person("b", 0.5, {BodySide.LEFT: (0.55, 0.4)}),
        ]
    )
    wrist_pair = result.pairs[0].wrist_pairs[0]
    diagonal = BBox(0.1, 0.2, 0.2, 0.4).diagonal
    assert wrist_pair.distance_relative_to_person_a_diagonal == pytest.approx(
        wrist_pair.distance / diagonal
    )
    assert wrist_pair.distance_relative_to_mean_person_diagonal == pytest.approx(
        wrist_pair.distance / diagonal
    )


# ------------------------------------------------------------------ geometry


def test_person_centers_and_center_distance() -> None:
    pair = pair_frame([person("a", 0.1), person("b", 0.5)]).pairs[0]
    assert pair.person_a_center == pytest.approx((0.2, 0.4))
    assert pair.person_b_center == pytest.approx((0.6, 0.4))
    assert pair.center_distance == pytest.approx(0.4)


def test_disjoint_boxes_have_positive_separation_and_zero_overlap() -> None:
    pair = pair_frame([person("a", 0.1), person("b", 0.5)]).pairs[0]
    assert pair.bbox_min_separation == pytest.approx(0.2)
    assert pair.bbox_intersection_area == pytest.approx(0.0)
    assert pair.bbox_iou == pytest.approx(0.0)


def test_overlapping_boxes_have_zero_separation_and_positive_iou() -> None:
    pair = pair_frame([person("a", 0.1), person("b", 0.2)]).pairs[0]
    assert pair.bbox_min_separation == pytest.approx(0.0)
    assert pair.bbox_intersection_area > 0.0
    assert pair.bbox_iou > 0.0


def test_touching_boundaries_have_zero_separation_and_zero_area() -> None:
    pair = pair_frame([person("a", 0.1), person("b", 0.3)]).pairs[0]
    assert pair.bbox_min_separation == pytest.approx(0.0)
    assert pair.bbox_intersection_area == pytest.approx(0.0)


def test_center_distance_normalizations() -> None:
    pair = pair_frame([person("a", 0.1), person("b", 0.5)]).pairs[0]
    diagonal = BBox(0.1, 0.2, 0.2, 0.4).diagonal
    assert pair.center_distance_relative_to_person_a_diagonal == pytest.approx(
        0.4 / diagonal
    )
    assert pair.center_distance_relative_to_person_b_diagonal == pytest.approx(
        0.4 / diagonal
    )
    assert pair.center_distance_relative_to_mean_person_diagonal == pytest.approx(
        0.4 / diagonal
    )


def test_normalization_is_scale_equivalent() -> None:
    small = pair_frame([person("a", 0.1), person("b", 0.5)]).pairs[0]
    big = build_person_pair_frame_from_tracked_pose(
        frame(
            [
                ("a", BBox(0.1, 0.1, 0.4, 0.8), {}),
                ("b", BBox(0.5, 0.1, 0.4, 0.8), {}),
            ]
        )
    ).pairs[0]
    assert big.center_distance_relative_to_mean_person_diagonal == pytest.approx(
        small.center_distance_relative_to_mean_person_diagonal
    )


def _projection(pair, tracking_id: str, side: BodySide):
    for item in pair.wrist_axis_projections:
        if item.wrist_owner_tracking_id == tracking_id and item.side is side:
            return item
    raise AssertionError("projection not found")


def test_axis_projection_at_zero_one_and_between() -> None:
    pair = pair_frame(
        [
            person("a", 0.1, {BodySide.LEFT: (0.2, 0.4), BodySide.RIGHT: (0.4, 0.4)}),
            person("b", 0.5, {BodySide.LEFT: (0.6, 0.4)}),
        ]
    ).pairs[0]
    at_a = _projection(pair, "a", BodySide.LEFT)
    between = _projection(pair, "a", BodySide.RIGHT)
    at_b = _projection(pair, "b", BodySide.LEFT)
    assert at_a.available is True and at_a.t == pytest.approx(0.0)
    assert between.t == pytest.approx(0.5)
    assert at_b.t == pytest.approx(1.0)
    assert at_a.perpendicular_distance == pytest.approx(0.0)


def test_axis_projection_outside_zero_one_is_not_clamped() -> None:
    pair = pair_frame(
        [
            person("a", 0.1, {BodySide.LEFT: (0.05, 0.4)}),
            person("b", 0.5, {BodySide.RIGHT: (0.9, 0.4)}),
        ]
    ).pairs[0]
    assert _projection(pair, "a", BodySide.LEFT).t < 0.0
    assert _projection(pair, "b", BodySide.RIGHT).t > 1.0


def test_perpendicular_distance_is_measured() -> None:
    pair = pair_frame(
        [
            person("a", 0.1, {BodySide.LEFT: (0.4, 0.1)}),
            person("b", 0.5),
        ]
    ).pairs[0]
    projection = _projection(pair, "a", BodySide.LEFT)
    assert projection.t == pytest.approx(0.5)
    assert projection.perpendicular_distance == pytest.approx(0.3)


def test_coincident_centers_make_projection_unavailable() -> None:
    result = build_person_pair_frame_from_tracked_pose(
        frame(
            [
                (
                    "a",
                    BBox(0.2, 0.2, 0.2, 0.4),
                    {PoseKeypointName.LEFT_WRIST: (0.25, 0.3)},
                ),
                ("b", BBox(0.2, 0.2, 0.2, 0.4), {}),
            ]
        )
    )
    projection = result.pairs[0].wrist_axis_projections[0]
    assert projection.available is False
    assert projection.t is None
    assert projection.perpendicular_distance is None
    assert projection.reason == "degenerate_axis_coincident_centers"


def test_all_geometry_values_are_finite() -> None:
    result = pair_frame(
        [
            person("a", 0.05, {BodySide.LEFT: (0.1, 0.3), BodySide.RIGHT: (0.2, 0.5)}),
            person("b", 0.35, {BodySide.LEFT: (0.4, 0.3)}),
            person("c", 0.65, {BodySide.RIGHT: (0.8, 0.5)}),
        ]
    )
    for pair in result.pairs:
        for value in (
            pair.center_distance,
            pair.bbox_min_separation,
            pair.bbox_intersection_area,
            pair.bbox_iou,
        ):
            assert math.isfinite(value)
        for wrist_pair in pair.wrist_pairs:
            assert math.isfinite(wrist_pair.distance)
        for projection in pair.wrist_axis_projections:
            if projection.available:
                assert math.isfinite(projection.t)
                assert math.isfinite(projection.perpendicular_distance)


# -------------------------------------------------------- degraded semantics


@pytest.mark.parametrize(
    "status",
    [
        TrackedPoseFrameStatus.POSE_UNAVAILABLE,
        TrackedPoseFrameStatus.ASSOCIATION_UNAVAILABLE,
        TrackedPoseFrameStatus.INCONSISTENT_INPUT,
    ],
)
def test_degraded_input_is_not_a_valid_empty_pair_frame(status) -> None:
    result = build_person_pair_frame_from_tracked_pose(
        TrackedPoseFrameResult(status=status, camera_id="cam-1", reason="degraded")
    )
    assert result.status.value == status.value
    assert result.pairs == ()
    assert result.subject_count == 0
    assert result.reason == "degraded"


def test_valid_frame_with_people_but_no_wrists_is_still_ok() -> None:
    result = pair_frame([person("a", 0.1), person("b", 0.5)])
    assert result.status is PairFrameStatus.OK
    assert result.pair_count == 1
    assert result.pairs[0].wrist_pairs == ()


def test_metadata_is_carried_through_to_the_pair_frame() -> None:
    result = pair_frame([person("a", 0.1), person("b", 0.5)])
    assert result.camera_id == "cam-1"
    assert result.frame_sequence == 3
    assert result.observed_at == OBSERVED_AT
    assert result.source_mode == "live"
    assert result.source_pose_status is PoseStatus.OK
    assert result.source_frame_status is TrackedPoseFrameStatus.OK


def test_public_api_requires_a_frame_scoped_source() -> None:
    with pytest.raises(TypeError):
        build_person_pair_frame(object())  # type: ignore[arg-type]
    subjects = build_body_feature_frame(
        frame([("a", BBox(0.1, 0.2, 0.2, 0.4), {})])
    ).subjects
    with pytest.raises(TypeError):
        build_person_pair_frame(subjects)  # type: ignore[arg-type]


def test_pair_frame_rejects_wrong_pair_count_for_subject_count() -> None:
    pairs = pair_frame([person("a", 0.1), person("b", 0.5)]).pairs
    with pytest.raises(PairGeometryContractError):
        PersonPairFrameResult(status=PairFrameStatus.OK, pairs=pairs, subject_count=3)


def test_pair_frame_rejects_duplicate_pairs() -> None:
    pairs = pair_frame([person("a", 0.1), person("b", 0.5)]).pairs
    with pytest.raises(PairGeometryContractError):
        PersonPairFrameResult(
            status=PairFrameStatus.OK, pairs=pairs + pairs, subject_count=2
        )


def test_results_are_immutable() -> None:
    result = pair_frame([person("a", 0.1), person("b", 0.5)])
    assert isinstance(result.pairs, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.pairs[0].center_distance = 0.0  # type: ignore[misc]
    assert dataclasses.is_dataclass(result)


# ---------------------------------------------------- vocabulary / no-thresholds

_MODULES = (
    Path("app/domain/pair_geometry.py"),
    Path("app/ai/person_pair_geometry_builder.py"),
    Path("app/ai/body_feature_frame_builder.py"),
)

_FORBIDDEN_NAMES = (
    "wrists_close",
    "people_close",
    "interaction_active",
    "interaction_zone",
    "reaching",
    "contact",
    "touching",
    "handoff",
    "handoff_candidate",
    "paper_exchange_candidate",
    "paper_present",
    "document_present",
    "book_as_paper",
    "sheet_detected",
    "exchange",
    "giver",
    "receiver",
    "velocity",
    "dwell",
    "cooldown",
    "previous_frame",
)


def test_no_behavioural_or_threshold_identifiers_exist() -> None:
    for path in _MODULES:
        tree = ast.parse(path.read_text())
        identifiers: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute, ast.arg)):
                identifiers.add(
                    getattr(node, "id", None)
                    or getattr(node, "attr", None)
                    or getattr(node, "arg", "")
                )
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                identifiers.add(node.name)
        for forbidden in _FORBIDDEN_NAMES:
            assert not any(
                forbidden in identifier.lower() for identifier in identifiers if identifier
            ), f"{path} exposes forbidden concept '{forbidden}'"


def test_no_temporal_or_runtime_imports() -> None:
    for path in _MODULES:
        source = path.read_text()
        for banned in (
            "temporal_state",
            "pose_runtime",
            "orchestrator",
            "engine_registry",
            "phone_rule_engine",
            "event_publisher",
        ):
            assert banned not in source


def test_nothing_in_the_live_runtime_imports_task_3c() -> None:
    for path in Path("app").rglob("*.py"):
        if path.name in {
            "pair_geometry.py",
            "person_pair_geometry_builder.py",
            "body_feature_frame_builder.py",
        }:
            continue
        source = path.read_text()
        assert "pair_geometry" not in source
        assert "person_pair_geometry_builder" not in source
        assert "body_feature_frame_builder" not in source
