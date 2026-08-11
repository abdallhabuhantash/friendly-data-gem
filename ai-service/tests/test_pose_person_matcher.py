"""Deterministic tests for pure pose <-> tracked-person association (Task 2D-B)."""

from __future__ import annotations

import math

import pytest

from app.ai.pose_person_matcher import (
    associate_pose_frame,
    build_pair_facts,
    dominates,
    is_valid_person_observation,
)
from app.domain.geometry import BBox
from app.domain.observations import FrameObservations, PersonObservation
from app.domain.pose import (
    COCO_17_KEYPOINTS,
    PoseFrameResult,
    PoseInstance,
    PoseKeypoint,
    PoseStatus,
)
from app.domain.pose_association import (
    PoseAssociationError,
    PoseAssociationFrameStatus,
    PoseAssociationSpec,
    PoseMatch,
    PoseMatchStatus,
)

SPEC = PoseAssociationSpec(
    min_bbox_iou=0.3,
    min_pose_bbox_containment=0.6,
    min_available_keypoints=4,
    min_keypoint_inside_ratio=0.7,
)


def keypoints(points: dict[int, tuple[float, float]]) -> tuple[PoseKeypoint, ...]:
    out = []
    for index, name in enumerate(COCO_17_KEYPOINTS):
        if index in points:
            x, y = points[index]
            out.append(
                PoseKeypoint(name=name, index=index, available=True, x=x, y=y, confidence=0.9)
            )
        else:
            out.append(PoseKeypoint.unavailable(name, index))
    return tuple(out)


def pose_in(box: BBox, count: int = 6, inside: int | None = None) -> PoseInstance:
    """Pose whose keypoints all sit inside `box` unless `inside` is smaller."""
    inside = count if inside is None else inside
    cx, cy = box.center
    points: dict[int, tuple[float, float]] = {}
    for i in range(count):
        if i < inside:
            points[i] = (cx, cy)
        else:
            points[i] = (0.99, 0.99)
    return PoseInstance(bbox=box, keypoints=keypoints(points), confidence=0.8)


def person(box: BBox, track: str | None, confidence: float = 0.9) -> PersonObservation:
    return PersonObservation(person_tracking_id=track, person_bbox=box, confidence=confidence)


def frame(*persons: PersonObservation) -> FrameObservations:
    return FrameObservations(camera_id="cam-1", persons=tuple(persons))


def ok_pose(*instances: PoseInstance) -> PoseFrameResult:
    return PoseFrameResult(status=PoseStatus.OK, instances=tuple(instances))


BOX_A = BBox(0.10, 0.10, 0.20, 0.40)
BOX_B = BBox(0.60, 0.10, 0.20, 0.40)


# ---------------------------------------------------------------- core matrix


def test_one_pose_one_tracked_person_associates():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, "11")),
        spec=SPEC,
    )
    assert result.status is PoseAssociationFrameStatus.OK
    assert [m.status for m in result.matches] == [PoseMatchStatus.ASSOCIATED]
    assert result.matches[0].person_tracking_id == "11"


def test_two_poses_two_tracked_persons():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A), pose_in(BOX_B)),
        observations=frame(person(BOX_A, "11"), person(BOX_B, "24")),
        spec=SPEC,
    )
    mapping = {m.pose_index: m.person_tracking_id for m in result.matches}
    assert mapping == {0: "11", 1: "24"}


def test_reversed_person_order_keeps_mapping():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A), pose_in(BOX_B)),
        observations=frame(person(BOX_B, "24"), person(BOX_A, "11")),
        spec=SPEC,
    )
    mapping = {m.pose_index: m.person_tracking_id for m in result.matches}
    assert mapping == {0: "11", 1: "24"}


def test_reversed_pose_order_keeps_mapping():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_B), pose_in(BOX_A)),
        observations=frame(person(BOX_A, "11"), person(BOX_B, "24")),
        spec=SPEC,
    )
    mapping = {m.pose_index: m.person_tracking_id for m in result.matches}
    assert mapping == {0: "24", 1: "11"}


def test_exact_tie_is_ambiguous():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, "11"), person(BOX_A, "12")),
        spec=SPEC,
    )
    assert result.matches[0].status is PoseMatchStatus.AMBIGUOUS
    assert result.matches[0].person_tracking_id is None


def test_overlapping_candidates_without_dominance_are_ambiguous():
    pose_box = BBox(0.30, 0.10, 0.20, 0.40)
    # Person 11: higher bbox IoU but only partial pose containment.
    # Person 24: full pose containment but lower IoU. Neither dominates.
    left = BBox(0.35, 0.10, 0.20, 0.40)
    right = BBox(0.25, 0.05, 0.35, 0.50)
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(pose_box)),
        observations=frame(person(left, "11"), person(right, "24")),
        spec=SPEC,
    )
    assert result.matches[0].status is PoseMatchStatus.AMBIGUOUS


def test_two_poses_competing_for_one_person_produce_no_duplicate_identity():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A), pose_in(BOX_A)),
        observations=frame(person(BOX_A, "11")),
        spec=SPEC,
    )
    assert all(m.status is PoseMatchStatus.AMBIGUOUS for m in result.matches)
    assert result.associated_matches == ()
    assert result.matches[0].reason == "multiple_pose_instances_compete_for_person"


def test_unique_best_candidate_untracked_is_blocked():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, None), person(BOX_B, "11")),
        spec=SPEC,
    )
    match = result.matches[0]
    assert match.status is PoseMatchStatus.UNTRACKED_BLOCKED
    assert match.person_tracking_id is None


def test_untracked_and_tracked_tie_is_ambiguous():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, None), person(BOX_A, "11")),
        spec=SPEC,
    )
    assert result.matches[0].status is PoseMatchStatus.AMBIGUOUS


def test_tracked_dominating_untracked_associates():
    pose_box = BBox(0.12, 0.12, 0.16, 0.34)
    tracked = BBox(0.10, 0.10, 0.20, 0.40)
    untracked = BBox(0.10, 0.10, 0.60, 0.80)  # weaker containment/IoU
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(pose_box)),
        observations=frame(person(untracked, None), person(tracked, "11")),
        spec=SPEC,
    )
    assert result.matches[0].status is PoseMatchStatus.ASSOCIATED
    assert result.matches[0].person_tracking_id == "11"


def test_two_untracked_people_yield_no_identity():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, None), person(BOX_B, None)),
        spec=SPEC,
    )
    assert result.associated_matches == ()
    assert result.matches[0].status in {
        PoseMatchStatus.UNTRACKED_BLOCKED,
        PoseMatchStatus.AMBIGUOUS,
    }


def test_insufficient_keypoints_blocks_association():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A, count=2)),
        observations=frame(person(BOX_A, "11")),
        spec=SPEC,
    )
    assert result.matches[0].status is PoseMatchStatus.INSUFFICIENT_KEYPOINTS
    assert result.matches[0].candidates == ()


def test_no_persons_gives_valid_unassociated_pose():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(),
        spec=SPEC,
    )
    assert result.status is PoseAssociationFrameStatus.OK
    assert result.matches[0].status is PoseMatchStatus.UNASSOCIATED
    assert result.matches[0].reason == "no person candidates"


def test_zero_poses_is_valid_empty_frame():
    result = associate_pose_frame(
        pose_result=ok_pose(),
        observations=frame(person(BOX_A, "11")),
        spec=SPEC,
    )
    assert result.status is PoseAssociationFrameStatus.OK
    assert result.matches == ()


@pytest.mark.parametrize(
    "status",
    [
        PoseStatus.MODEL_UNAVAILABLE,
        PoseStatus.INFERENCE_FAILED,
        PoseStatus.KEYPOINT_CONFIDENCE_ABSENT,
        PoseStatus.MALFORMED_RESULT,
        PoseStatus.UNSUPPORTED_POSE_SCHEMA,
        PoseStatus.KEYPOINTS_ABSENT,
    ],
)
def test_degraded_pose_result_is_pose_unavailable(status):
    result = associate_pose_frame(
        pose_result=PoseFrameResult.failure(status),
        observations=frame(person(BOX_A, "11")),
        spec=SPEC,
    )
    assert result.status is PoseAssociationFrameStatus.POSE_UNAVAILABLE
    assert result.matches == ()
    assert result.source_pose_status is status


# ------------------------------------------------------- malformed observations


@pytest.mark.parametrize(
    "bad",
    [
        BBox(0.1, 0.1, 0.0, 0.3),
        BBox(0.1, 0.1, 0.3, 0.0),
        BBox(0.9, 0.1, 0.3, 0.3),
        BBox(-0.1, 0.1, 0.3, 0.3),
        BBox(float("nan"), 0.1, 0.3, 0.3),
        BBox(0.1, 0.1, float("inf"), 0.3),
    ],
)
def test_malformed_person_bbox_invalidates_frame(bad):
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, "11"), person(bad, "12")),
        spec=SPEC,
    )
    assert result.status is PoseAssociationFrameStatus.INVALID_PERSON_OBSERVATIONS
    assert result.matches == ()


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.2])
def test_malformed_person_confidence_invalidates_frame(confidence):
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, "11", confidence=confidence)),
        spec=SPEC,
    )
    assert result.status is PoseAssociationFrameStatus.INVALID_PERSON_OBSERVATIONS


def test_duplicate_non_null_tracking_ids_invalidate_frame():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, "11"), person(BOX_B, "11")),
        spec=SPEC,
    )
    assert result.status is PoseAssociationFrameStatus.INVALID_PERSON_OBSERVATIONS
    assert result.reason == "duplicate non-null person_tracking_id"


def test_multiple_untracked_persons_are_valid_observations():
    assert is_valid_person_observation(person(BOX_A, None))
    result = associate_pose_frame(
        pose_result=ok_pose(),
        observations=frame(person(BOX_A, None), person(BOX_B, None)),
        spec=SPEC,
    )
    assert result.status is PoseAssociationFrameStatus.OK


# ------------------------------------------------------------- pair geometry


def test_pair_facts_geometry_and_ratio():
    pose = pose_in(BOX_A, count=6, inside=5)
    facts = build_pair_facts(
        pose_index=0, pose=pose, person_index=0, person=person(BOX_A, "11"), spec=SPEC
    )
    assert facts.available_keypoint_count == 6
    assert facts.keypoints_inside_person_count == 5
    assert facts.keypoint_inside_person_ratio == pytest.approx(5 / 6)
    assert facts.bbox_iou == pytest.approx(1.0)
    assert facts.pose_bbox_containment_in_person == pytest.approx(1.0)
    assert facts.pose_center_inside_person is True


def test_unavailable_keypoints_never_counted():
    pose = pose_in(BOX_A, count=4)
    facts = build_pair_facts(
        pose_index=0, pose=pose, person_index=0, person=person(BOX_A, "11"), spec=SPEC
    )
    assert facts.available_keypoint_count == 4
    assert facts.keypoint_inside_person_ratio == pytest.approx(1.0)


def test_pose_containment_is_pose_inside_person():
    pose_box = BBox(0.10, 0.10, 0.10, 0.20)
    facts = build_pair_facts(
        pose_index=0,
        pose=pose_in(pose_box),
        person_index=0,
        person=person(BOX_A, "11"),
        spec=SPEC,
    )
    assert facts.pose_bbox_containment_in_person == pytest.approx(1.0)
    assert facts.person_bbox_containment_in_pose < 1.0


def test_boundary_keypoint_is_inclusive():
    pose_box = BBox(0.10, 0.10, 0.20, 0.40)
    points = {i: (BOX_A.x, BOX_A.y) for i in range(6)}
    pose = PoseInstance(bbox=pose_box, keypoints=keypoints(points), confidence=0.7)
    facts = build_pair_facts(
        pose_index=0, pose=pose, person_index=0, person=person(BOX_A, "11"), spec=SPEC
    )
    assert facts.keypoints_inside_person_count == 6


def test_zero_available_keypoints_no_division_error():
    pose = PoseInstance(bbox=BOX_A, keypoints=keypoints({}), confidence=0.5)
    facts = build_pair_facts(
        pose_index=0, pose=pose, person_index=0, person=person(BOX_A, "11"), spec=SPEC
    )
    assert facts.available_keypoint_count == 0
    assert facts.keypoint_inside_person_ratio == 0.0
    assert facts.eligible is False


def test_pair_facts_carry_no_behaviour_fields():
    facts = build_pair_facts(
        pose_index=0, pose=pose_in(BOX_A), person_index=0, person=person(BOX_A, "11"), spec=SPEC
    )
    forbidden = {"behavior", "behaviour", "score", "region", "wrist", "head", "concealed"}
    for field in facts.__slots__:
        assert not any(token in field for token in forbidden)


def test_dominance_requires_strict_improvement_beyond_epsilon():
    base = build_pair_facts(
        pose_index=0, pose=pose_in(BOX_A), person_index=0, person=person(BOX_A, "11"), spec=SPEC
    )
    same = build_pair_facts(
        pose_index=0, pose=pose_in(BOX_A), person_index=1, person=person(BOX_A, "12"), spec=SPEC
    )
    assert dominates(base, same) is False
    assert dominates(same, base) is False


# ------------------------------------------------------------------- spec


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_bbox_iou": -0.1},
        {"min_bbox_iou": 1.1},
        {"min_bbox_iou": float("nan")},
        {"min_pose_bbox_containment": 1.5},
        {"min_keypoint_inside_ratio": float("inf")},
        {"min_available_keypoints": 0},
        {"min_available_keypoints": 18},
        {"min_available_keypoints": 3.5},
        {"min_available_keypoints": True},
    ],
)
def test_spec_rejects_invalid_values_without_clamping(kwargs):
    base = {
        "min_bbox_iou": 0.3,
        "min_pose_bbox_containment": 0.6,
        "min_available_keypoints": 4,
        "min_keypoint_inside_ratio": 0.7,
    }
    base.update(kwargs)
    with pytest.raises(PoseAssociationError):
        PoseAssociationSpec(**base)


def test_spec_is_immutable():
    with pytest.raises(Exception):
        SPEC.min_bbox_iou = 0.9  # type: ignore[misc]


def test_spec_has_no_production_defaults():
    with pytest.raises(TypeError):
        PoseAssociationSpec()  # type: ignore[call-arg]


# ------------------------------------------------- immutability & contracts


def test_inputs_are_not_modified():
    pose = pose_in(BOX_A)
    observed = person(BOX_A, "11")
    pose_result = ok_pose(pose)
    observations = frame(observed)
    snapshot = (
        pose.bbox,
        pose.keypoints,
        observed.person_bbox,
        observed.person_tracking_id,
        observations.persons,
        pose_result.instances,
    )
    associate_pose_frame(pose_result=pose_result, observations=observations, spec=SPEC)
    assert snapshot == (
        pose.bbox,
        pose.keypoints,
        observed.person_bbox,
        observed.person_tracking_id,
        observations.persons,
        pose_result.instances,
    )


def test_pose_instance_remains_identity_free():
    pose = pose_in(BOX_A)
    assert not hasattr(pose, "person_tracking_id")
    assert not hasattr(pose, "tracking_id")
    assert "tracking_id" not in PoseInstance.__slots__


def test_only_associated_status_may_carry_tracking_id():
    with pytest.raises(PoseAssociationError):
        PoseMatch(pose_index=0, status=PoseMatchStatus.AMBIGUOUS, person_tracking_id="11")
    with pytest.raises(PoseAssociationError):
        PoseMatch(pose_index=0, status=PoseMatchStatus.ASSOCIATED)


def test_matcher_is_deterministic():
    pose_result = ok_pose(pose_in(BOX_A), pose_in(BOX_B))
    observations = frame(person(BOX_A, "11"), person(BOX_B, "24"))
    first = associate_pose_frame(
        pose_result=pose_result, observations=observations, spec=SPEC
    )
    second = associate_pose_frame(
        pose_result=pose_result, observations=observations, spec=SPEC
    )
    assert first == second


def test_matcher_module_has_no_region_or_behaviour_imports():
    import app.ai.pose_person_matcher as matcher

    source = open(matcher.__file__, encoding="utf-8").read()
    for token in ("region_resolver", "RegionResolver", "PersonRegionSpec", "concealed"):
        assert token not in source


def test_no_runtime_module_imports_the_matcher():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "app"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name in {"pose_person_matcher.py", "pose_association.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "pose_person_matcher" in text or "pose_association" in text:
            offenders.append(path.name)
    assert offenders == []


def test_epsilon_tolerance_is_finite():
    from app.ai.pose_person_matcher import METRIC_EPSILON

    assert math.isfinite(METRIC_EPSILON) and METRIC_EPSILON > 0


# ------------------------------------- hardening: blank tracking identities


@pytest.mark.parametrize("bad", ["", "   ", "\t", "\n", " \t "])
def test_blank_tracking_id_is_invalid_observation(bad):
    assert is_valid_person_observation(person(BOX_A, bad)) is False
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, bad)),
        spec=SPEC,
    )
    assert result.status is PoseAssociationFrameStatus.INVALID_PERSON_OBSERVATIONS
    assert result.matches == ()


def test_blank_tracking_id_rejects_whole_frame_even_with_valid_person():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, "11"), person(BOX_B, "  ")),
        spec=SPEC,
    )
    assert result.status is PoseAssociationFrameStatus.INVALID_PERSON_OBSERVATIONS
    assert result.matches == ()


def test_none_and_normal_tracking_ids_remain_valid():
    assert is_valid_person_observation(person(BOX_A, None)) is True
    assert is_valid_person_observation(person(BOX_A, "11")) is True


def test_padded_tracking_id_is_not_normalised():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, " 11 ")),
        spec=SPEC,
    )
    assert result.matches[0].person_tracking_id == " 11 "


# ------------------------------------------- hardening: facts preservation


def _tracks(match: PoseMatch) -> set[str | None]:
    return {c.person_tracking_id for c in match.candidates}


def test_associated_match_keeps_all_evaluated_candidates():
    # Person 11 encloses the pose; person 24 is a poorer overlapping candidate.
    near = BBox(0.10, 0.10, 0.24, 0.44)
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, "11"), person(near, "24")),
        spec=SPEC,
    )
    match = result.matches[0]
    assert match.status is PoseMatchStatus.ASSOCIATED
    assert match.person_tracking_id == "11"
    assert _tracks(match) == {"11", "24"}
    rejected = [c for c in match.candidates if c.person_tracking_id == "24"][0]
    assert math.isfinite(rejected.bbox_iou)
    assert math.isfinite(rejected.pose_bbox_containment_in_person)
    assert math.isfinite(rejected.keypoint_inside_person_ratio)
    assert isinstance(rejected.eligible, bool)


def test_global_collision_keeps_full_candidate_facts_per_pose():
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A), pose_in(BOX_A)),
        observations=frame(person(BOX_A, "11"), person(BOX_B, "24")),
        spec=SPEC,
    )
    assert len(result.matches) == 2
    for match in result.matches:
        assert match.status is PoseMatchStatus.AMBIGUOUS
        assert match.person_tracking_id is None
        assert _tracks(match) == {"11", "24"}


def test_candidate_preservation_does_not_change_winner():
    near = BBox(0.10, 0.10, 0.24, 0.44)
    result = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, "11"), person(near, "24")),
        spec=SPEC,
    )
    assert result.matches[0].person_tracking_id == "11"
    assert len(result.matches[0].candidates) == 2


def test_reversed_person_order_keeps_same_winning_identity_and_facts():
    near = BBox(0.10, 0.10, 0.24, 0.44)
    forward = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(BOX_A, "11"), person(near, "24")),
        spec=SPEC,
    ).matches[0]
    reverse = associate_pose_frame(
        pose_result=ok_pose(pose_in(BOX_A)),
        observations=frame(person(near, "24"), person(BOX_A, "11")),
        spec=SPEC,
    ).matches[0]
    assert forward.person_tracking_id == reverse.person_tracking_id == "11"
    assert _tracks(forward) == _tracks(reverse) == {"11", "24"}
    by_track = {c.person_tracking_id: c for c in reverse.candidates}
    assert by_track["11"].person_index == 1
    assert by_track["24"].person_index == 0
