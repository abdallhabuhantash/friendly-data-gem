"""Deterministic pure tests for the derived tracked-pose observation layer."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from app.domain.geometry import BBox
from app.domain.models import SourceMode
from app.domain.observations import FrameObservations, PersonObservation
from app.domain.pose import (
    COCO_17_KEYPOINTS,
    PoseFrameResult,
    PoseInstance,
    PoseKeypoint,
    PoseKeypointName,
    PoseStatus,
    coco_17_index,
)
from app.domain.pose_association import (
    PoseAssociationFrameResult,
    PoseAssociationFrameStatus,
    PoseMatch,
    PoseMatchStatus,
)
from app.domain.tracked_pose_observations import (
    TrackedPoseContractError,
    TrackedPoseFrameStatus,
)
from app.ai.tracked_pose_observation_builder import build_tracked_pose_observations


def keypoints(points: dict[PoseKeypointName, tuple[float, float]]) -> tuple[PoseKeypoint, ...]:
    built = []
    for name in COCO_17_KEYPOINTS:
        index = coco_17_index(name)
        if name in points:
            x, y = points[name]
            built.append(
                PoseKeypoint(
                    name=name, index=index, available=True, x=x, y=y, confidence=0.8
                )
            )
        else:
            built.append(PoseKeypoint.unavailable(name, index, confidence=0.1))
    return tuple(built)


def pose_instance(
    box: BBox, points: dict[PoseKeypointName, tuple[float, float]]
) -> PoseInstance:
    return PoseInstance(bbox=box, keypoints=keypoints(points), confidence=0.9)


def observations(*persons: PersonObservation) -> FrameObservations:
    return FrameObservations(
        camera_id="cam-a",
        persons=persons,
        frame_sequence=42,
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_mode=SourceMode.RTSP if hasattr(SourceMode, "RTSP") else list(SourceMode)[0],
    )


def person(track: str | None, box: BBox, confidence: float = 0.7) -> PersonObservation:
    return PersonObservation(person_tracking_id=track, person_bbox=box, confidence=confidence)


def association(*matches: PoseMatch, status=PoseAssociationFrameStatus.OK) -> PoseAssociationFrameResult:
    return PoseAssociationFrameResult(
        status=status, matches=matches, source_pose_status=PoseStatus.OK
    )


PERSON_BOX = BBox(0.2, 0.2, 0.2, 0.4)
POSE_BOX = BBox(0.21, 0.21, 0.18, 0.38)
SIMPLE_POINTS = {
    PoseKeypointName.NOSE: (0.3, 0.25),
    PoseKeypointName.LEFT_WRIST: (0.25, 0.4),
}


def simple_frame():
    pose_result = PoseFrameResult(
        status=PoseStatus.OK, instances=(pose_instance(POSE_BOX, SIMPLE_POINTS),)
    )
    frame = observations(person("24", PERSON_BOX))
    assoc = association(
        PoseMatch(
            pose_index=0,
            status=PoseMatchStatus.ASSOCIATED,
            person_tracking_id="24",
            person_index=0,
        )
    )
    return pose_result, assoc, frame


def build(pose_result, assoc, frame):
    return build_tracked_pose_observations(
        pose_result=pose_result, association_result=assoc, frame_observations=frame
    )


# --- basic --------------------------------------------------------------


def test_single_associated_pose_yields_one_observation():
    result = build(*simple_frame())
    assert result.status is TrackedPoseFrameStatus.OK
    assert result.tracked_count == 1


def test_tracking_id_and_indices_preserved():
    result = build(*simple_frame())
    observation = result.observations[0]
    assert observation.person_tracking_id == "24"
    assert (observation.pose_index, observation.person_index) == (0, 0)
    assert observation.person_confidence == pytest.approx(0.7)
    assert observation.pose_confidence == pytest.approx(0.9)


def test_canonical_seventeen_keypoint_slots_preserved():
    result = build(*simple_frame())
    observation = result.observations[0]
    assert len(observation.keypoints) == 17
    assert tuple(kp.name for kp in observation.keypoints) == COCO_17_KEYPOINTS
    assert observation.available_keypoint_count == 2


def test_semantic_lookup_is_safe():
    result = build(*simple_frame())
    observation = result.observations[0]
    wrist = observation.keypoint(PoseKeypointName.LEFT_WRIST)
    assert wrist is not None and wrist.available
    assert observation.available_keypoint(PoseKeypointName.RIGHT_ANKLE) is None
    assert observation.keypoint("left_wrist") is None  # type: ignore[arg-type]


def test_inputs_remain_unchanged():
    pose_result, assoc, frame = simple_frame()
    snapshot = (pose_result, assoc, frame)
    build(pose_result, assoc, frame)
    assert snapshot == (pose_result, assoc, frame)
    assert frame.persons[0].person_bbox == PERSON_BOX


def test_derived_objects_are_immutable():
    result = build(*simple_frame())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.observations[0].person_tracking_id = "99"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.observations[0].keypoints[0].available = False  # type: ignore[misc]


def test_frame_metadata_preserved():
    result = build(*simple_frame())
    assert result.camera_id == "cam-a"
    assert result.frame_sequence == 42
    assert result.observed_at == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert result.source_pose_status is PoseStatus.OK
    assert result.pose_instance_count == 1


# --- relative geometry --------------------------------------------------


def test_same_anatomical_point_in_different_person_sizes_matches():
    def relative_for(box: BBox) -> tuple[float, float]:
        point = (box.x + box.width * 0.25, box.y + box.height * 0.75)
        pose_result = PoseFrameResult(
            status=PoseStatus.OK,
            instances=(
                pose_instance(
                    BBox(box.x, box.y, box.width, box.height),
                    {PoseKeypointName.LEFT_WRIST: point},
                ),
            ),
        )
        frame = observations(person("7", box))
        assoc = association(
            PoseMatch(
                pose_index=0,
                status=PoseMatchStatus.ASSOCIATED,
                person_tracking_id="7",
                person_index=0,
            )
        )
        wrist = build(pose_result, assoc, frame).observations[0].keypoint(
            PoseKeypointName.LEFT_WRIST
        )
        assert wrist is not None and wrist.relative_position is not None
        return (wrist.relative_position.relative_x, wrist.relative_position.relative_y)

    small = relative_for(BBox(0.1, 0.1, 0.1, 0.2))
    large = relative_for(BBox(0.4, 0.3, 0.3, 0.6))
    assert small[0] == pytest.approx(large[0])
    assert small[1] == pytest.approx(large[1])
    assert small == pytest.approx((0.25, 0.75))


def test_keypoint_inside_person_is_true():
    result = build(*simple_frame())
    nose = result.observations[0].keypoint(PoseKeypointName.NOSE)
    assert nose is not None and nose.inside_person is True


def test_keypoint_outside_person_keeps_unclamped_relative_position():
    outside = {PoseKeypointName.LEFT_WRIST: (0.05, 0.9)}
    pose_result = PoseFrameResult(
        status=PoseStatus.OK, instances=(pose_instance(POSE_BOX, outside),)
    )
    frame = observations(person("24", PERSON_BOX))
    assoc = association(
        PoseMatch(
            pose_index=0,
            status=PoseMatchStatus.ASSOCIATED,
            person_tracking_id="24",
            person_index=0,
        )
    )
    wrist = build(pose_result, assoc, frame).observations[0].keypoint(
        PoseKeypointName.LEFT_WRIST
    )
    assert wrist is not None and wrist.relative_position is not None
    assert wrist.relative_position.relative_x < 0.0
    assert wrist.relative_position.relative_y > 1.0
    assert wrist.inside_person is False


def test_unavailable_keypoint_carries_no_geometry():
    result = build(*simple_frame())
    knee = result.observations[0].keypoint(PoseKeypointName.LEFT_KNEE)
    assert knee is not None
    assert knee.available is False
    assert (knee.x, knee.y) == (None, None)
    assert knee.relative_position is None
    assert knee.inside_person is None


def test_no_unavailable_keypoint_is_treated_as_origin():
    result = build(*simple_frame())
    for keypoint in result.observations[0].keypoints:
        if not keypoint.available:
            assert keypoint.x is None and keypoint.y is None
            assert keypoint.relative_position is None


# --- association safety -------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        PoseMatchStatus.AMBIGUOUS,
        PoseMatchStatus.UNASSOCIATED,
        PoseMatchStatus.UNTRACKED_BLOCKED,
        PoseMatchStatus.INSUFFICIENT_KEYPOINTS,
    ],
)
def test_non_associated_matches_create_no_subject_but_stay_visible(status):
    pose_result = PoseFrameResult(
        status=PoseStatus.OK, instances=(pose_instance(POSE_BOX, SIMPLE_POINTS),)
    )
    frame = observations(person("24", PERSON_BOX))
    assoc = association(PoseMatch(pose_index=0, status=status, reason="diagnostic"))
    result = build(pose_result, assoc, frame)
    assert result.status is TrackedPoseFrameStatus.OK
    assert result.observations == ()
    assert len(result.unresolved) == 1
    assert result.unresolved[0].match_status is status
    assert result.unresolved[0].reason == "diagnostic"


def two_person_frame(reverse: bool = False):
    box_a = BBox(0.1, 0.2, 0.15, 0.4)
    box_b = BBox(0.6, 0.2, 0.15, 0.4)
    pose_a = pose_instance(box_a, {PoseKeypointName.NOSE: (0.15, 0.25)})
    pose_b = pose_instance(box_b, {PoseKeypointName.NOSE: (0.65, 0.25)})
    persons = [person("11", box_a), person("24", box_b)]
    index_a, index_b = 0, 1
    if reverse:
        persons.reverse()
        index_a, index_b = 1, 0
    frame = observations(*persons)
    pose_result = PoseFrameResult(status=PoseStatus.OK, instances=(pose_a, pose_b))
    assoc = association(
        PoseMatch(
            pose_index=0,
            status=PoseMatchStatus.ASSOCIATED,
            person_tracking_id="11",
            person_index=index_a,
        ),
        PoseMatch(
            pose_index=1,
            status=PoseMatchStatus.ASSOCIATED,
            person_tracking_id="24",
            person_index=index_b,
        ),
    )
    return pose_result, assoc, frame, box_a, box_b


def test_two_associated_poses_are_independent():
    pose_result, assoc, frame, box_a, box_b = two_person_frame()
    result = build(pose_result, assoc, frame)
    assert result.status is TrackedPoseFrameStatus.OK
    assert [o.person_tracking_id for o in result.observations] == ["11", "24"]
    assert result.observation_for("11").person_bbox == box_a
    assert result.observation_for("24").person_bbox == box_b


def test_reversed_person_order_still_resolves_correct_person():
    pose_result, assoc, frame, box_a, box_b = two_person_frame(reverse=True)
    result = build(pose_result, assoc, frame)
    assert result.status is TrackedPoseFrameStatus.OK
    assert result.observation_for("11").person_bbox == box_a
    assert result.observation_for("24").person_bbox == box_b
    assert result.observation_for("11").person_index == 1


# --- cross-contract corruption -----------------------------------------


def test_pose_index_out_of_range_is_inconsistent():
    pose_result = PoseFrameResult(
        status=PoseStatus.OK, instances=(pose_instance(POSE_BOX, SIMPLE_POINTS),)
    )
    frame = observations(person("24", PERSON_BOX))
    assoc = association(
        PoseMatch(
            pose_index=5,
            status=PoseMatchStatus.ASSOCIATED,
            person_tracking_id="24",
            person_index=0,
        )
    )
    result = build(pose_result, assoc, frame)
    assert result.status is TrackedPoseFrameStatus.INCONSISTENT_INPUT
    assert result.observations == ()


def test_duplicate_pose_index_is_inconsistent():
    pose_result = PoseFrameResult(
        status=PoseStatus.OK,
        instances=(
            pose_instance(POSE_BOX, SIMPLE_POINTS),
            pose_instance(POSE_BOX, SIMPLE_POINTS),
        ),
    )
    frame = observations(person("24", PERSON_BOX))
    assoc = association(
        PoseMatch(pose_index=0, status=PoseMatchStatus.AMBIGUOUS),
        PoseMatch(pose_index=0, status=PoseMatchStatus.AMBIGUOUS),
    )
    assert build(pose_result, assoc, frame).status is TrackedPoseFrameStatus.INCONSISTENT_INPUT


def test_missing_pose_match_is_inconsistent():
    pose_result = PoseFrameResult(
        status=PoseStatus.OK,
        instances=(
            pose_instance(POSE_BOX, SIMPLE_POINTS),
            pose_instance(POSE_BOX, SIMPLE_POINTS),
        ),
    )
    frame = observations(person("24", PERSON_BOX))
    assoc = association(PoseMatch(pose_index=0, status=PoseMatchStatus.AMBIGUOUS))
    assert build(pose_result, assoc, frame).status is TrackedPoseFrameStatus.INCONSISTENT_INPUT


def test_person_index_out_of_range_is_inconsistent():
    pose_result = PoseFrameResult(
        status=PoseStatus.OK, instances=(pose_instance(POSE_BOX, SIMPLE_POINTS),)
    )
    frame = observations(person("24", PERSON_BOX))
    assoc = association(
        PoseMatch(
            pose_index=0,
            status=PoseMatchStatus.ASSOCIATED,
            person_tracking_id="24",
            person_index=3,
        )
    )
    assert build(pose_result, assoc, frame).status is TrackedPoseFrameStatus.INCONSISTENT_INPUT


def test_tracking_id_mismatch_is_inconsistent():
    pose_result = PoseFrameResult(
        status=PoseStatus.OK, instances=(pose_instance(POSE_BOX, SIMPLE_POINTS),)
    )
    frame = observations(person("99", PERSON_BOX))
    assoc = association(
        PoseMatch(
            pose_index=0,
            status=PoseMatchStatus.ASSOCIATED,
            person_tracking_id="24",
            person_index=0,
        )
    )
    assert build(pose_result, assoc, frame).status is TrackedPoseFrameStatus.INCONSISTENT_INPUT


def test_two_matches_claiming_same_person_is_inconsistent():
    pose_result = PoseFrameResult(
        status=PoseStatus.OK,
        instances=(
            pose_instance(POSE_BOX, SIMPLE_POINTS),
            pose_instance(POSE_BOX, SIMPLE_POINTS),
        ),
    )
    frame = observations(person("24", PERSON_BOX))
    assoc = association(
        PoseMatch(
            pose_index=0,
            status=PoseMatchStatus.ASSOCIATED,
            person_tracking_id="24",
            person_index=0,
        ),
        PoseMatch(
            pose_index=1,
            status=PoseMatchStatus.ASSOCIATED,
            person_tracking_id="24",
            person_index=0,
        ),
    )
    result = build(pose_result, assoc, frame)
    assert result.status is TrackedPoseFrameStatus.INCONSISTENT_INPUT
    assert result.observations == ()


def test_contradicting_source_pose_status_is_inconsistent():
    pose_result = PoseFrameResult(
        status=PoseStatus.OK, instances=(pose_instance(POSE_BOX, SIMPLE_POINTS),)
    )
    frame = observations(person("24", PERSON_BOX))
    assoc = PoseAssociationFrameResult(
        status=PoseAssociationFrameStatus.OK,
        matches=(
            PoseMatch(
                pose_index=0,
                status=PoseMatchStatus.ASSOCIATED,
                person_tracking_id="24",
                person_index=0,
            ),
        ),
        source_pose_status=PoseStatus.MALFORMED_RESULT,
    )
    assert build(pose_result, assoc, frame).status is TrackedPoseFrameStatus.INCONSISTENT_INPUT


# --- degraded sources ---------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        PoseStatus.MODEL_UNAVAILABLE,
        PoseStatus.INFERENCE_FAILED,
        PoseStatus.KEYPOINT_CONFIDENCE_ABSENT,
    ],
)
def test_degraded_pose_status_yields_pose_unavailable(status):
    pose_result = PoseFrameResult(status=status, instances=())
    frame = observations(person("24", PERSON_BOX))
    result = build(pose_result, association(), frame)
    assert result.status is TrackedPoseFrameStatus.POSE_UNAVAILABLE
    assert result.observations == ()
    assert result.source_pose_status is status


def test_degraded_association_yields_association_unavailable():
    pose_result = PoseFrameResult(status=PoseStatus.OK, instances=())
    frame = observations(person("24", PERSON_BOX))
    assoc = PoseAssociationFrameResult(
        status=PoseAssociationFrameStatus.INVALID_PERSON_OBSERVATIONS,
        matches=(),
        source_pose_status=PoseStatus.OK,
        reason="malformed person observation",
    )
    result = build(pose_result, assoc, frame)
    assert result.status is TrackedPoseFrameStatus.ASSOCIATION_UNAVAILABLE
    assert result.observations == ()


def test_zero_pose_and_zero_matches_is_valid_empty_ok_frame():
    pose_result = PoseFrameResult(status=PoseStatus.OK, instances=())
    frame = observations()
    result = build(pose_result, association(), frame)
    assert result.status is TrackedPoseFrameStatus.OK
    assert result.observations == ()
    assert result.unresolved == ()


def test_unresolved_diagnostic_rejects_associated_status():
    with pytest.raises(TrackedPoseContractError):
        from app.domain.tracked_pose_observations import UnresolvedPoseDiagnostic

        UnresolvedPoseDiagnostic(pose_index=0, match_status=PoseMatchStatus.ASSOCIATED)
