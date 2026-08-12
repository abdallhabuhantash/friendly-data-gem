"""Deterministic pure tests for the frame-level body-feature layer (Task 3C)."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from app.ai.body_feature_frame_builder import build_body_feature_frame
from app.domain.body_features import TrackedBodyFeatures
from app.domain.geometry import BBox
from app.domain.pair_geometry import (
    PairFrameStatus,
    PairGeometryContractError,
    TrackedBodyFeatureFrame,
)
from app.domain.pose import COCO_17_KEYPOINTS, PoseKeypointName, PoseStatus, coco_17_index
from app.domain.pose_association import PoseMatchStatus
from app.domain.regions import RelativePoint
from app.domain.tracked_pose_observations import (
    TrackedPoseFrameResult,
    TrackedPoseFrameStatus,
    TrackedPoseKeypoint,
    TrackedPoseObservation,
    UnresolvedPoseDiagnostic,
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
    person_index: int = 0,
    pose_index: int = 0,
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


def ok_frame(observations: tuple[TrackedPoseObservation, ...], **kwargs) -> TrackedPoseFrameResult:
    return TrackedPoseFrameResult(
        status=TrackedPoseFrameStatus.OK,
        observations=observations,
        camera_id="cam-1",
        frame_sequence=7,
        observed_at=OBSERVED_AT,
        source_mode="live",
        source_pose_status=PoseStatus.OK,
        pose_instance_count=len(observations) + len(kwargs.get("unresolved", ())),
        **kwargs,
    )


def test_zero_people_is_a_valid_empty_frame() -> None:
    result = build_body_feature_frame(ok_frame(()))
    assert result.status is PairFrameStatus.OK
    assert result.subjects == ()
    assert result.subject_count == 0


def test_metadata_is_preserved_from_the_source_frame() -> None:
    result = build_body_feature_frame(
        ok_frame((observation("a", BBox(0.1, 0.1, 0.2, 0.4), {}),))
    )
    assert result.camera_id == "cam-1"
    assert result.frame_sequence == 7
    assert result.observed_at == OBSERVED_AT
    assert result.source_mode == "live"
    assert result.source_pose_status is PoseStatus.OK
    assert result.source_frame_status is TrackedPoseFrameStatus.OK


def test_every_observation_becomes_one_body_subject() -> None:
    frame = ok_frame(
        (
            observation("a", BBox(0.1, 0.1, 0.2, 0.4), {PoseKeypointName.LEFT_WRIST: (0.2, 0.3)}),
            observation("b", BBox(0.5, 0.1, 0.2, 0.4), {}, person_index=1, pose_index=1),
        )
    )
    result = build_body_feature_frame(frame)
    assert [s.person_tracking_id for s in result.subjects] == ["a", "b"]
    assert all(isinstance(s, TrackedBodyFeatures) for s in result.subjects)
    assert result.subjects[0].left_arm.wrist.available is True
    assert result.subjects[1].left_arm.wrist.available is False
    assert result.subjects[1].left_arm.wrist.x is None


def test_unresolved_poses_are_counted_not_fabricated() -> None:
    frame = ok_frame(
        (observation("a", BBox(0.1, 0.1, 0.2, 0.4), {}),),
        unresolved=(UnresolvedPoseDiagnostic(
                pose_index=1,
                match_status=PoseMatchStatus.UNASSOCIATED,
                reason="no_candidate",
            ),),
    )
    result = build_body_feature_frame(frame)
    assert result.subject_count == 1
    assert result.unresolved_pose_count == 1


@pytest.mark.parametrize(
    "status",
    [
        TrackedPoseFrameStatus.POSE_UNAVAILABLE,
        TrackedPoseFrameStatus.ASSOCIATION_UNAVAILABLE,
        TrackedPoseFrameStatus.INCONSISTENT_INPUT,
    ],
)
def test_degraded_input_preserves_status_and_fabricates_nothing(status) -> None:
    frame = TrackedPoseFrameResult(
        status=status, camera_id="cam-1", reason="model_missing"
    )
    result = build_body_feature_frame(frame)
    assert result.status.value == status.value
    assert result.subjects == ()
    assert result.source_frame_status is status
    assert result.reason == "model_missing"


def test_builder_rejects_foreign_input_types() -> None:
    with pytest.raises(TypeError):
        build_body_feature_frame(object())  # type: ignore[arg-type]


def test_frame_is_frozen_and_slotted() -> None:
    result = build_body_feature_frame(ok_frame(()))
    assert dataclasses.is_dataclass(result)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = PairFrameStatus.OK  # type: ignore[misc]


def test_frame_rejects_degraded_status_with_subjects() -> None:
    subject = build_body_feature_frame(
        ok_frame((observation("a", BBox(0.1, 0.1, 0.2, 0.4), {}),))
    ).subjects[0]
    with pytest.raises(PairGeometryContractError):
        TrackedBodyFeatureFrame(
            status=PairFrameStatus.POSE_UNAVAILABLE, subjects=(subject,)
        )


def test_frame_rejects_duplicate_tracking_ids() -> None:
    subject = build_body_feature_frame(
        ok_frame((observation("a", BBox(0.1, 0.1, 0.2, 0.4), {}),))
    ).subjects[0]
    with pytest.raises(PairGeometryContractError):
        TrackedBodyFeatureFrame(status=PairFrameStatus.OK, subjects=(subject, subject))


def test_frame_rejects_bad_metadata() -> None:
    with pytest.raises(PairGeometryContractError):
        TrackedBodyFeatureFrame(status=PairFrameStatus.OK, camera_id="   ")
    with pytest.raises(PairGeometryContractError):
        TrackedBodyFeatureFrame(status=PairFrameStatus.OK, frame_sequence=-1)
    with pytest.raises(PairGeometryContractError):
        TrackedBodyFeatureFrame(status=PairFrameStatus.OK, observed_at="now")  # type: ignore[arg-type]
    with pytest.raises(PairGeometryContractError):
        TrackedBodyFeatureFrame(status=PairFrameStatus.OK, source_mode="telepathy")  # type: ignore[arg-type]


def test_subject_lookup_is_exact_and_safe() -> None:
    result = build_body_feature_frame(
        ok_frame((observation("track-9", BBox(0.1, 0.1, 0.2, 0.4), {}),))
    )
    assert result.subject("track-9") is result.subjects[0]
    assert result.subject("track-8") is None
