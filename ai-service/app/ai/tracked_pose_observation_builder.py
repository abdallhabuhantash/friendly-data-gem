"""Pure derived tracked-pose observation builder (one frame, no behaviour).

Joins an already-validated ``PoseFrameResult``, the same-frame
``FrameObservations`` and the authoritative ``PoseAssociationFrameResult`` into
:class:`TrackedPoseFrameResult`.

Purity contract: no model inference, no runtime/CameraManager access, no
database, no clock, no global state, no temporal history, no region thresholds,
no behavioural features. Source objects are read-only inputs and are never
mutated. Only ``PoseMatchStatus.ASSOCIATED`` matches become subjects; every
other match survives as an immutable unresolved diagnostic so that "no pose
evidence" can never be confused with "pose existed but could not be assigned".
"""

from __future__ import annotations

from ..domain.observations import FrameObservations
from ..domain.pose import COCO_17_KEYPOINTS, PoseFrameResult
from ..domain.pose_association import (
    PoseAssociationFrameResult,
    PoseAssociationFrameStatus,
    PoseMatchStatus,
)
from ..domain.tracked_pose_observations import (
    TrackedPoseFrameResult,
    TrackedPoseFrameStatus,
    TrackedPoseKeypoint,
    TrackedPoseObservation,
    UnresolvedPoseDiagnostic,
)
from .region_resolver import relative_point


def _frame_metadata(frame_observations: FrameObservations) -> dict[str, object]:
    return {
        "camera_id": frame_observations.camera_id,
        "frame_sequence": frame_observations.frame_sequence,
        "observed_at": frame_observations.observed_at,
        "source_mode": frame_observations.source_mode,
    }


def _inconsistent(
    frame_observations: FrameObservations,
    pose_result: PoseFrameResult,
    reason: str,
) -> TrackedPoseFrameResult:
    return TrackedPoseFrameResult(
        status=TrackedPoseFrameStatus.INCONSISTENT_INPUT,
        source_pose_status=pose_result.status,
        pose_instance_count=len(pose_result.instances),
        reason=reason,
        **_frame_metadata(frame_observations),
    )


def build_tracked_pose_observations(
    *,
    pose_result: PoseFrameResult,
    association_result: PoseAssociationFrameResult,
    frame_observations: FrameObservations,
) -> TrackedPoseFrameResult:
    """Pure one-frame derived tracked-pose observation view."""
    if not isinstance(pose_result, PoseFrameResult):
        raise TypeError("pose_result must be a PoseFrameResult")
    if not isinstance(association_result, PoseAssociationFrameResult):
        raise TypeError("association_result must be a PoseAssociationFrameResult")
    if not isinstance(frame_observations, FrameObservations):
        raise TypeError("frame_observations must be a FrameObservations")

    metadata = _frame_metadata(frame_observations)

    if not pose_result.ok:
        return TrackedPoseFrameResult(
            status=TrackedPoseFrameStatus.POSE_UNAVAILABLE,
            source_pose_status=pose_result.status,
            reason="pose inference did not produce usable evidence",
            **metadata,
        )

    if not association_result.ok:
        return TrackedPoseFrameResult(
            status=TrackedPoseFrameStatus.ASSOCIATION_UNAVAILABLE,
            source_pose_status=pose_result.status,
            pose_instance_count=len(pose_result.instances),
            reason=association_result.reason or association_result.status.value,
            **metadata,
        )

    instances = tuple(pose_result.instances)
    persons = tuple(frame_observations.persons)
    matches = tuple(association_result.matches)

    # A. the association must describe the SAME pose frame, with explicit provenance.
    if (
        association_result.source_pose_status is not PoseStatus.OK
        or association_result.source_pose_status is not pose_result.status
    ):
        return _inconsistent(
            frame_observations,
            pose_result,
            "association source pose status is missing or contradicts the supplied pose result",
        )

    # B. every match index must be a real non-negative int BEFORE any indexing.
    for match in matches:
        if not strict_index(match.pose_index):
            return _inconsistent(
                frame_observations, pose_result, "association match pose_index is malformed"
            )
        if match.person_index is not None and not strict_index(match.person_index):
            return _inconsistent(
                frame_observations, pose_result, "association match person_index is malformed"
            )

    # C. matches must cover the pose instances exactly once each.
    pose_indices = [match.pose_index for match in matches]
    if len(set(pose_indices)) != len(pose_indices):
        return _inconsistent(
            frame_observations, pose_result, "duplicate pose_index in association matches"
        )
    if any(index < 0 or index >= len(instances) for index in pose_indices):
        return _inconsistent(
            frame_observations, pose_result, "association match pose_index out of range"
        )
    if set(pose_indices) != set(range(len(instances))):
        return _inconsistent(
            frame_observations, pose_result, "association matches do not cover every pose instance"
        )

    observations: list[TrackedPoseObservation] = []
    unresolved: list[UnresolvedPoseDiagnostic] = []
    claimed_persons: set[int] = set()
    claimed_tracks: set[str] = set()

    for match in matches:
        if match.status is not PoseMatchStatus.ASSOCIATED:
            unresolved.append(UnresolvedPoseDiagnostic.from_match(match))
            continue

        person_index = match.person_index
        # C. person_index must address a real same-frame person observation.
        if person_index is None or person_index < 0 or person_index >= len(persons):
            return _inconsistent(
                frame_observations, pose_result, "associated person_index out of range"
            )
        person = persons[person_index]
        tracking_id = match.person_tracking_id
        # E. tracking identity must be valid under the association contract.
        if not isinstance(tracking_id, str) or not tracking_id.strip():
            return _inconsistent(
                frame_observations, pose_result, "associated match carries a blank tracking id"
            )
        # D. authoritative person must agree on identity.
        if person.person_tracking_id != tracking_id:
            return _inconsistent(
                frame_observations,
                pose_result,
                "pose match tracking id disagrees with the person observation",
            )
        # F. one-to-one.
        if person_index in claimed_persons or tracking_id in claimed_tracks:
            return _inconsistent(
                frame_observations,
                pose_result,
                "two associated matches claim the same person identity",
            )

        pose = instances[match.pose_index]
        person_box = person.person_bbox
        tracked_keypoints: list[TrackedPoseKeypoint] = []
        for expected_name, keypoint in zip(COCO_17_KEYPOINTS, pose.keypoints):
            if not keypoint.available:
                tracked_keypoints.append(
                    TrackedPoseKeypoint(
                        name=expected_name,
                        index=keypoint.index,
                        available=False,
                        confidence=keypoint.confidence,
                    )
                )
                continue
            relative = relative_point(person_box, (float(keypoint.x), float(keypoint.y)))
            if relative is None:
                return _inconsistent(
                    frame_observations,
                    pose_result,
                    "person geometry cannot support person-relative keypoint resolution",
                )
            tracked_keypoints.append(
                TrackedPoseKeypoint(
                    name=expected_name,
                    index=keypoint.index,
                    available=True,
                    confidence=keypoint.confidence,
                    x=float(keypoint.x),
                    y=float(keypoint.y),
                    relative_position=relative,
                    inside_person=relative.inside_person,
                )
            )

        try:
            observation = TrackedPoseObservation(
                person_tracking_id=tracking_id,
                person_index=person_index,
                pose_index=match.pose_index,
                person_bbox=person_box,
                person_confidence=float(person.confidence),
                pose_bbox=pose.bbox,
                keypoints=tuple(tracked_keypoints),
                pose_confidence=pose.confidence,
            )
        except ValueError as error:
            return _inconsistent(
                frame_observations, pose_result, f"derived observation rejected: {error}"
            )

        claimed_persons.add(person_index)
        claimed_tracks.add(tracking_id)
        observations.append(observation)

    return TrackedPoseFrameResult(
        status=TrackedPoseFrameStatus.OK,
        observations=tuple(observations),
        unresolved=tuple(unresolved),
        source_pose_status=pose_result.status,
        pose_instance_count=len(instances),
        **metadata,
    )


__all__ = ["build_tracked_pose_observations"]
