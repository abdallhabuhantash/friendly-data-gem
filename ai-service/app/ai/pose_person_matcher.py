"""Pure single-frame pose <-> tracked-person association.

Stateless and deterministic: same inputs always produce the same output. No
history, no locks, no database, no image work, no model inference, no camera
state. Nothing here inspects behaviour, regions or joint semantics — this module
answers identity correspondence ONLY.

Contract summary
----------------
Eligibility (a pose/person pair may be a candidate) requires ALL of:

1. ``available_keypoint_count >= spec.min_available_keypoints``
2. ``keypoint_inside_person_ratio >= spec.min_keypoint_inside_ratio``
   (ratio over AVAILABLE keypoints only; unavailable keypoints are never
   treated as ``(0, 0)``)
3. ``bbox_iou >= spec.min_bbox_iou``
   OR ``pose_bbox_containment_in_person >= spec.min_pose_bbox_containment``

``pose_center_inside_person`` is exposed as a diagnostic fact only and never
substitutes for the gates above.

Selection is Pareto/dominance based over the core metrics
(``pose_bbox_containment_in_person``, ``keypoint_inside_person_ratio``,
``bbox_iou``). Exactly one non-dominated candidate -> provisional match; more
than one -> AMBIGUOUS; none eligible -> UNASSOCIATED. Array order and identity
strings are never tie-breakers. Detector confidence is never a ranking signal.
"""

from __future__ import annotations

import math

from ..domain.geometry import BBox, contains_point, containment_ratio, iou
from ..domain.observations import FrameObservations, PersonObservation
from ..domain.pose import PoseFrameResult, PoseInstance
from ..domain.pose_association import (
    PoseAssociationFrameResult,
    PoseAssociationFrameStatus,
    PoseAssociationSpec,
    PoseMatch,
    PoseMatchStatus,
    PosePersonPairFacts,
)

#: Floating-point tolerance: differences at or below this are considered equal.
METRIC_EPSILON = 1e-9


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _unit(value: object) -> bool:
    return _finite(value) and 0.0 <= float(value) <= 1.0


def is_valid_person_observation(person: object) -> bool:
    """True when the observation is safe to use as matching geometry."""
    if not isinstance(person, PersonObservation):
        return False
    box = person.person_bbox
    if not isinstance(box, BBox):
        return False
    for value in (box.x, box.y, box.width, box.height):
        if not _finite(value):
            return False
    if box.width <= 0.0 or box.height <= 0.0:
        return False
    if not (_unit(box.x) and _unit(box.y) and _unit(box.x2) and _unit(box.y2)):
        return False
    if not _unit(person.confidence):
        return False
    if person.person_tracking_id is not None and not isinstance(
        person.person_tracking_id, str
    ):
        return False
    return True


def _duplicate_tracking_ids(persons: tuple[PersonObservation, ...]) -> bool:
    seen: set[str] = set()
    for person in persons:
        track = person.person_tracking_id
        if track is None:
            continue
        if track in seen:
            return True
        seen.add(track)
    return False


def build_pair_facts(
    *,
    pose_index: int,
    pose: PoseInstance,
    person_index: int,
    person: PersonObservation,
    spec: PoseAssociationSpec,
) -> PosePersonPairFacts:
    """Raw pairwise geometry for one (pose, person) pair."""
    person_box = person.person_bbox
    pose_box = pose.bbox

    box_iou = iou(pose_box, person_box)
    pose_in_person = containment_ratio(pose_box, person_box)
    person_in_pose = containment_ratio(person_box, pose_box)
    center_inside = contains_point(person_box, pose_box.center)

    available = tuple(kp for kp in pose.keypoints if kp.available)
    available_count = len(available)
    inside_count = sum(
        1
        for kp in available
        if contains_point(person_box, (float(kp.x), float(kp.y)))
    )
    ratio = inside_count / available_count if available_count > 0 else 0.0

    keypoints_ok = available_count >= spec.min_available_keypoints
    ratio_ok = ratio >= spec.min_keypoint_inside_ratio
    boxes_ok = (
        box_iou >= spec.min_bbox_iou
        or pose_in_person >= spec.min_pose_bbox_containment
    )

    return PosePersonPairFacts(
        pose_index=pose_index,
        person_index=person_index,
        person_tracking_id=person.person_tracking_id,
        bbox_iou=box_iou,
        pose_bbox_containment_in_person=pose_in_person,
        person_bbox_containment_in_pose=person_in_pose,
        pose_center_inside_person=center_inside,
        available_keypoint_count=available_count,
        keypoints_inside_person_count=inside_count,
        keypoint_inside_person_ratio=ratio,
        eligible=bool(keypoints_ok and ratio_ok and boxes_ok),
    )


def _core_metrics(facts: PosePersonPairFacts) -> tuple[float, float, float]:
    return (
        facts.pose_bbox_containment_in_person,
        facts.keypoint_inside_person_ratio,
        facts.bbox_iou,
    )


def dominates(a: PosePersonPairFacts, b: PosePersonPairFacts) -> bool:
    """True when `a` is >= `b` on every core metric and strictly better on one."""
    metrics_a = _core_metrics(a)
    metrics_b = _core_metrics(b)
    strictly_better = False
    for value_a, value_b in zip(metrics_a, metrics_b):
        if value_a < value_b - METRIC_EPSILON:
            return False
        if value_a > value_b + METRIC_EPSILON:
            strictly_better = True
    return strictly_better


def _non_dominated(
    candidates: tuple[PosePersonPairFacts, ...],
) -> tuple[PosePersonPairFacts, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if not any(dominates(other, candidate) for other in candidates if other is not candidate)
    )


def associate_pose_frame(
    *,
    pose_result: PoseFrameResult,
    observations: FrameObservations,
    spec: PoseAssociationSpec,
) -> PoseAssociationFrameResult:
    """Pure one-frame pose -> tracked-person association."""
    if not isinstance(spec, PoseAssociationSpec):
        raise TypeError("spec must be a PoseAssociationSpec")

    if not pose_result.ok:
        return PoseAssociationFrameResult(
            status=PoseAssociationFrameStatus.POSE_UNAVAILABLE,
            matches=(),
            source_pose_status=pose_result.status,
            reason="pose result is degraded",
        )

    persons = tuple(observations.persons)
    if any(not is_valid_person_observation(person) for person in persons):
        return PoseAssociationFrameResult(
            status=PoseAssociationFrameStatus.INVALID_PERSON_OBSERVATIONS,
            matches=(),
            source_pose_status=pose_result.status,
            reason="malformed person observation",
        )
    if _duplicate_tracking_ids(persons):
        return PoseAssociationFrameResult(
            status=PoseAssociationFrameStatus.INVALID_PERSON_OBSERVATIONS,
            matches=(),
            source_pose_status=pose_result.status,
            reason="duplicate non-null person_tracking_id",
        )

    provisional: dict[int, PosePersonPairFacts] = {}
    resolved: dict[int, PoseMatch] = {}

    for pose_index, pose in enumerate(pose_result.instances):
        available_count = sum(1 for kp in pose.keypoints if kp.available)
        if available_count < spec.min_available_keypoints:
            resolved[pose_index] = PoseMatch(
                pose_index=pose_index,
                status=PoseMatchStatus.INSUFFICIENT_KEYPOINTS,
                reason="available keypoints below spec minimum",
            )
            continue

        facts = tuple(
            build_pair_facts(
                pose_index=pose_index,
                pose=pose,
                person_index=person_index,
                person=person,
                spec=spec,
            )
            for person_index, person in enumerate(persons)
        )
        eligible = tuple(item for item in facts if item.eligible)
        if not eligible:
            resolved[pose_index] = PoseMatch(
                pose_index=pose_index,
                status=PoseMatchStatus.UNASSOCIATED,
                reason="no person candidates" if not persons else "no eligible person candidate",
                candidates=facts,
            )
            continue

        best = _non_dominated(eligible)
        if len(best) != 1:
            resolved[pose_index] = PoseMatch(
                pose_index=pose_index,
                status=PoseMatchStatus.AMBIGUOUS,
                reason="multiple non-dominated person candidates",
                candidates=facts,
            )
            continue

        winner = best[0]
        if winner.person_tracking_id is None:
            resolved[pose_index] = PoseMatch(
                pose_index=pose_index,
                status=PoseMatchStatus.UNTRACKED_BLOCKED,
                person_index=winner.person_index,
                reason="preferred person candidate has no tracking identity",
                candidates=facts,
            )
            continue

        provisional[pose_index] = winner

    # Global one-to-one conflict resolution: no greedy second choice.
    claims: dict[int, list[int]] = {}
    for pose_index, winner in provisional.items():
        claims.setdefault(winner.person_index, []).append(pose_index)

    for person_index, pose_indices in claims.items():
        conflicted = len(pose_indices) > 1
        for pose_index in pose_indices:
            winner = provisional[pose_index]
            if conflicted:
                resolved[pose_index] = PoseMatch(
                    pose_index=pose_index,
                    status=PoseMatchStatus.AMBIGUOUS,
                    reason="multiple_pose_instances_compete_for_person",
                    candidates=(winner,),
                )
            else:
                resolved[pose_index] = PoseMatch(
                    pose_index=pose_index,
                    status=PoseMatchStatus.ASSOCIATED,
                    person_tracking_id=winner.person_tracking_id,
                    person_index=person_index,
                    candidates=(winner,),
                )

    matches = tuple(resolved[index] for index in sorted(resolved))
    return PoseAssociationFrameResult(
        status=PoseAssociationFrameStatus.OK,
        matches=matches,
        source_pose_status=pose_result.status,
    )
