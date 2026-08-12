"""Pure builder: ONE body-feature frame -> ONE PersonPairFrameResult.

Frame safety by construction: the public API consumes ONE authoritative
frame-level source and enumerates every unordered pair inside that call, so
cross-frame mixing is impossible through the normal API. The low-level two-person
geometry helper is private on purpose.

Geometry only:

* no proximity / contact / reaching / handoff / exchange boolean,
* no distance constants or thresholds of any kind,
* no directionality (giver/receiver),
* no paper / document / sheet claim (no such detector exists),
* no previous frame, velocity, dwell, cooldown or state machine.

Architectural note for a later phase (NOT implemented here): a future
``document_exchange`` engine MUST require an explicit armed exam-monitoring
state. Nothing in this module arms anything or is imported by the live runtime.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Optional

from app.ai.body_feature_frame_builder import build_body_feature_frame
from app.ai.region_resolver import relative_point
from app.domain.body_features import BodySide, TrackedBodyFeatures, WristFeatures
from app.domain.geometry import BBox, intersection_area, iou
from app.domain.pair_geometry import (
    PairFrameStatus,
    PersonPairFrameResult,
    PersonPairGeometry,
    PersonPairKey,
    TrackedBodyFeatureFrame,
    WristAxisProjection,
    WristPairGeometry,
    WristRelativeToOtherPerson,
)
from app.domain.tracked_pose_observations import TrackedPoseFrameResult

_SIDES: tuple[BodySide, ...] = (BodySide.LEFT, BodySide.RIGHT)


def _ratio(value: float, denominator: float) -> Optional[float]:
    if denominator <= 0.0:
        return None
    result = value / denominator
    return result if math.isfinite(result) else None


def _bbox_min_separation(a: BBox, b: BBox) -> float:
    dx = max(0.0, a.x - b.x2, b.x - a.x2)
    dy = max(0.0, a.y - b.y2, b.y - a.y2)
    return math.hypot(dx, dy)


def _available_wrist(
    subject: TrackedBodyFeatures, side: BodySide
) -> Optional[WristFeatures]:
    arm = subject.arm(side)
    if arm is None:
        return None
    wrist = arm.wrist
    if not wrist.available or wrist.x is None or wrist.y is None:
        return None
    return wrist


def _axis_projection(
    owner_tracking_id: str,
    side: BodySide,
    point: tuple[float, float],
    center_a: tuple[float, float],
    center_b: tuple[float, float],
) -> WristAxisProjection:
    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 0.0:
        return WristAxisProjection(
            wrist_owner_tracking_id=owner_tracking_id,
            side=side,
            available=False,
            reason="degenerate_axis_coincident_centers",
        )
    px = point[0] - center_a[0]
    py = point[1] - center_a[1]
    # t is NOT clamped: outside 0..1 is a legitimate geometric fact.
    t = (px * dx + py * dy) / length_squared
    perpendicular = abs(px * dy - py * dx) / math.sqrt(length_squared)
    if not (math.isfinite(t) and math.isfinite(perpendicular)):
        return WristAxisProjection(
            wrist_owner_tracking_id=owner_tracking_id,
            side=side,
            available=False,
            reason="non_finite_projection",
        )
    return WristAxisProjection(
        wrist_owner_tracking_id=owner_tracking_id,
        side=side,
        available=True,
        t=t,
        perpendicular_distance=perpendicular,
    )


def _pair_geometry(
    person_a: TrackedBodyFeatures, person_b: TrackedBodyFeatures
) -> PersonPairGeometry:
    """PRIVATE low-level helper. Public enumeration is always frame-scoped."""
    key = PersonPairKey.of(person_a.person_tracking_id, person_b.person_tracking_id)
    # Canonical ordering also fixes which member is A and which is B, so the
    # geometry is identical regardless of input order.
    if person_a.person_tracking_id != key.first_tracking_id:
        person_a, person_b = person_b, person_a

    box_a = person_a.person_bbox
    box_b = person_b.person_bbox
    center_a = box_a.center
    center_b = box_b.center
    center_distance = math.hypot(center_b[0] - center_a[0], center_b[1] - center_a[1])
    diagonal_a = box_a.diagonal
    diagonal_b = box_b.diagonal
    mean_diagonal = (diagonal_a + diagonal_b) / 2.0

    wrist_pairs: list[WristPairGeometry] = []
    for side_a in _SIDES:
        wrist_a = _available_wrist(person_a, side_a)
        if wrist_a is None:
            continue
        for side_b in _SIDES:
            wrist_b = _available_wrist(person_b, side_b)
            if wrist_b is None:
                continue
            distance = math.hypot(
                float(wrist_b.x) - float(wrist_a.x),
                float(wrist_b.y) - float(wrist_a.y),
            )
            if not math.isfinite(distance):
                continue
            wrist_pairs.append(
                WristPairGeometry(
                    side_a=side_a,
                    side_b=side_b,
                    a_x=float(wrist_a.x),
                    a_y=float(wrist_a.y),
                    b_x=float(wrist_b.x),
                    b_y=float(wrist_b.y),
                    distance=distance,
                    distance_relative_to_person_a_diagonal=_ratio(distance, diagonal_a),
                    distance_relative_to_person_b_diagonal=_ratio(distance, diagonal_b),
                    distance_relative_to_mean_person_diagonal=_ratio(
                        distance, mean_diagonal
                    ),
                )
            )

    nearest = (
        min(
            wrist_pairs,
            # Deterministic tie-break on anatomical side names, never input order.
            key=lambda item: (item.distance, item.side_a.value, item.side_b.value),
        )
        if wrist_pairs
        else None
    )

    relative_facts: list[WristRelativeToOtherPerson] = []
    projections: list[WristAxisProjection] = []
    for owner, other in ((person_a, person_b), (person_b, person_a)):
        for side in _SIDES:
            wrist = _available_wrist(owner, side)
            if wrist is None:
                continue
            # Authoritative, single geometry policy (validation + MIN_PERSON_EXTENT
            # + unclamped relative coordinates) lives in region_resolver.
            relative = relative_point(
                other.person_bbox, (float(wrist.x), float(wrist.y))
            )
            if relative is not None:
                relative_facts.append(
                    WristRelativeToOtherPerson(
                        wrist_owner_tracking_id=owner.person_tracking_id,
                        other_tracking_id=other.person_tracking_id,
                        side=side,
                        relative_position=relative,
                        inside_other_person_bbox=relative.inside_person,
                    )
                )
            projections.append(
                _axis_projection(
                    owner.person_tracking_id,
                    side,
                    (float(wrist.x), float(wrist.y)),
                    center_a,
                    center_b,
                )
            )

    return PersonPairGeometry(
        key=key,
        person_a=person_a,
        person_b=person_b,
        center_distance=center_distance,
        bbox_min_separation=_bbox_min_separation(box_a, box_b),
        bbox_intersection_area=intersection_area(box_a, box_b),
        bbox_iou=iou(box_a, box_b),
        center_distance_relative_to_person_a_diagonal=_ratio(
            center_distance, diagonal_a
        ),
        center_distance_relative_to_person_b_diagonal=_ratio(
            center_distance, diagonal_b
        ),
        center_distance_relative_to_mean_person_diagonal=_ratio(
            center_distance, mean_diagonal
        ),
        wrist_pairs=tuple(wrist_pairs),
        nearest_available_wrist_pair=nearest,
        wrists_relative_to_other_person=tuple(relative_facts),
        wrist_axis_projections=tuple(projections),
    )


def build_person_pair_frame(
    frame: TrackedBodyFeatureFrame,
) -> PersonPairFrameResult:
    """Enumerates every unordered pair of ONE body-feature frame, exactly once."""
    if not isinstance(frame, TrackedBodyFeatureFrame):
        raise TypeError("frame must be a TrackedBodyFeatureFrame")

    common = {
        "camera_id": frame.camera_id,
        "frame_sequence": frame.frame_sequence,
        "observed_at": frame.observed_at,
        "source_mode": frame.source_mode,
        "source_pose_status": frame.source_pose_status,
        "source_frame_status": frame.source_frame_status,
        "reason": frame.reason,
    }

    if frame.status is not PairFrameStatus.OK:
        return PersonPairFrameResult(status=frame.status, subject_count=0, **common)

    subjects = sorted(frame.subjects, key=lambda item: item.person_tracking_id)
    pairs = tuple(
        _pair_geometry(a, b) for a, b in combinations(subjects, 2)
    )
    return PersonPairFrameResult(
        status=PairFrameStatus.OK,
        pairs=pairs,
        subject_count=len(subjects),
        **common,
    )


def build_person_pair_frame_from_tracked_pose(
    frame: TrackedPoseFrameResult,
) -> PersonPairFrameResult:
    """Convenience: tracked-pose frame -> body-feature frame -> pair frame."""
    return build_person_pair_frame(build_body_feature_frame(frame))
