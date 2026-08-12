"""Task 3C consistency patch: wrist-relative facts use the authoritative policy.

``app.ai.region_resolver.relative_point`` is the single geometry policy for
person-relative coordinates (validation + MIN_PERSON_EXTENT + unclamped output).
"""

from __future__ import annotations

import inspect

import pytest

from app.ai import person_pair_geometry_builder as builder
from app.ai.region_resolver import MIN_PERSON_EXTENT, relative_point
from app.domain.body_features import BodySide
from app.domain.geometry import BBox

from tests.test_person_pair_geometry_builder import pair_frame  # noqa: F401
from app.domain.pose import PoseKeypointName


def person_with_box(
    tid: str, box: BBox, wrists: dict[BodySide, tuple[float, float]] | None = None
):
    points: dict[PoseKeypointName, tuple[float, float]] = {}
    for side, point in (wrists or {}).items():
        points[
            PoseKeypointName.LEFT_WRIST
            if side is BodySide.LEFT
            else PoseKeypointName.RIGHT_WRIST
        ] = point
    return (tid, box, points)


def relative_facts(result):
    return result.pairs[0].wrists_relative_to_other_person


def test_duplicated_local_geometry_policy_is_gone() -> None:
    assert not hasattr(builder, "_relative_to_bbox")
    assert "relative_point" in inspect.getsource(builder)


def test_tiny_other_person_bbox_yields_no_wrist_relative_fact() -> None:
    tiny = BBox(0.5, 0.5, MIN_PERSON_EXTENT / 10.0, MIN_PERSON_EXTENT / 10.0)
    result = pair_frame(
        [
            person_with_box(
                "a", BBox(0.1, 0.2, 0.2, 0.4), {BodySide.LEFT: (0.15, 0.4)}
            ),
            person_with_box("b", tiny),
        ]
    )
    assert relative_point(tiny, (0.15, 0.4)) is None
    owners = {fact.wrist_owner_tracking_id for fact in relative_facts(result)}
    assert "a" not in owners


def test_normal_bbox_relative_result_matches_authoritative_policy() -> None:
    box_b = BBox(0.5, 0.2, 0.2, 0.4)
    result = pair_frame(
        [
            person_with_box(
                "a", BBox(0.1, 0.2, 0.2, 0.4), {BodySide.LEFT: (0.52, 0.3)}
            ),
            person_with_box("b", box_b),
        ]
    )
    fact = next(
        f
        for f in relative_facts(result)
        if f.wrist_owner_tracking_id == "a" and f.side is BodySide.LEFT
    )
    expected = relative_point(box_b, (0.52, 0.3))
    assert expected is not None
    assert fact.relative_position.relative_x == pytest.approx(expected.relative_x)
    assert fact.relative_position.relative_y == pytest.approx(expected.relative_y)
    assert fact.inside_other_person_bbox is expected.inside_person


def test_relative_coordinates_outside_unit_range_remain_unclamped() -> None:
    box_b = BBox(0.5, 0.2, 0.2, 0.4)
    result = pair_frame(
        [
            person_with_box(
                "a", BBox(0.1, 0.2, 0.2, 0.4), {BodySide.LEFT: (0.15, 0.25)}
            ),
            person_with_box("b", box_b),
        ]
    )
    fact = next(
        f
        for f in relative_facts(result)
        if f.wrist_owner_tracking_id == "a" and f.side is BodySide.LEFT
    )
    assert fact.relative_position.relative_x < 0.0
    assert fact.inside_other_person_bbox is False
