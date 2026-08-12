"""Task 3F final contract tests: explicit same-frame join + fail-closed geometry.

Pure geometry only: no temporal fusion, no ownership, no events, no runtime.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from app.ai.paper_pair_spatial_builder import build_paper_pair_spatial_frame
from app.domain.geometry import BBox
from app.domain.paper_evidence import (
    PaperDetection,
    PaperEvidenceFrame,
    PaperEvidenceStatus,
)
from app.domain.paper_pair_spatial import (
    PaperPairSpatialStatus,
    SameFrameJoin,
)

from tests.test_paper_pair_spatial_geometry import (
    CAMERA_ID,
    FRAME_SEQUENCE,
    JOIN,
    OBSERVED_AT,
    PAPER_FRAME_INDEX,
    pair_frame,
    paper,
    person,
)

PEOPLE = [person("a", 0.1), person("b", 0.6)]
DETECTION = paper(BBox(0.4, 0.3, 0.04, 0.04))


def papers(
    *,
    frame_index: int | None = PAPER_FRAME_INDEX,
    timestamp_seconds: float | None = 1.5,
    detections: tuple[PaperDetection, ...] = (DETECTION,),
) -> PaperEvidenceFrame:
    return PaperEvidenceFrame(
        status=PaperEvidenceStatus.OK,
        detections=detections,
        model_name="yolo-world",
        backend="open_vocab",
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
    )


# ------------------------------------------------ independent frame counters


def test_different_counters_join_successfully() -> None:
    """pair sequence 210 <-> paper index 7 is valid when explicitly declared."""
    result = build_paper_pair_spatial_frame(pair_frame(PEOPLE), papers(), JOIN)
    assert FRAME_SEQUENCE == 210
    assert PAPER_FRAME_INDEX == 7
    assert result.status is PaperPairSpatialStatus.OK
    assert result.pair_frame_sequence == 210
    assert result.paper_frame_index == 7
    assert len(result.facts) == 1


def test_wrong_declared_pair_frame_sequence_rejected() -> None:
    result = build_paper_pair_spatial_frame(
        pair_frame(PEOPLE),
        papers(),
        dataclasses.replace(JOIN, pair_frame_sequence=211),
    )
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.reason == "pair_frame_sequence_mismatch"
    assert result.facts == ()


def test_wrong_declared_paper_frame_index_rejected() -> None:
    result = build_paper_pair_spatial_frame(
        pair_frame(PEOPLE),
        papers(),
        dataclasses.replace(JOIN, paper_frame_index=8),
    )
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.reason == "paper_frame_index_mismatch"
    assert result.facts == ()


def test_equal_counters_alone_do_not_create_provenance() -> None:
    """Matching numbers are NOT evidence: the declared join is authoritative."""
    equal = papers(frame_index=FRAME_SEQUENCE)
    result = build_paper_pair_spatial_frame(pair_frame(PEOPLE), equal, JOIN)
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.reason == "paper_frame_index_mismatch"

    # And a join that declares the coincidence still validates each side
    # independently rather than inferring correspondence from equality.
    declared = dataclasses.replace(JOIN, paper_frame_index=FRAME_SEQUENCE)
    assert (
        build_paper_pair_spatial_frame(pair_frame(PEOPLE), equal, declared).status
        is PaperPairSpatialStatus.OK
    )


def test_join_is_mandatory() -> None:
    with pytest.raises(TypeError):
        build_paper_pair_spatial_frame(pair_frame(PEOPLE), papers(), None)  # type: ignore[arg-type]


# ------------------------------------------------------- timestamp domains


def test_pair_observed_at_validated_only_against_pair_side() -> None:
    result = build_paper_pair_spatial_frame(
        pair_frame(PEOPLE),
        papers(),
        dataclasses.replace(
            JOIN, pair_observed_at=datetime(2030, 5, 5, tzinfo=timezone.utc)
        ),
    )
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.reason == "pair_observed_at_mismatch"


def test_paper_relative_seconds_validated_only_against_paper_side() -> None:
    result = build_paper_pair_spatial_frame(
        pair_frame(PEOPLE), papers(timestamp_seconds=99.5), JOIN
    )
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.reason == "paper_timestamp_disagreement"


def test_absolute_datetime_never_compared_to_relative_seconds() -> None:
    """An absolute pair datetime and media-relative paper seconds coexist."""
    join = SameFrameJoin(
        camera_id=CAMERA_ID,
        pair_frame_sequence=FRAME_SEQUENCE,
        paper_frame_index=PAPER_FRAME_INDEX,
        paper_timestamp_seconds=1.5,
        pair_observed_at=OBSERVED_AT,
    )
    assert OBSERVED_AT.timestamp() != pytest.approx(1.5)
    result = build_paper_pair_spatial_frame(pair_frame(PEOPLE), papers(), join)
    assert result.status is PaperPairSpatialStatus.OK
    assert result.pair_observed_at == OBSERVED_AT
    assert result.paper_timestamp_seconds == pytest.approx(1.5)


def test_pair_side_clock_may_be_left_undeclared() -> None:
    join = SameFrameJoin(
        camera_id=CAMERA_ID,
        pair_frame_sequence=FRAME_SEQUENCE,
        paper_frame_index=PAPER_FRAME_INDEX,
        paper_timestamp_seconds=1.5,
    )
    result = build_paper_pair_spatial_frame(pair_frame(PEOPLE), papers(), join)
    assert result.status is PaperPairSpatialStatus.OK


# --------------------------------------------- unusable person geometry


def test_unusable_person_a_geometry_fails_whole_frame() -> None:
    people = [person("a", 0.1, box=BBox(0.1, 0.2, 5e-5, 0.4)), person("b", 0.6)]
    result = build_paper_pair_spatial_frame(pair_frame(people), papers(), JOIN)
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.reason == "person_geometry_unusable"
    assert result.facts == ()


def test_unusable_person_b_geometry_fails_whole_frame() -> None:
    people = [person("a", 0.1), person("b", 0.6, box=BBox(0.6, 0.2, 5e-5, 0.4))]
    result = build_paper_pair_spatial_frame(pair_frame(people), papers(), JOIN)
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.reason == "person_geometry_unusable"
    assert result.facts == ()


def test_no_partial_facts_when_a_later_pair_is_unusable() -> None:
    people = [
        person("a", 0.1),
        person("b", 0.4),
        person("c", 0.7, box=BBox(0.7, 0.2, 5e-5, 0.4)),
    ]
    result = build_paper_pair_spatial_frame(pair_frame(people), papers(), JOIN)
    assert result.status is PaperPairSpatialStatus.INCONSISTENT_INPUT
    assert result.facts == ()


def test_valid_geometry_still_produces_unclamped_facts() -> None:
    outside = paper(BBox(0.9, 0.02, 0.03, 0.03))
    result = build_paper_pair_spatial_frame(
        pair_frame(PEOPLE), papers(detections=(outside,)), JOIN
    )
    assert result.status is PaperPairSpatialStatus.OK
    fact = result.facts[0]
    assert len(fact.person_facts) == 2
    relatives = [
        (item.relative_position.relative_x, item.relative_position.relative_y)
        for item in fact.person_facts
    ]
    # Unclamped: at least one coordinate legitimately falls outside 0..1.
    assert any(x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0 for x, y in relatives)
