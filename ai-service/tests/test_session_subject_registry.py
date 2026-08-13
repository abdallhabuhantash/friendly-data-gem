"""Deterministic proof of the anonymous exam-subject identity rules.

Every case below is pure: no clocks, no models, no Supabase. Timestamps are
supplied explicitly so temporal qualification, gap ageing and short-gap
recovery are exercised exactly, not approximately.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.ai.subject_registry import ExamSubjectRegistry
from app.domain.geometry import BBox
from app.domain.observations import PersonObservation
from app.domain.session_subjects import (
    AssociationMethod,
    SubjectEventKind,
    SubjectRegistryConfig,
    SubjectTrackingStatus,
    subject_label,
)

T0 = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def config(**overrides) -> SubjectRegistryConfig:
    base = dict(
        min_frames_to_qualify=3,
        min_seconds_to_qualify=0.3,
        short_gap_seconds=2.0,
        lost_after_seconds=0.5,
        end_after_seconds=4.0,
        reassociation_min_confidence=0.5,
        reassociation_margin=0.15,
        anchor_smoothing=0.3,
        pending_gap_seconds=0.5,
    )
    base.update(overrides)
    return SubjectRegistryConfig(**base)  # type: ignore[arg-type]


def registry(**overrides) -> ExamSubjectRegistry:
    return ExamSubjectRegistry(
        exam_session_id="session-1",
        camera_id="camera-1",
        config=config(**overrides),
    )


def person(tracking_id, box: BBox, confidence: float = 0.9) -> PersonObservation:
    return PersonObservation(
        person_tracking_id=tracking_id,
        person_bbox=box,
        confidence=confidence,
    )


LEFT = BBox(0.10, 0.40, 0.10, 0.30)
RIGHT = BBox(0.70, 0.40, 0.10, 0.30)
LEFT_SHIFTED = BBox(0.12, 0.41, 0.10, 0.30)


def qualify(reg: ExamSubjectRegistry, raw_id: str, box: BBox, start: float = 0.0):
    """Feeds exactly enough frames for one raw track to earn a subject."""
    result = None
    for index in range(3):
        result = reg.update([person(raw_id, box)], observed_at=at(start + index * 0.2))
    return result


# --------------------------------------------------------------- label policy


def test_subject_label_is_zero_padded_and_one_based():
    assert subject_label(1) == "S001"
    assert subject_label(17) == "S017"
    with pytest.raises(ValueError):
        subject_label(0)


def test_config_rejects_incoherent_windows():
    with pytest.raises(ValueError):
        config(lost_after_seconds=3.0, short_gap_seconds=2.0)
    with pytest.raises(ValueError):
        config(end_after_seconds=1.0, short_gap_seconds=2.0)
    with pytest.raises(ValueError):
        config(min_frames_to_qualify=0)
    with pytest.raises(ValueError):
        config(reassociation_margin=1.5)


# ------------------------------------------------------ temporal qualification


def test_flicker_track_never_creates_a_subject():
    reg = registry()
    first = reg.update([person("7", LEFT)], observed_at=at(0.0))
    second = reg.update([person("7", LEFT)], observed_at=at(0.2))
    assert first.subjects == () and second.subjects == ()
    assert reg.active_subject_count == 0


def test_persistent_track_earns_a_subject_with_initial_segment():
    reg = registry()
    result = qualify(reg, "7", LEFT)
    assert [item.label for item in result.subjects] == ["S001"]
    subject = result.subjects[0]
    assert subject.status is SubjectTrackingStatus.STABLE
    assert subject.active_tracking_id == "7"
    assert subject.reassociation_count == 0
    assert len(subject.segments) == 1
    assert subject.segments[0].method is AssociationMethod.INITIAL
    assert subject.segments[0].is_open
    kinds = [event.kind for event in result.events]
    assert kinds == [SubjectEventKind.SUBJECT_CREATED, SubjectEventKind.TRACK_ATTACHED]


def test_qualification_requires_duration_not_only_frames():
    reg = registry(min_seconds_to_qualify=5.0)
    for index in range(10):
        result = reg.update([person("7", LEFT)], observed_at=at(index * 0.1))
    assert result.subjects == ()


def test_numbering_is_sequential_and_never_reused():
    reg = registry()
    qualify(reg, "7", LEFT)
    result = qualify(reg, "9", RIGHT, start=1.0)
    assert [item.label for item in result.subjects] == ["S001", "S002"]
    # Ending S001 must not free its number.
    reg.update([person("9", RIGHT)], observed_at=at(20.0))
    result = qualify(reg, "11", LEFT, start=21.0)
    assert [item.label for item in result.subjects] == ["S001", "S002", "S003"]


def test_untracked_person_is_ignored_completely():
    reg = registry()
    for index in range(5):
        result = reg.update([person(None, LEFT), person("   ", RIGHT)], observed_at=at(index * 0.2))
    assert result.subjects == ()
    assert result.decisions == ()


def test_pending_progress_expires_after_a_gap():
    reg = registry()
    reg.update([person("7", LEFT)], observed_at=at(0.0))
    reg.update([person("7", LEFT)], observed_at=at(0.2))
    reg.update([], observed_at=at(1.5))  # pending gap exceeded, progress dropped
    result = reg.update([person("7", LEFT)], observed_at=at(1.7))
    assert result.subjects == ()


# ------------------------------------------------------------- gap and ageing


def test_short_gap_marks_subject_temporarily_lost_and_detaches_track():
    reg = registry()
    qualify(reg, "7", LEFT)
    result = reg.update([], observed_at=at(1.2))
    subject = result.subjects[0]
    assert subject.status is SubjectTrackingStatus.TEMPORARILY_LOST
    assert subject.active_tracking_id is None
    assert subject.segments[0].ended_at == at(1.2)
    assert SubjectEventKind.TRACK_DETACHED in {event.kind for event in result.events}


def test_expired_recovery_window_ends_the_subject():
    reg = registry()
    qualify(reg, "7", LEFT)
    result = reg.update([], observed_at=at(10.0))
    subject = result.subjects[0]
    assert subject.status is SubjectTrackingStatus.ENDED
    assert subject.ended_at == at(10.0)
    assert reg.active_subject_count == 0


def test_same_raw_track_reappearing_restores_stability_without_reassociation():
    reg = registry()
    qualify(reg, "7", LEFT)
    reg.update([], observed_at=at(1.0))
    result = reg.update([person("7", LEFT)], observed_at=at(1.4))
    subject = result.subjects[0]
    assert subject.status is SubjectTrackingStatus.STABLE
    assert subject.active_tracking_id == "7"
    assert subject.reassociation_count == 1
    assert subject.segments[-1].method is AssociationMethod.SHORT_GAP_REASSOCIATION


# ------------------------------------------------------- short-gap recovery


def test_new_raw_id_in_same_place_recovers_the_lost_subject():
    reg = registry()
    qualify(reg, "7", LEFT)
    reg.update([], observed_at=at(1.0))
    result = reg.update([person("42", LEFT_SHIFTED)], observed_at=at(1.4))
    assert len(result.subjects) == 1
    subject = result.subjects[0]
    assert subject.label == "S001"
    assert subject.active_tracking_id == "42"
    assert subject.reassociation_count == 1
    assert subject.last_association_confidence is not None
    decision = result.decisions[0]
    assert decision.accepted and decision.reason == "recovered"
    assert decision.subject_number == 1


def test_recovery_is_refused_outside_the_short_gap_window():
    reg = registry(end_after_seconds=30.0)
    qualify(reg, "7", LEFT)
    reg.update([], observed_at=at(1.0))
    result = reg.update([person("42", LEFT)], observed_at=at(5.0))
    decision = result.decisions[0]
    assert not decision.accepted and decision.reason == "no_recoverable_subject"
    assert result.subjects[0].active_tracking_id is None


def test_far_away_track_is_refused_and_becomes_its_own_subject():
    reg = registry()
    qualify(reg, "7", LEFT)
    reg.update([], observed_at=at(1.0))
    result = reg.update([person("42", RIGHT)], observed_at=at(1.2))
    assert result.decisions[0].reason == "below_recovery_threshold"
    for index in range(1, 4):
        result = reg.update([person("42", RIGHT)], observed_at=at(1.2 + index * 0.2))
    labels = {item.label: item for item in result.subjects}
    assert set(labels) == {"S001", "S002"}
    assert labels["S002"].active_tracking_id == "42"
    assert labels["S001"].reassociation_count == 0


def test_ambiguous_recovery_marks_uncertain_and_never_guesses():
    reg = registry()
    qualify(reg, "7", BBox(0.40, 0.40, 0.10, 0.30))
    qualify(reg, "8", BBox(0.50, 0.40, 0.10, 0.30), start=1.0)
    reg.update([], observed_at=at(2.4))
    result = reg.update(
        [person("99", BBox(0.45, 0.40, 0.10, 0.30))],
        observed_at=at(2.6),
    )
    decision = result.decisions[0]
    assert not decision.accepted and decision.reason == "ambiguous"
    assert decision.subject_number is None
    assert len(decision.candidates) == 2
    assert {item.status for item in result.subjects} == {SubjectTrackingStatus.UNCERTAIN}
    assert all(item.active_tracking_id is None for item in result.subjects)


def test_uncertain_subject_can_still_recover_when_evidence_becomes_clear():
    reg = registry()
    qualify(reg, "7", BBox(0.40, 0.40, 0.10, 0.30))
    qualify(reg, "8", BBox(0.50, 0.40, 0.10, 0.30), start=1.0)
    reg.update([], observed_at=at(2.4))
    reg.update([person("99", BBox(0.45, 0.40, 0.10, 0.30))], observed_at=at(2.6))
    result = reg.update([person("100", BBox(0.401, 0.401, 0.10, 0.30))], observed_at=at(2.8))
    assert result.decisions[0].accepted
    assert result.decisions[0].subject_number == 1


def test_live_track_is_never_stolen_by_another_subject():
    reg = registry()
    qualify(reg, "7", LEFT)
    reg.update([], observed_at=at(1.0))  # S001 lost
    result = qualify(reg, "7", LEFT, start=1.4)
    # The raw id belongs to S001 again; no second subject appears for it.
    assert len(result.subjects) == 1


def test_duplicate_raw_id_in_one_frame_is_reported_as_conflict():
    reg = registry()
    qualify(reg, "7", LEFT)
    result = reg.update(
        [person("7", LEFT), person("7", RIGHT)],
        observed_at=at(0.6),
    )
    subject = result.subjects[0]
    assert subject.status is SubjectTrackingStatus.CONFLICT
    assert any(
        event.reason == "duplicate_raw_tracking_id_in_frame" for event in result.events
    )


def test_conflicted_subject_is_never_silently_recovered():
    reg = registry()
    qualify(reg, "7", LEFT)
    reg.update([person("7", LEFT), person("7", RIGHT)], observed_at=at(0.6))
    reg.update([], observed_at=at(1.4))
    result = reg.update([person("55", LEFT)], observed_at=at(1.6))
    assert result.decisions[0].reason == "no_recoverable_subject"
    assert result.subjects[0].status is SubjectTrackingStatus.CONFLICT


# ------------------------------------------------------------ anchors & close


def test_anchor_is_a_smoothed_observation_region_not_a_seat():
    reg = registry(anchor_smoothing=0.5)
    result = qualify(reg, "7", LEFT)
    anchor = result.subjects[0].anchor
    assert anchor is not None
    result = reg.update([person("7", RIGHT)], observed_at=at(0.6))
    moved = result.subjects[0].anchor
    assert moved is not None
    assert anchor.x < moved.x < RIGHT.x


def test_out_of_order_frame_never_moves_identity_backwards():
    reg = registry()
    qualify(reg, "7", LEFT)
    result = reg.update([person("7", LEFT)], observed_at=at(-5.0))
    assert result.observed_at == at(0.4)
    assert result.subjects[0].last_seen_at == at(0.4)


def test_close_ends_every_open_subject_and_segment():
    reg = registry()
    qualify(reg, "7", LEFT)
    qualify(reg, "8", RIGHT, start=1.0)
    events = reg.close(ended_at=at(60.0))
    assert {event.kind for event in events} >= {
        SubjectEventKind.TRACK_DETACHED,
        SubjectEventKind.SUBJECT_ENDED,
    }
    for subject in reg.snapshots():
        assert subject.status is SubjectTrackingStatus.ENDED
        assert subject.ended_at == at(60.0)
        assert all(segment.ended_at is not None for segment in subject.segments)
    assert reg.active_subject_count == 0


def test_registry_never_reads_roster_or_identity_data():
    source = (
        __import__("pathlib").Path("app/ai/subject_registry.py").read_text(encoding="utf-8").lower()
    )
    for forbidden in ("university_id", "full_name", "roster", "face", "biometric_match"):
        assert forbidden not in source
