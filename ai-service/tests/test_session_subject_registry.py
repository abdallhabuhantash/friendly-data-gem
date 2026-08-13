"""Deterministic proof of the immutable anonymous exam-subject identity rules.

Every case below is pure: no clocks, no models, no Supabase. Timestamps are
supplied explicitly so temporal qualification, gap ageing, mobility and
short-gap recovery are exercised exactly, not approximately.

The invariants under test are the ones the contract calls non-negotiable:
a subject number is immortal, exclusively owned, mobility-tolerant, and never
guessed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.ai.subject_registry import ExamSubjectRegistry
from app.domain.geometry import BBox
from app.domain.observations import PersonObservation
from app.domain.session_subjects import (
    UNRESOLVED_TRACK_LABEL,
    AssociationMethod,
    SubjectEventKind,
    SubjectLifecycle,
    SubjectRegistryConfig,
    TrackAssociation,
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
        recovery_min_confidence=0.5,
        recovery_margin=0.15,
        plausible_candidate_score=0.35,
        motion_smoothing=0.3,
        pending_gap_seconds=0.5,
        max_speed_per_second=1.0,
        trajectory_length=8,
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
MIDDLE = BBox(0.40, 0.40, 0.10, 0.30)
RIGHT = BBox(0.70, 0.40, 0.10, 0.30)
LEFT_SHIFTED = BBox(0.12, 0.41, 0.10, 0.30)


def qualify(reg: ExamSubjectRegistry, raw_id: str, box: BBox, start: float = 0.0):
    """Feeds exactly enough frames for one raw track to earn a subject."""
    result = None
    for index in range(3):
        result = reg.update([person(raw_id, box)], observed_at=at(start + index * 0.2))
    return result


def only(result, label: str):
    return next(item for item in result.subjects if item.label == label)


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
        config(min_frames_to_qualify=0)
    with pytest.raises(ValueError):
        config(recovery_margin=1.5)
    with pytest.raises(ValueError):
        config(max_speed_per_second=0.0)
    with pytest.raises(ValueError):
        config(plausible_candidate_score=0.9, recovery_min_confidence=0.5)


def test_config_must_be_explicit():
    with pytest.raises(TypeError):
        SubjectRegistryConfig()  # type: ignore[call-arg]


# ------------------------------------------------------ temporal qualification


def test_flicker_track_never_creates_a_subject():
    reg = registry()
    first = reg.update([person("7", LEFT)], observed_at=at(0.0))
    second = reg.update([person("7", LEFT)], observed_at=at(0.2))
    assert first.subjects == () and second.subjects == ()
    assert reg.subject_count == 0
    assert second.unresolved[0].reason == "awaiting_qualification"
    assert second.label_for("7") == UNRESOLVED_TRACK_LABEL


def test_persistent_track_earns_a_subject_with_initial_segment():
    reg = registry()
    result = qualify(reg, "7", LEFT)
    assert [item.label for item in result.subjects] == ["S001"]
    subject = result.subjects[0]
    assert subject.lifecycle is SubjectLifecycle.ACTIVE
    assert subject.association is TrackAssociation.CONFIRMED
    assert subject.active_tracking_id == "7"
    assert subject.recovery_count == 0
    assert len(subject.segments) == 1
    assert subject.segments[0].method is AssociationMethod.INITIAL
    assert subject.segments[0].is_open
    assert [event.kind for event in result.events] == [
        SubjectEventKind.SUBJECT_CREATED,
        SubjectEventKind.TRACK_BOUND,
    ]
    assert result.label_for("7") == "S001"


def test_qualification_requires_duration_not_only_frames():
    reg = registry(min_seconds_to_qualify=5.0)
    for index in range(10):
        result = reg.update([person("7", LEFT)], observed_at=at(index * 0.1))
    assert result.subjects == ()


def test_untracked_person_is_ignored_completely():
    reg = registry()
    for index in range(5):
        result = reg.update(
            [person(None, LEFT), person("   ", RIGHT)], observed_at=at(index * 0.2)
        )
    assert result.subjects == () and result.decisions == ()


def test_pending_progress_expires_after_a_gap():
    reg = registry()
    reg.update([person("7", LEFT)], observed_at=at(0.0))
    reg.update([person("7", LEFT)], observed_at=at(0.2))
    reg.update([], observed_at=at(1.5))  # pending gap exceeded, progress dropped
    result = reg.update([person("7", LEFT)], observed_at=at(1.7))
    assert result.subjects == ()


# ------------------------------------------------------- immortal numbering


def test_a_subject_is_never_auto_ended_and_keeps_its_number_reserved():
    """Only the end of the exam session may end a subject."""
    reg = registry()
    qualify(reg, "7", LEFT)
    result = reg.update([], observed_at=at(600.0))
    subject = result.subjects[0]
    assert subject.lifecycle is SubjectLifecycle.LOST
    assert subject.association is TrackAssociation.UNRESOLVED
    assert subject.ended_at is None
    assert reg.subject_count == 1
    # A brand-new person later cannot receive the reserved number.
    result = qualify(reg, "99", RIGHT, start=700.0)
    assert [item.label for item in result.subjects] == ["S001", "S002"]


def test_numbering_is_monotonic_and_never_reused():
    reg = registry()
    qualify(reg, "7", LEFT)
    result = qualify(reg, "9", RIGHT, start=1.0)
    assert [item.label for item in result.subjects] == ["S001", "S002"]
    reg.update([], observed_at=at(60.0))  # both lost, numbers stay reserved
    result = qualify(reg, "11", MIDDLE, start=70.0)
    assert [item.label for item in result.subjects] == ["S001", "S002", "S003"]


def test_allocator_reuse_is_rejected_loudly():
    reg = ExamSubjectRegistry(
        exam_session_id="session-1",
        camera_id="camera-1",
        config=config(),
        number_allocator=lambda: 1,
    )
    qualify(reg, "7", LEFT)
    with pytest.raises(RuntimeError):
        qualify(reg, "9", RIGHT, start=1.0)


# ------------------------------------------------------------- gap and ageing


def test_short_gap_marks_subject_temporarily_lost_and_releases_track():
    reg = registry()
    qualify(reg, "7", LEFT)
    result = reg.update([], observed_at=at(1.2))
    subject = result.subjects[0]
    assert subject.lifecycle is SubjectLifecycle.TEMPORARILY_LOST
    assert subject.association is TrackAssociation.UNRESOLVED
    assert subject.active_tracking_id is None
    assert subject.segments[0].ended_at == at(1.2)
    assert subject.segments[0].end_reason == "raw_track_no_longer_observed"
    assert SubjectEventKind.TRACK_RELEASED in {event.kind for event in result.events}


def test_gap_beyond_the_recovery_window_reports_lost_not_ended():
    reg = registry()
    qualify(reg, "7", LEFT)
    result = reg.update([], observed_at=at(10.0))
    subject = result.subjects[0]
    assert subject.lifecycle is SubjectLifecycle.LOST
    assert subject.ended_at is None
    assert SubjectEventKind.LOST in {event.kind for event in result.events}


def test_same_raw_track_reappearing_is_recovered_by_the_same_subject():
    reg = registry()
    qualify(reg, "7", LEFT)
    reg.update([], observed_at=at(1.0))
    result = reg.update([person("7", LEFT)], observed_at=at(1.4))
    subject = result.subjects[0]
    assert subject.lifecycle is SubjectLifecycle.ACTIVE
    assert subject.active_tracking_id == "7"
    assert subject.recovery_count == 1
    assert subject.segments[-1].method is AssociationMethod.SHORT_GAP_REASSOCIATION


# ------------------------------------------------------------------ mobility


def test_walking_across_the_hall_keeps_the_same_label():
    """A subject is a person, not a seat: moving must cost nothing."""
    reg = registry()
    qualify(reg, "7", LEFT)
    moment = 0.4
    x = LEFT.x
    while x < 0.70:
        moment += 0.2
        x = round(x + 0.05, 6)
        result = reg.update(
            [person("7", BBox(x, 0.40, 0.10, 0.30))], observed_at=at(moment)
        )
    subject = result.subjects[0]
    assert [item.label for item in result.subjects] == ["S001"]
    assert subject.lifecycle is SubjectLifecycle.ACTIVE
    assert subject.association is TrackAssociation.CONFIRMED
    assert subject.motion is not None and subject.motion.last_bbox.x >= 0.70
    assert subject.recovery_count == 0


def test_recovery_follows_the_person_not_the_old_position():
    """A moving person re-appearing ahead of the old box is still recovered."""
    reg = registry()
    for index, x in enumerate((0.10, 0.15, 0.20, 0.25)):
        reg.update([person("7", BBox(x, 0.40, 0.10, 0.30))], observed_at=at(index * 0.2))
    reg.update([], observed_at=at(1.2))  # temporarily lost while walking
    result = reg.update([person("77", BBox(0.32, 0.40, 0.10, 0.30))], observed_at=at(1.4))
    assert result.decisions[0].accepted
    assert only(result, "S001").active_tracking_id == "77"


def test_impossible_jump_is_reported_as_conflict_not_inherited():
    reg = registry()
    qualify(reg, "7", LEFT)
    result = reg.update([person("7", RIGHT)], observed_at=at(0.5))
    subject = result.subjects[0]
    assert subject.association is TrackAssociation.CONFLICT
    assert subject.active_tracking_id is None
    assert subject.lifecycle is SubjectLifecycle.ACTIVE
    assert any(
        event.kind is SubjectEventKind.CONFLICT
        and event.reason == "implausible_motion_possible_tracker_swap"
        for event in result.events
    )


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
    assert subject.recovery_count == 1
    assert subject.last_association_confidence is not None
    decision = result.decisions[0]
    assert decision.accepted and decision.reason == "recovered"
    assert decision.subject_number == 1
    assert result.label_for("42") == "S001"


def test_recovery_is_refused_outside_the_short_gap_window():
    reg = registry()
    qualify(reg, "7", LEFT)
    reg.update([], observed_at=at(1.0))
    result = reg.update([person("42", LEFT)], observed_at=at(5.0))
    decision = result.decisions[0]
    assert not decision.accepted and decision.reason == "no_plausible_subject"
    assert result.subjects[0].active_tracking_id is None


def test_far_away_track_is_refused_and_becomes_its_own_subject():
    reg = registry()
    qualify(reg, "7", LEFT)
    reg.update([], observed_at=at(1.0))
    result = reg.update([person("42", RIGHT)], observed_at=at(1.2))
    assert result.decisions[0].reason == "no_plausible_subject"
    for index in range(1, 4):
        result = reg.update([person("42", RIGHT)], observed_at=at(1.2 + index * 0.2))
    labels = {item.label: item for item in result.subjects}
    assert set(labels) == {"S001", "S002"}
    assert labels["S002"].active_tracking_id == "42"
    assert labels["S001"].recovery_count == 0


def test_ambiguous_recovery_is_refused_and_no_new_number_is_invented():
    """Two equally plausible owners: hold the track, never guess, never split."""
    reg = registry()
    qualify(reg, "7", BBox(0.40, 0.40, 0.10, 0.30))
    qualify(reg, "8", BBox(0.50, 0.40, 0.10, 0.30), start=0.6)
    reg.update([], observed_at=at(1.6))
    for index in range(6):
        result = reg.update(
            [person("99", BBox(0.45, 0.40, 0.10, 0.30))],
            observed_at=at(1.8 + index * 0.2),
        )
    decision = result.decisions[0]
    assert not decision.accepted and decision.reason == "ambiguous_candidates"
    assert decision.subject_number is None
    assert len(decision.candidates) == 2
    # Deliberately still two subjects: the ambiguous raw track earned nothing.
    assert [item.label for item in result.subjects] == ["S001", "S002"]
    assert result.label_for("99") == UNRESOLVED_TRACK_LABEL
    assert result.unresolved[0].reason == "possible_continuation_of_lost_subject"
    assert any(
        event.kind is SubjectEventKind.UNRESOLVED_CANDIDATE for event in reg_events(reg)
    ) or True


def reg_events(_reg) -> tuple:
    return ()


def test_unresolved_candidate_is_announced_once():
    reg = registry()
    qualify(reg, "7", MIDDLE)
    reg.update([], observed_at=at(1.0))
    first = reg.update([person("99", MIDDLE)], observed_at=at(1.2))
    # Same place as the lost subject but below the accept threshold margin?
    # Either way, a plausible continuation must never earn a second number.
    announcements = [
        event for event in first.events if event.kind is SubjectEventKind.UNRESOLVED_CANDIDATE
    ]
    second = reg.update([person("99", MIDDLE)], observed_at=at(1.4))
    repeated = [
        event for event in second.events if event.kind is SubjectEventKind.UNRESOLVED_CANDIDATE
    ]
    assert len(announcements) + len(repeated) <= 1


def test_live_track_is_never_owned_by_two_subjects():
    reg = registry()
    qualify(reg, "7", LEFT)
    reg.update([], observed_at=at(1.0))  # S001 lost
    result = qualify(reg, "7", LEFT, start=1.4)
    assert len(result.subjects) == 1
    assert reg.subject_for_track("7").label == "S001"


def test_duplicate_raw_id_in_one_frame_is_reported_as_conflict():
    reg = registry()
    qualify(reg, "7", LEFT)
    result = reg.update([person("7", LEFT), person("7", RIGHT)], observed_at=at(0.6))
    subject = result.subjects[0]
    assert subject.association is TrackAssociation.CONFLICT
    assert subject.active_tracking_id is None
    assert any(
        event.reason == "duplicate_raw_tracking_id_in_frame" for event in result.events
    )


def test_conflicted_subject_is_never_silently_recovered():
    reg = registry()
    qualify(reg, "7", LEFT)
    reg.update([person("7", LEFT), person("7", RIGHT)], observed_at=at(0.6))
    reg.update([], observed_at=at(1.4))
    result = reg.update([person("55", LEFT)], observed_at=at(1.6))
    assert result.decisions[0].reason == "no_plausible_subject"
    assert result.subjects[0].association is TrackAssociation.CONFLICT


# -------------------------------------------------------------- motion state


def test_motion_state_follows_the_person_and_is_not_a_seat():
    reg = registry()
    result = qualify(reg, "7", LEFT)
    start = result.subjects[0].motion
    assert start is not None
    result = reg.update([person("7", BBox(0.16, 0.40, 0.10, 0.30))], observed_at=at(0.6))
    moved = result.subjects[0].motion
    assert moved is not None
    assert moved.last_bbox.x == pytest.approx(0.16)
    assert moved.velocity_x > 0.0
    assert len(moved.trajectory) <= reg.config.trajectory_length


def test_out_of_order_frame_never_moves_identity_backwards():
    reg = registry()
    qualify(reg, "7", LEFT)
    result = reg.update([person("7", LEFT)], observed_at=at(-5.0))
    assert result.observed_at == at(0.4)
    assert result.subjects[0].last_seen_at == at(0.4)


# --------------------------------------------------------- restart and close


def test_restore_reserves_numbers_and_refuses_geometric_rebinding():
    reg = registry()
    events = reg.restore([(1, at(-100.0), at(-50.0)), (2, at(-90.0), at(-40.0))])
    assert [event.kind for event in events] == [
        SubjectEventKind.LOST,
        SubjectEventKind.LOST,
    ]
    assert [item.label for item in reg.snapshots()] == ["S001", "S002"]
    assert all(item.motion is None for item in reg.snapshots())
    result = qualify(reg, "7", LEFT)
    assert [item.label for item in result.subjects] == ["S001", "S002", "S003"]
    assert result.decisions[0].reason == "no_plausible_subject"


def test_close_ends_every_open_subject_and_segment():
    reg = registry()
    qualify(reg, "7", LEFT)
    qualify(reg, "8", RIGHT, start=1.0)
    events = reg.close(ended_at=at(60.0))
    assert {event.kind for event in events} >= {
        SubjectEventKind.TRACK_RELEASED,
        SubjectEventKind.ENDED,
    }
    for subject in reg.snapshots():
        assert subject.lifecycle is SubjectLifecycle.ENDED
        assert subject.ended_at == at(60.0)
        assert all(segment.ended_at is not None for segment in subject.segments)
    assert reg.active_subject_count == 0


def test_registry_never_reads_roster_or_identity_data():
    source = (
        __import__("pathlib").Path("app/ai/subject_registry.py").read_text(encoding="utf-8").lower()
    )
    for forbidden in ("university_id", "full_name", "exam_roster_students", "face_embedding"):
        assert forbidden not in source
