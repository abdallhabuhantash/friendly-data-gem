"""Arming lifecycle and persistence buffering for anonymous exam subjects."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.domain.geometry import BBox
from app.domain.observations import FrameObservations, PersonObservation
from app.domain.session_subjects import SubjectRegistryConfig
from app.events.subject_state_publisher import SubjectStatePublisher
from app.runtime.subject_runtime import ArmedSession, SubjectRuntime

T0 = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
BOX = BBox(0.20, 0.40, 0.10, 0.30)


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


CONFIG = SubjectRegistryConfig(
    min_frames_to_qualify=2,
    min_seconds_to_qualify=0.2,
    short_gap_seconds=2.0,
    lost_after_seconds=0.5,
    end_after_seconds=4.0,
    reassociation_min_confidence=0.5,
    reassociation_margin=0.15,
    anchor_smoothing=0.3,
    pending_gap_seconds=0.5,
)


class FakeRepository:
    """Records exactly what the publisher would write to the database."""

    def __init__(self, *, fail_first: bool = False) -> None:
        self.subjects: list[dict] = []
        self.opened: list[dict] = []
        self.closed: list[dict] = []
        self._fail_first = fail_first

    def upsert_session_subject(self, payload: dict) -> str:
        if self._fail_first:
            self._fail_first = False
            raise RuntimeError("network down")
        self.subjects.append(payload)
        return f"row-{payload['exam_session_id']}-{payload['subject_number']}"

    def open_subject_track(self, **kwargs) -> None:
        self.opened.append(kwargs)

    def close_subject_track(self, **kwargs) -> None:
        self.closed.append(kwargs)


def frame(camera_id: str, tracking_id, box: BBox, moment: datetime) -> FrameObservations:
    return FrameObservations(
        camera_id=camera_id,
        persons=(PersonObservation(tracking_id, box, 0.9),),
        observed_at=moment,
    )


def build(repository=None):
    repo = repository or FakeRepository()
    publisher = SubjectStatePublisher(repo, heartbeat_seconds=5.0)
    return repo, publisher, SubjectRuntime(CONFIG, publisher)


def test_unarmed_camera_produces_no_subject_state():
    repo, publisher, runtime = build()
    for index in range(5):
        assert runtime.observe(frame("cam-1", "7", BOX, at(index * 0.2))) is None
    publisher.flush()
    assert repo.subjects == [] and repo.opened == []


def test_arming_is_required_before_any_monitoring_state_exists():
    repo, publisher, runtime = build()
    runtime.observe(frame("cam-1", "7", BOX, at(0.0)))
    runtime.arm(ArmedSession("session-1", ("cam-1",)))
    assert runtime.is_armed("session-1")
    for index in range(3):
        result = runtime.observe(frame("cam-1", "7", BOX, at(1.0 + index * 0.2)))
    assert result is not None
    assert [item.label for item in result.subjects] == ["S001"]


def test_subject_numbers_are_unique_across_cameras_of_one_session():
    repo, publisher, runtime = build()
    runtime.arm(ArmedSession("session-1", ("cam-1", "cam-2")))
    for index in range(3):
        runtime.observe(frame("cam-1", "7", BOX, at(index * 0.2)))
    for index in range(3):
        runtime.observe(frame("cam-2", "7", BOX, at(index * 0.2)))
    labels = [item.label for item in runtime.snapshots("session-1")]
    assert labels == ["S001", "S002"]


def test_flush_writes_anonymous_facts_only():
    repo, publisher, runtime = build()
    runtime.arm(ArmedSession("session-1", ("cam-1",)))
    for index in range(3):
        runtime.observe(frame("cam-1", "7", BOX, at(index * 0.2)))
    publisher.flush()
    assert len(repo.subjects) == 1
    payload = repo.subjects[0]
    assert payload["subject_number"] == 1
    assert payload["tracking_status"] == "stable"
    assert set(payload) == {
        "exam_session_id",
        "subject_number",
        "camera_id",
        "tracking_status",
        "first_seen_at",
        "last_seen_at",
        "ended_at",
        "anchor",
        "anchor_updated_at",
        "reassociation_count",
        "last_association_confidence",
    }
    assert repo.opened[0]["raw_tracking_id"] == "7"
    assert repo.opened[0]["association_method"] == "initial"


def test_failed_write_is_retried_and_never_lost():
    repo, publisher, runtime = build(FakeRepository(fail_first=True))
    runtime.arm(ArmedSession("session-1", ("cam-1",)))
    for index in range(3):
        runtime.observe(frame("cam-1", "7", BOX, at(index * 0.2)))
    publisher.flush()
    assert repo.subjects == [] and repo.opened == []
    assert publisher.pending_depth > 0
    publisher.flush()
    assert len(repo.subjects) == 1 and len(repo.opened) == 1
    assert publisher.pending_depth == 0


def test_disarm_closes_subjects_and_segments():
    repo, publisher, runtime = build()
    runtime.arm(ArmedSession("session-1", ("cam-1",)))
    for index in range(3):
        runtime.observe(frame("cam-1", "7", BOX, at(index * 0.2)))
    publisher.flush()
    runtime.disarm("session-1", ended_at=at(90.0))
    publisher.flush()
    assert not runtime.is_armed("session-1")
    assert repo.subjects[-1]["tracking_status"] == "ended"
    assert repo.closed[-1]["raw_tracking_id"] == "7"
    assert runtime.observe(frame("cam-1", "7", BOX, at(91.0))) is None


def test_sync_arms_and_disarms_to_match_the_console():
    repo, publisher, runtime = build()
    runtime.sync([ArmedSession("session-1", ("cam-1",))])
    assert runtime.armed_session_ids == ("session-1",)
    runtime.sync([ArmedSession("session-2", ("cam-2",))])
    assert runtime.armed_session_ids == ("session-2",)
    runtime.sync([])
    assert runtime.armed_session_ids == ()


def test_camera_reset_closes_that_cameras_subjects_only():
    repo, publisher, runtime = build()
    runtime.arm(ArmedSession("session-1", ("cam-1", "cam-2")))
    for index in range(3):
        runtime.observe(frame("cam-1", "7", BOX, at(index * 0.2)))
        runtime.observe(frame("cam-2", "9", BOX, at(index * 0.2)))
    runtime.reset_camera("cam-1")
    publisher.flush()
    remaining = {item.label: item.status.value for item in runtime.snapshots("session-1")}
    assert remaining == {"S002": "stable"}
    assert any(row["tracking_status"] == "ended" for row in repo.subjects)


def test_status_reports_measured_facts_only():
    repo, publisher, runtime = build()
    assert runtime.status() == {"armed_sessions": {}}
    runtime.arm(ArmedSession("session-1", ("cam-1",)))
    for index in range(3):
        runtime.observe(frame("cam-1", "7", BOX, at(index * 0.2)))
    status = runtime.status()
    assert status["armed_sessions"]["session-1"]["active_subjects"] == 1
    assert status["armed_sessions"]["session-1"]["cameras"] == ["cam-1"]


def test_runtime_never_touches_roster_tables():
    source = (
        __import__("pathlib")
        .Path("app/runtime/subject_runtime.py")
        .read_text(encoding="utf-8")
        .lower()
    )
    for forbidden in ("exam_roster_students", "university_id", "full_name"):
        assert forbidden not in source


def test_config_must_be_explicit():
    with pytest.raises(TypeError):
        SubjectRegistryConfig()  # type: ignore[call-arg]
