"""Persists anonymous exam-subject state without blocking inference.

The inference thread only *buffers* what changed; the control loop flushes to
Supabase. A failed flush is retried on the next tick and never turns into a
guessed value: nothing is written that the registry did not observe.

Only anonymous facts leave this process: subject number/label, lifecycle, track
association state, timestamps, the current observation region with its motion
estimate, raw track segments and the recovery confidence. No name, no university
ID, no image, no biometrics.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Optional

from ..domain.session_subjects import (
    SubjectEvent,
    SubjectEventKind,
    SubjectFrameResult,
    SubjectLifecycle,
    SubjectSnapshot,
)

logger = logging.getLogger(__name__)


class SubjectStatePublisher:
    """Buffers subject changes per (session, camera) and flushes them in order."""

    def __init__(self, repository, *, heartbeat_seconds: float = 5.0) -> None:  # noqa: ANN001
        self._repository = repository
        self._heartbeat_seconds = float(heartbeat_seconds)
        self._lock = threading.Lock()
        #: (exam_session_id, subject_number) -> database row id
        self._row_ids: dict[tuple[str, int], str] = {}
        #: Pending subject upserts, newest state wins.
        self._subject_writes: dict[tuple[str, int], dict[str, Any]] = {}
        #: Pending track segment operations, strictly ordered.
        self._track_writes: list[dict[str, Any]] = []
        self._last_heartbeat: dict[tuple[str, int], datetime] = {}

    # ------------------------------------------------------------- buffering

    def record(self, result: SubjectFrameResult) -> None:
        """Buffers one frame result. Called from the inference thread."""
        with self._lock:
            for event in result.events:
                self._record_event(result, event)
            for subject in result.subjects:
                self._maybe_heartbeat(result, subject)

    def record_events(
        self,
        *,
        exam_session_id: str,
        camera_id: str,
        subjects: tuple[SubjectSnapshot, ...],
        events: tuple[SubjectEvent, ...],
    ) -> None:
        """Buffers events produced outside a frame (e.g. session end)."""
        with self._lock:
            by_number = {item.subject_number: item for item in subjects}
            for event in events:
                subject = by_number.get(event.subject_number)
                if subject is None:
                    continue
                self._queue_subject(exam_session_id, camera_id, subject)
                self._queue_track(exam_session_id, event, subject)

    def _record_event(self, result: SubjectFrameResult, event: SubjectEvent) -> None:
        subject = next(
            (item for item in result.subjects if item.subject_number == event.subject_number),
            None,
        )
        if subject is None:
            return
        self._queue_subject(result.exam_session_id, result.camera_id, subject)
        self._queue_track(result.exam_session_id, event, subject)

    def _queue_subject(
        self, exam_session_id: str, camera_id: str, subject: SubjectSnapshot
    ) -> None:
        key = (exam_session_id, subject.subject_number)
        self._subject_writes[key] = {
            "exam_session_id": exam_session_id,
            "subject_number": subject.subject_number,
            "camera_id": camera_id,
            "lifecycle_status": subject.lifecycle.value,
            "track_association": subject.association.value,
            "active_raw_tracking_id": subject.active_tracking_id,
            "first_seen_at": subject.first_seen_at,
            "last_seen_at": subject.last_seen_at,
            "ended_at": subject.ended_at,
            "motion": subject.motion,
            "reassociation_count": subject.recovery_count,
            "last_association_confidence": subject.last_association_confidence,
        }
        self._last_heartbeat[key] = subject.last_seen_at

    def _queue_track(
        self, exam_session_id: str, event: SubjectEvent, subject: SubjectSnapshot
    ) -> None:
        if (
            event.kind in (SubjectEventKind.TRACK_BOUND, SubjectEventKind.TRACK_RECOVERED)
            and event.tracking_id
        ):
            self._track_writes.append(
                {
                    "operation": "open",
                    "exam_session_id": exam_session_id,
                    "subject_number": subject.subject_number,
                    "raw_tracking_id": event.tracking_id,
                    "started_at": event.at,
                    "association_method": event.method.value if event.method else "initial",
                    "association_confidence": event.association_confidence,
                    "start_reason": event.kind.value,
                }
            )
        elif event.kind in (SubjectEventKind.TRACK_RELEASED, SubjectEventKind.ENDED):
            if event.tracking_id:
                self._track_writes.append(
                    {
                        "operation": "close",
                        "exam_session_id": exam_session_id,
                        "raw_tracking_id": event.tracking_id,
                        "ended_at": event.at,
                        "end_reason": event.reason,
                    }
                )

    def _maybe_heartbeat(self, result: SubjectFrameResult, subject: SubjectSnapshot) -> None:
        """Keeps `last_seen_at` fresh for observed subjects, throttled."""
        if subject.lifecycle is SubjectLifecycle.ENDED:
            return
        key = (result.exam_session_id, subject.subject_number)
        if key not in self._row_ids and key not in self._subject_writes:
            return
        previous = self._last_heartbeat.get(key)
        if previous is not None and (
            (subject.last_seen_at - previous).total_seconds() < self._heartbeat_seconds
        ):
            return
        self._queue_subject(result.exam_session_id, result.camera_id, subject)

    # ---------------------------------------------------------------- flushing

    def flush(self) -> None:
        """Writes everything buffered. Called from the control thread only."""
        with self._lock:
            subjects = dict(self._subject_writes)
            tracks = list(self._track_writes)
            self._subject_writes.clear()
            self._track_writes.clear()

        unwritten_subjects: dict[tuple[str, int], dict[str, Any]] = {}
        for key, payload in subjects.items():
            try:
                row_id = self._repository.upsert_session_subject(payload)
                if row_id:
                    self._row_ids[key] = row_id
            except Exception as exc:
                logger.warning("Subject state write failed: %s", type(exc).__name__)
                unwritten_subjects[key] = payload

        unwritten_tracks: list[dict[str, Any]] = []
        for entry in tracks:
            try:
                if entry["operation"] == "open":
                    subject_id = self._row_ids.get(
                        (entry["exam_session_id"], entry["subject_number"])
                    )
                    if subject_id is None:
                        unwritten_tracks.append(entry)
                        continue
                    self._repository.open_subject_track(
                        session_subject_id=subject_id,
                        exam_session_id=entry["exam_session_id"],
                        raw_tracking_id=entry["raw_tracking_id"],
                        started_at=entry["started_at"],
                        association_method=entry["association_method"],
                        association_confidence=entry["association_confidence"],
                        start_reason=entry.get("start_reason"),
                    )
                else:
                    self._repository.close_subject_track(
                        exam_session_id=entry["exam_session_id"],
                        raw_tracking_id=entry["raw_tracking_id"],
                        ended_at=entry["ended_at"],
                        end_reason=entry.get("end_reason"),
                    )
            except Exception as exc:
                logger.warning("Subject track write failed: %s", type(exc).__name__)
                unwritten_tracks.append(entry)

        if unwritten_subjects or unwritten_tracks:
            with self._lock:
                for key, payload in unwritten_subjects.items():
                    self._subject_writes.setdefault(key, payload)
                self._track_writes = unwritten_tracks + self._track_writes

    def forget_session(self, exam_session_id: str) -> None:
        with self._lock:
            for key in [key for key in self._row_ids if key[0] == exam_session_id]:
                del self._row_ids[key]
            for key in [key for key in self._last_heartbeat if key[0] == exam_session_id]:
                del self._last_heartbeat[key]

    @property
    def pending_depth(self) -> int:
        with self._lock:
            return len(self._subject_writes) + len(self._track_writes)

    def known_numbers(self, exam_session_id: str) -> set[int]:
        with self._lock:
            return {number for session, number in self._row_ids if session == exam_session_id}

    def bind_existing(self, exam_session_id: str, rows: dict[int, str]) -> None:
        """Adopts subject rows that already exist (service restart, resume)."""
        with self._lock:
            for number, row_id in rows.items():
                self._row_ids[(exam_session_id, number)] = row_id

    def optional_row_id(self, exam_session_id: str, subject_number: int) -> Optional[str]:
        with self._lock:
            return self._row_ids.get((exam_session_id, subject_number))
