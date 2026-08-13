"""Armed-session ownership of the anonymous subject registries.

One registry exists per (armed exam session, camera). Nothing runs until an
exam session is explicitly ARMED — handing out papers at the start of an exam
must never be monitored, so arming is an operator action, not a side effect of
configuring a session (see ``docs/exam-session-identity-contract.md`` §11).

Thread model: `observe()` runs on a camera's inference thread and only touches
that camera's registry plus the shared lock; database writes are buffered and
flushed by the control thread through ``SubjectStatePublisher``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

from ..domain.observations import FrameObservations
from ..domain.session_subjects import (
    SubjectFrameResult,
    SubjectRegistryConfig,
    SubjectSnapshot,
)
from ..ai.subject_registry import ExamSubjectRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ArmedSession:
    """An exam session that is armed for monitoring right now."""

    exam_session_id: str
    camera_ids: tuple[str, ...]
    started_at: Optional[datetime] = None


class _SessionState:
    def __init__(self, session: ArmedSession, config: SubjectRegistryConfig) -> None:
        self.session = session
        self.config = config
        self.registries: dict[str, ExamSubjectRegistry] = {}
        self._next_number = 1

    def allocate(self) -> int:
        number = self._next_number
        self._next_number += 1
        return number

    def registry_for(self, camera_id: str) -> ExamSubjectRegistry:
        registry = self.registries.get(camera_id)
        if registry is None:
            registry = ExamSubjectRegistry(
                exam_session_id=self.session.exam_session_id,
                camera_id=camera_id,
                config=self.config,
                number_allocator=self.allocate,
            )
            self.registries[camera_id] = registry
        return registry


class SubjectRuntime:
    """Anonymous subject tracking for every armed exam session."""

    def __init__(self, config: SubjectRegistryConfig, publisher) -> None:  # noqa: ANN001
        self._config = config
        self._publisher = publisher
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionState] = {}
        #: camera_id -> exam_session_id (a camera serves one armed session)
        self._camera_sessions: dict[str, str] = {}

    # ---------------------------------------------------------------- arming

    def arm(self, session: ArmedSession) -> None:
        """Arms one session from a clean state; re-arming resets nothing else."""
        with self._lock:
            if session.exam_session_id in self._sessions:
                return
            state = _SessionState(session, self._config)
            self._sessions[session.exam_session_id] = state
            for camera_id in session.camera_ids:
                self._camera_sessions[camera_id] = session.exam_session_id
        logger.info(
            "Exam session armed for anonymous subject tracking (%d camera(s))",
            len(session.camera_ids),
        )

    def disarm(self, exam_session_id: str, *, ended_at: Optional[datetime] = None) -> None:
        """Closes every subject of the session truthfully, then forgets it."""
        moment = ended_at or datetime.now(timezone.utc)
        with self._lock:
            state = self._sessions.pop(exam_session_id, None)
            for camera_id in [
                camera_id
                for camera_id, session_id in self._camera_sessions.items()
                if session_id == exam_session_id
            ]:
                del self._camera_sessions[camera_id]
        if state is None:
            return
        for camera_id, registry in state.registries.items():
            snapshots_before = registry.snapshots()
            events = registry.close(ended_at=moment)
            if events:
                self._publisher.record_events(
                    exam_session_id=exam_session_id,
                    camera_id=camera_id,
                    subjects=registry.snapshots() or snapshots_before,
                    events=events,
                )

    def sync(self, armed: Iterable[ArmedSession]) -> None:
        """Reconciles with the database: arms new sessions, disarms finished ones."""
        wanted = {item.exam_session_id: item for item in armed}
        with self._lock:
            current = set(self._sessions)
        for session_id in current - set(wanted):
            self.disarm(session_id)
        for session_id, session in wanted.items():
            if session_id not in current:
                self.arm(session)
            else:
                with self._lock:
                    state = self._sessions.get(session_id)
                    if state is not None:
                        for camera_id in session.camera_ids:
                            self._camera_sessions[camera_id] = session_id

    def is_armed(self, exam_session_id: str) -> bool:
        with self._lock:
            return exam_session_id in self._sessions

    @property
    def armed_session_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._sessions))

    # ------------------------------------------------------------- observing

    def observe(self, observations: FrameObservations) -> Optional[SubjectFrameResult]:
        """Applies one analysed frame; returns None when the camera is unarmed."""
        camera_id = observations.camera_id
        with self._lock:
            session_id = self._camera_sessions.get(camera_id)
            state = self._sessions.get(session_id) if session_id else None
            registry = state.registry_for(camera_id) if state else None
        if registry is None:
            return None
        result = registry.update(
            observations.persons,
            observed_at=observations.observed_at or datetime.now(timezone.utc),
        )
        self._publisher.record(result)
        return result

    def reset_camera(self, camera_id: str) -> None:
        """A new stream incarnation must not inherit subject state."""
        with self._lock:
            session_id = self._camera_sessions.get(camera_id)
            state = self._sessions.get(session_id) if session_id else None
            registry = state.registries.pop(camera_id, None) if state else None
        if registry is None or state is None:
            return
        events = registry.close(ended_at=datetime.now(timezone.utc))
        if events:
            self._publisher.record_events(
                exam_session_id=state.session.exam_session_id,
                camera_id=camera_id,
                subjects=registry.snapshots(),
                events=events,
            )

    def snapshots(self, exam_session_id: str) -> tuple[SubjectSnapshot, ...]:
        with self._lock:
            state = self._sessions.get(exam_session_id)
            registries = list(state.registries.values()) if state else []
        collected: list[SubjectSnapshot] = []
        for registry in registries:
            collected.extend(registry.snapshots())
        return tuple(sorted(collected, key=lambda item: item.subject_number))

    def status(self) -> dict:
        """Measured facts only — never a promised capability."""
        with self._lock:
            sessions = {
                session_id: {
                    "cameras": sorted(state.registries),
                    "active_subjects": sum(
                        registry.active_subject_count for registry in state.registries.values()
                    ),
                }
                for session_id, state in self._sessions.items()
            }
        return {"armed_sessions": sessions}
