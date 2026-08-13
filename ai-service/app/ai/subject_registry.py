"""Immutable anonymous exam-subject identity registry (Phase 2, corrected).

One registry instance owns the anonymous subjects of exactly one
(exam session, camera) pair and converts unstable raw tracker ids into stable
per-session labels (``S001``, ``S002``, …).

Guarantees implemented here:

* **Immortal numbering** — a subject number is allocated once and never
  renumbered, transferred or reused, whatever the raw tracker does. A subject
  is never auto-ended by a timeout; only the end of the exam session ends it.
* **Exclusive ownership** — a raw tracking id belongs to at most one subject at
  a time, and a subject owns at most one raw track at a time. Ownership changes
  are explicit, logged transitions, never implicit reassignments.
* **Mobility awareness** — walking, standing up and changing desk are normal.
  Motion state is recovery *evidence*, never a seat, place or identity.
* **Fail-closed ambiguity** — impossible motion, duplicate ids and ambiguous
  candidates produce ``CONFLICT``/``UNRESOLVED``, never a guessed identity, and
  a raw track that plausibly continues a lost subject never receives a new
  number.

Deliberately absent: face recognition, appearance/clothing features, seat maps,
roster matching and any link to a real student — forbidden by
``docs/exam-session-identity-contract.md``.

Pure logic: no clocks (the caller supplies ``observed_at``), no I/O, no models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Optional

from ..domain.geometry import BBox
from ..domain.observations import PersonObservation
from ..domain.session_subjects import (
    UNRESOLVED_TRACK_LABEL,
    AssociationMethod,
    MotionState,
    PendingTrack,
    RecoveryCandidate,
    RecoveryDecision,
    SubjectEvent,
    SubjectEventKind,
    SubjectFrameResult,
    SubjectLifecycle,
    SubjectRegistryConfig,
    SubjectSnapshot,
    SubjectState,
    TrackAssociation,
    TrackSegment,
    advance_motion,
    displacement_is_plausible,
    initial_motion,
    recovery_score,
)

#: Lifecycles from which a subject may still reclaim a raw track.
_RECOVERABLE = (SubjectLifecycle.TEMPORARILY_LOST, SubjectLifecycle.LOST)


class ExamSubjectRegistry:
    """Stable anonymous subject bookkeeping for one exam session camera."""

    def __init__(
        self,
        *,
        exam_session_id: str,
        camera_id: str,
        config: SubjectRegistryConfig,
        number_allocator: Optional[Callable[[], int]] = None,
    ) -> None:
        self.exam_session_id = exam_session_id
        self.camera_id = camera_id
        self.config = config
        # Subject numbers are unique per EXAM SESSION, not per camera, so a
        # multi-camera session injects one shared atomic allocator.
        self._allocator = number_allocator
        self._subjects: dict[int, SubjectState] = {}
        self._pending: dict[str, PendingTrack] = {}
        self._next_number = 1
        self._last_frame_at: Optional[datetime] = None

    # ------------------------------------------------------------------ reads

    def snapshots(self) -> tuple[SubjectSnapshot, ...]:
        return tuple(self._subjects[number].snapshot() for number in sorted(self._subjects))

    def subject_for_track(self, raw_tracking_id: str) -> Optional[SubjectSnapshot]:
        state = self._owner_of(raw_tracking_id)
        return None if state is None else state.snapshot()

    def label_for_track(self, raw_tracking_id: str) -> str:
        """Anonymous label for a raw track, or ``UNRESOLVED`` when unowned."""
        state = self._owner_of(raw_tracking_id)
        return UNRESOLVED_TRACK_LABEL if state is None else state.label

    @property
    def subject_count(self) -> int:
        """Every subject ever created in this session — lost ones included."""
        return len(self._subjects)

    @property
    def active_subject_count(self) -> int:
        return sum(
            1
            for state in self._subjects.values()
            if state.lifecycle is SubjectLifecycle.ACTIVE
        )

    # ------------------------------------------------------------------ resume

    def restore(
        self,
        rows: Iterable[tuple[int, datetime, datetime]],
    ) -> tuple[SubjectEvent, ...]:
        """Reloads persisted subjects after a restart.

        Numbers stay reserved and history is preserved. Restored subjects have no
        motion evidence, so they are ``LOST``/``UNRESOLVED`` and can only be
        re-observed as new tracks — never silently re-bound from stale geometry.
        """
        events: list[SubjectEvent] = []
        for number, first_seen_at, last_seen_at in rows:
            if number in self._subjects:
                continue
            state = SubjectState(
                subject_number=int(number),
                first_seen_at=first_seen_at,
                last_seen_at=last_seen_at,
                lifecycle=SubjectLifecycle.LOST,
                association=TrackAssociation.UNRESOLVED,
            )
            self._subjects[state.subject_number] = state
            self._next_number = max(self._next_number, state.subject_number + 1)
            events.append(
                SubjectEvent(
                    kind=SubjectEventKind.LOST,
                    at=last_seen_at,
                    subject_number=state.subject_number,
                    label=state.label,
                    lifecycle=SubjectLifecycle.LOST,
                    association=TrackAssociation.UNRESOLVED,
                    method=AssociationMethod.RESTORED_AFTER_RESTART,
                    reason="restored_after_restart",
                )
            )
        return tuple(events)

    # ----------------------------------------------------------------- update

    def update(
        self,
        observations: Iterable[PersonObservation],
        *,
        observed_at: datetime,
    ) -> SubjectFrameResult:
        """Applies one analysed frame and reports exactly what changed."""
        # Frames may arrive slightly out of order across threads; identity must
        # never travel backwards in time.
        if self._last_frame_at is not None and observed_at < self._last_frame_at:
            observed_at = self._last_frame_at
        self._last_frame_at = observed_at

        events: list[SubjectEvent] = []
        decisions: list[RecoveryDecision] = []

        frame, duplicates = self._frame_tracks(observations)

        # A raw id reported twice in one frame is contradictory evidence: the
        # owner is flagged and detached instead of following either box.
        for raw_id in sorted(duplicates):
            owner = self._owner_of(raw_id)
            if owner is None:
                continue
            events.extend(
                self._release(
                    owner,
                    observed_at,
                    association=TrackAssociation.CONFLICT,
                    reason="duplicate_raw_tracking_id_in_frame",
                )
            )

        events.extend(self._advance_attached(frame, observed_at))
        events.extend(self._age_detached(frame, observed_at))

        for raw_id in sorted(frame):
            if self._owner_of(raw_id) is not None:
                continue
            decision = self._decide_recovery(raw_id, frame[raw_id], observed_at)
            decisions.append(decision)
            if decision.accepted and decision.subject_number is not None:
                events.extend(
                    self._bind(
                        self._subjects[decision.subject_number],
                        raw_id,
                        frame[raw_id],
                        observed_at,
                        method=AssociationMethod.SHORT_GAP_REASSOCIATION,
                        confidence=decision.score,
                        kind=SubjectEventKind.TRACK_RECOVERED,
                    )
                )
                self._pending.pop(raw_id, None)
                continue
            events.extend(self._track_pending(raw_id, frame[raw_id], observed_at, decision))

        self._expire_pending(frame, observed_at)

        return SubjectFrameResult(
            exam_session_id=self.exam_session_id,
            camera_id=self.camera_id,
            observed_at=observed_at,
            subjects=self.snapshots(),
            events=tuple(events),
            decisions=tuple(decisions),
            unresolved=self._unresolved_snapshot(),
            labels=self._labels(frame),
        )

    def close(self, *, ended_at: datetime) -> tuple[SubjectEvent, ...]:
        """Ends the exam session: the only legitimate way a subject ends."""
        events: list[SubjectEvent] = []
        for number in sorted(self._subjects):
            state = self._subjects[number]
            if state.lifecycle is SubjectLifecycle.ENDED:
                continue
            events.extend(self._end_subject(state, ended_at, reason="exam_session_ended"))
        self._pending.clear()
        return tuple(events)

    # --------------------------------------------------------------- internals

    def _frame_tracks(
        self, observations: Iterable[PersonObservation]
    ) -> tuple[dict[str, BBox], set[str]]:
        """Tracked persons of this frame; blank ids are never identities."""
        frame: dict[str, BBox] = {}
        duplicates: set[str] = set()
        for observation in observations:
            raw_id = (observation.person_tracking_id or "").strip()
            if not raw_id:
                # An untracked person cannot own a stable subject; dropping it
                # is the only truthful option.
                continue
            if raw_id in frame:
                duplicates.add(raw_id)
                continue
            frame[raw_id] = observation.person_bbox
        for raw_id in duplicates:
            frame.pop(raw_id, None)
        return frame, duplicates

    def _owner_of(self, raw_tracking_id: str) -> Optional[SubjectState]:
        for number in sorted(self._subjects):
            state = self._subjects[number]
            if state.active_tracking_id == raw_tracking_id:
                return state
        return None

    def _labels(self, frame: dict[str, BBox]) -> tuple[tuple[str, str], ...]:
        return tuple(
            (raw_id, self.label_for_track(raw_id)) for raw_id in sorted(frame)
        )

    def _unresolved_snapshot(self):  # noqa: ANN202 - tuple of UnresolvedCandidate
        from ..domain.session_subjects import UnresolvedCandidate

        return tuple(
            UnresolvedCandidate(
                raw_tracking_id=pending.raw_tracking_id,
                first_seen_at=pending.first_seen_at,
                last_seen_at=pending.last_seen_at,
                frames=pending.frames,
                reason=pending.reason,
            )
            for pending in sorted(self._pending.values(), key=lambda item: item.raw_tracking_id)
        )

    # -- attached subjects ---------------------------------------------------

    def _advance_attached(
        self, frame: dict[str, BBox], observed_at: datetime
    ) -> list[SubjectEvent]:
        events: list[SubjectEvent] = []
        for number in sorted(self._subjects):
            state = self._subjects[number]
            raw_id = state.active_tracking_id
            if raw_id is None or raw_id not in frame:
                continue
            observed = frame[raw_id]
            motion = state.motion
            if motion is not None and not displacement_is_plausible(
                motion,
                observed,
                observed_at,
                max_speed_per_second=self.config.max_speed_per_second,
            ):
                # Physically impossible jump: the raw tracker probably swapped
                # two people. Refuse to inherit the swap.
                events.extend(
                    self._release(
                        state,
                        observed_at,
                        association=TrackAssociation.CONFLICT,
                        reason="implausible_motion_possible_tracker_swap",
                    )
                )
                continue
            state.last_seen_at = observed_at
            state.motion = (
                initial_motion(
                    observed, observed_at, trajectory_length=self.config.trajectory_length
                )
                if motion is None
                else advance_motion(
                    motion,
                    observed,
                    observed_at,
                    smoothing=self.config.motion_smoothing,
                    trajectory_length=self.config.trajectory_length,
                )
            )
            events.extend(
                self._set_lifecycle(
                    state, SubjectLifecycle.ACTIVE, observed_at, reason="raw_track_observed"
                )
            )
            events.extend(
                self._set_association(
                    state,
                    TrackAssociation.CONFIRMED,
                    observed_at,
                    reason="raw_track_continues_plausibly",
                )
            )
        return events

    def _age_detached(
        self, frame: dict[str, BBox], observed_at: datetime
    ) -> list[SubjectEvent]:
        """Ages subjects whose raw track is missing. A subject NEVER expires."""
        events: list[SubjectEvent] = []
        for number in sorted(self._subjects):
            state = self._subjects[number]
            if state.lifecycle is SubjectLifecycle.ENDED:
                continue
            raw_id = state.active_tracking_id
            if raw_id is not None and raw_id in frame:
                continue
            gap = (observed_at - state.last_seen_at).total_seconds()
            if gap < self.config.lost_after_seconds:
                continue
            if raw_id is not None:
                events.extend(
                    self._release(
                        state,
                        observed_at,
                        association=TrackAssociation.UNRESOLVED,
                        reason="raw_track_no_longer_observed",
                    )
                )
            target = (
                SubjectLifecycle.LOST
                if gap > self.config.short_gap_seconds
                else SubjectLifecycle.TEMPORARILY_LOST
            )
            events.extend(
                self._set_lifecycle(
                    state,
                    target,
                    observed_at,
                    reason=(
                        "recovery_window_expired_number_stays_reserved"
                        if target is SubjectLifecycle.LOST
                        else "inside_short_gap_recovery_window"
                    ),
                )
            )
        return events

    # -- recovery ------------------------------------------------------------

    def _recoverable(self, observed_at: datetime) -> list[SubjectState]:
        recoverable: list[SubjectState] = []
        for number in sorted(self._subjects):
            state = self._subjects[number]
            if state.active_tracking_id is not None:
                continue
            if state.lifecycle not in _RECOVERABLE:
                continue
            if state.association is TrackAssociation.CONFLICT:
                # Conflicts are never silently repaired by geometry.
                continue
            if state.motion is None:
                # No spatial evidence at all (e.g. restored after restart).
                continue
            if (observed_at - state.last_seen_at).total_seconds() > self.config.short_gap_seconds:
                continue
            recoverable.append(state)
        return recoverable

    def _decide_recovery(
        self, raw_tracking_id: str, bbox: BBox, observed_at: datetime
    ) -> RecoveryDecision:
        """Scores every recoverable subject and refuses anything ambiguous.

        The winner must clear the absolute threshold AND beat the runner-up by
        the configured margin. The competition is symmetric: the winning subject
        must also not be a similarly good match for another unowned raw track,
        which the caller enforces by processing tracks deterministically and
        only ever binding one track per subject.
        """
        candidates = [
            RecoveryCandidate(
                state.subject_number,
                recovery_score(
                    state.motion,  # type: ignore[arg-type] - filtered above
                    bbox,
                    observed_at,
                    max_speed_per_second=self.config.max_speed_per_second,
                ),
            )
            for state in self._recoverable(observed_at)
        ]
        plausible = tuple(
            candidate
            for candidate in sorted(
                candidates, key=lambda item: (-item.score, item.subject_number)
            )
            if candidate.score >= self.config.plausible_candidate_score
        )

        if not plausible:
            return RecoveryDecision(
                raw_tracking_id=raw_tracking_id,
                accepted=False,
                subject_number=None,
                score=None,
                runner_up_score=None,
                reason="no_plausible_subject",
                candidates=(),
            )

        best = plausible[0]
        runner_up = plausible[1].score if len(plausible) > 1 else 0.0
        if best.score < self.config.recovery_min_confidence:
            reason = "below_recovery_threshold"
        elif (best.score - runner_up) < self.config.recovery_margin:
            reason = "ambiguous_candidates"
        else:
            reason = "recovered"
        return RecoveryDecision(
            raw_tracking_id=raw_tracking_id,
            accepted=reason == "recovered",
            subject_number=best.subject_number if reason == "recovered" else None,
            score=best.score,
            runner_up_score=runner_up or None,
            reason=reason,
            candidates=plausible,
        )

    # -- pending tracks ------------------------------------------------------

    def _track_pending(
        self,
        raw_tracking_id: str,
        bbox: BBox,
        observed_at: datetime,
        decision: RecoveryDecision,
    ) -> list[SubjectEvent]:
        """Progresses an unowned raw track towards (maybe) earning a subject."""
        blocked = decision.has_plausible_candidate
        reason = (
            "possible_continuation_of_lost_subject" if blocked else "awaiting_qualification"
        )
        pending = self._pending.get(raw_tracking_id)
        if pending is None:
            pending = PendingTrack(
                raw_tracking_id=raw_tracking_id,
                first_seen_at=observed_at,
                last_seen_at=observed_at,
                bbox=bbox,
                reason=reason,
            )
            self._pending[raw_tracking_id] = pending
        else:
            pending.last_seen_at = observed_at
            pending.bbox = bbox
            pending.frames += 1
            pending.reason = reason

        events: list[SubjectEvent] = []
        if blocked:
            # Allocating a fresh number here would duplicate a person who
            # already owns one. Hold instead, and say so once.
            if not pending.announced:
                pending.announced = True
                events.append(
                    SubjectEvent(
                        kind=SubjectEventKind.UNRESOLVED_CANDIDATE,
                        at=observed_at,
                        tracking_id=raw_tracking_id,
                        association=TrackAssociation.UNRESOLVED,
                        association_confidence=decision.score,
                        reason=decision.reason,
                    )
                )
            return events

        if not pending.qualifies(self.config):
            return events
        del self._pending[raw_tracking_id]
        return self._create_subject(raw_tracking_id, bbox, observed_at, pending.first_seen_at)

    def _expire_pending(self, frame: dict[str, BBox], observed_at: datetime) -> None:
        for raw_id in [
            key
            for key, pending in self._pending.items()
            if key not in frame
            and (observed_at - pending.last_seen_at).total_seconds()
            > self.config.pending_gap_seconds
        ]:
            del self._pending[raw_id]

    # -- state transitions ---------------------------------------------------

    def _create_subject(
        self,
        raw_tracking_id: str,
        bbox: BBox,
        observed_at: datetime,
        first_seen_at: datetime,
    ) -> list[SubjectEvent]:
        number = self._allocator() if self._allocator is not None else self._next_number
        number = int(number)
        if number in self._subjects:
            raise RuntimeError(
                f"subject number {number} is already assigned in this session; "
                "identity numbers must never be reused"
            )
        self._next_number = max(self._next_number, number + 1)
        state = SubjectState(
            subject_number=number,
            first_seen_at=first_seen_at,
            last_seen_at=observed_at,
            lifecycle=SubjectLifecycle.ACTIVE,
            association=TrackAssociation.UNRESOLVED,
            motion=initial_motion(
                bbox, observed_at, trajectory_length=self.config.trajectory_length
            ),
        )
        self._subjects[number] = state
        events = [
            SubjectEvent(
                kind=SubjectEventKind.SUBJECT_CREATED,
                at=observed_at,
                subject_number=number,
                label=state.label,
                lifecycle=SubjectLifecycle.ACTIVE,
                association=TrackAssociation.UNRESOLVED,
                reason="temporal_qualification_reached",
            )
        ]
        events.extend(
            self._bind(
                state,
                raw_tracking_id,
                bbox,
                observed_at,
                method=AssociationMethod.INITIAL,
                confidence=None,
                kind=SubjectEventKind.TRACK_BOUND,
                count_recovery=False,
            )
        )
        return events

    def _bind(
        self,
        state: SubjectState,
        raw_tracking_id: str,
        bbox: BBox,
        observed_at: datetime,
        *,
        method: AssociationMethod,
        confidence: Optional[float],
        kind: SubjectEventKind,
        count_recovery: bool = True,
    ) -> list[SubjectEvent]:
        """Gives exclusive ownership of one raw track to one subject."""
        other = self._owner_of(raw_tracking_id)
        if other is not None and other.subject_number != state.subject_number:
            raise RuntimeError(
                f"raw track {raw_tracking_id!r} is already owned by "
                f"{other.label}; a raw track has exactly one owner"
            )
        if state.active_tracking_id is not None and state.active_tracking_id != raw_tracking_id:
            raise RuntimeError(
                f"{state.label} still owns {state.active_tracking_id!r}; "
                "release it before binding another raw track"
            )
        state.active_tracking_id = raw_tracking_id
        state.last_seen_at = observed_at
        state.motion = (
            initial_motion(bbox, observed_at, trajectory_length=self.config.trajectory_length)
            if state.motion is None
            else advance_motion(
                state.motion,
                bbox,
                observed_at,
                smoothing=self.config.motion_smoothing,
                trajectory_length=self.config.trajectory_length,
            )
        )
        state.last_association_confidence = confidence
        state.segments.append(
            TrackSegment(
                raw_tracking_id=raw_tracking_id,
                started_at=observed_at,
                method=method,
                association_state=TrackAssociation.CONFIRMED,
                association_confidence=confidence,
                start_reason=kind.value,
            )
        )
        if count_recovery:
            state.recovery_count += 1
        events = [
            SubjectEvent(
                kind=kind,
                at=observed_at,
                subject_number=state.subject_number,
                label=state.label,
                tracking_id=raw_tracking_id,
                method=method,
                association_confidence=confidence,
                lifecycle=SubjectLifecycle.ACTIVE,
                association=TrackAssociation.CONFIRMED,
            )
        ]
        events.extend(
            self._set_lifecycle(
                state, SubjectLifecycle.ACTIVE, observed_at, reason="raw_track_bound"
            )
        )
        events.extend(
            self._set_association(
                state, TrackAssociation.CONFIRMED, observed_at, reason="raw_track_bound"
            )
        )
        return events

    def _release(
        self,
        state: SubjectState,
        observed_at: datetime,
        *,
        association: TrackAssociation,
        reason: str,
    ) -> list[SubjectEvent]:
        """Drops raw-track ownership. The subject and its number remain."""
        raw_id = state.active_tracking_id
        state.active_tracking_id = None
        self._close_open_segment(state, observed_at, reason=reason)
        events = [
            SubjectEvent(
                kind=SubjectEventKind.TRACK_RELEASED,
                at=observed_at,
                subject_number=state.subject_number,
                label=state.label,
                tracking_id=raw_id,
                lifecycle=state.lifecycle,
                reason=reason,
            )
        ]
        events.extend(
            self._set_association(state, association, observed_at, reason=reason)
        )
        return events

    def _close_open_segment(
        self, state: SubjectState, ended_at: datetime, *, reason: str
    ) -> None:
        for index in range(len(state.segments) - 1, -1, -1):
            segment = state.segments[index]
            if segment.is_open:
                state.segments[index] = TrackSegment(
                    raw_tracking_id=segment.raw_tracking_id,
                    started_at=segment.started_at,
                    method=segment.method,
                    association_state=segment.association_state,
                    association_confidence=segment.association_confidence,
                    start_reason=segment.start_reason,
                    ended_at=ended_at,
                    end_reason=reason,
                )
                return

    def _end_subject(
        self, state: SubjectState, ended_at: datetime, *, reason: str
    ) -> list[SubjectEvent]:
        events: list[SubjectEvent] = []
        if state.active_tracking_id is not None:
            events.extend(
                self._release(
                    state, ended_at, association=TrackAssociation.UNRESOLVED, reason=reason
                )
            )
        else:
            self._close_open_segment(state, ended_at, reason=reason)
        previous = state.lifecycle
        state.lifecycle = SubjectLifecycle.ENDED
        state.ended_at = ended_at
        events.append(
            SubjectEvent(
                kind=SubjectEventKind.ENDED,
                at=ended_at,
                subject_number=state.subject_number,
                label=state.label,
                previous_lifecycle=previous,
                lifecycle=SubjectLifecycle.ENDED,
                association=state.association,
                reason=reason,
            )
        )
        return events

    def _set_lifecycle(
        self,
        state: SubjectState,
        lifecycle: SubjectLifecycle,
        observed_at: datetime,
        *,
        reason: str,
    ) -> list[SubjectEvent]:
        if state.lifecycle is lifecycle or state.lifecycle is SubjectLifecycle.ENDED:
            return []
        previous = state.lifecycle
        state.lifecycle = lifecycle
        kind = {
            SubjectLifecycle.TEMPORARILY_LOST: SubjectEventKind.TEMPORARILY_LOST,
            SubjectLifecycle.LOST: SubjectEventKind.LOST,
            SubjectLifecycle.ACTIVE: SubjectEventKind.TRACK_RECOVERED
            if previous in _RECOVERABLE
            else SubjectEventKind.TRACK_BOUND,
        }[lifecycle]
        return [
            SubjectEvent(
                kind=kind,
                at=observed_at,
                subject_number=state.subject_number,
                label=state.label,
                tracking_id=state.active_tracking_id,
                previous_lifecycle=previous,
                lifecycle=lifecycle,
                association=state.association,
                reason=reason,
            )
        ]

    def _set_association(
        self,
        state: SubjectState,
        association: TrackAssociation,
        observed_at: datetime,
        *,
        reason: str,
    ) -> list[SubjectEvent]:
        if state.association is association:
            return []
        state.association = association
        if association is not TrackAssociation.CONFLICT:
            return []
        return [
            SubjectEvent(
                kind=SubjectEventKind.CONFLICT,
                at=observed_at,
                subject_number=state.subject_number,
                label=state.label,
                tracking_id=state.active_tracking_id,
                lifecycle=state.lifecycle,
                association=association,
                reason=reason,
            )
        ]
