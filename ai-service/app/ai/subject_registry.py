"""Anonymous exam-session subject registry (Phase 2 runtime logic).

One registry instance owns the anonymous subjects of exactly one
(exam session, camera) pair. It converts unstable raw tracker ids into stable
per-session labels (``S001``, ``S002``, …) using two conservative rules only:

1. **Temporal qualification** — a brand-new raw track must persist for a
   configured number of frames *and* a configured wall-clock duration before it
   earns a subject. Flicker never creates identities.
2. **Short-gap spatial recovery** — a lost subject may reclaim a *new* raw track
   only inside a configured gap window, only when the geometric recovery score
   clears the threshold, and only when it beats the runner-up by a margin.
   Anything less stays ``UNCERTAIN``; the registry never guesses.

Deliberately absent: face recognition, appearance/clothing features, seat maps,
roster matching and any link to a real student. Those are forbidden by
``docs/exam-session-identity-contract.md``.

Pure logic: no clocks (the caller supplies ``observed_at``), no I/O, no models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Optional

from ..domain.geometry import BBox
from ..domain.observations import PersonObservation
from ..domain.session_subjects import (
    AssociationMethod,
    PendingTrack,
    ReassociationCandidate,
    ReassociationDecision,
    SubjectEvent,
    SubjectEventKind,
    SubjectFrameResult,
    SubjectRegistryConfig,
    SubjectSnapshot,
    SubjectState,
    SubjectTrackingStatus,
    TrackSegment,
    blend_anchor,
    spatial_recovery_score,
)


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
        # Subject numbers must be unique per EXAM SESSION, not per camera, so a
        # multi-camera session injects one shared allocator.
        self._allocator = number_allocator
        self._subjects: dict[int, SubjectState] = {}
        self._pending: dict[str, PendingTrack] = {}
        self._next_number = 1
        self._last_frame_at: Optional[datetime] = None

    # ------------------------------------------------------------------ reads

    def snapshots(self) -> tuple[SubjectSnapshot, ...]:
        return tuple(
            self._subjects[number].snapshot() for number in sorted(self._subjects)
        )

    def subject_for_track(self, raw_tracking_id: str) -> Optional[SubjectSnapshot]:
        """The subject currently owning a raw track, if any."""
        for state in self._subjects.values():
            if state.active_tracking_id == raw_tracking_id:
                return state.snapshot()
        return None

    @property
    def active_subject_count(self) -> int:
        return sum(
            1
            for state in self._subjects.values()
            if state.status is not SubjectTrackingStatus.ENDED
        )

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
        decisions: list[ReassociationDecision] = []

        frame, duplicates = self._frame_tracks(observations)
        for raw_id in duplicates:
            owner = self._owner_of(raw_id)
            if owner is not None:
                events.extend(
                    self._set_status(
                        owner,
                        SubjectTrackingStatus.CONFLICT,
                        observed_at,
                        reason="duplicate_raw_tracking_id_in_frame",
                    )
                )

        events.extend(self._advance_attached(frame, observed_at))
        events.extend(self._age_detached(frame, observed_at))

        claimed = {
            state.active_tracking_id
            for state in self._subjects.values()
            if state.active_tracking_id
        }
        for raw_id, bbox in frame.items():
            if raw_id in claimed:
                continue
            decision = self._try_recover(raw_id, bbox, observed_at)
            decisions.append(decision)
            if decision.accepted and decision.subject_number is not None:
                events.extend(
                    self._attach(
                        self._subjects[decision.subject_number],
                        raw_id,
                        bbox,
                        observed_at,
                        method=AssociationMethod.SHORT_GAP_REASSOCIATION,
                        confidence=decision.score,
                    )
                )
                self._pending.pop(raw_id, None)
                continue
            if decision.reason == "ambiguous":
                for candidate in decision.candidates[:2]:
                    state = self._subjects.get(candidate.subject_number)
                    if state is not None and state.status is SubjectTrackingStatus.TEMPORARILY_LOST:
                        events.extend(
                            self._set_status(
                                state,
                                SubjectTrackingStatus.UNCERTAIN,
                                observed_at,
                                reason="ambiguous_short_gap_recovery",
                            )
                        )
            events.extend(self._track_pending(raw_id, bbox, observed_at))

        self._expire_pending(frame, observed_at)

        return SubjectFrameResult(
            exam_session_id=self.exam_session_id,
            camera_id=self.camera_id,
            observed_at=observed_at,
            subjects=self.snapshots(),
            events=tuple(events),
            decisions=tuple(decisions),
        )

    def close(self, *, ended_at: datetime) -> tuple[SubjectEvent, ...]:
        """Ends the session: every open segment is closed truthfully."""
        events: list[SubjectEvent] = []
        for number in sorted(self._subjects):
            state = self._subjects[number]
            if state.status is SubjectTrackingStatus.ENDED:
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
                # An untracked person cannot own a stable subject: dropping it
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
        for state in self._subjects.values():
            if state.active_tracking_id == raw_tracking_id:
                return state
        return None

    def _advance_attached(self, frame: dict[str, BBox], observed_at: datetime) -> list[SubjectEvent]:
        events: list[SubjectEvent] = []
        for number in sorted(self._subjects):
            state = self._subjects[number]
            raw_id = state.active_tracking_id
            if raw_id is None or raw_id not in frame:
                continue
            state.last_seen_at = observed_at
            state.anchor = blend_anchor(state.anchor, frame[raw_id], self.config.anchor_smoothing)
            state.anchor_updated_at = observed_at
            if state.status in (
                SubjectTrackingStatus.TEMPORARILY_LOST,
                SubjectTrackingStatus.UNCERTAIN,
            ):
                events.extend(
                    self._set_status(
                        state,
                        SubjectTrackingStatus.STABLE,
                        observed_at,
                        reason="raw_track_observed_again",
                    )
                )
        return events

    def _age_detached(self, frame: dict[str, BBox], observed_at: datetime) -> list[SubjectEvent]:
        events: list[SubjectEvent] = []
        for number in sorted(self._subjects):
            state = self._subjects[number]
            if state.status is SubjectTrackingStatus.ENDED:
                continue
            if state.active_tracking_id is not None and state.active_tracking_id in frame:
                continue
            gap = (observed_at - state.last_seen_at).total_seconds()
            if gap >= self.config.end_after_seconds:
                events.extend(self._end_subject(state, observed_at, reason="recovery_window_expired"))
                continue
            if gap >= self.config.lost_after_seconds:
                if state.active_tracking_id is not None:
                    events.append(self._detach(state, observed_at))
                if state.status is SubjectTrackingStatus.STABLE:
                    events.extend(
                        self._set_status(
                            state,
                            SubjectTrackingStatus.TEMPORARILY_LOST,
                            observed_at,
                            reason="raw_track_lost",
                        )
                    )
        return events

    def _try_recover(
        self, raw_tracking_id: str, bbox: BBox, observed_at: datetime
    ) -> ReassociationDecision:
        candidates: list[ReassociationCandidate] = []
        for number in sorted(self._subjects):
            state = self._subjects[number]
            if state.active_tracking_id is not None:
                continue
            if state.status not in (
                SubjectTrackingStatus.TEMPORARILY_LOST,
                SubjectTrackingStatus.UNCERTAIN,
            ):
                # CONFLICT and ENDED subjects are never silently repaired.
                continue
            if (observed_at - state.last_seen_at).total_seconds() > self.config.short_gap_seconds:
                continue
            candidates.append(
                ReassociationCandidate(number, spatial_recovery_score(state.anchor, bbox))
            )
        candidates.sort(key=lambda item: (-item.score, item.subject_number))

        if not candidates:
            return ReassociationDecision(
                raw_tracking_id=raw_tracking_id,
                accepted=False,
                subject_number=None,
                score=None,
                runner_up_score=None,
                reason="no_recoverable_subject",
                candidates=(),
            )

        best = candidates[0]
        runner_up = candidates[1].score if len(candidates) > 1 else 0.0
        if best.score < self.config.reassociation_min_confidence:
            reason = "below_recovery_threshold"
        elif (best.score - runner_up) < self.config.reassociation_margin:
            reason = "ambiguous"
        else:
            reason = "recovered"
        return ReassociationDecision(
            raw_tracking_id=raw_tracking_id,
            accepted=reason == "recovered",
            subject_number=best.subject_number if reason == "recovered" else None,
            score=best.score,
            runner_up_score=runner_up or None,
            reason=reason,
            candidates=tuple(candidates),
        )

    def _track_pending(
        self, raw_tracking_id: str, bbox: BBox, observed_at: datetime
    ) -> list[SubjectEvent]:
        pending = self._pending.get(raw_tracking_id)
        if pending is None:
            self._pending[raw_tracking_id] = PendingTrack(
                raw_tracking_id=raw_tracking_id,
                first_seen_at=observed_at,
                last_seen_at=observed_at,
                bbox=bbox,
            )
            return []
        pending.last_seen_at = observed_at
        pending.bbox = bbox
        pending.frames += 1
        if not pending.qualifies(self.config):
            return []
        del self._pending[raw_tracking_id]
        return self._create_subject(raw_tracking_id, bbox, observed_at, pending.first_seen_at)

    def _expire_pending(self, frame: dict[str, BBox], observed_at: datetime) -> None:
        for raw_id in [
            key
            for key, pending in self._pending.items()
            if key not in frame
            and (observed_at - pending.last_seen_at).total_seconds() > self.config.pending_gap_seconds
        ]:
            del self._pending[raw_id]

    def _create_subject(
        self,
        raw_tracking_id: str,
        bbox: BBox,
        observed_at: datetime,
        first_seen_at: datetime,
    ) -> list[SubjectEvent]:
        number = self._allocator() if self._allocator is not None else self._next_number
        self._next_number = max(self._next_number, number + 1)
        state = SubjectState(
            subject_number=number,
            first_seen_at=first_seen_at,
            last_seen_at=observed_at,
            anchor=bbox,
            anchor_updated_at=observed_at,
            status=SubjectTrackingStatus.STABLE,
        )
        self._subjects[number] = state
        events = [
            SubjectEvent(
                kind=SubjectEventKind.SUBJECT_CREATED,
                subject_number=number,
                label=state.label,
                at=observed_at,
                status=SubjectTrackingStatus.STABLE,
                reason="temporal_qualification_reached",
            )
        ]
        events.extend(
            self._attach(
                state,
                raw_tracking_id,
                bbox,
                observed_at,
                method=AssociationMethod.INITIAL,
                confidence=None,
                count_reassociation=False,
            )
        )
        return events

    def _attach(
        self,
        state: SubjectState,
        raw_tracking_id: str,
        bbox: BBox,
        observed_at: datetime,
        *,
        method: AssociationMethod,
        confidence: Optional[float],
        count_reassociation: bool = True,
    ) -> list[SubjectEvent]:
        state.active_tracking_id = raw_tracking_id
        state.last_seen_at = observed_at
        state.anchor = blend_anchor(state.anchor, bbox, self.config.anchor_smoothing)
        state.anchor_updated_at = observed_at
        state.last_association_confidence = confidence
        state.segments.append(
            TrackSegment(
                raw_tracking_id=raw_tracking_id,
                started_at=observed_at,
                method=method,
                association_confidence=confidence,
            )
        )
        if count_reassociation:
            state.reassociation_count += 1
        events = [
            SubjectEvent(
                kind=SubjectEventKind.TRACK_ATTACHED,
                subject_number=state.subject_number,
                label=state.label,
                at=observed_at,
                tracking_id=raw_tracking_id,
                method=method,
                association_confidence=confidence,
            )
        ]
        if state.status is not SubjectTrackingStatus.STABLE:
            events.extend(
                self._set_status(
                    state,
                    SubjectTrackingStatus.STABLE,
                    observed_at,
                    reason="raw_track_attached",
                )
            )
        return events

    def _detach(self, state: SubjectState, observed_at: datetime) -> SubjectEvent:
        raw_id = state.active_tracking_id
        state.active_tracking_id = None
        self._close_open_segment(state, observed_at)
        return SubjectEvent(
            kind=SubjectEventKind.TRACK_DETACHED,
            subject_number=state.subject_number,
            label=state.label,
            at=observed_at,
            tracking_id=raw_id,
        )

    def _close_open_segment(self, state: SubjectState, ended_at: datetime) -> None:
        for index in range(len(state.segments) - 1, -1, -1):
            segment = state.segments[index]
            if segment.is_open:
                state.segments[index] = TrackSegment(
                    raw_tracking_id=segment.raw_tracking_id,
                    started_at=segment.started_at,
                    method=segment.method,
                    association_confidence=segment.association_confidence,
                    ended_at=ended_at,
                )
                return

    def _end_subject(
        self, state: SubjectState, ended_at: datetime, *, reason: str
    ) -> list[SubjectEvent]:
        events: list[SubjectEvent] = []
        if state.active_tracking_id is not None:
            events.append(self._detach(state, ended_at))
        else:
            self._close_open_segment(state, ended_at)
        previous = state.status
        state.status = SubjectTrackingStatus.ENDED
        state.ended_at = ended_at
        events.append(
            SubjectEvent(
                kind=SubjectEventKind.SUBJECT_ENDED,
                subject_number=state.subject_number,
                label=state.label,
                at=ended_at,
                previous_status=previous,
                status=SubjectTrackingStatus.ENDED,
                reason=reason,
            )
        )
        return events

    def _set_status(
        self,
        state: SubjectState,
        status: SubjectTrackingStatus,
        observed_at: datetime,
        *,
        reason: str,
    ) -> list[SubjectEvent]:
        if state.status is status:
            return []
        previous = state.status
        state.status = status
        return [
            SubjectEvent(
                kind=SubjectEventKind.STATUS_CHANGED,
                subject_number=state.subject_number,
                label=state.label,
                at=observed_at,
                previous_status=previous,
                status=status,
                reason=reason,
            )
        ]
