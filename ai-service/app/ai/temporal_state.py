"""Temporal confirmation and cooldown.

A single frame never produces an event. A candidate must persist for both a
minimum wall-clock duration and a minimum number of matching frames, and the
same logical alert is suppressed for the configured cooldown window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..domain.models import AssociationStatus


@dataclass
class Candidate:
    """Accumulated evidence for one (camera, rule, subject, type) alert key."""

    key: tuple[str, ...]
    first_seen: float
    last_seen: float
    frames: int = 1
    best_trigger_confidence: float = 0.0
    best_association_confidence: Optional[float] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


@dataclass
class ConfirmationResult:
    confirmed: bool
    duration_seconds: float
    frame_count: int
    suppressed_by_cooldown: bool = False


def alert_key(
    camera_id: str,
    rule_id: str,
    event_type: str,
    subject_id: Optional[str],
) -> tuple[str, ...]:
    """Cooldown/candidate scope. `subject_id` is a tracking id, never identity."""
    return (camera_id, rule_id, event_type, subject_id or "unassigned")


def subject_for(status: AssociationStatus, person_tracking_id: Optional[str], phone_id: str) -> str:
    """A definitive person scopes the alert; otherwise the phone track does."""
    if status is AssociationStatus.ASSOCIATED and person_tracking_id:
        return f"person:{person_tracking_id}"
    return f"object:{phone_id}"


class TemporalConfirmer:
    """Per-camera temporal state machine. Uses injected monotonic timestamps."""

    def __init__(self, gap_tolerance_seconds: float = 0.5) -> None:
        self.gap_tolerance = float(gap_tolerance_seconds)
        self._candidates: dict[tuple[str, ...], Candidate] = {}
        self._cooldowns: dict[tuple[str, ...], float] = {}

    def observe(
        self,
        key: tuple[str, ...],
        *,
        now: float,
        min_duration_seconds: float,
        min_matching_frames: int,
        cooldown_seconds: float,
        trigger_confidence: float = 0.0,
        association_confidence: Optional[float] = None,
    ) -> ConfirmationResult:
        """Records one matching frame and reports whether an event fires."""
        candidate = self._candidates.get(key)
        if candidate is None or (now - candidate.last_seen) > self.gap_tolerance:
            candidate = Candidate(key=key, first_seen=now, last_seen=now, frames=1)
        else:
            candidate.last_seen = now
            candidate.frames += 1
        candidate.best_trigger_confidence = max(
            candidate.best_trigger_confidence, float(trigger_confidence)
        )
        if association_confidence is not None:
            candidate.best_association_confidence = max(
                candidate.best_association_confidence or 0.0, float(association_confidence)
            )
        self._candidates[key] = candidate

        satisfied = (
            candidate.duration >= float(min_duration_seconds)
            and candidate.frames >= int(min_matching_frames)
        )
        if not satisfied:
            return ConfirmationResult(False, candidate.duration, candidate.frames)

        released_at = self._cooldowns.get(key)
        if released_at is not None and (now - released_at) < float(cooldown_seconds):
            return ConfirmationResult(
                False, candidate.duration, candidate.frames, suppressed_by_cooldown=True
            )

        self._cooldowns[key] = now
        result = ConfirmationResult(True, candidate.duration, candidate.frames)
        # The candidate restarts so the next event needs fresh persistence.
        self._candidates.pop(key, None)
        return result

    def expire(self, now: float) -> None:
        """Drops candidates whose detection disappeared for too long."""
        stale = [
            key
            for key, candidate in self._candidates.items()
            if (now - candidate.last_seen) > self.gap_tolerance
        ]
        for key in stale:
            del self._candidates[key]

    @property
    def active_candidates(self) -> int:
        return len(self._candidates)


class InstantGate:
    """Deduplication for single-frame ("instant") evidence.

    Completely separate state from :class:`TemporalConfirmer`, so an instant
    warning can never reset, consume or suppress a temporal candidate. Scope is
    always (camera, rule, subject), so no camera or rule suppresses another.
    """

    def __init__(self) -> None:
        self._fired: dict[tuple[str, ...], float] = {}

    def allow(self, key: tuple[str, ...], *, now: float, cooldown_seconds: float) -> bool:
        last = self._fired.get(key)
        if last is not None and (now - last) < float(cooldown_seconds):
            return False
        self._fired[key] = now
        return True

    def reset(self) -> None:
        self._fired.clear()



@dataclass
class AssociationMemory:
    """Short-lived per-phone association history used for temporal continuity."""

    ttl_seconds: float = 2.0
    _entries: dict[str, tuple[float, dict[str, float]]] = field(default_factory=dict)

    def remember(self, phone_id: str, person_tracking_id: str, confidence: float, now: float) -> None:
        self._entries[phone_id] = (now, {person_tracking_id: float(confidence)})

    def recall(self, phone_id: str, now: float) -> dict[str, float]:
        entry = self._entries.get(phone_id)
        if not entry:
            return {}
        stamped, mapping = entry
        if (now - stamped) > self.ttl_seconds:
            self._entries.pop(phone_id, None)
            return {}
        return mapping