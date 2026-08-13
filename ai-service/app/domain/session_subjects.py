"""Immutable anonymous exam-subject identity contract (Phase 2, corrected).

An *exam session subject* is a stable anonymous handle (``S001``, ``S002``, …)
for **one logical physical person** inside **one exam session**. See
``docs/exam-session-identity-contract.md``.

The primary invariant of this module:

    Once ``S017`` has been assigned to a physical person, ``S017`` belongs to
    that person for the rest of the session. It is never renumbered, never
    transferred to somebody else, and never reused — not when the person walks
    to another desk, not when the raw tracker id changes, not when the person is
    lost entirely.

``S017`` is therefore **person-scoped**, never seat-scoped, location-scoped,
bounding-box-scoped or raw-tracker-scoped.

Two concepts are kept strictly separate:

```text
subject lifecycle     ACTIVE -> TEMPORARILY_LOST -> LOST -> ENDED   (existence)
track association     CONFIRMED / PROVISIONAL / UNRESOLVED / CONFLICT (binding)
```

A subject keeps existing (and keeps its number reserved) while its current
track association is ``UNRESOLVED``. Motion state is mobility-aware evidence for
short-gap recovery only: it moves with the person and is never a seat, a place
registration or an identity.

Everything here is pure data: no OpenCV, no Supabase, no clocks, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .geometry import BBox, clamp01, iou, normalized_distance

#: Subject labels are ``S`` + zero-padded number, matching the database
#: generated ``session_subjects.subject_label`` column exactly.
SUBJECT_LABEL_DIGITS = 3


def subject_label(subject_number: int) -> str:
    """Deterministic anonymous label for a per-session subject number."""
    if subject_number < 1:
        raise ValueError("subject_number must be >= 1")
    return f"S{subject_number:0{SUBJECT_LABEL_DIGITS}d}"


#: Human-facing text for a raw track that must NOT be attached to any subject
#: yet. Displayed instead of inventing an S-number.
UNRESOLVED_TRACK_LABEL = "UNRESOLVED"


class SubjectLifecycle(str, Enum):
    """Existence of the subject — independent of any raw tracker binding."""

    #: Currently observed (its raw track was seen recently).
    ACTIVE = "active"
    #: Not observed right now, still inside the short-gap recovery window.
    TEMPORARILY_LOST = "temporarily_lost"
    #: Recovery window expired. The number stays reserved for this session.
    LOST = "lost"
    #: The exam session ended. History is preserved, never deleted.
    ENDED = "ended"


class TrackAssociation(str, Enum):
    """State of the subject's binding to a raw tracker id right now."""

    #: Bound to a live raw track that behaves plausibly.
    CONFIRMED = "confirmed"
    #: Bound, but the evidence is weaker than a confirmed continuation.
    PROVISIONAL = "provisional"
    #: No raw track is attached (lost, or ambiguity refused).
    UNRESOLVED = "unresolved"
    #: Contradictory raw-track evidence (possible tracker swap). Never silently
    #: repaired: a human-visible conflict is preferred over a guessed identity.
    CONFLICT = "conflict"


class AssociationMethod(str, Enum):
    INITIAL = "initial"
    SHORT_GAP_REASSOCIATION = "short_gap_reassociation"
    RESTORED_AFTER_RESTART = "restored_after_restart"


class SubjectEventKind(str, Enum):
    """Structured transition names, also used verbatim in logs."""

    SUBJECT_CREATED = "exam_subject_created"
    TRACK_BOUND = "exam_subject_track_bound"
    TRACK_RECOVERED = "exam_subject_track_recovered"
    TRACK_RELEASED = "exam_subject_track_released"
    TEMPORARILY_LOST = "exam_subject_temporarily_lost"
    LOST = "exam_subject_lost"
    CONFLICT = "exam_subject_conflict"
    ENDED = "exam_subject_ended"
    UNRESOLVED_CANDIDATE = "unresolved_track_candidate"


@dataclass(frozen=True, slots=True)
class SubjectRegistryConfig:
    """Explicit, uncalibrated registry policy.

    No field has a hidden "sensible" default: qualification, gap tolerance and
    plausible motion depend entirely on frame rate, hall size and tracker
    quality, none of which this service may assume.
    """

    #: Frames a new raw track must persist before it can earn a subject.
    min_frames_to_qualify: int
    #: Wall-clock persistence a new raw track must show before it can earn one.
    min_seconds_to_qualify: float
    #: Longest gap in which a lost subject may still reclaim a new raw track.
    short_gap_seconds: float
    #: Gap after which a subject is reported TEMPORARILY_LOST.
    lost_after_seconds: float
    #: Minimum recovery score for reassociation to be accepted at all.
    recovery_min_confidence: float
    #: The winner must beat every runner-up (both directions) by this margin.
    recovery_margin: float
    #: Score above which a lost subject counts as a *plausible* continuation, so
    #: a new raw track must NOT be given a fresh S-number yet.
    plausible_candidate_score: float
    #: Weight of the newest observation when smoothing motion state.
    motion_smoothing: float
    #: Frames a pending raw track may miss before its progress is discarded.
    pending_gap_seconds: float
    #: Physically plausible motion, in normalized frame widths per second.
    #: Used to reject impossible jumps (tracker swaps) — fail-closed.
    max_speed_per_second: float
    #: Bounded trajectory history kept per subject (memory is deterministic).
    trajectory_length: int

    def __post_init__(self) -> None:
        if self.min_frames_to_qualify < 1:
            raise ValueError("min_frames_to_qualify must be >= 1")
        if self.trajectory_length < 1:
            raise ValueError("trajectory_length must be >= 1")
        for name in (
            "min_seconds_to_qualify",
            "short_gap_seconds",
            "lost_after_seconds",
            "pending_gap_seconds",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if float(self.max_speed_per_second) <= 0.0:
            raise ValueError("max_speed_per_second must be > 0")
        for name in (
            "recovery_min_confidence",
            "recovery_margin",
            "plausible_candidate_score",
            "motion_smoothing",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within 0..1")
        if self.lost_after_seconds > self.short_gap_seconds:
            raise ValueError("lost_after_seconds must not exceed short_gap_seconds")
        if self.plausible_candidate_score > self.recovery_min_confidence:
            raise ValueError(
                "plausible_candidate_score must not exceed recovery_min_confidence: "
                "a score good enough to accept must also count as plausible"
            )


@dataclass(frozen=True, slots=True)
class MotionState:
    """Mobility-aware short-term state. Evidence only, never identity.

    This deliberately replaces the earlier fixed "anchor" idea: the state moves
    with the person, so standing up, walking across the hall and sitting down
    somewhere else are normal and never cost a subject its label.
    """

    last_bbox: BBox
    #: Normalized units per second, estimated from consecutive observations.
    velocity_x: float
    velocity_y: float
    updated_at: datetime
    #: Bounded recent history, newest last.
    trajectory: tuple[tuple[datetime, BBox], ...] = ()

    def predict(self, at: datetime) -> BBox:
        """Where the person plausibly is now, given the last observed motion."""
        elapsed = max(0.0, (at - self.updated_at).total_seconds())
        return BBox(
            clamp01(self.last_bbox.x + self.velocity_x * elapsed),
            clamp01(self.last_bbox.y + self.velocity_y * elapsed),
            self.last_bbox.width,
            self.last_bbox.height,
        )


@dataclass(frozen=True, slots=True)
class TrackSegment:
    """One continuous stretch of raw-track ownership. Append-only history."""

    raw_tracking_id: str
    started_at: datetime
    method: AssociationMethod
    association_state: TrackAssociation = TrackAssociation.CONFIRMED
    association_confidence: Optional[float] = None
    start_reason: Optional[str] = None
    ended_at: Optional[datetime] = None
    end_reason: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.ended_at is None


@dataclass(frozen=True, slots=True)
class SubjectSnapshot:
    """Immutable read-only view of one subject at one moment."""

    subject_number: int
    label: str
    lifecycle: SubjectLifecycle
    association: TrackAssociation
    first_seen_at: datetime
    last_seen_at: datetime
    ended_at: Optional[datetime]
    active_tracking_id: Optional[str]
    motion: Optional[MotionState]
    recovery_count: int
    last_association_confidence: Optional[float]
    segments: tuple[TrackSegment, ...]

    @property
    def is_open(self) -> bool:
        return self.lifecycle is not SubjectLifecycle.ENDED


@dataclass(frozen=True, slots=True)
class SubjectEvent:
    """Something that actually happened. Never a prediction or a guess."""

    kind: SubjectEventKind
    at: datetime
    subject_number: Optional[int] = None
    label: Optional[str] = None
    tracking_id: Optional[str] = None
    method: Optional[AssociationMethod] = None
    association_confidence: Optional[float] = None
    previous_lifecycle: Optional[SubjectLifecycle] = None
    lifecycle: Optional[SubjectLifecycle] = None
    association: Optional[TrackAssociation] = None
    reason: Optional[str] = None


@dataclass(frozen=True, slots=True)
class RecoveryCandidate:
    subject_number: int
    score: float


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """Complete, diagnosable outcome of one reassociation attempt."""

    raw_tracking_id: str
    accepted: bool
    subject_number: Optional[int]
    score: Optional[float]
    runner_up_score: Optional[float]
    reason: str
    candidates: tuple[RecoveryCandidate, ...] = ()

    @property
    def has_plausible_candidate(self) -> bool:
        return bool(self.candidates)


@dataclass(frozen=True, slots=True)
class UnresolvedCandidate:
    """A raw track that is deliberately NOT an identity yet."""

    raw_tracking_id: str
    first_seen_at: datetime
    last_seen_at: datetime
    frames: int
    #: Why no S-number was allocated: `awaiting_qualification` or
    #: `possible_continuation_of_lost_subject`.
    reason: str


@dataclass(frozen=True, slots=True)
class SubjectFrameResult:
    """What one analysed frame did to the subject registry."""

    exam_session_id: str
    camera_id: str
    observed_at: datetime
    subjects: tuple[SubjectSnapshot, ...] = ()
    events: tuple[SubjectEvent, ...] = ()
    decisions: tuple[RecoveryDecision, ...] = ()
    unresolved: tuple[UnresolvedCandidate, ...] = ()
    #: raw tracking id -> anonymous label, for the annotated stream only.
    labels: tuple[tuple[str, str], ...] = ()

    def label_for(self, raw_tracking_id: str) -> str:
        """Human-facing label for a raw track: an S-number or ``UNRESOLVED``."""
        for raw_id, label in self.labels:
            if raw_id == raw_tracking_id:
                return label
        return UNRESOLVED_TRACK_LABEL

    def count_of(self, lifecycle: SubjectLifecycle) -> int:
        return sum(1 for item in self.subjects if item.lifecycle is lifecycle)


@dataclass
class PendingTrack:
    """A raw track in the internal UNRESOLVED_CANDIDATE state."""

    raw_tracking_id: str
    first_seen_at: datetime
    last_seen_at: datetime
    bbox: BBox
    frames: int = 1
    reason: str = "awaiting_qualification"
    announced: bool = False

    def qualifies(self, config: SubjectRegistryConfig) -> bool:
        observed = (self.last_seen_at - self.first_seen_at).total_seconds()
        return (
            self.frames >= config.min_frames_to_qualify
            and observed >= config.min_seconds_to_qualify
        )


@dataclass
class SubjectState:
    """Mutable per-subject bookkeeping owned by the registry only.

    ``subject_number`` is written once, at creation, and never changed here or
    anywhere else — the database enforces the same rule with a trigger.
    """

    subject_number: int
    first_seen_at: datetime
    last_seen_at: datetime
    lifecycle: SubjectLifecycle = SubjectLifecycle.ACTIVE
    association: TrackAssociation = TrackAssociation.UNRESOLVED
    active_tracking_id: Optional[str] = None
    motion: Optional[MotionState] = None
    ended_at: Optional[datetime] = None
    recovery_count: int = 0
    last_association_confidence: Optional[float] = None
    segments: list[TrackSegment] = field(default_factory=list)

    @property
    def label(self) -> str:
        return subject_label(self.subject_number)

    def snapshot(self) -> SubjectSnapshot:
        return SubjectSnapshot(
            subject_number=self.subject_number,
            label=self.label,
            lifecycle=self.lifecycle,
            association=self.association,
            first_seen_at=self.first_seen_at,
            last_seen_at=self.last_seen_at,
            ended_at=self.ended_at,
            active_tracking_id=self.active_tracking_id,
            motion=self.motion,
            recovery_count=self.recovery_count,
            last_association_confidence=self.last_association_confidence,
            segments=tuple(self.segments),
        )


def initial_motion(bbox: BBox, at: datetime, *, trajectory_length: int) -> MotionState:
    return MotionState(
        last_bbox=bbox,
        velocity_x=0.0,
        velocity_y=0.0,
        updated_at=at,
        trajectory=((at, bbox),)[-max(1, trajectory_length) :],
    )


def advance_motion(
    motion: MotionState,
    observed: BBox,
    at: datetime,
    *,
    smoothing: float,
    trajectory_length: int,
) -> MotionState:
    """Updates motion state with a fresh observation of the same person."""
    elapsed = (at - motion.updated_at).total_seconds()
    weight = min(1.0, max(0.0, float(smoothing)))
    if elapsed > 0:
        raw_vx = (observed.center[0] - motion.last_bbox.center[0]) / elapsed
        raw_vy = (observed.center[1] - motion.last_bbox.center[1]) / elapsed
        velocity_x = motion.velocity_x * (1.0 - weight) + raw_vx * weight
        velocity_y = motion.velocity_y * (1.0 - weight) + raw_vy * weight
    else:
        velocity_x, velocity_y = motion.velocity_x, motion.velocity_y
    trajectory = (motion.trajectory + ((at, observed),))[-max(1, trajectory_length) :]
    return MotionState(
        last_bbox=observed,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        updated_at=at,
        trajectory=trajectory,
    )


def displacement_is_plausible(
    motion: MotionState, observed: BBox, at: datetime, *, max_speed_per_second: float
) -> bool:
    """Could one person physically have moved from `motion` to `observed`?

    Fail-closed: an impossible jump means the raw tracker probably swapped two
    people, which must surface as a conflict rather than a silent identity swap.
    """
    elapsed = max(0.0, (at - motion.updated_at).total_seconds())
    travelled = (
        (observed.center[0] - motion.last_bbox.center[0]) ** 2
        + (observed.center[1] - motion.last_bbox.center[1]) ** 2
    ) ** 0.5
    # One frame of tolerance for jitter, expressed in the same speed unit.
    allowance = float(max_speed_per_second) * (elapsed + 0.04)
    return travelled <= allowance


def recovery_score(
    motion: MotionState,
    observed: BBox,
    at: datetime,
    *,
    max_speed_per_second: float,
) -> float:
    """0..1 plausibility that ``observed`` continues this subject.

    Purely geometric and motion based: predicted position overlap plus
    size-normalized centre proximity, hard-gated by physically possible travel.
    No appearance, colour, clothing, face or biometric feature is used anywhere.
    """
    if not displacement_is_plausible(
        motion, observed, at, max_speed_per_second=max_speed_per_second
    ):
        return 0.0
    predicted = motion.predict(at)
    overlap = iou(predicted, observed)
    distance = normalized_distance(observed.center, predicted)
    proximity = 0.0 if distance >= 1.5 else max(0.0, 1.0 - distance / 1.5)
    return round(min(1.0, max(0.0, 0.6 * overlap + 0.4 * proximity)), 6)
