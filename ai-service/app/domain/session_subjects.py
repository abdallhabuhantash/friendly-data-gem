"""Anonymous exam-session subject identity contract (Phase 2).

An *exam session subject* is a stable anonymous handle (``S001``, ``S002``, …)
for "a person this exam session has been tracking". It is deliberately **not**
an identity: no name, no university ID, no face, no biometric signature, no
seat. See ``docs/exam-session-identity-contract.md``.

Everything in this module is pure data: no OpenCV, no Supabase, no clocks and
no I/O, so the identity rules can be exercised deterministically in tests.

Layering (never collapsed):

```text
raw tracker id  ->  track segment  ->  anonymous session subject
```

A raw tracker id is a *temporary* label. When tracking breaks the raw id is
lost, and a later raw id may or may not be the same physical person. This layer
therefore records what was observed (segments, gaps, confidences) and never
upgrades a guess into a fact: ambiguity is preserved as ``UNCERTAIN``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .geometry import BBox, clamp01, iou, normalized_distance

#: Subject labels are always ``S`` + zero-padded number, matching the
#: database-generated ``session_subjects.subject_label`` column exactly.
SUBJECT_LABEL_DIGITS = 3


def subject_label(subject_number: int) -> str:
    """Deterministic anonymous label for a per-session subject number."""
    if subject_number < 1:
        raise ValueError("subject_number must be >= 1")
    return f"S{subject_number:0{SUBJECT_LABEL_DIGITS}d}"


class SubjectTrackingStatus(str, Enum):
    """Truthful tracking state of one anonymous subject."""

    #: Currently attached to a live raw track.
    STABLE = "stable"
    #: Raw track lost, still inside the short-gap recovery window.
    TEMPORARILY_LOST = "temporarily_lost"
    #: Recovery was plausible but ambiguous — deliberately not resolved.
    UNCERTAIN = "uncertain"
    #: Contradictory raw-track evidence; never silently repaired.
    CONFLICT = "conflict"
    #: Recovery window expired; the subject is closed for this session.
    ENDED = "ended"


class AssociationMethod(str, Enum):
    INITIAL = "initial"
    SHORT_GAP_REASSOCIATION = "short_gap_reassociation"


class SubjectEventKind(str, Enum):
    SUBJECT_CREATED = "subject_created"
    TRACK_ATTACHED = "track_attached"
    TRACK_DETACHED = "track_detached"
    STATUS_CHANGED = "status_changed"
    SUBJECT_ENDED = "subject_ended"


@dataclass(frozen=True, slots=True)
class SubjectRegistryConfig:
    """Explicit, uncalibrated registry policy.

    There is no field with a "sensible" hidden default here on purpose: gap
    tolerance and qualification thresholds depend entirely on camera frame
    rate, hall layout and tracker quality, none of which this service can
    assume.
    """

    #: Consecutive-ish observed frames a new raw track needs before it earns a
    #: subject. Prevents flicker detections from creating S001..S099.
    min_frames_to_qualify: int
    #: Wall-clock persistence a new raw track needs before it earns a subject.
    min_seconds_to_qualify: float
    #: Longest gap after which a lost subject may still reclaim a raw track.
    short_gap_seconds: float
    #: Gap after which a subject is reported as TEMPORARILY_LOST.
    lost_after_seconds: float
    #: Gap after which the subject is closed (ENDED) for this session.
    end_after_seconds: float
    #: Minimum spatial score for reassociation to be accepted at all.
    reassociation_min_confidence: float
    #: The winner must beat the runner-up by at least this margin.
    reassociation_margin: float
    #: Weight of the newest observation when updating the observation anchor.
    anchor_smoothing: float
    #: Frames a pending raw track may miss before its qualification progress
    #: is discarded.
    pending_gap_seconds: float

    def __post_init__(self) -> None:
        if self.min_frames_to_qualify < 1:
            raise ValueError("min_frames_to_qualify must be >= 1")
        for name in (
            "min_seconds_to_qualify",
            "short_gap_seconds",
            "lost_after_seconds",
            "end_after_seconds",
            "pending_gap_seconds",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("reassociation_min_confidence", "reassociation_margin", "anchor_smoothing"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within 0..1")
        if self.lost_after_seconds > self.short_gap_seconds:
            raise ValueError("lost_after_seconds must not exceed short_gap_seconds")
        if self.end_after_seconds < self.short_gap_seconds:
            raise ValueError("end_after_seconds must be >= short_gap_seconds")


@dataclass(frozen=True, slots=True)
class TrackSegment:
    """One continuous stretch of raw-track ownership by a subject."""

    raw_tracking_id: str
    started_at: datetime
    method: AssociationMethod
    association_confidence: Optional[float] = None
    ended_at: Optional[datetime] = None

    @property
    def is_open(self) -> bool:
        return self.ended_at is None


@dataclass(frozen=True, slots=True)
class SubjectSnapshot:
    """Immutable read-only view of one subject at one moment."""

    subject_number: int
    label: str
    status: SubjectTrackingStatus
    first_seen_at: datetime
    last_seen_at: datetime
    ended_at: Optional[datetime]
    active_tracking_id: Optional[str]
    anchor: Optional[BBox]
    anchor_updated_at: Optional[datetime]
    reassociation_count: int
    last_association_confidence: Optional[float]
    segments: tuple[TrackSegment, ...]


@dataclass(frozen=True, slots=True)
class SubjectEvent:
    """Something that actually happened to a subject in one frame."""

    kind: SubjectEventKind
    subject_number: int
    label: str
    at: datetime
    tracking_id: Optional[str] = None
    method: Optional[AssociationMethod] = None
    association_confidence: Optional[float] = None
    previous_status: Optional[SubjectTrackingStatus] = None
    status: Optional[SubjectTrackingStatus] = None
    reason: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ReassociationCandidate:
    subject_number: int
    score: float


@dataclass(frozen=True, slots=True)
class ReassociationDecision:
    """Complete, diagnosable outcome of one reassociation attempt."""

    raw_tracking_id: str
    accepted: bool
    subject_number: Optional[int]
    score: Optional[float]
    runner_up_score: Optional[float]
    reason: str
    candidates: tuple[ReassociationCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class SubjectFrameResult:
    """What one analysed frame did to the subject registry."""

    exam_session_id: str
    camera_id: str
    observed_at: datetime
    subjects: tuple[SubjectSnapshot, ...] = ()
    events: tuple[SubjectEvent, ...] = ()
    decisions: tuple[ReassociationDecision, ...] = ()

    @property
    def active_subject_count(self) -> int:
        return sum(1 for item in self.subjects if item.status is not SubjectTrackingStatus.ENDED)


@dataclass
class PendingTrack:
    """A raw track that has not yet earned an anonymous subject."""

    raw_tracking_id: str
    first_seen_at: datetime
    last_seen_at: datetime
    bbox: BBox
    frames: int = 1

    def qualifies(self, config: SubjectRegistryConfig) -> bool:
        observed = (self.last_seen_at - self.first_seen_at).total_seconds()
        return self.frames >= config.min_frames_to_qualify and observed >= config.min_seconds_to_qualify


@dataclass
class SubjectState:
    """Mutable per-subject bookkeeping owned by the registry only."""

    subject_number: int
    first_seen_at: datetime
    last_seen_at: datetime
    anchor: BBox
    anchor_updated_at: datetime
    status: SubjectTrackingStatus = SubjectTrackingStatus.STABLE
    active_tracking_id: Optional[str] = None
    ended_at: Optional[datetime] = None
    reassociation_count: int = 0
    last_association_confidence: Optional[float] = None
    segments: list[TrackSegment] = field(default_factory=list)

    @property
    def label(self) -> str:
        return subject_label(self.subject_number)

    def snapshot(self) -> SubjectSnapshot:
        return SubjectSnapshot(
            subject_number=self.subject_number,
            label=self.label,
            status=self.status,
            first_seen_at=self.first_seen_at,
            last_seen_at=self.last_seen_at,
            ended_at=self.ended_at,
            active_tracking_id=self.active_tracking_id,
            anchor=self.anchor,
            anchor_updated_at=self.anchor_updated_at,
            reassociation_count=self.reassociation_count,
            last_association_confidence=self.last_association_confidence,
            segments=tuple(self.segments),
        )


def blend_anchor(previous: BBox, observed: BBox, smoothing: float) -> BBox:
    """Exponentially smoothed *observation region* — never a seat assignment.

    The anchor is only "where this subject has recently been observed". It is
    advisory spatial context for short-gap recovery and carries no claim about
    seating, ownership of a place, or physical registration.
    """
    weight = min(1.0, max(0.0, float(smoothing)))
    keep = 1.0 - weight
    return BBox(
        clamp01(previous.x * keep + observed.x * weight),
        clamp01(previous.y * keep + observed.y * weight),
        clamp01(previous.width * keep + observed.width * weight),
        clamp01(previous.height * keep + observed.height * weight),
    )


def spatial_recovery_score(anchor: BBox, observed: BBox) -> float:
    """0..1 plausibility that ``observed`` continues the anchored subject.

    Purely geometric: overlap plus size-normalized centre proximity. No
    appearance, colour, clothing or face features are used anywhere.
    """
    overlap = iou(anchor, observed)
    distance = normalized_distance(observed.center, anchor)
    proximity = 0.0 if distance >= 1.5 else max(0.0, 1.0 - distance / 1.5)
    return round(min(1.0, max(0.0, 0.6 * overlap + 0.4 * proximity)), 6)
