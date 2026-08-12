"""Immutable contract for the TEMPORAL interpretation of Task 3C pair geometry.

Scope and honesty rules (Task 3D):

* This layer interprets ALREADY-DERIVED same-frame person-pair geometry over
  time. It never re-derives persons, wrists, pose or pair geometry.
* A completed sequence means EXACTLY ONE thing: for one camera, one stream
  generation, one rule, one tracked-person pair and ONE locked wrist-side
  combination, the configured geometric sequence
  ``approach -> near-interaction dwell -> separation`` was observed using the
  source frames' own timestamps.
* It is NOT proof of paper, a document, an object transfer, cheating, contact,
  touching, or of who gave anything to whom. A plain visual handshake produces
  the very same approach -> near -> separation geometry, so this state machine
  alone can NEVER distinguish a handshake from a paper handoff. Strengthening
  that decision is the responsibility of future object evidence plus exam
  context, not of this module.
* There is no paper/document/sheet detector in this system, so no
  ``paper_present``-style field exists here, and none may be added.
* No probability, no fused "confidence" score, no giver/receiver field.

Explicit arming (mandatory): temporal processing only ever happens when the
caller passes ``armed=True``. Construction of the tracker, and starting a
camera, arm nothing. While disarmed, no candidate history is created or kept,
so legitimate exam paper distribution observed before arming can never
contribute to a later completion.

Future integration requirement (deliberately NOT implemented here): before any
Document Exchange event may be emitted, an exam-session layer must supply a
student roster, seat/student resolution, staff/invigilator roles and
student-to-student eligibility, so that a lecturer legitimately handing an extra
paper mid-exam is excluded by policy. That roster logic must NOT be coupled into
this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Union

from .body_features import BodySide
from .pair_geometry import PersonPairKey

#: A stream generation / incarnation identifier is always supplied by the caller
#: and is never invented from frame numbers.
StreamGeneration = Union[int, str]


class HandoffTemporalContractError(ValueError):
    """Raised when a temporal configuration or fact would violate its invariants."""


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive(label: str, value: object) -> float:
    if not _finite(value) or float(value) <= 0.0:
        raise HandoffTemporalContractError(f"{label} must be a finite positive number")
    return float(value)


def _non_negative(label: str, value: object) -> float:
    if not _finite(value) or float(value) < 0.0:
        raise HandoffTemporalContractError(
            f"{label} must be a finite non-negative number"
        )
    return float(value)


def validate_stream_generation(value: object) -> StreamGeneration:
    """Caller-supplied generation identity: a real int or a non-blank string."""
    if isinstance(value, bool):
        raise HandoffTemporalContractError("stream_generation must not be a bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return value
    raise HandoffTemporalContractError(
        "stream_generation must be an int or a non-blank string"
    )


class HandoffPhase(str, Enum):
    """Explicit temporal phases. Neutral names only, no behaviour vocabulary."""

    IDLE = "idle"
    APPROACHING = "approaching"
    INTERACTION = "interaction"
    SEPARATING = "separating"
    COMPLETED = "completed"


class HandoffTemporalStatus(str, Enum):
    """Outcome of ONE temporal processing call for ONE source frame."""

    #: Armed and the frame was a usable temporal continuation.
    OK = "ok"
    #: The caller did not arm monitoring: nothing was processed or retained.
    DISARMED = "disarmed"
    #: Pose/association was unavailable upstream: unknown evidence, no advance.
    DEGRADED_FRAME = "degraded_frame"
    #: Duplicate / out-of-order / non-increasing frame input: state reset.
    NON_MONOTONIC = "non_monotonic"
    #: Frame lacked the metadata required for timestamp-based temporal logic.
    INVALID_INPUT = "invalid_input"


#: Deterministic abort/reset reasons (raw diagnostics, never user-facing alerts).
ABORT_EVIDENCE_GAP_EXCEEDED = "evidence_gap_exceeded"
ABORT_APPROACH_LOST = "approach_lost"
ABORT_APPROACH_TIMEOUT = "approach_timeout"
ABORT_INTERACTION_DWELL_TOO_SHORT = "interaction_dwell_too_short"
RESET_RECOVERED = "recovered_after_completion"
RESET_DISARMED = "disarmed"
RESET_NON_MONOTONIC = "non_monotonic_frame"


@dataclass(frozen=True, slots=True)
class HandoffTemporalSpec:
    """Caller-supplied temporal/geometric configuration. NO production defaults.

    Every distance is a wrist-pair distance normalized by the pair's MEAN person
    diagonal, exactly as produced by Task 3C
    (``distance_relative_to_mean_person_diagonal``). The metric is symmetric on
    purpose: the pair is symmetric and direction is unknown. It is not pixels
    and not a physical distance.

    Durations are seconds measured between source-frame ``observed_at`` values.
    No frame-count rule exists anywhere: "3 frames" means different things at
    1 FPS and 10 FPS.
    """

    #: Pair eligibility: person-centre distance relative to mean person diagonal.
    max_person_center_distance: float
    #: Wrist distance at or below which a candidate may start approaching.
    approach_start_wrist_distance: float
    #: Stricter wrist distance defining the near-interaction criterion.
    interaction_wrist_distance: float
    #: Required measured reduction of the locked wrist distance while approaching.
    min_approach_distance_reduction: float
    #: Required observed near-interaction dwell.
    min_interaction_dwell_seconds: float
    #: Tolerated unknown-evidence (occlusion/absent/degraded) gap.
    max_unknown_gap_seconds: float
    #: Wrist distance that must be reached again after the interaction.
    min_separation_wrist_distance: float
    #: Required increase from the closest observed distance to count as separation.
    min_separation_distance_increase: float
    #: Required observed separation dwell.
    min_separation_dwell_seconds: float
    #: Distance required to release a completed (latched) candidate.
    recovery_wrist_distance: float
    #: Dwell required at recovery distance before a new sequence may ever start.
    recovery_dwell_seconds: float
    #: Optional cap on how long a candidate may stay in APPROACHING.
    max_approach_duration_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        _positive("max_person_center_distance", self.max_person_center_distance)
        _positive("approach_start_wrist_distance", self.approach_start_wrist_distance)
        _positive("interaction_wrist_distance", self.interaction_wrist_distance)
        _positive(
            "min_approach_distance_reduction", self.min_approach_distance_reduction
        )
        _positive("min_interaction_dwell_seconds", self.min_interaction_dwell_seconds)
        _non_negative("max_unknown_gap_seconds", self.max_unknown_gap_seconds)
        _positive("min_separation_wrist_distance", self.min_separation_wrist_distance)
        _positive(
            "min_separation_distance_increase", self.min_separation_distance_increase
        )
        _positive("min_separation_dwell_seconds", self.min_separation_dwell_seconds)
        _positive("recovery_wrist_distance", self.recovery_wrist_distance)
        _positive("recovery_dwell_seconds", self.recovery_dwell_seconds)
        if self.max_approach_duration_seconds is not None:
            _positive(
                "max_approach_duration_seconds", self.max_approach_duration_seconds
            )

        # Only genuinely logical relationships are enforced. No deployment values
        # are recommended or implied here.
        if self.interaction_wrist_distance >= self.approach_start_wrist_distance:
            raise HandoffTemporalContractError(
                "interaction_wrist_distance must be strictly smaller than "
                "approach_start_wrist_distance"
            )
        if self.min_separation_wrist_distance <= self.interaction_wrist_distance:
            raise HandoffTemporalContractError(
                "min_separation_wrist_distance must be strictly larger than "
                "interaction_wrist_distance"
            )
        if self.recovery_wrist_distance < self.min_separation_wrist_distance:
            raise HandoffTemporalContractError(
                "recovery_wrist_distance must not be logically inside the "
                "separation/interaction geometry"
            )


@dataclass(frozen=True, slots=True)
class HandoffTemporalResult:
    """Immutable temporal facts for ONE pair candidate on ONE processed frame.

    ``completed_this_frame`` is a one-frame transition fact, not an EventDraft
    and not an alert. Nothing here is published anywhere.
    """

    camera_id: str
    stream_generation: StreamGeneration
    rule_id: str
    pair_key: PersonPairKey
    phase: HandoffPhase
    locked_side_a: BodySide
    locked_side_b: BodySide
    candidate_started_at: datetime
    last_valid_evidence_at: datetime
    evidence_available_this_frame: bool
    closest_wrist_distance: float
    current_wrist_distance: Optional[float] = None
    approach_distance_reduction: float = 0.0
    separation_distance_increase: float = 0.0
    interaction_started_at: Optional[datetime] = None
    separation_started_at: Optional[datetime] = None
    #: GENUINELY OBSERVED accumulated dwell, not wall time: tolerated UNKNOWN /
    #: degraded / missing-pair intervals pause accumulation and are never credited.
    interaction_duration_seconds: float = 0.0
    separation_duration_seconds: float = 0.0

    completed_this_frame: bool = False
    completed_at: Optional[datetime] = None
    abort_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise HandoffTemporalContractError("camera_id must be a non-blank string")
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise HandoffTemporalContractError("rule_id must be a non-blank string")
        validate_stream_generation(self.stream_generation)
        if not isinstance(self.pair_key, PersonPairKey):
            raise HandoffTemporalContractError("pair_key must be a PersonPairKey")
        if not isinstance(self.phase, HandoffPhase):
            raise HandoffTemporalContractError("phase must be a HandoffPhase")
        for label, value in (
            ("locked_side_a", self.locked_side_a),
            ("locked_side_b", self.locked_side_b),
        ):
            if not isinstance(value, BodySide):
                raise HandoffTemporalContractError(f"{label} must be a BodySide")
        for label, value in (
            ("candidate_started_at", self.candidate_started_at),
            ("last_valid_evidence_at", self.last_valid_evidence_at),
        ):
            if not isinstance(value, datetime):
                raise HandoffTemporalContractError(f"{label} must be a datetime")
        for label, value in (
            ("interaction_started_at", self.interaction_started_at),
            ("separation_started_at", self.separation_started_at),
            ("completed_at", self.completed_at),
        ):
            if value is not None and not isinstance(value, datetime):
                raise HandoffTemporalContractError(f"{label} must be a datetime")
        if type(self.evidence_available_this_frame) is not bool:
            raise HandoffTemporalContractError(
                "evidence_available_this_frame must be a real bool"
            )
        if type(self.completed_this_frame) is not bool:
            raise HandoffTemporalContractError("completed_this_frame must be a real bool")
        _non_negative("closest_wrist_distance", self.closest_wrist_distance)
        if self.current_wrist_distance is not None:
            _non_negative("current_wrist_distance", self.current_wrist_distance)
        _non_negative("approach_distance_reduction", self.approach_distance_reduction)
        _non_negative("separation_distance_increase", self.separation_distance_increase)
        _non_negative(
            "interaction_duration_seconds", self.interaction_duration_seconds
        )
        _non_negative("separation_duration_seconds", self.separation_duration_seconds)
        if self.abort_reason is not None and (
            not isinstance(self.abort_reason, str) or not self.abort_reason.strip()
        ):
            raise HandoffTemporalContractError(
                "abort_reason must be None or a non-blank string"
            )
        if self.completed_this_frame and self.phase is not HandoffPhase.COMPLETED:
            raise HandoffTemporalContractError(
                "completed_this_frame requires the COMPLETED phase"
            )
        if self.completed_this_frame and self.completed_at is None:
            raise HandoffTemporalContractError(
                "a completion must carry its completed_at timestamp"
            )


@dataclass(frozen=True, slots=True)
class HandoffTemporalFrameResult:
    """All temporal candidate facts produced from ONE source frame."""

    status: HandoffTemporalStatus
    camera_id: Optional[str] = None
    stream_generation: Optional[StreamGeneration] = None
    rule_id: Optional[str] = None
    frame_sequence: Optional[int] = None
    observed_at: Optional[datetime] = None
    candidates: tuple[HandoffTemporalResult, ...] = ()
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, HandoffTemporalStatus):
            raise HandoffTemporalContractError("status must be a HandoffTemporalStatus")
        if not isinstance(self.candidates, tuple):
            raise HandoffTemporalContractError("candidates must be an immutable tuple")
        for item in self.candidates:
            if not isinstance(item, HandoffTemporalResult):
                raise HandoffTemporalContractError(
                    "candidates must contain HandoffTemporalResult values"
                )
        if self.status is not HandoffTemporalStatus.OK:
            for item in self.candidates:
                if item.completed_this_frame:
                    raise HandoffTemporalContractError(
                        "a non-OK temporal frame can never carry a completion"
                    )

    @property
    def completed(self) -> tuple[HandoffTemporalResult, ...]:
        return tuple(item for item in self.candidates if item.completed_this_frame)

    def candidate(self, key: PersonPairKey) -> Optional[HandoffTemporalResult]:
        for item in self.candidates:
            if item.pair_key == key:
                return item
        return None
