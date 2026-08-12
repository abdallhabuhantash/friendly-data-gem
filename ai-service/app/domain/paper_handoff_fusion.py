"""Immutable contract for TASK 3G — paper-corroborated handoff fusion.

Scope and honesty rules:

* This layer FUSES two already-derived, authoritative immutable inputs:
  Task 3D ``HandoffTemporalResult`` (temporal handoff-like pattern) and Task 3F
  ``PaperPairSpatialFrame`` (same-frame paper <-> person-pair geometry). It never
  re-derives pose, wrists, pair geometry or paper detections.
* A fused completion means EXACTLY ONE thing: for one camera, one stream
  generation, one rule, one canonical tracked-person pair and ONE Task 3D
  temporal candidate with ONE locked wrist-side combination, the configured
  handoff-like temporal sequence completed AND paper-like detector evidence was
  observed sufficiently near the locked wrist geometry of that same pair during
  that same temporal candidate, according to explicit caller-supplied criteria.
* It does NOT mean: a confirmed paper transfer, ownership change, physical
  contact, grasping, a giver, a receiver, or cheating. There is no paper
  tracking identity anywhere in this system, so it never means "the same sheet
  was followed from A to B".
* No fused confidence, no probability, no score. ``min_paper_detector_confidence``
  constrains ONLY the raw paper-detector confidence and is never combined with
  pose, keypoint, person, duration or distance values.
* Distances are Task 3F's normalized paper<->wrist distance relative to the mean
  pair diagonal. Not pixels, not physical distance, never contact.
* Dormant by design: no EventDraft, EventPublisher, snapshot, notification,
  Supabase, frontend, orchestrator or engine-registry coupling, and no threads.
* Fusion correctness is NOT paper-detector real-world accuracy. The
  open-vocabulary paper detector still requires offline real-video acceptance
  before any runtime or event integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from .body_features import BodySide
from .handoff_temporal import (
    HandoffPhase,
    StreamGeneration,
    validate_stream_generation,
)
from .pair_geometry import PersonPairKey


class PaperHandoffFusionContractError(ValueError):
    """Raised when a fusion configuration or fact would violate its invariants."""


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive(label: str, value: object) -> float:
    if not _finite(value) or float(value) <= 0.0:
        raise PaperHandoffFusionContractError(
            f"{label} must be a finite positive number"
        )
    return float(value)


def _non_negative(label: str, value: object) -> float:
    if not _finite(value) or float(value) < 0.0:
        raise PaperHandoffFusionContractError(
            f"{label} must be a finite non-negative number"
        )
    return float(value)


def _non_blank(label: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PaperHandoffFusionContractError(f"{label} must be a non-blank string")
    return value


def _strict_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class PaperSupportMode(str, Enum):
    """Explicit caller-chosen spatial support policy. Never inferred."""

    #: Paper must qualify near AT LEAST ONE of the two locked wrists.
    EITHER_LOCKED_WRIST = "either_locked_wrist"
    #: Paper must qualify near BOTH locked wrists on the same frame.
    BOTH_LOCKED_WRISTS = "both_locked_wrists"


class PaperSupportStatus(str, Enum):
    """Categorical GEOMETRIC evidence state for ONE processed frame.

    These are spatial evidence states only. None of them means holder,
    ownership, transfer or grasping.
    """

    #: Valid paper-evidence frame, no detection passed the confidence floor
    #: (including a valid frame with zero paper detections). Negative detector
    #: evidence, NOT unknown.
    NONE = "none"
    #: Qualifying paper evidence near the locked wrist of pair member A.
    NEAR_LOCKED_WRIST_A = "near_locked_wrist_a"
    #: Qualifying paper evidence near the locked wrist of pair member B.
    NEAR_LOCKED_WRIST_B = "near_locked_wrist_b"
    #: Qualifying paper evidence near BOTH locked wrists.
    NEAR_BOTH_LOCKED_WRISTS = "near_both_locked_wrists"
    #: Confidence-passing paper evidence exists but is not sufficiently near the
    #: locked wrist geometry of this pair. Never counts as support.
    PAPER_PRESENT_BUT_NOT_CORRELATED = "paper_present_but_not_correlated"
    #: Paper detector degraded, Task 3F degraded, paper frame unavailable, or a
    #: required locked-wrist spatial fact unavailable. NOT negative proof.
    UNKNOWN = "unknown"


class PaperHandoffFusionStatus(str, Enum):
    """Outcome of ONE fusion call for ONE temporal candidate on ONE frame."""

    OK = "ok"
    #: The caller did not arm monitoring: nothing accumulated, state cleared.
    DISARMED = "disarmed"
    #: Duplicate / out-of-order / non-increasing fusion frame input: state reset.
    NON_MONOTONIC = "non_monotonic"
    #: The two contracts could not be proven to describe the same authoritative
    #: camera / generation / rule / pair / candidate / frame.
    INCONSISTENT_INPUT = "inconsistent_input"


#: Deterministic non-qualification / abort reasons (raw diagnostics only).
ABORT_PAPER_UNKNOWN_GAP_EXCEEDED = "paper_unknown_gap_exceeded"
NOT_QUALIFIED_INSUFFICIENT_INTERACTION_PAPER = "insufficient_interaction_paper_seconds"
NOT_QUALIFIED_INSUFFICIENT_TOTAL_PAPER = "insufficient_total_paper_seconds"
CLOSED_ALREADY_DECIDED = "candidate_already_decided"
RESET_DISARMED = "disarmed"
RESET_NON_MONOTONIC = "non_monotonic_frame"
RESET_NEW_CANDIDATE = "new_temporal_candidate"


@dataclass(frozen=True, slots=True)
class PaperHandoffFusionSpec:
    """Caller-supplied fusion configuration. NO behavioural production defaults.

    Every field must be supplied explicitly, including ``support_mode``: there is
    no silent policy fallback between EITHER and BOTH locked wrists.
    """

    #: Applies ONLY to the raw paper-detector confidence (never fused).
    min_paper_detector_confidence: float
    #: Task 3F normalized paper<->wrist distance relative to the mean pair
    #: diagonal at or below which paper counts as near a locked wrist.
    max_paper_to_locked_wrist_distance_relative_to_mean_diagonal: float
    #: Genuinely OBSERVED qualifying paper seconds required while Task 3D is in
    #: its INTERACTION phase.
    min_interaction_paper_observed_seconds: float
    #: Genuinely OBSERVED qualifying paper seconds required across the candidate.
    min_total_paper_observed_seconds: float
    #: Tolerated UNKNOWN paper-evidence gap inside an active candidate.
    max_paper_unknown_gap_seconds: float
    #: Explicit spatial support policy.
    support_mode: PaperSupportMode

    def __post_init__(self) -> None:
        confidence = self.min_paper_detector_confidence
        if not _finite(confidence) or not 0.0 <= float(confidence) <= 1.0:
            raise PaperHandoffFusionContractError(
                "min_paper_detector_confidence must be finite within 0..1"
            )
        _positive(
            "max_paper_to_locked_wrist_distance_relative_to_mean_diagonal",
            self.max_paper_to_locked_wrist_distance_relative_to_mean_diagonal,
        )
        _positive(
            "min_interaction_paper_observed_seconds",
            self.min_interaction_paper_observed_seconds,
        )
        _positive(
            "min_total_paper_observed_seconds", self.min_total_paper_observed_seconds
        )
        _non_negative(
            "max_paper_unknown_gap_seconds", self.max_paper_unknown_gap_seconds
        )
        if not isinstance(self.support_mode, PaperSupportMode):
            raise PaperHandoffFusionContractError(
                "support_mode must be an explicit PaperSupportMode"
            )
        if (
            float(self.min_total_paper_observed_seconds)
            < float(self.min_interaction_paper_observed_seconds)
        ):
            raise PaperHandoffFusionContractError(
                "min_total_paper_observed_seconds must not be smaller than "
                "min_interaction_paper_observed_seconds"
            )


@dataclass(frozen=True, slots=True)
class TemporalCandidateIdentity:
    """Deterministic identity of ONE Task 3D temporal candidate.

    Contains no permanent person identity: tracking ids are per-generation
    tracker labels, and a tracking-id change is simply a different candidate.
    """

    camera_id: str
    stream_generation: StreamGeneration
    rule_id: str
    pair_key: PersonPairKey
    locked_side_a: BodySide
    locked_side_b: BodySide
    candidate_started_at: datetime

    def __post_init__(self) -> None:
        _non_blank("camera_id", self.camera_id)
        _non_blank("rule_id", self.rule_id)
        validate_stream_generation(self.stream_generation)
        if not isinstance(self.pair_key, PersonPairKey):
            raise PaperHandoffFusionContractError("pair_key must be a PersonPairKey")
        for label, value in (
            ("locked_side_a", self.locked_side_a),
            ("locked_side_b", self.locked_side_b),
        ):
            if not isinstance(value, BodySide):
                raise PaperHandoffFusionContractError(f"{label} must be a BodySide")
        if not isinstance(self.candidate_started_at, datetime):
            raise PaperHandoffFusionContractError(
                "candidate_started_at must be a datetime"
            )


@dataclass(frozen=True, slots=True)
class PaperHandoffFusionJoin:
    """EXPLICIT cross-contract provenance declared by the caller.

    Task 3D and Task 3F expose different identity fields: Task 3D carries
    camera / generation / rule / pair / candidate facts but no paper counters,
    while Task 3F carries an explicitly declared ``pair_frame_sequence`` and
    ``paper_frame_index`` plus the pair-side absolute ``pair_observed_at``.
    Nothing is inferred from matching numeric counters: this object IS the
    caller's declaration that both inputs describe the same authoritative source
    frame, and every declared field is validated against its own contract.
    """

    camera_id: str
    stream_generation: StreamGeneration
    rule_id: str
    pair_key: PersonPairKey
    pair_frame_sequence: int
    #: Absolute pair-pipeline observation time; the ONLY clock used for dwell.
    pair_observed_at: datetime
    candidate_started_at: datetime
    locked_side_a: BodySide
    locked_side_b: BodySide

    def __post_init__(self) -> None:
        _non_blank("camera_id", self.camera_id)
        _non_blank("rule_id", self.rule_id)
        validate_stream_generation(self.stream_generation)
        if not isinstance(self.pair_key, PersonPairKey):
            raise PaperHandoffFusionContractError("pair_key must be a PersonPairKey")
        if not _strict_index(self.pair_frame_sequence):
            raise PaperHandoffFusionContractError(
                "pair_frame_sequence must be a non-negative int"
            )
        for label, value in (
            ("pair_observed_at", self.pair_observed_at),
            ("candidate_started_at", self.candidate_started_at),
        ):
            if not isinstance(value, datetime):
                raise PaperHandoffFusionContractError(f"{label} must be a datetime")
        for label, value in (
            ("locked_side_a", self.locked_side_a),
            ("locked_side_b", self.locked_side_b),
        ):
            if not isinstance(value, BodySide):
                raise PaperHandoffFusionContractError(f"{label} must be a BodySide")

    @property
    def identity(self) -> TemporalCandidateIdentity:
        return TemporalCandidateIdentity(
            camera_id=self.camera_id,
            stream_generation=self.stream_generation,
            rule_id=self.rule_id,
            pair_key=self.pair_key,
            locked_side_a=self.locked_side_a,
            locked_side_b=self.locked_side_b,
            candidate_started_at=self.candidate_started_at,
        )


@dataclass(frozen=True, slots=True)
class PaperHandoffFusionResult:
    """Immutable raw fusion / calibration facts for ONE frame, ONE candidate.

    ``fused_completed_this_frame`` is a one-frame transition fact. It is not an
    EventDraft, not an alert and is published nowhere. The eventual human-facing
    wording would be "Possible Paper Exchange", never "Cheating Detected".
    """

    status: PaperHandoffFusionStatus
    identity: Optional[TemporalCandidateIdentity] = None
    pair_frame_sequence: Optional[int] = None
    pair_observed_at: Optional[datetime] = None
    temporal_phase: Optional[HandoffPhase] = None
    paper_support_status: PaperSupportStatus = PaperSupportStatus.UNKNOWN
    support_qualified_this_frame: bool = False
    #: Frame-local index only. Detection indexes are NOT stable across frames and
    #: the selected detection may differ per frame, so this never claims that one
    #: physical sheet was tracked.
    qualifying_paper_detection_index: Optional[int] = None
    qualifying_paper_raw_prompt: Optional[str] = None
    qualifying_paper_confidence: Optional[float] = None
    paper_distance_to_locked_wrist_a: Optional[float] = None
    paper_distance_to_locked_wrist_b: Optional[float] = None
    observed_interaction_paper_seconds: float = 0.0
    observed_total_paper_seconds: float = 0.0
    last_valid_paper_evidence_at: Optional[datetime] = None
    paper_unknown_gap_seconds: float = 0.0
    temporal_completed_this_frame: bool = False
    fused_completed_this_frame: bool = False
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PaperHandoffFusionStatus):
            raise PaperHandoffFusionContractError(
                "status must be a PaperHandoffFusionStatus"
            )
        if self.identity is not None and not isinstance(
            self.identity, TemporalCandidateIdentity
        ):
            raise PaperHandoffFusionContractError(
                "identity must be a TemporalCandidateIdentity"
            )
        if self.pair_frame_sequence is not None and not _strict_index(
            self.pair_frame_sequence
        ):
            raise PaperHandoffFusionContractError(
                "pair_frame_sequence must be a non-negative int"
            )
        for label, value in (
            ("pair_observed_at", self.pair_observed_at),
            ("last_valid_paper_evidence_at", self.last_valid_paper_evidence_at),
        ):
            if value is not None and not isinstance(value, datetime):
                raise PaperHandoffFusionContractError(f"{label} must be a datetime")
        if self.temporal_phase is not None and not isinstance(
            self.temporal_phase, HandoffPhase
        ):
            raise PaperHandoffFusionContractError(
                "temporal_phase must be a HandoffPhase"
            )
        if not isinstance(self.paper_support_status, PaperSupportStatus):
            raise PaperHandoffFusionContractError(
                "paper_support_status must be a PaperSupportStatus"
            )
        for label, value in (
            ("support_qualified_this_frame", self.support_qualified_this_frame),
            ("temporal_completed_this_frame", self.temporal_completed_this_frame),
            ("fused_completed_this_frame", self.fused_completed_this_frame),
        ):
            if type(value) is not bool:
                raise PaperHandoffFusionContractError(f"{label} must be a real bool")
        if self.qualifying_paper_detection_index is not None and not _strict_index(
            self.qualifying_paper_detection_index
        ):
            raise PaperHandoffFusionContractError(
                "qualifying_paper_detection_index must be a non-negative int"
            )
        if self.qualifying_paper_raw_prompt is not None:
            _non_blank("qualifying_paper_raw_prompt", self.qualifying_paper_raw_prompt)
        if self.qualifying_paper_confidence is not None:
            value = self.qualifying_paper_confidence
            if not _finite(value) or not 0.0 <= float(value) <= 1.0:
                raise PaperHandoffFusionContractError(
                    "qualifying_paper_confidence must be finite within 0..1"
                )
        for label, value in (
            ("paper_distance_to_locked_wrist_a", self.paper_distance_to_locked_wrist_a),
            ("paper_distance_to_locked_wrist_b", self.paper_distance_to_locked_wrist_b),
        ):
            if value is not None:
                _non_negative(label, value)
        _non_negative(
            "observed_interaction_paper_seconds", self.observed_interaction_paper_seconds
        )
        _non_negative(
            "observed_total_paper_seconds", self.observed_total_paper_seconds
        )
        _non_negative("paper_unknown_gap_seconds", self.paper_unknown_gap_seconds)
        if self.reason is not None:
            _non_blank("reason", self.reason)
        if self.status is not PaperHandoffFusionStatus.OK and (
            self.fused_completed_this_frame or self.support_qualified_this_frame
        ):
            raise PaperHandoffFusionContractError(
                "a non-OK fusion frame can never qualify or complete"
            )
        if self.fused_completed_this_frame and not self.temporal_completed_this_frame:
            raise PaperHandoffFusionContractError(
                "a fused completion requires the Task 3D completion transition"
            )
        if self.fused_completed_this_frame and self.identity is None:
            raise PaperHandoffFusionContractError(
                "a fused completion must carry its candidate identity"
            )
