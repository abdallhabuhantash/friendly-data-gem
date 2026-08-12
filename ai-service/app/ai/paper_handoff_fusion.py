"""TASK 3G — paper-corroborated handoff fusion (PURE / DORMANT).

Consumes Task 3D (``HandoffTemporalResult``) and Task 3F
(``PaperPairSpatialFrame``) READ-ONLY. Neither
``app/ai/exchange_temporal_state.py`` nor ``app/ai/paper_pair_spatial_builder.py``
is modified, and no input object is ever mutated.

What a fused completion means: the configured handoff-like temporal pattern
completed for one canonical tracked-person pair, and paper-like detector evidence
was genuinely OBSERVED sufficiently near the LOCKED wrist geometry of that same
pair during that same temporal candidate.

What it never means: confirmed paper transfer, ownership, contact, grasping, a
giver, a receiver, or cheating. There is no paper tracking identity, so evidence
across frames never proves that one physical sheet was followed.

Dormant: no orchestrator/engine-registry wiring, no EventDraft, no
EventPublisher, no snapshots, no notifications, no Supabase, no frontend, no
threads. Fusion tests passing does NOT make the paper detector production ready:
offline real-video acceptance of the open-vocabulary paper detector is still
required before any runtime or event integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.domain.handoff_temporal import (
    ABORT_APPROACH_LOST,
    ABORT_APPROACH_TIMEOUT,
    ABORT_EVIDENCE_GAP_EXCEEDED,
    ABORT_INTERACTION_DWELL_TOO_SHORT,
    RESET_DISARMED as TEMPORAL_RESET_DISARMED,
    RESET_NON_MONOTONIC as TEMPORAL_RESET_NON_MONOTONIC,
    RESET_RECOVERED,
    HandoffPhase,
    HandoffTemporalResult,
)
from app.domain.pair_geometry import PersonPairKey
from app.domain.paper_handoff_fusion import (
    ABORT_PAPER_UNKNOWN_GAP_EXCEEDED,
    CLOSED_ALREADY_DECIDED,
    NOT_QUALIFIED_INSUFFICIENT_INTERACTION_PAPER,
    NOT_QUALIFIED_INSUFFICIENT_TOTAL_PAPER,
    RESET_DISARMED,
    RESET_NEW_CANDIDATE,
    RESET_NON_MONOTONIC,
    PaperHandoffFusionContractError,
    PaperHandoffFusionJoin,
    PaperHandoffFusionResult,
    PaperHandoffFusionSpec,
    PaperHandoffFusionStatus,
    PaperSupportMode,
    PaperSupportStatus,
    TemporalCandidateIdentity,
)
from app.domain.paper_pair_spatial import (
    PaperPairSpatialFrame,
    PaperPairSpatialStatus,
)

#: Full fusion state identity. A change in ANY member is different state:
#: cameras, stream incarnations, rules and pairs never share or migrate state.
StateKey = tuple[str, object, str, PersonPairKey]

#: Task 3D reasons that TERMINATE that exact temporal candidate. Task 3D remains
#: authoritative for the lifecycle: this layer only mirrors it and never decides
#: on its own that a candidate ended.
TERMINAL_TEMPORAL_REASONS: frozenset[str] = frozenset(
    {
        ABORT_EVIDENCE_GAP_EXCEEDED,
        ABORT_APPROACH_LOST,
        ABORT_APPROACH_TIMEOUT,
        ABORT_INTERACTION_DWELL_TOO_SHORT,
        RESET_RECOVERED,
        TEMPORAL_RESET_DISARMED,
        TEMPORAL_RESET_NON_MONOTONIC,
    }
)
ContextKey = tuple[str, str]


def _seconds(later: datetime, earlier: datetime) -> float:
    return max(0.0, (later - earlier).total_seconds())


@dataclass
class _Evaluation:
    """Frame-local paper support evaluation. Bounded scalars only."""

    status: PaperSupportStatus
    qualified: bool = False
    detection_index: Optional[int] = None
    raw_prompt: Optional[str] = None
    confidence: Optional[float] = None
    distance_a: Optional[float] = None
    distance_b: Optional[float] = None


@dataclass
class _FusionCandidate:
    """PRIVATE mutable per-candidate state. Bounded accumulators, no history."""

    identity: TemporalCandidateIdentity
    observed_total_seconds: float = 0.0
    observed_interaction_seconds: float = 0.0
    #: Anchor of the last continuously-qualifying frame (None = paused).
    anchor_at: Optional[datetime] = None
    anchor_in_interaction: bool = False
    last_valid_paper_at: Optional[datetime] = None
    aborted_reason: Optional[str] = None
    #: A temporal completion already closed this candidate deterministically.
    decided: bool = False

    def pause(self) -> None:
        """Break qualifying continuity; keep accumulators (no blind credit)."""
        self.anchor_at = None
        self.anchor_in_interaction = False

    def invalidate(self, reason: str) -> None:
        self.pause()
        self.observed_total_seconds = 0.0
        self.observed_interaction_seconds = 0.0
        self.aborted_reason = reason


class PaperHandoffFusionTracker:
    """Stateful, synchronous, deterministic fusion state machine.

    Not thread-safe by design: this phase is not runtime-integrated.
    """

    def __init__(self) -> None:
        self._candidates: dict[StateKey, _FusionCandidate] = {}
        #: Explicit fusion frame-order guard, because paper frames arrive from a
        #: separate pipeline. Fails closed on duplicate / out-of-order input.
        self._order: dict[StateKey, tuple[int, datetime]] = {}

    # ------------------------------------------------------------- lifecycle

    def reset_camera(self, camera_id: str) -> None:
        """Removes fusion state for ONE camera only."""
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise PaperHandoffFusionContractError(
                "camera_id must be a non-blank string"
            )
        self._drop(lambda key: key[0] == camera_id)

    def reset_context(self, camera_id: str, rule_id: str) -> None:
        """Removes fusion state for ONE (camera, rule) across its generations."""
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise PaperHandoffFusionContractError(
                "camera_id must be a non-blank string"
            )
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise PaperHandoffFusionContractError("rule_id must be a non-blank string")
        self._drop(lambda key: key[0] == camera_id and key[2] == rule_id)

    @property
    def active_candidate_count(self) -> int:
        return len(self._candidates)

    def _drop(self, predicate) -> None:
        for key in [key for key in self._candidates if predicate(key)]:
            del self._candidates[key]
        for key in [key for key in self._order if predicate(key)]:
            del self._order[key]

    # --------------------------------------------------------------- observe

    def observe(
        self,
        *,
        temporal: HandoffTemporalResult,
        spatial_frame: PaperPairSpatialFrame,
        join: PaperHandoffFusionJoin,
        armed: bool,
        spec: PaperHandoffFusionSpec,
    ) -> PaperHandoffFusionResult:
        """Fuses ONE Task 3D candidate result with ONE Task 3F spatial frame."""
        if not isinstance(temporal, HandoffTemporalResult):
            raise PaperHandoffFusionContractError(
                "temporal must be a HandoffTemporalResult (Task 3D is authoritative)"
            )
        if not isinstance(spatial_frame, PaperPairSpatialFrame):
            raise PaperHandoffFusionContractError(
                "spatial_frame must be a PaperPairSpatialFrame (Task 3F is authoritative)"
            )
        if not isinstance(join, PaperHandoffFusionJoin):
            raise PaperHandoffFusionContractError(
                "join must be an explicit PaperHandoffFusionJoin"
            )
        if not isinstance(spec, PaperHandoffFusionSpec):
            raise PaperHandoffFusionContractError(
                "spec must be a PaperHandoffFusionSpec"
            )
        if type(armed) is not bool:
            raise PaperHandoffFusionContractError(
                "armed must be an explicit real bool"
            )

        identity = join.identity
        key: StateKey = (
            join.camera_id,
            join.stream_generation,
            join.rule_id,
            join.pair_key,
        )

        # --- explicit arming gate (Task 3D remains authoritative) ------------
        if not armed:
            self._candidates.pop(key, None)
            self._order.pop(key, None)
            return PaperHandoffFusionResult(
                status=PaperHandoffFusionStatus.DISARMED,
                identity=identity,
                pair_frame_sequence=join.pair_frame_sequence,
                pair_observed_at=join.pair_observed_at,
                paper_support_status=PaperSupportStatus.UNKNOWN,
                reason=RESET_DISARMED,
            )

        # --- strict cross-contract provenance --------------------------------
        mismatch = self._provenance_mismatch(temporal, spatial_frame, join)
        if mismatch is not None:
            self._candidates.pop(key, None)
            self._order.pop(key, None)
            return PaperHandoffFusionResult(
                status=PaperHandoffFusionStatus.INCONSISTENT_INPUT,
                identity=identity,
                pair_frame_sequence=join.pair_frame_sequence,
                pair_observed_at=join.pair_observed_at,
                paper_support_status=PaperSupportStatus.UNKNOWN,
                reason=mismatch,
            )

        # --- Task 3D lifecycle: retire stale generations of this context ------
        # Generation isolation alone would keep replaced stream incarnations
        # stored forever. Only OTHER generations of this exact (camera, rule) are
        # retired: never another camera and never another rule.
        self._drop(
            lambda other: other[0] == join.camera_id
            and other[2] == join.rule_id
            and other[1] != join.stream_generation
        )

        # Terminal Task 3D lifecycle outcomes retire candidate-local fusion state.
        # --- Task 3D abort/reset is authoritative -----------------------------
        # A terminal Task 3D reason ends that exact temporal candidate. The
        # terminal frame itself may never add paper dwell, and no paper evidence
        # of the terminated candidate may survive it.
        if temporal.abort_reason in TERMINAL_TEMPORAL_REASONS:
            self._candidates.pop(key, None)
            self._order.pop(key, None)
            return PaperHandoffFusionResult(
                status=PaperHandoffFusionStatus.OK,
                identity=identity,
                pair_frame_sequence=join.pair_frame_sequence,
                pair_observed_at=join.pair_observed_at,
                temporal_phase=temporal.phase,
                paper_support_status=PaperSupportStatus.UNKNOWN,
                temporal_completed_this_frame=bool(temporal.completed_this_frame),
                reason=temporal.abort_reason,
            )

        # --- explicit fail-closed fusion frame-order guard -------------------

        now = join.pair_observed_at
        previous = self._order.get(key)
        if previous is not None and (
            join.pair_frame_sequence <= previous[0] or now <= previous[1]
        ):
            self._candidates.pop(key, None)
            self._order.pop(key, None)
            return PaperHandoffFusionResult(
                status=PaperHandoffFusionStatus.NON_MONOTONIC,
                identity=identity,
                pair_frame_sequence=join.pair_frame_sequence,
                pair_observed_at=now,
                paper_support_status=PaperSupportStatus.UNKNOWN,
                reason=RESET_NON_MONOTONIC,
            )
        self._order[key] = (join.pair_frame_sequence, now)

        # --- temporal candidate identity -------------------------------------
        candidate = self._candidates.get(key)
        restarted = candidate is not None and candidate.identity != identity
        if candidate is None or restarted:
            candidate = _FusionCandidate(identity=identity)
            self._candidates[key] = candidate

        evaluation = self._evaluate(temporal, spatial_frame, spec)

        # --- observed dwell accounting (UNKNOWN never credited) ---------------
        if evaluation.status is PaperSupportStatus.UNKNOWN:
            candidate.pause()
            if candidate.last_valid_paper_at is not None and (
                _seconds(now, candidate.last_valid_paper_at)
                > float(spec.max_paper_unknown_gap_seconds)
            ):
                candidate.invalidate(ABORT_PAPER_UNKNOWN_GAP_EXCEEDED)
        else:
            candidate.last_valid_paper_at = now
            if evaluation.qualified:
                if candidate.anchor_at is not None:
                    delta = _seconds(now, candidate.anchor_at)
                    candidate.observed_total_seconds += delta
                    if (
                        candidate.anchor_in_interaction
                        and temporal.phase is HandoffPhase.INTERACTION
                    ):
                        candidate.observed_interaction_seconds += delta
                candidate.anchor_at = now
                candidate.anchor_in_interaction = (
                    temporal.phase is HandoffPhase.INTERACTION
                )
            else:
                # Valid negative / non-correlated evidence breaks continuity.
                candidate.pause()

        unknown_gap = (
            _seconds(now, candidate.last_valid_paper_at)
            if candidate.last_valid_paper_at is not None
            else 0.0
        )

        # --- deterministic, non-retroactive completion decision --------------
        fused = False
        reason = candidate.aborted_reason
        if temporal.completed_this_frame:
            if candidate.decided:
                reason = CLOSED_ALREADY_DECIDED
            elif candidate.aborted_reason is not None:
                candidate.decided = True
            elif (
                candidate.observed_interaction_seconds
                < float(spec.min_interaction_paper_observed_seconds)
            ):
                candidate.decided = True
                reason = NOT_QUALIFIED_INSUFFICIENT_INTERACTION_PAPER
            elif (
                candidate.observed_total_seconds
                < float(spec.min_total_paper_observed_seconds)
            ):
                candidate.decided = True
                reason = NOT_QUALIFIED_INSUFFICIENT_TOTAL_PAPER
            else:
                candidate.decided = True
                fused = True
                reason = None
        elif candidate.decided:
            # The candidate closed deterministically: no later paper evidence may
            # retroactively qualify it.
            reason = CLOSED_ALREADY_DECIDED

        if restarted and reason is None:
            reason = RESET_NEW_CANDIDATE

        return PaperHandoffFusionResult(
            status=PaperHandoffFusionStatus.OK,
            identity=identity,
            pair_frame_sequence=join.pair_frame_sequence,
            pair_observed_at=now,
            temporal_phase=temporal.phase,
            paper_support_status=evaluation.status,
            support_qualified_this_frame=bool(evaluation.qualified),
            qualifying_paper_detection_index=(
                evaluation.detection_index if evaluation.qualified else None
            ),
            qualifying_paper_raw_prompt=(
                evaluation.raw_prompt if evaluation.qualified else None
            ),
            qualifying_paper_confidence=evaluation.confidence,
            paper_distance_to_locked_wrist_a=evaluation.distance_a,
            paper_distance_to_locked_wrist_b=evaluation.distance_b,
            observed_interaction_paper_seconds=candidate.observed_interaction_seconds,
            observed_total_paper_seconds=candidate.observed_total_seconds,
            last_valid_paper_evidence_at=candidate.last_valid_paper_at,
            paper_unknown_gap_seconds=unknown_gap,
            temporal_completed_this_frame=bool(temporal.completed_this_frame),
            fused_completed_this_frame=fused,
            reason=reason,
        )

    # ------------------------------------------------------------ provenance

    @staticmethod
    def _provenance_mismatch(
        temporal: HandoffTemporalResult,
        spatial_frame: PaperPairSpatialFrame,
        join: PaperHandoffFusionJoin,
    ) -> Optional[str]:
        """Returns a deterministic reason string on ANY identity contradiction."""
        if temporal.camera_id != join.camera_id:
            return "camera_id_mismatch"
        if temporal.stream_generation != join.stream_generation:
            return "stream_generation_mismatch"
        if temporal.rule_id != join.rule_id:
            return "rule_id_mismatch"
        if temporal.pair_key != join.pair_key:
            return "pair_key_mismatch"
        if temporal.candidate_started_at != join.candidate_started_at:
            return "temporal_candidate_identity_mismatch"
        if (
            temporal.locked_side_a != join.locked_side_a
            or temporal.locked_side_b != join.locked_side_b
        ):
            return "locked_wrist_side_mismatch"
        if (
            spatial_frame.camera_id is not None
            and spatial_frame.camera_id != join.camera_id
        ):
            return "spatial_camera_id_mismatch"
        if (
            spatial_frame.pair_frame_sequence is not None
            and spatial_frame.pair_frame_sequence != join.pair_frame_sequence
        ):
            return "spatial_frame_provenance_mismatch"
        if (
            spatial_frame.pair_observed_at is not None
            and spatial_frame.pair_observed_at != join.pair_observed_at
        ):
            return "spatial_observed_at_mismatch"
        return None

    # -------------------------------------------------------------- evidence

    @staticmethod
    def _evaluate(
        temporal: HandoffTemporalResult,
        spatial_frame: PaperPairSpatialFrame,
        spec: PaperHandoffFusionSpec,
    ) -> _Evaluation:
        """Frame-local paper support, strictly against Task 3D's LOCKED wrists.

        Wrist sides are NEVER substituted: Task 3D is authoritative for the
        locked combination, so paper near a non-locked wrist can never support
        this candidate.
        """
        if spatial_frame.status is not PaperPairSpatialStatus.OK:
            return _Evaluation(PaperSupportStatus.UNKNOWN)

        facts = spatial_frame.facts_for_pair(temporal.pair_key)
        if not facts:
            if spatial_frame.paper_detection_count == 0:
                # Valid negative detector evidence, NOT unknown.
                return _Evaluation(PaperSupportStatus.NONE)
            return _Evaluation(PaperSupportStatus.UNKNOWN)

        owner_a, owner_b = temporal.pair_key.tracking_ids
        threshold = float(
            spec.max_paper_to_locked_wrist_distance_relative_to_mean_diagonal
        )
        floor = float(spec.min_paper_detector_confidence)

        passing = 0
        measurable = 0
        qualifying: list[tuple[float, int, _Evaluation]] = []
        nearest_non_qualifying: Optional[tuple[float, int, _Evaluation]] = None

        for fact in sorted(facts, key=lambda item: item.paper.detection_index):
            if float(fact.paper.confidence) < floor:
                continue
            passing += 1
            distance_a = _locked_distance(fact, owner_a, temporal.locked_side_a)
            distance_b = _locked_distance(fact, owner_b, temporal.locked_side_b)
            if spec.support_mode is PaperSupportMode.BOTH_LOCKED_WRISTS:
                if distance_a is None or distance_b is None:
                    continue
                measurable += 1
                metric = max(distance_a, distance_b)
                qualified = distance_a <= threshold and distance_b <= threshold
                status = (
                    PaperSupportStatus.NEAR_BOTH_LOCKED_WRISTS
                    if qualified
                    else PaperSupportStatus.PAPER_PRESENT_BUT_NOT_CORRELATED
                )
            else:
                if distance_a is None and distance_b is None:
                    continue
                measurable += 1
                available = [value for value in (distance_a, distance_b) if value is not None]
                metric = min(available)
                near_a = distance_a is not None and distance_a <= threshold
                near_b = distance_b is not None and distance_b <= threshold
                qualified = near_a or near_b
                if near_a and near_b:
                    status = PaperSupportStatus.NEAR_BOTH_LOCKED_WRISTS
                elif near_a:
                    status = PaperSupportStatus.NEAR_LOCKED_WRIST_A
                elif near_b:
                    status = PaperSupportStatus.NEAR_LOCKED_WRIST_B
                else:
                    status = PaperSupportStatus.PAPER_PRESENT_BUT_NOT_CORRELATED

            evaluation = _Evaluation(
                status=status,
                qualified=qualified,
                detection_index=fact.paper.detection_index,
                raw_prompt=fact.paper.raw_prompt,
                confidence=float(fact.paper.confidence),
                distance_a=distance_a,
                distance_b=distance_b,
            )
            entry = (metric, fact.paper.detection_index, evaluation)
            if qualified:
                qualifying.append(entry)
            elif nearest_non_qualifying is None or entry[:2] < nearest_non_qualifying[:2]:
                nearest_non_qualifying = entry

        if qualifying:
            # Deterministic selection: nearest first, then lowest frame-local
            # detection index. Bboxes are never merged and no paper identity is
            # created, so the selected detection may differ between frames.
            qualifying.sort(key=lambda item: (item[0], item[1]))
            return qualifying[0][2]
        if measurable:
            return nearest_non_qualifying[2] if nearest_non_qualifying else _Evaluation(
                PaperSupportStatus.PAPER_PRESENT_BUT_NOT_CORRELATED
            )
        if passing:
            # Confidence-passing paper exists but the required locked-wrist
            # spatial facts were unavailable: unknown, never negative proof.
            return _Evaluation(PaperSupportStatus.UNKNOWN)
        return _Evaluation(PaperSupportStatus.NONE)


def _locked_distance(fact, owner_tracking_id: str, side) -> Optional[float]:
    """Task 3F normalized paper<->wrist distance relative to the mean pair diagonal."""
    for wrist in fact.wrist_facts:
        if wrist.wrist_owner_tracking_id == owner_tracking_id and wrist.side is side:
            value = wrist.distance_relative_to_mean_pair_diagonal
            return None if value is None else float(value)
    return None
