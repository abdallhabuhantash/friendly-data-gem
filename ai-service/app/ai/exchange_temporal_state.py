"""Deterministic temporal state machine over Task 3C person-pair geometry.

Task 3D owns its temporal state independently: Task 1's
``app/ai/temporal_state.py`` is neither imported nor modified here, and this
module is intentionally dormant (no runtime, orchestrator, engine-registry,
event, notification or Supabase coupling, no background threads).

What this module can and cannot say:

* CAN say: "for this camera, stream generation, rule, tracked-person pair and
  ONE locked wrist-side combination, the configured
  approach -> near-interaction dwell -> separation sequence was observed."
* CANNOT say: paper, document, sheet, object transfer, cheating, contact,
  touching, or who gave what to whom. A handshake yields the same geometry.

Timing uses ONLY the source frames' ``observed_at`` values. There is no
``time.time()``, no monotonic clock, no sleep and no frame-count heuristic, so
variable pose cadence and skipped frame numbers are handled correctly.

Arming is a pure caller input. Constructing this tracker arms nothing; a camera
starting arms nothing. While ``armed=False`` no candidate state exists at all,
which is what makes legitimate pre-arm paper distribution incapable of priming a
later completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.domain.body_features import BodySide
from app.domain.handoff_temporal import (
    ABORT_APPROACH_LOST,
    ABORT_APPROACH_TIMEOUT,
    ABORT_EVIDENCE_GAP_EXCEEDED,
    ABORT_INTERACTION_DWELL_TOO_SHORT,
    RESET_DISARMED,
    RESET_NON_MONOTONIC,
    RESET_RECOVERED,
    HandoffPhase,
    HandoffTemporalContractError,
    HandoffTemporalFrameResult,
    HandoffTemporalResult,
    HandoffTemporalSpec,
    HandoffTemporalStatus,
    StreamGeneration,
    validate_stream_generation,
)
from app.domain.pair_geometry import (
    PairFrameStatus,
    PersonPairFrameResult,
    PersonPairGeometry,
    PersonPairKey,
    WristPairGeometry,
)

#: Full temporal state identity. A change in ANY member is a different state:
#: cameras, stream incarnations, rules and pairs never share or migrate state,
#: and a tracking-id change is simply a different pair (no re-identification).
StateKey = tuple[str, StreamGeneration, str, PersonPairKey]
ContextKey = tuple[str, StreamGeneration, str]


def _seconds(later: datetime, earlier: datetime) -> float:
    return max(0.0, (later - earlier).total_seconds())


@dataclass
class _Candidate:
    """PRIVATE mutable per-pair state. Only bounded scalars: no frame history.

    Dwell accounting is ACCUMULATED OBSERVED time, never wall time. Each dwell
    keeps a bounded accumulator plus an optional anchor timestamp. The anchor is
    cleared whenever the supporting evidence stops being continuously valid
    (UNKNOWN / degraded / missing pair or wrist, or valid-but-non-qualifying
    geometry), so blind intervals can never be credited as observed dwell. The
    accumulator itself is retained, so evidence returning inside
    ``max_unknown_gap_seconds`` resumes from the previously observed dwell.
    """

    side_a: BodySide
    side_b: BodySide
    started_at: datetime
    first_distance: float
    closest_distance: float
    last_distance: float
    last_valid_evidence_at: datetime
    phase: HandoffPhase = HandoffPhase.APPROACHING
    interaction_started_at: Optional[datetime] = None
    #: Anchor of the last continuously-valid near-interaction frame (None = paused).
    interaction_last_at: Optional[datetime] = None
    interaction_observed_seconds: float = 0.0
    separation_started_at: Optional[datetime] = None
    #: Anchor of the last continuously-valid separated frame (None = paused).
    separation_last_at: Optional[datetime] = None
    separation_observed_seconds: float = 0.0
    separation_max_distance: float = 0.0
    completed_at: Optional[datetime] = None
    recovery_started_at: Optional[datetime] = None
    #: Anchor of the last continuously-valid recovery frame (None = paused).
    recovery_last_at: Optional[datetime] = None
    recovery_observed_seconds: float = 0.0

    @property
    def approach_reduction(self) -> float:
        return max(0.0, self.first_distance - self.closest_distance)

    @property
    def separation_increase(self) -> float:
        return max(0.0, self.separation_max_distance - self.closest_distance)

    @property
    def interaction_duration(self) -> float:
        """Genuinely observed accumulated interaction dwell (unknown excluded)."""
        return self.interaction_observed_seconds

    @property
    def separation_duration(self) -> float:
        """Genuinely observed accumulated separation dwell (unknown excluded)."""
        return self.separation_observed_seconds

    def pause_dwell_accounting(self) -> None:
        """UNKNOWN evidence: keep accumulators, drop anchors (no blind credit)."""
        self.interaction_last_at = None
        self.separation_last_at = None
        self.recovery_last_at = None



class HandoffTemporalTracker:
    """Stateful, synchronous, deterministic temporal tracker.

    Not thread-safe by design: this phase is not runtime-integrated, and runtime
    concurrency belongs to the future integration layer rather than speculative
    locking here.
    """

    def __init__(self) -> None:
        self._candidates: dict[StateKey, _Candidate] = {}
        #: Last accepted (frame_sequence, observed_at) per context, for strict
        #: monotonic ordering. Duplicates never accumulate dwell time.
        self._order: dict[ContextKey, tuple[int, datetime]] = {}

    # ------------------------------------------------------------- lifecycle

    def reset_camera(self, camera_id: str) -> None:
        """Removes state for ONE camera only. No other camera is affected."""
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise HandoffTemporalContractError("camera_id must be a non-blank string")
        self._drop(lambda key: key[0] == camera_id)

    def reset_context(self, camera_id: str, rule_id: str) -> None:
        """Removes state for ONE (camera, rule) context across its generations."""
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise HandoffTemporalContractError("camera_id must be a non-blank string")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise HandoffTemporalContractError("rule_id must be a non-blank string")
        self._drop(lambda key: key[0] == camera_id and key[2] == rule_id)

    @property
    def active_candidate_count(self) -> int:
        return len(self._candidates)

    def _drop(self, predicate) -> None:
        for key in [key for key in self._candidates if predicate(key)]:
            del self._candidates[key]
        for key in [key for key in self._order if predicate((key[0], key[1], key[2]))]:
            del self._order[key]

    # --------------------------------------------------------------- observe

    def observe(
        self,
        frame: PersonPairFrameResult,
        *,
        rule_id: str,
        stream_generation: StreamGeneration,
        armed: bool,
        spec: HandoffTemporalSpec,
    ) -> HandoffTemporalFrameResult:
        """Processes ONE same-frame Task 3C result. The frame is never mutated."""
        if not isinstance(frame, PersonPairFrameResult):
            raise HandoffTemporalContractError(
                "frame must be a PersonPairFrameResult (Task 3C is authoritative)"
            )
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise HandoffTemporalContractError("rule_id must be a non-blank string")
        if not isinstance(spec, HandoffTemporalSpec):
            raise HandoffTemporalContractError("spec must be a HandoffTemporalSpec")
        if type(armed) is not bool:
            raise HandoffTemporalContractError("armed must be an explicit real bool")
        generation = validate_stream_generation(stream_generation)
        camera_id = frame.camera_id

        # --- explicit arming gate -------------------------------------------
        if not armed:
            if isinstance(camera_id, str) and camera_id.strip():
                self.reset_context(camera_id, rule_id)
            return HandoffTemporalFrameResult(
                status=HandoffTemporalStatus.DISARMED,
                camera_id=camera_id,
                stream_generation=generation,
                rule_id=rule_id,
                frame_sequence=frame.frame_sequence,
                observed_at=frame.observed_at,
                reason=RESET_DISARMED,
            )

        if (
            not isinstance(camera_id, str)
            or not camera_id.strip()
            or not isinstance(frame.frame_sequence, int)
            or isinstance(frame.frame_sequence, bool)
            or frame.frame_sequence < 0
            or not isinstance(frame.observed_at, datetime)
        ):
            if isinstance(camera_id, str) and camera_id.strip():
                self.reset_context(camera_id, rule_id)
            return HandoffTemporalFrameResult(
                status=HandoffTemporalStatus.INVALID_INPUT,
                camera_id=camera_id if isinstance(camera_id, str) else None,
                stream_generation=generation,
                rule_id=rule_id,
                frame_sequence=frame.frame_sequence,
                observed_at=frame.observed_at,
                reason="missing_frame_metadata",
            )

        sequence = int(frame.frame_sequence)
        now: datetime = frame.observed_at
        context: ContextKey = (camera_id, generation, rule_id)

        # A new generation retires every older incarnation of the same context.
        self._drop(
            lambda key: key[0] == camera_id and key[2] == rule_id and key[1] != generation
        )

        # --- strict frame ordering ------------------------------------------
        previous = self._order.get(context)
        if previous is not None:
            last_sequence, last_observed_at = previous
            if sequence <= last_sequence or now <= last_observed_at:
                self._drop(lambda key: key[:3] == context)
                return HandoffTemporalFrameResult(
                    status=HandoffTemporalStatus.NON_MONOTONIC,
                    camera_id=camera_id,
                    stream_generation=generation,
                    rule_id=rule_id,
                    frame_sequence=sequence,
                    observed_at=now,
                    reason=RESET_NON_MONOTONIC,
                )
        self._order[context] = (sequence, now)

        degraded = frame.status is not PairFrameStatus.OK
        pairs: dict[PersonPairKey, PersonPairGeometry] = (
            {} if degraded else {pair.key: pair for pair in frame.pairs}
        )

        results: list[HandoffTemporalResult] = []
        #: Keys already processed this frame never restart in the same frame.
        processed: set[PersonPairKey] = set()

        # --- existing candidates --------------------------------------------
        for key in [key for key in self._candidates if key[:3] == context]:
            candidate = self._candidates[key]
            pair = pairs.get(key[3])
            evidence = (
                None if pair is None else self._locked_distance(pair, candidate)
            )
            result = self._advance(
                key, candidate, evidence, now=now, spec=spec, degraded=degraded
            )
            results.append(result)
            processed.add(key[3])

        # --- new candidates -------------------------------------------------
        if not degraded:
            for pair_key in sorted(
                pairs, key=lambda item: item.tracking_ids
            ):
                key = (camera_id, generation, rule_id, pair_key)
                if key in self._candidates or pair_key in processed:
                    continue
                started = self._maybe_start(pairs[pair_key], now=now, spec=spec)
                if started is None:
                    continue
                self._candidates[key] = started
                results.append(self._result(key, started, started.last_distance, True))

        status = (
            HandoffTemporalStatus.DEGRADED_FRAME
            if degraded
            else HandoffTemporalStatus.OK
        )
        return HandoffTemporalFrameResult(
            status=status,
            camera_id=camera_id,
            stream_generation=generation,
            rule_id=rule_id,
            frame_sequence=sequence,
            observed_at=now,
            candidates=tuple(
                sorted(results, key=lambda item: item.pair_key.tracking_ids)
            ),
            reason=frame.reason if degraded else None,
        )

    # ------------------------------------------------------------- internals

    @staticmethod
    def _pair_distances(pair: PersonPairGeometry) -> list[WristPairGeometry]:
        """Wrist pairs carrying the symmetric mean-diagonal normalized distance."""
        return [
            wrist_pair
            for wrist_pair in pair.wrist_pairs
            if wrist_pair.distance_relative_to_mean_person_diagonal is not None
        ]

    @staticmethod
    def _locked_distance(
        pair: PersonPairGeometry, candidate: _Candidate
    ) -> Optional[float]:
        """ONLY the locked wrist combination is ever read: no substitution."""
        for wrist_pair in pair.wrist_pairs:
            if (
                wrist_pair.side_a is candidate.side_a
                and wrist_pair.side_b is candidate.side_b
                and wrist_pair.distance_relative_to_mean_person_diagonal is not None
            ):
                return float(wrist_pair.distance_relative_to_mean_person_diagonal)
        return None

    def _maybe_start(
        self, pair: PersonPairGeometry, *, now: datetime, spec: HandoffTemporalSpec
    ) -> Optional[_Candidate]:
        eligible_center = pair.center_distance_relative_to_mean_person_diagonal
        if (
            eligible_center is None
            or float(eligible_center) > spec.max_person_center_distance
        ):
            return None
        options = self._pair_distances(pair)
        if not options:
            return None
        # Deterministic selection: nearest distance, ties broken by stable
        # semantic side names. Input tuple ordering can never change this.
        chosen = min(
            options,
            key=lambda item: (
                float(item.distance_relative_to_mean_person_diagonal),
                item.side_a.value,
                item.side_b.value,
            ),
        )
        distance = float(chosen.distance_relative_to_mean_person_diagonal)
        # A candidate may only START strictly outside the interaction criterion:
        # continuously-close static posture therefore never becomes a candidate,
        # and one single near frame can never create an interaction.
        if not (
            spec.interaction_wrist_distance
            < distance
            <= spec.approach_start_wrist_distance
        ):
            return None
        return _Candidate(
            side_a=chosen.side_a,
            side_b=chosen.side_b,
            started_at=now,
            first_distance=distance,
            closest_distance=distance,
            last_distance=distance,
            last_valid_evidence_at=now,
        )

    def _advance(
        self,
        key: StateKey,
        candidate: _Candidate,
        evidence: Optional[float],
        *,
        now: datetime,
        spec: HandoffTemporalSpec,
        degraded: bool,
    ) -> HandoffTemporalResult:
        # Missing evidence (occluded locked wrist, pair absent from the frame, or
        # a degraded pose frame) is UNKNOWN. It is never "far away", never
        # "interaction ended" and never "separated".
        if evidence is None:
            if _seconds(now, candidate.last_valid_evidence_at) > spec.max_unknown_gap_seconds:
                self._candidates.pop(key, None)
                return self._result(
                    key,
                    candidate,
                    None,
                    False,
                    phase=HandoffPhase.IDLE,
                    abort_reason=ABORT_EVIDENCE_GAP_EXCEEDED,
                )
            # Tolerated gap: candidate stays alive with its locked wrist pair,
            # but every dwell accumulator pauses (no blind time is credited).
            candidate.pause_dwell_accounting()
            return self._result(key, candidate, None, False)


        distance = float(evidence)
        candidate.last_distance = distance
        candidate.last_valid_evidence_at = now
        completed_now = False
        abort_reason: Optional[str] = None

        if candidate.phase is HandoffPhase.APPROACHING:
            candidate.closest_distance = min(candidate.closest_distance, distance)
            if (
                spec.max_approach_duration_seconds is not None
                and _seconds(now, candidate.started_at)
                > spec.max_approach_duration_seconds
            ):
                self._candidates.pop(key, None)
                return self._result(
                    key,
                    candidate,
                    distance,
                    True,
                    phase=HandoffPhase.IDLE,
                    abort_reason=ABORT_APPROACH_TIMEOUT,
                )
            if (
                distance <= spec.interaction_wrist_distance
                and candidate.approach_reduction >= spec.min_approach_distance_reduction
            ):
                candidate.phase = HandoffPhase.INTERACTION
                candidate.interaction_started_at = now
                candidate.interaction_last_at = now
                candidate.interaction_observed_seconds = 0.0

            elif distance > spec.approach_start_wrist_distance:
                self._candidates.pop(key, None)
                return self._result(
                    key,
                    candidate,
                    distance,
                    True,
                    phase=HandoffPhase.IDLE,
                    abort_reason=ABORT_APPROACH_LOST,
                )

        if candidate.phase is HandoffPhase.INTERACTION:
            candidate.closest_distance = min(candidate.closest_distance, distance)
            if distance <= spec.interaction_wrist_distance:
                # Accumulate ONLY the interval between two consecutive frames
                # that both carried valid near-interaction evidence.
                if candidate.interaction_last_at is not None:
                    candidate.interaction_observed_seconds += _seconds(
                        now, candidate.interaction_last_at
                    )
                candidate.interaction_last_at = now
            elif candidate.interaction_duration >= spec.min_interaction_dwell_seconds:
                candidate.phase = HandoffPhase.SEPARATING
                candidate.interaction_last_at = None
                candidate.separation_started_at = None
                candidate.separation_last_at = None
                candidate.separation_observed_seconds = 0.0
                candidate.separation_max_distance = distance
            else:
                self._candidates.pop(key, None)
                return self._result(
                    key,
                    candidate,
                    distance,
                    True,
                    phase=HandoffPhase.IDLE,
                    abort_reason=ABORT_INTERACTION_DWELL_TOO_SHORT,
                )

        if candidate.phase is HandoffPhase.SEPARATING:
            candidate.separation_max_distance = max(
                candidate.separation_max_distance, distance
            )
            separated = (
                distance >= spec.min_separation_wrist_distance
                and (distance - candidate.closest_distance)
                >= spec.min_separation_distance_increase
            )
            if separated:
                if candidate.separation_started_at is None:
                    candidate.separation_started_at = now
                    candidate.separation_observed_seconds = 0.0
                elif candidate.separation_last_at is not None:
                    candidate.separation_observed_seconds += _seconds(
                        now, candidate.separation_last_at
                    )
                candidate.separation_last_at = now
                if (
                    candidate.separation_duration
                    >= spec.min_separation_dwell_seconds
                ):
                    candidate.phase = HandoffPhase.COMPLETED
                    candidate.completed_at = now
                    candidate.recovery_started_at = None
                    candidate.recovery_last_at = None
                    candidate.recovery_observed_seconds = 0.0
                    completed_now = True
            else:
                # Separation progress restarts; near geometry resumes interaction.
                candidate.separation_started_at = None
                candidate.separation_last_at = None
                candidate.separation_observed_seconds = 0.0
                if distance <= spec.interaction_wrist_distance:
                    candidate.phase = HandoffPhase.INTERACTION
                    candidate.interaction_last_at = now

        if candidate.phase is HandoffPhase.COMPLETED and not completed_now:
            # Latched: a completion is reported EXACTLY ONCE. This is
            # state-machine duplicate prevention, not alert cooldown.
            if distance >= spec.recovery_wrist_distance:
                if candidate.recovery_started_at is None:
                    candidate.recovery_started_at = now
                    candidate.recovery_observed_seconds = 0.0
                elif candidate.recovery_last_at is not None:
                    candidate.recovery_observed_seconds += _seconds(
                        now, candidate.recovery_last_at
                    )
                candidate.recovery_last_at = now
                if candidate.recovery_observed_seconds >= spec.recovery_dwell_seconds:
                    self._candidates.pop(key, None)
                    return self._result(
                        key,
                        candidate,
                        distance,
                        True,
                        abort_reason=RESET_RECOVERED,
                    )
            else:
                candidate.recovery_started_at = None
                candidate.recovery_last_at = None
                candidate.recovery_observed_seconds = 0.0


        return self._result(
            key,
            candidate,
            distance,
            True,
            completed_this_frame=completed_now,
            abort_reason=abort_reason,
        )

    @staticmethod
    def _result(
        key: StateKey,
        candidate: _Candidate,
        distance: Optional[float],
        evidence_available: bool,
        *,
        phase: Optional[HandoffPhase] = None,
        completed_this_frame: bool = False,
        abort_reason: Optional[str] = None,
    ) -> HandoffTemporalResult:
        camera_id, generation, rule_id, pair_key = key
        # Raw calibration facts are genuinely OBSERVED accumulated durations,
        # never wall time spanning tolerated UNKNOWN intervals.
        separation_duration = candidate.separation_duration

        return HandoffTemporalResult(
            camera_id=camera_id,
            stream_generation=generation,
            rule_id=rule_id,
            pair_key=pair_key,
            phase=phase or candidate.phase,
            locked_side_a=candidate.side_a,
            locked_side_b=candidate.side_b,
            candidate_started_at=candidate.started_at,
            last_valid_evidence_at=candidate.last_valid_evidence_at,
            evidence_available_this_frame=evidence_available,
            closest_wrist_distance=candidate.closest_distance,
            current_wrist_distance=distance,
            approach_distance_reduction=candidate.approach_reduction,
            separation_distance_increase=candidate.separation_increase,
            interaction_started_at=candidate.interaction_started_at,
            separation_started_at=candidate.separation_started_at,
            interaction_duration_seconds=candidate.interaction_duration,
            separation_duration_seconds=separation_duration,
            completed_this_frame=completed_this_frame,
            completed_at=candidate.completed_at,
            abort_reason=abort_reason,
        )
