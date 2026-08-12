"""Deterministic tests for TASK 3G paper-corroborated handoff fusion.

Pure/dormant layer: no events, no notifications, no UI, no runtime integration,
no giver/receiver inference, no cheating vocabulary, no fused score. Fusion
correctness here is NOT paper-detector real-world accuracy: the open-vocabulary
paper detector still requires offline real-video acceptance.
"""

from __future__ import annotations

import dataclasses
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.ai.paper_handoff_fusion import PaperHandoffFusionTracker
from app.ai.paper_pair_spatial_builder import build_paper_pair_spatial_frame
from app.ai.person_pair_geometry_builder import build_person_pair_frame_from_tracked_pose
from app.domain.body_features import BodySide
from app.domain.geometry import BBox
from app.domain.handoff_temporal import (
    ABORT_APPROACH_LOST,
    ABORT_APPROACH_TIMEOUT,
    ABORT_EVIDENCE_GAP_EXCEEDED,
    ABORT_INTERACTION_DWELL_TOO_SHORT,
    RESET_RECOVERED,
    HandoffPhase,
    HandoffTemporalResult,
)
from app.domain.pair_geometry import PersonPairKey
from app.domain.paper_evidence import PaperEvidenceFrame, PaperEvidenceStatus
from app.domain.paper_handoff_fusion import (
    ABORT_PAPER_UNKNOWN_GAP_EXCEEDED,
    PaperHandoffFusionContractError,
    PaperHandoffFusionJoin,
    PaperHandoffFusionSpec,
    PaperHandoffFusionStatus,
    PaperSupportMode,
    PaperSupportStatus,
)
from app.domain.paper_pair_spatial import PaperPairSpatialStatus, SameFrameJoin
from app.domain.pose import PoseStatus
from app.domain.tracked_pose_observations import (
    TrackedPoseFrameResult,
    TrackedPoseFrameStatus,
)

from tests._source_scan import code_text
from tests.test_paper_pair_spatial_geometry import observation, paper, person

CAMERA_ID = "cam-1"
RULE_ID = "rule-exchange"
GENERATION = 4
PAIR = PersonPairKey.of("a", "b")
START_AT = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
SIDE_A = BodySide.RIGHT
SIDE_B = BodySide.LEFT

#: Explicit spec: every criterion is caller supplied, no production defaults.
SPEC = PaperHandoffFusionSpec(
    min_paper_detector_confidence=0.5,
    max_paper_to_locked_wrist_distance_relative_to_mean_diagonal=0.25,
    min_interaction_paper_observed_seconds=0.5,
    min_total_paper_observed_seconds=0.5,
    max_paper_unknown_gap_seconds=1.0,
    support_mode=PaperSupportMode.EITHER_LOCKED_WRIST,
)
BOTH_SPEC = dataclasses.replace(SPEC, support_mode=PaperSupportMode.BOTH_LOCKED_WRISTS)

NEAR_PAPER = paper(BBox(0.415, 0.39, 0.02, 0.02))          # between both wrists
NEAR_A_ONLY = paper(BBox(0.34, 0.39, 0.02, 0.02))          # near a-right only
FAR_PAPER = paper(BBox(0.05, 0.85, 0.03, 0.03))            # far from both wrists
LOW_CONFIDENCE = paper(BBox(0.415, 0.39, 0.02, 0.02), confidence=0.2)


def at(step: float) -> datetime:
    return START_AT + timedelta(seconds=step)


def people(*, wrong_wrist: bool = False, third_person: bool = False):
    a_wrists = {SIDE_A: (0.32, 0.4)}
    if wrong_wrist:
        a_wrists = {BodySide.LEFT: (0.32, 0.4)}
    result = [
        person("a", 0.15, a_wrists),
        person("b", 0.55, {SIDE_B: (0.53, 0.4)}),
    ]
    if third_person:
        result.append(person("c", 0.75, {BodySide.RIGHT: (0.78, 0.4)}))
    return result


def spatial(
    detections,
    *,
    sequence: int,
    observed_at: datetime,
    paper_index: int,
    paper_status: PaperEvidenceStatus = PaperEvidenceStatus.OK,
    persons=None,
):
    persons = persons if persons is not None else people()
    observations = tuple(
        observation(tid, box, points, index)
        for index, (tid, box, points) in enumerate(persons)
    )
    pair_frame = build_person_pair_frame_from_tracked_pose(
        TrackedPoseFrameResult(
            status=TrackedPoseFrameStatus.OK,
            observations=observations,
            camera_id=CAMERA_ID,
            frame_sequence=sequence,
            observed_at=observed_at,
            source_mode="live",
            source_pose_status=PoseStatus.OK,
            pose_instance_count=len(observations),
        )
    )
    paper_evidence = PaperEvidenceFrame(
        status=paper_status,
        detections=tuple(detections),
        model_name="yolo-world",
        backend="open_vocab",
        reason=None if paper_status is PaperEvidenceStatus.OK else "detector failure",
        frame_index=paper_index,
    )
    join = SameFrameJoin(
        camera_id=CAMERA_ID,
        pair_frame_sequence=sequence,
        paper_frame_index=paper_index,
        pair_observed_at=observed_at,
    )
    return build_paper_pair_spatial_frame(pair_frame, paper_evidence, join)


def temporal(
    *,
    phase: HandoffPhase = HandoffPhase.INTERACTION,
    observed_at: datetime,
    completed: bool = False,
    started_at: datetime = START_AT,
    pair: PersonPairKey = PAIR,
    generation=GENERATION,
    rule_id: str = RULE_ID,
    camera_id: str = CAMERA_ID,
    side_a: BodySide = SIDE_A,
    side_b: BodySide = SIDE_B,
) -> HandoffTemporalResult:
    return HandoffTemporalResult(
        camera_id=camera_id,
        stream_generation=generation,
        rule_id=rule_id,
        pair_key=pair,
        phase=HandoffPhase.COMPLETED if completed else phase,
        locked_side_a=side_a,
        locked_side_b=side_b,
        candidate_started_at=started_at,
        last_valid_evidence_at=observed_at,
        evidence_available_this_frame=True,
        closest_wrist_distance=0.1,
        completed_this_frame=completed,
        completed_at=observed_at if completed else None,
    )


def join_for(result: HandoffTemporalResult, sequence: int, observed_at: datetime):
    return PaperHandoffFusionJoin(
        camera_id=result.camera_id,
        stream_generation=result.stream_generation,
        rule_id=result.rule_id,
        pair_key=result.pair_key,
        pair_frame_sequence=sequence,
        pair_observed_at=observed_at,
        candidate_started_at=result.candidate_started_at,
        locked_side_a=result.locked_side_a,
        locked_side_b=result.locked_side_b,
    )


def run(
    tracker: PaperHandoffFusionTracker,
    detections,
    *,
    step: int,
    completed: bool = False,
    armed: bool = True,
    spec: PaperHandoffFusionSpec = SPEC,
    paper_status: PaperEvidenceStatus = PaperEvidenceStatus.OK,
    persons=None,
    started_at: datetime = START_AT,
    pair: PersonPairKey = PAIR,
    generation=GENERATION,
    rule_id: str = RULE_ID,
    camera_id: str = CAMERA_ID,
    side_a: BodySide = SIDE_A,
    side_b: BodySide = SIDE_B,
    observed_at: datetime | None = None,
    sequence: int | None = None,
    spatial_frame=None,
):
    moment = observed_at if observed_at is not None else at(step * 0.3)
    seq = sequence if sequence is not None else 100 + step
    result = temporal(
        observed_at=moment,
        completed=completed,
        started_at=started_at,
        pair=pair,
        generation=generation,
        rule_id=rule_id,
        camera_id=camera_id,
        side_a=side_a,
        side_b=side_b,
    )
    frame = (
        spatial_frame
        if spatial_frame is not None
        else spatial(
            detections,
            sequence=seq,
            observed_at=moment,
            paper_index=step,
            paper_status=paper_status,
            persons=persons,
        )
    )
    return tracker.observe(
        temporal=result,
        spatial_frame=frame,
        join=join_for(result, seq, moment),
        armed=armed,
        spec=spec,
    )


def complete_with_support(tracker, detections=(NEAR_PAPER,), spec=SPEC, **kwargs):
    """Three qualifying INTERACTION frames, then the Task 3D completion frame."""
    for step in range(3):
        run(tracker, detections, step=step, spec=spec, **kwargs)
    return run(tracker, detections, step=3, completed=True, spec=spec, **kwargs)


# ------------------------------------------------------------- geometry sanity


def test_geometry_fixture_places_paper_near_locked_wrists() -> None:
    frame = spatial((NEAR_PAPER,), sequence=1, observed_at=at(0), paper_index=0)
    assert frame.status is PaperPairSpatialStatus.OK
    fact = frame.facts_for_pair(PAIR)[0]
    distances = {
        (wrist.wrist_owner_tracking_id, wrist.side): wrist.distance_relative_to_mean_pair_diagonal
        for wrist in fact.wrist_facts
    }
    assert distances[("a", SIDE_A)] < 0.25
    assert distances[("b", SIDE_B)] < 0.25


# --------------------------------------------------------------- core fusion


def test_handshake_without_any_paper_never_fuses() -> None:
    """MANDATORY: full temporal completion, zero paper evidence -> no fusion."""
    tracker = PaperHandoffFusionTracker()
    result = complete_with_support(tracker, detections=())
    assert result.temporal_completed_this_frame is True
    assert result.fused_completed_this_frame is False
    assert result.paper_support_status is PaperSupportStatus.NONE


def test_paper_far_from_locked_wrists_never_fuses() -> None:
    tracker = PaperHandoffFusionTracker()
    result = complete_with_support(tracker, detections=(FAR_PAPER,))
    assert result.paper_support_status is PaperSupportStatus.PAPER_PRESENT_BUT_NOT_CORRELATED
    assert result.fused_completed_this_frame is False
    assert result.observed_total_paper_seconds == pytest.approx(0.0)


def test_paper_near_non_locked_wrist_never_fuses() -> None:
    """Locked A-right; the only available a-wrist is LEFT -> no substitution."""
    tracker = PaperHandoffFusionTracker()
    result = complete_with_support(
        tracker, detections=(NEAR_A_ONLY,), persons=people(wrong_wrist=True)
    )
    assert result.fused_completed_this_frame is False
    assert result.paper_distance_to_locked_wrist_a is None


def test_paper_correlated_to_other_pair_does_not_support_this_pair() -> None:
    tracker = PaperHandoffFusionTracker()
    near_bc = paper(BBox(0.66, 0.39, 0.02, 0.02))
    result = complete_with_support(
        tracker, detections=(near_bc,), persons=people(third_person=True)
    )
    assert result.fused_completed_this_frame is False
    assert result.paper_support_status is PaperSupportStatus.PAPER_PRESENT_BUT_NOT_CORRELATED


def test_paper_below_confidence_floor_never_fuses() -> None:
    tracker = PaperHandoffFusionTracker()
    result = complete_with_support(tracker, detections=(LOW_CONFIDENCE,))
    assert result.paper_support_status is PaperSupportStatus.NONE
    assert result.fused_completed_this_frame is False


def test_qualifying_paper_without_temporal_completion_never_fuses() -> None:
    tracker = PaperHandoffFusionTracker()
    last = None
    for step in range(4):
        last = run(tracker, (NEAR_PAPER,), step=step)
    assert last.support_qualified_this_frame is True
    assert last.observed_total_paper_seconds > 0.0
    assert last.fused_completed_this_frame is False


def test_full_completion_with_support_fuses_exactly_once() -> None:
    tracker = PaperHandoffFusionTracker()
    result = complete_with_support(tracker)
    assert result.status is PaperHandoffFusionStatus.OK
    assert result.fused_completed_this_frame is True
    assert result.observed_interaction_paper_seconds == pytest.approx(0.6)
    assert result.qualifying_paper_detection_index == 0
    assert result.qualifying_paper_raw_prompt == "sheet of paper"
    assert result.qualifying_paper_confidence == pytest.approx(0.6)
    assert result.reason is None
    # No repeat on following frames.
    for step in range(4, 7):
        again = run(tracker, (NEAR_PAPER,), step=step)
        assert again.fused_completed_this_frame is False


def test_paper_after_completion_cannot_retroactively_qualify() -> None:
    tracker = PaperHandoffFusionTracker()
    for step in range(3):
        run(tracker, (), step=step)
    closed = run(tracker, (), step=3, completed=True)
    assert closed.fused_completed_this_frame is False
    for step in range(4, 9):
        later = run(tracker, (NEAR_PAPER,), step=step)
        assert later.fused_completed_this_frame is False


def test_new_temporal_candidate_starts_with_zero_paper_history() -> None:
    tracker = PaperHandoffFusionTracker()
    for step in range(3):
        run(tracker, (NEAR_PAPER,), step=step)
    fresh = run(tracker, (NEAR_PAPER,), step=3, started_at=at(5.0))
    assert fresh.observed_total_paper_seconds == pytest.approx(0.0)
    assert fresh.observed_interaction_paper_seconds == pytest.approx(0.0)


# ---------------------------------------------------------- support mode tests


def test_either_locked_wrist_qualifies_on_one_wrist() -> None:
    tracker = PaperHandoffFusionTracker()
    result = complete_with_support(tracker, detections=(NEAR_A_ONLY,))
    assert result.paper_support_status is PaperSupportStatus.NEAR_LOCKED_WRIST_A
    assert result.fused_completed_this_frame is True


def test_both_locked_wrists_policy_rejects_single_wrist_support() -> None:
    tracker = PaperHandoffFusionTracker()
    result = complete_with_support(
        tracker, detections=(NEAR_A_ONLY,), spec=BOTH_SPEC
    )
    assert result.paper_support_status is PaperSupportStatus.PAPER_PRESENT_BUT_NOT_CORRELATED
    assert result.fused_completed_this_frame is False


def test_both_locked_wrists_policy_accepts_dual_support() -> None:
    tracker = PaperHandoffFusionTracker()
    result = complete_with_support(tracker, detections=(NEAR_PAPER,), spec=BOTH_SPEC)
    assert result.paper_support_status is PaperSupportStatus.NEAR_BOTH_LOCKED_WRISTS
    assert result.fused_completed_this_frame is True


def test_multiple_detections_resolve_deterministically_and_fuse_once() -> None:
    tracker = PaperHandoffFusionTracker()
    detections = (FAR_PAPER, NEAR_PAPER, NEAR_A_ONLY)
    result = complete_with_support(tracker, detections=detections)
    assert result.fused_completed_this_frame is True
    # Deterministic selection: nearest qualifying detection wins (index 2 here),
    # which is a frame-local choice and never a tracked paper identity.
    assert result.qualifying_paper_detection_index == 2
    repeat = run(tracker, detections, step=4, completed=True)
    assert repeat.fused_completed_this_frame is False


def test_equal_candidate_detections_break_ties_by_lowest_index() -> None:
    mirror = paper(BBox(0.415, 0.39, 0.02, 0.02))
    tracker = PaperHandoffFusionTracker()
    result = run(tracker, (NEAR_PAPER, mirror), step=0)
    assert result.qualifying_paper_detection_index == 0


# ------------------------------------------------------- unknown / dwell tests


def test_unknown_paper_evidence_pauses_observed_dwell() -> None:
    tracker = PaperHandoffFusionTracker()
    run(tracker, (NEAR_PAPER,), step=0)
    run(tracker, (NEAR_PAPER,), step=1)
    degraded = run(
        tracker, (), step=2, paper_status=PaperEvidenceStatus.MODEL_UNAVAILABLE
    )
    assert degraded.paper_support_status is PaperSupportStatus.UNKNOWN
    assert degraded.observed_total_paper_seconds == pytest.approx(0.3)
    resumed = run(tracker, (NEAR_PAPER,), step=3)
    # 0.3 observed + unknown interval skipped + anchor restart -> still 0.3.
    assert resumed.observed_total_paper_seconds == pytest.approx(0.3)
    later = run(tracker, (NEAR_PAPER,), step=4)
    assert later.observed_total_paper_seconds == pytest.approx(0.6)


def test_unknown_gap_beyond_tolerance_invalidates_corroboration() -> None:
    tracker = PaperHandoffFusionTracker()
    for step in range(3):
        run(tracker, (NEAR_PAPER,), step=step)
    gone = run(
        tracker,
        (),
        step=3,
        paper_status=PaperEvidenceStatus.MODEL_UNAVAILABLE,
        observed_at=at(6.0),
    )
    assert gone.reason == ABORT_PAPER_UNKNOWN_GAP_EXCEEDED
    assert gone.observed_total_paper_seconds == pytest.approx(0.0)
    closed = run(tracker, (NEAR_PAPER,), step=4, completed=True, observed_at=at(6.3))
    assert closed.fused_completed_this_frame is False


def test_degraded_spatial_frame_is_unknown_not_negative() -> None:
    tracker = PaperHandoffFusionTracker()
    degraded_frame = spatial(
        (NEAR_PAPER,),
        sequence=100,
        observed_at=at(0.0),
        paper_index=0,
        persons=[person("a", 0.15, {SIDE_A: (0.32, 0.4)})],
    )
    result = run(tracker, (), step=0, spatial_frame=degraded_frame)
    # A single person yields no pair facts; the required spatial fact is absent.
    assert result.paper_support_status is PaperSupportStatus.UNKNOWN


def test_valid_zero_paper_frame_is_not_unknown() -> None:
    tracker = PaperHandoffFusionTracker()
    result = run(tracker, (), step=0)
    assert result.paper_support_status is PaperSupportStatus.NONE
    assert result.last_valid_paper_evidence_at == at(0.0)


def test_valid_non_correlated_paper_is_not_unknown() -> None:
    tracker = PaperHandoffFusionTracker()
    result = run(tracker, (FAR_PAPER,), step=0)
    assert result.paper_support_status is PaperSupportStatus.PAPER_PRESENT_BUT_NOT_CORRELATED
    assert result.last_valid_paper_evidence_at == at(0.0)


def test_zero_paper_frame_breaks_qualifying_continuity() -> None:
    tracker = PaperHandoffFusionTracker()
    run(tracker, (NEAR_PAPER,), step=0)
    run(tracker, (), step=1)
    resumed = run(tracker, (NEAR_PAPER,), step=2)
    assert resumed.observed_total_paper_seconds == pytest.approx(0.0)


# ----------------------------------------------------------------- arming


def test_disarmed_sequence_with_perfect_paper_never_fuses() -> None:
    """MANDATORY pre-arm distribution: disarmed evidence cannot prime arming."""
    tracker = PaperHandoffFusionTracker()
    for step in range(4):
        pre = run(tracker, (NEAR_PAPER,), step=step, armed=False)
        assert pre.status is PaperHandoffFusionStatus.DISARMED
        assert pre.fused_completed_this_frame is False
        assert pre.observed_total_paper_seconds == pytest.approx(0.0)
    assert tracker.active_candidate_count == 0
    first_armed = run(tracker, (NEAR_PAPER,), step=4, completed=True)
    assert first_armed.observed_total_paper_seconds == pytest.approx(0.0)
    assert first_armed.fused_completed_this_frame is False


def test_disarm_mid_candidate_clears_fusion_state() -> None:
    tracker = PaperHandoffFusionTracker()
    run(tracker, (NEAR_PAPER,), step=0)
    run(tracker, (NEAR_PAPER,), step=1)
    run(tracker, (NEAR_PAPER,), step=2, armed=False)
    assert tracker.active_candidate_count == 0
    resumed = run(tracker, (NEAR_PAPER,), step=3)
    assert resumed.observed_total_paper_seconds == pytest.approx(0.0)


# ---------------------------------------------------------------- identity


def test_pairs_cameras_rules_and_generations_are_isolated() -> None:
    tracker = PaperHandoffFusionTracker()
    for step in range(3):
        run(tracker, (NEAR_PAPER,), step=step)
    other_pair = run(
        tracker, (NEAR_PAPER,), step=3, pair=PersonPairKey.of("a", "c"),
        persons=people(third_person=True),
    )
    assert other_pair.observed_total_paper_seconds == pytest.approx(0.0)
    other_camera = run(tracker, (NEAR_PAPER,), step=3, camera_id="cam-2")
    assert other_camera.observed_total_paper_seconds == pytest.approx(0.0)
    other_rule = run(tracker, (NEAR_PAPER,), step=3, rule_id="rule-other")
    assert other_rule.observed_total_paper_seconds == pytest.approx(0.0)
    other_generation = run(tracker, (NEAR_PAPER,), step=3, generation=5)
    assert other_generation.observed_total_paper_seconds == pytest.approx(0.0)


def test_tracking_id_change_does_not_migrate_evidence() -> None:
    tracker = PaperHandoffFusionTracker()
    for step in range(3):
        run(tracker, (NEAR_PAPER,), step=step)
    renamed = run(
        tracker,
        (NEAR_PAPER,),
        step=3,
        pair=PersonPairKey.of("a", "c"),
        persons=people(third_person=True),
    )
    assert renamed.observed_total_paper_seconds == pytest.approx(0.0)


def test_reset_camera_and_context_only_remove_intended_state() -> None:
    tracker = PaperHandoffFusionTracker()
    run(tracker, (NEAR_PAPER,), step=0)
    run(tracker, (NEAR_PAPER,), step=0, camera_id="cam-2")
    tracker.reset_camera("cam-2")
    assert tracker.active_candidate_count == 1
    tracker.reset_context(CAMERA_ID, RULE_ID)
    assert tracker.active_candidate_count == 0


# --------------------------------------------------------------- provenance


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"camera_id": "cam-9"}, "camera_id_mismatch"),
        ({"stream_generation": 9}, "stream_generation_mismatch"),
        ({"rule_id": "rule-9"}, "rule_id_mismatch"),
        ({"pair_key": PersonPairKey.of("a", "z")}, "pair_key_mismatch"),
        ({"candidate_started_at": at(99.0)}, "temporal_candidate_identity_mismatch"),
        ({"locked_side_a": BodySide.LEFT}, "locked_wrist_side_mismatch"),
        ({"pair_frame_sequence": 999}, "spatial_frame_provenance_mismatch"),
        ({"pair_observed_at": at(42.0)}, "spatial_observed_at_mismatch"),
    ],
)
def test_provenance_contradictions_fail_closed(kwargs, expected) -> None:
    tracker = PaperHandoffFusionTracker()
    result = temporal(observed_at=at(0.0))
    frame = spatial((NEAR_PAPER,), sequence=100, observed_at=at(0.0), paper_index=0)
    base = join_for(result, 100, at(0.0))
    fused = tracker.observe(
        temporal=result,
        spatial_frame=frame,
        join=dataclasses.replace(base, **kwargs),
        armed=True,
        spec=SPEC,
    )
    assert fused.status is PaperHandoffFusionStatus.INCONSISTENT_INPUT
    assert fused.reason == expected
    assert fused.fused_completed_this_frame is False
    assert tracker.active_candidate_count == 0


def test_duplicate_and_out_of_order_frames_fail_closed() -> None:
    tracker = PaperHandoffFusionTracker()
    run(tracker, (NEAR_PAPER,), step=1)
    duplicate = run(tracker, (NEAR_PAPER,), step=1)
    assert duplicate.status is PaperHandoffFusionStatus.NON_MONOTONIC
    assert duplicate.support_qualified_this_frame is False


# ------------------------------------------------------------ contract safety


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_paper_detector_confidence": True},
        {"min_paper_detector_confidence": 1.5},
        {"min_paper_detector_confidence": float("nan")},
        {"max_paper_to_locked_wrist_distance_relative_to_mean_diagonal": 0.0},
        {"max_paper_to_locked_wrist_distance_relative_to_mean_diagonal": float("inf")},
        {"min_interaction_paper_observed_seconds": 0.0},
        {"min_total_paper_observed_seconds": 0.1},
        {"max_paper_unknown_gap_seconds": -1.0},
        {"max_paper_unknown_gap_seconds": True},
        {"support_mode": "either"},
    ],
)
def test_invalid_specs_are_rejected(kwargs) -> None:
    with pytest.raises(PaperHandoffFusionContractError):
        dataclasses.replace(SPEC, **kwargs)


def test_results_are_immutable() -> None:
    tracker = PaperHandoffFusionTracker()
    result = run(tracker, (NEAR_PAPER,), step=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.fused_completed_this_frame = True  # type: ignore[misc]


def test_inputs_are_not_modified() -> None:
    tracker = PaperHandoffFusionTracker()
    result = temporal(observed_at=at(0.0))
    frame = spatial((NEAR_PAPER,), sequence=100, observed_at=at(0.0), paper_index=0)
    before = (repr(result), repr(frame))
    tracker.observe(
        temporal=result,
        spatial_frame=frame,
        join=join_for(result, 100, at(0.0)),
        armed=True,
        spec=SPEC,
    )
    assert (repr(result), repr(frame)) == before


def test_no_forbidden_vocabulary_or_coupling_in_fusion_sources() -> None:
    banned = (
        "giver",
        "receiver",
        "from_student",
        "to_student",
        "cheating",
        "violation_confirmed",
        "ownership",
        "holder",
        "grasp",
        "probability",
        "fused_confidence",
        "EventDraft",
        "EventPublisher",
        "snapshot",
        "telegram",
        "supabase",
        "orchestrator",
        "engine_registry",
        "train",
    )
    for path in (
        Path("app/domain/paper_handoff_fusion.py"),
        Path("app/ai/paper_handoff_fusion.py"),
    ):
        source = code_text(path).lower()
        for word in banned:
            assert not re.search(rf"\b{re.escape(word.lower())}\w*", source), (
                f"{path} must not contain {word}"
            )


def test_fusion_layer_is_not_wired_into_the_live_runtime() -> None:
    for path in Path("app").rglob("*.py"):
        if path.name == "paper_handoff_fusion.py":
            continue
        assert "paper_handoff_fusion" not in path.read_text()


# ------------------------------------------------- Task 3D lifecycle cleanup


TERMINAL_REASONS = (
    ABORT_EVIDENCE_GAP_EXCEEDED,
    ABORT_APPROACH_LOST,
    ABORT_APPROACH_TIMEOUT,
    ABORT_INTERACTION_DWELL_TOO_SHORT,
    RESET_RECOVERED,
)


def terminal_frame(tracker, reason: str, *, step: int, detections=(NEAR_PAPER,)):
    """One frame whose authoritative Task 3D result carries a terminal reason."""
    moment = at(step * 0.3)
    sequence = 100 + step
    result = dataclasses.replace(
        temporal(observed_at=moment), abort_reason=reason
    )
    frame = spatial(detections, sequence=sequence, observed_at=moment, paper_index=step)
    return tracker.observe(
        temporal=result,
        spatial_frame=frame,
        join=join_for(result, sequence, moment),
        armed=True,
        spec=SPEC,
    )


@pytest.mark.parametrize("reason", TERMINAL_REASONS)
def test_terminal_temporal_reason_discards_fusion_state(reason) -> None:
    tracker = PaperHandoffFusionTracker()
    for step in range(3):
        run(tracker, (NEAR_PAPER,), step=step)
    assert tracker.active_candidate_count == 1
    ended = terminal_frame(tracker, reason, step=3)
    assert ended.status is PaperHandoffFusionStatus.OK
    assert ended.reason == reason
    assert ended.fused_completed_this_frame is False
    assert ended.support_qualified_this_frame is False
    assert ended.paper_support_status is PaperSupportStatus.UNKNOWN
    assert tracker.active_candidate_count == 0


@pytest.mark.parametrize("reason", TERMINAL_REASONS)
def test_terminal_frame_itself_adds_no_paper_dwell(reason) -> None:
    tracker = PaperHandoffFusionTracker()
    run(tracker, (NEAR_PAPER,), step=0)
    run(tracker, (NEAR_PAPER,), step=1)
    ended = terminal_frame(tracker, reason, step=2)
    assert ended.observed_total_paper_seconds == pytest.approx(0.0)
    assert ended.observed_interaction_paper_seconds == pytest.approx(0.0)


def test_new_candidate_after_abort_starts_with_zero_paper_support() -> None:
    tracker = PaperHandoffFusionTracker()
    for step in range(3):
        run(tracker, (NEAR_PAPER,), step=step)
    terminal_frame(tracker, ABORT_APPROACH_LOST, step=3)
    fresh = run(tracker, (NEAR_PAPER,), step=4, started_at=at(1.5))
    assert fresh.observed_total_paper_seconds == pytest.approx(0.0)
    assert fresh.observed_interaction_paper_seconds == pytest.approx(0.0)
    closed = run(tracker, (NEAR_PAPER,), step=5, completed=True, started_at=at(1.5))
    assert closed.fused_completed_this_frame is False


def test_terminated_candidate_evidence_cannot_be_reused_for_completion() -> None:
    tracker = PaperHandoffFusionTracker()
    for step in range(3):
        run(tracker, (NEAR_PAPER,), step=step)
    terminal_frame(tracker, ABORT_EVIDENCE_GAP_EXCEEDED, step=3)
    revived = run(tracker, (NEAR_PAPER,), step=4, completed=True)
    assert revived.fused_completed_this_frame is False
    assert revived.observed_total_paper_seconds == pytest.approx(0.0)


def test_terminal_frame_clears_frame_order_state() -> None:
    tracker = PaperHandoffFusionTracker()
    run(tracker, (NEAR_PAPER,), step=5)
    terminal_frame(tracker, ABORT_APPROACH_TIMEOUT, step=6)
    # Order state was discarded with the candidate, so a lower sequence from the
    # next candidate is accepted instead of being read as out-of-order.
    resumed = run(tracker, (NEAR_PAPER,), step=1, started_at=at(1.0))
    assert resumed.status is PaperHandoffFusionStatus.OK


def test_new_generation_retires_stale_generation_state() -> None:
    tracker = PaperHandoffFusionTracker()
    for step in range(3):
        run(tracker, (NEAR_PAPER,), step=step, generation=4)
    assert tracker.active_candidate_count == 1
    fresh = run(tracker, (NEAR_PAPER,), step=0, generation=5)
    assert fresh.observed_total_paper_seconds == pytest.approx(0.0)
    # Generation 4 fusion state was retired, not merely isolated.
    assert tracker.active_candidate_count == 1


def test_generation_retirement_leaves_other_cameras_and_rules_untouched() -> None:
    tracker = PaperHandoffFusionTracker()
    run(tracker, (NEAR_PAPER,), step=0, generation=4)
    run(tracker, (NEAR_PAPER,), step=0, generation=4, camera_id="cam-2")
    run(tracker, (NEAR_PAPER,), step=0, generation=4, rule_id="rule-other")
    assert tracker.active_candidate_count == 3
    run(tracker, (NEAR_PAPER,), step=1, generation=5)
    assert tracker.active_candidate_count == 3
    survivor = run(
        tracker, (NEAR_PAPER,), step=1, generation=4, camera_id="cam-2"
    )
    assert survivor.observed_total_paper_seconds == pytest.approx(0.3)
