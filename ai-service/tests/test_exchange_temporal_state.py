"""Deterministic tests for the Task 3D handoff temporal state machine.

All timing is driven by explicit ``observed_at`` timestamps. There is no
``time.sleep``, no wall clock and no frame-count assumption anywhere.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.ai.exchange_temporal_state import HandoffTemporalTracker
from app.ai.person_pair_geometry_builder import build_person_pair_frame_from_tracked_pose
from app.domain.body_features import BodySide
from app.domain.geometry import BBox
from app.domain.handoff_temporal import (
    ABORT_APPROACH_LOST,
    ABORT_EVIDENCE_GAP_EXCEEDED,
    ABORT_INTERACTION_DWELL_TOO_SHORT,
    RESET_RECOVERED,
    HandoffPhase,
    HandoffTemporalContractError,
    HandoffTemporalSpec,
    HandoffTemporalStatus,
)
from app.domain.pair_geometry import PersonPairKey
from app.domain.pose import COCO_17_KEYPOINTS, PoseKeypointName, PoseStatus, coco_17_index
from app.domain.regions import RelativePoint
from app.domain.tracked_pose_observations import (
    TrackedPoseFrameResult,
    TrackedPoseFrameStatus,
    TrackedPoseKeypoint,
    TrackedPoseObservation,
)

START = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
BOX_WIDTH = 0.2
BOX_HEIGHT = 0.4
#: Both people use identical box sizes, so the symmetric mean person diagonal is
#: this value and a wrist gap dx maps to dx / MEAN_DIAGONAL.
MEAN_DIAGONAL = math.hypot(BOX_WIDTH, BOX_HEIGHT)

SPEC = HandoffTemporalSpec(
    max_person_center_distance=1.5,
    approach_start_wrist_distance=0.9,
    interaction_wrist_distance=0.2,
    min_approach_distance_reduction=0.3,
    min_interaction_dwell_seconds=1.0,
    max_unknown_gap_seconds=1.0,
    min_separation_wrist_distance=0.6,
    min_separation_distance_increase=0.3,
    min_separation_dwell_seconds=1.0,
    recovery_wrist_distance=0.8,
    recovery_dwell_seconds=1.0,
)

PAIR = PersonPairKey.of("a", "b")


def dx_for(relative: float) -> float:
    return relative * MEAN_DIAGONAL


def at(seconds: float) -> datetime:
    return START + timedelta(seconds=seconds)


def _relative(point: tuple[float, float], box: BBox) -> RelativePoint:
    return RelativePoint(
        relative_x=(point[0] - box.x) / box.width,
        relative_y=(point[1] - box.y) / box.height,
    )


def _observation(
    tracking_id: str,
    box: BBox,
    wrists: dict[BodySide, tuple[float, float]],
    index: int,
) -> TrackedPoseObservation:
    points = {
        (
            PoseKeypointName.LEFT_WRIST
            if side is BodySide.LEFT
            else PoseKeypointName.RIGHT_WRIST
        ): point
        for side, point in wrists.items()
    }
    keypoints = []
    for name in COCO_17_KEYPOINTS:
        keypoint_index = coco_17_index(name)
        if name in points:
            x, y = points[name]
            rel = _relative((x, y), box)
            keypoints.append(
                TrackedPoseKeypoint(
                    name=name,
                    index=keypoint_index,
                    available=True,
                    confidence=0.8,
                    x=x,
                    y=y,
                    relative_position=rel,
                    inside_person=rel.inside_person,
                )
            )
        else:
            keypoints.append(
                TrackedPoseKeypoint(name=name, index=keypoint_index, available=False)
            )
    return TrackedPoseObservation(
        person_tracking_id=tracking_id,
        person_index=index,
        pose_index=index,
        person_bbox=box,
        person_confidence=0.7,
        pose_bbox=box,
        pose_confidence=0.9,
        keypoints=tuple(keypoints),
    )


def box_for(index: int) -> BBox:
    return BBox(0.1 + 0.4 * index, 0.2, BOX_WIDTH, BOX_HEIGHT)


def frame(
    people: list[tuple[str, dict[BodySide, tuple[float, float]]]],
    *,
    sequence: int,
    observed_at: datetime,
    camera_id: str = "cam-1",
):
    observations = tuple(
        _observation(tracking_id, box_for(index), wrists, index)
        for index, (tracking_id, wrists) in enumerate(people)
    )
    return build_person_pair_frame_from_tracked_pose(
        TrackedPoseFrameResult(
            status=TrackedPoseFrameStatus.OK,
            observations=observations,
            camera_id=camera_id,
            frame_sequence=sequence,
            observed_at=observed_at,
            source_mode="live",
            source_pose_status=PoseStatus.OK,
            pose_instance_count=len(observations),
        )
    )


def degraded_frame(*, sequence: int, observed_at: datetime, camera_id: str = "cam-1"):
    return build_person_pair_frame_from_tracked_pose(
        TrackedPoseFrameResult(
            status=TrackedPoseFrameStatus.POSE_UNAVAILABLE,
            camera_id=camera_id,
            frame_sequence=sequence,
            observed_at=observed_at,
            source_mode="live",
            source_pose_status=PoseStatus.MODEL_UNAVAILABLE,
            reason="pose model unavailable",
        )
    )


def gap_frame(
    relative: float,
    *,
    sequence: int,
    observed_at: datetime,
    camera_id: str = "cam-1",
    sides: tuple[BodySide, BodySide] = (BodySide.RIGHT, BodySide.LEFT),
    ids: tuple[str, str] = ("a", "b"),
):
    """One two-person frame whose locked wrist gap equals ``relative``."""
    ax = 0.28
    return frame(
        [
            (ids[0], {sides[0]: (ax, 0.4)}),
            (ids[1], {sides[1]: (ax + dx_for(relative), 0.4)}),
        ],
        sequence=sequence,
        observed_at=observed_at,
        camera_id=camera_id,
    )


def observe(
    tracker: HandoffTemporalTracker,
    frame_result,
    *,
    rule_id: str = "rule-1",
    generation: int = 4,
    armed: bool = True,
    spec: HandoffTemporalSpec = SPEC,
):
    return tracker.observe(
        frame_result,
        rule_id=rule_id,
        stream_generation=generation,
        armed=armed,
        spec=spec,
    )


def run(
    tracker: HandoffTemporalTracker,
    steps: list[tuple[int, float, float]],
    **kwargs,
):
    """Runs (sequence, seconds, relative wrist distance) steps in order."""
    results = []
    for sequence, seconds, relative in steps:
        results.append(
            observe(
                tracker,
                gap_frame(relative, sequence=sequence, observed_at=at(seconds)),
                **kwargs,
            )
        )
    return results


#: A deterministic full sequence: approach -> interaction dwell -> separation.
FULL_SEQUENCE = [
    (1, 0.0, 0.8),
    (2, 1.0, 0.5),
    (3, 2.0, 0.1),
    (4, 3.5, 0.1),
    (5, 4.5, 0.7),
    (6, 6.0, 0.7),
]


# ------------------------------------------------------------- core sequence


def test_idle_with_far_wrists_creates_no_candidate() -> None:
    tracker = HandoffTemporalTracker()
    result = observe(tracker, gap_frame(2.0, sequence=1, observed_at=at(0)))
    assert result.status is HandoffTemporalStatus.OK
    assert result.candidates == ()
    assert tracker.active_candidate_count == 0


def test_static_close_wrists_never_complete() -> None:
    tracker = HandoffTemporalTracker()
    for index in range(10):
        result = run(tracker, [(index + 1, float(index), 0.05)])[0]
        assert result.completed == ()
    assert tracker.active_candidate_count == 0


def test_single_close_frame_never_completes() -> None:
    tracker = HandoffTemporalTracker()
    result = observe(tracker, gap_frame(0.05, sequence=1, observed_at=at(0)))
    assert result.completed == ()
    assert result.candidates == ()


def test_valid_approach_starts_candidate() -> None:
    tracker = HandoffTemporalTracker()
    result = run(tracker, [(1, 0.0, 0.8)])[0]
    candidate = result.candidate(PAIR)
    assert candidate is not None
    assert candidate.phase is HandoffPhase.APPROACHING
    assert candidate.completed_this_frame is False


def test_insufficient_reduction_does_not_reach_interaction() -> None:
    strict = dataclasses.replace(SPEC, min_approach_distance_reduction=0.85)
    tracker = HandoffTemporalTracker()
    results = run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1)], spec=strict)
    assert results[-1].candidate(PAIR).phase is HandoffPhase.APPROACHING


def test_valid_approach_reaches_interaction() -> None:
    tracker = HandoffTemporalTracker()
    results = run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1)])
    candidate = results[-1].candidate(PAIR)
    assert candidate.phase is HandoffPhase.INTERACTION
    assert candidate.interaction_started_at == at(1.0)


def test_interaction_shorter_than_dwell_aborts() -> None:
    tracker = HandoffTemporalTracker()
    results = run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1), (3, 1.2, 0.7)])
    candidate = results[-1].candidate(PAIR)
    assert candidate.abort_reason == ABORT_INTERACTION_DWELL_TOO_SHORT
    assert tracker.active_candidate_count == 0


def test_interaction_dwell_satisfied_moves_to_separating() -> None:
    tracker = HandoffTemporalTracker()
    results = run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1), (3, 2.5, 0.1), (4, 3.0, 0.7)])
    candidate = results[-1].candidate(PAIR)
    assert candidate.phase is HandoffPhase.SEPARATING
    assert candidate.interaction_duration_seconds == pytest.approx(1.5)


def test_interaction_without_separation_never_completes() -> None:
    tracker = HandoffTemporalTracker()
    steps = [(index + 1, float(index), 0.1) for index in range(8)]
    results = run(tracker, [(1, 0.0, 0.8)] + [(s + 1, t + 1.0, d) for s, t, d in steps])
    assert all(result.completed == () for result in results)


def test_separation_shorter_than_dwell_does_not_complete() -> None:
    tracker = HandoffTemporalTracker()
    results = run(
        tracker,
        [(1, 0.0, 0.8), (2, 1.0, 0.1), (3, 2.5, 0.1), (4, 3.0, 0.7), (5, 3.4, 0.7)],
    )
    assert all(result.completed == () for result in results)
    assert results[-1].candidate(PAIR).phase is HandoffPhase.SEPARATING


def test_full_sequence_completes_exactly_once() -> None:
    tracker = HandoffTemporalTracker()
    results = run(tracker, FULL_SEQUENCE)
    completions = [result for result in results if result.completed]
    assert len(completions) == 1
    completed = completions[0].completed[0]
    assert completed.phase is HandoffPhase.COMPLETED
    assert completed.completed_at == at(6.0)
    assert completed.locked_side_a is BodySide.RIGHT
    assert completed.locked_side_b is BodySide.LEFT


def test_completion_is_not_repeated_on_later_frames() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, FULL_SEQUENCE)
    later = run(tracker, [(7, 6.5, 0.7), (8, 7.0, 0.7)])
    assert all(result.completed == () for result in later)


def test_recovery_allows_a_genuinely_new_later_sequence() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, FULL_SEQUENCE)
    recovery = run(tracker, [(7, 7.0, 1.2), (8, 9.0, 1.2)])
    assert recovery[-1].candidate(PAIR).abort_reason == RESET_RECOVERED
    assert tracker.active_candidate_count == 0
    second = run(
        tracker,
        [
            (9, 10.0, 0.8),
            (10, 11.0, 0.1),
            (11, 12.5, 0.1),
            (12, 13.5, 0.7),
            (13, 15.0, 0.7),
        ],
    )
    assert sum(len(result.completed) for result in second) == 1


# ------------------------------------------------------------ missing evidence


def test_brief_locked_wrist_loss_is_a_tolerated_unknown_gap() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1)])
    blind = observe(
        tracker,
        frame(
            [("a", {}), ("b", {})],
            sequence=3,
            observed_at=at(1.5),
        ),
    )
    candidate = blind.candidate(PAIR)
    assert candidate.evidence_available_this_frame is False
    assert candidate.phase is HandoffPhase.INTERACTION
    assert candidate.abort_reason is None


def test_locked_wrist_missing_beyond_gap_aborts() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1)])
    blind = observe(
        tracker,
        frame([("a", {}), ("b", {})], sequence=3, observed_at=at(3.0)),
    )
    assert blind.candidate(PAIR).abort_reason == ABORT_EVIDENCE_GAP_EXCEEDED
    assert tracker.active_candidate_count == 0


def test_other_wrist_is_never_substituted_for_the_locked_pair() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1)])
    # The locked A-right/B-left combination disappears; a very close
    # A-left/B-right pair appears instead and MUST NOT be used.
    substitute = observe(
        tracker,
        frame(
            [
                ("a", {BodySide.LEFT: (0.28, 0.4)}),
                ("b", {BodySide.RIGHT: (0.29, 0.4)}),
            ],
            sequence=3,
            observed_at=at(1.4),
        ),
    )
    candidate = substitute.candidate(PAIR)
    assert candidate.locked_side_a is BodySide.RIGHT
    assert candidate.locked_side_b is BodySide.LEFT
    assert candidate.evidence_available_this_frame is False


def test_pair_absent_briefly_is_unknown_not_separation() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1)])
    absent = observe(
        tracker,
        frame([("a", {BodySide.RIGHT: (0.28, 0.4)})], sequence=3, observed_at=at(1.5)),
    )
    assert absent.candidate(PAIR).phase is HandoffPhase.INTERACTION
    assert tracker.active_candidate_count == 1


def test_pair_absent_too_long_removes_state() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1)])
    observe(
        tracker,
        frame([("a", {BodySide.RIGHT: (0.28, 0.4)})], sequence=3, observed_at=at(4.0)),
    )
    assert tracker.active_candidate_count == 0


def test_brief_degraded_frame_does_not_advance_but_is_tolerated() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1)])
    result = observe(tracker, degraded_frame(sequence=3, observed_at=at(1.5)))
    assert result.status is HandoffTemporalStatus.DEGRADED_FRAME
    candidate = result.candidate(PAIR)
    assert candidate.phase is HandoffPhase.INTERACTION
    assert candidate.interaction_duration_seconds == pytest.approx(0.0)


def test_long_degraded_period_aborts_candidate() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1)])
    result = observe(tracker, degraded_frame(sequence=3, observed_at=at(5.0)))
    assert result.candidate(PAIR).abort_reason == ABORT_EVIDENCE_GAP_EXCEEDED
    assert tracker.active_candidate_count == 0


def test_approach_lost_aborts_candidate() -> None:
    tracker = HandoffTemporalTracker()
    results = run(tracker, [(1, 0.0, 0.8), (2, 1.0, 1.5)])
    assert results[-1].candidate(PAIR).abort_reason == ABORT_APPROACH_LOST


# -------------------------------------------------------------------- arming


def test_default_construction_is_not_armed_and_disarmed_never_completes() -> None:
    tracker = HandoffTemporalTracker()
    results = run(tracker, FULL_SEQUENCE, armed=False)
    assert all(
        result.status is HandoffTemporalStatus.DISARMED and result.completed == ()
        for result in results
    )
    assert tracker.active_candidate_count == 0


def test_disarmed_sequence_cannot_prime_post_arm_detection() -> None:
    """Professor-like paper distribution while disarmed must leave nothing behind."""
    tracker = HandoffTemporalTracker()
    run(tracker, FULL_SEQUENCE, armed=False)
    assert tracker.active_candidate_count == 0
    # Arming starts clean from IDLE: a separation-looking frame proves nothing.
    first_armed = observe(tracker, gap_frame(0.7, sequence=7, observed_at=at(7.0)))
    assert first_armed.completed == ()
    assert first_armed.candidates == ()


def test_disarm_during_approaching_clears_candidate() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8)])
    assert tracker.active_candidate_count == 1
    observe(tracker, gap_frame(0.8, sequence=2, observed_at=at(1.0)), armed=False)
    assert tracker.active_candidate_count == 0


def test_disarm_during_interaction_clears_and_cannot_resume() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1), (3, 2.5, 0.1)])
    observe(tracker, gap_frame(0.1, sequence=4, observed_at=at(3.0)), armed=False)
    assert tracker.active_candidate_count == 0
    rearmed = run(tracker, [(5, 4.0, 0.7), (6, 5.5, 0.7)])
    assert all(result.completed == () for result in rearmed)


# ------------------------------------------------------------ identity scopes


def test_pairs_are_independent() -> None:
    tracker = HandoffTemporalTracker()
    ax = 0.28
    people = [
        ("a", {BodySide.RIGHT: (ax, 0.4)}),
        ("b", {BodySide.LEFT: (ax + dx_for(0.8), 0.4)}),
        ("c", {BodySide.LEFT: (ax + dx_for(3.0), 0.4)}),
    ]
    result = observe(tracker, frame(people, sequence=1, observed_at=at(0)))
    keys = {candidate.pair_key for candidate in result.candidates}
    assert keys == {PersonPairKey.of("a", "b")}
    assert tracker.active_candidate_count == 1


def test_two_cameras_with_the_same_tracking_ids_are_isolated() -> None:
    tracker = HandoffTemporalTracker()
    observe(tracker, gap_frame(0.8, sequence=1, observed_at=at(0), camera_id="cam-1"))
    observe(tracker, gap_frame(0.8, sequence=1, observed_at=at(0), camera_id="cam-2"))
    assert tracker.active_candidate_count == 2
    tracker.reset_camera("cam-1")
    assert tracker.active_candidate_count == 1


def test_two_rules_on_one_camera_are_isolated() -> None:
    tracker = HandoffTemporalTracker()
    observe(tracker, gap_frame(0.8, sequence=1, observed_at=at(0)), rule_id="rule-1")
    observe(tracker, gap_frame(0.8, sequence=1, observed_at=at(0)), rule_id="rule-2")
    assert tracker.active_candidate_count == 2
    tracker.reset_context("cam-1", "rule-2")
    assert tracker.active_candidate_count == 1


def test_generation_change_discards_old_candidate() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1), (3, 2.5, 0.1)], generation=4)
    later = run(tracker, [(4, 3.5, 0.7), (5, 5.0, 0.7)], generation=5)
    assert all(result.completed == () for result in later)
    assert all(key[1] == 5 for key in tracker._candidates)


def test_tracking_id_change_does_not_migrate_state() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1)])
    renamed = observe(
        tracker,
        gap_frame(0.1, sequence=3, observed_at=at(2.0), ids=("a", "z")),
    )
    assert renamed.candidate(PersonPairKey.of("a", "z")) is None
    assert renamed.candidate(PAIR).evidence_available_this_frame is False


def test_reset_camera_and_reset_context_are_narrow() -> None:
    tracker = HandoffTemporalTracker()
    observe(tracker, gap_frame(0.8, sequence=1, observed_at=at(0), camera_id="cam-1"))
    observe(tracker, gap_frame(0.8, sequence=1, observed_at=at(0), camera_id="cam-2"))
    observe(
        tracker,
        gap_frame(0.8, sequence=1, observed_at=at(0), camera_id="cam-2"),
        rule_id="rule-2",
    )
    tracker.reset_context("cam-2", "rule-2")
    assert tracker.active_candidate_count == 2
    tracker.reset_camera("cam-2")
    assert tracker.active_candidate_count == 1


# --------------------------------------------------------------- frame order


def test_duplicate_frame_does_not_advance_duration() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1), (3, 2.5, 0.1)])
    duplicate = observe(tracker, gap_frame(0.1, sequence=3, observed_at=at(2.5)))
    assert duplicate.status is HandoffTemporalStatus.NON_MONOTONIC
    assert tracker.active_candidate_count == 0


def test_older_sequence_is_rejected() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(5, 0.0, 0.8)])
    older = observe(tracker, gap_frame(0.1, sequence=4, observed_at=at(1.0)))
    assert older.status is HandoffTemporalStatus.NON_MONOTONIC


def test_equal_timestamp_with_newer_sequence_is_rejected() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8)])
    same_time = observe(tracker, gap_frame(0.1, sequence=2, observed_at=at(0.0)))
    assert same_time.status is HandoffTemporalStatus.NON_MONOTONIC


def test_older_timestamp_is_rejected() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 1.0, 0.8)])
    older = observe(tracker, gap_frame(0.1, sequence=2, observed_at=at(0.5)))
    assert older.status is HandoffTemporalStatus.NON_MONOTONIC


def test_skipped_sequence_with_increasing_time_stays_valid() -> None:
    tracker = HandoffTemporalTracker()
    results = run(tracker, [(1, 0.0, 0.8), (900, 1.0, 0.1)])
    assert results[-1].status is HandoffTemporalStatus.OK
    assert results[-1].candidate(PAIR).phase is HandoffPhase.INTERACTION


def test_variable_cadence_uses_timestamps_not_frame_counts() -> None:
    tracker = HandoffTemporalTracker()
    # Three near frames in 0.3s: far below the 1.0s dwell -> abort, no completion.
    fast = run(
        tracker,
        [(1, 0.0, 0.8), (2, 1.0, 0.1), (3, 1.1, 0.1), (4, 1.2, 0.1), (5, 1.3, 0.7)],
    )
    assert all(result.completed == () for result in fast)
    # Two near frames spanning 1.5s satisfy the same dwell.
    slow = HandoffTemporalTracker()
    ok = run(slow, FULL_SEQUENCE)
    assert sum(len(result.completed) for result in ok) == 1


def test_invalid_frame_metadata_is_reported_and_resets() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8)])
    from app.domain.pair_geometry import PersonPairFrameResult

    naked = PersonPairFrameResult(status=HandoffPhaseFrameStatus := None) if False else None
    bare = PersonPairFrameResult(
        status=__import__(
            "app.domain.pair_geometry", fromlist=["PairFrameStatus"]
        ).PairFrameStatus.OK,
        camera_id="cam-1",
    )
    result = observe(tracker, bare)
    assert result.status is HandoffTemporalStatus.INVALID_INPUT
    assert tracker.active_candidate_count == 0
    assert naked is None


# ---------------------------------------------------------------- wrist lock


def test_candidate_locks_a_deterministic_wrist_pair() -> None:
    tracker = HandoffTemporalTracker()
    ax = 0.28
    result = observe(
        tracker,
        frame(
            [
                (
                    "a",
                    {
                        BodySide.LEFT: (ax - 0.05, 0.4),
                        BodySide.RIGHT: (ax, 0.4),
                    },
                ),
                (
                    "b",
                    {
                        BodySide.LEFT: (ax + dx_for(0.8), 0.4),
                        BodySide.RIGHT: (ax + dx_for(1.4), 0.4),
                    },
                ),
            ],
            sequence=1,
            observed_at=at(0),
        ),
    )
    candidate = result.candidate(PAIR)
    assert (candidate.locked_side_a, candidate.locked_side_b) == (
        BodySide.RIGHT,
        BodySide.LEFT,
    )


def test_equal_distance_selection_is_stable_by_side_name() -> None:
    tracker = HandoffTemporalTracker()
    ax = 0.28
    # A-left and A-right are equidistant from B-left: LEFT wins by side ordering.
    result = observe(
        tracker,
        frame(
            [
                ("a", {BodySide.LEFT: (ax, 0.35), BodySide.RIGHT: (ax, 0.45)}),
                ("b", {BodySide.LEFT: (ax + dx_for(0.8), 0.4)}),
            ],
            sequence=1,
            observed_at=at(0),
        ),
    )
    candidate = result.candidate(PAIR)
    assert candidate.locked_side_a is BodySide.LEFT


def test_nearer_other_wrist_does_not_switch_the_locked_pair() -> None:
    tracker = HandoffTemporalTracker()
    ax = 0.28
    observe(tracker, gap_frame(0.8, sequence=1, observed_at=at(0)))
    result = observe(
        tracker,
        frame(
            [
                ("a", {BodySide.RIGHT: (ax, 0.4), BodySide.LEFT: (ax + 0.001, 0.4)}),
                (
                    "b",
                    {
                        BodySide.LEFT: (ax + dx_for(0.8), 0.4),
                        BodySide.RIGHT: (ax + 0.002, 0.4),
                    },
                ),
            ],
            sequence=2,
            observed_at=at(1.0),
        ),
    )
    candidate = result.candidate(PAIR)
    assert (candidate.locked_side_a, candidate.locked_side_b) == (
        BodySide.RIGHT,
        BodySide.LEFT,
    )
    assert candidate.current_wrist_distance == pytest.approx(0.8, abs=1e-6)


def test_reset_allows_a_different_wrist_pair_later() -> None:
    tracker = HandoffTemporalTracker()
    observe(tracker, gap_frame(0.8, sequence=1, observed_at=at(0)))
    observe(tracker, gap_frame(1.5, sequence=2, observed_at=at(1.0)))  # abort
    assert tracker.active_candidate_count == 0
    result = observe(
        tracker,
        gap_frame(
            0.8,
            sequence=3,
            observed_at=at(2.0),
            sides=(BodySide.LEFT, BodySide.RIGHT),
        ),
    )
    candidate = result.candidate(PAIR)
    assert (candidate.locked_side_a, candidate.locked_side_b) == (
        BodySide.LEFT,
        BodySide.RIGHT,
    )


# ------------------------------------------------------------------ contract


@pytest.mark.parametrize(
    "field",
    [
        "max_person_center_distance",
        "approach_start_wrist_distance",
        "interaction_wrist_distance",
        "min_interaction_dwell_seconds",
        "min_separation_dwell_seconds",
        "recovery_dwell_seconds",
    ],
)
def test_spec_rejects_bool_masquerading_as_numeric(field: str) -> None:
    with pytest.raises(HandoffTemporalContractError):
        dataclasses.replace(SPEC, **{field: True})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0, "0.5", None])
def test_spec_rejects_non_finite_and_non_numeric(value) -> None:
    with pytest.raises(HandoffTemporalContractError):
        dataclasses.replace(SPEC, interaction_wrist_distance=value)


def test_spec_rejects_invalid_threshold_ordering() -> None:
    with pytest.raises(HandoffTemporalContractError):
        dataclasses.replace(SPEC, interaction_wrist_distance=0.95)
    with pytest.raises(HandoffTemporalContractError):
        dataclasses.replace(SPEC, min_separation_wrist_distance=0.1)
    with pytest.raises(HandoffTemporalContractError):
        dataclasses.replace(SPEC, recovery_wrist_distance=0.3)


def test_spec_has_no_default_behavioural_thresholds() -> None:
    required = {
        name
        for name, field in HandoffTemporalSpec.__dataclass_fields__.items()
        if field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    }
    assert "max_approach_duration_seconds" not in required
    assert len(required) == 11


def test_results_are_immutable_and_input_frame_unchanged() -> None:
    tracker = HandoffTemporalTracker()
    source = gap_frame(0.8, sequence=1, observed_at=at(0))
    before = dataclasses.astuple(source.pairs[0].key)
    result = observe(tracker, source)
    candidate = result.candidate(PAIR)
    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate.phase = HandoffPhase.COMPLETED  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = HandoffTemporalStatus.OK  # type: ignore[misc]
    assert dataclasses.astuple(source.pairs[0].key) == before


def test_state_memory_is_bounded_scalars_only() -> None:
    tracker = HandoffTemporalTracker()
    run(tracker, [(1, 0.0, 0.8), (2, 1.0, 0.1), (3, 2.5, 0.1)])
    candidate = next(iter(tracker._candidates.values()))
    for value in vars(candidate).values():
        assert not isinstance(value, (list, dict, set, tuple))


def test_module_makes_no_forbidden_claims() -> None:
    root = Path(__file__).resolve().parents[1] / "app"
    sources = (
        (root / "ai" / "exchange_temporal_state.py").read_text(),
        (root / "domain" / "handoff_temporal.py").read_text(),
    )
    for text in sources:
        lowered = text.lower()
        for forbidden in (
            "paper_present",
            "document_present",
            "sheet_present",
            "paper_confidence",
            "giver",
            "receiver",
            "eventdraft",
            "eventpublisher",
            "supabase",
            "telegram",
            "notification",
            "snapshot_service",
            "time.time(",
            "time.sleep",
            "monotonic",
            "threading",
            "orchestrator",
            "engineregistry",
            "confidence_score",
        ):
            assert forbidden not in lowered, forbidden


def test_temporal_module_does_not_import_task_one_state_or_runtime() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "ai"
        / "exchange_temporal_state.py"
    )
    tree = ast.parse(path.read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("temporal_state" == name.split(".")[-1] for name in imported)
    assert not any(
        part in name
        for name in imported
        for part in ("runtime", "events", "notifications", "infrastructure", "camera")
    )
