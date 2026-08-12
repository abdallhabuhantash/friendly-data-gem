"""Task 3D unknown-gap dwell accounting.

A tolerated UNKNOWN gap keeps a candidate alive with its locked wrist pair, but
UNKNOWN time is BLIND time: it can never be credited to any observed dwell
requirement (interaction, separation or post-completion recovery). Every
assertion below is driven by explicit source-frame timestamps only.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.ai.exchange_temporal_state import HandoffTemporalTracker
from app.domain.body_features import BodySide
from app.domain.handoff_temporal import (
    ABORT_EVIDENCE_GAP_EXCEEDED,
    ABORT_INTERACTION_DWELL_TOO_SHORT,
    HandoffPhase,
    HandoffTemporalStatus,
)

from tests.test_exchange_temporal_state import (
    PAIR,
    SPEC,
    at,
    degraded_frame,
    frame,
    gap_frame,
    observe,
)

TOL = 1e-9


def missing_pair_frame(*, sequence: int, observed_at):
    """Only ONE tracked person: the locked pair is absent -> UNKNOWN."""
    return frame(
        [("a", {BodySide.RIGHT: (0.45, 0.4)})],
        sequence=sequence,
        observed_at=observed_at,
    )


def missing_wrist_frame(*, sequence: int, observed_at):
    """Both people present, but the locked wrist side is absent -> UNKNOWN."""
    return frame(
        [
            ("a", {BodySide.LEFT: (0.45, 0.4)}),
            ("b", {BodySide.RIGHT: (0.85, 0.4)}),
        ],
        sequence=sequence,
        observed_at=observed_at,
    )


def near_step(tracker, sequence: int, seconds: float, relative: float = 0.1):
    return observe(
        tracker, gap_frame(relative, sequence=sequence, observed_at=at(seconds))
    )


def start_interaction(tracker) -> None:
    """seq1 far approach start -> seq2 near: INTERACTION with 0.0 observed dwell."""
    near_step(tracker, 1, 0.0, 0.8)
    result = near_step(tracker, 2, 0.5, 0.1)
    candidate = result.candidate(PAIR)
    assert candidate is not None
    assert candidate.phase is HandoffPhase.INTERACTION
    assert candidate.interaction_duration_seconds == pytest.approx(0.0, abs=TOL)


# ------------------------------------------- A. unknown never funds interaction


@pytest.mark.parametrize(
    "unknown_frame_factory",
    [degraded_frame, missing_pair_frame, missing_wrist_frame],
    ids=["degraded_frame", "missing_pair", "missing_wrist"],
)
def test_unknown_gap_does_not_fund_interaction_dwell(unknown_frame_factory) -> None:
    """A. 0.2s valid + 0.9s UNKNOWN + 0.2s valid != a 1.0s interaction dwell.

    E. degraded frames and missing pair/wrist frames obey identical semantics.
    """
    tracker = HandoffTemporalTracker()
    start_interaction(tracker)

    observed = near_step(tracker, 3, 0.7).candidate(PAIR)
    assert observed.interaction_duration_seconds == pytest.approx(0.2, abs=1e-6)

    # UNKNOWN interval: last valid evidence 0.7 -> next valid 1.6 (0.9s blind).
    blind = observe(tracker, unknown_frame_factory(sequence=4, observed_at=at(1.2)))
    blind_candidate = blind.candidate(PAIR)
    assert blind_candidate is not None, "a tolerated gap keeps the candidate alive"
    assert blind_candidate.evidence_available_this_frame is False
    assert blind_candidate.phase is HandoffPhase.INTERACTION
    assert blind_candidate.locked_side_a is BodySide.RIGHT
    assert blind_candidate.locked_side_b is BodySide.LEFT
    assert blind_candidate.interaction_duration_seconds == pytest.approx(0.2, abs=1e-6)

    resumed = near_step(tracker, 5, 1.6).candidate(PAIR)
    assert resumed.interaction_duration_seconds == pytest.approx(0.2, abs=1e-6)
    assert resumed.locked_side_a is BodySide.RIGHT

    after = near_step(tracker, 6, 1.8).candidate(PAIR)
    # Wall time since interaction start is 1.3s; observed dwell is only 0.4s.
    assert (at(1.8) - after.interaction_started_at) == timedelta(seconds=1.3)
    assert after.interaction_duration_seconds == pytest.approx(0.4, abs=1e-6)
    assert after.completed_this_frame is False

    # The dwell requirement is NOT satisfied: separating aborts the candidate.
    aborted = observe(
        tracker, gap_frame(0.7, sequence=7, observed_at=at(2.0))
    ).candidate(PAIR)
    assert aborted.abort_reason == ABORT_INTERACTION_DWELL_TOO_SHORT
    assert tracker.active_candidate_count == 0


# ------------------------------- B. genuine valid evidence eventually satisfies


def test_additional_valid_evidence_eventually_satisfies_interaction_dwell() -> None:
    """B. The same candidate completes once 1.0s of REAL near evidence exists."""
    tracker = HandoffTemporalTracker()
    start_interaction(tracker)
    near_step(tracker, 3, 0.7)
    observe(tracker, degraded_frame(sequence=4, observed_at=at(1.2)))
    near_step(tracker, 5, 1.6)
    near_step(tracker, 6, 1.8)

    # 0.4s observed so far; add continuous valid near evidence up to 1.0s.
    mid = near_step(tracker, 7, 2.4).candidate(PAIR)
    assert mid.interaction_duration_seconds == pytest.approx(1.0, abs=1e-6)
    assert mid.phase is HandoffPhase.INTERACTION

    # Separation with genuinely observed dwell then completes exactly once.
    separating = observe(
        tracker, gap_frame(0.7, sequence=8, observed_at=at(2.6))
    ).candidate(PAIR)
    assert separating.phase is HandoffPhase.SEPARATING
    assert separating.abort_reason is None
    assert separating.separation_duration_seconds == pytest.approx(0.0, abs=TOL)

    completed = observe(tracker, gap_frame(0.7, sequence=9, observed_at=at(3.8)))
    result = completed.candidate(PAIR)
    assert result.completed_this_frame is True
    assert result.phase is HandoffPhase.COMPLETED
    assert result.separation_duration_seconds == pytest.approx(1.2, abs=1e-6)
    assert result.interaction_duration_seconds == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------ C. unknown never funds separation


def _reach_separating(tracker) -> None:
    """Approach -> 1.0s genuinely observed interaction dwell -> SEPARATING."""
    near_step(tracker, 1, 0.0, 0.8)
    near_step(tracker, 2, 0.5, 0.1)
    near_step(tracker, 3, 1.5, 0.1)
    separating = observe(
        tracker, gap_frame(0.7, sequence=4, observed_at=at(1.7))
    ).candidate(PAIR)
    assert separating.phase is HandoffPhase.SEPARATING
    assert separating.interaction_duration_seconds == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "unknown_frame_factory",
    [degraded_frame, missing_pair_frame, missing_wrist_frame],
    ids=["degraded_frame", "missing_pair", "missing_wrist"],
)
def test_unknown_gap_does_not_fund_separation_dwell(unknown_frame_factory) -> None:
    """C + E. Blind time cannot complete a separation dwell."""
    tracker = HandoffTemporalTracker()
    _reach_separating(tracker)

    first = observe(tracker, gap_frame(0.7, sequence=5, observed_at=at(1.9)))
    assert first.candidate(PAIR).separation_duration_seconds == pytest.approx(
        0.2, abs=1e-6
    )

    blind = observe(tracker, unknown_frame_factory(sequence=6, observed_at=at(2.4)))
    blind_candidate = blind.candidate(PAIR)
    assert blind_candidate is not None
    assert blind_candidate.phase is HandoffPhase.SEPARATING
    assert blind_candidate.separation_duration_seconds == pytest.approx(0.2, abs=1e-6)

    # 0.9s blind interval (1.9 -> 2.8) is tolerated but never credited.
    resumed = observe(tracker, gap_frame(0.7, sequence=7, observed_at=at(2.8)))
    resumed_candidate = resumed.candidate(PAIR)
    assert resumed_candidate.completed_this_frame is False
    assert resumed_candidate.separation_duration_seconds == pytest.approx(0.2, abs=1e-6)

    still_open = observe(tracker, gap_frame(0.7, sequence=8, observed_at=at(3.2)))
    assert still_open.candidate(PAIR).completed_this_frame is False
    assert still_open.candidate(PAIR).separation_duration_seconds == pytest.approx(
        0.6, abs=1e-6
    )

    # Only real observed separation time completes the sequence.
    finished = observe(tracker, gap_frame(0.7, sequence=9, observed_at=at(3.8)))
    assert finished.candidate(PAIR).completed_this_frame is True
    assert finished.candidate(PAIR).separation_duration_seconds == pytest.approx(
        1.2, abs=1e-6
    )


# -------------------------------------------- D. unknown never funds recovery


def _reach_completed(tracker) -> None:
    _reach_separating(tracker)
    completed = observe(tracker, gap_frame(0.7, sequence=5, observed_at=at(2.9)))
    assert completed.candidate(PAIR).completed_this_frame is True


@pytest.mark.parametrize(
    "unknown_frame_factory",
    [degraded_frame, missing_pair_frame, missing_wrist_frame],
    ids=["degraded_frame", "missing_pair", "missing_wrist"],
)
def test_unknown_gap_does_not_fund_recovery_dwell(unknown_frame_factory) -> None:
    """D + E. Blind time cannot satisfy the post-completion recovery dwell."""
    tracker = HandoffTemporalTracker()
    _reach_completed(tracker)

    observe(tracker, gap_frame(0.9, sequence=6, observed_at=at(3.1)))
    assert tracker.active_candidate_count == 1

    blind = observe(tracker, unknown_frame_factory(sequence=7, observed_at=at(3.6)))
    assert blind.candidate(PAIR) is not None
    assert tracker.active_candidate_count == 1

    # 0.9s blind (3.1 -> 4.0) is tolerated; recovery dwell is still unsatisfied.
    resumed = observe(tracker, gap_frame(0.9, sequence=8, observed_at=at(4.0)))
    assert resumed.candidate(PAIR).abort_reason is None
    assert tracker.active_candidate_count == 1

    partial = observe(tracker, gap_frame(0.9, sequence=9, observed_at=at(4.4)))
    assert partial.candidate(PAIR).abort_reason is None

    recovered = observe(tracker, gap_frame(0.9, sequence=10, observed_at=at(5.1)))
    assert recovered.candidate(PAIR).abort_reason is not None
    assert tracker.active_candidate_count == 0


# --------------------------------------- F. exceeding the gap still aborts


@pytest.mark.parametrize(
    "unknown_frame_factory",
    [degraded_frame, missing_pair_frame, missing_wrist_frame],
    ids=["degraded_frame", "missing_pair", "missing_wrist"],
)
def test_gap_beyond_tolerance_still_aborts(unknown_frame_factory) -> None:
    """F. A gap over max_unknown_gap_seconds aborts exactly as before."""
    tracker = HandoffTemporalTracker()
    start_interaction(tracker)
    near_step(tracker, 3, 0.7)

    blind = observe(tracker, unknown_frame_factory(sequence=4, observed_at=at(1.2)))
    assert blind.candidate(PAIR) is not None

    aborted = observe(tracker, unknown_frame_factory(sequence=5, observed_at=at(2.0)))
    candidate = aborted.candidate(PAIR)
    assert candidate is not None
    assert candidate.abort_reason == ABORT_EVIDENCE_GAP_EXCEEDED
    assert candidate.phase is HandoffPhase.IDLE
    assert candidate.evidence_available_this_frame is False
    assert tracker.active_candidate_count == 0
    assert aborted.status in (
        HandoffTemporalStatus.OK,
        HandoffTemporalStatus.DEGRADED_FRAME,
    )
