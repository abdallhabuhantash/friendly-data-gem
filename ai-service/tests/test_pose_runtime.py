"""Deterministic tests for the optional, asynchronous Pose runtime.

Nothing here loads a pose model: the provider is a controllable fake, so every
assertion is about scheduling, stream-incarnation safety and the invariant that
pose can never become a synchronous dependency of Task 1 phone detection.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.domain.geometry import BBox
from app.domain.observations import FrameObservations, PersonObservation
from app.domain.pose import PoseFrameResult, PoseStatus
from app.domain.pose_association import PoseAssociationSpec
from app.runtime.pose_runtime import ASSOCIATION_UNCONFIGURED, PoseJob, PoseRuntime


class FakeClock:
    """Explicit monotonic clock: no sleeps, no wall-clock flakiness."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeProvider:
    available = True
    model_name = "/private/path/yolo11n-pose.pt"

    def __init__(self, result: PoseFrameResult | None = None) -> None:
        self.calls: list[object] = []
        self.result = result or PoseFrameResult(status=PoseStatus.OK)
        self.gate: threading.Event | None = None
        self.entered = threading.Event()

    def infer(self, frame):  # noqa: ANN001
        self.calls.append(frame)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(2.0)
        return self.result


class ExplodingProvider(FakeProvider):
    def infer(self, frame):  # noqa: ANN001
        raise RuntimeError("secret-model-path-should-never-leak")


def observations(camera_id: str = "cam-a") -> FrameObservations:
    return FrameObservations(
        camera_id=camera_id,
        persons=(
            PersonObservation(
                person_tracking_id="t1",
                person_bbox=BBox(0.1, 0.1, 0.2, 0.4),
                confidence=0.9,
            ),
        ),
        frame_sequence=1,
    )


def job(camera_id: str, generation: int, sequence: int) -> PoseJob:
    return PoseJob(
        camera_id=camera_id,
        generation=generation,
        frame_sequence=sequence,
        observed_at=None,
        observations=observations(camera_id),
        frame=f"frame-{camera_id}-{sequence}",
    )


def runtime(provider=None, interval: float = 0.0, spec=None, clock=None) -> PoseRuntime:
    return PoseRuntime(
        provider or FakeProvider(),
        min_interval_seconds=interval,
        association_spec=spec,
        clock=clock or FakeClock(),
    )


# --- activation / generation gating -------------------------------------
def test_submission_requires_an_activated_generation() -> None:
    pose = runtime()
    assert pose.submit(job("cam-a", 1, 1)) is False
    assert pose.pending_count() == 0

    pose.activate("cam-a", 1)
    assert pose.submit(job("cam-a", 1, 2)) is True
    assert pose.pending_count() == 1


def test_job_from_previous_incarnation_is_refused() -> None:
    pose = runtime()
    pose.activate("cam-a", 2)
    assert pose.submit(job("cam-a", 1, 1)) is False


# --- latest-job-wins ----------------------------------------------------
def test_only_the_newest_pending_job_per_camera_survives() -> None:
    pose = runtime()
    pose.activate("cam-a", 1)
    for sequence in (1, 2, 3):
        assert pose.submit(job("cam-a", 1, sequence)) is True

    assert pose.pending_count() == 1
    assert pose.pending_frame_sequence("cam-a") == 3
    metrics = pose.metrics("cam-a")
    assert metrics["submitted"] == 3
    assert metrics["replaced_pending"] == 2


def test_pending_slots_are_independent_per_camera() -> None:
    pose = runtime()
    pose.activate("cam-a", 1)
    pose.activate("cam-b", 1)
    pose.submit(job("cam-a", 1, 5))
    pose.submit(job("cam-b", 1, 9))
    assert pose.pending_count() == 2
    assert pose.pending_frame_sequence("cam-a") == 5
    assert pose.pending_frame_sequence("cam-b") == 9


def test_round_robin_serves_cameras_fairly() -> None:
    provider = FakeProvider()
    pose = runtime(provider)
    pose.activate("cam-a", 1)
    pose.activate("cam-b", 1)
    pose.submit(job("cam-a", 1, 1))
    pose.submit(job("cam-b", 1, 1))

    assert pose.process_pending_once() is True
    assert pose.process_pending_once() is True
    assert pose.process_pending_once() is False
    served = [str(frame) for frame in provider.calls]
    assert served == ["frame-cam-a-1", "frame-cam-b-1"]

    # The next scan starts after the camera served last.
    pose.submit(job("cam-a", 1, 2))
    pose.submit(job("cam-b", 1, 2))
    pose.process_pending_once()
    assert str(provider.calls[-1]) == "frame-cam-a-2"


# --- cadence ------------------------------------------------------------
def test_cadence_skips_frames_and_never_copies_them() -> None:
    clock = FakeClock()
    pose = runtime(interval=1.0, clock=clock)
    pose.activate("cam-a", 1)
    copies: list[int] = []

    def submit(sequence: int) -> bool:
        return pose.maybe_submit(
            camera_id="cam-a",
            generation=1,
            frame_sequence=sequence,
            observed_at=None,
            observations=observations(),
            copy_frame=lambda: copies.append(sequence) or f"copy-{sequence}",
        )

    assert submit(1) is True
    assert submit(2) is False
    clock.advance(0.5)
    assert submit(3) is False
    clock.advance(0.5)
    assert submit(4) is True

    assert copies == [1, 4]
    assert pose.metrics("cam-a")["cadence_skipped"] == 2


def test_cadence_refuses_inactive_camera_without_copying() -> None:
    pose = runtime(interval=0.0)
    copied = []
    admitted = pose.maybe_submit(
        camera_id="cam-x",
        generation=1,
        frame_sequence=1,
        observed_at=None,
        observations=observations("cam-x"),
        copy_frame=lambda: copied.append(1),
    )
    assert admitted is False
    assert copied == []


# --- incarnation safety -------------------------------------------------
def test_result_of_ended_incarnation_is_discarded() -> None:
    provider = FakeProvider()
    pose = runtime(provider)
    pose.activate("cam-a", 1)
    pose.submit(job("cam-a", 1, 1))
    # Reconfiguration happens while the job sits in the worker.
    pose.reset_camera("cam-a")
    pose.activate("cam-a", 2)

    assert pose.process_pending_once() is False
    assert pose.latest_result("cam-a") is None


def test_inflight_result_cannot_be_stored_under_the_new_generation() -> None:
    provider = FakeProvider()
    pose = runtime(provider)
    pose.activate("cam-a", 1)
    pending = job("cam-a", 1, 7)
    pose.submit(pending)
    taken = pose._next_job()  # noqa: SLF001 - deterministic worker step
    assert taken is not None

    pose.reset_camera("cam-a")
    pose.activate("cam-a", 2)
    pose._run_job(taken)  # noqa: SLF001

    assert pose.latest_result("cam-a") is None
    # Stale accounting stays OUT of the new incarnation's metrics.
    assert pose.stale_discards("cam-a") == 1
    metrics = pose.metrics("cam-a")
    assert metrics["active_generation"] == 2
    assert metrics["processed"] == 0



def test_activate_same_generation_twice_keeps_state() -> None:
    pose = runtime()
    pose.activate("cam-a", 1)
    pose.submit(job("cam-a", 1, 4))
    pose.activate("cam-a", 1)
    assert pose.pending_frame_sequence("cam-a") == 4


def test_deactivate_removes_camera_entirely() -> None:
    pose = runtime()
    pose.activate("cam-a", 1)
    pose.submit(job("cam-a", 1, 1))
    pose.deactivate("cam-a")
    assert pose.pending_count() == 0
    assert pose.active_generation("cam-a") is None
    assert pose.metrics("cam-a") is None
    assert pose.status()["cameras"] == {}


def test_deactivate_of_one_camera_leaves_others_untouched() -> None:
    pose = runtime()
    pose.activate("cam-a", 1)
    pose.activate("cam-b", 1)
    pose.submit(job("cam-b", 1, 3))
    pose.deactivate("cam-a")
    assert pose.pending_frame_sequence("cam-b") == 3
    assert pose.process_pending_once() is True


# --- results, association, diagnostics ----------------------------------
def test_result_is_cached_with_frame_identity_and_measured_timing() -> None:
    clock = FakeClock()
    pose = runtime(FakeProvider(), clock=clock)
    pose.activate("cam-a", 1)
    pose.submit(job("cam-a", 1, 12))
    pose.process_pending_once()

    result = pose.latest_result("cam-a")
    assert result is not None
    assert result.generation == 1
    assert result.frame_sequence == 12
    assert result.association is None
    assert result.association_state == ASSOCIATION_UNCONFIGURED
    metrics = pose.metrics("cam-a")
    assert metrics["processed"] == 1
    assert metrics["last_pose_frame_sequence"] == 12


def test_association_runs_only_when_configured() -> None:
    spec = PoseAssociationSpec(
        min_bbox_iou=0.3,
        min_pose_bbox_containment=0.6,
        min_available_keypoints=5,
        min_keypoint_inside_ratio=0.6,
    )
    pose = runtime(FakeProvider(), spec=spec)
    assert pose.association_configured is True
    pose.activate("cam-a", 1)
    pose.submit(job("cam-a", 1, 1))
    pose.process_pending_once()

    result = pose.latest_result("cam-a")
    assert result is not None
    assert result.association is not None
    assert result.association_state == "ok"


def test_degraded_pose_result_is_counted_as_a_provider_failure() -> None:
    provider = FakeProvider(
        PoseFrameResult(status=PoseStatus.MODEL_UNAVAILABLE, reason="unavailable")
    )
    pose = runtime(provider)
    pose.activate("cam-a", 1)
    pose.submit(job("cam-a", 1, 1))
    pose.process_pending_once()
    assert pose.metrics("cam-a")["provider_failures"] == 1


def test_status_never_exposes_the_model_path() -> None:
    pose = runtime(FakeProvider())
    status = pose.status()
    assert status["model"] == "yolo11n-pose.pt"
    assert "/private/path" not in str(status)


# --- worker resilience --------------------------------------------------
def test_worker_survives_a_failing_provider() -> None:
    pose = runtime(ExplodingProvider())
    pose.activate("cam-a", 1)
    pose.start()
    try:
        pose.submit(job("cam-a", 1, 1))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            metrics = pose.metrics("cam-a")
            if metrics and metrics["worker_errors"] == 1:
                break
            time.sleep(0.01)
        assert pose.metrics("cam-a")["worker_errors"] == 1
        assert pose.worker_running is True
    finally:
        pose.stop(timeout=2.0)
    assert pose.worker_running is False


def test_submission_never_blocks_on_inference() -> None:
    """Task 1's frame path must return while pose inference is still running."""
    provider = FakeProvider()
    provider.gate = threading.Event()
    pose = runtime(provider)
    pose.activate("cam-a", 1)
    pose.start()
    try:
        pose.submit(job("cam-a", 1, 1))
        assert provider.entered.wait(2.0) is True

        started = time.monotonic()
        assert pose.submit(job("cam-a", 1, 2)) is True
        assert (time.monotonic() - started) < 0.5
        assert pose.pending_frame_sequence("cam-a") == 2
    finally:
        provider.gate.set()
        pose.stop(timeout=2.0)


def test_global_inference_concurrency_is_one() -> None:
    provider = FakeProvider()
    provider.gate = threading.Event()
    pose = runtime(provider)
    pose.activate("cam-a", 1)
    pose.activate("cam-b", 1)
    pose.start()
    try:
        pose.submit(job("cam-a", 1, 1))
        assert provider.entered.wait(2.0) is True
        pose.submit(job("cam-b", 1, 1))
        # While one inference is gated, no second inference may start.
        time.sleep(0.1)
        assert len(provider.calls) == 1
    finally:
        provider.gate.set()
        pose.stop(timeout=2.0)


def test_negative_interval_is_rejected() -> None:
    with pytest.raises(ValueError):
        PoseRuntime(FakeProvider(), min_interval_seconds=-1.0)
