"""Asynchronous, optional Pose runtime (infrastructure + diagnostics only).

Primary invariant
-----------------
POSE IS NEVER A SYNCHRONOUS DEPENDENCY OF TASK 1 PHONE DETECTION. The Task 1
frame path may only *submit* work here: a cadence check, at most one frame copy,
a small immutable job and a worker wake-up. Model inference happens exclusively
on this module's own dedicated worker thread, outside any camera lifecycle lock.

Scheduling
----------
* ONE dedicated worker for the ONE shared pose model (global pose inference
  concurrency = 1 for this MVP; it matches the provider's shared model + lock
  and stops N cameras from launching N expensive pose calls at once).
* At most ONE pending job per camera: a newer eligible frame REPLACES the older
  pending frame, which is intentionally dropped. Memory bound:
  ``pending jobs <= active cameras``. There is no FIFO backlog.
* Deterministic round-robin over activated cameras, so a continuously
  submitting camera cannot starve another camera's pending job.

Stream incarnation
------------------
A pose result belongs to ``(camera_id, generation)``. A result produced by
generation N is discarded when the camera has moved to N+1 or been removed, so
no pose state ever crosses an incarnation boundary.

Deliberately absent: regions, wrist/head features, concealed-device activity,
temporal behaviour, behaviour scores, evidence, events, notifications and any
database write. Nothing here concludes anything about behaviour.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from ..ai.pose_person_matcher import associate_pose_frame
from ..domain.observations import FrameObservations
from ..domain.pose import PoseFrameResult, PoseStatus
from ..domain.pose_association import (
    PoseAssociationFrameResult,
    PoseAssociationSpec,
)

logger = logging.getLogger(__name__)

#: Reported when pose inference ran but no association spec is configured.
ASSOCIATION_UNCONFIGURED = "association_unconfigured"


@dataclass(frozen=True, slots=True)
class PoseJob:
    """Immutable description of ONE frame admitted for pose processing."""

    camera_id: str
    generation: int
    frame_sequence: Optional[int]
    observed_at: Optional[datetime]
    #: Person observations built from the SAME detector frame as ``frame``.
    observations: FrameObservations
    #: Independent frame copy owned by the pose runtime.
    frame: Any
    source_mode: Optional[str] = None
    submitted_monotonic: Optional[float] = None


@dataclass(frozen=True, slots=True)
class PoseRuntimeResult:
    """Immutable latest-pose snapshot for one camera incarnation."""

    camera_id: str
    generation: int
    frame_sequence: Optional[int]
    observed_at: Optional[datetime]
    completed_at: datetime
    pose: PoseFrameResult
    association: Optional[PoseAssociationFrameResult] = None
    association_state: str = ASSOCIATION_UNCONFIGURED
    inference_ms: float = 0.0
    latency_ms: Optional[float] = None


@dataclass
class _CameraMetrics:
    """Measured per-camera diagnostics for ONE camera incarnation.

    Nothing produced by an ended incarnation is ever counted here: stale
    completions are accounted separately by the runtime.
    """

    submitted: int = 0
    processed: int = 0
    replaced_pending: int = 0
    cadence_skipped: int = 0
    provider_failures: int = 0
    association_degraded: int = 0
    worker_errors: int = 0
    last_inference_ms: Optional[float] = None
    last_completed_at: Optional[datetime] = None
    last_frame_sequence: Optional[int] = None
    _window_start: Optional[float] = field(default=None, repr=False)
    _window_count: int = field(default=0, repr=False)
    measured_pose_fps: Optional[float] = None

    def snapshot(self, generation: Optional[int]) -> dict:
        return {
            "active_generation": generation,
            "submitted": self.submitted,
            "processed": self.processed,
            "replaced_pending": self.replaced_pending,
            "cadence_skipped": self.cadence_skipped,
            "provider_failures": self.provider_failures,
            "association_degraded": self.association_degraded,
            "worker_errors": self.worker_errors,
            "last_pose_inference_ms": (
                round(self.last_inference_ms, 2) if self.last_inference_ms is not None else None
            ),
            "last_pose_completed_at": (
                self.last_completed_at.isoformat() if self.last_completed_at else None
            ),
            "last_pose_frame_sequence": self.last_frame_sequence,
            "measured_pose_fps": (
                round(self.measured_pose_fps, 2) if self.measured_pose_fps is not None else None
            ),
        }



class PoseRuntime:
    """Owns the pose worker, the per-camera pending slot and the result cache."""

    def __init__(
        self,
        provider,  # noqa: ANN001 - PoseProvider protocol
        *,
        min_interval_seconds: float,
        association_spec: Optional[PoseAssociationSpec] = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = None,  # type: ignore[assignment]
    ) -> None:
        if min_interval_seconds < 0.0:
            raise ValueError("min_interval_seconds must be >= 0")
        self._provider = provider
        self._min_interval = float(min_interval_seconds)
        self._spec = association_spec
        self._clock = clock
        self._wall_clock = wall_clock or (lambda: datetime.now())
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        #: Guards worker thread creation/teardown only (never held over inference).
        self._thread_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._generations: dict[str, int] = {}
        self._order: list[str] = []
        self._cursor = 0
        self._pending: dict[str, PoseJob] = {}
        self._results: dict[str, PoseRuntimeResult] = {}
        self._last_submitted: dict[str, float] = {}
        self._metrics: dict[str, _CameraMetrics] = {}
        #: Completions of ended incarnations, kept OUT of per-camera metrics.
        self._stale_discards: int = 0
        self._stale_by_camera: dict[str, int] = {}

        self._stop_timed_out: bool = False

    # --- lifecycle --------------------------------------------------------
    @property
    def association_configured(self) -> bool:
        return self._spec is not None

    @property
    def worker_running(self) -> bool:
        """True while a worker thread for THIS instance is genuinely alive."""
        with self._thread_lock:
            thread = self._thread
            if thread is None:
                return False
            if thread.is_alive():
                return True
            # The worker finished: forget the dead handle so start() may run again.
            self._thread = None
            return False

    def start(self) -> None:
        """Duplicate-safe: at most ONE live worker thread per instance."""
        with self._thread_lock:
            thread = self._thread
            if thread is not None:
                if thread.is_alive():
                    # A previous worker is still alive (possibly blocked inside a
                    # provider call after a timed-out stop). Never start a second.
                    return
                self._thread = None
            self._stop.clear()
            self._stop_timed_out = False
            worker = threading.Thread(target=self._worker_loop, name="pose", daemon=True)
            self._thread = worker
            worker.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Signals shutdown and joins with a bound; never lies about liveness."""
        self._stop.set()
        self._wake.set()
        with self._thread_lock:
            thread = self._thread
            if thread is None:
                # Nothing alive: any earlier timeout condition is resolved.
                self._stop_timed_out = False
        if thread is None:
            return

        thread.join(timeout=timeout)
        with self._thread_lock:
            if thread.is_alive():
                # Truthful: the worker is still inside a provider call. It will
                # observe the stop flag and exit once that call returns; the
                # handle stays so start() cannot create a second worker.
                self._stop_timed_out = True
                logger.warning(
                    "Pose worker still running after stop timeout of %.1fs "
                    "(shutdown signalled; no second worker will be started)",
                    timeout,
                )
                return
            self._stop_timed_out = False
            if self._thread is thread:
                self._thread = None


    def activate(self, camera_id: str, generation: int) -> None:
        """Marks one camera incarnation as the only one allowed to store state."""
        with self._lock:
            if self._generations.get(camera_id) != generation:
                self._discard_locked(camera_id)
            self._generations[camera_id] = generation
            if camera_id not in self._order:
                self._order.append(camera_id)
            self._metrics.setdefault(camera_id, _CameraMetrics())

    def deactivate(self, camera_id: str) -> None:
        """Removes a camera entirely: pending job, result, cadence, generation."""
        with self._lock:
            self._discard_locked(camera_id)
            self._generations.pop(camera_id, None)
            self._metrics.pop(camera_id, None)
            self._stale_by_camera.pop(camera_id, None)

            if camera_id in self._order:
                index = self._order.index(camera_id)
                self._order.remove(camera_id)
                if self._cursor > index:
                    self._cursor -= 1

    #: Old name kept intentionally narrow; reset == invalidate this incarnation.
    def reset_camera(self, camera_id: str) -> None:
        """Drops incarnation-specific state but keeps the camera registered."""
        with self._lock:
            self._discard_locked(camera_id)
            self._generations.pop(camera_id, None)
            self._metrics[camera_id] = _CameraMetrics()

    def _discard_locked(self, camera_id: str) -> None:
        self._pending.pop(camera_id, None)
        self._results.pop(camera_id, None)
        self._last_submitted.pop(camera_id, None)

    # --- submission (Task 1 side; must stay cheap) -------------------------
    def cadence_admits(self, camera_id: str, generation: int) -> bool:
        """Pure cadence/activation check: no frame copy, no allocation."""
        with self._lock:
            return self._cadence_admits_locked(camera_id, generation)

    def _cadence_admits_locked(self, camera_id: str, generation: int) -> bool:
        if self._generations.get(camera_id) != generation:
            return False
        last = self._last_submitted.get(camera_id)
        if last is None:
            return True
        return (self._clock() - last) >= self._min_interval

    def maybe_submit(
        self,
        *,
        camera_id: str,
        generation: int,
        frame_sequence: Optional[int],
        observed_at: Optional[datetime],
        observations: FrameObservations,
        copy_frame: Callable[[], Any],
        source_mode: Optional[str] = None,
    ) -> bool:
        """Cadence-gated submission. Copies the frame ONLY when admitted.

        Cadence accounting is *reserved* around the copy and rolled back when no
        job is actually accepted, so a failed copy or an ended incarnation never
        consumes the camera's next pose cadence slot.
        """
        with self._lock:
            if not self._cadence_admits_locked(camera_id, generation):
                metrics = self._metrics.get(camera_id)
                if metrics is not None:
                    metrics.cadence_skipped += 1
                return False
            previous = self._last_submitted.get(camera_id)
            reserved = self._clock()
            self._last_submitted[camera_id] = reserved

        accepted = False
        try:
            # Exactly one intentional frame copy, for an admitted frame only.
            frame = copy_frame()
            accepted = self.submit(
                PoseJob(
                    camera_id=camera_id,
                    generation=generation,
                    frame_sequence=frame_sequence,
                    observed_at=observed_at,
                    observations=observations,
                    frame=frame,
                    source_mode=source_mode,
                    submitted_monotonic=self._clock(),
                )
            )
            return accepted
        finally:
            if not accepted:
                self._release_reservation(camera_id, reserved, previous)

    def _release_reservation(
        self, camera_id: str, reserved: float, previous: Optional[float]
    ) -> None:
        """Rolls the cadence reservation back, but only if still ours."""
        with self._lock:
            if self._last_submitted.get(camera_id) != reserved:
                # A newer accepted submission (or a reset) already replaced it.
                return
            if previous is None:
                self._last_submitted.pop(camera_id, None)
            else:
                self._last_submitted[camera_id] = previous


    def submit(self, job: PoseJob) -> bool:
        """Stores the job in this camera's single pending slot (newest wins)."""
        with self._lock:
            if self._generations.get(job.camera_id) != job.generation:
                return False
            metrics = self._metrics.setdefault(job.camera_id, _CameraMetrics())
            if job.camera_id in self._pending:
                metrics.replaced_pending += 1
            self._pending[job.camera_id] = job
            metrics.submitted += 1
        self._wake.set()
        return True

    # --- worker -----------------------------------------------------------
    def _next_job(self) -> Optional[PoseJob]:
        """Deterministic round-robin pick over activated cameras."""
        with self._lock:
            if not self._pending or not self._order:
                return None
            count = len(self._order)
            for offset in range(count):
                index = (self._cursor + offset) % count
                camera_id = self._order[index]
                job = self._pending.pop(camera_id, None)
                if job is None:
                    continue
                # Next scan starts AFTER the camera just served.
                self._cursor = (index + 1) % count
                return job
            return None

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            job = self._next_job()
            if job is None:
                self._wake.wait(0.05)
                self._wake.clear()
                continue
            try:
                self._run_job(job)
            except Exception as error:  # noqa: BLE001 - worker must stay alive
                # Safe context only: never frame contents, URLs or raw messages.
                logger.warning(
                    "Pose job failed (camera=%s generation=%s frame=%s %s)",
                    job.camera_id,
                    job.generation,
                    job.frame_sequence,
                    type(error).__name__,
                )
                with self._lock:
                    metrics = self._metrics.get(job.camera_id)
                    if metrics is not None:
                        metrics.worker_errors += 1

    def process_pending_once(self) -> bool:
        """Test/diagnostic hook: runs at most one pending job on this thread."""
        job = self._next_job()
        if job is None:
            return False
        self._run_job(job)
        return True

    def _run_job(self, job: PoseJob) -> None:
        started = self._clock()
        # Inference runs here: no camera lifecycle lock, no runtime lock held.
        pose = self._provider.infer(job.frame)
        inference_ms = (self._clock() - started) * 1000.0

        association: Optional[PoseAssociationFrameResult] = None
        state = ASSOCIATION_UNCONFIGURED
        if self._spec is not None:
            # Association uses ONLY the observations captured with this frame.
            association = associate_pose_frame(
                pose_result=pose, observations=job.observations, spec=self._spec
            )
            state = association.status.value

        completed_at = self._wall_clock()
        latency_ms = None
        if job.submitted_monotonic is not None:
            latency_ms = (self._clock() - job.submitted_monotonic) * 1000.0

        result = PoseRuntimeResult(
            camera_id=job.camera_id,
            generation=job.generation,
            frame_sequence=job.frame_sequence,
            observed_at=job.observed_at,
            completed_at=completed_at,
            pose=pose,
            association=association,
            association_state=state,
            inference_ms=inference_ms,
            latency_ms=latency_ms,
        )

        with self._lock:
            if self._generations.get(job.camera_id) != job.generation:
                # The incarnation ended while pose was running: discard, never
                # publish a generation-N result as generation-N+1 state, and
                # never attribute this old work to the NEW incarnation's
                # per-camera metrics. Stale accounting is kept separately.
                self._stale_discards += 1
                self._stale_by_camera[job.camera_id] = (
                    self._stale_by_camera.get(job.camera_id, 0) + 1
                )
                return
            metrics = self._metrics.setdefault(job.camera_id, _CameraMetrics())

            self._results[job.camera_id] = result
            metrics.processed += 1
            metrics.last_inference_ms = inference_ms
            metrics.last_completed_at = completed_at
            metrics.last_frame_sequence = job.frame_sequence
            if pose.status is not PoseStatus.OK:
                metrics.provider_failures += 1
            if association is not None and not association.ok:
                metrics.association_degraded += 1
            self._record_fps_locked(metrics)

    def _record_fps_locked(self, metrics: _CameraMetrics) -> None:
        """Genuinely measured pose throughput; never capture FPS."""
        now = self._clock()
        if metrics._window_start is None:
            metrics._window_start = now
            metrics._window_count = 1
            return
        metrics._window_count += 1
        elapsed = now - metrics._window_start
        if elapsed >= 2.0:
            metrics.measured_pose_fps = metrics._window_count / elapsed
            metrics._window_start = now
            metrics._window_count = 0

    # --- introspection ----------------------------------------------------
    def latest_result(self, camera_id: str) -> Optional[PoseRuntimeResult]:
        with self._lock:
            return self._results.get(camera_id)

    def pending_frame_sequence(self, camera_id: str) -> Optional[int]:
        with self._lock:
            job = self._pending.get(camera_id)
            return job.frame_sequence if job else None

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def active_generation(self, camera_id: str) -> Optional[int]:
        with self._lock:
            return self._generations.get(camera_id)

    def metrics(self, camera_id: str) -> Optional[dict]:
        with self._lock:
            metrics = self._metrics.get(camera_id)
            if metrics is None:
                return None
            return metrics.snapshot(self._generations.get(camera_id))

    def stale_discards(self, camera_id: Optional[str] = None) -> int:
        """Completions of ENDED incarnations; never part of camera metrics."""
        with self._lock:
            if camera_id is None:
                return self._stale_discards
            return self._stale_by_camera.get(camera_id, 0)

    def status(self) -> dict:
        with self._lock:
            cameras = {
                camera_id: metrics.snapshot(self._generations.get(camera_id))
                for camera_id, metrics in self._metrics.items()
            }
            pending = len(self._pending)
            stale = self._stale_discards
            stale_by_camera = dict(self._stale_by_camera)
        return {
            "enabled": True,
            "configured": True,
            "worker_running": self.worker_running,
            "stop_timed_out": self._stop_timed_out,
            "provider_available": bool(getattr(self._provider, "available", False)),
            "model": getattr(self._provider, "model_name", None) and _basename(
                getattr(self._provider, "model_name")
            ),
            "association_configured": self.association_configured,
            "pending_jobs": pending,
            "stale_discards": stale,
            "stale_discards_by_camera": stale_by_camera,
            "cameras": cameras,
        }



def _basename(value: str) -> str:
    """Model file name only: never a full, possibly sensitive path."""
    import os

    return os.path.basename(value) or value


__all__ = [
    "ASSOCIATION_UNCONFIGURED",
    "PoseJob",
    "PoseRuntime",
    "PoseRuntimeResult",
]
