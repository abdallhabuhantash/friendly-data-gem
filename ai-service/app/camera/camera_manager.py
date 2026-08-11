"""Keeps running capture workers in sync with the console configuration.

Stream incarnations
-------------------
The same camera UUID may represent several sequential *stream incarnations*:
camera A generation 1 reads source X, generation 2 reads source Y. Every time a
worker is created for a camera id, that camera's generation is incremented, so
consumers can detect an incarnation boundary explicitly instead of guessing from
frame sequence numbers or object identity.

`snapshot(camera_id)` returns worker + config + generation read together under
one lock, so an inference loop can never mix a new worker with a stale
generation. `lock(camera_id)` hands out one narrow per-camera lifecycle lock:
camera A's transition never blocks camera B's work.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Iterable, Optional

from ..domain.models import CameraConfig
from .capture_worker import CaptureWorker
from .source_builder import build_source

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraRuntime:
    """Atomic view of one camera's current stream incarnation."""

    camera_id: str
    generation: int
    worker: CaptureWorker
    config: CameraConfig


class CameraManager:
    """Starts, stops and replaces capture workers as configuration changes."""

    def __init__(self, settings, credentials) -> None:  # noqa: ANN001 - injected
        self._settings = settings
        self._credentials = credentials
        self._workers: dict[str, CaptureWorker] = {}
        self._configs: dict[str, CameraConfig] = {}
        self._signatures: dict[str, tuple] = {}
        self._generations: dict[str, int] = {}
        self._state_lock = threading.RLock()
        self._camera_locks: dict[str, threading.RLock] = {}

    @staticmethod
    def _signature(camera: CameraConfig) -> tuple:
        return (
            camera.source_type.value,
            camera.host,
            camera.rtsp_port,
            camera.stream_path,
            camera.channel,
        )

    # --- per-camera synchronisation ---------------------------------------
    def lock(self, camera_id: str) -> threading.RLock:
        """One lifecycle lock per camera id; stable for the process lifetime.

        Deliberately NOT a global lock: serialising every camera would let one
        slow camera stall the others. The lock object is kept even after the
        camera is removed, so a late inference thread and a later incarnation
        still synchronise on the same object.
        """
        with self._state_lock:
            existing = self._camera_locks.get(camera_id)
            if existing is None:
                existing = threading.RLock()
                self._camera_locks[camera_id] = existing
            return existing

    def generation(self, camera_id: str) -> Optional[int]:
        with self._state_lock:
            if camera_id not in self._workers:
                return None
            return self._generations.get(camera_id)

    def snapshot(self, camera_id: str) -> Optional[CameraRuntime]:
        """Worker, config and generation read together, never mixed."""
        with self._state_lock:
            worker = self._workers.get(camera_id)
            config = self._configs.get(camera_id)
            generation = self._generations.get(camera_id)
            if worker is None or config is None or generation is None:
                return None
            return CameraRuntime(
                camera_id=camera_id, generation=generation, worker=worker, config=config
            )

    def sync(self, cameras: Iterable[CameraConfig]) -> set[str]:
        """Syncs workers and reports camera ids whose source was replaced.

        A camera is "reconfigured" when a running worker for the SAME camera id
        is replaced because its capture-affecting signature changed. Harmless
        metadata edits (name, location, ...) never appear here, because they are
        not part of `_signature`. Callers use the returned ids to reset runtime
        state that belongs to the previous stream incarnation.
        """
        desired = {camera.id: camera for camera in cameras}
        reconfigured: set[str] = set()

        with self._state_lock:
            running = list(self._workers)
        for camera_id in running:
            if camera_id not in desired:
                self.stop_camera(camera_id)
                continue
            with self._state_lock:
                changed = self._signatures.get(camera_id) != self._signature(desired[camera_id])
            if changed:
                logger.info("Camera %s source signature changed; replacing worker", camera_id)
                reconfigured.add(camera_id)
                self.stop_camera(camera_id)

        for camera_id, camera in desired.items():
            with self._state_lock:
                self._configs[camera_id] = camera
                already_running = camera_id in self._workers
            if already_running:
                continue
            username, password = self._credentials.get(camera_id)
            source = build_source(
                camera,
                username=username,
                password=password,
                demo_video_path=self._settings.demo_video_for(camera_id),
                demo_loop=self._settings.demo_video_loop,
            )
            if source is None:
                logger.warning(
                    "Camera %s (%s) has no usable source; skipping", camera.name, camera_id
                )
                continue
            worker = CaptureWorker(camera_id, camera.name, source)
            worker.start()
            with self._state_lock:
                # A new worker is always a NEW stream incarnation.
                self._generations[camera_id] = self._generations.get(camera_id, 0) + 1
                self._workers[camera_id] = worker
                self._signatures[camera_id] = self._signature(camera)

        return reconfigured

    def stop_camera(self, camera_id: str) -> None:
        with self._state_lock:
            worker = self._workers.pop(camera_id, None)
            self._signatures.pop(camera_id, None)
        if worker:
            logger.info("Stopping capture for camera %s", camera_id)
            worker.stop()

    def stop_all(self) -> None:
        with self._state_lock:
            camera_ids = list(self._workers)
        for camera_id in camera_ids:
            self.stop_camera(camera_id)

    def worker(self, camera_id: str) -> Optional[CaptureWorker]:
        with self._state_lock:
            return self._workers.get(camera_id)

    def config(self, camera_id: str) -> Optional[CameraConfig]:
        with self._state_lock:
            return self._configs.get(camera_id)

    @property
    def active(self) -> dict[str, CaptureWorker]:
        with self._state_lock:
            return dict(self._workers)


__all__ = ["CameraManager", "CameraRuntime"]
