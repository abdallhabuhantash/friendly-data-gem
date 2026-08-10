"""Keeps running capture workers in sync with the console configuration."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from ..domain.models import CameraConfig
from .capture_worker import CaptureWorker
from .source_builder import build_source

logger = logging.getLogger(__name__)


class CameraManager:
    """Starts, stops and replaces capture workers as configuration changes."""

    def __init__(self, settings, credentials) -> None:  # noqa: ANN001 - injected
        self._settings = settings
        self._credentials = credentials
        self._workers: dict[str, CaptureWorker] = {}
        self._configs: dict[str, CameraConfig] = {}
        self._signatures: dict[str, tuple] = {}

    @staticmethod
    def _signature(camera: CameraConfig) -> tuple:
        return (
            camera.source_type.value,
            camera.host,
            camera.rtsp_port,
            camera.stream_path,
            camera.channel,
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

        for camera_id in list(self._workers):
            if camera_id not in desired:
                self.stop_camera(camera_id)
                continue
            if self._signatures.get(camera_id) != self._signature(desired[camera_id]):
                logger.info("Camera %s source signature changed; replacing worker", camera_id)
                reconfigured.add(camera_id)
                self.stop_camera(camera_id)




        for camera_id, camera in desired.items():
            self._configs[camera_id] = camera
            if camera_id in self._workers:
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
            self._workers[camera_id] = worker
            self._signatures[camera_id] = self._signature(camera)

        return reconfigured


    def stop_camera(self, camera_id: str) -> None:
        worker = self._workers.pop(camera_id, None)
        self._signatures.pop(camera_id, None)
        if worker:
            logger.info("Stopping capture for camera %s", camera_id)
            worker.stop()

    def stop_all(self) -> None:
        for camera_id in list(self._workers):
            self.stop_camera(camera_id)

    def worker(self, camera_id: str) -> Optional[CaptureWorker]:
        return self._workers.get(camera_id)

    def config(self, camera_id: str) -> Optional[CameraConfig]:
        return self._configs.get(camera_id)

    @property
    def active(self) -> dict[str, CaptureWorker]:
        return dict(self._workers)