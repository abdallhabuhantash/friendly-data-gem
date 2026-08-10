"""AI service and camera heartbeats.

Only genuinely measured values are published. The service never writes an NVR
heartbeat and never claims recording state.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HealthReporter:
    def __init__(self, repository, settings, queue) -> None:  # noqa: ANN001
        self._repository = repository
        self._settings = settings
        self._queue = queue
        self._started = time.monotonic()

    def payload(
        self,
        *,
        model: str,
        device: str,
        inference_fps: float,
        gpu_load_percent: Optional[float] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self._settings.service_version,
            "model": model,
            "device": device,
            "inference_fps": round(float(inference_fps), 2),
            "queue_depth": self._queue.event_depth() + self._queue.notification_depth(),
            "uptime_seconds": int(time.monotonic() - self._started),
            "notification_channels": {
                "telegram": {
                    "configured": bool(self._settings.telegram_configured),
                    "ready": bool(self._settings.telegram_ready),
                }
            },
        }
        # Only reported when genuinely measurable; never invented.
        if gpu_load_percent is not None:
            payload["gpu_load_percent"] = round(float(gpu_load_percent), 1)
        return payload

    def beat(self, *, online: bool, is_demo: bool, payload: dict[str, Any]) -> None:
        try:
            self._repository.write_ai_health(online=online, is_demo=is_demo, payload=payload)
        except Exception as exc:
            logger.warning("AI heartbeat write failed: %s", type(exc).__name__)

    def camera_beat(self, camera_id: str, *, status: str, fps: float, heartbeat_at) -> None:
        """Only called with an actual observed frame time; never optimistic."""
        try:
            self._repository.update_camera_runtime(
                camera_id, status=status, fps=fps, heartbeat_at=heartbeat_at
            )
        except Exception as exc:
            logger.warning("Camera heartbeat write failed for %s: %s", camera_id, type(exc).__name__)


def measure_gpu_load() -> Optional[float]:
    """Returns GPU utilisation only when the driver actually reports it."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.utilization())
    except Exception:
        return None