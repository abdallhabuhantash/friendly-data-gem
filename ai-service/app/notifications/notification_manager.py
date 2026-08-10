"""Bounded, de-duplicated notification delivery.

Delivery runs off the inference path: jobs are persisted in the same local
SQLite store and drained by a retry worker, so a failed send can never stop
detection and a delivered message is never repeated.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import NotificationProvider, payload_from_row, should_send

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 5


class NotificationManager:
    def __init__(
        self,
        queue,  # noqa: ANN001 - OfflineQueue
        provider: Optional[NotificationProvider] = None,
        *,
        send_warnings: bool = False,
    ) -> None:
        self._queue = queue
        self._provider = provider
        self._send_warnings = send_warnings

    @property
    def enabled(self) -> bool:
        return self._provider is not None

    def enqueue(self, event) -> bool:  # noqa: ANN001 - AiEvent
        row = event.to_row()
        snapshot_file = None
        return self.enqueue_row(row, snapshot_file)

    def enqueue_row(self, row: dict[str, Any], snapshot_file: Optional[str] = None) -> bool:
        if not self._provider:
            return False
        if not should_send(
            severity=str(row.get("severity") or "warning"),
            association_status=str(row.get("association_status") or "not_applicable"),
            send_warnings=self._send_warnings,
        ):
            return False
        payload = payload_from_row(row, snapshot_file)
        return self._queue.enqueue_notification(
            payload.event_id, self._provider.name, payload.to_dict()
        )

    def drain(self, limit: int = 5) -> int:
        """Attempts pending sends. Never raises into the inference loop."""
        if not self._provider:
            return 0
        delivered = 0
        for job in self._queue.due_notifications(limit=limit, max_attempts=MAX_ATTEMPTS):
            from .base import NotificationPayload

            try:
                payload = NotificationPayload(**job.payload)
                ok = self._provider.send(payload)
            except Exception as exc:
                logger.warning("Notification send error: %s", type(exc).__name__)
                ok = False
            if ok:
                self._queue.mark_notification_delivered(job.event_id, job.provider)
                delivered += 1
            else:
                self._queue.mark_notification_failed(job.event_id, job.provider, "send failed")
        return delivered