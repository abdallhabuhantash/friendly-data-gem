"""Persists confirmed events, with a durable queue behind every network call.

Publishing order: UUID -> snapshot -> upload -> insert -> notify. A failing
upload never discards a critical detection; the event is stored with
`snapshot_path = null` and the queue retries the row later.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EventPublisher:
    """Coordinates snapshot upload, Supabase insert, queueing and notification."""

    def __init__(
        self,
        repository,  # noqa: ANN001 - SupabaseRepository
        queue,  # noqa: ANN001 - OfflineQueue
        snapshots=None,  # noqa: ANN001 - SnapshotService
        notifications=None,  # noqa: ANN001 - NotificationManager
        duplicate_error: type[Exception] = Exception,
    ) -> None:
        self._repository = repository
        self._queue = queue
        self._snapshots = snapshots
        self._notifications = notifications
        self._duplicate_error = duplicate_error

    # --- public API -------------------------------------------------------
    def publish(self, event, frame=None, save_snapshot: bool = False) -> bool:
        """Publishes one confirmed event. Returns True when Supabase accepted it."""
        local_file: Optional[Path] = None
        if save_snapshot and frame is not None and self._snapshots is not None:
            try:
                local_file = self._snapshots.write_local(event, frame)
                if local_file is not None:
                    event.snapshot_path = self._snapshots.upload(event, local_file)
            except Exception as exc:  # isolation: inference must keep running
                logger.error("Snapshot handling failed for %s: %s", event.id, type(exc).__name__)

        row = event.to_row()
        stored = self._insert(row)
        if not stored:
            self._queue.enqueue_event(
                event.id, row, str(local_file) if local_file else None
            )
            logger.warning("Event %s queued locally for retry", event.id)
        else:
            logger.info(
                "Event %s persisted (%s, %s)", event.id, row["type"], row["association_status"]
            )
            if self._notifications is not None:
                self._notifications.enqueue(event)

        if stored and local_file is not None and event.snapshot_path:
            self._snapshots.cleanup(local_file)
        return stored

    def retry_pending(self, limit: int = 5) -> int:
        """Drains the durable queue. Duplicates count as success."""
        sent = 0
        for pending in self._queue.due_events(limit=limit):
            row: dict[str, Any] = pending.payload
            if self._insert(row):
                self._queue.mark_event_sent(pending.event_id)
                if self._notifications is not None:
                    self._notifications.enqueue_row(row)
                if pending.snapshot_path and self._snapshots is not None:
                    self._snapshots.cleanup(Path(pending.snapshot_path))
                sent += 1
            else:
                self._queue.mark_event_failed(
                    pending.event_id, "insert failed", backoff_seconds=min(300, 15 * (pending.attempts + 1))
                )
        return sent

    # --- internals --------------------------------------------------------
    def _insert(self, row: dict[str, Any]) -> bool:
        try:
            self._repository.insert_event(row)
            return True
        except self._duplicate_error:
            # The same UUID already exists: never re-insert or upsert, because
            # that could reset a human review decision back to `new`.
            logger.info("Event %s already persisted; treating retry as success", row.get("id"))
            return True
        except Exception as exc:
            logger.error("Event insert failed for %s: %s", row.get("id"), type(exc).__name__)
            return False