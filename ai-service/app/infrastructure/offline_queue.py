"""Durable local queue for events and notifications.

Confirmed AI events survive a temporary Internet or Supabase outage. Event IDs
are generated before any I/O, so a retry can never create a duplicate logical
event, and an already-existing row is treated as success (never upserted, so a
human review decision can never be reset to `new`).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_events (
    event_id      TEXT PRIMARY KEY,
    payload       TEXT NOT NULL,
    snapshot_path TEXT,
    created_at    REAL NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    next_attempt  REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS pending_notifications (
    event_id   TEXT NOT NULL,
    provider   TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    delivered  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (event_id, provider)
);
CREATE TABLE IF NOT EXISTS pending_evidence (
    event_id     TEXT PRIMARY KEY,
    object_path  TEXT NOT NULL,
    local_path   TEXT NOT NULL,
    created_at   REAL NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    next_attempt REAL NOT NULL DEFAULT 0
);
"""


@dataclass
class PendingEvent:
    event_id: str
    payload: dict[str, Any]
    snapshot_path: Optional[str]
    attempts: int


@dataclass
class PendingNotification:
    event_id: str
    provider: str
    payload: dict[str, Any]
    attempts: int


@dataclass
class PendingEvidence:
    """A stored event whose snapshot still has to reach Supabase Storage."""

    event_id: str
    object_path: str
    local_path: str
    attempts: int



class OfflineQueue:
    """Thread-safe SQLite-backed store. Never contains camera passwords."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # --- events -----------------------------------------------------------
    def enqueue_event(
        self, event_id: str, payload: dict[str, Any], snapshot_path: Optional[str] = None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO pending_events (event_id, payload, snapshot_path, created_at)"
                " VALUES (?, ?, ?, ?)",
                (event_id, json.dumps(payload), snapshot_path, time.time()),
            )
            self._conn.commit()

    def due_events(self, limit: int = 10, now: Optional[float] = None) -> list[PendingEvent]:
        moment = time.time() if now is None else now
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM pending_events WHERE next_attempt <= ? ORDER BY created_at LIMIT ?",
                (moment, limit),
            ).fetchall()
        return [
            PendingEvent(
                event_id=row["event_id"],
                payload=json.loads(row["payload"]),
                snapshot_path=row["snapshot_path"],
                attempts=row["attempts"],
            )
            for row in rows
        ]

    def mark_event_sent(self, event_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM pending_events WHERE event_id = ?", (event_id,))
            self._conn.commit()

    def mark_event_failed(self, event_id: str, error: str, backoff_seconds: float = 15.0) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE pending_events SET attempts = attempts + 1, last_error = ?,"
                " next_attempt = ? WHERE event_id = ?",
                (error[:500], time.time() + backoff_seconds, event_id),
            )
            self._conn.commit()

    def event_depth(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS total FROM pending_events").fetchone()
        return int(row["total"])

    # --- notifications ----------------------------------------------------
    def enqueue_notification(self, event_id: str, provider: str, payload: dict[str, Any]) -> bool:
        """Returns False when this (event, provider) pair was already handled."""
        with self._lock:
            existing = self._conn.execute(
                "SELECT delivered FROM pending_notifications WHERE event_id = ? AND provider = ?",
                (event_id, provider),
            ).fetchone()
            if existing is not None:
                return False
            self._conn.execute(
                "INSERT INTO pending_notifications (event_id, provider, payload, created_at)"
                " VALUES (?, ?, ?, ?)",
                (event_id, provider, json.dumps(payload), time.time()),
            )
            self._conn.commit()
        return True

    def due_notifications(self, limit: int = 10, max_attempts: int = 5) -> list[PendingNotification]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM pending_notifications WHERE delivered = 0 AND attempts < ?"
                " ORDER BY created_at LIMIT ?",
                (max_attempts, limit),
            ).fetchall()
        return [
            PendingNotification(
                event_id=row["event_id"],
                provider=row["provider"],
                payload=json.loads(row["payload"]),
                attempts=row["attempts"],
            )
            for row in rows
        ]

    def mark_notification_delivered(self, event_id: str, provider: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE pending_notifications SET delivered = 1 WHERE event_id = ? AND provider = ?",
                (event_id, provider),
            )
            self._conn.commit()

    def mark_notification_failed(self, event_id: str, provider: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE pending_notifications SET attempts = attempts + 1, last_error = ?"
                " WHERE event_id = ? AND provider = ?",
                (error[:500], event_id, provider),
            )
            self._conn.commit()

    def notification_depth(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS total FROM pending_notifications WHERE delivered = 0"
            ).fetchone()
        return int(row["total"])

    # --- pending evidence (stored event, snapshot not uploaded yet) --------
    def enqueue_evidence(self, event_id: str, object_path: str, local_path: str) -> None:
        """Remembers that a persisted event still owes its snapshot evidence."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO pending_evidence"
                " (event_id, object_path, local_path, created_at) VALUES (?, ?, ?, ?)",
                (event_id, object_path, local_path, time.time()),
            )
            self._conn.commit()

    def due_evidence(self, limit: int = 5, now: Optional[float] = None) -> list[PendingEvidence]:
        moment = time.time() if now is None else now
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM pending_evidence WHERE next_attempt <= ?"
                " ORDER BY created_at LIMIT ?",
                (moment, limit),
            ).fetchall()
        return [
            PendingEvidence(
                event_id=row["event_id"],
                object_path=row["object_path"],
                local_path=row["local_path"],
                attempts=row["attempts"],
            )
            for row in rows
        ]

    def mark_evidence_sent(self, event_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM pending_evidence WHERE event_id = ?", (event_id,))
            self._conn.commit()

    def mark_evidence_failed(
        self, event_id: str, error: str, backoff_seconds: float = 30.0
    ) -> None:
        """Bounded backoff: the row stays until the upload finally succeeds."""
        with self._lock:
            self._conn.execute(
                "UPDATE pending_evidence SET attempts = attempts + 1, last_error = ?,"
                " next_attempt = ? WHERE event_id = ?",
                (error[:500], time.time() + backoff_seconds, event_id),
            )
            self._conn.commit()

    def evidence_depth(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS total FROM pending_evidence").fetchone()
        return int(row["total"])

    # --- local file ownership ---------------------------------------------
    def references_file(self, local_path: str) -> bool:
        """True while any pending job still needs this local snapshot file.

        Deleting a referenced file would either orphan the stored event's
        evidence or downgrade a queued Telegram photo to text, so cleanup must
        consult this first.
        """
        target = str(local_path)
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM pending_evidence WHERE local_path = ? LIMIT 1", (target,)
            ).fetchone()
            if row is not None:
                return True
            row = self._conn.execute(
                "SELECT 1 FROM pending_events WHERE snapshot_path = ? LIMIT 1", (target,)
            ).fetchone()
            if row is not None:
                return True
            rows = self._conn.execute(
                "SELECT payload FROM pending_notifications WHERE delivered = 0"
            ).fetchall()
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
            if str(payload.get("snapshot_file") or "") == target:
                return True
        return False

    def close(self) -> None:

        with self._lock:
            self._conn.close()