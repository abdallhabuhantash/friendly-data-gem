"""Durable snapshot/evidence retry behaviour.

These tests use fakes for Supabase and the snapshot uploader, so they need no
network, no model weights and no camera.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.events.event_publisher import EventPublisher
from app.infrastructure.offline_queue import OfflineQueue


class DuplicateError(Exception):
    pass


class FakeRepository:
    def __init__(self, fail_insert: bool = False) -> None:
        self.rows: dict[str, dict] = {}
        self.snapshot_updates: list[tuple[str, str]] = []
        self.insert_calls = 0
        self.fail_insert = fail_insert

    def insert_event(self, row: dict) -> None:
        self.insert_calls += 1
        if self.fail_insert:
            raise RuntimeError("network down")
        if row["id"] in self.rows:
            raise DuplicateError(row["id"])
        self.rows[row["id"]] = dict(row)

    def set_event_snapshot(self, event_id: str, snapshot_path: str) -> None:
        self.snapshot_updates.append((event_id, snapshot_path))
        self.rows[event_id]["snapshot_path"] = snapshot_path


class FakeSnapshots:
    """Writes a real local file; the upload can be made to fail on demand."""

    def __init__(self, tmp_path: Path, upload_ok: bool = True) -> None:
        self.dir = tmp_path
        self.upload_ok = upload_ok
        self.upload_calls: list[str] = []
        self.cleaned: list[Path] = []

    def write_local(self, event, frame):  # noqa: ANN001
        path = self.dir / f"{event.id}.jpg"
        path.write_bytes(b"jpeg")
        return path

    @staticmethod
    def object_path(event):  # noqa: ANN001
        return f"cam/{event.id}.jpg"

    def upload(self, event, local_file):  # noqa: ANN001
        return self.upload_file(self.object_path(event), local_file)

    def upload_file(self, object_path: str, local_file: Path):
        self.upload_calls.append(object_path)
        return object_path if self.upload_ok else None

    def cleanup(self, local_file) -> None:  # noqa: ANN001
        self.cleaned.append(Path(local_file))
        Path(local_file).unlink(missing_ok=True)


class FakeNotifications:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def enqueue(self, event, snapshot_file=None) -> bool:  # noqa: ANN001
        self.calls.append((event.id, snapshot_file))
        return True

    def enqueue_row(self, row, snapshot_file=None) -> bool:  # noqa: ANN001
        self.calls.append((row["id"], snapshot_file))
        return True


class FakeEvent:
    def __init__(self, event_id: str = "event-1") -> None:
        self.id = event_id
        self.snapshot_path = None

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "type": "mobile_phone_detected",
            "association_status": "associated",
            "status": "new",
            "snapshot_path": self.snapshot_path,
        }


@pytest.fixture()
def queue(tmp_path: Path) -> OfflineQueue:
    q = OfflineQueue(tmp_path / "queue.db")
    yield q
    q.close()


def _publisher(repo, queue, snapshots, notifications=None) -> EventPublisher:
    return EventPublisher(
        repository=repo,
        queue=queue,
        snapshots=snapshots,
        notifications=notifications,
        duplicate_error=DuplicateError,
    )


def test_failed_upload_still_persists_event_and_queues_evidence(tmp_path, queue):
    repo = FakeRepository()
    snapshots = FakeSnapshots(tmp_path, upload_ok=False)
    publisher = _publisher(repo, queue, snapshots)
    event = FakeEvent()

    assert publisher.publish(event, frame=object(), save_snapshot=True) is True
    # Event is stored even though the evidence upload failed.
    assert repo.rows["event-1"]["snapshot_path"] is None
    assert queue.evidence_depth() == 1
    # The local file must survive so the retry has something to upload.
    assert (tmp_path / "event-1.jpg").exists()
    assert snapshots.cleaned == []


def test_evidence_retry_attaches_snapshot_to_same_event(tmp_path, queue):
    repo = FakeRepository()
    snapshots = FakeSnapshots(tmp_path, upload_ok=False)
    publisher = _publisher(repo, queue, snapshots)
    publisher.publish(FakeEvent(), frame=object(), save_snapshot=True)

    snapshots.upload_ok = True
    assert publisher.retry_pending_evidence() == 1

    assert repo.snapshot_updates == [("event-1", "cam/event-1.jpg")]
    assert repo.rows["event-1"]["snapshot_path"] == "cam/event-1.jpg"
    # No duplicate row was created for the retry.
    assert repo.insert_calls == 1
    assert queue.evidence_depth() == 0
    # Local file is released only after the evidence landed.
    assert not (tmp_path / "event-1.jpg").exists()


def test_evidence_retry_never_overwrites_review_decision(tmp_path, queue):
    repo = FakeRepository()
    snapshots = FakeSnapshots(tmp_path, upload_ok=False)
    publisher = _publisher(repo, queue, snapshots)
    publisher.publish(FakeEvent(), frame=object(), save_snapshot=True)

    # A human reviews the event before the snapshot upload recovers.
    repo.rows["event-1"].update(
        {"status": "confirmed", "reviewed_by": "Alice", "reviewed_at": "2026-01-01T00:00:00Z"}
    )

    snapshots.upload_ok = True
    publisher.retry_pending_evidence()

    row = repo.rows["event-1"]
    assert row["status"] == "confirmed"
    assert row["reviewed_by"] == "Alice"
    assert row["snapshot_path"] == "cam/event-1.jpg"


def test_evidence_retry_backs_off_and_keeps_job(tmp_path, queue):
    repo = FakeRepository()
    snapshots = FakeSnapshots(tmp_path, upload_ok=False)
    publisher = _publisher(repo, queue, snapshots)
    publisher.publish(FakeEvent(), frame=object(), save_snapshot=True)

    assert publisher.retry_pending_evidence() == 0
    # Job survives, but is not retried immediately (bounded backoff).
    assert queue.evidence_depth() == 1
    assert queue.due_evidence() == []
    assert (tmp_path / "event-1.jpg").exists()


def test_local_file_kept_while_notification_is_pending(tmp_path, queue):
    repo = FakeRepository()
    snapshots = FakeSnapshots(tmp_path, upload_ok=True)
    notifications = FakeNotifications()
    publisher = _publisher(repo, queue, snapshots, notifications)
    event = FakeEvent()

    queue.enqueue_notification("event-1", "telegram", {"snapshot_file": str(tmp_path / "event-1.jpg")})
    publisher.publish(event, frame=object(), save_snapshot=True)

    assert notifications.calls == [("event-1", str(tmp_path / "event-1.jpg"))]
    # A pending Telegram photo still owns the file, so cleanup must wait.
    assert (tmp_path / "event-1.jpg").exists()
    assert snapshots.cleaned == []


def test_notification_receives_snapshot_path(tmp_path, queue):
    repo = FakeRepository()
    snapshots = FakeSnapshots(tmp_path, upload_ok=True)
    notifications = FakeNotifications()
    publisher = _publisher(repo, queue, snapshots, notifications)

    publisher.publish(FakeEvent("event-2"), frame=object(), save_snapshot=True)

    assert notifications.calls[0][0] == "event-2"
    assert notifications.calls[0][1] is not None


def test_duplicate_insert_is_treated_as_success(tmp_path, queue):
    repo = FakeRepository()
    snapshots = FakeSnapshots(tmp_path)
    publisher = _publisher(repo, queue, snapshots)

    assert publisher.publish(FakeEvent("dup"), frame=None) is True
    assert publisher.publish(FakeEvent("dup"), frame=None) is True
    assert queue.event_depth() == 0
