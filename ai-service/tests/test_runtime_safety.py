"""Tests for the non-AI safety guarantees: redaction, queueing, streaming."""

from __future__ import annotations

from app.camera.source_builder import build_rtsp_url, build_source, redact
from app.domain.models import CameraConfig, SourceType
from app.infrastructure.offline_queue import OfflineQueue
from app.notifications.base import should_send
from app.runtime.stream_hub import StreamHub
from app.security import verify_service_key


def test_rtsp_url_is_redacted_before_logging():
    camera = CameraConfig(
        id="c1", name="Cam", source_type=SourceType.DIRECT_CAMERA, host="10.0.0.5", stream_path="/s1"
    )
    url = build_rtsp_url(camera, "admin", "sup3rs3cret")
    assert "sup3rs3cret" in url
    assert "sup3rs3cret" not in redact(url)
    assert "admin" not in redact(url)


def test_demo_camera_without_video_has_no_source():
    camera = CameraConfig(id="c2", name="Demo", source_type=SourceType.DEMO)
    assert build_source(camera) is None


def test_stream_hub_expires_stale_frames():
    hub = StreamHub(max_age_seconds=0.0)
    hub.publish("c1", b"jpeg")
    assert hub.latest("c1") is None
    fresh = StreamHub(max_age_seconds=10.0)
    fresh.publish("c1", b"jpeg")
    assert fresh.latest("c1") == b"jpeg"


def test_queue_survives_and_deduplicates(tmp_path):
    queue = OfflineQueue(tmp_path / "queue.db")
    queue.enqueue_event("e1", {"id": "e1"}, None)
    queue.enqueue_event("e1", {"id": "e1"}, None)
    assert queue.event_depth() == 1
    assert queue.enqueue_notification("e1", "telegram", {"event_id": "e1"}) is True
    assert queue.enqueue_notification("e1", "telegram", {"event_id": "e1"}) is False
    queue.mark_event_sent("e1")
    assert queue.event_depth() == 0
    queue.close()


def test_stream_endpoint_stays_closed_without_a_configured_key():
    assert verify_service_key("", "anything") is False
    assert verify_service_key("secret", None) is False
    assert verify_service_key("secret", "secret") is True


def test_only_definitive_critical_alerts_notify_by_default():
    assert should_send(severity="critical", association_status="associated", send_warnings=False)
    assert not should_send(severity="critical", association_status="uncertain", send_warnings=False)
    assert should_send(severity="warning", association_status="unassociated", send_warnings=True)