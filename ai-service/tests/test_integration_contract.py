"""Web <-> Python integration contract tests.

These cover the generic, source-agnostic guarantees: NVR independence, per-camera
isolation and unknown-camera behaviour. Nothing here assumes a camera count.
"""

from __future__ import annotations

from app.camera.source_builder import build_rtsp_url, build_source
from app.domain.models import CameraConfig, RuleConfig, SourceType
from app.runtime.stream_hub import StreamHub


def _camera(camera_id: str, source_type: SourceType, **kwargs) -> CameraConfig:
    return CameraConfig(id=camera_id, name=f"Cam {camera_id}", source_type=source_type, **kwargs)


def test_direct_camera_source_needs_no_nvr():
    camera = _camera("a", SourceType.DIRECT_CAMERA, host="10.0.0.9", stream_path="stream1")
    source = build_source(camera)
    assert source is not None
    assert source.kind == "rtsp"
    assert source.url.startswith("rtsp://10.0.0.9:554/stream1")


def test_nvr_channel_source_uses_the_same_rtsp_abstraction():
    channel = _camera(
        "b", SourceType.NVR_CHANNEL, host="10.0.0.2", rtsp_port=5541, stream_path="/Streaming/1"
    )
    direct = _camera("c", SourceType.DIRECT_CAMERA, host="10.0.0.3", stream_path="/Streaming/1")
    assert build_source(channel).kind == build_source(direct).kind == "rtsp"
    assert build_rtsp_url(channel) == "rtsp://10.0.0.2:5541/Streaming/1"


def test_sources_depend_on_configuration_not_on_ordering():
    first = _camera("1", SourceType.DIRECT_CAMERA, host="10.0.0.10", stream_path="/a")
    second = _camera("2", SourceType.NVR_CHANNEL, host="10.0.0.11", stream_path="/b")
    third = _camera("3", SourceType.DEMO)
    urls = [build_source(c, demo_video_path="./samples/demo.mp4").url for c in (first, second, third)]
    assert len(set(urls)) == 3


def test_stream_hub_never_returns_another_cameras_frame():
    hub = StreamHub(max_age_seconds=10.0)
    hub.publish("cam-a", b"frame-a")
    hub.publish("cam-b", b"frame-b")
    assert hub.latest("cam-a") == b"frame-a"
    assert hub.latest("cam-b") == b"frame-b"
    assert hub.latest("cam-unknown") is None
    hub.drop("cam-a")
    assert hub.latest("cam-a") is None
    assert hub.latest("cam-b") == b"frame-b"


def test_rule_assignment_is_by_camera_uuid_not_position():
    rule = RuleConfig(
        id="r1",
        name="Phone",
        engine_key="mobile_phone_detection",
        enabled=True,
        available=True,
        camera_ids=("cam-a", "cam-c"),
    )
    assert rule.is_phone_engine
    assert rule.applies_to("cam-a")
    assert rule.applies_to("cam-c")
    assert not rule.applies_to("cam-b")


def test_unscoped_rule_applies_to_every_camera_generically():
    """An empty scope is fleet-wide by design: it must not be camera-count aware."""
    rule = RuleConfig(
        id="r2",
        name="Phone",
        engine_key="mobile_phone_detection",
        enabled=True,
        available=True,
        camera_ids=(),
    )
    assert all(rule.applies_to(cid) for cid in ("cam-a", "cam-b", "cam-c"))
