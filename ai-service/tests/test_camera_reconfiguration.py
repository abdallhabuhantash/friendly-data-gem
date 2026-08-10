"""Camera reconfiguration lifecycle: same UUID, replaced video source.

Pure logic: capture workers, sources and the detector are replaced by tiny
fakes, so nothing here opens a stream, a model or a socket.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.camera import camera_manager as camera_manager_module
from app.camera.camera_manager import CameraManager
from app.domain.models import ENGINE_MOBILE_PHONE, CameraConfig, RuleConfig, SourceType
from app.runtime.frame_gate import FrameGate
from app.runtime.orchestrator import Orchestrator
from app.runtime.stream_hub import StreamHub

CAM_A = "11111111-1111-4111-8111-111111111111"
CAM_B = "22222222-2222-4222-8222-222222222222"


class FakeWorker:
    def __init__(self, camera_id, camera_name, source):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.source = source
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


@pytest.fixture()
def manager(monkeypatch):
    monkeypatch.setattr(camera_manager_module, "CaptureWorker", FakeWorker)
    monkeypatch.setattr(
        camera_manager_module, "build_source", lambda camera, **kw: SimpleNamespace(kind="fake")
    )
    settings = SimpleNamespace(demo_video_for=lambda _cid: None, demo_video_loop=True)
    credentials = SimpleNamespace(get=lambda _cid: (None, None))
    return CameraManager(settings, credentials)


def camera(camera_id=CAM_A, **overrides) -> CameraConfig:
    base = dict(
        id=camera_id,
        name="Hall",
        source_type=SourceType.NVR_CHANNEL,
        host="10.0.0.5",
        rtsp_port=554,
        channel=1,
        stream_path="/cam/realmonitor",
    )
    base.update(overrides)
    return CameraConfig(**base)


# --- A/B/C: reconfiguration reporting ------------------------------------
def test_unchanged_signature_reports_no_reconfiguration(manager):
    assert manager.sync([camera()]) == set()
    worker = manager.worker(CAM_A)
    assert manager.sync([camera()]) == set()
    assert manager.worker(CAM_A) is worker  # worker preserved


def test_metadata_only_change_is_not_reconfiguration(manager):
    manager.sync([camera()])
    worker = manager.worker(CAM_A)
    assert manager.sync([camera(name="Hall renamed", location="Floor 2")]) == set()
    assert manager.worker(CAM_A) is worker


@pytest.mark.parametrize(
    "change",
    [
        {"host": "10.0.0.9"},
        {"stream_path": "/other/path"},
        {"channel": 4},
        {"rtsp_port": 5540},
        {"source_type": SourceType.DIRECT_CAMERA},
    ],
)
def test_source_signature_change_reports_reconfiguration(manager, change):
    manager.sync([camera()])
    old_worker = manager.worker(CAM_A)
    assert manager.sync([camera(**change)]) == {CAM_A}
    new_worker = manager.worker(CAM_A)
    assert old_worker.stopped is True
    assert new_worker is not old_worker
    assert new_worker.started is True


def test_reconfiguration_is_isolated_between_cameras(manager):
    manager.sync([camera(), camera(CAM_B, host="10.0.0.6")])
    worker_b = manager.worker(CAM_B)
    assert manager.sync([camera(host="10.0.0.99"), camera(CAM_B, host="10.0.0.6")]) == {CAM_A}
    assert manager.worker(CAM_B) is worker_b
    assert worker_b.stopped is False


def test_removed_camera_is_not_reported_as_reconfigured(manager):
    manager.sync([camera(), camera(CAM_B, host="10.0.0.6")])
    assert manager.sync([camera(CAM_B, host="10.0.0.6")]) == set()
    assert manager.worker(CAM_A) is None


# --- shared runtime reset path -------------------------------------------
class FakeRegistry:
    def __init__(self):
        self.resets = []

    def reset(self, camera_id):
        self.resets.append(camera_id)


class FakeDetector:
    def __init__(self):
        self.resets = []

    def reset_camera(self, camera_id):
        self.resets.append(camera_id)


def runtime_stub():
    stub = SimpleNamespace(
        registry=FakeRegistry(),
        stream_hub=StreamHub(),
        _inference_fps={CAM_A: 11.0, CAM_B: 7.0},
        _frame_gate=FrameGate(),
        detector=FakeDetector(),
    )
    stub.stream_hub.publish(CAM_A, b"old-source-jpeg")
    stub.stream_hub.publish(CAM_B, b"b-jpeg")
    stub._frame_gate.accept(CAM_A, 500)
    stub._frame_gate.accept(CAM_B, 500)
    return stub


def test_runtime_reset_clears_only_the_target_camera():
    stub = runtime_stub()
    Orchestrator._reset_camera_runtime(stub, CAM_A)

    assert stub.registry.resets == [CAM_A]
    assert stub.detector.resets == [CAM_A]
    assert stub.stream_hub.has(CAM_A) is False
    assert stub.stream_hub.latest(CAM_B) == b"b-jpeg"
    assert CAM_A not in stub._inference_fps
    assert stub._inference_fps[CAM_B] == 7.0
    # E: a fresh, LOWER sequence from the replacement worker is accepted at once.
    assert stub._frame_gate.accept(CAM_A, 1) is True
    assert stub._frame_gate.accept(CAM_B, 500) is False


def test_runtime_reset_is_idempotent_and_tolerates_missing_detector():
    stub = runtime_stub()
    Orchestrator._reset_camera_runtime(stub, CAM_A)
    Orchestrator._reset_camera_runtime(stub, CAM_A)
    assert stub.registry.resets == [CAM_A, CAM_A]
    stub.detector = None
    Orchestrator._reset_camera_runtime(stub, CAM_A)


# --- refresh wiring: replacement and removal use the same cleanup ---------
class AliveThread:
    def is_alive(self):
        return True


def refresh_stub(*, reconfigured, active):
    stub = runtime_stub()
    stub.repository = SimpleNamespace(
        system_config=lambda: SimpleNamespace(operation_mode="live"),
        cameras=lambda mode: [],
        rules=lambda: [],
    )
    stub.cameras = SimpleNamespace(
        sync=lambda cameras: set(reconfigured), active={cid: object() for cid in active}
    )
    stub._threads = {CAM_A: AliveThread(), CAM_B: AliveThread()}
    stub._rules = []
    stub.system = SimpleNamespace(operation_mode="live")
    return stub


def test_reconfigured_camera_is_cleaned_without_dropping_its_thread():
    stub = refresh_stub(reconfigured={CAM_A}, active=[CAM_A, CAM_B])
    Orchestrator._refresh_configuration(stub)

    assert stub.registry.resets == [CAM_A]
    assert stub.detector.resets == [CAM_A]
    assert stub.stream_hub.has(CAM_A) is False
    assert CAM_A not in stub._inference_fps
    assert stub._frame_gate.accept(CAM_A, 1) is True
    # The existing inference thread keeps serving the replacement worker.
    assert set(stub._threads) == {CAM_A, CAM_B}


def test_removed_camera_still_gets_full_cleanup():
    stub = refresh_stub(reconfigured=set(), active=[CAM_B])
    Orchestrator._refresh_configuration(stub)

    assert stub.registry.resets == [CAM_A]
    assert stub.detector.resets == [CAM_A]
    assert stub.stream_hub.has(CAM_A) is False
    assert CAM_A not in stub._inference_fps
    assert stub._frame_gate.accept(CAM_A, 1) is True
    assert set(stub._threads) == {CAM_B}
    assert stub.stream_hub.latest(CAM_B) == b"b-jpeg"
    assert stub._inference_fps[CAM_B] == 7.0


# --- J: the real production phone-rule filter ----------------------------
def rule(engine_key, rule_id, **overrides) -> RuleConfig:
    base = dict(
        id=rule_id,
        name=rule_id,
        engine_key=engine_key,
        available=True,
        enabled=True,
        confidence_threshold=0.7,
        person_confidence_threshold=0.6,
        association_confidence_threshold=0.65,
    )
    base.update(overrides)
    return RuleConfig(**base)


def test_production_phone_rules_excludes_behavioural_engines():
    phone = rule(ENGINE_MOBILE_PHONE, "phone")
    behavioural = [
        rule("concealed_device_activity", "concealed", confidence_threshold=0.01),
        rule("document_exchange", "documents", person_confidence_threshold=0.99),
        rule("peer_interaction", "peers", association_confidence_threshold=0.99),
        rule(None, "unset"),
    ]
    selected = Orchestrator._phone_rules([*behavioural, phone])
    assert selected == [phone]
    assert min(r.confidence_threshold for r in selected) == pytest.approx(0.7)
    assert min(r.person_confidence_threshold for r in selected) == pytest.approx(0.6)
    assert min(r.association_confidence_threshold for r in selected) == pytest.approx(0.65)
    assert Orchestrator._phone_rules(behavioural) == []
