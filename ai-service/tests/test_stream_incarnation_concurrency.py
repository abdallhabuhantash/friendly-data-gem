"""Deterministic concurrency tests for camera stream incarnations.

Every race is forced with `threading.Event` / `Barrier`; no test sleeps and hopes
the interleaving happens. The real production methods
(`Orchestrator._guarded_process`, `_transition_generation`,
`_reset_camera_runtime`, `_refresh_configuration`, `YoloDetector.reset_camera`)
are exercised directly on light harnesses, so nothing opens a stream or a model.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace

from app.ai.detector import TrackerStateStore, YoloDetector
from app.camera.camera_manager import CameraRuntime
from app.runtime.frame_gate import FrameGate
from app.runtime.orchestrator import Orchestrator
from app.runtime.stream_hub import StreamHub

CAM_A = "11111111-1111-4111-8111-111111111111"
CAM_B = "22222222-2222-4222-8222-222222222222"
TIMEOUT = 5.0


# --- fakes ----------------------------------------------------------------
class FakeRegistry:
    def __init__(self):
        self.resets: list[str] = []
        self.state: dict[str, int] = {}

    def reset(self, camera_id):
        self.resets.append(camera_id)
        self.state.pop(camera_id, None)


class FakeDetector:
    def __init__(self):
        self.resets: list[str] = []
        self.trackers: dict[str, int] = {}

    def reset_camera(self, camera_id):
        self.resets.append(camera_id)
        self.trackers.pop(camera_id, None)


@dataclass
class FakeWorker:
    camera_id: str
    frame: object = "frame"
    sequence: int = 1

    def latest_frame_with_sequence(self):
        return self.frame, self.sequence


class FakeCameras:
    """Minimal CameraManager surface: generations, per-camera locks, snapshots."""

    def __init__(self, generations: dict[str, int]):
        self._generations = dict(generations)
        self._workers = {cid: FakeWorker(cid) for cid in generations}
        self._configs = {
            cid: SimpleNamespace(id=cid, name=f"cam-{cid[:2]}") for cid in generations
        }
        self._locks: dict[str, threading.RLock] = {}
        self.reconfigured: set[str] = set()

    def lock(self, camera_id):
        return self._locks.setdefault(camera_id, threading.RLock())

    def generation(self, camera_id):
        if camera_id not in self._workers:
            return None
        return self._generations.get(camera_id)

    def snapshot(self, camera_id):
        worker = self._workers.get(camera_id)
        config = self._configs.get(camera_id)
        generation = self._generations.get(camera_id)
        if worker is None or config is None or generation is None:
            return None
        return CameraRuntime(camera_id, generation, worker, config)

    def sync(self, cameras):
        return set(self.reconfigured)

    @property
    def active(self):
        return dict(self._workers)

    # test-only mutations
    def replace_source(self, camera_id):
        self._generations[camera_id] += 1
        self._workers[camera_id] = FakeWorker(camera_id)
        self.reconfigured = {camera_id}

    def remove(self, camera_id):
        self._workers.pop(camera_id, None)
        self.reconfigured = set()


class Harness:
    """Real orchestrator lifecycle methods over fake collaborators."""

    _reset_camera_runtime = Orchestrator._reset_camera_runtime
    _transition_generation = Orchestrator._transition_generation
    _guarded_process = Orchestrator._guarded_process
    _record_inference_fps = Orchestrator._record_inference_fps
    _refresh_configuration = Orchestrator._refresh_configuration

    def __init__(self, generations: dict[str, int]):
        self.cameras = FakeCameras(generations)
        self.registry = FakeRegistry()
        self.detector = FakeDetector()
        self.stream_hub = StreamHub()
        self._frame_gate = FrameGate()
        self._inference_fps: dict[str, float] = {}
        self._processed_frames: dict[str, int] = {}
        self._fps_window: dict[str, tuple[float, int]] = {}
        self._seen_generation: dict[str, int] = dict(generations)
        self._threads: dict[str, threading.Thread] = {}
        self._rules: list = []
        self.system = SimpleNamespace(operation_mode="live")
        self.settings = SimpleNamespace(process_every_n_frames=1, inference_max_fps=0.0)
        self.repository = SimpleNamespace(
            system_config=lambda: SimpleNamespace(operation_mode="live"),
            cameras=lambda mode: [],
            rules=lambda: [],
        )
        self.trace: list[str] = []
        self.processed: list[tuple[str, int]] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.pause_camera: str | None = None
        self._stop = threading.Event()

    # stands in for the real AI pass: records state mutations
    def _process_frame(self, camera, frame, frame_sequence=None):
        self.trace.append(f"process-enter:{camera.id}")
        if self.pause_camera == camera.id:
            self.entered.set()
            assert self.release.wait(TIMEOUT), "release never signalled"
        self.processed.append((camera.id, frame_sequence))
        self.registry.state[camera.id] = frame_sequence
        self.detector.trackers[camera.id] = frame_sequence
        self.stream_hub.publish(camera.id, b"jpeg")
        self.trace.append(f"process-exit:{camera.id}")

    def _reset_traced(self, camera_id):
        self.trace.append(f"reset:{camera_id}")
        return Orchestrator._reset_camera_runtime(self, camera_id)


def harness(generations=None) -> Harness:
    return Harness(generations or {CAM_A: 1, CAM_B: 1})


def spawn(fn) -> threading.Thread:
    thread = threading.Thread(target=fn, daemon=True)
    thread.start()
    return thread


def seed_old_state(h: Harness, camera_id: str, sequence: int = 500) -> None:
    h._frame_gate.accept(camera_id, sequence)
    h.registry.state[camera_id] = sequence
    h.detector.trackers[camera_id] = sequence
    h.stream_hub.publish(camera_id, b"old")
    h._inference_fps[camera_id] = 12.0
    h._fps_window[camera_id] = (1.0, 5)
    h._processed_frames[camera_id] = sequence


def assert_no_state(h: Harness, camera_id: str) -> None:
    assert camera_id not in h.registry.state
    assert camera_id not in h.detector.trackers
    assert h.stream_hub.has(camera_id) is False
    assert camera_id not in h._inference_fps
    assert camera_id not in h._fps_window
    assert camera_id not in h._processed_frames


# --- TEST A: in-flight old frame + reconfigure ----------------------------
def test_a_reconfigure_waits_for_in_flight_frame_then_resets():
    h = harness()
    h.pause_camera = CAM_A
    runtime = h.cameras.snapshot(CAM_A)

    worker = spawn(lambda: h._guarded_process(runtime, "frame", 500))
    assert h.entered.wait(TIMEOUT)  # old-incarnation frame is inside the section

    # Replacement worker exists before the control thread cleans up.
    h.cameras.replace_source(CAM_A)
    control = spawn(lambda: h.__class__._transition_generation(h, CAM_A))

    # The reset cannot complete while the same camera's frame holds the lock.
    control.join(0.2)
    assert control.is_alive() is True
    assert h.registry.resets == []

    h.release.set()
    worker.join(TIMEOUT)
    control.join(TIMEOUT)
    assert control.is_alive() is False

    assert h.registry.resets == [CAM_A]
    assert h.detector.resets == [CAM_A]
    assert h._seen_generation[CAM_A] == 2
    assert_no_state(h, CAM_A)
    # ordering: the paused frame finished before the reset ran
    assert h.trace.index("process-exit:" + CAM_A) < len(h.trace)


# --- TEST B: new worker wins scheduling ----------------------------------
def test_b_new_generation_frame_is_refused_before_transition():
    h = harness()
    seed_old_state(h, CAM_A)
    h.cameras.replace_source(CAM_A)
    new_runtime = h.cameras.snapshot(CAM_A)

    # Inference thread sees the new worker first: the frame is refused because
    # the generation transition has not happened yet.
    assert h._guarded_process(new_runtime, "frame", 1) is False
    assert h.processed == []
    assert h.registry.state[CAM_A] == 500  # untouched old state, not mixed

    assert h._transition_generation(CAM_A) == 2
    assert h.registry.resets == [CAM_A]
    assert h._guarded_process(new_runtime, "frame", 1) is True
    assert h.processed == [(CAM_A, 1)]


def test_b_transition_is_performed_exactly_once():
    h = harness()
    h.cameras.replace_source(CAM_A)
    barrier = threading.Barrier(2)
    results: list = []

    def racer():
        barrier.wait(TIMEOUT)
        results.append(h._transition_generation(CAM_A))

    threads = [spawn(racer) for _ in range(2)]
    for thread in threads:
        thread.join(TIMEOUT)
    assert results == [2, 2]
    assert h.registry.resets == [CAM_A]  # destructive transition ran once


# --- TEST C: detector reset vs in-flight detect --------------------------
def build_detector() -> tuple[YoloDetector, threading.Event, threading.Event]:
    detector = object.__new__(YoloDetector)
    detector.device = "cpu"
    detector.imgsz = 320
    detector.tracker = "bytetrack.yaml"
    detector._lock = threading.Lock()
    detector._classes = {}
    detector._trackers = TrackerStateStore()
    entered = threading.Event()
    release = threading.Event()

    predictor = SimpleNamespace(trackers=["tracker-obj"], vid_path=["path"])

    def track(**kwargs):
        entered.set()
        assert release.wait(TIMEOUT)
        return []

    detector._model = SimpleNamespace(track=track, predictor=predictor)
    return detector, entered, release


def test_c_reset_camera_waits_for_in_flight_detect():
    detector, entered, release = build_detector()
    frame = SimpleNamespace(shape=(480, 640, 3))

    detect_thread = spawn(lambda: detector.detect(frame, CAM_A))
    assert entered.wait(TIMEOUT)

    reset_thread = spawn(lambda: detector.reset_camera(CAM_A))
    reset_thread.join(0.2)
    assert reset_thread.is_alive() is True  # blocked on the detector lock

    release.set()
    detect_thread.join(TIMEOUT)
    reset_thread.join(TIMEOUT)
    assert reset_thread.is_alive() is False
    # The in-flight detect captured state, then the reset removed it for good.
    assert detector._trackers.known(CAM_A) is False


def test_c_reset_of_one_camera_keeps_other_tracker_state():
    detector, entered, release = build_detector()
    release.set()
    frame = SimpleNamespace(shape=(480, 640, 3))
    detector.detect(frame, CAM_A)
    detector.detect(frame, CAM_B)
    assert detector._trackers.known(CAM_A) is True

    detector.reset_camera(CAM_A)
    assert detector._trackers.known(CAM_A) is False
    assert detector._trackers.known(CAM_B) is True


# --- TEST D: camera isolation --------------------------------------------
def test_d_camera_b_is_untouched_while_camera_a_transitions():
    h = harness()
    seed_old_state(h, CAM_A)
    seed_old_state(h, CAM_B, sequence=77)
    b_runtime = h.cameras.snapshot(CAM_B)

    h.cameras.replace_source(CAM_A)
    h._transition_generation(CAM_A)

    assert h.registry.resets == [CAM_A]
    assert h.detector.resets == [CAM_A]
    assert h._seen_generation[CAM_B] == 1
    assert h.registry.state[CAM_B] == 77
    assert h.detector.trackers[CAM_B] == 77
    assert h.stream_hub.has(CAM_B) is True
    assert h._inference_fps[CAM_B] == 12.0
    # Camera B keeps processing normally with its own generation.
    assert h._guarded_process(b_runtime, "frame", 78) is True
    assert h.processed == [(CAM_B, 78)]


def test_d_camera_b_processing_is_not_blocked_by_camera_a_lock():
    h = harness()
    h.pause_camera = CAM_A
    a_runtime = h.cameras.snapshot(CAM_A)
    b_runtime = h.cameras.snapshot(CAM_B)

    a_thread = spawn(lambda: h._guarded_process(a_runtime, "frame", 1))
    assert h.entered.wait(TIMEOUT)
    # Camera A holds only its own lock, so B completes while A is paused.
    assert h._guarded_process(b_runtime, "frame", 1) is True
    h.release.set()
    a_thread.join(TIMEOUT)


# --- TEST E: removal with in-flight frame --------------------------------
def test_e_removal_waits_for_in_flight_frame_and_state_cannot_reappear():
    h = harness()
    h.pause_camera = CAM_A
    runtime = h.cameras.snapshot(CAM_A)
    h._threads = {CAM_A: SimpleNamespace(is_alive=lambda: True)}

    worker = spawn(lambda: h._guarded_process(runtime, "frame", 500))
    assert h.entered.wait(TIMEOUT)

    h.cameras.remove(CAM_A)
    control = spawn(lambda: h.__class__._refresh_configuration(h))
    control.join(0.2)
    assert control.is_alive() is True
    assert h.registry.resets == []

    h.release.set()
    worker.join(TIMEOUT)
    control.join(TIMEOUT)
    assert control.is_alive() is False

    assert h.registry.resets == [CAM_A]
    assert_no_state(h, CAM_A)
    assert CAM_A not in h._seen_generation
    assert CAM_A not in h._threads
    # A late frame of the removed incarnation can never restore state.
    assert h._guarded_process(runtime, "frame", 501) is False
    assert_no_state(h, CAM_A)


# --- TEST F: generation 500 -> 1 -----------------------------------------
def test_f_new_generation_restarts_sequence_at_one():
    h = harness()
    for sequence in (499, 500):
        assert h._guarded_process(h.cameras.snapshot(CAM_A), "frame", sequence) is True
    # Same sequence twice is still refused inside one incarnation.
    assert h._guarded_process(h.cameras.snapshot(CAM_A), "frame", 500) is False

    h.cameras.replace_source(CAM_A)
    new_runtime = h.cameras.snapshot(CAM_A)
    assert h._guarded_process(new_runtime, "frame", 1) is False  # pre-transition
    h._transition_generation(CAM_A)
    assert h._frame_gate.accept(CAM_A, 1) is True  # gate is fresh after reset

    h._frame_gate.reset(CAM_A)
    assert h._guarded_process(new_runtime, "frame", 1) is True
    assert h.processed[-1] == (CAM_A, 1)


# --- TEST G: no duplicate inference thread -------------------------------
def test_g_repeated_refresh_never_duplicates_an_inference_thread():
    h = harness()
    started: list[str] = []
    lock = threading.Lock()

    def loop(camera_id):
        with lock:
            started.append(camera_id)
        h._stop.wait(TIMEOUT)

    h._inference_loop = loop
    for _ in range(4):
        h.__class__._refresh_configuration(h)

    assert sorted(h._threads) == sorted([CAM_A, CAM_B])
    for thread in h._threads.values():
        assert thread.is_alive() is True
    h._stop.set()
    for thread in list(h._threads.values()):
        thread.join(TIMEOUT)
    assert sorted(started) == sorted([CAM_A, CAM_B])

    names = [t.name for t in threading.enumerate()]
    assert len([n for n in names if n.startswith(f"infer-{CAM_A[:8]}")]) == 0
