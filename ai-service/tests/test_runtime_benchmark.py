"""Deterministic tests for the runtime benchmark tooling (no GPU, no models)."""

from __future__ import annotations

import json
import sys

import pytest

from app.benchmark import runtime_benchmark as rb
from app.benchmark.statistics import (
    LatencySummary,
    fps,
    mean,
    median,
    percentage_change,
    percentile,
)


class FakeClock:
    """Deterministic monotonic clock: every read advances by a fixed step."""

    def __init__(self, step: float = 0.010) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def _config(**overrides) -> rb.BenchmarkConfig:
    base = dict(source_video="/videos/demo.mp4", warmup_frames=0, max_measured_frames=0)
    base.update(overrides)
    return rb.BenchmarkConfig(**base)


# --- A. statistics ------------------------------------------------------
def test_statistics_basic_calculations():
    values = [10.0, 20.0, 30.0, 40.0]
    assert mean(values) == 25.0
    assert median(values) == 25.0
    assert median([1.0, 2.0, 3.0]) == 2.0
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 100.0) == 40.0


# --- B. empty metrics handled truthfully -------------------------------
def test_empty_series_is_unavailable_not_zero():
    assert mean([]) is None
    assert median([]) is None
    assert percentile([], 95.0) is None
    summary = LatencySummary.from_samples([])
    assert summary.count == 0
    assert summary.mean_ms is None and summary.p95_ms is None and summary.max_ms is None
    assert fps(0, 10.0) is None


# --- C. FPS uses measured wall-clock duration ---------------------------
def test_fps_uses_measured_duration_and_rejects_zero():
    assert fps(30, 2.0) == 15.0
    assert fps(30, 0.0) is None
    assert fps(-1, 2.0) is None


# --- D. deterministic p95 ----------------------------------------------
def test_p95_is_deterministic():
    samples = [float(value) for value in range(1, 101)]
    assert percentile(samples, 95.0) == pytest.approx(95.05)
    assert LatencySummary.from_samples(samples).p95_ms == pytest.approx(95.05)
    assert percentile([5.0], 95.0) == 5.0


# --- E. comparison percentage ------------------------------------------
def test_percentage_change_and_comparison():
    assert percentage_change(20.0, 15.0) == pytest.approx(-25.0)
    assert percentage_change(0.0, 15.0) is None
    assert percentage_change(None, 15.0) is None

    baseline = rb.ModeMetrics(
        mode=rb.MODE_TASK1_ONLY,
        analysed_frames=20,
        elapsed_seconds=1.0,
        detector=LatencySummary.from_samples([50.0] * 20),
    )
    with_pose = rb.ModeMetrics(
        mode=rb.MODE_TASK1_PLUS_POSE,
        analysed_frames=15,
        elapsed_seconds=1.0,
        detector=LatencySummary.from_samples([60.0] * 15),
        pose=rb.PoseMeasurements(submitted=5, processed=4),
    )
    comparison = rb.build_comparison(baseline, with_pose)
    assert comparison["task1_fps_without_pose"] == 20.0
    assert comparison["task1_fps_with_pose"] == 15.0
    assert comparison["absolute_fps_difference"] == -5.0
    assert comparison["percentage_change"] == pytest.approx(-25.0)
    assert comparison["detector_latency"]["mean_ms_percentage_change"] == pytest.approx(20.0)
    assert comparison["verdict"] == "not_decided_measurement_only"
    assert rb.build_comparison(baseline, None) is None


# --- F. baseline claims no pose ----------------------------------------
def test_baseline_result_contains_no_pose_metrics():
    metrics = rb.measure_mode(
        mode=rb.MODE_TASK1_ONLY,
        frames=[object()] * 4,
        analyse=lambda frame, index: None,
        clock=FakeClock(),
    )
    payload = metrics.to_dict()
    assert payload["pose_enabled"] is False
    assert payload["pose"] is None
    report = rb.build_report(config=_config(), baseline=metrics, hardware={"cuda_available": False})
    assert report["with_pose"] is None
    assert report["comparison"] is None


# --- G. pose diagnostics come from PoseRuntime -------------------------
class _StubRuntime:
    def __init__(self, metrics: dict, stale: int) -> None:
        self._metrics = metrics
        self._stale = stale

    def metrics(self, camera_id: str) -> dict:
        return self._metrics

    def stale_discards(self, camera_id: str) -> int:
        return self._stale


def test_pose_measurements_reuse_runtime_diagnostics():
    runtime = _StubRuntime(
        {
            "submitted": 9,
            "processed": 4,
            "replaced_pending": 3,
            "provider_failures": 1,
            "association_degraded": 2,
            "cadence_skipped": 5,
            "worker_errors": 0,
            "measured_pose_fps": 1.75,
        },
        stale=2,
    )
    measurements = rb.pose_measurements_from_runtime(
        runtime, "cam", durations_ms=[100.0, 200.0, 300.0, 400.0]
    )
    assert measurements.submitted == 9
    assert measurements.processed == 4
    assert measurements.replaced_pending == 3
    assert measurements.stale_discards == 2
    assert measurements.provider_failures == 1
    assert measurements.association_degraded == 2
    assert measurements.runtime_measured_pose_fps == 1.75
    assert measurements.inference.mean_ms == 250.0
    assert measurements.inference.max_ms == 400.0
    assert measurements.effective_pose_fps == pytest.approx(4.0)

    empty = rb.pose_measurements_from_runtime(_StubRuntime({}, 0), "cam", durations_ms=[])
    assert empty.processed == 0
    assert empty.effective_pose_fps is None
    assert empty.inference.mean_ms is None


def test_timed_pose_provider_does_not_change_results():
    class Provider:
        available = True
        model_name = "pose.pt"

        def infer(self, frame):  # noqa: ANN001, ANN202
            return ("result", frame)

    timed = rb.TimedPoseProvider(Provider())
    assert timed.available is True
    assert timed.model_name == "pose.pt"
    assert timed.infer("frame-a") == ("result", "frame-a")
    assert len(timed.durations_ms) == 1


# --- H. CUDA unavailable → valid report --------------------------------
def test_report_valid_without_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    info = rb.safe_hardware_info()
    assert info["cuda_available"] is False
    assert info["gpu_name"] is None
    rb.reset_cuda_peak_memory()
    memory = rb.cuda_peak_memory()
    assert memory == {"peak_allocated_bytes": None, "peak_reserved_bytes": None}

    metrics = rb.measure_mode(
        mode=rb.MODE_TASK1_ONLY,
        frames=[object()] * 3,
        analyse=lambda frame, index: None,
        clock=FakeClock(),
    )
    report = rb.build_report(config=_config(), baseline=metrics, hardware=info)
    assert "Task 1 FPS" in rb.format_summary(report)


# --- I. safe hardware fields contain no secrets ------------------------
def test_report_contains_no_secrets():
    config = _config(
        source_video="C:/Users/secret-user/videos/demo.mp4",
        pose_model="C:/Users/secret-user/models/yolo11n-pose.pt",
        pose_device="cuda:0",
        pose_imgsz=640,
        pose_confidence=0.3,
        pose_max_fps=2.0,
        detector_model="C:/Users/secret-user/models/yolo11n.pt",
    )
    metrics = rb.measure_mode(
        mode=rb.MODE_TASK1_ONLY,
        frames=[object()] * 2,
        analyse=lambda frame, index: None,
        clock=FakeClock(),
    )
    report = rb.build_report(config=config, baseline=metrics, hardware=rb.safe_hardware_info())
    blob = json.dumps(report)
    for forbidden in ("secret-user", "rtsp://", "Users", "service_role", "token"):
        assert forbidden not in blob
    assert report["configuration"]["source_video"] == "demo.mp4"
    assert report["configuration"]["pose"]["model"] == "yolo11n-pose.pt"


# --- J. JSON roundtrip -------------------------------------------------
def test_json_serialization_roundtrip():
    baseline = rb.measure_mode(
        mode=rb.MODE_TASK1_ONLY,
        frames=[object()] * 3,
        analyse=lambda frame, index: None,
        clock=FakeClock(),
    )
    with_pose = rb.measure_mode(
        mode=rb.MODE_TASK1_PLUS_POSE,
        frames=[object()] * 3,
        analyse=lambda frame, index: None,
        clock=FakeClock(0.02),
        pose_measurements=lambda: rb.PoseMeasurements(
            submitted=3, processed=2, inference=LatencySummary.from_samples([10.0, 30.0])
        ),
    )
    report = rb.build_report(
        config=_config(), baseline=baseline, with_pose=with_pose, hardware={"cuda_available": False}
    )
    restored = json.loads(json.dumps(report))
    assert restored["with_pose"]["pose"]["processed"] == 2
    assert restored["comparison"]["task1_fps_without_pose"] is not None


# --- K/L. warm-up excluded, measured limit respected -------------------
def test_warmup_frames_excluded_and_limit_respected():
    seen: list[int] = []

    def analyse(frame, index):  # noqa: ANN001, ANN202
        seen.append(index)

    metrics = rb.measure_mode(
        mode=rb.MODE_TASK1_ONLY,
        frames=[object()] * 20,
        analyse=analyse,
        warmup_frames=3,
        max_measured_frames=5,
        clock=FakeClock(),
    )
    # 3 warm-up frames ran through the pipeline but were not measured.
    assert seen[:3] == [0, 1, 2]
    assert len(seen) == 8
    assert metrics.analysed_frames == 5
    assert metrics.detector.count == 5
    assert metrics.warmup_frames == 3
    assert metrics.elapsed_seconds > 0.0
    assert metrics.task1_fps is not None


def test_measure_mode_rejects_negative_configuration():
    with pytest.raises(ValueError):
        rb.measure_mode(
            mode=rb.MODE_TASK1_ONLY,
            frames=[],
            analyse=lambda frame, index: None,
            warmup_frames=-1,
        )


def test_zero_measured_frames_reports_unavailable_fps():
    metrics = rb.measure_mode(
        mode=rb.MODE_TASK1_ONLY,
        frames=[object()],
        analyse=lambda frame, index: None,
        warmup_frames=5,
        clock=FakeClock(),
    )
    assert metrics.analysed_frames == 0
    assert metrics.task1_fps is None
    assert metrics.to_dict()["task1_fps"] is None


# --- M. import performs no model initialisation ------------------------
def test_benchmark_import_loads_no_models():
    for module in ("ultralytics", "torch", "cv2"):
        assert module not in getattr(rb, "__dict__", {})
    source = rb.__doc__ or ""
    assert "Measurement tooling" in source
    # Nothing heavy is imported at module scope.
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(rb.__file__).read_text(encoding="utf-8"))
    top_level_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")
    for forbidden in ("torch", "cv2", "ultralytics"):
        assert forbidden not in top_level_imports


def test_pose_configuration_requires_explicit_values():
    assert _config().pose_configured is False
    partial = _config(pose_model="p.pt", pose_device="cuda:0")
    assert partial.pose_configured is False
    complete = _config(
        pose_model="p.pt",
        pose_device="cuda:0",
        pose_imgsz=640,
        pose_confidence=0.3,
        pose_max_fps=2.0,
    )
    assert complete.pose_configured is True
    assert complete.pose_min_interval_seconds == pytest.approx(0.5)


def test_summary_reports_unavailable_instead_of_fake_zero():
    metrics = rb.ModeMetrics(
        mode=rb.MODE_TASK1_ONLY,
        analysed_frames=0,
        elapsed_seconds=0.0,
        detector=LatencySummary(),
    )
    text = rb.format_summary(
        rb.build_report(config=_config(), baseline=metrics, hardware={"gpu_name": None})
    )
    assert "unavailable" in text
    assert "TASK 1 + POSE: not run" in text
