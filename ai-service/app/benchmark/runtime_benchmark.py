"""Real runtime benchmark: Task 1 alone vs Task 1 + asynchronous Pose.

What this module is
-------------------
Measurement tooling ONLY. It observes the production detection path (the same
``YoloDetector``, association, ``PhoneRuleEngine`` and ``PoseRuntime``) and
reports measured numbers. It never changes a decision, never adds behaviour,
regions or events, and never invents a pass/fail threshold.

Import safety
-------------
Importing this module loads no model and downloads nothing: OpenCV, torch,
ultralytics and the detector are imported inside the execution functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

from .statistics import LatencySummary, fps, percentage_change

logger = logging.getLogger(__name__)

MODE_TASK1_ONLY = "TASK1_ONLY"
MODE_TASK1_PLUS_POSE = "TASK1_PLUS_POSE"

BENCHMARK_CAMERA_ID = "benchmark-camera"
BENCHMARK_ENGINE_KEY = "mobile_phone_detection"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Immutable benchmark configuration. Both modes MUST share one instance."""

    source_video: str
    warmup_frames: int = 0
    #: 0 means "every frame in the file after warm-up".
    max_measured_frames: int = 0

    detector_model: str = "yolo11n.pt"
    detector_device: str = "auto"
    detector_imgsz: int = 960
    detector_tracker: str = "bytetrack.yaml"

    # Task 1 thresholds (production semantics, never hidden defaults of their own)
    phone_confidence: float = 0.35
    person_confidence: float = 0.40
    association_confidence: float = 0.55
    association_margin: float = 0.12
    gap_tolerance_seconds: float = 0.5
    min_duration_seconds: float = 1.5
    min_matching_frames: int = 5
    instant_detection_enabled: bool = True
    instant_confidence_threshold: float = 0.85

    # Pose (mode B only). Explicit values required; nothing is calibrated here.
    pose_model: Optional[str] = None
    pose_device: Optional[str] = None
    pose_imgsz: Optional[int] = None
    pose_confidence: Optional[float] = None
    pose_max_fps: Optional[float] = None

    @property
    def pose_configured(self) -> bool:
        return all(
            value is not None and str(value).strip() != ""
            for value in (
                self.pose_model,
                self.pose_device,
                self.pose_imgsz,
                self.pose_confidence,
                self.pose_max_fps,
            )
        )

    @property
    def pose_min_interval_seconds(self) -> float:
        if not self.pose_max_fps or float(self.pose_max_fps) <= 0.0:
            return 0.0
        return 1.0 / float(self.pose_max_fps)

    def to_dict(self) -> dict:
        """Safe, path-free view of the configuration (no absolute user paths)."""
        return {
            "source_video": Path(self.source_video).name,
            "warmup_frames": self.warmup_frames,
            "max_measured_frames": self.max_measured_frames,
            "detector_model": Path(self.detector_model).name,
            "detector_device": self.detector_device,
            "detector_imgsz": self.detector_imgsz,
            "detector_tracker": self.detector_tracker,
            "task1_thresholds": {
                "phone_confidence": self.phone_confidence,
                "person_confidence": self.person_confidence,
                "association_confidence": self.association_confidence,
                "association_margin": self.association_margin,
                "gap_tolerance_seconds": self.gap_tolerance_seconds,
                "min_duration_seconds": self.min_duration_seconds,
                "min_matching_frames": self.min_matching_frames,
                "instant_detection_enabled": self.instant_detection_enabled,
                "instant_confidence_threshold": self.instant_confidence_threshold,
            },
            "pose": {
                "configured": self.pose_configured,
                "model": Path(self.pose_model).name if self.pose_model else None,
                "device": self.pose_device,
                "imgsz": self.pose_imgsz,
                "confidence": self.pose_confidence,
                "max_fps": self.pose_max_fps,
                "min_interval_seconds": (
                    round(self.pose_min_interval_seconds, 6) if self.pose_configured else None
                ),
            },
        }


# --------------------------------------------------------------------------
# Measured results
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PoseMeasurements:
    """Pose diagnostics captured from ``PoseRuntime`` plus timed inference."""

    submitted: int = 0
    processed: int = 0
    replaced_pending: int = 0
    stale_discards: int = 0
    provider_failures: int = 0
    association_degraded: int = 0
    cadence_skipped: int = 0
    worker_errors: int = 0
    runtime_measured_pose_fps: Optional[float] = None
    inference: LatencySummary = field(default_factory=LatencySummary)
    effective_pose_fps: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "submitted": self.submitted,
            "processed": self.processed,
            "replaced_pending": self.replaced_pending,
            "stale_discards": self.stale_discards,
            "provider_failures": self.provider_failures,
            "association_degraded": self.association_degraded,
            "cadence_skipped": self.cadence_skipped,
            "worker_errors": self.worker_errors,
            "runtime_measured_pose_fps": _round(self.runtime_measured_pose_fps),
            "effective_pose_fps": _round(self.effective_pose_fps),
            "inference": self.inference.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ModeMetrics:
    """Measured metrics for ONE benchmark mode."""

    mode: str
    analysed_frames: int
    elapsed_seconds: float
    detector: LatencySummary
    pose: Optional[PoseMeasurements] = None
    warmup_frames: int = 0

    @property
    def task1_fps(self) -> Optional[float]:
        return fps(self.analysed_frames, self.elapsed_seconds)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "pose_enabled": self.pose is not None,
            "warmup_frames_excluded": self.warmup_frames,
            "analysed_frames": self.analysed_frames,
            "elapsed_seconds": _round(self.elapsed_seconds, 6),
            "task1_fps": _round(self.task1_fps),
            "detector_latency": self.detector.to_dict(),
            # A baseline run never claims pose numbers: the key is null.
            "pose": self.pose.to_dict() if self.pose is not None else None,
        }


# --------------------------------------------------------------------------
# Generic measurement loop (pure; deterministic under a fake clock)
# --------------------------------------------------------------------------
def measure_mode(
    *,
    mode: str,
    frames: Iterable[Any],
    analyse: Callable[[Any, int], None],
    warmup_frames: int = 0,
    max_measured_frames: int = 0,
    clock: Callable[[], float] = None,  # type: ignore[assignment]
    pose_measurements: Optional[Callable[[], PoseMeasurements]] = None,
) -> ModeMetrics:
    """Runs ``analyse`` per frame and measures only the non-warm-up frames."""
    if warmup_frames < 0 or max_measured_frames < 0:
        raise ValueError("warmup_frames and max_measured_frames must be >= 0")
    if clock is None:
        import time

        clock = time.perf_counter

    durations: list[float] = []
    measured = 0
    window_start: Optional[float] = None
    window_end: Optional[float] = None

    for index, frame in enumerate(frames):
        started = clock()
        analyse(frame, index)
        finished = clock()
        if index < warmup_frames:
            continue  # warm-up frames run through the models but are not counted
        if window_start is None:
            window_start = started
        window_end = finished
        durations.append((finished - started) * 1000.0)
        measured += 1
        if max_measured_frames and measured >= max_measured_frames:
            break

    elapsed = 0.0
    if window_start is not None and window_end is not None:
        elapsed = max(0.0, window_end - window_start)

    pose = pose_measurements() if pose_measurements is not None else None
    return ModeMetrics(
        mode=mode,
        analysed_frames=measured,
        elapsed_seconds=elapsed,
        detector=LatencySummary.from_samples(durations),
        pose=pose,
        warmup_frames=warmup_frames,
    )


# --------------------------------------------------------------------------
# Hardware diagnostics (safe facts only)
# --------------------------------------------------------------------------
def safe_hardware_info() -> dict:
    """Safe hardware/runtime facts. Never fails, never contains secrets."""
    import platform

    info: dict = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu_count": None,
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_current_device": None,
        "gpu_name": None,
        "cuda_version": None,
        "process_rss_bytes": None,
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch  # noqa: PLC0415

        info["torch_available"] = True
        info["torch_version"] = str(torch.__version__)
        info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        available = bool(torch.cuda.is_available())
        info["cuda_available"] = available
        if available:
            info["cuda_device_count"] = int(torch.cuda.device_count())
            index = int(torch.cuda.current_device())
            info["cuda_current_device"] = index
            # Device name is READ at runtime; no GPU model is ever hard-coded.
            info["gpu_name"] = str(torch.cuda.get_device_name(index))
    except Exception:  # noqa: BLE001 - CUDA diagnostics are optional
        pass
    return info


def reset_cuda_peak_memory() -> None:
    """Best-effort peak-memory reset before a mode; silent when unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def cuda_peak_memory() -> dict:
    """Peak CUDA memory if measurable, otherwise truthful ``None`` values."""
    result = {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    try:
        import torch

        if torch.cuda.is_available():
            result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    except Exception:  # noqa: BLE001
        pass
    return result


# --------------------------------------------------------------------------
# Comparison + report
# --------------------------------------------------------------------------
def build_comparison(baseline: ModeMetrics, with_pose: Optional[ModeMetrics]) -> Optional[dict]:
    """Measured differences only: no verdict, no invented acceptance limit."""
    if with_pose is None:
        return None
    without = baseline.task1_fps
    withp = with_pose.task1_fps
    return {
        "task1_fps_without_pose": _round(without),
        "task1_fps_with_pose": _round(withp),
        "absolute_fps_difference": (
            None if without is None or withp is None else _round(withp - without)
        ),
        "percentage_change": _round(percentage_change(without, withp)),
        "detector_latency": {
            "mean_ms_without_pose": _round(baseline.detector.mean_ms),
            "mean_ms_with_pose": _round(with_pose.detector.mean_ms),
            "mean_ms_percentage_change": _round(
                percentage_change(baseline.detector.mean_ms, with_pose.detector.mean_ms)
            ),
            "median_ms_without_pose": _round(baseline.detector.median_ms),
            "median_ms_with_pose": _round(with_pose.detector.median_ms),
            "median_ms_percentage_change": _round(
                percentage_change(baseline.detector.median_ms, with_pose.detector.median_ms)
            ),
            "p95_ms_without_pose": _round(baseline.detector.p95_ms),
            "p95_ms_with_pose": _round(with_pose.detector.p95_ms),
            "p95_ms_percentage_change": _round(
                percentage_change(baseline.detector.p95_ms, with_pose.detector.p95_ms)
            ),
        },
        "verdict": "not_decided_measurement_only",
    }


def build_report(
    *,
    config: BenchmarkConfig,
    baseline: ModeMetrics,
    with_pose: Optional[ModeMetrics] = None,
    hardware: Optional[dict] = None,
    baseline_memory: Optional[dict] = None,
    pose_memory: Optional[dict] = None,
    timestamp: Optional[datetime] = None,
) -> dict:
    """JSON-serialisable benchmark report. Contains no credentials or secrets."""
    moment = timestamp or datetime.now(timezone.utc)
    return {
        "timestamp": moment.isoformat(),
        "benchmark_version": 1,
        "configuration": config.to_dict(),
        "hardware": hardware if hardware is not None else safe_hardware_info(),
        "baseline": baseline.to_dict(),
        "baseline_cuda_memory": baseline_memory,
        "with_pose": with_pose.to_dict() if with_pose is not None else None,
        "with_pose_cuda_memory": pose_memory,
        "comparison": build_comparison(baseline, with_pose),
    }


def format_summary(report: dict) -> str:
    """Human-readable console summary of a report produced above."""
    hardware = report.get("hardware") or {}
    baseline = report.get("baseline") or {}
    with_pose = report.get("with_pose")
    comparison = report.get("comparison")
    lines: list[str] = []
    lines.append("VIGILANT EYE RUNTIME BENCHMARK")
    lines.append(f"Timestamp: {report.get('timestamp')}")
    lines.append("")
    lines.append("Hardware:")
    lines.append(f"  Platform:  {hardware.get('platform')} {hardware.get('platform_release')}")
    lines.append(f"  GPU:       {hardware.get('gpu_name') or 'not available'}")
    lines.append(
        f"  CUDA:      available={hardware.get('cuda_available')} "
        f"version={hardware.get('cuda_version')} devices={hardware.get('cuda_device_count')}"
    )
    lines.append(f"  Torch:     {hardware.get('torch_version') or 'not installed'}")
    lines.append("")
    lines.append("TASK 1 ONLY")
    lines.extend(_mode_lines(baseline))
    lines.append("")
    if with_pose is None:
        lines.append("TASK 1 + POSE: not run")
    else:
        lines.append("TASK 1 + POSE")
        lines.extend(_mode_lines(with_pose))
        pose = with_pose.get("pose") or {}
        inference = pose.get("inference") or {}
        lines.append(f"  Pose submitted:    {pose.get('submitted')}")
        lines.append(f"  Pose processed:    {pose.get('processed')}")
        lines.append(f"  Pose FPS:          {_show(pose.get('effective_pose_fps'))}")
        lines.append(f"  Pose mean (ms):    {_show(inference.get('mean_ms'))}")
        lines.append(f"  Pose median (ms):  {_show(inference.get('median_ms'))}")
        lines.append(f"  Pose p95 (ms):     {_show(inference.get('p95_ms'))}")
        lines.append(f"  Pose max (ms):     {_show(inference.get('max_ms'))}")
        lines.append(f"  Pose replacements: {pose.get('replaced_pending')}")
        lines.append(f"  Pose stale:        {pose.get('stale_discards')}")
        lines.append(f"  Provider failures: {pose.get('provider_failures')}")
    if comparison:
        latency = comparison.get("detector_latency") or {}
        lines.append("")
        lines.append("COMPARISON")
        lines.append(
            f"  Task 1 FPS difference:        {_show(comparison.get('absolute_fps_difference'))}"
        )
        lines.append(
            f"  Task 1 FPS percentage change: {_show(comparison.get('percentage_change'))}"
        )
        lines.append(
            f"  Detector mean change (%):     {_show(latency.get('mean_ms_percentage_change'))}"
        )
        lines.append(
            f"  Detector p95 change (%):      {_show(latency.get('p95_ms_percentage_change'))}"
        )
        lines.append("  Verdict: measurement only (no acceptance threshold applied)")
    return "\n".join(lines)


def _mode_lines(mode: dict) -> list[str]:
    detector = mode.get("detector_latency") or {}
    return [
        f"  Frames:            {mode.get('analysed_frames')}",
        f"  Elapsed (s):       {_show(mode.get('elapsed_seconds'))}",
        f"  Task 1 FPS:        {_show(mode.get('task1_fps'))}",
        f"  Detector mean:     {_show(detector.get('mean_ms'))} ms",
        f"  Detector median:   {_show(detector.get('median_ms'))} ms",
        f"  Detector p95:      {_show(detector.get('p95_ms'))} ms",
        f"  Detector max:      {_show(detector.get('max_ms'))} ms",
    ]


def _show(value: Any) -> str:
    return "unavailable" if value is None else str(value)


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), digits)


# --------------------------------------------------------------------------
# Real execution (models load only here)
# --------------------------------------------------------------------------
class TimedPoseProvider:
    """Transparent timing wrapper around a real ``PoseProvider``.

    It adds measurement only: the wrapped provider's result is returned
    untouched, so no pose decision is affected.
    """

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self.durations_ms: list[float] = []

    @property
    def available(self) -> bool:
        return bool(getattr(self._provider, "available", False))

    @property
    def model_name(self) -> Any:
        return getattr(self._provider, "model_name", None)

    def infer(self, frame: Any):  # noqa: ANN201 - PoseFrameResult
        import time

        started = time.perf_counter()
        try:
            return self._provider.infer(frame)
        finally:
            self.durations_ms.append((time.perf_counter() - started) * 1000.0)


def iter_video_frames(path: str) -> Iterator[Any]:
    """Yields frames from a local video file using the real OpenCV reader."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the benchmark video: {Path(path).name}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def _rule_config(config: BenchmarkConfig):  # noqa: ANN202
    from ..domain.models import RuleConfig

    return RuleConfig(
        id="benchmark-rule",
        name="Benchmark phone rule",
        engine_key=BENCHMARK_ENGINE_KEY,
        available=True,
        enabled=True,
        severity="critical",
        confidence_threshold=config.phone_confidence,
        person_confidence_threshold=config.person_confidence,
        association_confidence_threshold=config.association_confidence,
        min_duration_seconds=config.min_duration_seconds,
        min_matching_frames=config.min_matching_frames,
        instant_detection_enabled=config.instant_detection_enabled,
        instant_confidence_threshold=config.instant_confidence_threshold,
        save_snapshot=False,
    )


def run_mode(config: BenchmarkConfig, *, mode: str) -> tuple[ModeMetrics, dict]:
    """Executes ONE benchmark mode over the configured video file."""
    import time

    from ..ai.detector import YoloDetector
    from ..ai.observation_builder import build_frame_observations
    from ..ai.phone_rule_engine import PhoneRuleEngine
    from ..domain.models import CameraConfig, SourceType

    video = Path(config.source_video)
    if not video.exists():
        raise FileNotFoundError(f"benchmark video not found: {video.name}")

    detector = YoloDetector(
        config.detector_model,
        config.detector_device,
        config.detector_imgsz,
        config.detector_tracker,
    )
    engine = PhoneRuleEngine(
        association_margin=config.association_margin,
        gap_tolerance_seconds=config.gap_tolerance_seconds,
    )
    camera = CameraConfig(
        id=BENCHMARK_CAMERA_ID,
        name="Benchmark",
        source_type=SourceType.DEMO,
        is_demo=True,
    )
    rule = _rule_config(config)

    pose_runtime = None
    timed_provider: Optional[TimedPoseProvider] = None
    if mode == MODE_TASK1_PLUS_POSE:
        if not config.pose_configured:
            raise ValueError(
                "pose mode requires explicit pose model, device, imgsz, confidence and max fps"
            )
        from ..ai.pose_provider import UltralyticsPoseProvider
        from ..runtime.pose_runtime import PoseRuntime

        timed_provider = TimedPoseProvider(
            UltralyticsPoseProvider(
                model_name=str(config.pose_model),
                device=str(config.pose_device),
                imgsz=int(config.pose_imgsz),  # type: ignore[arg-type]
                confidence=float(config.pose_confidence),  # type: ignore[arg-type]
            )
        )
        pose_runtime = PoseRuntime(
            timed_provider,
            min_interval_seconds=config.pose_min_interval_seconds,
            association_spec=None,
        )
        pose_runtime.activate(BENCHMARK_CAMERA_ID, 1)
        pose_runtime.start()

    def analyse(frame: Any, index: int) -> None:
        detections = detector.detect(frame, camera.id)
        engine.process_frame(
            camera=camera,
            rule=rule,
            detections=detections,
            now=time.monotonic(),
            source_mode="demo",
            detected_at=datetime.now(timezone.utc),
        )
        if pose_runtime is not None:
            observations = build_frame_observations(
                camera_id=camera.id,
                detections=detections,
                frame_sequence=index,
                observed_at=datetime.now(timezone.utc),
                source_mode="demo",
            )
            pose_runtime.maybe_submit(
                camera_id=camera.id,
                generation=1,
                frame_sequence=index,
                observed_at=observations.observed_at,
                observations=observations,
                copy_frame=lambda: frame.copy(),
                source_mode="demo",
            )

    reset_cuda_peak_memory()
    try:
        metrics = measure_mode(
            mode=mode,
            frames=iter_video_frames(str(video)),
            analyse=analyse,
            warmup_frames=config.warmup_frames,
            max_measured_frames=config.max_measured_frames,
            pose_measurements=(
                None
                if pose_runtime is None
                else lambda: pose_measurements_from_runtime(
                    pose_runtime,
                    BENCHMARK_CAMERA_ID,
                    durations_ms=timed_provider.durations_ms if timed_provider else [],
                )
            ),
        )
    finally:
        if pose_runtime is not None:
            pose_runtime.stop(timeout=5.0)
    return metrics, cuda_peak_memory()


def pose_measurements_from_runtime(
    runtime: Any, camera_id: str, *, durations_ms: Iterable[float]
) -> PoseMeasurements:
    """Reuses ``PoseRuntime``'s own measured accounting; nothing recomputed."""
    metrics = runtime.metrics(camera_id) or {}
    samples = list(durations_ms)
    inference = LatencySummary.from_samples(samples)
    processed = int(metrics.get("processed") or 0)
    total_ms = sum(samples)
    effective = fps(processed, total_ms / 1000.0) if samples else None
    return PoseMeasurements(
        submitted=int(metrics.get("submitted") or 0),
        processed=processed,
        replaced_pending=int(metrics.get("replaced_pending") or 0),
        stale_discards=int(runtime.stale_discards(camera_id)),
        provider_failures=int(metrics.get("provider_failures") or 0),
        association_degraded=int(metrics.get("association_degraded") or 0),
        cadence_skipped=int(metrics.get("cadence_skipped") or 0),
        worker_errors=int(metrics.get("worker_errors") or 0),
        runtime_measured_pose_fps=metrics.get("measured_pose_fps"),
        inference=inference,
        effective_pose_fps=effective,
    )


def run_benchmark(config: BenchmarkConfig, *, include_pose: bool = True) -> dict:
    """Runs the baseline and, when requested, the Task 1 + Pose mode."""
    hardware = safe_hardware_info()
    baseline, baseline_memory = run_mode(config, mode=MODE_TASK1_ONLY)
    with_pose = None
    pose_memory = None
    if include_pose and config.pose_configured:
        with_pose, pose_memory = run_mode(config, mode=MODE_TASK1_PLUS_POSE)
    return build_report(
        config=config,
        baseline=baseline,
        with_pose=with_pose,
        hardware=hardware,
        baseline_memory=baseline_memory,
        pose_memory=pose_memory,
    )


def write_report(report: dict, output_path: str) -> Path:
    """Writes the JSON report, creating parent folders as needed."""
    import json

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


__all__ = [
    "BENCHMARK_CAMERA_ID",
    "BenchmarkConfig",
    "MODE_TASK1_ONLY",
    "MODE_TASK1_PLUS_POSE",
    "ModeMetrics",
    "PoseMeasurements",
    "TimedPoseProvider",
    "build_comparison",
    "build_report",
    "cuda_peak_memory",
    "format_summary",
    "iter_video_frames",
    "measure_mode",
    "pose_measurements_from_runtime",
    "reset_cuda_peak_memory",
    "run_benchmark",
    "run_mode",
    "safe_hardware_info",
    "write_report",
]
