"""Offline open-vocabulary paper evaluation (measurement only — Task 3E-B).

What this module is
-------------------
Descriptive measurement tooling for the DORMANT open-vocabulary paper detector.
It is never imported by the live runtime, adds no GPU cost to Task 1, and makes
no behavioural claim: it reports what the model did, nothing more.

Import safety
-------------
Importing this module loads no model, opens no video and downloads nothing:
OpenCV and the detector are imported inside the execution functions.

Truthfulness rules
------------------
* Without labelled ground truth, only DESCRIPTIVE metrics are produced: frames
  sampled, frames with detections, detection counts, per-prompt counts,
  confidence distribution, latency and processing FPS. The words precision,
  recall, accuracy and mAP are deliberately NOT produced anywhere.
* Every detection keeps the exact prompt that fired.
* Reports contain BASENAMES only — never absolute private source paths.
* Manual review is a report-side workflow (``review`` slots left as ``null``),
  never something the detector itself decides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from ..domain.paper_evidence import PaperEvidenceFrame, PaperEvidenceStatus
from .statistics import fps, mean, median, percentile

logger = logging.getLogger(__name__)

#: Allowed manual-review verdicts (applied by a HUMAN after the run).
REVIEW_VERDICTS: tuple[str, ...] = ("true_paper", "false_positive", "uncertain")

#: Negative scenes an acceptable evaluation MUST include.
REQUIRED_NEGATIVE_SCENES: tuple[str, ...] = (
    "empty desk",
    "hands only",
    "handshake / hands near each other",
    "phone",
    "pen or pencil",
    "notebook or book",
    "clothing",
    "white desk surface",
    "monitor or tablet screen",
    "printed signs or background posters",
)

#: Positive scenes that must be measured before trusting the detector.
REQUIRED_POSITIVE_SCENES: tuple[str, ...] = (
    "normal exam sheet",
    "small slip",
    "folded sheet",
    "partially occluded sheet",
    "one hand holding paper",
    "two hands holding paper",
    "paper moving between two people",
    "paper on desk",
    "paper angled to camera",
    "distant or small paper",
    "motion blur",
)


def safe_source_label(source: str) -> str:
    """Basename only: absolute private paths never reach a report."""
    return Path(str(source)).name or "source"


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """One sampled frame: what the model reported plus measured latency."""

    frame_index: int
    result: PaperEvidenceFrame
    latency_ms: Optional[float] = None
    timestamp_seconds: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frame_index": self.frame_index,
            "status": self.result.status.value,
            "detection_count": len(self.result.detections),
            "detections": [
                {
                    **detection.to_dict(),
                    # Manual review is filled in by a human, never by the model.
                    "review": None,
                }
                for detection in self.result.detections
            ],
        }
        if self.timestamp_seconds is not None:
            payload["timestamp_seconds"] = round(float(self.timestamp_seconds), 6)
        if self.latency_ms is not None:
            payload["latency_ms"] = round(float(self.latency_ms), 3)
        if self.result.reason is not None:
            payload["reason"] = self.result.reason
        return payload


@dataclass(frozen=True, slots=True)
class DescriptiveMetrics:
    """Descriptive-only measurements. NOT precision/recall/accuracy/mAP."""

    frames_sampled: int
    frames_with_detections: int
    total_detections: int
    detections_by_prompt: dict[str, int]
    frames_by_status: dict[str, int]
    confidence_mean: Optional[float] = None
    confidence_median: Optional[float] = None
    confidence_p95: Optional[float] = None
    confidence_max: Optional[float] = None
    latency_mean_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    processing_fps: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        def rounded(value: Optional[float], digits: int = 6) -> Optional[float]:
            return None if value is None else round(float(value), digits)

        return {
            "measurement_kind": "descriptive_only",
            "note": (
                "No labelled ground truth exists, so precision/recall/accuracy/mAP "
                "are intentionally not reported."
            ),
            "frames_sampled": self.frames_sampled,
            "frames_with_detections": self.frames_with_detections,
            "total_detections": self.total_detections,
            "detections_by_prompt": dict(sorted(self.detections_by_prompt.items())),
            "frames_by_status": dict(sorted(self.frames_by_status.items())),
            "confidence": {
                "mean": rounded(self.confidence_mean),
                "median": rounded(self.confidence_median),
                "p95": rounded(self.confidence_p95),
                "max": rounded(self.confidence_max),
            },
            "latency_ms": {
                "mean": rounded(self.latency_mean_ms, 3),
                "p95": rounded(self.latency_p95_ms, 3),
            },
            "processing_fps": rounded(self.processing_fps, 3),
        }


def summarize(
    observations: Sequence[FrameObservation],
    prompts: Sequence[str] = (),
    elapsed_seconds: Optional[float] = None,
) -> DescriptiveMetrics:
    """Deterministic descriptive summary; safe with zero frames/detections."""
    confidences: list[float] = []
    latencies: list[float] = []
    by_prompt: dict[str, int] = {prompt: 0 for prompt in prompts}
    by_status: dict[str, int] = {}
    frames_with_detections = 0
    total_detections = 0

    for observation in observations:
        status = observation.result.status.value
        by_status[status] = by_status.get(status, 0) + 1
        if observation.latency_ms is not None:
            latencies.append(float(observation.latency_ms))
        detections = observation.result.detections
        if detections:
            frames_with_detections += 1
            total_detections += len(detections)
        for detection in detections:
            confidences.append(float(detection.confidence))
            key = detection.raw_prompt or detection.class_name
            by_prompt[key] = by_prompt.get(key, 0) + 1

    frames_sampled = len(observations)
    return DescriptiveMetrics(
        frames_sampled=frames_sampled,
        frames_with_detections=frames_with_detections,
        total_detections=total_detections,
        detections_by_prompt=by_prompt,
        frames_by_status=by_status,
        confidence_mean=mean(confidences),
        confidence_median=median(confidences),
        confidence_p95=percentile(confidences, 95.0),
        confidence_max=max(confidences) if confidences else None,
        latency_mean_ms=mean(latencies),
        latency_p95_ms=percentile(latencies, 95.0),
        processing_fps=(
            fps(frames_sampled, float(elapsed_seconds))
            if elapsed_seconds is not None
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """One prompt configuration evaluated over one source."""

    prompts: tuple[str, ...]
    observations: tuple[FrameObservation, ...]
    metrics: DescriptiveMetrics
    mode: str = "full_frame"
    crop: Optional[dict[str, float]] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompts": list(self.prompts),
            "mode": self.mode,
            "metrics": self.metrics.to_dict(),
            "frames": [observation.to_dict() for observation in self.observations],
        }
        if self.crop is not None:
            payload["crop"] = self.crop
        return payload


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Full JSON-serialisable report; never contains absolute source paths."""

    source_name: str
    backend: str
    model_name: str
    device: str
    imgsz: int
    confidence_threshold: float
    frame_stride: int
    runs: tuple[EvaluationRun, ...] = field(default_factory=tuple)
    generated_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": "3E-B open-vocabulary paper evidence evaluation",
            "generated_at": self.generated_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_name": safe_source_label(self.source_name),
            "backend": self.backend,
            "model_name": safe_source_label(self.model_name),
            "device": self.device,
            "imgsz": self.imgsz,
            "confidence_threshold": round(float(self.confidence_threshold), 6),
            "frame_stride": self.frame_stride,
            "review_verdicts": list(REVIEW_VERDICTS),
            "required_negative_scenes": list(REQUIRED_NEGATIVE_SCENES),
            "required_positive_scenes": list(REQUIRED_POSITIVE_SCENES),
            "production_ready": False,
            "runs": [run.to_dict() for run in self.runs],
        }


def render_console_summary(report: EvaluationReport) -> str:
    """Human-readable summary. Descriptive wording only."""
    lines: list[str] = []
    lines.append(f"Source            : {safe_source_label(report.source_name)}")
    lines.append(f"Backend / weights : {report.backend} / {safe_source_label(report.model_name)}")
    lines.append(
        f"Device / imgsz    : {report.device} / {report.imgsz} "
        f"(conf {report.confidence_threshold}, stride {report.frame_stride})"
    )
    for run in report.runs:
        metrics = run.metrics
        lines.append("")
        lines.append(f"[{run.mode}] prompts: {', '.join(run.prompts)}")
        lines.append(
            f"  frames sampled {metrics.frames_sampled}, "
            f"frames with detections {metrics.frames_with_detections}, "
            f"total detections {metrics.total_detections}"
        )
        if metrics.detections_by_prompt:
            fired = ", ".join(
                f"{prompt}={count}"
                for prompt, count in sorted(metrics.detections_by_prompt.items())
            )
            lines.append(f"  detections by prompt: {fired}")
        if metrics.frames_by_status:
            statuses = ", ".join(
                f"{status}={count}" for status, count in sorted(metrics.frames_by_status.items())
            )
            lines.append(f"  frame statuses: {statuses}")
        lines.append(
            "  confidence mean/median/p95/max: "
            f"{_fmt(metrics.confidence_mean)}/{_fmt(metrics.confidence_median)}/"
            f"{_fmt(metrics.confidence_p95)}/{_fmt(metrics.confidence_max)}"
        )
        lines.append(
            f"  latency mean/p95 ms: {_fmt(metrics.latency_mean_ms, 1)}/"
            f"{_fmt(metrics.latency_p95_ms, 1)}   processing fps: "
            f"{_fmt(metrics.processing_fps, 2)}"
        )
    lines.append("")
    lines.append(
        "Descriptive measurements only: no labelled ground truth exists, so no "
        "precision/recall/accuracy/mAP is claimed. Zero detections means 'no paper "
        "evidence was detected by this model', not 'there is no paper'. Review each "
        "detection manually as true_paper / false_positive / uncertain."
    )
    return "\n".join(lines)


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


# --------------------------------------------------------------------------
# Execution (lazy heavy imports; never touched at import time)
# --------------------------------------------------------------------------
def iter_frames(source: str, frame_stride: int, max_frames: int = 0) -> Iterable[tuple[int, Any, Optional[float]]]:
    """Yields ``(frame_index, frame, timestamp_seconds)`` from a local file."""
    if not isinstance(frame_stride, int) or isinstance(frame_stride, bool) or frame_stride <= 0:
        raise ValueError("frame_stride must be a positive integer")
    import cv2  # lazy: heavy dependency

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source: {safe_source_label(source)}")
    try:
        index = 0
        emitted = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % frame_stride == 0:
                position = capture.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = (
                    float(position) / 1000.0
                    if isinstance(position, (int, float)) and position and position > 0
                    else None
                )
                yield index, frame, timestamp
                emitted += 1
                if max_frames and emitted >= max_frames:
                    break
            index += 1
    finally:
        capture.release()


def evaluate_source(
    detector: Any,
    source: str,
    frame_stride: int,
    max_frames: int = 0,
    crop: Any = None,
    mode: str = "full_frame",
    annotated_output: Optional[str] = None,
) -> EvaluationRun:
    """Runs ONE prompt configuration over a local video/image source."""
    import time

    observations: list[FrameObservation] = []
    writer = None
    started = time.perf_counter()
    try:
        for frame_index, frame, timestamp in iter_frames(source, frame_stride, max_frames):
            call_started = time.perf_counter()
            result = detector.infer(frame, crop=crop) if crop is not None else detector.infer(frame)
            latency_ms = (time.perf_counter() - call_started) * 1000.0
            result = result.with_frame_metadata(
                frame_index=frame_index, timestamp_seconds=timestamp
            )
            observations.append(
                FrameObservation(
                    frame_index=frame_index,
                    result=result,
                    latency_ms=latency_ms,
                    timestamp_seconds=timestamp,
                )
            )
            if annotated_output:
                writer = _write_annotated(writer, annotated_output, frame, result)
    finally:
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started
    prompts = tuple(getattr(detector, "prompts", ()))
    return EvaluationRun(
        prompts=prompts,
        observations=tuple(observations),
        metrics=summarize(observations, prompts, elapsed),
        mode=mode,
        crop=crop.to_dict() if crop is not None and hasattr(crop, "to_dict") else None,
    )


def _write_annotated(writer: Any, output_path: str, frame: Any, result: PaperEvidenceFrame) -> Any:
    """Draws paper boxes + raw prompt labels for manual visual review."""
    import cv2  # lazy: heavy dependency

    height, width = frame.shape[:2]
    canvas = frame.copy()
    for detection in result.detections:
        x1, y1, x2, y2 = detection.bbox.to_pixels(width, height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 2)
        caption = f"{detection.raw_prompt or detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            canvas, caption, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 200, 255), 1, cv2.LINE_AA,
        )
    if result.status is not PaperEvidenceStatus.OK:
        cv2.putText(
            canvas, result.status.value, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 255), 2, cv2.LINE_AA,
        )
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
    writer.write(canvas)
    return writer
