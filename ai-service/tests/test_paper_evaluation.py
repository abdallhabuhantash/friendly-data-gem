"""Offline paper-evaluation tooling tests (Task 3E-B).

Descriptive measurement only; no model, no video, no network.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from app.benchmark.paper_evaluation import (
    REQUIRED_NEGATIVE_SCENES,
    REQUIRED_POSITIVE_SCENES,
    REVIEW_VERDICTS,
    DescriptiveMetrics,
    EvaluationReport,
    EvaluationRun,
    FrameObservation,
    iter_frames,
    render_console_summary,
    safe_source_label,
    summarize,
)
from app.domain.geometry import BBox
from app.domain.paper_evidence import (
    PaperDetection,
    PaperEvidenceFrame,
    PaperEvidenceStatus,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "evaluate_paper_open_vocab.py"
MODULE_PATH = ROOT / "app" / "benchmark" / "paper_evaluation.py"


def detection(prompt: str = "paper", confidence: float = 0.5) -> PaperDetection:
    return PaperDetection(
        bbox=BBox(0.1, 0.1, 0.4, 0.4),
        confidence=confidence,
        raw_prompt=prompt,
        backend="ultralytics-open-vocab",
        model_name="yolov8s-worldv2.pt",
    )


def frame(*detections: PaperDetection, index: int = 0, latency: float = 10.0) -> FrameObservation:
    return FrameObservation(
        frame_index=index,
        result=PaperEvidenceFrame(
            status=PaperEvidenceStatus.OK,
            detections=tuple(detections),
            model_name="yolov8s-worldv2.pt",
        ),
        latency_ms=latency,
        timestamp_seconds=index / 10.0,
    )


# --- descriptive metrics -------------------------------------------------


def test_summary_counts_frames_and_detections_per_prompt() -> None:
    observations = [
        frame(detection("paper", 0.4), detection("folded paper", 0.6), index=0),
        frame(index=5),
        frame(detection("paper", 0.8), index=10),
    ]
    metrics = summarize(observations, ("paper", "folded paper"), elapsed_seconds=1.0)
    assert metrics.frames_sampled == 3
    assert metrics.frames_with_detections == 2
    assert metrics.total_detections == 3
    assert metrics.detections_by_prompt == {"paper": 2, "folded paper": 1}
    assert metrics.confidence_max == pytest.approx(0.8)
    assert metrics.processing_fps == pytest.approx(3.0)


def test_unfired_prompts_are_reported_as_zero() -> None:
    metrics = summarize([frame(index=0)], ("paper", "exam paper"))
    assert metrics.detections_by_prompt == {"paper": 0, "exam paper": 0}


def test_empty_run_is_safe() -> None:
    metrics = summarize([], ("paper",), elapsed_seconds=0.0)
    payload = metrics.to_dict()
    assert metrics.frames_sampled == 0
    assert metrics.total_detections == 0
    assert payload["confidence"]["mean"] is None
    assert payload["measurement_kind"] == "descriptive_only"


def test_degraded_frames_are_counted_by_status() -> None:
    degraded = FrameObservation(
        frame_index=3,
        result=PaperEvidenceFrame(
            status=PaperEvidenceStatus.MODEL_UNAVAILABLE, reason="model unavailable"
        ),
        latency_ms=1.0,
    )
    metrics = summarize([frame(detection()), degraded], ("paper",))
    assert metrics.frames_by_status == {"ok": 1, "model_unavailable": 1}
    assert metrics.frames_with_detections == 1


def test_summary_is_deterministic() -> None:
    observations = [frame(detection("paper", 0.4), index=0), frame(detection("paper", 0.9), index=1)]
    assert summarize(observations, ("paper",)) == summarize(observations, ("paper",))


# --- report / truthfulness ----------------------------------------------


def report(runs=()) -> EvaluationReport:
    return EvaluationReport(
        source_name="/home/secret/videos/exam_clip.mp4",
        backend="ultralytics-open-vocab",
        model_name="/home/secret/models/yolov8s-worldv2.pt",
        device="cpu",
        imgsz=960,
        confidence_threshold=0.15,
        frame_stride=5,
        runs=tuple(runs),
    )


def build_run() -> EvaluationRun:
    observations = (frame(detection("paper", 0.42), index=0), frame(index=5))
    return EvaluationRun(
        prompts=("paper",),
        observations=observations,
        metrics=summarize(observations, ("paper",), elapsed_seconds=0.5),
    )


def test_report_is_json_serialisable_and_hides_private_paths() -> None:
    payload = report([build_run()]).to_dict()
    text = json.dumps(payload)
    assert "/home/secret" not in text
    assert payload["source_name"] == "exam_clip.mp4"
    assert payload["model_name"] == "yolov8s-worldv2.pt"
    assert payload["production_ready"] is False
    assert payload["review_verdicts"] == list(REVIEW_VERDICTS)


def test_report_lists_required_positive_and_negative_scenes() -> None:
    payload = report([build_run()]).to_dict()
    for scene in ("empty desk", "hands only", "phone", "notebook or book", "white desk surface"):
        assert scene in payload["required_negative_scenes"]
    for scene in ("small slip", "folded sheet", "motion blur", "paper moving between two people"):
        assert scene in payload["required_positive_scenes"]
    assert set(REQUIRED_NEGATIVE_SCENES) and set(REQUIRED_POSITIVE_SCENES)


def test_no_precision_recall_or_map_language_anywhere() -> None:
    text = json.dumps(report([build_run()]).to_dict()).lower()
    console = render_console_summary(report([build_run()])).lower()
    for banned in ("precision", "recall", " map", "mean average", "accuracy", "f1"):
        assert banned not in text
        assert banned not in console


def test_console_summary_states_zero_detection_meaning() -> None:
    text = render_console_summary(report([build_run()]))
    assert "no paper evidence was detected" in text.lower()
    assert "true_paper" in text


def test_every_detection_row_carries_prompt_and_empty_review_slot() -> None:
    payload = build_run().to_dict()
    row = payload["frames"][0]["detections"][0]
    assert row["raw_prompt"] == "paper"
    assert row["review"] is None


def test_crop_mode_recorded_when_used() -> None:
    run = EvaluationRun(
        prompts=("paper",),
        observations=(),
        metrics=summarize([], ("paper",)),
        mode="explicit_crop",
        crop={"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5},
    )
    payload = run.to_dict()
    assert payload["mode"] == "explicit_crop"
    assert payload["crop"]["x2"] == 0.5


def test_safe_source_label_strips_directories() -> None:
    assert safe_source_label("/a/b/c/clip.mp4") == "clip.mp4"


def test_metrics_dataclasses_are_immutable() -> None:
    metrics = summarize([frame(detection())], ("paper",))
    with pytest.raises(Exception):
        metrics.frames_sampled = 99  # type: ignore[misc]
    assert isinstance(metrics, DescriptiveMetrics)


def test_iter_frames_rejects_invalid_stride() -> None:
    for stride in (0, -1, True):
        with pytest.raises(ValueError):
            list(iter_frames("clip.mp4", stride))  # type: ignore[arg-type]


# --- import safety / isolation ------------------------------------------


def test_evaluation_module_has_no_module_level_side_effects() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    assert [
        node for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ] == []
    top_level_imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "cv2" not in top_level_imports
    assert "ultralytics" not in top_level_imports


def test_cli_script_imports_without_loading_models_or_videos() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert [
        node for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ] == []
    assert "cv2" not in {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }


def test_cli_requires_all_explicit_parameters() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("evaluate_paper_open_vocab", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    required = {
        action.dest for action in parser._actions if getattr(action, "required", False)
    }
    assert required == {"source", "weights", "prompts", "device", "imgsz", "confidence", "frame_stride"}


def test_cli_declares_no_training_and_no_runtime_integration() -> None:
    doc = SCRIPT_PATH.read_text(encoding="utf-8").lower()
    assert "trains nothing" in doc
    assert "never imported by the live runtime" in doc


def test_evaluation_tooling_does_not_touch_events_or_temporal_layers() -> None:
    for path in (MODULE_PATH, SCRIPT_PATH):
        source = path.read_text(encoding="utf-8")
        for forbidden in (
            "handoff_temporal",
            "exchange_temporal",
            "event_publisher",
            "notification_manager",
            "orchestrator",
            "supabase",
        ):
            assert forbidden not in source
