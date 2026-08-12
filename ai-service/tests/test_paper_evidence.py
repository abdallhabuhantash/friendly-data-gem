"""Domain contract tests for the paper-evidence layer (Task 3E).

Paper evidence is raw object evidence: nothing here may imply transfer,
handoff, exchange or cheating.
"""

from __future__ import annotations

import dataclasses
import math

import pathlib

import pytest

from ._source_scan import code_text

from app.domain.geometry import BBox
from app.domain.paper_evidence import (
    CANONICAL_PAPER_CLASS,
    FORBIDDEN_PAPER_ALIASES,
    PaperDetection,
    PaperDetectorContractError,
    PaperEvidenceFrame,
    PaperEvidenceStatus,
)


def box(x1: float, y1: float, x2: float, y2: float) -> BBox:
    return BBox(x1, y1, x2 - x1, y2 - y1)


def test_canonical_class_is_single_generic_paper_class() -> None:
    assert CANONICAL_PAPER_CLASS == "paper"
    assert "book" in FORBIDDEN_PAPER_ALIASES
    assert "notebook" in FORBIDDEN_PAPER_ALIASES
    assert "phone" in FORBIDDEN_PAPER_ALIASES


def test_valid_paper_bbox_accepted() -> None:
    detection = PaperDetection(bbox=box(0.1, 0.2, 0.3, 0.5), confidence=0.42)
    assert detection.class_name == CANONICAL_PAPER_CLASS
    assert detection.confidence == pytest.approx(0.42)
    assert detection.bbox.area > 0.0


def test_border_touching_bbox_accepted() -> None:
    detection = PaperDetection(bbox=box(0.0, 0.0, 1.0, 1.0), confidence=1.0)
    assert detection.bbox.x2 == pytest.approx(1.0)


@pytest.mark.parametrize(
    "coords",
    [
        (0.6, 0.2, 0.3, 0.5),  # reversed x
        (0.1, 0.7, 0.3, 0.4),  # reversed y
    ],
)
def test_reversed_bbox_rejected(coords) -> None:
    x1, y1, x2, y2 = coords
    with pytest.raises(PaperDetectorContractError):
        PaperDetection(bbox=BBox(x1, y1, x2 - x1, y2 - y1), confidence=0.5)


@pytest.mark.parametrize(
    "bbox",
    [BBox(0.2, 0.2, 0.0, 0.3), BBox(0.2, 0.2, 0.3, 0.0), BBox(0.2, 0.2, 0.0, 0.0)],
)
def test_zero_area_bbox_rejected(bbox: BBox) -> None:
    with pytest.raises(PaperDetectorContractError):
        PaperDetection(bbox=bbox, confidence=0.5)


@pytest.mark.parametrize(
    "coords",
    [(-0.01, 0.2, 0.3, 0.5), (0.1, -0.2, 0.3, 0.5), (0.1, 0.2, 1.01, 0.5), (0.1, 0.2, 0.3, 1.2)],
)
def test_out_of_range_bbox_rejected(coords) -> None:
    x1, y1, x2, y2 = coords
    with pytest.raises(PaperDetectorContractError):
        PaperDetection(bbox=BBox(x1, y1, x2 - x1, y2 - y1), confidence=0.5)


def test_non_finite_bbox_rejected() -> None:
    with pytest.raises(PaperDetectorContractError):
        PaperDetection(bbox=BBox(0.1, 0.1, float("nan"), 0.2), confidence=0.5)


def test_bool_confidence_rejected() -> None:
    with pytest.raises(PaperDetectorContractError):
        PaperDetection(bbox=box(0.1, 0.1, 0.2, 0.2), confidence=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nan_inf_confidence_rejected(value: float) -> None:
    with pytest.raises(PaperDetectorContractError):
        PaperDetection(bbox=box(0.1, 0.1, 0.2, 0.2), confidence=value)


@pytest.mark.parametrize("value", [-0.01, 1.01, 5.0])
def test_confidence_outside_unit_range_rejected(value: float) -> None:
    with pytest.raises(PaperDetectorContractError):
        PaperDetection(bbox=box(0.1, 0.1, 0.2, 0.2), confidence=value)


def test_non_numeric_confidence_rejected() -> None:
    with pytest.raises(PaperDetectorContractError):
        PaperDetection(bbox=box(0.1, 0.1, 0.2, 0.2), confidence="0.5")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["book", "notebook", "phone", "hand", "sheet"])
def test_non_canonical_class_rejected(name: str) -> None:
    with pytest.raises(PaperDetectorContractError):
        PaperDetection(bbox=box(0.1, 0.1, 0.2, 0.2), confidence=0.5, class_name=name)


def test_model_and_crop_labels_reject_paths() -> None:
    with pytest.raises(PaperDetectorContractError):
        PaperDetection(
            bbox=box(0.1, 0.1, 0.2, 0.2),
            confidence=0.5,
            model_name="/home/private/models/paper.pt",
        )
    with pytest.raises(PaperDetectorContractError):
        PaperDetection(
            bbox=box(0.1, 0.1, 0.2, 0.2), confidence=0.5, crop_source="C:\\crops\\a"
        )
    ok = PaperDetection(
        bbox=box(0.1, 0.1, 0.2, 0.2),
        confidence=0.5,
        model_name="paper_v1.pt",
        crop_source="explicit_crop",
    )
    assert ok.model_name == "paper_v1.pt"


def test_detection_is_immutable() -> None:
    detection = PaperDetection(bbox=box(0.1, 0.1, 0.2, 0.2), confidence=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        detection.confidence = 0.9  # type: ignore[misc]


def test_frame_result_is_immutable() -> None:
    frame = PaperEvidenceFrame.empty("paper_v1.pt")
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.status = PaperEvidenceStatus.INFERENCE_FAILED  # type: ignore[misc]
    assert isinstance(frame.detections, tuple)


def test_ok_zero_detections_is_valid_and_not_a_no_paper_claim() -> None:
    frame = PaperEvidenceFrame.empty("paper_v1.pt")
    assert frame.ok is True
    assert frame.detections == ()
    assert frame.has_paper_evidence is False
    assert frame.reason is None


def test_ok_with_detections_reports_paper_evidence() -> None:
    detection = PaperDetection(bbox=box(0.2, 0.2, 0.4, 0.4), confidence=0.7)
    frame = PaperEvidenceFrame(
        status=PaperEvidenceStatus.OK, detections=(detection,), model_name="paper_v1.pt"
    )
    assert frame.has_paper_evidence is True
    assert frame.to_dict()["detections"][0]["class_name"] == "paper"


@pytest.mark.parametrize(
    "status",
    [
        PaperEvidenceStatus.MODEL_UNAVAILABLE,
        PaperEvidenceStatus.MODEL_SCHEMA_MISMATCH,
        PaperEvidenceStatus.INFERENCE_FAILED,
        PaperEvidenceStatus.MALFORMED_RESULT,
    ],
)
def test_degraded_result_cannot_carry_detections(status: PaperEvidenceStatus) -> None:
    detection = PaperDetection(bbox=box(0.2, 0.2, 0.4, 0.4), confidence=0.7)
    with pytest.raises(PaperDetectorContractError):
        PaperEvidenceFrame(status=status, detections=(detection,), reason="x")
    degraded = PaperEvidenceFrame.failure(status, "stage (ValueError)", "paper_v1.pt")
    assert degraded.detections == ()
    assert degraded.ok is False
    assert degraded.has_paper_evidence is False


def test_degraded_requires_reason_and_ok_forbids_reason() -> None:
    with pytest.raises(PaperDetectorContractError):
        PaperEvidenceFrame(status=PaperEvidenceStatus.INFERENCE_FAILED)
    with pytest.raises(PaperDetectorContractError):
        PaperEvidenceFrame(status=PaperEvidenceStatus.OK, reason="unexpected")
    with pytest.raises(PaperDetectorContractError):
        PaperEvidenceFrame.failure(PaperEvidenceStatus.OK, "not degraded")


def test_detections_must_be_a_tuple_of_detections() -> None:
    detection = PaperDetection(bbox=box(0.2, 0.2, 0.4, 0.4), confidence=0.7)
    with pytest.raises(PaperDetectorContractError):
        PaperEvidenceFrame(status=PaperEvidenceStatus.OK, detections=[detection])  # type: ignore[arg-type]
    with pytest.raises(PaperDetectorContractError):
        PaperEvidenceFrame(status=PaperEvidenceStatus.OK, detections=("paper",))  # type: ignore[arg-type]


def test_no_object_identity_or_fused_score_fields() -> None:
    fields = {field.name for field in dataclasses.fields(PaperDetection)}
    assert fields == {
        "bbox",
        "confidence",
        "class_name",
        "model_name",
        "crop_source",
        "raw_prompt",
        "backend",
    }
    assert not any("track" in name or "id" == name for name in fields)


def test_domain_module_does_not_import_temporal_or_event_layers() -> None:
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "app"
        / "domain"
        / "paper_evidence.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("handoff_temporal", "exchange_temporal", "event", "notification"):
        assert f"import {forbidden}" not in source
    # Only prose may discuss what the layer refuses to claim; code must not.
    assert "cheat" not in code_text(
        pathlib.Path(__file__).resolve().parents[1] / "app" / "domain" / "paper_evidence.py"
    ).lower()
    assert math.isfinite(1.0)  # sanity
