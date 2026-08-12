"""Provider tests for the paper detector (Task 3E).

No real weights and no network access: the Ultralytics model is injected via
``model_factory``. These tests prove there is NO stock/COCO/book fallback and no
class-index guessing.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from app.ai.crop_transform import CropTransform
from app.ai.paper_detector import (
    PaperDetectorConfigError,
    UltralyticsPaperDetector,
    model_class_names,
    paper_class_index,
    parse_paper_result,
)
from app.domain.paper_evidence import PaperEvidenceStatus

AI_DIR = pathlib.Path(__file__).resolve().parents[1] / "app" / "ai"
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"


class FakeBoxes:
    def __init__(self, xyxyn, conf, cls) -> None:
        self.xyxyn = xyxyn
        self.conf = conf
        self.cls = cls


class FakeResult:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


class FakeModel:
    def __init__(self, names, results=None, raise_on_predict: BaseException | None = None) -> None:
        self.names = names
        self._results = results
        self._raise = raise_on_predict
        self.predict_calls: list[dict] = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        if self._raise is not None:
            raise self._raise
        return self._results


PAPER_ONLY = {0: "paper"}
PAPER_SECOND = {0: "hand", 1: "paper"}


def detector(model: FakeModel | None = None, **kwargs) -> UltralyticsPaperDetector:
    factory = (lambda path: model) if model is not None else None
    return UltralyticsPaperDetector(
        weights_path=kwargs.pop("weights_path", "models/paper_v1.pt"),
        model_factory=factory,
        require_existing_weights=kwargs.pop("require_existing_weights", False),
        **kwargs,
    )


# --- configuration -------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights_path": ""},
        {"imgsz": 0},
        {"imgsz": True},
        {"confidence": 1.5},
        {"confidence": float("nan")},
        {"confidence": True},
        {"device": " "},
        {"paper_class": ""},
        {"paper_class": "book"},
        {"paper_class": "notebook"},
    ],
)
def test_invalid_configuration_raises(kwargs) -> None:
    base = {"weights_path": "models/paper_v1.pt", "require_existing_weights": False}
    base.update(kwargs)
    with pytest.raises(PaperDetectorConfigError):
        UltralyticsPaperDetector(**base)


# --- model availability --------------------------------------------------


def test_missing_model_file_reports_model_unavailable() -> None:
    provider = UltralyticsPaperDetector(weights_path="models/does_not_exist_paper.pt")
    frame = provider.infer(object())
    assert frame.status is PaperEvidenceStatus.MODEL_UNAVAILABLE
    assert provider.available is False
    assert frame.detections == ()
    assert frame.model_name == "does_not_exist_paper.pt"


def test_model_load_failure_reported_safely_without_path_leak() -> None:
    secret_path = "/private/creds/rtsp:pass@paper.pt"

    def boom(path: str):
        raise RuntimeError(f"cannot open {secret_path} token=abc123")

    provider = UltralyticsPaperDetector(
        weights_path=secret_path, model_factory=boom, require_existing_weights=False
    )
    frame = provider.infer(object())
    assert frame.status is PaperEvidenceStatus.MODEL_UNAVAILABLE
    assert frame.reason is not None
    assert "RuntimeError" in frame.reason
    assert "token=abc123" not in frame.reason
    assert "/private/creds" not in frame.reason
    assert provider.available is False


def test_load_failure_is_sticky() -> None:
    calls = {"n": 0}

    def boom(path: str):
        calls["n"] += 1
        raise OSError("nope")

    provider = UltralyticsPaperDetector(
        weights_path="models/paper_v1.pt", model_factory=boom, require_existing_weights=False
    )
    assert provider.infer(object()).status is PaperEvidenceStatus.MODEL_UNAVAILABLE
    assert provider.infer(object()).status is PaperEvidenceStatus.MODEL_UNAVAILABLE
    assert calls["n"] == 1


# --- schema validation ---------------------------------------------------


@pytest.mark.parametrize(
    "names",
    [
        {0: "book"},
        {0: "notebook", 1: "phone"},
        {0: "sheet"},
        {0: "person", 1: "cell phone"},
        [],
    ],
)
def test_missing_canonical_paper_class_is_schema_mismatch(names) -> None:
    provider = detector(FakeModel(names))
    frame = provider.infer(object())
    assert frame.status is PaperEvidenceStatus.MODEL_SCHEMA_MISMATCH
    assert provider.available is False


def test_unreadable_class_names_is_schema_mismatch() -> None:
    frame = detector(FakeModel(None)).infer(object())
    assert frame.status is PaperEvidenceStatus.MODEL_SCHEMA_MISMATCH


def test_no_class_index_zero_guessing() -> None:
    # A single-class checkpoint whose only class is 'book' must never be used.
    model = FakeModel({0: "book"}, results=[FakeResult(FakeBoxes([[0.1, 0.1, 0.2, 0.2]], [0.9], [0]))])
    frame = detector(model).infer(object())
    assert frame.status is PaperEvidenceStatus.MODEL_SCHEMA_MISMATCH
    assert model.predict_calls == []  # inference never even attempted


def test_paper_class_index_resolution_is_name_based() -> None:
    assert paper_class_index(PAPER_SECOND, "paper") == 1
    assert paper_class_index({0: "PAPER"}, "paper") == 0
    assert paper_class_index({0: "book"}, "paper") is None
    assert model_class_names(FakeModel(["paper"])) == {0: "paper"}
    assert model_class_names(FakeModel({"0": "paper"})) == {0: "paper"}
    assert model_class_names(FakeModel({0: 3})) is None


# --- valid parsing -------------------------------------------------------


def test_valid_paper_detection_parsed() -> None:
    model = FakeModel(
        PAPER_SECOND,
        results=[FakeResult(FakeBoxes([[0.2, 0.3, 0.5, 0.7]], [0.81], [1]))],
    )
    frame = detector(model).infer(object())
    assert frame.status is PaperEvidenceStatus.OK
    assert len(frame.detections) == 1
    detection = frame.detections[0]
    assert detection.class_name == "paper"
    assert detection.confidence == pytest.approx(0.81)
    assert detection.bbox.x == pytest.approx(0.2)
    assert detection.bbox.x2 == pytest.approx(0.5)
    assert detection.model_name == "paper_v1.pt"
    assert detection.crop_source is None


def test_zero_detections_is_ok() -> None:
    model = FakeModel(PAPER_ONLY, results=[FakeResult(FakeBoxes([], [], []))])
    frame = detector(model).infer(object())
    assert frame.status is PaperEvidenceStatus.OK
    assert frame.detections == ()
    assert frame.has_paper_evidence is False


def test_non_paper_classes_are_ignored_not_mapped() -> None:
    model = FakeModel(
        PAPER_SECOND,
        results=[
            FakeResult(
                FakeBoxes(
                    [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.45]],
                    [0.95, 0.6],
                    [0, 1],
                )
            )
        ],
    )
    frame = detector(model).infer(object())
    assert frame.status is PaperEvidenceStatus.OK
    assert len(frame.detections) == 1
    assert frame.detections[0].bbox.x == pytest.approx(0.3)


def test_epsilon_drift_snapped_but_substantive_range_rejected() -> None:
    model = FakeModel(
        PAPER_ONLY,
        results=[FakeResult(FakeBoxes([[-1e-12, 0.1, 0.5, 1.0 + 1e-12]], [0.5], [0]))],
    )
    frame = detector(model).infer(object())
    assert frame.status is PaperEvidenceStatus.OK
    assert frame.detections[0].bbox.x == pytest.approx(0.0)


# --- malformed provider output ------------------------------------------


@pytest.mark.parametrize(
    "boxes",
    [
        FakeBoxes([[0.5, 0.1, 0.2, 0.4]], [0.7], [0]),  # reversed x
        FakeBoxes([[0.1, 0.6, 0.4, 0.2]], [0.7], [0]),  # reversed y
        FakeBoxes([[0.1, 0.1, 0.1, 0.4]], [0.7], [0]),  # zero width
        FakeBoxes([[-0.4, 0.1, 0.4, 0.4]], [0.7], [0]),  # out of range
        FakeBoxes([[0.1, 0.1, 1.6, 0.4]], [0.7], [0]),  # out of range
        FakeBoxes([[0.1, 0.1, 0.4]], [0.7], [0]),  # short row
        FakeBoxes([[float("nan"), 0.1, 0.4, 0.4]], [0.7], [0]),
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], [1.5], [0]),  # bad confidence
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], [float("inf")], [0]),
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], [0.7, 0.8], [0]),  # length mismatch
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], [0.7], []),  # cls mismatch
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], [0.7], [0.5]),  # non-integral class
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], None, [0]),  # missing conf channel
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], [0.7], None),  # missing cls channel
        FakeBoxes(None, [0.7], [0]),  # missing boxes
    ],
)
def test_malformed_boxes_yield_malformed_result(boxes) -> None:
    model = FakeModel(PAPER_ONLY, results=[FakeResult(boxes)])
    frame = detector(model).infer(object())
    assert frame.status is PaperEvidenceStatus.MALFORMED_RESULT
    assert frame.detections == ()


def test_whole_frame_strictness_no_silent_row_dropping() -> None:
    model = FakeModel(
        PAPER_ONLY,
        results=[
            FakeResult(
                FakeBoxes([[0.1, 0.1, 0.4, 0.4], [0.7, 0.1, 0.3, 0.4]], [0.9, 0.8], [0, 0])
            )
        ],
    )
    frame = detector(model).infer(object())
    assert frame.status is PaperEvidenceStatus.MALFORMED_RESULT


def test_result_without_boxes_attribute_is_malformed() -> None:
    class Bare:
        boxes = None

    model = FakeModel(PAPER_ONLY, results=[Bare()])
    assert detector(model).infer(object()).status is PaperEvidenceStatus.MALFORMED_RESULT


@pytest.mark.parametrize("results", [[], [FakeResult(FakeBoxes([], [], [])), FakeResult(None)], None])
def test_wrong_result_count_is_malformed(results) -> None:
    model = FakeModel(PAPER_ONLY, results=results)
    frame = detector(model).infer(object())
    assert frame.status is PaperEvidenceStatus.MALFORMED_RESULT


# --- inference failure ---------------------------------------------------


def test_inference_failure_reported_without_leakage() -> None:
    model = FakeModel(
        PAPER_ONLY,
        raise_on_predict=RuntimeError("rtsp://admin:secret@10.0.0.9/stream failed"),
    )
    provider = detector(model)
    frame = provider.infer(object())
    assert frame.status is PaperEvidenceStatus.INFERENCE_FAILED
    assert frame.reason == "paper inference failed (RuntimeError)"
    assert "secret" not in frame.reason
    # An inference failure is transient: the provider stays usable.
    assert provider.available is True


# --- crop support --------------------------------------------------------


def test_crop_detection_mapped_back_to_full_frame() -> None:
    model = FakeModel(PAPER_ONLY, results=[FakeResult(FakeBoxes([[0.0, 0.0, 0.5, 0.5]], [0.6], [0]))])
    frame = detector(model).infer(object(), crop=CropTransform(0.4, 0.4, 0.8, 0.8))
    assert frame.status is PaperEvidenceStatus.OK
    detection = frame.detections[0]
    assert detection.bbox.x == pytest.approx(0.4)
    assert detection.bbox.x2 == pytest.approx(0.6)
    assert detection.crop_source == "explicit_crop"


def test_identity_crop_leaves_coordinates_and_provenance_untouched() -> None:
    model = FakeModel(PAPER_ONLY, results=[FakeResult(FakeBoxes([[0.1, 0.2, 0.3, 0.4]], [0.6], [0]))])
    frame = detector(model).infer(object(), crop=CropTransform.full_frame())
    assert frame.detections[0].bbox.x == pytest.approx(0.1)
    assert frame.detections[0].crop_source is None


def test_invalid_crop_argument_is_a_programming_error() -> None:
    model = FakeModel(PAPER_ONLY, results=[FakeResult(FakeBoxes([], [], []))])
    with pytest.raises(PaperDetectorConfigError):
        detector(model).infer(object(), crop=(0.1, 0.1, 0.5, 0.5))  # type: ignore[arg-type]


def test_parse_paper_result_is_pure_and_deterministic() -> None:
    result = FakeResult(FakeBoxes([[0.2, 0.2, 0.4, 0.4]], [0.5], [0]))
    first = parse_paper_result(result, 0, "paper_v1.pt")
    second = parse_paper_result(result, 0, "paper_v1.pt")
    assert first == second


# --- static guarantees ---------------------------------------------------


def test_provider_declares_no_stock_or_book_fallback() -> None:
    source = (AI_DIR / "paper_detector.py").read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in ("yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "coco.yaml"):
        assert forbidden not in lowered
    # 'book' appears only in prose forbidding it, never as a mapped class value.
    assert '"book"' not in source and "'book'" not in source
    for forbidden in ("handoff_temporal", "exchange_temporal", "event_publisher", "notification"):
        assert forbidden not in source


def test_provider_never_calls_track_or_downloads_at_import() -> None:
    tree = ast.parse((AI_DIR / "paper_detector.py").read_text(encoding="utf-8"))
    module_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert module_calls == []
    assert ".track(" not in (AI_DIR / "paper_detector.py").read_text(encoding="utf-8")


def test_paper_module_is_not_wired_into_runtime() -> None:
    for name in ("orchestrator.py", "camera_manager.py"):
        candidates = list((AI_DIR.parent).rglob(name))
        for path in candidates:
            source = path.read_text(encoding="utf-8")
            assert "paper_detector" not in source
            assert "paper_evidence" not in source
    for name in ("engine_registry.py", "exchange_temporal_state.py", "pose_provider.py"):
        source = (AI_DIR / name).read_text(encoding="utf-8")
        assert "paper" not in source.lower()
