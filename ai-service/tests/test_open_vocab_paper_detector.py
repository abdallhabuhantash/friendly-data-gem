"""Open-vocabulary paper provider tests (Task 3E-B).

No real weights, no network: the backend is injected via ``model_factory``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from ._source_scan import code_text
from app.ai.crop_transform import CropTransform
from app.ai.open_vocab_paper_detector import (
    PAPER_PROMPT_CANDIDATES,
    OpenVocabConfigError,
    OpenVocabPaperDetector,
    PaperPromptConfig,
    PaperPromptConfigError,
    PROHIBITED_PROMPT_TERMS,
    parse_open_vocab_result,
)
from app.domain.paper_evidence import CANONICAL_PAPER_CLASS, PaperEvidenceStatus

AI_DIR = pathlib.Path(__file__).resolve().parents[1] / "app" / "ai"
MODULE_PATH = AI_DIR / "open_vocab_paper_detector.py"


class FakeBoxes:
    def __init__(self, xyxyn, conf, cls) -> None:
        self.xyxyn = xyxyn
        self.conf = conf
        self.cls = cls


class FakeResult:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


class FakeWorldModel:
    """Minimal stand-in for an Ultralytics open-vocabulary model."""

    def __init__(self, results=None, raise_on_predict=None, raise_on_set_classes=None) -> None:
        self._results = results
        self._raise_on_predict = raise_on_predict
        self._raise_on_set_classes = raise_on_set_classes
        self.classes: list[str] | None = None
        self.predict_calls: list[dict] = []

    def set_classes(self, classes):
        if self._raise_on_set_classes is not None:
            raise self._raise_on_set_classes
        self.classes = list(classes)

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        if self._raise_on_predict is not None:
            raise self._raise_on_predict
        return self._results


class NoPromptModel:
    """A closed-vocabulary checkpoint: no ``set_classes``."""

    names = {0: "paper"}

    def predict(self, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("inference must not be attempted")


PROMPTS = PaperPromptConfig(("paper", "small paper slip"))


def detector(model=None, prompts: PaperPromptConfig = PROMPTS, **kwargs):
    factory = (lambda path: model) if model is not None else None
    return OpenVocabPaperDetector(
        weights_path=kwargs.pop("weights_path", "models/yolov8s-worldv2.pt"),
        prompts=prompts,
        model_factory=factory,
        require_existing_weights=kwargs.pop("require_existing_weights", False),
        **kwargs,
    )


# --- prompt configuration ------------------------------------------------


def test_valid_prompt_configuration() -> None:
    config = PaperPromptConfig(tuple(PAPER_PROMPT_CANDIDATES))
    assert config.prompts == PAPER_PROMPT_CANDIDATES
    assert config.prompt_for_index(0) == "paper"
    assert config.prompt_for_index(len(config.prompts)) is None
    assert config.prompt_for_index(-1) is None
    assert config.prompt_for_index(True) is None
    assert "paper" in config.label
    assert config.to_list() == list(PAPER_PROMPT_CANDIDATES)


def test_candidate_prompt_terms_are_explicit_paper_concepts() -> None:
    assert PAPER_PROMPT_CANDIDATES == (
        "paper",
        "sheet of paper",
        "exam paper",
        "paper slip",
        "small paper slip",
        "folded paper",
    )


@pytest.mark.parametrize("prompts", [(), [], "paper", None, 5])
def test_empty_or_non_list_prompts_rejected(prompts) -> None:
    with pytest.raises(PaperPromptConfigError):
        PaperPromptConfig(prompts)  # type: ignore[arg-type]


@pytest.mark.parametrize("prompt", ["", "   ", "\t\n", None, 3])
def test_blank_or_non_string_prompt_rejected(prompt) -> None:
    with pytest.raises(PaperPromptConfigError):
        PaperPromptConfig(("paper", prompt))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "prompts",
    [
        ("paper", "paper"),
        ("paper", "PAPER"),
        ("sheet of paper", "sheet  of   paper"),
    ],
)
def test_duplicate_prompt_rejected(prompts) -> None:
    with pytest.raises(PaperPromptConfigError):
        PaperPromptConfig(prompts)


@pytest.mark.parametrize(
    "prompt",
    ["book", "notebook", "folder", "phone", "cell phone", "pen", "hand", "desk", "object", "thing"],
)
def test_prohibited_fallback_prompts_rejected(prompt: str) -> None:
    with pytest.raises(PaperPromptConfigError):
        PaperPromptConfig((prompt,))
    with pytest.raises(PaperPromptConfigError):
        PaperPromptConfig(("paper", prompt))


@pytest.mark.parametrize("prompt", ["white rectangle", "document", "card", "exam"])
def test_non_paper_concept_prompts_rejected(prompt: str) -> None:
    with pytest.raises(PaperPromptConfigError):
        PaperPromptConfig((prompt,))


def test_provider_requires_validated_prompt_config() -> None:
    with pytest.raises(OpenVocabConfigError):
        OpenVocabPaperDetector(
            weights_path="models/w.pt",
            prompts=["paper"],  # type: ignore[arg-type]
            require_existing_weights=False,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights_path": " "},
        {"imgsz": 0},
        {"imgsz": True},
        {"confidence": 1.2},
        {"confidence": float("nan")},
        {"confidence": True},
        {"device": ""},
        {"backend_label": " "},
    ],
)
def test_invalid_provider_configuration_raises(kwargs) -> None:
    base = {
        "weights_path": "models/w.pt",
        "prompts": PROMPTS,
        "require_existing_weights": False,
    }
    base.update(kwargs)
    with pytest.raises(OpenVocabConfigError):
        OpenVocabPaperDetector(**base)


# --- model / prompt failure statuses -------------------------------------


def test_missing_weights_report_model_unavailable() -> None:
    provider = OpenVocabPaperDetector(
        weights_path="models/definitely_missing_world.pt", prompts=PROMPTS
    )
    frame = provider.infer(object())
    assert frame.status is PaperEvidenceStatus.MODEL_UNAVAILABLE
    assert frame.detections == ()
    assert provider.available is False


def test_load_failure_hides_paths_and_credentials() -> None:
    def boom(path: str):
        raise RuntimeError("failed /home/secret/models/w.pt rtsp://u:p@10.0.0.5")

    provider = OpenVocabPaperDetector(
        weights_path="/home/secret/models/w.pt",
        prompts=PROMPTS,
        model_factory=boom,
        require_existing_weights=False,
    )
    frame = provider.infer(object())
    assert frame.status is PaperEvidenceStatus.MODEL_UNAVAILABLE
    assert frame.reason == "open-vocabulary model unavailable (RuntimeError)"
    assert "/home/secret" not in (frame.reason or "")
    assert "rtsp://" not in (frame.reason or "")
    assert frame.model_name == "w.pt"


def test_closed_vocabulary_checkpoint_is_prompt_configuration_invalid() -> None:
    provider = detector(NoPromptModel())
    frame = provider.infer(object())
    assert frame.status is PaperEvidenceStatus.PROMPT_CONFIGURATION_INVALID
    assert provider.available is False


def test_prompt_application_failure_is_not_downgraded_to_empty_evidence() -> None:
    model = FakeWorldModel(raise_on_set_classes=ValueError("/private/path token=x"))
    frame = detector(model).infer(object())
    assert frame.status is PaperEvidenceStatus.PROMPT_CONFIGURATION_INVALID
    assert frame.reason == "paper prompt configuration failed (ValueError)"
    assert frame.detections == ()
    assert model.predict_calls == []


def test_prompts_are_applied_to_backend_exactly_once() -> None:
    model = FakeWorldModel(results=[FakeResult(FakeBoxes([], [], []))])
    provider = detector(model)
    provider.infer(object())
    provider.infer(object())
    assert model.classes == ["paper", "small paper slip"]
    assert len(model.predict_calls) == 2


def test_inference_failure_status_without_leakage() -> None:
    model = FakeWorldModel(
        results=None, raise_on_predict=RuntimeError("rtsp://admin:pw@cam/stream broke")
    )
    provider = detector(model)
    frame = provider.infer(object())
    assert frame.status is PaperEvidenceStatus.INFERENCE_FAILED
    assert frame.reason == "paper inference failed (RuntimeError)"
    assert "pw@" not in (frame.reason or "")
    assert provider.available is True


# --- detections ----------------------------------------------------------


def test_detection_preserves_original_prompt_and_canonical_semantic() -> None:
    model = FakeWorldModel(results=[FakeResult(FakeBoxes([[0.2, 0.3, 0.4, 0.6]], [0.33], [1]))])
    frame = detector(model).infer(object())
    assert frame.status is PaperEvidenceStatus.OK
    detection = frame.detections[0]
    assert detection.raw_prompt == "small paper slip"
    assert detection.class_name == CANONICAL_PAPER_CLASS
    assert detection.backend == "ultralytics-open-vocab"
    assert detection.model_name == "yolov8s-worldv2.pt"
    assert detection.confidence == pytest.approx(0.33)
    assert detection.to_dict()["raw_prompt"] == "small paper slip"


def test_multiple_prompts_tracked_independently() -> None:
    model = FakeWorldModel(
        results=[
            FakeResult(
                FakeBoxes([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.7, 0.8]], [0.4, 0.6], [0, 1])
            )
        ]
    )
    frame = detector(model).infer(object())
    prompts = [detection.raw_prompt for detection in frame.detections]
    assert prompts == ["paper", "small paper slip"]


def test_valid_zero_detection_frame_is_ok() -> None:
    model = FakeWorldModel(results=[FakeResult(FakeBoxes([], [], []))])
    frame = detector(model).infer(object())
    assert frame.status is PaperEvidenceStatus.OK
    assert frame.detections == ()
    assert frame.has_paper_evidence is False


@pytest.mark.parametrize(
    "boxes",
    [
        FakeBoxes([[0.6, 0.1, 0.2, 0.4]], [0.5], [0]),  # reversed x
        FakeBoxes([[0.1, 0.7, 0.4, 0.3]], [0.5], [0]),  # reversed y
        FakeBoxes([[0.1, 0.1, 0.1, 0.4]], [0.5], [0]),  # zero area
        FakeBoxes([[-0.3, 0.1, 0.4, 0.4]], [0.5], [0]),  # out of range
        FakeBoxes([[0.1, 0.1, 0.4, 1.9]], [0.5], [0]),  # out of range
        FakeBoxes([[float("nan"), 0.1, 0.4, 0.4]], [0.5], [0]),
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], [float("inf")], [0]),
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], [1.4], [0]),
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], [0.5], [7]),  # index outside prompts
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], [0.5], [0.5]),  # non-integral class
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], [0.5, 0.5], [0]),  # length mismatch
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], None, [0]),
        FakeBoxes([[0.1, 0.1, 0.4, 0.4]], [0.5], None),
        FakeBoxes(None, [0.5], [0]),
    ],
)
def test_malformed_model_output_degrades_whole_frame(boxes) -> None:
    model = FakeWorldModel(results=[FakeResult(boxes)])
    frame = detector(model).infer(object())
    assert frame.status is PaperEvidenceStatus.MALFORMED_RESULT
    assert frame.detections == ()


def test_one_bad_row_discards_the_whole_frame() -> None:
    model = FakeWorldModel(
        results=[
            FakeResult(
                FakeBoxes([[0.1, 0.1, 0.3, 0.3], [0.9, 0.1, 0.4, 0.3]], [0.7, 0.7], [0, 0])
            )
        ]
    )
    assert detector(model).infer(object()).status is PaperEvidenceStatus.MALFORMED_RESULT


@pytest.mark.parametrize("results", [[], None, [FakeResult(FakeBoxes([], [], [])), FakeResult(None)]])
def test_wrong_result_count_is_malformed(results) -> None:
    model = FakeWorldModel(results=results)
    assert detector(model).infer(object()).status is PaperEvidenceStatus.MALFORMED_RESULT


def test_bool_confidence_from_backend_rejected() -> None:
    model = FakeWorldModel(results=[FakeResult(FakeBoxes([[0.1, 0.1, 0.4, 0.4]], [True], [0]))])
    assert detector(model).infer(object()).status is PaperEvidenceStatus.MALFORMED_RESULT


# --- optional offline crop experiment ------------------------------------


def test_optional_crop_maps_back_to_full_frame() -> None:
    model = FakeWorldModel(results=[FakeResult(FakeBoxes([[0.0, 0.0, 0.5, 0.5]], [0.5], [0]))])
    frame = detector(model).infer(object(), crop=CropTransform(0.2, 0.2, 0.6, 0.6))
    detection = frame.detections[0]
    assert detection.bbox.x == pytest.approx(0.2)
    assert detection.bbox.x2 == pytest.approx(0.4)
    assert detection.crop_source == "explicit_crop"


def test_invalid_crop_argument_is_a_programming_error() -> None:
    model = FakeWorldModel(results=[FakeResult(FakeBoxes([], [], []))])
    with pytest.raises(OpenVocabConfigError):
        detector(model).infer(object(), crop=(0.1, 0.1, 0.5, 0.5))  # type: ignore[arg-type]


def test_parser_is_deterministic() -> None:
    result = FakeResult(FakeBoxes([[0.2, 0.2, 0.4, 0.4]], [0.5], [0]))
    first = parse_open_vocab_result(result, PROMPTS, "w.pt", "backend")
    second = parse_open_vocab_result(result, PROMPTS, "w.pt", "backend")
    assert first == second


def test_frame_metadata_can_be_attached_without_mutation() -> None:
    model = FakeWorldModel(results=[FakeResult(FakeBoxes([[0.2, 0.2, 0.4, 0.4]], [0.5], [0]))])
    frame = detector(model).infer(object())
    stamped = frame.with_frame_metadata(frame_index=12, timestamp_seconds=1.5)
    assert (stamped.frame_index, stamped.timestamp_seconds) == (12, 1.5)
    assert frame.frame_index is None


# --- static guarantees ---------------------------------------------------


def test_module_never_falls_back_to_book_or_stock_classes() -> None:
    code = code_text(MODULE_PATH)
    lowered = code.lower()
    # No stock/COCO checkpoint or class list is referenced by executable code.
    for forbidden in ("coco", "yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "coco.yaml"):
        assert forbidden not in lowered
    # 'book' exists in code ONLY inside the prohibited-prompt set.
    assert PROHIBITED_PROMPT_TERMS >= {"book", "notebook", "folder", "phone"}
    assert '"book"' not in code.replace(str(sorted(PROHIBITED_PROMPT_TERMS)), "")
    assert "paper" == CANONICAL_PAPER_CLASS


def test_no_task_3d_or_event_imports() -> None:
    source = code_text(MODULE_PATH)
    for forbidden in (
        "handoff_temporal",
        "exchange_temporal",
        "event_publisher",
        "notification",
        "supabase",
    ):
        assert forbidden not in source
    lowered = source.lower()
    for word in ("cheating", "handoff", "transfer"):
        assert word not in lowered


def test_no_module_level_side_effects() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    calls = [
        node for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert calls == []


def test_no_production_runtime_import_of_paper_detector() -> None:
    app_dir = AI_DIR.parent
    for path in app_dir.rglob("*.py"):
        if "paper" in path.name or "benchmark" in path.parts:
            continue
        source = code_text(path)
        assert "open_vocab_paper_detector" not in source
        assert "paper_evidence" not in source


def test_task1_and_pose_modules_untouched_by_paper_work() -> None:
    for name in ("detector.py", "tracker.py", "phone_rule_engine.py", "pose_provider.py"):
        source = code_text(AI_DIR / name).lower()
        assert "paper" not in source
