"""Open-vocabulary paper evidence provider (offline/dormant — Task 3E-B).

Why open-vocabulary
-------------------
No trustworthy loose-paper checkpoint exists yet, and the stock COCO class list
has no paper class. Instead of shipping a lie (``book`` is NOT paper) we probe an
open-vocabulary detector (Ultralytics YOLO-World / YOLOE, whichever checkpoint is
supplied) with EXPLICIT loose-paper text prompts and measure whether it can see
paper at the real camera scale. Custom training (Task 3E, see
``docs/paper-detector-training.md``) remains Plan B.

Hard rules enforced here
------------------------
* Prompts are ALWAYS explicitly supplied. There is no hidden prompt default in
  behavioural code, no synonym expansion and no silent fallback prompt.
* ``book``, ``notebook``, ``folder``, ``phone``, ``pen``, ``hand``, ``desk`` and
  "generic object" style prompts are REJECTED as paper prompts.
* Every detection preserves the EXACT prompt that fired (``raw_prompt``) next to
  the canonical ``paper`` semantic; the firing prompt is never hidden.
* Whole-frame strictness: any unusable row makes the frame ``MALFORMED_RESULT``.
  A partially trusted frame is never produced.
* Statuses are explicit: ``MODEL_UNAVAILABLE``, ``PROMPT_CONFIGURATION_INVALID``,
  ``INFERENCE_FAILED``, ``MALFORMED_RESULT``. A failure is never downgraded into
  valid empty evidence.
* Failure reasons are ``stage (ExceptionClass)`` only — never raw exception text,
  model paths, stream URLs or credentials (same discipline as ``PoseProvider``).
* Evidence only: nothing here means transfer, exchange, handoff or cheating, and
  nothing from the Task 3D temporal layer is imported. No runtime integration,
  no scheduling, no GPU cost on the live path.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol

from ..domain.paper_evidence import (
    CANONICAL_PAPER_CLASS,
    FORBIDDEN_PAPER_ALIASES,
    PaperDetection,
    PaperDetectorContractError,
    PaperEvidenceFrame,
    PaperEvidenceStatus,
)
from .crop_transform import CropTransform, CropTransformError
from .paper_detector import (  # shared strict parsing helpers
    _is_finite,
    _normalized_bbox,
    _rows,
    _safe_reason,
    _unit_value,
)

logger = logging.getLogger(__name__)

#: Default BACKEND LABEL only (a diagnostic name, never a model path).
DEFAULT_BACKEND_LABEL = "ultralytics-open-vocab"

#: Candidate loose-paper prompt terms for offline experiments. These are a menu
#: for the evaluation CLI and tests — NOT an implicit default: the provider
#: always requires an explicit prompt configuration.
PAPER_PROMPT_CANDIDATES: tuple[str, ...] = (
    "paper",
    "sheet of paper",
    "exam paper",
    "paper slip",
    "small paper slip",
    "folded paper",
)

#: Prompt substrings that would make the evidence untruthful.
PROHIBITED_PROMPT_TERMS: frozenset[str] = frozenset(
    {
        "book",
        "notebook",
        "folder",
        "phone",
        "cell phone",
        "pen",
        "pencil",
        "hand",
        "desk",
        "object",
        "thing",
        "item",
    }
)


class PaperPromptConfigError(ValueError):
    """Raised for an invalid/prohibited open-vocabulary prompt configuration."""


class OpenVocabConfigError(ValueError):
    """Raised for provider configuration/programming errors (never degraded)."""


@dataclass(frozen=True, slots=True)
class PaperPromptConfig:
    """An explicit, validated, ordered open-vocabulary paper prompt list.

    Prompt ORDER is the class-index order handed to the backend, so index → raw
    prompt resolution is exact and never guessed.
    """

    prompts: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.prompts, (str, bytes)) or not isinstance(
            self.prompts, (tuple, list)
        ):
            raise PaperPromptConfigError("prompts must be a non-empty list/tuple of strings")
        cleaned: list[str] = []
        seen: set[str] = set()
        for prompt in self.prompts:
            if not isinstance(prompt, str) or not prompt.strip():
                raise PaperPromptConfigError(
                    "every prompt must be a non-empty, non-whitespace string"
                )
            label = " ".join(prompt.split())
            key = label.lower()
            if key in seen:
                raise PaperPromptConfigError(f"duplicate prompt: {label!r}")
            if key in FORBIDDEN_PAPER_ALIASES or key in PROHIBITED_PROMPT_TERMS:
                raise PaperPromptConfigError(
                    f"{label!r} is never a paper prompt (it is a different object)"
                )
            for term in PROHIBITED_PROMPT_TERMS:
                if term in key.split() and "paper" not in key:
                    raise PaperPromptConfigError(
                        f"{label!r} does not describe loose paper"
                    )
            if "paper" not in key and "sheet" not in key and "slip" not in key:
                raise PaperPromptConfigError(
                    f"{label!r} is not a loose-paper concept; prompts must be explicit"
                )
            cleaned.append(label)
            seen.add(key)
        if not cleaned:
            raise PaperPromptConfigError("prompts must contain at least one prompt")
        object.__setattr__(self, "prompts", tuple(cleaned))

    @classmethod
    def from_iterable(cls, prompts: Iterable[str]) -> "PaperPromptConfig":
        return cls(tuple(prompts))

    def prompt_for_index(self, index: int) -> Optional[str]:
        """Raw prompt for a backend class index, or ``None`` when out of range."""
        if isinstance(index, bool) or not isinstance(index, int):
            return None
        if 0 <= index < len(self.prompts):
            return self.prompts[index]
        return None

    @property
    def label(self) -> str:
        """Stable, human-readable configuration label for reports."""
        return " | ".join(self.prompts)

    def to_list(self) -> list[str]:
        return list(self.prompts)


class OpenVocabPaperDetectorProtocol(Protocol):
    """Narrow capability: one frame in, one immutable paper evidence frame out."""

    @property
    def available(self) -> bool: ...

    def infer(self, frame: Any, crop: Optional[CropTransform] = None) -> PaperEvidenceFrame: ...


class OpenVocabPaperDetector:
    """Open-vocabulary paper detector over an explicit prompt configuration.

    Nothing is loaded at import: weights load lazily on first use, and prompts
    are applied to the backend immediately after loading. A load failure or a
    prompt-application failure is STICKY for the instance and reported with its
    own status.
    """

    def __init__(
        self,
        weights_path: str,
        prompts: PaperPromptConfig,
        device: str = "cpu",
        imgsz: int = 960,
        confidence: float = 0.25,
        backend_label: str = DEFAULT_BACKEND_LABEL,
        model_factory: Optional[Callable[[str], Any]] = None,
        require_existing_weights: bool = True,
    ) -> None:
        if not isinstance(weights_path, str) or not weights_path.strip():
            raise OpenVocabConfigError("weights_path must be a non-empty string")
        if not isinstance(prompts, PaperPromptConfig):
            raise OpenVocabConfigError(
                "prompts must be an explicitly validated PaperPromptConfig"
            )
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
            raise OpenVocabConfigError(f"imgsz must be a positive integer, got {imgsz!r}")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise OpenVocabConfigError("confidence must be a number in 0..1")
        if not _is_finite(confidence) or not 0.0 <= float(confidence) <= 1.0:
            raise OpenVocabConfigError(
                f"confidence must be finite within 0..1, got {confidence!r}"
            )
        if not isinstance(device, str) or not device.strip():
            raise OpenVocabConfigError("device must be a non-empty string")
        if not isinstance(backend_label, str) or not backend_label.strip():
            raise OpenVocabConfigError("backend_label must be a non-empty string")

        self.weights_path = weights_path.strip()
        self.prompt_config = prompts
        self.device = device.strip()
        self.imgsz = int(imgsz)
        self.confidence = float(confidence)
        self.backend_label = backend_label.strip()
        self.require_existing_weights = bool(require_existing_weights)
        self._model_factory = model_factory
        self._lock = threading.Lock()
        self._model: Any = None
        self._failure_status: Optional[PaperEvidenceStatus] = None
        self._failure_reason: Optional[str] = None

    # -- introspection ----------------------------------------------------
    @property
    def available(self) -> bool:
        return self._failure_status is None

    @property
    def model_label(self) -> str:
        """Weights FILE NAME only: never a full, possibly sensitive path."""
        return os.path.basename(self.weights_path) or self.weights_path

    @property
    def prompts(self) -> tuple[str, ...]:
        return self.prompt_config.prompts

    # -- model lifecycle --------------------------------------------------
    def _default_factory(self, weights_path: str) -> Any:
        from ultralytics import YOLO  # imported lazily: heavy dependency

        return YOLO(weights_path)

    def initialize(self) -> bool:
        with self._lock:
            return self._ensure_model() is not None

    def _fail(self, status: PaperEvidenceStatus, reason: str) -> None:
        self._failure_status = status
        self._failure_reason = reason
        self._model = None

    def _ensure_model(self) -> Any:
        """Loads weights and applies prompts once. Caller MUST hold the lock."""
        if self._model is not None or self._failure_status is not None:
            return self._model

        if self.require_existing_weights and not os.path.isfile(self.weights_path):
            self._fail(
                PaperEvidenceStatus.MODEL_UNAVAILABLE,
                f"open-vocabulary weights missing: {self.model_label}",
            )
            logger.warning(
                "Open-vocabulary weights %s not found; no stock fallback", self.model_label
            )
            return None

        factory = self._model_factory or self._default_factory
        try:
            model = factory(self.weights_path)
        except Exception as error:  # noqa: BLE001 - degradation, not a crash
            self._fail(
                PaperEvidenceStatus.MODEL_UNAVAILABLE,
                _safe_reason("open-vocabulary model unavailable", error),
            )
            logger.warning(
                "Open-vocabulary model %s unavailable (%s)",
                self.model_label,
                type(error).__name__,
            )
            return None

        set_classes = getattr(model, "set_classes", None)
        if not callable(set_classes):
            self._fail(
                PaperEvidenceStatus.PROMPT_CONFIGURATION_INVALID,
                f"checkpoint {self.model_label} does not accept text prompts",
            )
            logger.warning(
                "Checkpoint %s is not open-vocabulary; refusing to guess classes",
                self.model_label,
            )
            return None
        try:
            set_classes(self.prompt_config.to_list())
        except Exception as error:  # noqa: BLE001 - degradation, not a crash
            self._fail(
                PaperEvidenceStatus.PROMPT_CONFIGURATION_INVALID,
                _safe_reason("paper prompt configuration failed", error),
            )
            logger.warning(
                "Applying paper prompts to %s failed (%s)",
                self.model_label,
                type(error).__name__,
            )
            return None

        self._model = model
        return self._model

    # -- inference --------------------------------------------------------
    def infer(self, frame: Any, crop: Optional[CropTransform] = None) -> PaperEvidenceFrame:
        """Full-frame (or explicitly supplied crop) paper inference on one frame.

        Full-frame evaluation is the phase-one contract; ``crop`` exists only so
        the OFFLINE evaluator can run an explicit, separately-labelled crop
        experiment. No runtime crop scheduling exists.
        """
        if crop is not None and not isinstance(crop, CropTransform):
            raise OpenVocabConfigError("crop must be a CropTransform when supplied")
        label = self.model_label
        crop_source = None if crop is None or crop.is_identity else "explicit_crop"

        with self._lock:
            model = self._ensure_model()
            if model is None:
                return PaperEvidenceFrame.failure(
                    self._failure_status or PaperEvidenceStatus.MODEL_UNAVAILABLE,
                    self._failure_reason or "open-vocabulary model could not be loaded",
                    label,
                    self.backend_label,
                )
            try:
                results = model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    device=self.device,
                    conf=self.confidence,
                    verbose=False,
                )
            except Exception as error:  # noqa: BLE001 - degradation, not a crash
                logger.warning(
                    "Paper inference failed on %s (%s)", label, type(error).__name__
                )
                return PaperEvidenceFrame.failure(
                    PaperEvidenceStatus.INFERENCE_FAILED,
                    _safe_reason("paper inference failed", error),
                    label,
                    self.backend_label,
                )

        rows = _rows(results)
        if rows is None:
            return PaperEvidenceFrame.failure(
                PaperEvidenceStatus.MALFORMED_RESULT,
                "predict() returned no readable results",
                label,
                self.backend_label,
            )
        if len(rows) != 1:
            return PaperEvidenceFrame.failure(
                PaperEvidenceStatus.MALFORMED_RESULT,
                f"expected exactly 1 result for 1 frame, got {len(rows)}",
                label,
                self.backend_label,
            )
        return parse_open_vocab_result(
            rows[0],
            self.prompt_config,
            model_name=label,
            backend=self.backend_label,
            crop=crop,
            crop_source=crop_source,
        )


def parse_open_vocab_result(
    result: Any,
    prompt_config: PaperPromptConfig,
    model_name: Optional[str] = None,
    backend: Optional[str] = None,
    crop: Optional[CropTransform] = None,
    crop_source: Optional[str] = None,
) -> PaperEvidenceFrame:
    """Pure, deterministic parser for ONE open-vocabulary ``Results`` object."""
    if not isinstance(prompt_config, PaperPromptConfig):
        raise OpenVocabConfigError("prompt_config must be a PaperPromptConfig")
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return PaperEvidenceFrame.failure(
            PaperEvidenceStatus.MALFORMED_RESULT, "result carries no boxes", model_name, backend
        )

    box_rows = _rows(getattr(boxes, "xyxyn", None))
    if box_rows is None:
        return PaperEvidenceFrame.failure(
            PaperEvidenceStatus.MALFORMED_RESULT,
            "missing normalized boxes array",
            model_name,
            backend,
        )
    if not box_rows:
        return PaperEvidenceFrame(
            status=PaperEvidenceStatus.OK,
            detections=(),
            model_name=model_name,
            backend=backend,
        )

    confidences = _rows(getattr(boxes, "conf", None))
    class_rows = _rows(getattr(boxes, "cls", None))
    if confidences is None or class_rows is None:
        return PaperEvidenceFrame.failure(
            PaperEvidenceStatus.MALFORMED_RESULT,
            "paper detections require both confidence and class channels",
            model_name,
            backend,
        )
    if len(confidences) != len(box_rows) or len(class_rows) != len(box_rows):
        return PaperEvidenceFrame.failure(
            PaperEvidenceStatus.MALFORMED_RESULT,
            f"boxes/conf/cls length mismatch ({len(box_rows)}/{len(confidences)}/"
            f"{len(class_rows)})",
            model_name,
            backend,
        )

    detections: list[PaperDetection] = []
    for index in range(len(box_rows)):
        raw_class = class_rows[index]
        if not _is_finite(raw_class) or float(raw_class) != float(int(float(raw_class))):
            return PaperEvidenceFrame.failure(
                PaperEvidenceStatus.MALFORMED_RESULT,
                f"unusable class id at detection {index}",
                model_name,
                backend,
            )
        raw_prompt = prompt_config.prompt_for_index(int(float(raw_class)))
        if raw_prompt is None:
            # An index outside the configured prompt list is NOT paper evidence
            # and is never guessed into one.
            return PaperEvidenceFrame.failure(
                PaperEvidenceStatus.MALFORMED_RESULT,
                f"class id outside configured prompt list at detection {index}",
                model_name,
                backend,
            )

        bbox = _normalized_bbox(box_rows[index])
        if bbox is None:
            return PaperEvidenceFrame.failure(
                PaperEvidenceStatus.MALFORMED_RESULT,
                f"unusable paper bbox at detection {index}",
                model_name,
                backend,
            )
        confidence = _unit_value(confidences[index])
        if confidence is None:
            return PaperEvidenceFrame.failure(
                PaperEvidenceStatus.MALFORMED_RESULT,
                f"invalid confidence at detection {index}",
                model_name,
                backend,
            )
        if crop is not None and not crop.is_identity:
            try:
                bbox = crop.to_full_frame(bbox)
            except CropTransformError:
                return PaperEvidenceFrame.failure(
                    PaperEvidenceStatus.MALFORMED_RESULT,
                    f"crop mapping rejected detection {index}",
                    model_name,
                    backend,
                )
        try:
            detections.append(
                PaperDetection(
                    bbox=bbox,
                    confidence=confidence,
                    class_name=CANONICAL_PAPER_CLASS,
                    raw_prompt=raw_prompt,
                    model_name=model_name,
                    backend=backend,
                    crop_source=crop_source,
                )
            )
        except PaperDetectorContractError:
            return PaperEvidenceFrame.failure(
                PaperEvidenceStatus.MALFORMED_RESULT,
                f"paper detection rejected at index {index}",
                model_name,
                backend,
            )

    return PaperEvidenceFrame(
        status=PaperEvidenceStatus.OK,
        detections=tuple(detections),
        model_name=model_name,
        backend=backend,
    )
