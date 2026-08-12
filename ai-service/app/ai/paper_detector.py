"""Paper-evidence provider (dormant: NO runtime integration in Task 3E).

Layering
--------
:class:`PaperDetector` is the ONLY interface future behaviour code may depend
on. Ultralytics objects never leave this module.

Hard truthfulness rules
-----------------------
* There is NO stock-model fallback of any kind. No COCO weights, no
  ``yolov8n.pt`` default, no ``book`` → ``paper`` mapping, no class-index
  guessing. Paper is claimed only when a real paper-specific checkpoint that
  exposes the canonical ``paper`` class produced the detection.
* Custom weights must be supplied explicitly; nothing is downloaded or loaded
  at import time.
* Loaded weights are schema-validated: if the checkpoint's class names do not
  contain the configured canonical paper class, the provider is
  :attr:`PaperEvidenceStatus.MODEL_SCHEMA_MISMATCH`. A checkpoint is never
  trusted merely because it loaded.
* Classes other than the configured paper class are ignored (they are not
  paper); the frame can still be ``OK``.
* Whole-frame strictness: if any *paper-class* row is unusable (reversed box,
  out-of-range coordinates, invalid confidence), the frame is
  ``MALFORMED_RESULT`` — never a clean ``OK`` with silently dropped rows.
* Failure reasons are built from stage + exception CLASS name only: raw
  exception text may embed private model paths or credential-bearing stream
  URLs. Same discipline as ``PoseProvider``.
* Evidence only: nothing here infers transfer, handoff, exchange or cheating,
  and nothing from the Task 3D temporal layer is imported.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, Callable, Optional, Protocol, Sequence

from ..domain.geometry import BBox
from ..domain.paper_evidence import (
    CANONICAL_PAPER_CLASS,
    FORBIDDEN_PAPER_ALIASES,
    PaperDetection,
    PaperDetectorContractError,
    PaperEvidenceFrame,
    PaperEvidenceStatus,
)
from .crop_transform import CropTransform, CropTransformError

logger = logging.getLogger(__name__)

#: Floating-point drift tolerance snapped at the parser boundary only.
BOUNDS_TOLERANCE = 1e-9


def _safe_reason(stage: str, error: BaseException) -> str:
    """Reason built ONLY from the stage and the exception class name."""
    return f"{stage} ({type(error).__name__})"


class PaperDetectorConfigError(ValueError):
    """Raised for provider configuration/programming errors (never degraded)."""


class PaperDetector(Protocol):
    """Narrow paper capability: one frame in, one immutable evidence frame out."""

    @property
    def available(self) -> bool:
        """False once the provider is known to be unusable."""

    def infer(self, frame: Any, crop: Optional[CropTransform] = None) -> PaperEvidenceFrame:
        """Never raises for ordinary model/runtime failure; returns a status."""


class _Malformed(Exception):
    """Internal signal: this frame is unusable."""

    def __init__(
        self,
        reason: str,
        status: PaperEvidenceStatus = PaperEvidenceStatus.MALFORMED_RESULT,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


def _is_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _unit_value(value: Any) -> Optional[float]:
    """Finite 0..1 value with epsilon drift snapped; ``None`` when unusable."""
    if not _is_finite(value):
        return None
    number = float(value)
    if number < 0.0:
        return 0.0 if number >= -BOUNDS_TOLERANCE else None
    if number > 1.0:
        return 1.0 if number <= 1.0 + BOUNDS_TOLERANCE else None
    return number


def _rows(source: Any) -> Optional[list]:
    """Converts a tensor/array/sequence into plain nested Python lists."""
    if source is None:
        return None
    if hasattr(source, "tolist"):
        try:
            source = source.tolist()
        except Exception:  # pragma: no cover - defensive
            return None
    if isinstance(source, (str, bytes)):
        return None
    if isinstance(source, Sequence):
        return list(source)
    try:
        return list(source)
    except TypeError:
        return None


def _normalized_bbox(row: Any) -> Optional[BBox]:
    """Strict xyxyn box: ``x1 < x2`` and ``y1 < y2``, never repaired."""
    values = _rows(row)
    if values is None or len(values) < 4:
        return None
    coords = [_unit_value(value) for value in values[:4]]
    if any(coord is None for coord in coords):
        return None
    x1, y1, x2, y2 = coords  # type: ignore[misc]
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1, y1, x2 - x1, y2 - y1)


def model_class_names(model: Any) -> Optional[dict[int, str]]:
    """Reads the checkpoint's class-name mapping, or ``None`` when unreadable."""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        mapping: dict[int, str] = {}
        for key, value in names.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                return None
            if not isinstance(value, str):
                return None
            mapping[index] = value
        return mapping
    if isinstance(names, (list, tuple)):
        if not all(isinstance(value, str) for value in names):
            return None
        return {index: value for index, value in enumerate(names)}
    return None


def paper_class_index(names: dict[int, str], paper_class: str) -> Optional[int]:
    """Index of the canonical paper class. No index-0 guessing, no aliases."""
    wanted = paper_class.strip().lower()
    for index, name in names.items():
        if not isinstance(name, str):
            continue
        candidate = name.strip().lower()
        if candidate == wanted:
            return index
    return None


def parse_paper_result(
    result: Any,
    paper_class_id: int,
    model_name: Optional[str] = None,
    crop: Optional[CropTransform] = None,
    crop_source: Optional[str] = None,
) -> PaperEvidenceFrame:
    """Pure parser for ONE Ultralytics detection ``Results`` object."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return PaperEvidenceFrame.failure(
            PaperEvidenceStatus.MALFORMED_RESULT, "result carries no boxes", model_name
        )
    try:
        detections = _parse_detections(
            boxes, paper_class_id, model_name, crop, crop_source
        )
    except _Malformed as error:
        return PaperEvidenceFrame.failure(error.status, error.reason, model_name)
    return PaperEvidenceFrame(
        status=PaperEvidenceStatus.OK, detections=detections, model_name=model_name
    )


def _parse_detections(
    boxes: Any,
    paper_class_id: int,
    model_name: Optional[str],
    crop: Optional[CropTransform],
    crop_source: Optional[str],
) -> tuple[PaperDetection, ...]:
    box_rows = _rows(getattr(boxes, "xyxyn", None))
    if box_rows is None:
        raise _Malformed("missing normalized boxes array")
    if not box_rows:
        return ()

    confidences = _rows(getattr(boxes, "conf", None))
    class_rows = _rows(getattr(boxes, "cls", None))
    if confidences is None or class_rows is None:
        raise _Malformed("paper detections require both confidence and class channels")
    if len(confidences) != len(box_rows) or len(class_rows) != len(box_rows):
        raise _Malformed(
            f"boxes/conf/cls length mismatch ({len(box_rows)}/{len(confidences)}/"
            f"{len(class_rows)})"
        )

    detections: list[PaperDetection] = []
    for index, row in enumerate(box_rows):
        raw_class = class_rows[index]
        if not _is_finite(raw_class):
            raise _Malformed(f"unreadable class id at detection {index}")
        class_id = int(float(raw_class))
        if float(raw_class) != float(class_id):
            raise _Malformed(f"non-integral class id at detection {index}")
        if class_id != paper_class_id:
            # A different class is simply not paper evidence.
            continue

        bbox = _normalized_bbox(row)
        if bbox is None:
            raise _Malformed(f"unusable paper bbox at detection {index}")
        confidence = _unit_value(confidences[index])
        if confidence is None:
            raise _Malformed(f"invalid confidence at detection {index}")

        if crop is not None and not crop.is_identity:
            try:
                bbox = crop.to_full_frame(bbox)
            except CropTransformError as error:
                raise _Malformed(f"crop mapping rejected detection {index}: {error}") from error
        try:
            detections.append(
                PaperDetection(
                    bbox=bbox,
                    confidence=confidence,
                    class_name=CANONICAL_PAPER_CLASS,
                    model_name=model_name,
                    crop_source=crop_source,
                )
            )
        except PaperDetectorContractError as error:
            raise _Malformed(f"paper detection rejected: {error}") from error
    return tuple(detections)


class UltralyticsPaperDetector:
    """Paper detector backed by a FUTURE custom-trained Ultralytics checkpoint.

    * ``weights_path`` is mandatory and explicit; there is no default model and
      nothing is downloaded. Missing/unloadable weights →
      :attr:`PaperEvidenceStatus.MODEL_UNAVAILABLE`.
    * After a successful load, class names are validated against
      ``paper_class`` → :attr:`PaperEvidenceStatus.MODEL_SCHEMA_MISMATCH` when
      absent. Both failures are STICKY for the instance.
    * One provider owns ONE model guarded by its OWN lock. It never touches the
      object detector's or pose provider's model/lock, and never calls
      ``model.track``.
    * Reasons expose only the weights FILE NAME and exception class names.
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        imgsz: int = 640,
        confidence: float = 0.25,
        paper_class: str = CANONICAL_PAPER_CLASS,
        model_factory: Optional[Callable[[str], Any]] = None,
        require_existing_weights: bool = True,
    ) -> None:
        if not isinstance(weights_path, str) or not weights_path.strip():
            raise PaperDetectorConfigError("weights_path must be a non-empty string")
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
            raise PaperDetectorConfigError(f"imgsz must be a positive integer, got {imgsz!r}")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise PaperDetectorConfigError("confidence must be a number in 0..1")
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise PaperDetectorConfigError(
                f"confidence must be finite within 0..1, got {confidence!r}"
            )
        if not isinstance(device, str) or not device.strip():
            raise PaperDetectorConfigError("device must be a non-empty string")
        if not isinstance(paper_class, str) or not paper_class.strip():
            raise PaperDetectorConfigError("paper_class must be a non-empty string")
        if paper_class.strip().lower() in FORBIDDEN_PAPER_ALIASES:
            raise PaperDetectorConfigError(
                f"{paper_class!r} is never an acceptable paper class"
            )

        self.weights_path = weights_path.strip()
        self.device = device.strip()
        self.imgsz = int(imgsz)
        self.confidence = float(confidence)
        self.paper_class = paper_class.strip()
        self.require_existing_weights = bool(require_existing_weights)
        self._model_factory = model_factory
        self._lock = threading.Lock()
        self._model: Any = None
        self._paper_class_id: Optional[int] = None
        self._failure_status: Optional[PaperEvidenceStatus] = None
        self._failure_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self._failure_status is None

    @property
    def model_label(self) -> str:
        """Weights FILE NAME only: never a full, possibly sensitive path."""
        return os.path.basename(self.weights_path) or self.weights_path

    def _default_factory(self, weights_path: str) -> Any:
        from ultralytics import YOLO  # imported lazily: heavy dependency

        return YOLO(weights_path)

    def initialize(self) -> bool:
        """Explicitly loads and schema-validates the checkpoint."""
        with self._lock:
            return self._ensure_model() is not None

    def _fail(self, status: PaperEvidenceStatus, reason: str) -> None:
        self._failure_status = status
        self._failure_reason = reason
        self._model = None
        self._paper_class_id = None

    def _ensure_model(self) -> Any:
        """Loads + validates the model once. Caller MUST hold ``self._lock``."""
        if self._model is not None or self._failure_status is not None:
            return self._model

        if self.require_existing_weights and not os.path.isfile(self.weights_path):
            self._fail(
                PaperEvidenceStatus.MODEL_UNAVAILABLE,
                f"custom paper weights missing: {self.model_label}",
            )
            logger.warning("Paper weights %s not found; no stock fallback", self.model_label)
            return None

        factory = self._model_factory or self._default_factory
        try:
            model = factory(self.weights_path)
        except Exception as error:  # noqa: BLE001 - degradation, not a crash
            self._fail(
                PaperEvidenceStatus.MODEL_UNAVAILABLE,
                _safe_reason("paper model unavailable", error),
            )
            # Only the exception CLASS is logged: messages may embed paths.
            logger.warning(
                "Paper model %s unavailable (%s)", self.model_label, type(error).__name__
            )
            return None

        names = model_class_names(model)
        if names is None:
            self._fail(
                PaperEvidenceStatus.MODEL_SCHEMA_MISMATCH,
                f"checkpoint {self.model_label} exposes no readable class names",
            )
            return None
        index = paper_class_index(names, self.paper_class)
        if index is None:
            self._fail(
                PaperEvidenceStatus.MODEL_SCHEMA_MISMATCH,
                f"checkpoint {self.model_label} does not expose the required "
                f"{self.paper_class!r} class",
            )
            logger.warning(
                "Paper checkpoint %s lacks the required %r class; refusing to guess",
                self.model_label,
                self.paper_class,
            )
            return None

        self._model = model
        self._paper_class_id = index
        return self._model

    def infer(self, frame: Any, crop: Optional[CropTransform] = None) -> PaperEvidenceFrame:
        """Runs paper detection on one full frame or one explicit crop."""
        if crop is not None and not isinstance(crop, CropTransform):
            raise PaperDetectorConfigError("crop must be a CropTransform when supplied")
        label = self.model_label
        crop_source = None if crop is None or crop.is_identity else "explicit_crop"

        with self._lock:
            model = self._ensure_model()
            if model is None:
                return PaperEvidenceFrame.failure(
                    self._failure_status or PaperEvidenceStatus.MODEL_UNAVAILABLE,
                    self._failure_reason or "paper model could not be loaded",
                    label,
                )
            paper_class_id = self._paper_class_id
            assert paper_class_id is not None  # guaranteed by schema validation
            try:
                results = model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    device=self.device,
                    conf=self.confidence,
                    verbose=False,
                )
            except Exception as error:  # noqa: BLE001 - degradation, not a crash
                # Never log the raw message: it can carry credentials/paths.
                logger.warning(
                    "Paper inference failed on %s (%s)", label, type(error).__name__
                )
                return PaperEvidenceFrame.failure(
                    PaperEvidenceStatus.INFERENCE_FAILED,
                    _safe_reason("paper inference failed", error),
                    label,
                )

        rows = _rows(results)
        if rows is None:
            return PaperEvidenceFrame.failure(
                PaperEvidenceStatus.MALFORMED_RESULT,
                "predict() returned no readable results",
                label,
            )
        if len(rows) != 1:
            return PaperEvidenceFrame.failure(
                PaperEvidenceStatus.MALFORMED_RESULT,
                f"expected exactly 1 result for 1 frame, got {len(rows)}",
                label,
            )
        return parse_paper_result(rows[0], paper_class_id, label, crop, crop_source)
