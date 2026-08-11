"""Optional pose-estimation provider (dormant: no runtime integration).

Layering
--------
:class:`PoseProvider` is the ONLY interface future behaviour code may depend
on. Ultralytics ``Results``/``Keypoints`` objects never leave this module, so a
future library upgrade is contained here.

Pinned-API contract (ultralytics==8.3.55)
-----------------------------------------
``model.predict(...)`` returns a list of ``Results``. For a pose model each
result exposes ``boxes`` (with ``xyxyn`` normalized boxes and ``conf``) and
``keypoints`` (with ``xyn`` normalized keypoints and ``conf``). In that pinned
release ``Keypoints`` masks low-confidence keypoints by setting their
coordinates to ``0.0`` when confidence < 0.5, therefore availability is decided
from the confidence channel and ``(0.0, 0.0)`` is never assumed to be an
observed joint.

Explicitly NOT done here: tracking (``model.track`` is never called — the
object detector owns person identity), pose-to-person matching, cadence
scheduling, GPU budgeting, region reasoning and any behavioural feature.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Callable, Optional, Protocol, Sequence

from ..domain.geometry import BBox
from ..domain.pose import (
    COCO_17_KEYPOINTS,
    COCO_17_KEYPOINT_COUNT,
    PoseFrameResult,
    PoseInstance,
    PoseKeypoint,
    PoseStatus,
)

logger = logging.getLogger(__name__)

#: Confidence floor below which pinned Ultralytics zeroes keypoint coordinates.
KEYPOINT_VISIBILITY_FLOOR = 0.5

#: Numeric tolerance for normalized bounds checks.
BOUNDS_TOLERANCE = 1e-9


class PoseProvider(Protocol):
    """Narrow pose capability: one frame in, one immutable pose result out."""

    @property
    def available(self) -> bool:
        """False once the provider is known to be unusable (e.g. load failed)."""

    def infer(self, frame: Any) -> PoseFrameResult:
        """Never raises for ordinary model/runtime failure; returns a status."""


def _is_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _valid_confidence(value: Any) -> Optional[float]:
    """Finite confidence within 0..1, else ``None``. Never clamps."""
    if not _is_finite(value):
        return None
    number = float(value)
    if number < -BOUNDS_TOLERANCE or number > 1.0 + BOUNDS_TOLERANCE:
        return None
    return number


def _valid_normalized(value: Any) -> Optional[float]:
    if not _is_finite(value):
        return None
    number = float(value)
    if number < -BOUNDS_TOLERANCE or number > 1.0 + BOUNDS_TOLERANCE:
        return None
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


def _pair(value: Any) -> Optional[tuple[float, float]]:
    row = _rows(value)
    if row is None or len(row) < 2:
        return None
    x = _valid_normalized(row[0])
    y = _valid_normalized(row[1])
    if x is None or y is None:
        return None
    return (x, y)


def _normalized_bbox(row: Any) -> Optional[BBox]:
    values = _rows(row)
    if values is None or len(values) < 4:
        return None
    coords = [_valid_normalized(value) for value in values[:4]]
    if any(coord is None for coord in coords):
        return None
    x1, y1, x2, y2 = coords  # type: ignore[misc]
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    width = right - left
    height = bottom - top
    if width <= 0.0 or height <= 0.0:
        return None
    return BBox(left, top, width, height)


def _build_keypoints(
    points: list, confidences: Optional[list]
) -> Optional[tuple[PoseKeypoint, ...]]:
    """Builds a COCO-17 keypoint tuple, or ``None`` for an unsupported schema."""
    if len(points) != COCO_17_KEYPOINT_COUNT:
        return None
    if confidences is not None and len(confidences) != COCO_17_KEYPOINT_COUNT:
        return None

    keypoints: list[PoseKeypoint] = []
    for index, name in enumerate(COCO_17_KEYPOINTS):
        confidence = (
            _valid_confidence(confidences[index]) if confidences is not None else None
        )
        coordinates = _pair(points[index])
        # Availability comes from the confidence contract, never coordinates.
        if confidences is not None and (
            confidence is None or confidence < KEYPOINT_VISIBILITY_FLOOR
        ):
            keypoints.append(PoseKeypoint.unavailable(name, index, confidence))
            continue
        if coordinates is None:
            keypoints.append(PoseKeypoint.unavailable(name, index, confidence))
            continue
        keypoints.append(
            PoseKeypoint(
                name=name,
                index=index,
                available=True,
                x=coordinates[0],
                y=coordinates[1],
                confidence=confidence,
            )
        )
    return tuple(keypoints)


def parse_pose_result(result: Any, model_name: Optional[str] = None) -> PoseFrameResult:
    """Pure parser for ONE pinned-Ultralytics pose ``Results`` object."""
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return PoseFrameResult.failure(
            PoseStatus.KEYPOINTS_ABSENT, "result carries no keypoints", model_name
        )
    if boxes is None:
        return PoseFrameResult.failure(
            PoseStatus.MALFORMED_RESULT, "result carries no boxes", model_name
        )

    box_rows = _rows(getattr(boxes, "xyxyn", None))
    point_rows = _rows(getattr(keypoints, "xyn", None))
    if box_rows is None or point_rows is None:
        return PoseFrameResult.failure(
            PoseStatus.MALFORMED_RESULT,
            "missing normalized boxes/keypoints arrays",
            model_name,
        )
    if len(box_rows) != len(point_rows):
        # No arbitrary zip truncation: the whole frame is treated as malformed.
        return PoseFrameResult.failure(
            PoseStatus.MALFORMED_RESULT,
            f"boxes/keypoints length mismatch ({len(box_rows)} vs {len(point_rows)})",
            model_name,
        )
    if not box_rows:
        return PoseFrameResult(status=PoseStatus.OK, instances=(), model_name=model_name)

    box_confidences = _rows(getattr(boxes, "conf", None))
    keypoint_confidences = _rows(getattr(keypoints, "conf", None))
    if keypoint_confidences is not None and len(keypoint_confidences) != len(point_rows):
        return PoseFrameResult.failure(
            PoseStatus.MALFORMED_RESULT,
            "keypoint confidence length mismatch",
            model_name,
        )

    instances: list[PoseInstance] = []
    for index, points in enumerate(point_rows):
        point_list = _rows(points)
        if point_list is None:
            return PoseFrameResult.failure(
                PoseStatus.MALFORMED_RESULT, "unreadable keypoint row", model_name
            )
        confidences = (
            _rows(keypoint_confidences[index])
            if keypoint_confidences is not None
            else None
        )
        if keypoint_confidences is not None and confidences is None:
            return PoseFrameResult.failure(
                PoseStatus.MALFORMED_RESULT,
                "unreadable keypoint confidence row",
                model_name,
            )
        built = _build_keypoints(point_list, confidences)
        if built is None:
            return PoseFrameResult.failure(
                PoseStatus.UNSUPPORTED_POSE_SCHEMA,
                f"expected {COCO_17_KEYPOINT_COUNT} keypoints, got {len(point_list)}",
                model_name,
            )
        bbox = _normalized_bbox(box_rows[index])
        if bbox is None:
            # Unusable instance geometry: skip this instance only.
            continue
        confidence = (
            _valid_confidence(box_confidences[index])
            if box_confidences is not None and index < len(box_confidences)
            else None
        )
        instances.append(
            PoseInstance(bbox=bbox, keypoints=built, confidence=confidence)
        )

    return PoseFrameResult(
        status=PoseStatus.OK, instances=tuple(instances), model_name=model_name
    )


class UltralyticsPoseProvider:
    """Pose provider backed by the existing pinned Ultralytics YOLO stack.

    * Weights are NEVER loaded at import time; loading happens lazily on the
      first :meth:`infer` (or via an explicit :meth:`initialize`).
    * ``model_factory`` is injectable so tests need no real weights.
    * One provider owns ONE pose model guarded by its OWN lock; it never
      touches ``YoloDetector``'s lock and never calls ``model.track``.
    * Ordinary failures (load error, inference exception, malformed output) are
      reported as degraded :class:`PoseFrameResult` values. Programming or
      configuration mistakes (e.g. an invalid factory signature) may still
      surface as exceptions.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        imgsz: int = 640,
        confidence: float = 0.25,
        model_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.imgsz = int(imgsz)
        self.confidence = float(confidence)
        self._model_factory = model_factory
        self._lock = threading.Lock()
        self._model: Any = None
        self._load_failed = False
        self._load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return not self._load_failed

    def _default_factory(self, model_name: str) -> Any:
        from ultralytics import YOLO  # imported lazily: heavy dependency

        return YOLO(model_name)

    def initialize(self) -> bool:
        """Explicitly loads the pose model. Returns False when unavailable."""
        with self._lock:
            return self._ensure_model() is not None

    def _ensure_model(self) -> Any:
        """Loads the model once. Caller MUST hold ``self._lock``."""
        if self._model is not None or self._load_failed:
            return self._model
        factory = self._model_factory or self._default_factory
        try:
            self._model = factory(self.model_name)
        except Exception as error:  # noqa: BLE001 - degradation, not a crash
            self._load_failed = True
            self._load_error = str(error)
            logger.warning("Pose model %s unavailable: %s", self.model_name, error)
            return None
        return self._model

    def infer(self, frame: Any) -> PoseFrameResult:
        """Runs untracked pose prediction on one frame; never raises normally."""
        with self._lock:
            model = self._ensure_model()
            if model is None:
                return PoseFrameResult.failure(
                    PoseStatus.MODEL_UNAVAILABLE,
                    self._load_error or "pose model could not be loaded",
                    self.model_name,
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
                logger.warning("Pose inference failed on %s: %s", self.model_name, error)
                return PoseFrameResult.failure(
                    PoseStatus.INFERENCE_FAILED, str(error), self.model_name
                )

        if not results:
            return PoseFrameResult.failure(
                PoseStatus.MALFORMED_RESULT, "empty result list", self.model_name
            )
        return parse_pose_result(results[0], self.model_name)
