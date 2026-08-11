"""Optional pose-estimation provider (dormant: no runtime integration).

Layering
--------
:class:`PoseProvider` is the ONLY interface future behaviour code may depend
on. Ultralytics ``Results``/``Keypoints`` objects never leave this module, so a
future library upgrade is contained here.

Pinned-API contract (ultralytics==8.3.55)
-----------------------------------------
``model.predict(...)`` returns a list of ``Results``. One input frame MUST
produce exactly ONE ``Results`` object; zero or several is malformed and is
never silently reduced to ``results[0]``. For a pose model each result exposes
``boxes`` (with ``xyxyn`` normalized boxes and ``conf``) and ``keypoints``
(with ``xyn`` normalized keypoints and ``conf``).

Truthfulness policies enforced here
-----------------------------------
* Confidence channel required: a ``Kx2`` coordinate-only keypoint array yields
  :attr:`PoseStatus.KEYPOINT_CONFIDENCE_ABSENT` with zero instances. Visibility
  is never inferred from coordinates, so ``(0.0, 0.0)`` and ``(0.4, 0.5)`` are
  equally unusable without confidence.
* Reversed pose boxes (``x2 <= x1`` or ``y2 <= y1``) are malformed, never
  repaired by min/max.
* Normalized bounds: only floating-point epsilon drift (``1e-9``) is snapped at
  this parser boundary; substantive out-of-range values are rejected.
* Array alignment: boxes/keypoints/box-confidence/keypoint-confidence counts
  must match exactly. A supplied-but-invalid confidence is malformed, never
  quietly turned into ``None``.
* Whole-frame strictness: if the model returned pose detections and ANY
  instance is unusable, the frame is ``MALFORMED_RESULT`` rather than a clean
  ``OK`` with silently dropped rows. ``OK`` + empty instances therefore means
  only one thing: the model genuinely detected no people.

Explicitly NOT done here: tracking (``model.track`` is never called — the
object detector owns person identity), pose-to-person matching, cadence
scheduling, GPU budgeting, region reasoning and any behavioural feature.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, Callable, Optional, Protocol, Sequence

from ..domain.geometry import BBox
from ..domain.pose import (
    COCO_17_KEYPOINTS,
    COCO_17_KEYPOINT_COUNT,
    PoseContractError,
    PoseFrameResult,
    PoseInstance,
    PoseKeypoint,
    PoseStatus,
)

logger = logging.getLogger(__name__)

#: Confidence floor below which pinned Ultralytics zeroes keypoint coordinates.
KEYPOINT_VISIBILITY_FLOOR = 0.5

#: Floating-point drift tolerance snapped at the parser boundary only.
BOUNDS_TOLERANCE = 1e-9


def _safe_reason(stage: str, error: BaseException) -> str:
    """Failure reason built ONLY from the stage and the exception class name.

    Raw exception text is deliberately discarded: model-load errors can embed
    private filesystem paths and inference errors can embed credential-bearing
    stream URLs. Neither is ever stored, returned or logged at warning level.
    """
    return f"{stage} ({type(error).__name__})"


class PoseProviderConfigError(ValueError):
    """Raised for provider configuration/programming errors (never degraded)."""



class PoseProvider(Protocol):
    """Narrow pose capability: one frame in, one immutable pose result out."""

    @property
    def available(self) -> bool:
        """False once the provider is known to be unusable (e.g. load failed)."""

    def infer(self, frame: Any) -> PoseFrameResult:
        """Never raises for ordinary model/runtime failure; returns a status."""


class _Malformed(Exception):
    """Internal signal: this frame is unusable."""

    def __init__(self, reason: str, status: PoseStatus = PoseStatus.MALFORMED_RESULT) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


def _is_finite(value: Any) -> bool:
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
        if number >= -BOUNDS_TOLERANCE:
            return 0.0
        return None
    if number > 1.0:
        if number <= 1.0 + BOUNDS_TOLERANCE:
            return 1.0
        return None
    return number


#: Confidence and normalized coordinates share the same 0..1 contract.
_valid_confidence = _unit_value
_valid_normalized = _unit_value


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
    """Strict xyxyn box: ``x1 < x2`` and ``y1 < y2``, never repaired."""
    values = _rows(row)
    if values is None or len(values) < 4:
        return None
    coords = [_valid_normalized(value) for value in values[:4]]
    if any(coord is None for coord in coords):
        return None
    x1, y1, x2, y2 = coords  # type: ignore[misc]
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1, y1, x2 - x1, y2 - y1)


def _build_keypoints(points: list, confidences: Optional[list]) -> tuple[PoseKeypoint, ...]:
    """Builds a confidence-bearing COCO-17 keypoint tuple or raises."""
    if len(points) != COCO_17_KEYPOINT_COUNT:
        raise _Malformed(
            f"expected {COCO_17_KEYPOINT_COUNT} keypoints, got {len(points)}",
            PoseStatus.UNSUPPORTED_POSE_SCHEMA,
        )
    if confidences is None:
        raise _Malformed(
            "keypoint confidence channel absent: coordinates alone are not "
            "behavioural pose evidence",
            PoseStatus.KEYPOINT_CONFIDENCE_ABSENT,
        )
    if len(confidences) != COCO_17_KEYPOINT_COUNT:
        raise _Malformed(
            f"keypoint confidence row has {len(confidences)} entries, "
            f"expected {COCO_17_KEYPOINT_COUNT}"
        )

    keypoints: list[PoseKeypoint] = []
    for index, name in enumerate(COCO_17_KEYPOINTS):
        confidence = _valid_confidence(confidences[index])
        coordinates = _pair(points[index])
        # Availability comes from the confidence contract, never coordinates.
        if confidence is None or confidence < KEYPOINT_VISIBILITY_FLOOR or coordinates is None:
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


def _parse_instances(boxes: Any, keypoints: Any) -> tuple[PoseInstance, ...]:
    box_rows = _rows(getattr(boxes, "xyxyn", None))
    point_rows = _rows(getattr(keypoints, "xyn", None))
    if box_rows is None or point_rows is None:
        raise _Malformed("missing normalized boxes/keypoints arrays")
    if len(box_rows) != len(point_rows):
        raise _Malformed(
            f"boxes/keypoints length mismatch ({len(box_rows)} vs {len(point_rows)})"
        )
    if not box_rows:
        return ()

    box_confidences = _rows(getattr(boxes, "conf", None))
    if box_confidences is not None and len(box_confidences) != len(box_rows):
        raise _Malformed(
            f"box confidence length mismatch ({len(box_confidences)} vs {len(box_rows)})"
        )
    keypoint_confidences = _rows(getattr(keypoints, "conf", None))
    if keypoint_confidences is not None and len(keypoint_confidences) != len(point_rows):
        raise _Malformed("keypoint confidence length mismatch")

    instances: list[PoseInstance] = []
    for index, points in enumerate(point_rows):
        point_list = _rows(points)
        if point_list is None:
            raise _Malformed("unreadable keypoint row")
        if keypoint_confidences is None:
            confidences = None
        else:
            confidences = _rows(keypoint_confidences[index])
            if confidences is None:
                raise _Malformed("unreadable keypoint confidence row")
        built = _build_keypoints(point_list, confidences)

        bbox = _normalized_bbox(box_rows[index])
        if bbox is None:
            raise _Malformed(f"unusable pose bbox at instance {index}")

        confidence: Optional[float] = None
        if box_confidences is not None:
            confidence = _valid_confidence(box_confidences[index])
            if confidence is None:
                raise _Malformed(f"invalid box confidence at instance {index}")
        try:
            instances.append(PoseInstance(bbox=bbox, keypoints=built, confidence=confidence))
        except PoseContractError as error:
            raise _Malformed(f"pose instance rejected: {error}") from error
    return tuple(instances)


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
    try:
        instances = _parse_instances(boxes, keypoints)
    except _Malformed as error:
        return PoseFrameResult.failure(error.status, error.reason, model_name)
    return PoseFrameResult(status=PoseStatus.OK, instances=instances, model_name=model_name)


class UltralyticsPoseProvider:
    """Pose provider backed by the existing pinned Ultralytics YOLO stack.

    * Configuration is validated in the constructor and never clamped; invalid
      configuration raises :class:`PoseProviderConfigError`.
    * Weights are NEVER loaded at import time; loading happens lazily on the
      first :meth:`infer` (or via an explicit :meth:`initialize`).
    * ``model_factory`` is injectable so tests need no real weights.
    * One provider owns ONE pose model guarded by its OWN lock; it never
      touches ``YoloDetector``'s lock and never calls ``model.track``.
    * A load failure is STICKY for the instance: the provider stays
      ``available=False`` until future runtime configuration management
      reconstructs it. No retry policy is implemented here.
    * Error reasons carry only the model file name and the error text — never
      full credential-bearing paths, tokens or URLs.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        imgsz: int = 640,
        confidence: float = 0.25,
        model_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise PoseProviderConfigError("model_name must be a non-empty string")
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
            raise PoseProviderConfigError(f"imgsz must be a positive integer, got {imgsz!r}")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise PoseProviderConfigError("confidence must be a number in 0..1")
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise PoseProviderConfigError(
                f"confidence must be finite within 0..1, got {confidence!r}"
            )
        if not isinstance(device, str) or not device.strip():
            raise PoseProviderConfigError("device must be a non-empty string")

        self.model_name = model_name.strip()
        self.device = device.strip()
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

    @property
    def _safe_model_label(self) -> str:
        """Model file name only: never a full, possibly sensitive path."""
        return os.path.basename(self.model_name) or self.model_name

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
            self._load_error = _safe_reason("pose model unavailable", error)
            # Only the exception CLASS is logged: messages may embed private
            # filesystem paths or credentials.
            logger.warning(
                "Pose model %s unavailable (%s)",
                self._safe_model_label,
                type(error).__name__,
            )
            return None
        return self._model

    def infer(self, frame: Any) -> PoseFrameResult:
        """Runs untracked pose prediction on one frame; never raises normally."""
        label = self._safe_model_label
        with self._lock:
            model = self._ensure_model()
            if model is None:
                return PoseFrameResult.failure(
                    PoseStatus.MODEL_UNAVAILABLE,
                    self._load_error or "pose model could not be loaded",
                    label,
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
                # Never log the raw message: it can carry RTSP credentials.
                logger.warning(
                    "Pose inference failed on %s (%s)", label, type(error).__name__
                )
                return PoseFrameResult.failure(
                    PoseStatus.INFERENCE_FAILED,
                    _safe_reason("pose inference failed", error),
                    label,
                )

        rows = _rows(results)
        if rows is None:
            return PoseFrameResult.failure(
                PoseStatus.MALFORMED_RESULT, "predict() returned no readable results", label
            )
        if len(rows) != 1:
            # One frame in must mean exactly one Results object out.
            return PoseFrameResult.failure(
                PoseStatus.MALFORMED_RESULT,
                f"expected exactly 1 result for 1 frame, got {len(rows)}",
                label,
            )
        return parse_pose_result(rows[0], label)
