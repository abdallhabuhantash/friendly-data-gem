"""Ultralytics YOLO wrapper with per-camera tracker isolation.

Class IDs are discovered from ``model.names``, never hard-coded, so switching
to another YOLO checkpoint requires no code change.

Tracker isolation
-----------------
Ultralytics ``model.track(persist=True)`` keeps the tracker state on the
*shared predictor* (``model.predictor.trackers``). Calling it with
``persist=True`` for camera A and then camera B would let ByteTrack track IDs
bleed across cameras, and ``persist=False`` would throw the history away every
frame (no tracking at all).

To guarantee that Camera A's tracking history can never influence Camera B,
:class:`TrackerStateStore` swaps each camera's own tracker state onto the
shared predictor immediately before ``track(persist=True)`` and captures it
again afterwards. This keeps exactly one shared YOLO model (no model copy per
camera) while tracker state stays strictly per camera, and uses only the
Ultralytics attributes the public ``track()`` API itself maintains.

"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from ..domain.geometry import normalize_xyxy
from ..domain.models import CLASS_PERSON, CLASS_PHONE, Detection, FrameDetections

logger = logging.getLogger(__name__)

#: Model label -> application class name.
LABEL_MAP = {
    "person": CLASS_PERSON,
    "cell phone": CLASS_PHONE,
    "cell_phone": CLASS_PHONE,
    "mobile phone": CLASS_PHONE,
}


def resolve_device(configured: str) -> str:
    """``auto`` picks CUDA when available, otherwise CPU. Never hard-coded GPU."""
    if configured and configured.lower() != "auto":
        return configured
    try:
        import torch  # imported lazily: heavy dependency

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # pragma: no cover - torch absent or broken
        pass
    return "cpu"


def wanted_class_ids(names: dict[int, str]) -> dict[int, str]:
    """Maps the model's own class IDs to the two classes this engine needs."""
    wanted: dict[int, str] = {}
    for class_id, label in names.items():
        mapped = LABEL_MAP.get(str(label).strip().lower())
        if mapped:
            wanted[int(class_id)] = mapped
    return wanted


#: Predictor attributes that hold Ultralytics' per-source tracking state.
TRACKER_STATE_ATTRS = ("trackers", "vid_path")


class TrackerStateStore:
    """Keeps one Ultralytics tracker state per camera for a shared predictor.

    ``model.track(persist=True)`` stores its tracker objects on the predictor.
    This store swaps the right camera's state in before inference and captures
    it back afterwards, so no two cameras ever share tracker history. Deleting
    the attributes makes Ultralytics build a fresh tracker set for a camera it
    has not seen yet (or one that was reset), which is exactly what its own
    ``on_predict_start`` hook does.

    It touches no private Ultralytics classes, so it works with any Ultralytics
    release whose ``track()`` API maintains predictor state.
    """

    def __init__(self) -> None:
        self._states: dict[str, dict[str, object]] = {}

    def known(self, camera_id: str) -> bool:
        return camera_id in self._states

    @property
    def cameras(self) -> list[str]:
        return sorted(self._states)

    def install(self, predictor: object, camera_id: str) -> None:
        """Puts this camera's tracker state on the predictor before inference."""
        state = self._states.get(camera_id)
        for attr in TRACKER_STATE_ATTRS:
            if state is not None and attr in state:
                setattr(predictor, attr, state[attr])
            else:
                # No state yet for this camera: force a fresh tracker set.
                try:
                    delattr(predictor, attr)
                except AttributeError:
                    pass

    def capture(self, predictor: object, camera_id: str) -> None:
        """Stores the tracker state produced by this camera's inference call."""
        state: dict[str, object] = {}
        for attr in TRACKER_STATE_ATTRS:
            if hasattr(predictor, attr):
                state[attr] = getattr(predictor, attr)
        if state:
            self._states[camera_id] = state

    def reset(self, camera_id: str) -> None:
        """Drops the state of exactly one camera; every other camera is kept."""
        self._states.pop(camera_id, None)


class YoloDetector:
    """Thread-safe front-end for one shared Ultralytics model.

    Inference is serialised behind a lock. Per-camera tracker state is kept
    isolated by :class:`TrackerStateStore`, which swaps the current camera's
    state onto the shared predictor around each ``track()`` call.
    """

    def __init__(self, model_name: str, device: str, imgsz: int, tracker: str) -> None:
        from ultralytics import YOLO  # imported lazily so unit tests stay light

        self.device = resolve_device(device)
        self.model_name = model_name
        self.imgsz = int(imgsz)
        self.tracker = tracker if tracker.endswith(".yaml") else f"{tracker}.yaml"
        self._lock = threading.Lock()
        self._model = YOLO(model_name)
        self._classes = wanted_class_ids(dict(self._model.names))
        self._trackers = TrackerStateStore()
        logger.info(
            "YOLO model %s ready on %s (classes: %s)",
            model_name,
            self.device,
            sorted(set(self._classes.values())),
        )

    @property
    def class_ids(self) -> list[int]:
        return sorted(self._classes)

    def reset_camera(self, camera_id: str) -> None:
        """Drops tracker state for ONE camera only (e.g. on config change)."""
        self._trackers.reset(camera_id)

    def detect(self, frame, camera_id: str, min_confidence: float = 0.20) -> FrameDetections:
        """Runs tracked inference on one BGR frame and returns typed detections."""
        height, width = frame.shape[:2]
        with self._lock:
            predictor = getattr(self._model, "predictor", None)
            if predictor is not None:
                self._trackers.install(predictor, camera_id)
            results = self._model.track(
                source=frame,
                persist=True,
                tracker=self.tracker,
                imgsz=self.imgsz,
                device=self.device,
                classes=self.class_ids,
                conf=min_confidence,
                verbose=False,
            )
            predictor = getattr(self._model, "predictor", None)
            if predictor is not None:
                self._trackers.capture(predictor, camera_id)


        persons: list[Detection] = []
        phones: list[Detection] = []
        if not results:
            return FrameDetections()

        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return FrameDetections()

        for index in range(len(boxes)):
            class_id = int(boxes.cls[index].item())
            class_name = self._classes.get(class_id)
            if not class_name:
                continue
            confidence = float(boxes.conf[index].item())
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[index].tolist())
            tracking_id: Optional[str] = None
            if getattr(boxes, "id", None) is not None:
                tracking_id = f"{int(boxes.id[index].item()):02d}"
            detection = Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=normalize_xyxy(x1, y1, x2, y2, width, height),
                tracking_id=tracking_id,
            )
            if class_name == CLASS_PERSON:
                persons.append(detection)
            else:
                phones.append(detection)

        return FrameDetections(persons=tuple(persons), phones=tuple(phones))
