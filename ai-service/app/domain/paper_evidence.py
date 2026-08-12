"""Immutable paper-evidence domain contract (raw object evidence ONLY).

Scope
-----
This module describes what a paper-specific detector reported for ONE analysed
frame, in normalized 0..1 full-frame coordinates. It is pure: no Ultralytics,
OpenCV, network, database, threading or configuration access.

What the ``paper`` class means
------------------------------
Exactly ONE canonical semantic class exists for the MVP custom detector:
:data:`CANONICAL_PAPER_CLASS` (``"paper"``). It means a *loose sheet of paper*
that a person could hold, pass or read:

* full exam sheet / answer sheet
* A4 / Letter-like loose sheet
* small paper slip / cheat sheet
* folded loose paper
* partially visible loose paper (visible extent only)

It explicitly does NOT include and has NO aliases for: ``book``, ``notebook``,
``phone``, ``pen``, ``hand``, ``desk``, folders, tablets or screens. The stock
COCO ``book`` class is NEVER an acceptable proxy for paper; a notebook/book may
become its own separate class later if a real need is demonstrated.

Truthfulness invariants
-----------------------
* A :class:`PaperDetection` carries only source facts: a strictly valid
  normalized bbox, a real finite confidence in 0..1, the canonical class and
  optional safe provenance labels. No behaviour, no fusion, no identity.
* A degraded (non-``OK``) :class:`PaperEvidenceFrame` can NEVER carry
  detections.
* ``OK`` with zero detections means "no paper was detected on this frame" — it
  is NOT a claim that no paper exists.
* Evidence is not an event: a paper detection never means transfer, handoff,
  exchange, cheating or suspicion. Nothing from the temporal handoff layer
  (Task 3D) is imported or referenced here.
* No object identity: detections are frame-local facts. There is deliberately
  no ``paper_tracking_id``, and a detection index is never a person identity.
* Confidence stays the detector's own confidence; it is never combined with
  person, pose, wrist or temporal confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .geometry import BBox

#: The single canonical internal class name of the MVP paper detector.
CANONICAL_PAPER_CLASS = "paper"

#: Class names that must never be accepted or aliased as paper evidence.
FORBIDDEN_PAPER_ALIASES: frozenset[str] = frozenset(
    {"book", "notebook", "phone", "cell phone", "pen", "pencil", "hand", "desk"}
)


class PaperEvidenceStatus(str, Enum):
    """Outcome of ONE paper-detection attempt on ONE frame.

    ``OK`` covers a successful inference, including one that found zero paper.
    In that case the ONLY truthful reading is: "no paper evidence was detected
    by this model on this frame" — never "there is definitely no paper".

    Every other member means the frame produced NO usable paper evidence and
    always carries an empty ``detections`` tuple.
    """

    OK = "ok"
    #: Weights absent/unloadable. No stock, COCO or ``book`` fallback exists.
    MODEL_UNAVAILABLE = "model_unavailable"
    #: A custom checkpoint loaded but does not expose the canonical paper class.
    MODEL_SCHEMA_MISMATCH = "model_schema_mismatch"
    #: Open-vocabulary prompt configuration was rejected or could not be
    #: applied to the backend. Never downgraded to valid empty evidence.
    PROMPT_CONFIGURATION_INVALID = "prompt_configuration_invalid"
    INFERENCE_FAILED = "inference_failed"
    MALFORMED_RESULT = "malformed_result"



class PaperDetectorContractError(ValueError):
    """Raised when a paper domain object would violate its own invariants."""


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _safe_label(value: object, field_name: str) -> Optional[str]:
    """Optional provenance label: a non-empty, path-free, short string."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PaperDetectorContractError(
            f"{field_name} must be a non-empty string when supplied"
        )
    label = value.strip()
    if "/" in label or "\\" in label:
        raise PaperDetectorContractError(
            f"{field_name} must not contain filesystem path separators"
        )
    if len(label) > 128:
        raise PaperDetectorContractError(f"{field_name} must be at most 128 characters")
    return label


@dataclass(frozen=True, slots=True)
class PaperDetection:
    """One paper object reported by a paper-specific detector on one frame.

    Coordinates are ALWAYS normalized full-frame coordinates. When a detection
    originated from a crop, the provider maps it back through an explicit
    validated crop transform before constructing this object, and records the
    crop via ``crop_source``.

    ``class_name`` is the canonical internal semantic (always ``"paper"``).
    ``raw_prompt`` preserves the EXACT open-vocabulary prompt / model class label
    that actually fired (for example ``"small paper slip"``), so diagnostics can
    always see which prompt produced the evidence. The canonical semantic never
    hides the raw prompt, and the raw prompt is never treated as a behaviour.
    ``backend`` records which model/backend produced it (safe identifier only).
    """

    bbox: BBox
    confidence: float
    class_name: str = CANONICAL_PAPER_CLASS
    raw_prompt: Optional[str] = None
    model_name: Optional[str] = None
    backend: Optional[str] = None
    crop_source: Optional[str] = None


    def __post_init__(self) -> None:
        if not isinstance(self.bbox, BBox):
            raise PaperDetectorContractError("bbox must be a BBox")
        for name in ("x", "y", "width", "height"):
            if not _finite_number(getattr(self.bbox, name)):
                raise PaperDetectorContractError(f"bbox.{name} must be a finite number")
        if self.bbox.width <= 0.0 or self.bbox.height <= 0.0:
            raise PaperDetectorContractError(
                "bbox must have positive area (reversed/zero-area boxes are malformed, "
                "never repaired)"
            )
        for value, name in (
            (self.bbox.x, "x1"),
            (self.bbox.y, "y1"),
            (self.bbox.x2, "x2"),
            (self.bbox.y2, "y2"),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise PaperDetectorContractError(
                    f"bbox {name}={value!r} is outside the normalized 0..1 frame"
                )

        if not _finite_number(self.confidence):
            raise PaperDetectorContractError(
                "confidence must be a real finite float (bool/NaN/inf rejected)"
            )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise PaperDetectorContractError(
                f"confidence must be within 0..1, got {self.confidence!r}"
            )
        object.__setattr__(self, "confidence", float(self.confidence))

        if self.class_name != CANONICAL_PAPER_CLASS:
            raise PaperDetectorContractError(
                f"class_name must be the canonical {CANONICAL_PAPER_CLASS!r} class; "
                f"{self.class_name!r} is not paper evidence"
            )
        raw_prompt = _safe_label(self.raw_prompt, "raw_prompt")
        if raw_prompt is not None and raw_prompt.strip().lower() in FORBIDDEN_PAPER_ALIASES:
            raise PaperDetectorContractError(
                f"{raw_prompt!r} is never paper evidence and can never be a paper prompt"
            )
        object.__setattr__(self, "raw_prompt", raw_prompt)
        object.__setattr__(self, "model_name", _safe_label(self.model_name, "model_name"))
        object.__setattr__(self, "backend", _safe_label(self.backend, "backend"))
        object.__setattr__(self, "crop_source", _safe_label(self.crop_source, "crop_source"))

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "class_name": self.class_name,
            "bbox": self.bbox.to_dict(),
            "confidence": round(self.confidence, 6),
        }
        if self.raw_prompt is not None:
            payload["raw_prompt"] = self.raw_prompt
        if self.model_name is not None:
            payload["model_name"] = self.model_name
        if self.backend is not None:
            payload["backend"] = self.backend
        if self.crop_source is not None:
            payload["crop_source"] = self.crop_source
        return payload



@dataclass(frozen=True, slots=True)
class PaperEvidenceFrame:
    """Result of one paper-detection attempt on one analysed frame.

    ``OK`` with ``detections=()`` means ONLY: "no paper evidence was detected by
    this model on this frame". It is never a claim that no paper exists.

    ``frame_index`` / ``timestamp_seconds`` are optional source-frame metadata
    (offline evaluation supplies them; live code has no obligation to).
    """

    status: PaperEvidenceStatus
    detections: tuple[PaperDetection, ...] = field(default_factory=tuple)
    model_name: Optional[str] = None
    backend: Optional[str] = None
    reason: Optional[str] = None
    frame_index: Optional[int] = None
    timestamp_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PaperEvidenceStatus):
            raise PaperDetectorContractError("status must be a PaperEvidenceStatus")
        if not isinstance(self.detections, tuple):
            raise PaperDetectorContractError("detections must be an immutable tuple")
        for detection in self.detections:
            if not isinstance(detection, PaperDetection):
                raise PaperDetectorContractError("detections must contain PaperDetection only")
        if self.status is not PaperEvidenceStatus.OK and self.detections:
            raise PaperDetectorContractError(
                "a degraded paper frame can never carry detections"
            )
        if self.status is PaperEvidenceStatus.OK and self.reason is not None:
            raise PaperDetectorContractError("an OK paper frame carries no failure reason")
        if self.status is not PaperEvidenceStatus.OK and not self.reason:
            raise PaperDetectorContractError("a degraded paper frame requires a safe reason")
        object.__setattr__(self, "model_name", _safe_label(self.model_name, "model_name"))
        object.__setattr__(self, "backend", _safe_label(self.backend, "backend"))
        if self.frame_index is not None:
            if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
                raise PaperDetectorContractError("frame_index must be a non-negative int")
            if self.frame_index < 0:
                raise PaperDetectorContractError("frame_index must be a non-negative int")
        if self.timestamp_seconds is not None:
            if not _finite_number(self.timestamp_seconds) or float(self.timestamp_seconds) < 0.0:
                raise PaperDetectorContractError(
                    "timestamp_seconds must be a finite non-negative number"
                )
            object.__setattr__(self, "timestamp_seconds", float(self.timestamp_seconds))

    @property
    def ok(self) -> bool:
        return self.status is PaperEvidenceStatus.OK

    @property
    def has_paper_evidence(self) -> bool:
        """True only when a paper model genuinely reported paper on this frame."""
        return self.ok and bool(self.detections)

    def with_frame_metadata(
        self, frame_index: Optional[int] = None, timestamp_seconds: Optional[float] = None
    ) -> "PaperEvidenceFrame":
        """Returns a copy carrying source-frame metadata (never mutates)."""
        return PaperEvidenceFrame(
            status=self.status,
            detections=self.detections,
            model_name=self.model_name,
            backend=self.backend,
            reason=self.reason,
            frame_index=frame_index if frame_index is not None else self.frame_index,
            timestamp_seconds=(
                timestamp_seconds
                if timestamp_seconds is not None
                else self.timestamp_seconds
            ),
        )

    @classmethod
    def empty(
        cls, model_name: Optional[str] = None, backend: Optional[str] = None
    ) -> "PaperEvidenceFrame":
        """Successful inference that detected no paper (NOT a no-paper claim)."""
        return cls(
            status=PaperEvidenceStatus.OK,
            detections=(),
            model_name=model_name,
            backend=backend,
        )

    @classmethod
    def failure(
        cls,
        status: PaperEvidenceStatus,
        reason: str,
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> "PaperEvidenceFrame":
        if status is PaperEvidenceStatus.OK:
            raise PaperDetectorContractError("failure() requires a degraded status")
        return cls(
            status=status,
            detections=(),
            model_name=model_name,
            backend=backend,
            reason=reason,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status.value,
            "detections": [detection.to_dict() for detection in self.detections],
        }
        if self.model_name is not None:
            payload["model_name"] = self.model_name
        if self.backend is not None:
            payload["backend"] = self.backend
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.frame_index is not None:
            payload["frame_index"] = self.frame_index
        if self.timestamp_seconds is not None:
            payload["timestamp_seconds"] = round(float(self.timestamp_seconds), 6)
        return payload


#: Task 3E-B naming alias: the detection type IS the paper evidence detection.
PaperEvidenceDetection = PaperDetection

        return payload
