"""Pure, immutable crop <-> full-frame coordinate transform.

Purpose
-------
The paper detector is designed so the SAME provider can run either on a full
frame or on an explicitly supplied crop (for example a future person-pair
interaction crop). Crop coordinates must never be ambiguous: a detection that
came from a crop is mapped back to full-frame normalized coordinates through an
explicit, validated transform before it becomes :class:`PaperDetection`.

This module is pure geometry only. It performs NO image cropping, NO scheduling
and NO runtime integration; nothing here decides when or whether a crop is
produced.

Policies
--------
* The crop rectangle is expressed in normalized 0..1 FULL-FRAME coordinates and
  must be finite, non-reversed and of positive area.
* A crop-relative bbox must itself be a strictly valid normalized box
  (finite, positive area, inside 0..1). Substantively invalid boxes are
  REJECTED, never clamped.
* Only floating-point drift (``COORDINATE_TOLERANCE``) is snapped, and only at
  the mapping boundary, so a border-touching box stays border-touching.
* Mapping is deterministic and round-trips within floating-point tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..domain.geometry import BBox

#: Floating-point drift tolerance snapped at the transform boundary only.
COORDINATE_TOLERANCE = 1e-9


class CropTransformError(ValueError):
    """Raised for an invalid crop rectangle or an invalid crop-relative box."""


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _snap_unit(value: float, label: str) -> float:
    number = float(value)
    if number < 0.0:
        if number >= -COORDINATE_TOLERANCE:
            return 0.0
        raise CropTransformError(f"{label}={value!r} is below the normalized 0..1 range")
    if number > 1.0:
        if number <= 1.0 + COORDINATE_TOLERANCE:
            return 1.0
        raise CropTransformError(f"{label}={value!r} is above the normalized 0..1 range")
    return number


def _validate_normalized_box(box: BBox, label: str) -> None:
    if not isinstance(box, BBox):
        raise CropTransformError(f"{label} must be a BBox")
    for name in ("x", "y", "width", "height"):
        if not _finite(getattr(box, name)):
            raise CropTransformError(f"{label}.{name} must be a finite number")
    if box.width <= 0.0 or box.height <= 0.0:
        raise CropTransformError(f"{label} must have positive area (never repaired)")
    for value, name in ((box.x, "x1"), (box.y, "y1"), (box.x2, "x2"), (box.y2, "y2")):
        _snap_unit(value, f"{label}.{name}")


@dataclass(frozen=True, slots=True)
class CropTransform:
    """A validated crop rectangle in normalized full-frame coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for value, name in ((self.x1, "x1"), (self.y1, "y1"), (self.x2, "x2"), (self.y2, "y2")):
            if not _finite(value):
                raise CropTransformError(f"crop {name} must be a finite number")
        x1 = _snap_unit(self.x1, "crop x1")
        y1 = _snap_unit(self.y1, "crop y1")
        x2 = _snap_unit(self.x2, "crop x2")
        y2 = _snap_unit(self.y2, "crop y2")
        if x2 <= x1 or y2 <= y1:
            raise CropTransformError(
                "crop must be non-reversed with positive width and height"
            )
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "y1", y1)
        object.__setattr__(self, "x2", x2)
        object.__setattr__(self, "y2", y2)

    @classmethod
    def full_frame(cls) -> "CropTransform":
        """Identity transform: the crop IS the whole frame."""
        return cls(0.0, 0.0, 1.0, 1.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def is_identity(self) -> bool:
        return (
            abs(self.x1) <= COORDINATE_TOLERANCE
            and abs(self.y1) <= COORDINATE_TOLERANCE
            and abs(self.x2 - 1.0) <= COORDINATE_TOLERANCE
            and abs(self.y2 - 1.0) <= COORDINATE_TOLERANCE
        )

    def to_full_frame(self, box: BBox) -> BBox:
        """Maps a crop-relative normalized box into full-frame coordinates."""
        _validate_normalized_box(box, "crop-relative bbox")
        x1 = _snap_unit(self.x1 + box.x * self.width, "mapped x1")
        y1 = _snap_unit(self.y1 + box.y * self.height, "mapped y1")
        x2 = _snap_unit(self.x1 + box.x2 * self.width, "mapped x2")
        y2 = _snap_unit(self.y1 + box.y2 * self.height, "mapped y2")
        if x2 <= x1 or y2 <= y1:
            raise CropTransformError("mapped bbox collapsed to zero area")
        return BBox(x1, y1, x2 - x1, y2 - y1)

    def to_crop_relative(self, box: BBox) -> BBox:
        """Inverse mapping (full-frame box -> crop-relative), for round-trips."""
        _validate_normalized_box(box, "full-frame bbox")
        x1 = _snap_unit((box.x - self.x1) / self.width, "unmapped x1")
        y1 = _snap_unit((box.y - self.y1) / self.height, "unmapped y1")
        x2 = _snap_unit((box.x2 - self.x1) / self.width, "unmapped x2")
        y2 = _snap_unit((box.y2 - self.y1) / self.height, "unmapped y2")
        if x2 <= x1 or y2 <= y1:
            raise CropTransformError("unmapped bbox collapsed to zero area")
        return BBox(x1, y1, x2 - x1, y2 - y1)

    def to_dict(self) -> dict[str, float]:
        return {
            "x1": round(self.x1, 6),
            "y1": round(self.y1, 6),
            "x2": round(self.x2, 6),
            "y2": round(self.y2, 6),
        }
