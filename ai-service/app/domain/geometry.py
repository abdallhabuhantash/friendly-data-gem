"""Pure geometry helpers.

All event-facing bounding boxes are normalized to 0..1 relative to the analysed
frame. Pixel coordinates are used only for rendering annotations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def clamp01(value: float) -> float:
    """Clamps a float safely into the 0..1 range."""
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class BBox:
    """Normalized bounding box (x, y = top-left corner)."""

    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def diagonal(self) -> float:
        return math.hypot(self.width, self.height)

    def to_dict(self) -> dict[str, float]:
        return {
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "width": round(self.width, 6),
            "height": round(self.height, 6),
        }

    def to_pixels(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        """Returns (x1, y1, x2, y2) in pixels for annotation rendering."""
        return (
            int(round(self.x * frame_width)),
            int(round(self.y * frame_height)),
            int(round(self.x2 * frame_width)),
            int(round(self.y2 * frame_height)),
        )


def normalize_xyxy(
    x1: float, y1: float, x2: float, y2: float, frame_width: int, frame_height: int
) -> BBox:
    """Converts pixel xyxy into a clamped normalized BBox."""
    if frame_width <= 0 or frame_height <= 0:
        return BBox(0.0, 0.0, 0.0, 0.0)
    left = clamp01(min(x1, x2) / frame_width)
    top = clamp01(min(y1, y2) / frame_height)
    right = clamp01(max(x1, x2) / frame_width)
    bottom = clamp01(max(y1, y2) / frame_height)
    return BBox(left, top, clamp01(right - left), clamp01(bottom - top))


def contains_point(box: BBox, point: tuple[float, float]) -> bool:
    """True when the point lies inside the box."""
    px, py = point
    return box.x <= px <= box.x2 and box.y <= py <= box.y2


def expand(box: BBox, factor: float) -> BBox:
    """Expands a box around its center by `factor` (0.2 == +20% each side)."""
    cx, cy = box.center
    half_w = box.width * (1.0 + factor) / 2.0
    half_h = box.height * (1.0 + factor) / 2.0
    x = clamp01(cx - half_w)
    y = clamp01(cy - half_h)
    return BBox(x, y, clamp01(min(1.0, cx + half_w) - x), clamp01(min(1.0, cy + half_h) - y))


def intersection_area(a: BBox, b: BBox) -> float:
    width = min(a.x2, b.x2) - max(a.x, b.x)
    height = min(a.y2, b.y2) - max(a.y, b.y)
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def containment_ratio(inner: BBox, outer: BBox) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    return intersection_area(inner, outer) / inner.area if inner.area > 0 else 0.0


def upper_body_region(person: BBox) -> BBox:
    """Torso/upper-body region: the top 55% of the person box."""
    return BBox(person.x, person.y, person.width, person.height * 0.55)


def normalized_distance(point: tuple[float, float], person: BBox) -> float:
    """Distance from a point to the person's upper-body center, scaled by size."""
    region = upper_body_region(person)
    cx, cy = region.center
    scale = max(person.diagonal, 1e-6)
    return math.hypot(point[0] - cx, point[1] - cy) / scale