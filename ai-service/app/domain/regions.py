"""Immutable, purely geometric region types.

Scope and truthfulness
----------------------
Everything here is deterministic 2D geometry in normalized 0..1 frame
coordinates. Nothing in this module performs inference, keeps history, reads a
database, or draws a behavioural conclusion.

Person-relative regions are expressed as fractions of a *detected person
bounding box*. A detector box is not a physical scene measurement: the lower
portion of a person's 2D box is NOT known to be a lap, a desk, or the space
under a desk. Names here stay geometric on purpose
(``inside_lower_person_region``, never ``on_lap`` / ``under_desk``).

Optional per-camera polygons (``NormalizedPolygon`` / ``ConfiguredRegion``) are
the only place where deployment-specific scene meaning may later enter, and only
because an operator explicitly configured that polygon with a label. Absence of
a configured polygon stays absence: no full-frame default is ever invented.

This is not homography, depth estimation, or 3D reconstruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .geometry import BBox


class RegionConfigError(ValueError):
    """Raised for invalid region *configuration* (a programming/config error).

    Distinct from unusable *observed* geometry, which yields an explicit
    unavailable result instead of an exception.
    """


def _is_finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class PersonRegionSpec:
    """Caller-supplied spec for a generic lower person-relative region.

    ``lower_start_fraction`` = 0.65 means, geometrically, "the sub-rectangle of
    the person's detected bounding box starting at 65% of its height". It does
    not mean a lap starts there and it does not mean cheating starts there.

    ``lower_end_fraction`` defaults to the bottom of the person box. The
    horizontal fractions allow narrowing the region inside the person's width.
    """

    lower_start_fraction: float = 0.65
    lower_end_fraction: float = 1.0
    left_fraction: float = 0.0
    right_fraction: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "lower_start_fraction",
            "lower_end_fraction",
            "left_fraction",
            "right_fraction",
        ):
            value = getattr(self, name)
            if not _is_finite(value):
                raise RegionConfigError(f"{name} must be a finite number, got {value!r}")
            if not (0.0 <= float(value) <= 1.0):
                raise RegionConfigError(f"{name} must be within 0..1, got {value!r}")
        if self.lower_start_fraction >= self.lower_end_fraction:
            raise RegionConfigError(
                "lower_start_fraction must be strictly less than lower_end_fraction"
            )
        if self.left_fraction >= self.right_fraction:
            raise RegionConfigError("left_fraction must be strictly less than right_fraction")


@dataclass(frozen=True, slots=True)
class NormalizedPolygon:
    """Immutable normalized (0..1) polygon with at least 3 distinct vertices."""

    vertices: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.vertices, tuple) or len(self.vertices) < 3:
            raise RegionConfigError("polygon requires at least 3 vertices")
        for vertex in self.vertices:
            if len(vertex) != 2:
                raise RegionConfigError("polygon vertices must be (x, y) pairs")
            for value in vertex:
                if not _is_finite(value):
                    raise RegionConfigError("polygon coordinates must be finite")
                if not (0.0 <= float(value) <= 1.0):
                    raise RegionConfigError("polygon coordinates must be normalized within 0..1")
        distinct = {(round(x, 9), round(y, 9)) for x, y in self.vertices}
        if len(distinct) < 3:
            raise RegionConfigError("polygon requires at least 3 distinct vertices")
        if self.area <= 1e-9:
            raise RegionConfigError("polygon is degenerate (zero or near-zero area)")

    @property
    def area(self) -> float:
        """Absolute shoelace area; orientation-independent."""
        total = 0.0
        count = len(self.vertices)
        for index in range(count):
            x1, y1 = self.vertices[index]
            x2, y2 = self.vertices[(index + 1) % count]
            total += (x1 * y2) - (x2 * y1)
        return abs(total) / 2.0


@dataclass(frozen=True, slots=True)
class ConfiguredRegion:
    """A per-camera configured polygon plus an opaque caller-supplied label.

    The label carries no meaning for this module: geometry never infers that a
    polygon is a desk, a doorway or anything else.
    """

    polygon: NormalizedPolygon
    region_id: Optional[str] = None
    label: Optional[str] = None


@dataclass(frozen=True, slots=True)
class RelativePoint:
    """A point expressed relative to a person's bounding box."""

    relative_x: float
    relative_y: float

    @property
    def inside_person(self) -> bool:
        return 0.0 <= self.relative_x <= 1.0 and 0.0 <= self.relative_y <= 1.0


@dataclass(frozen=True, slots=True)
class PointRegionFacts:
    """Distinct, explainable geometric facts about one point.

    Deliberately no aggregate score: person-relative membership and configured
    camera-region membership stay separate evidence sources.
    """

    available: bool
    relative_position: Optional[RelativePoint] = None
    inside_person: Optional[bool] = None
    inside_lower_person_region: Optional[bool] = None
    #: ``None`` means *no polygon was configured*, never a fabricated ``False``.
    configured_region_membership: Optional[bool] = None
    reason: Optional[str] = None


@dataclass(frozen=True, slots=True)
class BBoxRegionFacts:
    """Distinct geometric facts about an object bounding box.

    ``center_inside_lower_person_region`` and ``lower_region_containment_ratio``
    are measured separately and must not be treated as equivalent.
    """

    available: bool
    center_inside_lower_person_region: Optional[bool] = None
    lower_region_containment_ratio: Optional[float] = None
    lower_region_intersection_area: Optional[float] = None
    center_relative_position: Optional[RelativePoint] = None
    center_inside_person: Optional[bool] = None
    configured_region_center_membership: Optional[bool] = None
    reason: Optional[str] = None


__all__ = [
    "BBox",
    "BBoxRegionFacts",
    "ConfiguredRegion",
    "NormalizedPolygon",
    "PersonRegionSpec",
    "PointRegionFacts",
    "RegionConfigError",
    "RelativePoint",
]
