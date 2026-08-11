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

Calibration status
------------------
NO production lower-region threshold has been calibrated from real Vigilant Eye
footage. ``PersonRegionSpec.lower_start_fraction`` is therefore a REQUIRED
caller-supplied value: there is deliberately no default, so no invented
deployment threshold can leak in through ``PersonRegionSpec()``. Task 2G
real-video calibration will determine deployment defaults/presets. Values used
in tests are synthetic geometry fixtures only.

Optional per-camera polygons (``NormalizedPolygon`` / ``ConfiguredRegion``) are
the only place where deployment-specific scene meaning may later enter, and only
because an operator explicitly configured that polygon with a label. A label is
opaque metadata: geometry attaches no behavioural meaning to it. Absence of a
configured polygon stays absence: no full-frame default is ever invented.

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
    not mean a lap starts there and it does not mean cheating starts there. It
    is required precisely because no calibrated production value exists yet.

    ``lower_end_fraction`` and the horizontal fractions are pure structural
    defaults (the full lower band of the person box).
    """

    lower_start_fraction: float
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


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> int:
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if cross > 1e-12:
        return 1
    if cross < -1e-12:
        return -1
    return 0


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    eps = 1e-12
    return (
        _orientation(a, b, p) == 0
        and min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    """Deterministic proper/touching segment intersection test."""
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    return (
        _on_segment(p1, p2, q1)
        or _on_segment(p1, p2, q2)
        or _on_segment(q1, q2, p1)
        or _on_segment(q1, q2, p2)
    )


@dataclass(frozen=True, slots=True)
class NormalizedPolygon:
    """Immutable normalized (0..1) *simple* polygon with >= 3 distinct vertices.

    Vertices are implicitly closed: an explicit repeated closing vertex (first
    vertex repeated last) is rejected, as is any duplicate consecutive vertex,
    because it creates a zero-length edge. Self-intersecting polygons (bow-ties,
    crossing edges, non-adjacent edges touching at a vertex) are rejected.
    Concave simple polygons remain valid, in either winding order.
    """

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

        count = len(self.vertices)
        rounded = [(round(float(x), 9), round(float(y), 9)) for x, y in self.vertices]
        for index in range(count):
            if rounded[index] == rounded[(index + 1) % count]:
                raise RegionConfigError(
                    "polygon must not contain duplicate consecutive vertices "
                    "(the closing vertex is implicit)"
                )
        if len(set(rounded)) != count:
            raise RegionConfigError("polygon must not repeat any vertex")
        if len(set(rounded)) < 3:
            raise RegionConfigError("polygon requires at least 3 distinct vertices")
        if self.area <= 1e-9:
            raise RegionConfigError("polygon is degenerate (zero or near-zero area)")

        # Simplicity: non-adjacent edges must not intersect at all.
        for i in range(count):
            a1, a2 = rounded[i], rounded[(i + 1) % count]
            for j in range(i + 1, count):
                if j == i or (j + 1) % count == i or (i + 1) % count == j:
                    continue  # adjacent edges legitimately share an endpoint
                b1, b2 = rounded[j], rounded[(j + 1) % count]
                if _segments_intersect(a1, a2, b1, b2):
                    raise RegionConfigError("polygon is self-intersecting (must be a simple polygon)")

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
class ConfiguredRegionMembership:
    """Membership of one observed geometry in ONE configured region."""

    inside: bool
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

    ``configured_region_memberships`` is ``None`` when NO region was configured,
    and a tuple (possibly of all-``False`` entries) when regions were configured
    but did not match. The aggregate boolean is derived from it.
    """

    available: bool
    relative_position: Optional[RelativePoint] = None
    inside_person: Optional[bool] = None
    inside_lower_person_region: Optional[bool] = None
    #: ``None`` means *no polygon was configured*, never a fabricated ``False``.
    configured_region_memberships: Optional[tuple[ConfiguredRegionMembership, ...]] = None
    reason: Optional[str] = None

    @property
    def configured_regions_configured(self) -> bool:
        return self.configured_region_memberships is not None

    @property
    def configured_region_membership(self) -> Optional[bool]:
        """Derived convenience aggregate; never a substitute for per-region facts."""
        if self.configured_region_memberships is None:
            return None
        return any(item.inside for item in self.configured_region_memberships)

    @property
    def matched_configured_region_ids(self) -> tuple[Optional[str], ...]:
        if not self.configured_region_memberships:
            return ()
        return tuple(item.region_id for item in self.configured_region_memberships if item.inside)


@dataclass(frozen=True, slots=True)
class BBoxRegionFacts:
    """Distinct geometric facts about an object bounding box.

    ``center_inside_lower_person_region`` and ``lower_region_containment_ratio``
    are measured separately and must not be treated as equivalent. When the
    observed object geometry is unusable, ``available`` is ``False`` and every
    membership field stays ``None`` so malformed geometry can never be mistaken
    for positive evidence.
    """

    available: bool
    center_inside_lower_person_region: Optional[bool] = None
    lower_region_containment_ratio: Optional[float] = None
    lower_region_intersection_area: Optional[float] = None
    center_relative_position: Optional[RelativePoint] = None
    center_inside_person: Optional[bool] = None
    configured_region_center_memberships: Optional[tuple[ConfiguredRegionMembership, ...]] = None
    reason: Optional[str] = None

    @property
    def configured_regions_configured(self) -> bool:
        return self.configured_region_center_memberships is not None

    @property
    def configured_region_center_membership(self) -> Optional[bool]:
        """Derived convenience aggregate; never a substitute for per-region facts."""
        if self.configured_region_center_memberships is None:
            return None
        return any(item.inside for item in self.configured_region_center_memberships)

    @property
    def matched_configured_region_ids(self) -> tuple[Optional[str], ...]:
        if not self.configured_region_center_memberships:
            return ()
        return tuple(
            item.region_id for item in self.configured_region_center_memberships if item.inside
        )


__all__ = [
    "BBox",
    "BBoxRegionFacts",
    "ConfiguredRegion",
    "ConfiguredRegionMembership",
    "NormalizedPolygon",
    "PersonRegionSpec",
    "PointRegionFacts",
    "RegionConfigError",
    "RelativePoint",
]
