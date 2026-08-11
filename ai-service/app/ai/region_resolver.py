"""Pure person-relative and configured-region geometry resolution.

Stateless, deterministic, dependency-free (no OpenCV, no model, no database, no
temporal history). Every function takes the person's bounding box explicitly, so
two people in the same frame are classified independently and no camera's
configured polygon can leak into another camera.

Truthfulness notes (see also ``app/domain/regions``):

* Person-relative geometry scales with the detected person box, which makes it
  tolerant of camera distance/resolution, but it is NOT 3D scene understanding.
* "Lower person region" means the lower fraction of the person's 2D detector
  box. It is not a lap, not a desk, and not the space under a desk. Only an
  explicitly configured per-camera polygon may carry deployment scene meaning.
* Nothing here produces a confidence score; geometry is geometry.
* Observed geometry is validated strictly and never silently clamped: malformed
  boxes/points yield an explicit unavailable result with no positive membership
  fields, so no caller can mistake malformed input for evidence.

This module is intentionally NOT wired into the runtime pipeline, so it adds no
per-frame cost to the existing detection path.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from ..domain.geometry import BBox, containment_ratio, intersection_area
from ..domain.regions import (
    BBoxRegionFacts,
    ConfiguredRegion,
    ConfiguredRegionMembership,
    NormalizedPolygon,
    PersonRegionSpec,
    PointRegionFacts,
    RegionConfigError,
    RelativePoint,
)

#: Person boxes thinner/shorter than this (normalized) are unusable geometry.
MIN_PERSON_EXTENT = 1e-4

#: Floating-point rounding tolerance for normalized frame bounds only.
FRAME_EPS = 1e-9


def _finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def is_valid_normalized_box(box: BBox, *, min_extent: float = 0.0) -> bool:
    """Strict normalized-BBox contract check (no clamping, no repair)."""
    values = (box.x, box.y, box.width, box.height)
    if not all(_finite(value) for value in values):
        return False
    if box.width <= 0.0 or box.height <= 0.0:
        return False
    if box.width < min_extent or box.height < min_extent:
        return False
    if box.x < -FRAME_EPS or box.y < -FRAME_EPS:
        return False
    if box.x2 > 1.0 + FRAME_EPS or box.y2 > 1.0 + FRAME_EPS:
        return False
    return True


def is_valid_normalized_point(point: tuple[float, float]) -> bool:
    """A point is valid observed data only inside the normalized frame."""
    if point is None or len(point) != 2 or not all(_finite(value) for value in point):
        return False
    x, y = float(point[0]), float(point[1])
    return -FRAME_EPS <= x <= 1.0 + FRAME_EPS and -FRAME_EPS <= y <= 1.0 + FRAME_EPS


def is_person_box_usable(person: BBox) -> bool:
    """True when the person box can support relative geometry at all."""
    return is_valid_normalized_box(person, min_extent=MIN_PERSON_EXTENT)


def relative_point(person: BBox, point: tuple[float, float]) -> Optional[RelativePoint]:
    """Point expressed as fractions of the person box, or ``None`` if unusable."""
    if not is_person_box_usable(person):
        return None
    if not is_valid_normalized_point(point):
        return None
    return RelativePoint(
        relative_x=(float(point[0]) - person.x) / person.width,
        relative_y=(float(point[1]) - person.y) / person.height,
    )


def lower_person_region(person: BBox, spec: PersonRegionSpec) -> Optional[BBox]:
    """Sub-rectangle of the person box described by ``spec`` (frame coordinates)."""
    if not is_person_box_usable(person):
        return None
    x = person.x + person.width * spec.left_fraction
    y = person.y + person.height * spec.lower_start_fraction
    width = person.width * (spec.right_fraction - spec.left_fraction)
    height = person.height * (spec.lower_end_fraction - spec.lower_start_fraction)
    return BBox(x, y, width, height)


def point_in_polygon(polygon: NormalizedPolygon, point: tuple[float, float]) -> bool:
    """Deterministic point-in-polygon test (supports concave simple polygons).

    Boundary convention: a point lying exactly on an edge or vertex is treated as
    INSIDE. Interior membership uses a ray-casting (crossing-number) rule, which
    is orientation-independent, so clockwise and counter-clockwise vertex orders
    give identical results.
    """
    if not is_valid_normalized_point(point):
        return False
    px, py = float(point[0]), float(point[1])
    vertices = polygon.vertices
    count = len(vertices)
    eps = 1e-12

    # Boundary first: on-edge counts as inside.
    for index in range(count):
        x1, y1 = vertices[index]
        x2, y2 = vertices[(index + 1) % count]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) <= 1e-12:
            if (
                min(x1, x2) - eps <= px <= max(x1, x2) + eps
                and min(y1, y2) - eps <= py <= max(y1, y2) + eps
            ):
                return True

    inside = False
    for index in range(count):
        x1, y1 = vertices[index]
        x2, y2 = vertices[(index + 1) % count]
        if (y1 > py) != (y2 > py):
            x_at = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_at:
                inside = not inside
    return inside


def validate_configured_regions(
    regions: Optional[Sequence[ConfiguredRegion]],
) -> Optional[tuple[ConfiguredRegion, ...]]:
    """Validate a supplied configured-region set; ``None``/empty means unconfigured.

    Duplicate non-null ``region_id`` values are a configuration error, because
    ambiguous identity would make future evidence unexplainable. Labels need not
    be unique. No global storage is involved: only the supplied set is checked.
    """
    if not regions:
        return None
    seen: set[str] = set()
    for region in regions:
        if region.region_id is None:
            continue
        if region.region_id in seen:
            raise RegionConfigError(f"duplicate configured region_id: {region.region_id!r}")
        seen.add(region.region_id)
    return tuple(regions)


def _configured_memberships(
    regions: Optional[Sequence[ConfiguredRegion]], point: tuple[float, float]
) -> Optional[tuple[ConfiguredRegionMembership, ...]]:
    """``None`` when nothing is configured; otherwise per-region membership."""
    validated = validate_configured_regions(regions)
    if validated is None:
        return None
    return tuple(
        ConfiguredRegionMembership(
            inside=point_in_polygon(region.polygon, point),
            region_id=region.region_id,
            label=region.label,
        )
        for region in validated
    )


def resolve_point(
    *,
    person: BBox,
    point: tuple[float, float],
    spec: PersonRegionSpec,
    configured_regions: Optional[Sequence[ConfiguredRegion]] = None,
) -> PointRegionFacts:
    """Geometric facts for one point relative to one person (and optional zones)."""
    if not is_person_box_usable(person):
        return PointRegionFacts(available=False, reason="degenerate_person_box")
    if not is_valid_normalized_point(point):
        return PointRegionFacts(available=False, reason="invalid_point")
    relative = relative_point(person, point)
    if relative is None:
        return PointRegionFacts(available=False, reason="invalid_point")

    region = lower_person_region(person, spec)
    inside_person = relative.inside_person
    inside_lower = bool(
        inside_person
        and region is not None
        and region.x <= float(point[0]) <= region.x2
        and region.y <= float(point[1]) <= region.y2
    )
    return PointRegionFacts(
        available=True,
        relative_position=relative,
        inside_person=inside_person,
        inside_lower_person_region=inside_lower,
        configured_region_memberships=_configured_memberships(configured_regions, point),
    )


def resolve_bbox(
    *,
    person: BBox,
    obj: BBox,
    spec: PersonRegionSpec,
    configured_regions: Optional[Sequence[ConfiguredRegion]] = None,
) -> BBoxRegionFacts:
    """Geometric facts for an object bounding box relative to one person.

    A malformed object box (zero/negative extent, non-finite, or reaching outside
    the normalized frame) returns ``available=False`` with NO membership facts.
    """
    if not is_person_box_usable(person):
        return BBoxRegionFacts(available=False, reason="degenerate_person_box")
    if not all(_finite(value) for value in (obj.x, obj.y, obj.width, obj.height)):
        return BBoxRegionFacts(
            available=False,
            reason="invalid_object_box",
            lower_region_intersection_area=0.0,
        )
    if obj.width <= 0.0 or obj.height <= 0.0:
        return BBoxRegionFacts(
            available=False,
            reason="zero_area_object" if obj.width == 0.0 or obj.height == 0.0 else "negative_extent_object_box",
            lower_region_intersection_area=0.0,
        )
    if not is_valid_normalized_box(obj):
        return BBoxRegionFacts(
            available=False,
            reason="object_box_outside_frame",
            lower_region_intersection_area=0.0,
        )

    region = lower_person_region(person, spec)
    if region is None:
        return BBoxRegionFacts(available=False, reason="degenerate_person_box")

    center = obj.center
    center_facts = resolve_point(
        person=person, point=center, spec=spec, configured_regions=configured_regions
    )

    return BBoxRegionFacts(
        available=True,
        center_inside_lower_person_region=center_facts.inside_lower_person_region,
        lower_region_containment_ratio=containment_ratio(obj, region),
        lower_region_intersection_area=intersection_area(obj, region),
        center_relative_position=center_facts.relative_position,
        center_inside_person=center_facts.inside_person,
        configured_region_center_memberships=center_facts.configured_region_memberships,
    )


__all__ = [
    "FRAME_EPS",
    "MIN_PERSON_EXTENT",
    "is_person_box_usable",
    "is_valid_normalized_box",
    "is_valid_normalized_point",
    "lower_person_region",
    "point_in_polygon",
    "relative_point",
    "resolve_bbox",
    "resolve_point",
    "validate_configured_regions",
]
