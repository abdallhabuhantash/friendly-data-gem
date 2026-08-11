"""Deterministic pure-geometry tests for person-relative and polygon regions."""

from __future__ import annotations

import math

import pytest

from app.ai.region_resolver import (
    is_person_box_usable,
    is_valid_normalized_box,
    is_valid_normalized_point,
    lower_person_region,
    point_in_polygon,
    relative_point,
    resolve_bbox,
    resolve_point,
    validate_configured_regions,
)
from app.domain.geometry import BBox
from app.domain.regions import (
    ConfiguredRegion,
    NormalizedPolygon,
    PersonRegionSpec,
    RegionConfigError,
)

SPEC = PersonRegionSpec(lower_start_fraction=0.65)


# --- person-relative -------------------------------------------------------


def test_a_same_relative_point_in_two_sizes():
    small = BBox(0.1, 0.1, 0.1, 0.2)
    large = BBox(0.4, 0.2, 0.3, 0.6)
    p_small = (small.x + 0.5 * small.width, small.y + 0.8 * small.height)
    p_large = (large.x + 0.5 * large.width, large.y + 0.8 * large.height)
    a = relative_point(small, p_small)
    b = relative_point(large, p_large)
    assert a is not None and b is not None
    assert math.isclose(a.relative_x, b.relative_x, abs_tol=1e-9)
    assert math.isclose(a.relative_y, b.relative_y, abs_tol=1e-9)


def test_b_top_point_not_in_lower_region():
    person = BBox(0.2, 0.2, 0.2, 0.4)
    facts = resolve_point(person=person, point=(0.3, 0.25), spec=SPEC)
    assert facts.available
    assert facts.inside_person is True
    assert facts.inside_lower_person_region is False


def test_c_lower_point_in_lower_region():
    person = BBox(0.2, 0.2, 0.2, 0.4)
    facts = resolve_point(person=person, point=(0.3, 0.55), spec=SPEC)
    assert facts.inside_lower_person_region is True


def test_d_point_outside_person_never_lower_positive():
    person = BBox(0.2, 0.2, 0.2, 0.4)
    facts = resolve_point(person=person, point=(0.9, 0.55), spec=SPEC)
    assert facts.inside_person is False
    assert facts.inside_lower_person_region is False


def test_e_person_touching_boundary():
    person = BBox(0.0, 0.0, 0.25, 1.0)
    facts = resolve_point(person=person, point=(0.0, 0.99), spec=SPEC)
    assert facts.available
    assert facts.relative_position.relative_x == 0.0
    assert facts.inside_lower_person_region is True


@pytest.mark.parametrize(
    "person",
    [
        BBox(0.1, 0.1, 0.0, 0.4),
        BBox(0.1, 0.1, 0.3, 0.0),
        BBox(0.1, 0.1, 1e-9, 1e-9),
        BBox(float("nan"), 0.1, 0.2, 0.2),
        BBox(0.1, 0.1, float("inf"), 0.2),
    ],
)
def test_f_degenerate_person_unavailable(person):
    assert is_person_box_usable(person) is False
    facts = resolve_point(person=person, point=(0.15, 0.4), spec=SPEC)
    assert facts.available is False
    assert facts.inside_lower_person_region is None
    assert lower_person_region(person, SPEC) is None


def test_g_two_persons_independent():
    person_a = BBox(0.1, 0.1, 0.2, 0.6)
    person_b = BBox(0.6, 0.1, 0.2, 0.6)
    point = (0.2, 0.6)
    a = resolve_point(person=person_a, point=point, spec=SPEC)
    b = resolve_point(person=person_b, point=point, spec=SPEC)
    assert a.inside_lower_person_region is True
    assert b.inside_person is False
    assert b.inside_lower_person_region is False


def test_spec_validation_errors():
    with pytest.raises(RegionConfigError):
        PersonRegionSpec(lower_start_fraction=1.2)
    with pytest.raises(RegionConfigError):
        PersonRegionSpec(lower_start_fraction=0.8, lower_end_fraction=0.5)
    with pytest.raises(RegionConfigError):
        PersonRegionSpec(left_fraction=0.6, right_fraction=0.6)
    with pytest.raises(RegionConfigError):
        PersonRegionSpec(lower_start_fraction=float("nan"))


# --- object bbox -----------------------------------------------------------


def test_h_object_fully_inside_lower_region():
    person = BBox(0.2, 0.2, 0.4, 0.4)  # lower region y: 0.46..0.6
    obj = BBox(0.3, 0.5, 0.05, 0.05)
    facts = resolve_bbox(person=person, obj=obj, spec=SPEC)
    assert facts.center_inside_lower_person_region is True
    assert math.isclose(facts.lower_region_containment_ratio, 1.0, abs_tol=1e-9)


def test_i_object_partial_overlap():
    person = BBox(0.2, 0.2, 0.4, 0.4)
    obj = BBox(0.3, 0.42, 0.05, 0.08)  # straddles y=0.46
    facts = resolve_bbox(person=person, obj=obj, spec=SPEC)
    assert 0.0 < facts.lower_region_containment_ratio < 1.0


def test_j_object_outside_lower_region():
    person = BBox(0.2, 0.2, 0.4, 0.4)
    obj = BBox(0.8, 0.8, 0.05, 0.05)
    facts = resolve_bbox(person=person, obj=obj, spec=SPEC)
    assert facts.lower_region_containment_ratio == 0.0
    assert facts.lower_region_intersection_area == 0.0
    assert facts.center_inside_lower_person_region is False


def test_k_zero_area_object_safe():
    person = BBox(0.2, 0.2, 0.4, 0.4)
    obj = BBox(0.3, 0.5, 0.0, 0.0)
    facts = resolve_bbox(person=person, obj=obj, spec=SPEC)
    assert facts.available is False
    assert facts.center_inside_lower_person_region is None
    assert facts.configured_region_center_membership is None
    assert facts.lower_region_containment_ratio is None
    assert facts.lower_region_intersection_area == 0.0
    assert facts.reason == "zero_area_object"



def test_center_membership_and_overlap_are_separate():
    person = BBox(0.2, 0.2, 0.4, 0.4)
    obj = BBox(0.3, 0.44, 0.05, 0.04)  # center 0.46 boundary-ish, small overlap
    facts = resolve_bbox(person=person, obj=obj, spec=SPEC)
    assert facts.center_inside_lower_person_region is not None
    assert facts.lower_region_containment_ratio is not None


# --- camera polygon --------------------------------------------------------

SQUARE_CCW = NormalizedPolygon(((0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)))
SQUARE_CW = NormalizedPolygon(((0.2, 0.2), (0.2, 0.6), (0.6, 0.6), (0.6, 0.2)))
CONCAVE = NormalizedPolygon(
    ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.5, 0.4), (0.1, 0.9)),
)


def test_l_point_inside_polygon():
    assert point_in_polygon(SQUARE_CCW, (0.4, 0.4)) is True


def test_m_point_outside_polygon():
    assert point_in_polygon(SQUARE_CCW, (0.05, 0.4)) is False


def test_n_point_on_edge_is_inside():
    assert point_in_polygon(SQUARE_CCW, (0.2, 0.4)) is True
    assert point_in_polygon(SQUARE_CCW, (0.2, 0.2)) is True


def test_o_orientation_independent():
    for point in [(0.4, 0.4), (0.05, 0.4), (0.6, 0.3)]:
        assert point_in_polygon(SQUARE_CCW, point) == point_in_polygon(SQUARE_CW, point)


def test_p_concave_polygon():
    assert point_in_polygon(CONCAVE, (0.5, 0.2)) is True
    assert point_in_polygon(CONCAVE, (0.5, 0.8)) is False


def test_q_too_few_distinct_points_rejected():
    with pytest.raises(RegionConfigError):
        NormalizedPolygon(((0.1, 0.1), (0.2, 0.2)))
    with pytest.raises(RegionConfigError):
        NormalizedPolygon(((0.1, 0.1), (0.1, 0.1), (0.2, 0.2)))


def test_r_collinear_polygon_rejected():
    with pytest.raises(RegionConfigError):
        NormalizedPolygon(((0.1, 0.1), (0.2, 0.2), (0.3, 0.3)))


def test_s_invalid_coordinates_rejected():
    with pytest.raises(RegionConfigError):
        NormalizedPolygon(((0.1, 0.1), (float("nan"), 0.2), (0.3, 0.5)))
    with pytest.raises(RegionConfigError):
        NormalizedPolygon(((0.1, 0.1), (float("inf"), 0.2), (0.3, 0.5)))
    with pytest.raises(RegionConfigError):
        NormalizedPolygon(((0.1, 0.1), (1.4, 0.2), (0.3, 0.5)))


def test_polygon_touching_frame_borders():
    polygon = NormalizedPolygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    assert point_in_polygon(polygon, (0.0, 0.0)) is True
    assert point_in_polygon(polygon, (0.5, 0.5)) is True


# --- evidence source separation -------------------------------------------

ZONE = (ConfiguredRegion(polygon=SQUARE_CCW, region_id="z1", label="operator-defined"),)


def test_t_inside_lower_region_outside_configured_region():
    person = BBox(0.7, 0.2, 0.2, 0.6)
    facts = resolve_point(person=person, point=(0.8, 0.7), spec=SPEC, configured_regions=ZONE)
    assert facts.inside_lower_person_region is True
    assert facts.configured_region_membership is False


def test_u_outside_lower_region_inside_configured_region():
    person = BBox(0.3, 0.3, 0.2, 0.6)
    facts = resolve_point(person=person, point=(0.4, 0.35), spec=SPEC, configured_regions=ZONE)
    assert facts.inside_lower_person_region is False
    assert facts.configured_region_membership is True


def test_v_no_configured_region_is_none():
    person = BBox(0.3, 0.3, 0.2, 0.6)
    facts = resolve_point(person=person, point=(0.4, 0.8), spec=SPEC)
    assert facts.configured_region_membership is None
    empty = resolve_point(person=person, point=(0.4, 0.8), spec=SPEC, configured_regions=())
    assert empty.configured_region_membership is None


def test_resolution_invariance_across_frame_sizes():
    """Same normalized geometry -> same facts regardless of pixel resolution."""
    person = BBox(0.25, 0.25, 0.2, 0.5)
    point = (0.3, 0.7)
    facts = resolve_point(person=person, point=point, spec=SPEC)
    for width, height in ((640, 480), (1920, 1080), (2560, 1440)):
        px = person.to_pixels(width, height)
        rebuilt = BBox(px[0] / width, px[1] / height, (px[2] - px[0]) / width, (px[3] - px[1]) / height)
        again = resolve_point(person=rebuilt, point=point, spec=SPEC)
        assert again.inside_lower_person_region == facts.inside_lower_person_region


# --- final geometry safety hardening --------------------------------------


def test_a_zero_area_object_inside_lower_region_is_unavailable():
    person = BBox(0.2, 0.2, 0.4, 0.4)  # lower region y: 0.46..0.6
    obj = BBox(0.35, 0.55, 0.0, 0.0)
    facts = resolve_bbox(person=person, obj=obj, spec=SPEC, configured_regions=ZONE)
    assert facts.available is False
    assert facts.center_inside_lower_person_region is None
    assert facts.center_inside_person is None
    assert facts.configured_region_center_membership is None
    assert facts.lower_region_containment_ratio is None


@pytest.mark.parametrize(
    "obj",
    [
        BBox(0.3, 0.5, -0.1, 0.05),
        BBox(0.3, 0.5, 0.05, -0.1),
        BBox(0.3, 0.5, float("nan"), 0.05),
    ],
)
def test_b_invalid_extent_object_unavailable(obj):
    person = BBox(0.2, 0.2, 0.4, 0.4)
    facts = resolve_bbox(person=person, obj=obj, spec=SPEC)
    assert facts.available is False
    assert facts.center_inside_lower_person_region is None
    assert facts.reason in {"negative_extent_object_box", "invalid_object_box"}


@pytest.mark.parametrize(
    "person",
    [
        BBox(-0.1, 0.2, 0.3, 0.4),
        BBox(0.2, -0.05, 0.3, 0.4),
        BBox(0.8, 0.2, 0.4, 0.4),
        BBox(0.2, 0.8, 0.3, 0.5),
    ],
)
def test_c_person_outside_frame_unavailable(person):
    assert is_person_box_usable(person) is False
    facts = resolve_point(person=person, point=(0.3, 0.5), spec=SPEC)
    assert facts.available is False
    assert facts.reason == "degenerate_person_box"


def test_d_object_extending_outside_frame_unavailable():
    person = BBox(0.2, 0.2, 0.4, 0.4)
    facts = resolve_bbox(person=person, obj=BBox(0.9, 0.5, 0.3, 0.1), spec=SPEC)
    assert facts.available is False
    assert facts.reason == "object_box_outside_frame"
    assert facts.center_inside_lower_person_region is None


@pytest.mark.parametrize("point", [(-0.01, 0.5), (1.01, 0.5), (0.5, -0.01), (0.5, 1.01)])
def test_e_point_outside_frame_unavailable(point):
    person = BBox(0.2, 0.2, 0.4, 0.4)
    facts = resolve_point(person=person, point=point, spec=SPEC)
    assert facts.available is False
    assert facts.reason == "invalid_point"
    assert facts.inside_person is None


def test_f_valid_frame_point_outside_person_is_available():
    person = BBox(0.2, 0.2, 0.2, 0.2)
    facts = resolve_point(person=person, point=(0.9, 0.9), spec=SPEC)
    assert facts.available is True
    assert facts.inside_person is False
    assert facts.inside_lower_person_region is False


def test_g_person_region_spec_requires_explicit_lower_start():
    with pytest.raises(TypeError):
        PersonRegionSpec()  # type: ignore[call-arg]
    assert PersonRegionSpec(lower_start_fraction=0.65).lower_end_fraction == 1.0


REGION_A = ConfiguredRegion(
    polygon=NormalizedPolygon(((0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6))),
    region_id="A",
    label="a",
)
REGION_B = ConfiguredRegion(
    polygon=NormalizedPolygon(((0.7, 0.7), (0.95, 0.7), (0.95, 0.95), (0.7, 0.95))),
    region_id="B",
    label="b",
)
REGION_C = ConfiguredRegion(
    polygon=NormalizedPolygon(((0.3, 0.3), (0.9, 0.3), (0.9, 0.9), (0.3, 0.9))),
    region_id="C",
    label="c",
)


def test_h_three_regions_preserve_individual_membership():
    person = BBox(0.3, 0.3, 0.3, 0.4)
    facts = resolve_point(
        person=person,
        point=(0.4, 0.4),
        spec=SPEC,
        configured_regions=(REGION_A, REGION_B, REGION_C),
    )
    memberships = facts.configured_region_memberships
    assert memberships is not None
    result = {item.region_id: item.inside for item in memberships}
    assert result == {"A": True, "B": False, "C": True}
    assert facts.matched_configured_region_ids == ("A", "C")
    assert facts.configured_region_membership is True
    assert [item.label for item in memberships] == ["a", "b", "c"]


def test_i_no_config_differs_from_configured_no_match():
    person = BBox(0.05, 0.05, 0.1, 0.1)
    point = (0.1, 0.1)
    unconfigured = resolve_point(person=person, point=point, spec=SPEC)
    assert unconfigured.configured_regions_configured is False
    assert unconfigured.configured_region_memberships is None
    assert unconfigured.configured_region_membership is None

    configured = resolve_point(
        person=person, point=point, spec=SPEC, configured_regions=(REGION_A, REGION_B)
    )
    assert configured.configured_regions_configured is True
    assert configured.configured_region_memberships is not None
    assert all(item.inside is False for item in configured.configured_region_memberships)
    assert configured.configured_region_membership is False
    assert configured.matched_configured_region_ids == ()


def test_j_duplicate_region_ids_rejected():
    duplicate = ConfiguredRegion(polygon=REGION_C.polygon, region_id="A", label="other")
    with pytest.raises(RegionConfigError):
        validate_configured_regions((REGION_A, duplicate))
    with pytest.raises(RegionConfigError):
        resolve_point(
            person=BBox(0.3, 0.3, 0.2, 0.4),
            point=(0.4, 0.4),
            spec=SPEC,
            configured_regions=(REGION_A, duplicate),
        )
    # Null region_ids are allowed to repeat, and labels need not be unique.
    anonymous = (
        ConfiguredRegion(polygon=REGION_A.polygon, label="same"),
        ConfiguredRegion(polygon=REGION_B.polygon, label="same"),
    )
    assert validate_configured_regions(anonymous) is not None


def test_k_self_intersecting_polygon_rejected():
    with pytest.raises(RegionConfigError):  # bow-tie
        NormalizedPolygon(((0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)))
    with pytest.raises(RegionConfigError):  # crossing star-ish loop
        NormalizedPolygon(((0.2, 0.2), (0.8, 0.2), (0.2, 0.5), (0.8, 0.5)))


def test_l_valid_concave_polygon_accepted():
    polygon = NormalizedPolygon(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.5, 0.4), (0.1, 0.9)))
    assert polygon.area > 0.0
    assert point_in_polygon(polygon, (0.5, 0.2)) is True
    assert point_in_polygon(polygon, (0.5, 0.8)) is False


def test_repeated_closing_vertex_rejected():
    with pytest.raises(RegionConfigError):
        NormalizedPolygon(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.1)))
    with pytest.raises(RegionConfigError):
        NormalizedPolygon(((0.1, 0.1), (0.9, 0.1), (0.9, 0.1), (0.9, 0.9)))


def test_non_adjacent_edges_touching_vertex_rejected():
    with pytest.raises(RegionConfigError):
        NormalizedPolygon(((0.1, 0.1), (0.5, 0.5), (0.9, 0.1), (0.9, 0.9), (0.5, 0.5), (0.1, 0.9)))


def test_boundary_membership_convention_preserved():
    polygon = NormalizedPolygon(((0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)))
    assert point_in_polygon(polygon, (0.2, 0.4)) is True
    assert point_in_polygon(polygon, (0.6, 0.6)) is True
