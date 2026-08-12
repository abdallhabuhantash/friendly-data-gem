"""Crop <-> full-frame transform tests (Task 3E)."""

from __future__ import annotations

import dataclasses

import pytest

from app.ai.crop_transform import CropTransform, CropTransformError
from app.domain.geometry import BBox


def box(x1: float, y1: float, x2: float, y2: float) -> BBox:
    return BBox(x1, y1, x2 - x1, y2 - y1)


def test_full_frame_crop_is_identity_mapping() -> None:
    crop = CropTransform.full_frame()
    assert crop.is_identity is True
    original = box(0.2, 0.3, 0.5, 0.6)
    mapped = crop.to_full_frame(original)
    assert mapped.x == pytest.approx(original.x)
    assert mapped.y == pytest.approx(original.y)
    assert mapped.x2 == pytest.approx(original.x2)
    assert mapped.y2 == pytest.approx(original.y2)


def test_sub_crop_maps_to_full_frame_coordinates() -> None:
    crop = CropTransform(0.4, 0.2, 0.8, 0.6)
    mapped = crop.to_full_frame(box(0.5, 0.5, 1.0, 1.0))
    assert mapped.x == pytest.approx(0.6)
    assert mapped.y == pytest.approx(0.4)
    assert mapped.x2 == pytest.approx(0.8)
    assert mapped.y2 == pytest.approx(0.6)


def test_border_touching_crop_relative_box_stays_border_touching() -> None:
    crop = CropTransform(0.1, 0.1, 0.5, 0.5)
    mapped = crop.to_full_frame(box(0.0, 0.0, 1.0, 1.0))
    assert (mapped.x, mapped.y) == pytest.approx((0.1, 0.1))
    assert (mapped.x2, mapped.y2) == pytest.approx((0.5, 0.5))


@pytest.mark.parametrize(
    "coords",
    [
        (0.5, 0.1, 0.5, 0.4),  # zero width
        (0.1, 0.4, 0.4, 0.4),  # zero height
        (0.6, 0.1, 0.2, 0.4),  # reversed x
        (0.1, 0.6, 0.4, 0.2),  # reversed y
        (-0.1, 0.1, 0.4, 0.4),  # out of range
        (0.1, 0.1, 1.4, 0.4),  # out of range
        (float("nan"), 0.1, 0.4, 0.4),
        (0.1, 0.1, float("inf"), 0.4),
    ],
)
def test_invalid_crop_rejected(coords) -> None:
    with pytest.raises(CropTransformError):
        CropTransform(*coords)


@pytest.mark.parametrize(
    "bbox",
    [
        BBox(0.2, 0.2, 0.0, 0.3),  # zero area
        BBox(0.6, 0.2, -0.3, 0.3),  # reversed
        BBox(-0.2, 0.2, 0.3, 0.3),  # out of range
        BBox(0.8, 0.2, 0.5, 0.3),  # x2 > 1
        BBox(0.2, 0.2, float("nan"), 0.3),
    ],
)
def test_crop_relative_invalid_bbox_rejected(bbox: BBox) -> None:
    crop = CropTransform(0.1, 0.1, 0.9, 0.9)
    with pytest.raises(CropTransformError):
        crop.to_full_frame(bbox)


def test_no_substantive_out_of_frame_clamping() -> None:
    crop = CropTransform(0.5, 0.5, 1.0, 1.0)
    # A crop-relative box beyond 1.0 is invalid input, not something to clamp.
    with pytest.raises(CropTransformError):
        crop.to_full_frame(BBox(0.5, 0.5, 0.9, 0.9))


def test_mapping_is_deterministic_and_round_trips() -> None:
    crop = CropTransform(0.25, 0.125, 0.75, 0.625)
    original = box(0.2, 0.4, 0.9, 0.95)
    first = crop.to_full_frame(original)
    second = crop.to_full_frame(original)
    assert first == second
    back = crop.to_crop_relative(first)
    assert back.x == pytest.approx(original.x)
    assert back.y == pytest.approx(original.y)
    assert back.x2 == pytest.approx(original.x2)
    assert back.y2 == pytest.approx(original.y2)


def test_transform_is_immutable_and_serializable() -> None:
    crop = CropTransform(0.1, 0.2, 0.3, 0.4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        crop.x1 = 0.0  # type: ignore[misc]
    assert crop.to_dict() == {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}
    assert crop.width == pytest.approx(0.2)
    assert crop.height == pytest.approx(0.2)


def test_inverse_mapping_rejects_box_outside_crop() -> None:
    crop = CropTransform(0.4, 0.4, 0.6, 0.6)
    with pytest.raises(CropTransformError):
        crop.to_crop_relative(box(0.1, 0.1, 0.2, 0.2))
