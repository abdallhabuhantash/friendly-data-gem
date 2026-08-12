"""Deterministic pure tests for the wrist/arm feature layer (geometry only)."""

from __future__ import annotations

import ast
import copy
import dataclasses
import math
from pathlib import Path

import pytest

from app.ai.wrist_arm_feature_builder import build_wrist_arm_features
from app.domain.body_features import BodySide, TrackedBodyFeatures
from app.domain.geometry import BBox
from app.domain.pose import COCO_17_KEYPOINTS, PoseKeypointName, coco_17_index
from app.domain.regions import RelativePoint
from app.domain.tracked_pose_observations import (
    TrackedPoseKeypoint,
    TrackedPoseObservation,
)

PERSON_BOX = BBox(0.2, 0.2, 0.2, 0.4)


def relative(point: tuple[float, float], box: BBox) -> RelativePoint:
    return RelativePoint(
        relative_x=(point[0] - box.x) / box.width,
        relative_y=(point[1] - box.y) / box.height,
    )


def observation(
    points: dict[PoseKeypointName, tuple[float, float]],
    box: BBox = PERSON_BOX,
    tracking_id: str = "track-1",
) -> TrackedPoseObservation:
    keypoints = []
    for name in COCO_17_KEYPOINTS:
        index = coco_17_index(name)
        if name in points:
            x, y = points[name]
            rel = relative((x, y), box)
            keypoints.append(
                TrackedPoseKeypoint(
                    name=name,
                    index=index,
                    available=True,
                    confidence=0.8,
                    x=x,
                    y=y,
                    relative_position=rel,
                    inside_person=rel.inside_person,
                )
            )
        else:
            keypoints.append(
                TrackedPoseKeypoint(name=name, index=index, available=False)
            )
    return TrackedPoseObservation(
        person_tracking_id=tracking_id,
        person_index=0,
        pose_index=0,
        person_bbox=box,
        person_confidence=0.7,
        pose_bbox=box,
        pose_confidence=0.9,
        keypoints=tuple(keypoints),
    )


LEFT_CHAIN = {
    PoseKeypointName.LEFT_SHOULDER: (0.25, 0.30),
    PoseKeypointName.LEFT_ELBOW: (0.25, 0.40),
    PoseKeypointName.LEFT_WRIST: (0.35, 0.40),
}
RIGHT_CHAIN = {
    PoseKeypointName.RIGHT_SHOULDER: (0.35, 0.30),
    PoseKeypointName.RIGHT_ELBOW: (0.35, 0.45),
    PoseKeypointName.RIGHT_WRIST: (0.35, 0.55),
}


def test_full_left_arm_geometry() -> None:
    features = build_wrist_arm_features(observation(LEFT_CHAIN))
    left = features.left_arm
    assert left.availability.full_chain_available is True
    assert left.wrist.available is True
    assert left.wrist.x == pytest.approx(0.35)
    assert left.elbow_to_shoulder_distance == pytest.approx(0.10)
    assert left.wrist_to_elbow_distance == pytest.approx(0.10)
    assert left.shoulder_to_wrist_distance == pytest.approx(math.hypot(0.10, 0.10))
    assert left.elbow_angle_degrees == pytest.approx(90.0)
    assert left.shoulder_wrist_to_segment_sum_ratio == pytest.approx(
        math.hypot(0.10, 0.10) / 0.20
    )
    assert left.shoulder_to_wrist_distance_relative_to_person == pytest.approx(
        math.hypot(0.10, 0.10) / PERSON_BOX.diagonal
    )


def test_full_right_arm_geometry_straight_chain() -> None:
    right = build_wrist_arm_features(observation(RIGHT_CHAIN)).right_arm
    assert right.availability.full_chain_available is True
    assert right.shoulder_to_wrist_distance == pytest.approx(0.25)
    assert right.elbow_angle_degrees == pytest.approx(180.0)
    assert right.shoulder_wrist_to_segment_sum_ratio == pytest.approx(1.0)


def test_sides_are_independent() -> None:
    features = build_wrist_arm_features(observation({**LEFT_CHAIN, **RIGHT_CHAIN}))
    assert features.left_arm.elbow_angle_degrees == pytest.approx(90.0)
    assert features.right_arm.elbow_angle_degrees == pytest.approx(180.0)
    assert (
        features.left_arm.shoulder_to_wrist_distance
        != features.right_arm.shoulder_to_wrist_distance
    )


def test_missing_side_does_not_borrow_from_other_side() -> None:
    features = build_wrist_arm_features(observation(LEFT_CHAIN))
    right = features.right_arm
    assert right.availability.available_joint_count == 0
    assert right.wrist.available is False
    assert right.shoulder_to_wrist_distance is None
    assert right.elbow_angle_degrees is None
    assert features.left_arm.shoulder_to_wrist_distance is not None


def test_wrist_unavailable_makes_wrist_facts_unavailable() -> None:
    points = dict(LEFT_CHAIN)
    del points[PoseKeypointName.LEFT_WRIST]
    left = build_wrist_arm_features(observation(points)).left_arm
    assert left.wrist.available is False
    assert left.wrist.x is None and left.wrist.y is None
    assert left.wrist.relative_position is None
    assert left.wrist_to_elbow_distance is None
    assert left.shoulder_to_wrist_distance is None
    assert left.elbow_angle_degrees is None
    assert left.elbow_to_shoulder_distance == pytest.approx(0.10)


def test_elbow_unavailable_makes_elbow_angle_unavailable() -> None:
    points = dict(LEFT_CHAIN)
    del points[PoseKeypointName.LEFT_ELBOW]
    left = build_wrist_arm_features(observation(points)).left_arm
    assert left.elbow_angle_degrees is None
    assert left.wrist_to_elbow_distance is None
    assert left.elbow_to_shoulder_distance is None
    assert left.shoulder_to_wrist_distance is not None
    assert left.shoulder_wrist_to_segment_sum_ratio is None


def test_shoulder_unavailable_makes_shoulder_facts_unavailable() -> None:
    points = dict(LEFT_CHAIN)
    del points[PoseKeypointName.LEFT_SHOULDER]
    left = build_wrist_arm_features(observation(points)).left_arm
    assert left.elbow_to_shoulder_distance is None
    assert left.shoulder_to_wrist_distance is None
    assert left.shoulder_to_wrist_distance_relative_to_person is None
    assert left.elbow_angle_degrees is None
    assert left.shoulder_confidence is None
    assert left.wrist_to_elbow_distance == pytest.approx(0.10)


def test_missing_keypoints_never_become_zero_zero() -> None:
    features = build_wrist_arm_features(observation({}))
    for arm in (features.left_arm, features.right_arm):
        assert arm.wrist.x is None and arm.wrist.y is None
        assert arm.wrist.relative_position is None
        assert arm.wrist.inside_person is None


def test_relative_coordinates_outside_unit_range_are_not_clamped() -> None:
    points = {**LEFT_CHAIN, PoseKeypointName.LEFT_WRIST: (0.05, 0.70)}
    left = build_wrist_arm_features(observation(points)).left_arm
    assert left.wrist.available is True
    assert left.wrist.relative_position is not None
    assert left.wrist.relative_position.relative_x < 0.0
    assert left.wrist.relative_position.relative_y > 1.0
    assert left.wrist.inside_person is False


def test_distances_are_deterministic_and_finite() -> None:
    first = build_wrist_arm_features(observation({**LEFT_CHAIN, **RIGHT_CHAIN}))
    second = build_wrist_arm_features(observation({**LEFT_CHAIN, **RIGHT_CHAIN}))
    assert first == second
    for arm in (first.left_arm, first.right_arm):
        for value in (
            arm.wrist_to_elbow_distance,
            arm.elbow_to_shoulder_distance,
            arm.shoulder_to_wrist_distance,
            arm.shoulder_wrist_to_segment_sum_ratio,
            arm.elbow_angle_degrees,
        ):
            assert value is not None and math.isfinite(value)


def test_degenerate_zero_length_segment_is_handled_safely() -> None:
    points = {
        PoseKeypointName.LEFT_SHOULDER: (0.25, 0.30),
        PoseKeypointName.LEFT_ELBOW: (0.25, 0.30),
        PoseKeypointName.LEFT_WRIST: (0.25, 0.30),
    }
    left = build_wrist_arm_features(observation(points)).left_arm
    assert left.wrist_to_elbow_distance == pytest.approx(0.0)
    assert left.shoulder_to_wrist_distance == pytest.approx(0.0)
    assert left.elbow_angle_degrees is None
    assert left.shoulder_wrist_to_segment_sum_ratio is None


def test_builder_uses_semantic_keypoint_lookup_only() -> None:
    source = Path("app/ai/wrist_arm_feature_builder.py").read_text(encoding="utf-8")
    assert "available_keypoint" in source
    for index in range(17):
        assert f"keypoints[{index}]" not in source


def test_input_observation_is_not_mutated() -> None:
    subject = observation({**LEFT_CHAIN, **RIGHT_CHAIN})
    snapshot = copy.deepcopy(subject)
    build_wrist_arm_features(subject)
    assert subject == snapshot


def test_new_domain_objects_are_immutable() -> None:
    features = build_wrist_arm_features(observation(LEFT_CHAIN))
    assert isinstance(features, TrackedBodyFeatures)
    with pytest.raises(dataclasses.FrozenInstanceError):
        features.person_tracking_id = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        features.left_arm.elbow_angle_degrees = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        features.left_arm.wrist.x = 0.5  # type: ignore[misc]


def test_no_temporal_or_global_state_in_feature_layer() -> None:
    for path in (
        "app/ai/wrist_arm_feature_builder.py",
        "app/domain/body_features.py",
    ):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            assert not isinstance(node, ast.Global), path
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in {
                        "time",
                        "datetime",
                        "threading",
                        "random",
                    }, path
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {
                    "time",
                    "datetime",
                    "threading",
                    "random",
                }, path


def test_equivalent_normalized_pose_in_two_person_sizes_matches_relative_geometry() -> None:
    small = BBox(0.10, 0.10, 0.10, 0.20)
    large = BBox(0.40, 0.30, 0.20, 0.40)

    def scaled(box: BBox) -> dict[PoseKeypointName, tuple[float, float]]:
        fractions = {
            PoseKeypointName.LEFT_SHOULDER: (0.25, 0.25),
            PoseKeypointName.LEFT_ELBOW: (0.25, 0.50),
            PoseKeypointName.LEFT_WRIST: (0.75, 0.50),
        }
        return {
            name: (box.x + fx * box.width, box.y + fy * box.height)
            for name, (fx, fy) in fractions.items()
        }

    a = build_wrist_arm_features(observation(scaled(small), small)).left_arm
    b = build_wrist_arm_features(observation(scaled(large), large)).left_arm

    assert a.shoulder_to_wrist_distance_relative_to_person == pytest.approx(
        b.shoulder_to_wrist_distance_relative_to_person
    )
    assert a.shoulder_wrist_to_segment_sum_ratio == pytest.approx(
        b.shoulder_wrist_to_segment_sum_ratio
    )
    assert a.elbow_angle_degrees == pytest.approx(b.elbow_angle_degrees)
    assert a.wrist.relative_position == b.wrist.relative_position
    assert a.shoulder_to_wrist_distance != pytest.approx(b.shoulder_to_wrist_distance)


def test_confidences_are_preserved_separately_and_never_fused() -> None:
    features = build_wrist_arm_features(observation(LEFT_CHAIN))
    assert features.person_confidence == pytest.approx(0.7)
    assert features.pose_confidence == pytest.approx(0.9)
    assert features.left_arm.wrist.keypoint_confidence == pytest.approx(0.8)
    assert features.left_arm.shoulder_confidence == pytest.approx(0.8)
    assert features.left_arm.elbow_confidence == pytest.approx(0.8)


def test_no_behaviour_vocabulary_in_output_fields() -> None:
    forbidden = (
        "handing_paper",
        "reaching_other_person",
        "suspicious",
        "cheating",
        "exchange",
        "concealed",
        "wrist_low",
        "head_down",
        "arm_extended",
        "grasp",
        "palm",
        "finger",
        "hand_center",
    )
    names: list[str] = []
    features = build_wrist_arm_features(observation(LEFT_CHAIN))
    for obj in (
        features,
        features.left_arm,
        features.right_arm,
        features.left_arm.wrist,
        features.left_arm.availability,
    ):
        names.extend(field.name for field in dataclasses.fields(obj))
    joined = " ".join(names)
    for word in forbidden:
        assert word not in joined


def test_no_runtime_module_imports_the_feature_layer() -> None:
    roots = [Path("app/runtime"), Path("app/camera"), Path("app/events")]
    roots += [
        Path("app/ai/phone_rule_engine.py"),
        Path("app/ai/engine_registry.py"),
        Path("app/ai/detector.py"),
        Path("app/ai/pose_runtime.py") if Path("app/ai/pose_runtime.py").exists() else Path("app/ai/detector.py"),
    ]
    files: list[Path] = []
    for root in roots:
        files.extend(root.rglob("*.py") if root.is_dir() else [root])
    for path in files:
        source = path.read_text(encoding="utf-8")
        assert "wrist_arm_feature_builder" not in source, path
        assert "body_features" not in source, path
