"""Pose domain + provider tests (no real YOLO weights, no downloads)."""

from __future__ import annotations

import dataclasses
import math
import threading

import pytest

from app.ai.pose_provider import UltralyticsPoseProvider, parse_pose_result
from app.domain.pose import (
    COCO_17_KEYPOINT_COUNT,
    PoseFrameResult,
    PoseKeypointName,
    PoseStatus,
    coco_17_index,
)


class FakeSeq:
    """Stands in for an Ultralytics tensor: exposes ``tolist()``."""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class FakeBoxes:
    def __init__(self, xyxyn, conf=None):
        self.xyxyn = FakeSeq(xyxyn)
        self.conf = FakeSeq(conf) if conf is not None else None


class FakeKeypoints:
    def __init__(self, xyn, conf=None):
        self.xyn = FakeSeq(xyn)
        self.conf = FakeSeq(conf) if conf is not None else None


class FakeResult:
    def __init__(self, boxes=None, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


def points(count=COCO_17_KEYPOINT_COUNT, value=(0.5, 0.5)):
    return [list(value) for _ in range(count)]


def confs(count=COCO_17_KEYPOINT_COUNT, value=0.9):
    return [value] * count


def one_person_result(kp=None, kp_conf=None, box=(0.1, 0.1, 0.4, 0.9), box_conf=0.8):
    return FakeResult(
        boxes=FakeBoxes([list(box)], [box_conf]),
        keypoints=FakeKeypoints([kp or points()], [kp_conf or confs()]),
    )


# --- domain tests ---------------------------------------------------------


def test_a_seventeen_valid_keypoints_have_semantic_names():
    result = parse_pose_result(one_person_result())
    instance = result.instances[0]
    assert result.status is PoseStatus.OK
    assert len(instance.keypoints) == 17
    assert [kp.name for kp in instance.keypoints][0] == PoseKeypointName.NOSE
    assert all(kp.available for kp in instance.keypoints)


def test_b_left_wrist_is_index_nine():
    assert coco_17_index(PoseKeypointName.LEFT_WRIST) == 9
    instance = parse_pose_result(one_person_result()).instances[0]
    assert instance.keypoint(PoseKeypointName.LEFT_WRIST).index == 9


def test_c_right_wrist_is_index_ten():
    assert coco_17_index(PoseKeypointName.RIGHT_WRIST) == 10


def test_d_hips_are_eleven_and_twelve():
    assert coco_17_index(PoseKeypointName.LEFT_HIP) == 11
    assert coco_17_index(PoseKeypointName.RIGHT_HIP) == 12


def test_e_pose_domain_is_immutable():
    result = parse_pose_result(one_person_result())
    instance = result.instances[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.confidence = 0.1
    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.keypoints[0].x = 0.2
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = PoseStatus.INFERENCE_FAILED


def test_f_unavailable_lookup_is_safe():
    kp_conf = confs()
    kp_conf[9] = 0.1
    instance = parse_pose_result(one_person_result(kp_conf=kp_conf)).instances[0]
    assert instance.keypoint(PoseKeypointName.LEFT_WRIST).available is False
    assert instance.available_keypoint(PoseKeypointName.LEFT_WRIST) is None
    assert instance.available_keypoint_count == 16


# --- low confidence -------------------------------------------------------


def test_g_masked_zero_zero_keypoint_is_unavailable():
    kp = points()
    kp[9] = [0.0, 0.0]
    kp_conf = confs()
    kp_conf[9] = 0.31
    wrist = parse_pose_result(one_person_result(kp=kp, kp_conf=kp_conf)).instances[0].keypoint(
        PoseKeypointName.LEFT_WRIST
    )
    assert wrist.available is False
    assert wrist.x is None and wrist.y is None
    assert wrist.confidence == pytest.approx(0.31)


def test_h_valid_point_near_origin_stays_available():
    kp = points()
    kp[9] = [0.001, 0.002]
    wrist = parse_pose_result(one_person_result(kp=kp)).instances[0].keypoint(
        PoseKeypointName.LEFT_WRIST
    )
    assert wrist.available is True
    assert wrist.x == pytest.approx(0.001)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.4, -0.2])
def test_i_malformed_confidence_is_never_positive(bad):
    kp_conf = confs()
    kp_conf[10] = bad
    wrist = parse_pose_result(one_person_result(kp_conf=kp_conf)).instances[0].keypoint(
        PoseKeypointName.RIGHT_WRIST
    )
    assert wrist.available is False
    assert wrist.confidence is None
    assert wrist.x is None


# --- result parsing -------------------------------------------------------


def test_j_one_valid_pose_result_yields_one_instance():
    result = parse_pose_result(one_person_result())
    assert result.ok and result.instance_count == 1
    assert result.instances[0].bbox.width == pytest.approx(0.3)


def test_k_two_people_yield_two_instances():
    result = parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([[0.1, 0.1, 0.3, 0.9], [0.5, 0.1, 0.8, 0.9]], [0.8, 0.7]),
            keypoints=FakeKeypoints([points(), points()], [confs(), confs()]),
        )
    )
    assert result.ok and result.instance_count == 2


def test_l_zero_persons_is_success_with_no_instances():
    result = parse_pose_result(FakeResult(boxes=FakeBoxes([]), keypoints=FakeKeypoints([])))
    assert result.status is PoseStatus.OK
    assert result.instances == ()


def test_m_length_mismatch_is_malformed_not_truncated():
    result = parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([[0.1, 0.1, 0.3, 0.9], [0.4, 0.1, 0.6, 0.9]]),
            keypoints=FakeKeypoints([points()], [confs()]),
        )
    )
    assert result.status is PoseStatus.MALFORMED_RESULT
    assert result.instances == ()


def test_n_non_coco_keypoint_count_is_unsupported_schema():
    result = parse_pose_result(one_person_result(kp=points(5), kp_conf=confs(5)))
    assert result.status is PoseStatus.UNSUPPORTED_POSE_SCHEMA
    assert result.instances == ()


def test_o_malformed_pose_bbox_rejects_only_that_instance():
    result = parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([[0.2, 0.2, 0.2, 0.2], [0.5, 0.1, 0.8, 0.9]]),
            keypoints=FakeKeypoints([points(), points()], [confs(), confs()]),
        )
    )
    assert result.ok and result.instance_count == 1
    assert result.instances[0].bbox.x == pytest.approx(0.5)


def test_o2_out_of_frame_bbox_is_rejected():
    result = parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([[0.1, 0.1, 1.4, 0.9]]),
            keypoints=FakeKeypoints([points()], [confs()]),
        )
    )
    assert result.ok and result.instances == ()


def test_p_missing_keypoints_is_explicit_status():
    result = parse_pose_result(FakeResult(boxes=FakeBoxes([[0.1, 0.1, 0.3, 0.9]]), keypoints=None))
    assert result.status is PoseStatus.KEYPOINTS_ABSENT


def test_absent_keypoint_confidence_stays_none():
    result = parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([[0.1, 0.1, 0.3, 0.9]]),
            keypoints=FakeKeypoints([points()]),
        )
    )
    keypoint = result.instances[0].keypoint(PoseKeypointName.NOSE)
    assert keypoint.available is True and keypoint.confidence is None


# --- provider failure -----------------------------------------------------


class FakeModel:
    def __init__(self, results=None, error=None, hook=None):
        self._results = results if results is not None else [one_person_result()]
        self._error = error
        self._hook = hook
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        if self._hook is not None:
            self._hook()
        if self._error is not None:
            raise self._error
        return self._results


def test_q_inference_exception_is_reported_cleanly():
    provider = UltralyticsPoseProvider(
        "fake-pose.pt", model_factory=lambda _: FakeModel(error=RuntimeError("cuda oom"))
    )
    result = provider.infer(object())
    assert result.status is PoseStatus.INFERENCE_FAILED
    assert "cuda oom" in (result.reason or "")
    assert result.instances == ()


def test_r_model_load_failure_marks_provider_unavailable():
    def broken(_name):
        raise FileNotFoundError("weights missing")

    provider = UltralyticsPoseProvider("missing.pt", model_factory=broken)
    assert provider.initialize() is False
    assert provider.available is False
    result = provider.infer(object())
    assert result.status is PoseStatus.MODEL_UNAVAILABLE


def test_s_construction_does_not_load_the_model():
    loads = []
    provider = UltralyticsPoseProvider(
        "fake-pose.pt", model_factory=lambda name: loads.append(name) or FakeModel()
    )
    assert loads == []
    provider.infer(object())
    assert loads == ["fake-pose.pt"]


def test_provider_never_uses_tracking():
    source = (__file__.rsplit("tests", 1)[0]) + "app/ai/pose_provider.py"
    with open(source, encoding="utf-8") as handle:
        assert ".track(" not in handle.read()


def test_t_concurrent_infer_calls_are_serialized():
    barrier = threading.Barrier(2, timeout=0.2)
    overlapped = []

    def hook():
        try:
            barrier.wait()
            overlapped.append(True)
        except threading.BrokenBarrierError:
            overlapped.append(False)

    provider = UltralyticsPoseProvider(
        "fake-pose.pt", model_factory=lambda _: FakeModel(hook=hook)
    )
    provider.initialize()
    threads = [threading.Thread(target=provider.infer, args=(object(),)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert overlapped == [False, False]  # never inside predict() at the same time


def test_no_behavioural_features_leak_into_domain():
    instance = parse_pose_result(one_person_result()).instances[0]
    for forbidden in ("person_tracking_id", "head_is_down", "wrist_below", "behavior_score"):
        assert not hasattr(instance, forbidden)
    assert math.isfinite(instance.bbox.area)
    assert isinstance(PoseFrameResult.failure(PoseStatus.INFERENCE_FAILED).instances, tuple)
