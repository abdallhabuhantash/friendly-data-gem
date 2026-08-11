"""Pose domain + provider contract tests (no real YOLO weights, no downloads)."""

from __future__ import annotations

import dataclasses
import math
import threading

import pytest

from app.ai.pose_provider import (
    PoseProviderConfigError,
    UltralyticsPoseProvider,
    parse_pose_result,
)
from app.domain.geometry import BBox
from app.domain.pose import (
    COCO_17_KEYPOINT_COUNT,
    COCO_17_KEYPOINTS,
    PoseContractError,
    PoseFrameResult,
    PoseInstance,
    PoseKeypoint,
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
        boxes=FakeBoxes([list(box)], [box_conf] if box_conf is not None else None),
        keypoints=FakeKeypoints([kp or points()], [kp_conf or confs()]),
    )


def valid_keypoints():
    return tuple(
        PoseKeypoint(name=name, index=index, available=True, x=0.5, y=0.5, confidence=0.9)
        for index, name in enumerate(COCO_17_KEYPOINTS)
    )


# --- domain basics --------------------------------------------------------


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


# --- 14: confidence-absence policy ---------------------------------------


def test_14a_coordinate_only_result_is_confidence_absent():
    result = parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([[0.1, 0.1, 0.4, 0.9]]),
            keypoints=FakeKeypoints([points()]),
        )
    )
    assert result.status is PoseStatus.KEYPOINT_CONFIDENCE_ABSENT
    assert result.instances == ()


def test_14b_coordinate_only_origin_keypoint_is_never_available():
    kp = points()
    kp[9] = [0.0, 0.0]
    result = parse_pose_result(
        FakeResult(boxes=FakeBoxes([[0.1, 0.1, 0.4, 0.9]]), keypoints=FakeKeypoints([kp]))
    )
    assert result.status is PoseStatus.KEYPOINT_CONFIDENCE_ABSENT


def test_14c_plausible_coordinates_without_confidence_are_not_evidence():
    kp = points(value=(0.4, 0.5))
    result = parse_pose_result(
        FakeResult(boxes=FakeBoxes([[0.1, 0.1, 0.4, 0.9]]), keypoints=FakeKeypoints([kp]))
    )
    assert result.status is PoseStatus.KEYPOINT_CONFIDENCE_ABSENT
    assert result.instances == ()


def test_14d_confidence_bearing_result_still_works():
    result = parse_pose_result(one_person_result())
    assert result.ok and result.instance_count == 1
    assert result.instances[0].confidence == pytest.approx(0.8)


# --- 15: domain invariants -----------------------------------------------


def test_15e_available_without_coordinates_is_rejected():
    with pytest.raises(PoseContractError):
        PoseKeypoint(name=PoseKeypointName.NOSE, index=0, available=True, confidence=0.9)


def test_15f_unavailable_with_coordinates_is_rejected():
    with pytest.raises(PoseContractError):
        PoseKeypoint(name=PoseKeypointName.NOSE, index=0, available=False, x=0.5, y=0.5)


def test_15g_wrong_canonical_index_is_rejected():
    with pytest.raises(PoseContractError):
        PoseKeypoint(
            name=PoseKeypointName.LEFT_WRIST,
            index=3,
            available=True,
            x=0.5,
            y=0.5,
            confidence=0.9,
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.2])
def test_15h_invalid_confidence_is_rejected(bad):
    with pytest.raises(PoseContractError):
        PoseKeypoint(name=PoseKeypointName.NOSE, index=0, available=False, confidence=bad)


def test_15h2_available_without_confidence_is_rejected():
    with pytest.raises(PoseContractError):
        PoseKeypoint(name=PoseKeypointName.NOSE, index=0, available=True, x=0.4, y=0.4)


def test_15i_missing_or_duplicate_keypoint_is_rejected():
    box = BBox(0.1, 0.1, 0.3, 0.8)
    kps = valid_keypoints()
    with pytest.raises(PoseContractError):
        PoseInstance(bbox=box, keypoints=kps[:-1])
    duplicated = (kps[0],) + kps[2:] + (kps[0],)
    with pytest.raises(PoseContractError):
        PoseInstance(bbox=box, keypoints=duplicated)


def test_15j_failure_status_cannot_carry_instances():
    instance = PoseInstance(bbox=BBox(0.1, 0.1, 0.3, 0.8), keypoints=valid_keypoints())
    with pytest.raises(PoseContractError):
        PoseFrameResult(status=PoseStatus.INFERENCE_FAILED, instances=(instance,))
    assert PoseFrameResult(status=PoseStatus.OK, instances=(instance,)).instance_count == 1


# --- 16: bbox ------------------------------------------------------------


def _box_result(box):
    return parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([list(box)], [0.8]),
            keypoints=FakeKeypoints([points()], [confs()]),
        )
    )


def test_16k_reversed_x_is_malformed_never_repaired():
    result = _box_result((0.8, 0.1, 0.2, 0.9))
    assert result.status is PoseStatus.MALFORMED_RESULT
    assert result.instances == ()


def test_16l_reversed_y_is_malformed():
    assert _box_result((0.1, 0.9, 0.4, 0.2)).status is PoseStatus.MALFORMED_RESULT


def test_16m_zero_area_bbox_is_malformed():
    assert _box_result((0.2, 0.2, 0.2, 0.2)).status is PoseStatus.MALFORMED_RESULT


@pytest.mark.parametrize("box", [(0.1, 0.1, 1.4, 0.9), (-0.3, 0.1, 0.4, 0.9)])
def test_16n_out_of_range_bbox_is_malformed(box):
    assert _box_result(box).status is PoseStatus.MALFORMED_RESULT


def test_16o_valid_bbox_is_unchanged():
    instance = _box_result((0.1, 0.2, 0.4, 0.9)).instances[0]
    assert (instance.bbox.x, instance.bbox.y) == (pytest.approx(0.1), pytest.approx(0.2))
    assert instance.bbox.width == pytest.approx(0.3)
    assert instance.bbox.height == pytest.approx(0.7)


def test_16p_epsilon_drift_is_snapped_into_range():
    instance = _box_result((-1e-12, 0.1, 1.0 + 1e-12, 0.9)).instances[0]
    assert instance.bbox.x == 0.0
    assert instance.bbox.x2 == pytest.approx(1.0)


# --- 17: alignment -------------------------------------------------------


def test_17p_two_boxes_one_confidence_is_malformed():
    result = parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([[0.1, 0.1, 0.3, 0.9], [0.5, 0.1, 0.8, 0.9]], [0.8]),
            keypoints=FakeKeypoints([points(), points()], [confs(), confs()]),
        )
    )
    assert result.status is PoseStatus.MALFORMED_RESULT


def test_17q_one_box_two_confidences_is_malformed():
    result = parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([[0.1, 0.1, 0.3, 0.9]], [0.8, 0.7]),
            keypoints=FakeKeypoints([points()], [confs()]),
        )
    )
    assert result.status is PoseStatus.MALFORMED_RESULT


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.5, 1.5])
def test_17r_invalid_supplied_box_confidence_is_malformed(bad):
    result = parse_pose_result(one_person_result(box_conf=bad))
    assert result.status is PoseStatus.MALFORMED_RESULT
    assert result.instances == ()


def test_17s_boxes_keypoints_mismatch_is_malformed():
    result = parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([[0.1, 0.1, 0.3, 0.9], [0.4, 0.1, 0.6, 0.9]]),
            keypoints=FakeKeypoints([points()], [confs()]),
        )
    )
    assert result.status is PoseStatus.MALFORMED_RESULT


def test_17t_keypoint_confidence_row_mismatch_is_malformed():
    result = parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([[0.1, 0.1, 0.3, 0.9]]),
            keypoints=FakeKeypoints([points()], [confs(5)]),
        )
    )
    assert result.status is PoseStatus.MALFORMED_RESULT


def test_17u_non_coco_keypoint_count_is_unsupported_schema():
    result = parse_pose_result(one_person_result(kp=points(5), kp_conf=confs(5)))
    assert result.status is PoseStatus.UNSUPPORTED_POSE_SCHEMA
    assert result.instances == ()


def test_17v_zero_people_is_success_with_no_instances():
    result = parse_pose_result(FakeResult(boxes=FakeBoxes([]), keypoints=FakeKeypoints([])))
    assert result.status is PoseStatus.OK
    assert result.instances == ()


def test_17w_one_malformed_of_two_fails_the_whole_frame():
    result = parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([[0.2, 0.2, 0.2, 0.2], [0.5, 0.1, 0.8, 0.9]], [0.8, 0.7]),
            keypoints=FakeKeypoints([points(), points()], [confs(), confs()]),
        )
    )
    assert result.status is PoseStatus.MALFORMED_RESULT
    assert result.instances == ()


def test_17x_two_people_yield_two_instances():
    result = parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([[0.1, 0.1, 0.3, 0.9], [0.5, 0.1, 0.8, 0.9]], [0.8, 0.7]),
            keypoints=FakeKeypoints([points(), points()], [confs(), confs()]),
        )
    )
    assert result.ok and result.instance_count == 2


def test_17y_missing_keypoints_is_explicit_status():
    result = parse_pose_result(FakeResult(boxes=FakeBoxes([[0.1, 0.1, 0.3, 0.9]]), keypoints=None))
    assert result.status is PoseStatus.KEYPOINTS_ABSENT


def test_17z_absent_box_confidence_leaves_instance_confidence_none():
    result = parse_pose_result(
        FakeResult(
            boxes=FakeBoxes([[0.1, 0.1, 0.3, 0.9]]),
            keypoints=FakeKeypoints([points()], [confs()]),
        )
    )
    assert result.ok and result.instances[0].confidence is None


# --- 18: single-frame result cardinality ---------------------------------


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


def provider_with(results):
    provider = UltralyticsPoseProvider(
        "fake-pose.pt", model_factory=lambda _: FakeModel(results=results)
    )
    return provider


def test_18u_empty_results_is_malformed():
    result = provider_with([]).infer(object())
    assert result.status is PoseStatus.MALFORMED_RESULT


def test_18v_exactly_one_result_is_parsed():
    result = provider_with([one_person_result()]).infer(object())
    assert result.ok and result.instance_count == 1


def test_18w_two_results_for_one_frame_is_malformed():
    result = provider_with([one_person_result(), one_person_result()]).infer(object())
    assert result.status is PoseStatus.MALFORMED_RESULT
    assert "got 2" in (result.reason or "")
    assert result.instances == ()


# --- 19: provider configuration -----------------------------------------


@pytest.mark.parametrize("name", ["", "   "])
def test_19x_empty_model_name_rejected(name):
    with pytest.raises(PoseProviderConfigError):
        UltralyticsPoseProvider(name)


@pytest.mark.parametrize("imgsz", [0, -640, 1.5])
def test_19y_invalid_imgsz_rejected(imgsz):
    with pytest.raises(PoseProviderConfigError):
        UltralyticsPoseProvider("fake-pose.pt", imgsz=imgsz)


@pytest.mark.parametrize("conf", [-0.1, 1.5, float("nan"), float("inf")])
def test_19z_invalid_confidence_rejected(conf):
    with pytest.raises(PoseProviderConfigError):
        UltralyticsPoseProvider("fake-pose.pt", confidence=conf)


def test_19aa_empty_device_rejected():
    with pytest.raises(PoseProviderConfigError):
        UltralyticsPoseProvider("fake-pose.pt", device=" ")


def test_19ab_valid_config_stays_lazy():
    loads = []
    provider = UltralyticsPoseProvider(
        "fake-pose.pt",
        device="cpu",
        imgsz=640,
        confidence=0.25,
        model_factory=lambda name: loads.append(name) or FakeModel(),
    )
    assert loads == []
    provider.infer(object())
    assert loads == ["fake-pose.pt"]


# --- provider failure ----------------------------------------------------


def test_q_inference_exception_is_reported_cleanly():
    provider = UltralyticsPoseProvider(
        "fake-pose.pt", model_factory=lambda _: FakeModel(error=RuntimeError("cuda oom"))
    )
    result = provider.infer(object())
    assert result.reason == "pose inference failed (RuntimeError)"
    assert "cuda oom" not in (result.reason or "")
    assert result.instances == ()


def test_r_model_load_failure_is_sticky():
    def broken(_name):
        raise FileNotFoundError("weights missing")

    provider = UltralyticsPoseProvider("missing.pt", model_factory=broken)
    assert provider.initialize() is False
    assert provider.available is False
    assert provider.infer(object()).status is PoseStatus.MODEL_UNAVAILABLE
    assert provider.initialize() is False  # stays unavailable, no retry policy


def test_error_reason_uses_model_file_name_only():
    provider = UltralyticsPoseProvider(
        "/home/secret-user/models/pose.pt",
        model_factory=lambda _: FakeModel(error=RuntimeError("boom")),
    )
    result = provider.infer(object())
    assert result.model_name == "pose.pt"
    assert "secret-user" not in (result.reason or "")


def test_provider_never_uses_tracking():
    source = (__file__.rsplit("tests", 1)[0]) + "app/ai/pose_provider.py"
    with open(source, encoding="utf-8") as handle:
        assert ".track(" not in handle.read()


def test_t_concurrent_infer_calls_are_serialized():
    inside = threading.Event()
    overlapped: list[bool] = []
    entered = threading.Lock()
    depth = {"value": 0}

    def hook():
        with entered:
            depth["value"] += 1
            overlapped.append(depth["value"] > 1)
        inside.set()
        # Deterministic: wait for the sibling thread to be released, not a sleep.
        release.wait(timeout=5)
        with entered:
            depth["value"] -= 1

    release = threading.Event()
    provider = UltralyticsPoseProvider(
        "fake-pose.pt", model_factory=lambda _: FakeModel(hook=hook)
    )
    provider.initialize()
    threads = [threading.Thread(target=provider.infer, args=(object(),)) for _ in range(2)]
    threads[0].start()
    assert inside.wait(timeout=5)
    threads[1].start()
    release.set()
    for thread in threads:
        thread.join(timeout=5)

    assert overlapped == [False, False]  # never inside predict() at the same time


def test_no_behavioural_features_leak_into_domain():
    instance = parse_pose_result(one_person_result()).instances[0]
    for forbidden in ("person_tracking_id", "head_is_down", "wrist_below", "behavior_score"):
        assert not hasattr(instance, forbidden)
    assert math.isfinite(instance.bbox.area)
    assert isinstance(PoseFrameResult.failure(PoseStatus.INFERENCE_FAILED).instances, tuple)
