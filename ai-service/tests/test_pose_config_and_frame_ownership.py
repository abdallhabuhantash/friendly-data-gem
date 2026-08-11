"""Pose configuration explicitness and strict pose frame ownership.

Two contracts are proven here:

1. Enabling pose requires EXPLICIT deployment values. No model, device, input
   size, confidence floor or cadence is ever invented on the operator's behalf.
2. A frame handed to the pose worker is an INDEPENDENT object. There is no
   fallback that would let the background worker read the live capture buffer.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.runtime.orchestrator import _independent_frame_copy


def settings(**overrides) -> Settings:
    base = {
        "supabase_url": "https://example.invalid",
        "supabase_service_role_key": "k",
        "ai_service_key": "s",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


# --- pose disabled is the normal default --------------------------------
def test_pose_is_disabled_and_unconfigured_by_default() -> None:
    config = settings()
    assert config.pose_enabled is False
    assert config.pose_inference_problems == []
    assert config.pose_inference_configured is False
    assert config.pose_association_configured is False
    assert config.validate_runtime() == []


def test_disabled_pose_has_no_silent_deployment_defaults() -> None:
    config = settings()
    assert config.pose_device is None
    assert config.pose_imgsz is None
    assert config.pose_confidence is None
    assert config.pose_max_fps is None
    assert config.pose_min_interval_seconds is None


# --- pose enabled requires explicit values ------------------------------
REQUIRED = {
    "pose_model": "/models/pose.pt",
    "pose_device": "cpu",
    "pose_imgsz": 640,
    "pose_confidence": 0.4,
    "pose_max_fps": 2.0,
}


@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_each_missing_required_pose_setting_blocks_pose(missing: str) -> None:
    values = dict(REQUIRED)
    values[missing] = "" if missing == "pose_model" else None
    config = settings(pose_enabled=True, **values)

    assert config.pose_inference_configured is False
    problems = config.pose_inference_problems
    assert len(problems) == 1
    assert missing.upper() in problems[0]
    assert "unconfigured" in problems[0]
    # A configuration problem is reported, never fatal for Task 1.
    assert any("pose" in problem.lower() for problem in config.validate_runtime())


def test_all_required_values_present_configures_pose_inference() -> None:
    config = settings(pose_enabled=True, **REQUIRED)
    assert config.pose_inference_problems == []
    assert config.pose_inference_configured is True
    assert config.pose_min_interval_seconds == pytest.approx(0.5)
    # Association stays explicitly unconfigured: no calibrated values exist.
    assert config.pose_association_configured is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pose_imgsz", 0),
        ("pose_confidence", 1.5),
        ("pose_max_fps", 0.0),
    ],
)
def test_explicit_but_invalid_values_are_rejected(field: str, value) -> None:  # noqa: ANN001
    values = dict(REQUIRED)
    values[field] = value
    config = settings(pose_enabled=True, **values)
    assert config.pose_inference_configured is False
    assert any(field.upper() in problem for problem in config.pose_inference_problems)


# --- strict frame ownership ---------------------------------------------
class CopyableFrame:
    def __init__(self) -> None:
        self.copies = 0

    def copy(self) -> "CopyableFrame":
        self.copies += 1
        return CopyableFrame()


def test_copy_capable_frame_yields_exactly_one_independent_copy() -> None:
    frame = CopyableFrame()
    duplicate = _independent_frame_copy(frame)
    assert frame.copies == 1
    assert duplicate is not frame


def test_frame_without_copy_is_refused() -> None:
    with pytest.raises(TypeError):
        _independent_frame_copy(object())


def test_frame_returning_itself_is_refused() -> None:
    class SelfReturning:
        def copy(self):  # noqa: ANN202
            return self

    with pytest.raises(TypeError):
        _independent_frame_copy(SelfReturning())


def test_frame_returning_none_is_refused() -> None:
    class NoneCopy:
        def copy(self):  # noqa: ANN202
            return None

    with pytest.raises(TypeError):
        _independent_frame_copy(NoneCopy())
