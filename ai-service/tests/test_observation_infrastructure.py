"""Task 2B infrastructure: observations, registry dispatch, isolation.

Pure logic only: no model, no camera, no network, no database. Task 1 phone
behaviour must remain provably untouched by everything proven here.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.ai.engine_registry import EngineRegistry, FrameContext, PhoneEngineAdapter
from app.ai.observation_builder import build_frame_observations
from app.ai.phone_rule_engine import PhoneRuleEngine
from app.domain.geometry import BBox
from app.domain.models import (
    ENGINE_MOBILE_PHONE,
    CameraConfig,
    Detection,
    FrameDetections,
    RuleConfig,
    SourceType,
)
from app.domain.observations import FrameObservations, PersonObservation

CAM_A = CameraConfig(id="11111111-1111-4111-8111-111111111111", name="Hall A")
CAM_B = CameraConfig(id="22222222-2222-4222-8222-222222222222", name="Hall B")


def person(tid, x, conf=0.92) -> Detection:
    return Detection("person", conf, BBox(x, 0.30, 0.18, 0.60), tid)


def phone(x, y=0.55, conf=0.95, tid=None) -> Detection:
    return Detection("cell_phone", conf, BBox(x, y, 0.04, 0.07), tid)


def phone_rule(**overrides) -> RuleConfig:
    base = dict(
        id="rule-phone",
        name="Mobile phone",
        engine_key=ENGINE_MOBILE_PHONE,
        available=True,
        enabled=True,
        severity="critical",
        confidence_threshold=0.6,
        person_confidence_threshold=0.5,
        association_confidence_threshold=0.65,
        min_duration_seconds=1.0,
        min_matching_frames=3,
        cooldown_seconds=30,
        require_person_association=True,
        instant_detection_enabled=True,
        instant_confidence_threshold=0.85,
    )
    base.update(overrides)
    return RuleConfig(**base)


def behavioural_rule(engine_key="concealed_device_activity", **overrides) -> RuleConfig:
    base = dict(
        id="rule-behaviour",
        name="Behavioural",
        engine_key=engine_key,
        available=True,
        enabled=True,
        severity="warning",
        # Deliberately extreme thresholds: must never leak into phone logic.
        confidence_threshold=0.01,
        person_confidence_threshold=0.99,
        association_confidence_threshold=0.99,
    )
    base.update(overrides)
    return RuleConfig(**base)


def context(camera, detections, now=100.0) -> FrameContext:
    return FrameContext(
        camera=camera,
        detections=detections,
        observations=build_frame_observations(
            camera_id=camera.id, detections=detections, frame_sequence=7, source_mode="live"
        ),
        now=now,
        source_mode="live",
    )


# --- A/B/C: observation structures ---------------------------------------
def test_two_persons_build_immutable_observations():
    detections = FrameDetections(persons=(person("p1", 0.10), person("p2", 0.60)), phones=())
    obs = build_frame_observations(
        camera_id=CAM_A.id, detections=detections, frame_sequence=42, source_mode="live"
    )
    assert isinstance(obs, FrameObservations)
    assert obs.camera_id == CAM_A.id
    assert obs.frame_sequence == 42
    assert obs.person_count == 2
    assert isinstance(obs.persons, tuple)
    assert [p.person_tracking_id for p in obs.persons] == ["p1", "p2"]
    assert obs.persons[0].person_bbox == detections.persons[0].bbox
    assert obs.persons[1].confidence == pytest.approx(0.92)
    # No pose / region / behavioural fields exist on the observation layer.
    fields = {f.name for f in dataclasses.fields(PersonObservation)}
    assert fields == {"person_tracking_id", "person_bbox", "confidence"}


def test_builder_does_not_mutate_frame_detections():
    detections = FrameDetections(persons=(person("p1", 0.10),), phones=(phone(0.20),))
    before = (detections.persons, detections.phones, detections.persons[0])
    build_frame_observations(camera_id=CAM_A.id, detections=detections)
    assert (detections.persons, detections.phones, detections.persons[0]) == before
    assert detections.persons[0].tracking_id == "p1"


def test_observations_are_frozen():
    obs = build_frame_observations(
        camera_id=CAM_A.id, detections=FrameDetections(persons=(person("p1", 0.1),))
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.camera_id = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.persons[0].confidence = 0.1


# --- D: phone engine resolves through the registry ------------------------
def _confirm_phone(registry, rules, camera):
    detections = FrameDetections(persons=(person("p1", 0.40),), phones=(phone(0.46),))
    drafts = []
    for step in range(4):
        drafts.extend(registry.dispatch(rules, context(camera, detections, now=100.0 + step * 0.5)))
    return drafts


def test_phone_rule_dispatches_through_registry_and_still_fires():
    registry = EngineRegistry()
    registry.register(ENGINE_MOBILE_PHONE, PhoneEngineAdapter(PhoneRuleEngine()))
    drafts = _confirm_phone(registry, [phone_rule()], CAM_A)
    assert drafts, "phone engine produced no event through the registry"
    assert {d.origin for d in drafts} <= {"instant", "temporal"}
    assert any(d.origin == "instant" for d in drafts)


# --- E/F: independent dispatch + exception isolation ----------------------
class FakeEngine:
    def __init__(self, *, boom=False):
        self.calls = []
        self.reset_calls = []
        self.boom = boom

    def process(self, rule, ctx):
        self.calls.append((rule.id, ctx.camera.id, ctx.observations.frame_sequence))
        if self.boom:
            raise RuntimeError("fake engine failure")
        return [f"draft-{rule.id}"]

    def reset(self, camera_id):
        self.reset_calls.append(camera_id)


def test_two_fake_engines_dispatch_independently():
    registry = EngineRegistry()
    a, b = FakeEngine(), FakeEngine()
    registry.register("engine_a", a)
    registry.register("engine_b", b)
    rules = [
        behavioural_rule(engine_key="engine_a", id="rule-a"),
        behavioural_rule(engine_key="engine_b", id="rule-b"),
    ]
    drafts = registry.dispatch(rules, context(CAM_A, FrameDetections()))
    assert drafts == ["draft-rule-a", "draft-rule-b"]
    assert a.calls and b.calls
    assert a.calls[0][2] == 7  # accepted capture sequence reached the engine


def test_failing_engine_does_not_suppress_phone_events():
    registry = EngineRegistry()
    registry.register(ENGINE_MOBILE_PHONE, PhoneEngineAdapter(PhoneRuleEngine()))
    broken = FakeEngine(boom=True)
    registry.register("engine_b", broken)
    rules = [behavioural_rule(engine_key="engine_b", id="rule-b"), phone_rule()]
    drafts = _confirm_phone(registry, rules, CAM_A)
    assert broken.calls, "broken engine was never dispatched"
    assert drafts, "phone drafts were lost because another engine raised"


# --- G: unknown engine key -----------------------------------------------
def test_unknown_engine_key_is_skipped_without_phone_fallback():
    registry = EngineRegistry()
    phone_engine = PhoneRuleEngine()
    registry.register(ENGINE_MOBILE_PHONE, PhoneEngineAdapter(phone_engine))
    rules = [behavioural_rule(engine_key="totally_unknown", id="rule-x")]
    detections = FrameDetections(persons=(person("p1", 0.40),), phones=(phone(0.46),))
    drafts = []
    for step in range(5):
        drafts.extend(registry.dispatch(rules, context(CAM_A, detections, now=100.0 + step * 0.5)))
    assert drafts == []
    assert registry.handler("totally_unknown") is None
    assert phone_engine._confirmers == {}
    assert phone_engine._instant == {}


def test_missing_engine_key_is_skipped():
    registry = EngineRegistry()
    registry.register(ENGINE_MOBILE_PHONE, PhoneEngineAdapter(PhoneRuleEngine()))
    rules = [behavioural_rule(engine_key=None, id="rule-none")]
    assert registry.dispatch(rules, context(CAM_A, FrameDetections())) == []


# --- H/I: generic rule selection -----------------------------------------
def _selected(rules, camera):
    return [
        rule
        for rule in rules
        if rule.enabled and rule.available and rule.applies_to(camera.id)
    ]


def test_camera_scoped_rule_stays_isolated_by_uuid():
    scoped = phone_rule(id="rule-a-only", camera_ids=(CAM_A.id,))
    assert _selected([scoped], CAM_A) == [scoped]
    assert _selected([scoped], CAM_B) == []


def test_fleet_wide_rule_applies_to_every_camera():
    fleet = behavioural_rule(engine_key="engine_a", camera_ids=())
    assert _selected([fleet], CAM_A) == [fleet]
    assert _selected([fleet], CAM_B) == [fleet]


def test_disabled_or_unavailable_rules_are_never_selected():
    assert _selected([phone_rule(enabled=False)], CAM_A) == []
    assert _selected([phone_rule(available=False)], CAM_A) == []


# --- J: reset lifecycle ---------------------------------------------------
def test_registry_reset_clears_only_the_requested_camera():
    registry = EngineRegistry()
    phone_engine = PhoneRuleEngine()
    registry.register(ENGINE_MOBILE_PHONE, PhoneEngineAdapter(phone_engine))
    fake = FakeEngine()
    registry.register("engine_a", fake)

    _confirm_phone(registry, [phone_rule()], CAM_A)
    _confirm_phone(registry, [phone_rule()], CAM_B)
    assert CAM_A.id in phone_engine._confirmers
    assert CAM_B.id in phone_engine._confirmers

    registry.reset(CAM_A.id)
    assert CAM_A.id not in phone_engine._confirmers
    assert CAM_A.id not in phone_engine._memory
    assert CAM_A.id not in phone_engine._instant
    assert CAM_B.id in phone_engine._confirmers
    assert fake.reset_calls == [CAM_A.id]


# --- K: phone annotation thresholds isolated from behavioural rules -------
def phone_annotation_thresholds(rules):
    """Mirrors the orchestrator's phone-only threshold computation."""
    phone_rules = [r for r in rules if r.engine_key == ENGINE_MOBILE_PHONE]
    if not phone_rules:
        return None
    return (
        min(r.person_confidence_threshold for r in phone_rules),
        min(r.confidence_threshold for r in phone_rules),
        min(r.association_confidence_threshold for r in phone_rules),
    )


@pytest.mark.parametrize(
    "engine_key", ["concealed_device_activity", "document_exchange", "peer_interaction"]
)
def test_behavioural_rules_never_change_phone_annotation_thresholds(engine_key):
    only_phone = [phone_rule()]
    mixed = [phone_rule(), behavioural_rule(engine_key=engine_key)]
    assert phone_annotation_thresholds(mixed) == phone_annotation_thresholds(only_phone)


def test_behavioural_only_camera_has_no_phone_annotation_thresholds():
    assert phone_annotation_thresholds([behavioural_rule()]) is None
