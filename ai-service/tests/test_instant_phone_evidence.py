"""Instant (single-frame) visible-phone evidence: pure logic, no model, no I/O.

These tests verify SOFTWARE behaviour only. They do not and cannot prove that
YOLO visually detects every real, very brief phone appearance.
"""

from __future__ import annotations

from app.ai.phone_rule_engine import PhoneRuleEngine, classify
from app.domain.geometry import BBox
from app.domain.models import (
    AssociationStatus,
    CameraConfig,
    Detection,
    FrameDetections,
    RuleConfig,
    SourceType,
)

CAM_A = CameraConfig(id="cam-a", name="Hall A", source_type=SourceType.DEMO)
CAM_B = CameraConfig(id="cam-b", name="Hall B", source_type=SourceType.DEMO)


def person(tid: str, x: float, conf: float = 0.9) -> Detection:
    return Detection("person", conf, BBox(x, 0.3, 0.18, 0.6), tid)


def phone(x: float, y: float = 0.5, conf: float = 0.9, tid: str = "p1") -> Detection:
    return Detection("cell_phone", conf, BBox(x, y, 0.04, 0.07), tid)


def rule(**overrides) -> RuleConfig:
    base = dict(
        id="rule-1",
        name="Mobile phone",
        engine_key="mobile_phone_detection",
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


def run(engine: PhoneRuleEngine, *, camera=CAM_A, r=None, detections=None, now=0.0):
    return engine.process_frame(
        camera=camera,
        rule=r or rule(),
        detections=detections or FrameDetections((), (phone(0.42),)),
        now=now,
        source_mode="demo",
    )


# --- TEST A ---------------------------------------------------------------
def test_single_high_confidence_frame_creates_one_warning():
    engine = PhoneRuleEngine()
    drafts = run(engine, detections=FrameDetections((), (phone(0.42, conf=0.9),)))
    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.origin == "instant"
    event = draft.event
    assert event.type == "mobile_phone_detected"
    assert event.severity == "warning"
    assert event.status == "new"
    assert event.detection_frame_count == 1
    assert event.detection_duration_seconds == 0.0
    assert event.trigger_object_class == "cell_phone"
    assert event.trigger_confidence == 0.9
    assert event.evidence and event.evidence[0].role == "trigger_object"


# --- TEST B ---------------------------------------------------------------
def test_below_instant_threshold_and_below_temporal_produces_nothing():
    engine = PhoneRuleEngine()
    drafts = run(engine, detections=FrameDetections((), (phone(0.42, conf=0.7),)))
    assert drafts == []


# --- TEST C ---------------------------------------------------------------
def test_continuous_phone_does_not_spam_instant_events():
    engine = PhoneRuleEngine()
    detections = FrameDetections((), (phone(0.42, conf=0.9),))
    instant = []
    for step in range(20):
        instant += [
            d
            for d in run(engine, detections=detections, now=step * 0.05)
            if d.origin == "instant"
        ]
    assert len(instant) == 1


# --- TEST D ---------------------------------------------------------------
def test_instant_warning_does_not_block_later_temporal_escalation():
    engine = PhoneRuleEngine()
    detections = FrameDetections((person("01", 0.4),), (phone(0.42, conf=0.9),))
    instant, temporal = [], []
    for step in range(12):
        for draft in run(engine, detections=detections, now=step * 0.25):
            (instant if draft.origin == "instant" else temporal).append(draft)
    assert len(instant) == 1
    assert len(temporal) == 1
    assert temporal[0].event.type == "suspicious_cheating_activity"
    assert temporal[0].event.detection_frame_count >= 3


# --- TEST E ---------------------------------------------------------------
def test_single_associated_frame_is_only_a_warning():
    engine = PhoneRuleEngine()
    detections = FrameDetections((person("01", 0.4),), (phone(0.42, conf=0.95),))
    drafts = run(engine, detections=detections)
    assert len(drafts) == 1
    event = drafts[0].event
    assert event.type == "mobile_phone_detected"
    assert event.severity == "warning"
    assert event.severity != "critical"
    assert event.association_status is AssociationStatus.ASSOCIATED


# --- TEST F ---------------------------------------------------------------
def test_uncertain_association_never_names_a_person():
    engine = PhoneRuleEngine()
    detections = FrameDetections(
        (person("01", 0.40), person("02", 0.46)), (phone(0.5, conf=0.95),)
    )
    drafts = run(engine, detections=detections)
    assert len(drafts) == 1
    event = drafts[0].event
    assert event.association_status is AssociationStatus.UNCERTAIN
    assert event.person_tracking_id is None
    assert event.severity == "warning"
    assert event.type == "mobile_phone_detected"


def test_unassociated_phone_still_preserves_evidence():
    engine = PhoneRuleEngine()
    drafts = run(engine, detections=FrameDetections((), (phone(0.05, 0.05, conf=0.95),)))
    assert len(drafts) == 1
    assert drafts[0].event.association_status is AssociationStatus.UNASSOCIATED
    assert drafts[0].event.type == "mobile_phone_detected"


# --- TEST G ---------------------------------------------------------------
def test_instant_state_is_per_camera():
    engine = PhoneRuleEngine()
    detections = FrameDetections((), (phone(0.42, conf=0.9),))
    a = run(engine, camera=CAM_A, detections=detections, now=1.0)
    b = run(engine, camera=CAM_B, detections=detections, now=1.0)
    assert len(a) == 1 and len(b) == 1
    assert a[0].event.camera_id == "cam-a"
    assert b[0].event.camera_id == "cam-b"


# --- TEST H ---------------------------------------------------------------
def test_instant_state_is_per_rule():
    engine = PhoneRuleEngine()
    detections = FrameDetections((), (phone(0.42, conf=0.9),))
    first = run(engine, r=rule(id="rule-1"), detections=detections, now=1.0)
    second = run(engine, r=rule(id="rule-2"), detections=detections, now=1.0)
    assert len(first) == 1 and len(second) == 1
    assert {first[0].event.rule_id, second[0].event.rule_id} == {"rule-1", "rule-2"}


# --- instant threshold safety --------------------------------------------
def test_effective_instant_threshold_is_never_weaker_than_normal():
    r = rule(confidence_threshold=0.9, instant_confidence_threshold=0.5)
    assert r.effective_instant_threshold == 0.9
    assert 0.0 <= rule(instant_confidence_threshold=5.0).effective_instant_threshold <= 1.0


def test_instant_can_be_disabled():
    engine = PhoneRuleEngine()
    drafts = run(
        engine,
        r=rule(instant_detection_enabled=False),
        detections=FrameDetections((), (phone(0.42, conf=0.99),)),
    )
    assert drafts == []


# --- TEST K ---------------------------------------------------------------
def test_no_confirmed_cheating_claim_anywhere():
    for status in AssociationStatus:
        if status is AssociationStatus.NOT_APPLICABLE:
            continue
        event_type, severity = classify(status)
        assert "confirmed" not in event_type
        assert severity in {"critical", "warning", "info"}
