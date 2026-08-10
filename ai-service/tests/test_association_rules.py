"""Pure-logic tests: no camera, no model weights, no network."""

from __future__ import annotations

import time

from app.ai.association import associate
from app.ai.phone_rule_engine import PhoneRuleEngine, classify, overall_confidence
from app.domain.geometry import BBox
from app.domain.models import (
    AssociationStatus,
    CameraConfig,
    Detection,
    FrameDetections,
    RuleConfig,
    SourceType,
)
from app.notifications.base import effective_severity, format_message, payload_from_row


def person(tid: str, x: float, conf: float = 0.9) -> Detection:
    return Detection("person", conf, BBox(x, 0.3, 0.18, 0.6), tid)


def phone(x: float, y: float = 0.5, conf: float = 0.85, tid: str = "p1") -> Detection:
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
    )
    base.update(overrides)
    return RuleConfig(**base)


CAMERA = CameraConfig(id="cam-1", name="Hall A", source_type=SourceType.DEMO)


def test_phone_on_single_person_is_associated():
    result = associate(phone(0.42), (person("01", 0.4),), association_threshold=0.65)
    assert result.status is AssociationStatus.ASSOCIATED
    assert result.person_tracking_id == "01"


def test_phone_between_two_overlapping_people_is_uncertain_not_definitive():
    # Both persons plausibly own the phone: the engine must refuse to pick one.
    result = associate(
        phone(0.5), (person("01", 0.40), person("02", 0.46)), association_threshold=0.65
    )
    assert result.status is AssociationStatus.UNCERTAIN
    assert result.person_tracking_id is None


def test_phone_with_no_person_is_unassociated():
    result = associate(phone(0.05, 0.05), (), association_threshold=0.65)
    assert result.status is AssociationStatus.UNASSOCIATED


def test_classification_never_claims_confirmed_cheating():
    assert classify(AssociationStatus.ASSOCIATED) == ("suspicious_cheating_activity", "critical")
    assert classify(AssociationStatus.UNCERTAIN) == ("possible_cheating_activity", "warning")
    assert classify(AssociationStatus.UNASSOCIATED) == ("mobile_phone_detected", "warning")


def test_overall_confidence_is_the_weakest_link():
    assert overall_confidence(0.9, 0.7) == 0.7
    assert overall_confidence(0.6, None) == 0.6


def test_short_detection_does_not_create_an_event():
    engine = PhoneRuleEngine()
    now = time.monotonic()
    drafts = engine.process_frame(
        camera=CAMERA,
        rule=rule(),
        detections=FrameDetections((person("01", 0.4),), (phone(0.42),)),
        now=now,
        source_mode="demo",
    )
    assert drafts == []


def test_sustained_detection_creates_one_event_then_cools_down():
    engine = PhoneRuleEngine()
    detections = FrameDetections((person("01", 0.4),), (phone(0.42),))
    start = time.monotonic()
    produced = []
    for step in range(12):
        produced += engine.process_frame(
            camera=CAMERA,
            rule=rule(),
            detections=detections,
            now=start + step * 0.25,
            source_mode="demo",
        )
    assert len(produced) == 1
    event = produced[0].event
    assert event.type == "suspicious_cheating_activity"
    assert event.status == "new"
    assert event.source_mode == "demo"
    assert event.evidence and event.evidence[0].role == "trigger_object"


def test_event_row_matches_the_contract():
    engine = PhoneRuleEngine()
    detections = FrameDetections((person("01", 0.4),), (phone(0.42),))
    start = time.monotonic()
    drafts = []
    for step in range(12):
        drafts += engine.process_frame(
            camera=CAMERA,
            rule=rule(),
            detections=detections,
            now=start + step * 0.25,
            source_mode="live",
        )
    row = drafts[0].event.to_row()
    for key in (
        "id",
        "type",
        "severity",
        "status",
        "camera_id",
        "association_status",
        "detection_duration_seconds",
        "detection_frame_count",
        "evidence",
        "source_mode",
    ):
        assert key in row
    assert row["note"] is None
    box = row["evidence"][0]["bbox"]
    assert all(0.0 <= box[axis] <= 1.0 for axis in ("x", "y"))


def test_uncertain_association_is_never_critical_and_never_names_a_person():
    assert effective_severity("critical", "uncertain") == "warning"
    payload = payload_from_row(
        {
            "id": "e1",
            "type": "possible_cheating_activity",
            "severity": "critical",
            "camera_name": "Hall A",
            "association_status": "uncertain",
            "person_tracking_id": "07",
            "trigger_confidence": 0.81,
            "association_confidence": 0.5,
            "detection_duration_seconds": 3.2,
            "detected_at": "2026-01-01T10:00:00Z",
        }
    )
    assert payload.effective_severity == "warning"
    assert payload.person_tracking_id is None
    message = format_message(payload)
    assert "confirmed cheating" not in message.lower()
    assert "Requires Human Review" in message