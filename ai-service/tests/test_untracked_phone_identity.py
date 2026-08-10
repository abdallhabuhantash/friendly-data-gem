"""Untracked instant-phone deduplication identity: pure logic, no model, no I/O.

Verifies that array position is never used as object identity, and that the
spatial fallback key neither merges clearly distinct untracked phones nor lets a
single stationary untracked phone spam instant warnings.
"""

from __future__ import annotations

from app.ai.phone_rule_engine import PhoneRuleEngine
from app.ai.temporal_state import object_key, spatial_signature
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


def phone(x: float, y: float = 0.5, conf: float = 0.9, tid=None) -> Detection:
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


def run(engine, *, camera=CAM_A, r=None, detections, now=0.0):
    return engine.process_frame(
        camera=camera,
        rule=r or rule(),
        detections=detections,
        now=now,
        source_mode="demo",
    )


def instant(drafts):
    return [d for d in drafts if d.origin == "instant"]


# --- A: two clearly different untracked phones are both preserved ---------
def test_two_distinct_untracked_locations_are_not_suppressed():
    engine = PhoneRuleEngine()
    first = instant(run(engine, detections=FrameDetections((), (phone(0.10, 0.10),)), now=0.0))
    second = instant(run(engine, detections=FrameDetections((), (phone(0.80, 0.85),)), now=1.0))
    assert len(first) == 1
    assert len(second) == 1


def test_untracked_identity_never_uses_array_index():
    # Same single phone, but detection ordering changed: identity must follow
    # the object, not its position in the list.
    a, b = phone(0.10, 0.10), phone(0.80, 0.85)
    engine = PhoneRuleEngine()
    run(engine, detections=FrameDetections((), (a, b)), now=0.0)
    reordered = instant(run(engine, detections=FrameDetections((), (b, a)), now=0.5))
    assert reordered == []
    assert object_key(a) != object_key(b)
    assert "idx" not in object_key(a)


# --- B: one stationary untracked phone does not spam ----------------------
def test_stationary_untracked_phone_does_not_spam():
    engine = PhoneRuleEngine()
    fired = []
    for step in range(20):
        jitter = 0.001 * (step % 3)
        fired += instant(
            run(
                engine,
                detections=FrameDetections((), (phone(0.42 + jitter, 0.50 + jitter),)),
                now=step * 0.05,
            )
        )
    assert len(fired) == 1


def test_spatial_signature_is_deterministic_and_not_identity():
    box = BBox(0.42, 0.50, 0.04, 0.07)
    assert spatial_signature(box) == spatial_signature(BBox(0.421, 0.501, 0.04, 0.07))
    assert spatial_signature(box) != spatial_signature(BBox(0.80, 0.85, 0.04, 0.07))


# --- C: association context keeps dedup stable without a phone track ------
def test_associated_person_context_dedups_even_when_phone_untracked():
    engine = PhoneRuleEngine()
    persons = (person("01", 0.40),)
    fired = []
    for step in range(8):
        # The phone drifts and has no tracking id, but stays on the same person.
        fired += instant(
            run(
                engine,
                detections=FrameDetections(persons, (phone(0.42 + step * 0.01, 0.45),)),
                now=step * 0.1,
            )
        )
    assert len(fired) == 1
    assert fired[0].event.association_status is AssociationStatus.ASSOCIATED
    assert fired[0].event.person_tracking_id == "01"


# --- D / E: isolation -----------------------------------------------------
def test_camera_isolation_for_untracked_phones():
    engine = PhoneRuleEngine()
    detections = FrameDetections((), (phone(0.42),))
    a = instant(run(engine, camera=CAM_A, detections=detections, now=1.0))
    b = instant(run(engine, camera=CAM_B, detections=detections, now=1.0))
    assert len(a) == 1 and len(b) == 1


def test_rule_isolation_for_untracked_phones():
    engine = PhoneRuleEngine()
    detections = FrameDetections((), (phone(0.42),))
    first = instant(run(engine, r=rule(id="rule-1"), detections=detections, now=1.0))
    second = instant(run(engine, r=rule(id="rule-2"), detections=detections, now=1.0))
    assert len(first) == 1 and len(second) == 1


# --- threshold invariant --------------------------------------------------
def test_instant_threshold_never_weaker_than_trigger_threshold():
    r = rule(confidence_threshold=0.92, instant_confidence_threshold=0.85)
    assert r.effective_instant_threshold == 0.92
