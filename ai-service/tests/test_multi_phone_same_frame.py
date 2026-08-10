"""Same-frame multi-phone isolation: pure logic, no model, no I/O.

Deterministic proof that PhoneRuleEngine treats every visible phone inside ONE
analysed frame as an independent observation: independent attribution,
independent event identity, independent cooldown scope.
"""

from __future__ import annotations

from app.ai.phone_rule_engine import PhoneRuleEngine
from app.ai.temporal_state import object_key
from app.domain.geometry import BBox
from app.domain.models import (
    AssociationStatus,
    CameraConfig,
    Detection,
    FrameDetections,
    RuleConfig,
    SourceType,
)

CAM = CameraConfig(id="cam-multi", name="Hall Multi", source_type=SourceType.DEMO)


def person(tid: str, x: float, y: float = 0.30, conf: float = 0.92) -> Detection:
    return Detection("person", conf, BBox(x, y, 0.18, 0.60), tid)


def phone(x: float, y: float, conf: float = 0.95, tid=None) -> Detection:
    return Detection("cell_phone", conf, BBox(x, y, 0.04, 0.07), tid)


def rule(**overrides) -> RuleConfig:
    base = dict(
        id="rule-multi",
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


def run(engine, detections, now, r=None):
    return engine.process_frame(
        camera=CAM,
        rule=r or rule(),
        detections=detections,
        now=now,
        source_mode="demo",
    )


def instant(drafts):
    return [d for d in drafts if d.origin == "instant"]


def trigger_evidence(draft):
    return next(item for item in draft.event.evidence if item.role == "trigger_object")


# Clearly separated people: A on the left, B on the right.
PERSON_A = person("11", 0.10)
PERSON_B = person("24", 0.70)
# Phones sitting on each person's torso.
PHONE_A = phone(0.16, 0.50, tid="31")
PHONE_B = phone(0.76, 0.50, tid="48")


# --- 1: two people + two phones in ONE frame ------------------------------
def test_two_people_two_phones_same_frame_produce_two_instant_events():
    engine = PhoneRuleEngine()
    frame = FrameDetections((PERSON_A, PERSON_B), (PHONE_A, PHONE_B))
    drafts = instant(run(engine, frame, now=0.0))

    assert len(drafts) == 2
    for draft in drafts:
        assert draft.origin == "instant"
        assert draft.event.type == "mobile_phone_detected"
        assert draft.event.severity == "warning"
        assert draft.event.association_status is AssociationStatus.ASSOCIATED

    by_person = {d.event.person_tracking_id: d for d in drafts}
    assert set(by_person) == {"11", "24"}

    # Attribution follows the phone, not list position.
    assert trigger_evidence(by_person["11"]).tracking_id == "31"
    assert trigger_evidence(by_person["24"]).tracking_id == "48"
    assert trigger_evidence(by_person["11"]).bbox == PHONE_A.bbox
    assert trigger_evidence(by_person["24"]).bbox == PHONE_B.bbox
    assert trigger_evidence(by_person["11"]).associated_person_tracking_id == "11"
    assert trigger_evidence(by_person["24"]).associated_person_tracking_id == "24"

    # Independent identity and independent evidence objects.
    assert drafts[0].event.id != drafts[1].event.id
    assert drafts[0].event.evidence is not drafts[1].event.evidence


def test_detection_order_does_not_change_attribution():
    engine = PhoneRuleEngine()
    reversed_frame = FrameDetections((PERSON_B, PERSON_A), (PHONE_B, PHONE_A))
    drafts = instant(run(engine, reversed_frame, now=0.0))
    mapping = {trigger_evidence(d).tracking_id: d.event.person_tracking_id for d in drafts}
    assert mapping == {"31": "11", "48": "24"}


# --- 2: cooldown independence --------------------------------------------
def test_cooldown_is_per_phone_and_never_cross_suppresses():
    engine = PhoneRuleEngine()
    first = instant(run(engine, FrameDetections((PERSON_A, PERSON_B), (PHONE_A, PHONE_B)), now=0.0))
    assert len(first) == 2

    # Phone B disappears, phone A stays: A is inside its own cooldown.
    second = instant(run(engine, FrameDetections((PERSON_A, PERSON_B), (PHONE_A,)), now=1.0))
    assert second == []

    # A brand new phone C on person B must not be suppressed by A's cooldown.
    phone_c = phone(0.74, 0.62, tid="57")
    third = instant(
        run(engine, FrameDetections((PERSON_A, PERSON_B), (PHONE_A, phone_c)), now=2.0)
    )
    assert len(third) == 1
    assert trigger_evidence(third[0]).tracking_id == "57"


# --- 3: one uncertain + one clear phone in the same frame ----------------
def test_uncertain_phone_does_not_contaminate_clear_phone():
    engine = PhoneRuleEngine()
    # Two adjacent people; the ambiguous phone sits between them.
    left = person("11", 0.20)
    middle = person("12", 0.32)
    right = person("24", 0.75)
    ambiguous = phone(0.315, 0.50, tid="31")
    clear = phone(0.81, 0.50, tid="48")

    drafts = instant(
        run(engine, FrameDetections((left, middle, right), (ambiguous, clear)), now=0.0)
    )
    assert len(drafts) == 2

    by_phone = {trigger_evidence(d).tracking_id: d for d in drafts}
    uncertain_draft = by_phone["31"]
    clear_draft = by_phone["48"]

    assert uncertain_draft.event.association_status is AssociationStatus.UNCERTAIN
    assert uncertain_draft.event.severity == "warning"
    assert uncertain_draft.event.person_tracking_id is None

    assert clear_draft.event.association_status is AssociationStatus.ASSOCIATED
    assert clear_draft.event.severity == "warning"
    assert clear_draft.event.person_tracking_id == "24"


# --- 4: two untracked phones in ONE frame -------------------------------
def test_two_untracked_phones_same_frame_are_independent():
    engine = PhoneRuleEngine()
    left = phone(0.12, 0.12, tid=None)
    right = phone(0.85, 0.80, tid=None)

    drafts = instant(run(engine, FrameDetections((), (left, right)), now=0.0))
    assert len(drafts) == 2
    assert drafts[0].event.id != drafts[1].event.id

    keys = {object_key(left), object_key(right)}
    assert len(keys) == 2
    assert all(key.startswith("sig:") for key in keys)

    bboxes = {trigger_evidence(d).bbox for d in drafts}
    assert bboxes == {left.bbox, right.bbox}
