"""The `mobile_phone_detection` engine.

Turns per-frame detections into safe, contract-compliant event drafts. The
engine is pure: it takes detections plus a monotonic timestamp and returns
drafts. Snapshotting, uploading and persistence happen elsewhere.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..domain.models import (
    AiEvent,
    AssociationResult,
    AssociationStatus,
    CameraConfig,
    CLASS_PERSON,
    CLASS_PHONE,
    Detection,
    EvidenceItem,
    FrameDetections,
    RuleConfig,
    SourceMode,
)
from .association import DEFAULT_ASSOCIATION_MARGIN, associate
from .temporal_state import (
    AssociationMemory,
    InstantGate,
    TemporalConfirmer,
    alert_key,
    subject_for,
)

TYPE_SUSPICIOUS = "suspicious_cheating_activity"
TYPE_POSSIBLE = "possible_cheating_activity"
TYPE_PHONE_ONLY = "mobile_phone_detected"


def classify(status: AssociationStatus) -> tuple[str, str]:
    """Maps association status to (event type, severity). Never 'confirmed'."""
    if status is AssociationStatus.ASSOCIATED:
        return TYPE_SUSPICIOUS, "critical"
    if status is AssociationStatus.UNCERTAIN:
        return TYPE_POSSIBLE, "warning"
    return TYPE_PHONE_ONLY, "warning"


def overall_confidence(trigger_confidence: float, association_confidence: Optional[float]) -> float:
    """Deterministic and conservative: the weakest link of the chain."""
    if association_confidence is None:
        return round(float(trigger_confidence), 4)
    return round(min(float(trigger_confidence), float(association_confidence)), 4)


def build_evidence(
    phone: Detection,
    persons: tuple[Detection, ...],
    association: AssociationResult,
) -> list[EvidenceItem]:
    """Phone evidence plus the persons visible in the triggering frame."""
    definitive = (
        association.person_tracking_id
        if association.status is AssociationStatus.ASSOCIATED
        else None
    )
    items = [
        EvidenceItem(
            object_id=f"phone-{phone.tracking_id or 'na'}",
            class_name=CLASS_PHONE,
            confidence=phone.confidence,
            bbox=phone.bbox,
            role="trigger_object",
            tracking_id=phone.tracking_id,
            associated_person_tracking_id=definitive,
            association_confidence=association.confidence,
        )
    ]
    for person in persons:
        items.append(
            EvidenceItem(
                object_id=f"person-{person.tracking_id or 'na'}",
                class_name=CLASS_PERSON,
                confidence=person.confidence,
                bbox=person.bbox,
                role="person",
                tracking_id=person.tracking_id,
            )
        )
    return items


@dataclass
class EventDraft:
    """A confirmed event plus the extra context the publisher needs."""

    event: AiEvent
    rule: RuleConfig
    save_snapshot: bool
    #: "instant" = single-frame visible-phone evidence,
    #: "temporal" = duration + frame-count confirmed evidence.
    origin: str = "temporal"


class PhoneRuleEngine:
    """Stateful per-camera engine for one rule instance.

    Two independent evidence levels share one detection pass:

    * PATH A (instant): a single analysed frame with a phone confidence at or
      above the stricter instant threshold preserves the observation as a
      `mobile_phone_detected` warning. It never claims cheating.
    * PATH B (temporal): unchanged duration + matching-frame + association
      reasoning, the only path that may produce the stronger event types.

    The two paths keep separate state, so an instant warning never resets,
    consumes or suppresses a later temporal escalation.
    """

    engine_key = "mobile_phone_detection"

    def __init__(
        self,
        *,
        association_margin: float = DEFAULT_ASSOCIATION_MARGIN,
        gap_tolerance_seconds: float = 0.5,
    ) -> None:
        self.association_margin = association_margin
        self._confirmers: dict[str, TemporalConfirmer] = {}
        self._memory: dict[str, AssociationMemory] = {}
        self._instant: dict[str, InstantGate] = {}
        self._gap_tolerance = gap_tolerance_seconds

    def _confirmer(self, camera_id: str) -> TemporalConfirmer:
        if camera_id not in self._confirmers:
            self._confirmers[camera_id] = TemporalConfirmer(self._gap_tolerance)
        return self._confirmers[camera_id]

    def _memory_for(self, camera_id: str) -> AssociationMemory:
        if camera_id not in self._memory:
            self._memory[camera_id] = AssociationMemory()
        return self._memory[camera_id]

    def _instant_gate(self, camera_id: str) -> InstantGate:
        if camera_id not in self._instant:
            self._instant[camera_id] = InstantGate()
        return self._instant[camera_id]

    def reset(self, camera_id: str) -> None:
        self._confirmers.pop(camera_id, None)
        self._memory.pop(camera_id, None)
        self._instant.pop(camera_id, None)

    def process_frame(
        self,
        *,
        camera: CameraConfig,
        rule: RuleConfig,
        detections: FrameDetections,
        now: float,
        source_mode: SourceMode,
        detected_at: Optional[datetime] = None,
    ) -> list[EventDraft]:
        """Evaluates one analysed frame and returns any confirmed events."""
        if not rule.is_phone_engine or not rule.applies_to(camera.id):
            return []

        persons = tuple(
            person
            for person in detections.persons
            if person.confidence >= rule.person_confidence_threshold and person.tracking_id
        )
        phones = tuple(
            phone for phone in detections.phones if phone.confidence >= rule.confidence_threshold
        )

        confirmer = self._confirmer(camera.id)
        memory = self._memory_for(camera.id)
        instant_gate = self._instant_gate(camera.id)
        drafts: list[EventDraft] = []

        if not phones:
            confirmer.expire(now)
            return drafts

        for index, phone in enumerate(phones):
            phone_id = phone.tracking_id or f"idx{index}"
            association = associate(
                phone,
                persons,
                association_threshold=rule.association_confidence_threshold,
                margin=self.association_margin,
                previous=memory.recall(phone_id, now),
            )
            if (
                association.status is AssociationStatus.ASSOCIATED
                and association.person_tracking_id
                and association.confidence is not None
            ):
                memory.remember(phone_id, association.person_tracking_id, association.confidence, now)

            if rule.require_person_association and association.status is AssociationStatus.UNASSOCIATED:
                # The rule demands a person; a lone phone is still reported but
                # only at the weakest, non-accusatory level.
                pass

            # --- PATH A: instant visible-phone evidence -------------------
            # Independent of the temporal state machine below, so the same
            # phone can still escalate later through PATH B.
            if (
                rule.instant_detection_enabled
                and phone.confidence >= rule.effective_instant_threshold
            ):
                instant_key = alert_key(camera.id, rule.id, "instant", f"object:{phone_id}")
                if instant_gate.allow(
                    instant_key, now=now, cooldown_seconds=rule.cooldown_seconds
                ):
                    drafts.append(
                        EventDraft(
                            event=AiEvent(
                                id=str(uuid.uuid4()),
                                type=TYPE_PHONE_ONLY,
                                severity="warning",
                                camera_id=camera.id,
                                camera_name=camera.name,
                                rule_id=rule.id,
                                confidence=overall_confidence(
                                    phone.confidence, association.confidence
                                ),
                                trigger_object_class=CLASS_PHONE,
                                trigger_confidence=phone.confidence,
                                association_status=association.status,
                                association_confidence=association.confidence,
                                detection_duration_seconds=0.0,
                                detection_frame_count=1,
                                source_mode=source_mode,
                                detected_at=detected_at or datetime.now(timezone.utc),
                                person_tracking_id=(
                                    association.person_tracking_id
                                    if association.status is AssociationStatus.ASSOCIATED
                                    else None
                                ),
                                evidence=build_evidence(phone, persons, association),
                            ),
                            rule=rule,
                            save_snapshot=rule.save_snapshot,
                            origin="instant",
                        )
                    )

            # --- PATH B: temporal confirmation (unchanged) ----------------
            event_type, severity = classify(association.status)
            subject = subject_for(association.status, association.person_tracking_id, phone_id)
            key = alert_key(camera.id, rule.id, event_type, subject)

            outcome = confirmer.observe(
                key,
                now=now,
                min_duration_seconds=rule.min_duration_seconds,
                min_matching_frames=rule.min_matching_frames,
                cooldown_seconds=rule.cooldown_seconds,
                trigger_confidence=phone.confidence,
                association_confidence=association.confidence,
            )
            if not outcome.confirmed:
                continue

            event = AiEvent(
                id=str(uuid.uuid4()),
                type=event_type,
                severity=severity,
                camera_id=camera.id,
                camera_name=camera.name,
                rule_id=rule.id,
                confidence=overall_confidence(phone.confidence, association.confidence),
                trigger_object_class=CLASS_PHONE,
                trigger_confidence=phone.confidence,
                association_status=association.status,
                association_confidence=association.confidence,
                detection_duration_seconds=outcome.duration_seconds,
                detection_frame_count=outcome.frame_count,
                source_mode=source_mode,
                detected_at=detected_at or datetime.now(timezone.utc),
                person_tracking_id=(
                    association.person_tracking_id
                    if association.status is AssociationStatus.ASSOCIATED
                    else None
                ),
                evidence=build_evidence(phone, persons, association),
            )
            drafts.append(EventDraft(event=event, rule=rule, save_snapshot=rule.save_snapshot))

        return drafts