"""Notification provider contract and the safety policy shared by all channels.

The policy implements docs/notification-contract.md: an uncertain association
never identifies a person and nothing is ever phrased as confirmed cheating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

TITLES = {
    "suspicious_cheating_activity": "Suspicious Cheating Activity",
    "possible_cheating_activity": "Possible Cheating Activity",
    "mobile_phone_detected": "Mobile Phone Detected",
}


@dataclass
class NotificationPayload:
    """Safe, contract-shaped message content. Never holds secrets or RTSP URLs."""

    event_id: str
    event_title: str
    camera: str
    effective_severity: str
    person_tracking_id: Optional[str]
    phone_confidence: float
    association_confidence: Optional[float]
    detection_duration_seconds: float
    detected_at: str
    review_required: bool = True
    snapshot_file: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_title": self.event_title,
            "camera": self.camera,
            "effective_severity": self.effective_severity,
            "person_tracking_id": self.person_tracking_id,
            "phone_confidence": self.phone_confidence,
            "association_confidence": self.association_confidence,
            "detection_duration_seconds": self.detection_duration_seconds,
            "detected_at": self.detected_at,
            "review_required": self.review_required,
            "snapshot_file": self.snapshot_file,
        }


def effective_severity(severity: str, association_status: str) -> str:
    """An uncertain association is never presented as critical."""
    if association_status == "uncertain" and severity == "critical":
        return "warning"
    return severity


def should_send(
    *, severity: str, association_status: str, send_warnings: bool
) -> bool:
    """Default policy: only definitive critical alerts are pushed to a phone."""
    resolved = effective_severity(severity, association_status)
    if resolved == "critical" and association_status == "associated":
        return True
    return bool(send_warnings)


def payload_from_row(row: dict[str, Any], snapshot_file: Optional[str] = None) -> NotificationPayload:
    """Builds the safe payload from a persisted event row."""
    status = str(row.get("association_status") or "not_applicable")
    severity = effective_severity(str(row.get("severity") or "warning"), status)
    return NotificationPayload(
        event_id=str(row.get("id")),
        event_title=TITLES.get(str(row.get("type")), "Detection Event"),
        camera=str(row.get("camera_name") or "Camera"),
        effective_severity=severity,
        # A definitive person is named only for a definitive association.
        person_tracking_id=(row.get("person_tracking_id") if status == "associated" else None),
        phone_confidence=float(row.get("trigger_confidence") or 0.0),
        association_confidence=(
            None if row.get("association_confidence") is None
            else float(row["association_confidence"])
        ),
        detection_duration_seconds=float(row.get("detection_duration_seconds") or 0.0),
        detected_at=str(row.get("detected_at") or ""),
        snapshot_file=snapshot_file,
    )


def format_message(payload: NotificationPayload) -> str:
    """Advisory wording only. The words 'confirmed cheating' never appear."""
    lines = [
        "VIGILANT EYE",
        payload.event_title,
        f"Camera: {payload.camera}",
    ]
    if payload.person_tracking_id:
        lines.append(f"Person ID: {payload.person_tracking_id}")
    else:
        lines.append("Person ID: not definitively associated")
    lines.extend(
        [
            f"Phone Confidence: {payload.phone_confidence:.2f}",
            (
                "Association Confidence: n/a"
                if payload.association_confidence is None
                else f"Association Confidence: {payload.association_confidence:.2f}"
            ),
            f"Detection Duration: {payload.detection_duration_seconds:.1f}s",
            f"Timestamp: {payload.detected_at}",
            f"Severity: {payload.effective_severity}",
            f"Event ID: {payload.event_id}",
            "Requires Human Review",
        ]
    )
    return "\n".join(lines)


class NotificationProvider(Protocol):
    name: str

    def send(self, payload: NotificationPayload) -> bool: ...