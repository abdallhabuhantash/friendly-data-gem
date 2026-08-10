"""Typed domain models shared across the AI service.

Nothing here imports OpenCV, Ultralytics or Supabase so the pure logic stays
testable without hardware, model weights or network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from .geometry import BBox

CLASS_PERSON = "person"
CLASS_PHONE = "cell_phone"

#: Machine-readable identifier of the only implemented rule algorithm.
ENGINE_MOBILE_PHONE = "mobile_phone_detection"

SourceMode = Literal["live", "demo"]


class SourceType(str, Enum):
    DIRECT_CAMERA = "direct_camera"
    NVR_CHANNEL = "nvr_channel"
    DEMO = "demo"


class AssociationStatus(str, Enum):
    ASSOCIATED = "associated"
    UNCERTAIN = "uncertain"
    UNASSOCIATED = "unassociated"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class CameraConfig:
    """Camera configuration as owned by the web console. Never holds secrets."""

    id: str
    name: str
    location: str = ""
    source_type: SourceType = SourceType.DEMO
    host: str = ""
    rtsp_port: int = 554
    channel: int = 1
    stream_path: str = ""
    stream_profile: str = "main"
    ai_enabled: bool = True
    active: bool = True
    is_demo: bool = True


@dataclass(frozen=True)
class RuleConfig:
    """AI rule configuration, mirrored from public.ai_rules."""

    id: str
    name: str
    engine_key: Optional[str]
    available: bool
    enabled: bool
    severity: str = "warning"
    confidence_threshold: float = 0.7
    person_confidence_threshold: float = 0.6
    association_confidence_threshold: float = 0.65
    min_duration_seconds: float = 1.5
    min_matching_frames: int = 5
    cooldown_seconds: int = 20
    require_person_association: bool = True
    save_snapshot: bool = True
    sound_notification: bool = False
    camera_ids: tuple[str, ...] = ()

    def applies_to(self, camera_id: str) -> bool:
        """Empty scope means the rule applies to every AI-enabled camera."""
        return not self.camera_ids or camera_id in self.camera_ids

    @property
    def is_phone_engine(self) -> bool:
        return (
            self.enabled
            and self.available
            and self.engine_key == ENGINE_MOBILE_PHONE
        )


@dataclass(frozen=True)
class Detection:
    """A single detected object in one analysed frame."""

    class_name: str
    confidence: float
    bbox: BBox
    tracking_id: Optional[str] = None


@dataclass(frozen=True)
class FrameDetections:
    persons: tuple[Detection, ...] = ()
    phones: tuple[Detection, ...] = ()


@dataclass(frozen=True)
class AssociationResult:
    status: AssociationStatus
    person_tracking_id: Optional[str]
    confidence: Optional[float]
    runner_up_confidence: Optional[float] = None


@dataclass
class EvidenceItem:
    object_id: str
    class_name: str
    confidence: float
    bbox: BBox
    role: str
    tracking_id: Optional[str] = None
    associated_person_tracking_id: Optional[str] = None
    association_confidence: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "tracking_id": self.tracking_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.to_dict(),
            "role": self.role,
            "associated_person_tracking_id": self.associated_person_tracking_id,
            "association_confidence": (
                None
                if self.association_confidence is None
                else round(float(self.association_confidence), 4)
            ),
        }


@dataclass
class AiEvent:
    """An event ready for persistence. `id` is generated before any I/O."""

    id: str
    type: str
    severity: str
    camera_id: str
    camera_name: str
    rule_id: str
    confidence: float
    trigger_object_class: str
    trigger_confidence: float
    association_status: AssociationStatus
    association_confidence: Optional[float]
    detection_duration_seconds: float
    detection_frame_count: int
    source_mode: SourceMode
    detected_at: datetime
    person_tracking_id: Optional[str] = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    snapshot_path: Optional[str] = None
    status: str = "new"

    def to_row(self) -> dict[str, Any]:
        """Row shape defined by docs/ai-event-contract.md."""
        return {
            "id": self.id,
            "type": self.type,
            "severity": self.severity,
            "status": self.status,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "rule_id": self.rule_id,
            "confidence": round(float(self.confidence), 4),
            # Legacy integer column kept in sync with the precise value.
            "duration_seconds": int(round(self.detection_duration_seconds)),
            "detection_duration_seconds": round(float(self.detection_duration_seconds), 3),
            "detection_frame_count": int(self.detection_frame_count),
            "person_tracking_id": self.person_tracking_id,
            "trigger_object_class": self.trigger_object_class,
            "trigger_confidence": round(float(self.trigger_confidence), 4),
            "association_status": self.association_status.value,
            "association_confidence": (
                None
                if self.association_confidence is None
                else round(float(self.association_confidence), 4)
            ),
            "evidence": [item.to_dict() for item in self.evidence],
            "source_mode": self.source_mode,
            "snapshot_path": self.snapshot_path,
            "detected_at": self.detected_at.isoformat(),
            # `note` is human-review text only and is never written by the AI.
            "note": None,
        }


@dataclass
class SystemConfig:
    """Snapshot of public.system_settings relevant to the AI runtime."""

    operation_mode: SourceMode = "demo"
    timezone: str = "Asia/Amman"