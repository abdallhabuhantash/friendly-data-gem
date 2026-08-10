"""The single place that talks to Supabase with the service-role key.

Everything above this layer works with typed domain models, never with raw
dictionaries. The service-role key is read from the local environment only and
is never logged or returned by the API.
"""

from __future__ import annotations

import logging
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from supabase import Client, create_client

from ..domain.models import CameraConfig, RuleConfig, SourceType, SystemConfig

logger = logging.getLogger(__name__)


class DuplicateEventError(Exception):
    """The event UUID already exists: the event is already persisted."""


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class SupabaseRepository:
    """Typed access to the tables the AI service owns or observes."""

    def __init__(self, url: str, service_role_key: str, snapshot_bucket: str = "snapshots") -> None:
        self._client: Client = create_client(url, service_role_key)
        self._bucket = snapshot_bucket

    # --- configuration ----------------------------------------------------
    def system_config(self) -> SystemConfig:
        response = self._client.table("system_settings").select("*").limit(1).execute()
        rows = response.data or []
        if not rows:
            return SystemConfig()
        row = rows[0]
        mode = row.get("operation_mode") or "demo"
        return SystemConfig(
            operation_mode="live" if mode == "live" else "demo",
            timezone=row.get("timezone") or "Asia/Amman",
        )

    def cameras(self, operation_mode: str) -> list[CameraConfig]:
        """Active, AI-enabled cameras matching the current operation mode."""
        response = (
            self._client.table("cameras")
            .select("*")
            .eq("active", True)
            .eq("ai_enabled", True)
            .eq("is_demo", operation_mode == "demo")
            .execute()
        )
        cameras: list[CameraConfig] = []
        for row in response.data or []:
            try:
                source_type = SourceType(row.get("source_type") or "demo")
            except ValueError:
                source_type = SourceType.DEMO
            cameras.append(
                CameraConfig(
                    id=row["id"],
                    name=row.get("name") or "Camera",
                    location=row.get("location") or "",
                    source_type=source_type,
                    host=row.get("host") or "",
                    rtsp_port=_int(row.get("rtsp_port"), 554),
                    channel=_int(row.get("channel"), 1),
                    stream_path=row.get("stream_path") or "",
                    stream_profile=row.get("stream_profile") or "main",
                    ai_enabled=bool(row.get("ai_enabled")),
                    active=bool(row.get("active")),
                    is_demo=bool(row.get("is_demo")),
                )
            )
        return cameras

    def rules(self) -> list[RuleConfig]:
        """Enabled + available rules together with their camera scope."""
        response = (
            self._client.table("ai_rules")
            .select("*")
            .eq("enabled", True)
            .eq("available", True)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return []
        scope = self._client.table("ai_rule_cameras").select("*").execute().data or []
        by_rule: dict[str, list[str]] = {}
        for link in scope:
            by_rule.setdefault(link["rule_id"], []).append(link["camera_id"])

        rules: list[RuleConfig] = []
        for row in rows:
            rules.append(
                RuleConfig(
                    id=row["id"],
                    name=row.get("name") or "",
                    engine_key=row.get("engine_key"),
                    available=bool(row.get("available")),
                    enabled=bool(row.get("enabled")),
                    severity=row.get("severity") or "warning",
                    confidence_threshold=_float(row.get("confidence_threshold"), 0.7),
                    person_confidence_threshold=_float(row.get("person_confidence_threshold"), 0.6),
                    association_confidence_threshold=_float(
                        row.get("association_confidence_threshold"), 0.65
                    ),
                    min_duration_seconds=_float(row.get("min_duration_seconds"), 1.5),
                    min_matching_frames=_int(row.get("min_matching_frames"), 5),
                    cooldown_seconds=_int(row.get("cooldown_seconds"), 20),
                    require_person_association=bool(row.get("require_person_association")),
                    save_snapshot=bool(row.get("save_snapshot")),
                    sound_notification=bool(row.get("sound_notification")),
                    camera_ids=tuple(by_rule.get(row["id"], ())),
                )
            )
        return rules

    def camera_credentials(self, camera_id: str) -> tuple[Optional[str], Optional[str]]:
        """Service-role-only credential lookup. Values never leave the process."""
        response = (
            self._client.table("camera_credentials")
            .select("username,password")
            .eq("camera_id", camera_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return (None, None)
        return rows[0].get("username"), rows[0].get("password")

    # --- runtime writes ---------------------------------------------------
    def update_camera_runtime(
        self, camera_id: str, *, status: str, fps: float, heartbeat_at: Optional[datetime] = None
    ) -> None:
        """Truthful runtime state: only called when real frames were observed."""
        payload: dict[str, Any] = {"status": status, "fps": int(round(fps))}
        if heartbeat_at is not None:
            payload["last_heartbeat_at"] = heartbeat_at.astimezone(timezone.utc).isoformat()
        self._client.table("cameras").update(payload).eq("id", camera_id).execute()

    def write_ai_health(self, *, online: bool, is_demo: bool, payload: dict[str, Any]) -> None:
        self._client.table("service_health").upsert(
            {
                "service": "ai",
                "online": online,
                "is_demo": is_demo,
                "payload": payload,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="service",
        ).execute()

    def upload_snapshot(self, object_path: str, local_file: Path) -> str:
        """Uploads to the private bucket and returns the stored object path."""
        content_type = mimetypes.guess_type(local_file.name)[0] or "image/jpeg"
        self._client.storage.from_(self._bucket).upload(
            object_path,
            local_file.read_bytes(),
            {"content-type": content_type, "upsert": "true"},
        )
        return object_path

    def insert_event(self, row: dict[str, Any]) -> None:
        """Plain insert. A duplicate UUID means the event is already stored."""
        try:
            self._client.table("events").insert(row).execute()
        except Exception as exc:  # supabase raises APIError subclasses
            message = str(exc)
            if "duplicate key" in message or "23505" in message:
                raise DuplicateEventError(row.get("id", "")) from exc
            raise

    def set_event_snapshot(self, event_id: str, snapshot_path: str) -> None:
        self._client.table("events").update({"snapshot_path": snapshot_path}).eq(
            "id", event_id
        ).execute()