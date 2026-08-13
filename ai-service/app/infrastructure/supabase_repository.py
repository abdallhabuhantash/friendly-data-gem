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


def _iso(value: Any) -> Optional[str]:
    """UTC ISO-8601 for a datetime, or None. Never invents a timestamp."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


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
                    instant_detection_enabled=bool(row.get("instant_detection_enabled", True)),
                    instant_confidence_threshold=_float(
                        row.get("instant_confidence_threshold"), 0.85
                    ),
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
    # --- exam sessions (anonymous subject runtime) -------------------------
    def exam_session(self, exam_session_id: str) -> Optional[dict[str, Any]]:
        """Session row plus its linked camera ids. No roster data is read."""
        response = (
            self._client.table("exam_sessions")
            .select("id,title,status,started_at,ended_at")
            .eq("id", exam_session_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return None
        row = dict(rows[0])
        row["camera_ids"] = self._exam_session_camera_ids(exam_session_id)
        return row

    def armed_exam_sessions(self) -> list[dict[str, Any]]:
        """Every session the console has armed (status `active`)."""
        response = (
            self._client.table("exam_sessions")
            .select("id,started_at")
            .eq("status", "active")
            .execute()
        )
        sessions: list[dict[str, Any]] = []
        for row in response.data or []:
            sessions.append(
                {
                    "id": row.get("id"),
                    "started_at": row.get("started_at"),
                    "camera_ids": self._exam_session_camera_ids(str(row.get("id"))),
                }
            )
        return sessions

    def _exam_session_camera_ids(self, exam_session_id: str) -> list[str]:
        response = (
            self._client.table("exam_session_cameras")
            .select("camera_id")
            .eq("exam_session_id", exam_session_id)
            .execute()
        )
        return [str(row["camera_id"]) for row in (response.data or []) if row.get("camera_id")]

    def set_exam_session_runtime(
        self,
        exam_session_id: str,
        *,
        status: str,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
    ) -> None:
        payload: dict[str, Any] = {"status": status}
        if started_at is not None:
            payload["started_at"] = started_at.astimezone(timezone.utc).isoformat()
        if ended_at is not None:
            payload["ended_at"] = ended_at.astimezone(timezone.utc).isoformat()
        self._client.table("exam_sessions").update(payload).eq("id", exam_session_id).execute()

    # --- anonymous subject state -----------------------------------------
    def existing_subject_rows(self, exam_session_id: str) -> dict[int, str]:
        response = (
            self._client.table("session_subjects")
            .select("id,subject_number")
            .eq("exam_session_id", exam_session_id)
            .execute()
        )
        return {
            int(row["subject_number"]): str(row["id"])
            for row in (response.data or [])
            if row.get("subject_number") is not None
        }

    def open_subject_history(self, exam_session_id: str) -> list[dict[str, Any]]:
        """Subjects a previous run created, so their numbers stay reserved."""
        response = (
            self._client.table("session_subjects")
            .select("subject_number,first_seen_at,last_seen_at,lifecycle_status")
            .eq("exam_session_id", exam_session_id)
            .neq("lifecycle_status", "ended")
            .order("subject_number")
            .execute()
        )
        return [dict(row) for row in (response.data or [])]

    def allocate_subject_number(self, exam_session_id: str) -> int:
        """Atomic, gap-free, monotonic per-session number from the database.

        Delegating to the database is what makes numbering safe when several
        cameras — or several service instances — create subjects concurrently.
        """
        response = self._client.rpc(
            "allocate_session_subject_number",
            {"_exam_session_id": exam_session_id},
        ).execute()
        number = response.data
        if isinstance(number, list):
            number = number[0] if number else None
        if number is None:
            raise RuntimeError("subject number allocation returned no value")
        return int(number)

    def upsert_session_subject(self, payload: dict[str, Any]) -> Optional[str]:
        """Anonymous subject state only: number, lifecycle, times, motion.

        ``subject_number`` is written on insert and never changed afterwards; the
        database rejects any attempt to renumber or re-parent a subject.
        """
        motion = payload.get("motion")
        bbox = None if motion is None else motion.last_bbox
        row: dict[str, Any] = {
            "exam_session_id": payload["exam_session_id"],
            "subject_number": int(payload["subject_number"]),
            "camera_id": payload.get("camera_id"),
            "lifecycle_status": payload["lifecycle_status"],
            "track_association": payload["track_association"],
            "active_raw_tracking_id": payload.get("active_raw_tracking_id"),
            "first_seen_at": _iso(payload.get("first_seen_at")),
            "last_seen_at": _iso(payload.get("last_seen_at")),
            "ended_at": _iso(payload.get("ended_at")),
            "last_bbox_x": None if bbox is None else round(float(bbox.x), 6),
            "last_bbox_y": None if bbox is None else round(float(bbox.y), 6),
            "last_bbox_width": None if bbox is None else round(float(bbox.width), 6),
            "last_bbox_height": None if bbox is None else round(float(bbox.height), 6),
            "velocity_x": None if motion is None else round(float(motion.velocity_x), 6),
            "velocity_y": None if motion is None else round(float(motion.velocity_y), 6),
            "motion_updated_at": None if motion is None else _iso(motion.updated_at),
            "reassociation_count": int(payload.get("reassociation_count") or 0),
            "last_association_confidence": payload.get("last_association_confidence"),
        }
        response = (
            self._client.table("session_subjects")
            .upsert(row, on_conflict="exam_session_id,subject_number")
            .execute()
        )
        rows = response.data or []
        if rows and rows[0].get("id"):
            return str(rows[0]["id"])
        existing = self.existing_subject_rows(str(payload["exam_session_id"]))
        return existing.get(int(payload["subject_number"]))

    def open_subject_track(
        self,
        *,
        session_subject_id: str,
        exam_session_id: str,
        raw_tracking_id: str,
        started_at: datetime,
        association_method: str,
        association_confidence: Optional[float],
        start_reason: Optional[str] = None,
    ) -> None:
        """Records one raw-track segment. Raw ids are temporary labels only."""
        self._client.table("session_subject_tracks").insert(
            {
                "session_subject_id": session_subject_id,
                "exam_session_id": exam_session_id,
                "raw_tracking_id": raw_tracking_id,
                "started_at": _iso(started_at),
                "association_method": association_method,
                "association_confidence": association_confidence,
                "association_state": "confirmed",
                "start_reason": start_reason,
            }
        ).execute()

    def close_subject_track(
        self,
        *,
        exam_session_id: str,
        raw_tracking_id: str,
        ended_at: datetime,
        end_reason: Optional[str] = None,
    ) -> None:
        self._client.table("session_subject_tracks").update(
            {"ended_at": _iso(ended_at), "end_reason": end_reason}
        ).eq("exam_session_id", exam_session_id).eq(
            "raw_tracking_id", raw_tracking_id
        ).is_("ended_at", "null").execute()

