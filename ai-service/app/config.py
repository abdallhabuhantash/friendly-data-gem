"""Environment-driven configuration. Secrets live only in the local .env."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All runtime configuration. Never logged, never returned by the API."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "development"
    service_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # --- Supabase (service role, local backend only) ---
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    snapshot_bucket: str = "snapshots"

    # --- Operational endpoint auth ---
    ai_service_key: str = ""

    # --- Model ---
    yolo_model: str = "yolo11n.pt"
    yolo_device: str = "auto"
    yolo_imgsz: int = 960
    yolo_tracker: str = "bytetrack.yaml"

    # --- Loops ---
    config_refresh_seconds: float = 10.0
    health_heartbeat_seconds: float = 10.0
    camera_heartbeat_seconds: float = 10.0
    # 0 (default) removes the artificial ceiling: each camera's inference loop
    # then runs exactly as fast as real model execution allows, which matters
    # for phones that are visible for only a fraction of a second. Set a
    # positive value to cap CPU/GPU usage. Actual throughput always depends on
    # hardware, model, resolution and camera count — no FPS is promised.
    inference_max_fps: float = 0.0
    # Never skip frames by default: a skipped frame is evidence thrown away.
    process_every_n_frames: int = 1

    # --- Detection tuning ---
    association_margin: float = 0.12
    detection_gap_tolerance_seconds: float = 0.5

    # --- Pose (optional capability, OFF by default) ---
    # Nothing pose-related is constructed, loaded or copied while disabled.
    # When POSE_ENABLED=true every field below must be supplied EXPLICITLY:
    # there is no calibrated deployment default for a model, a device, an input
    # size, a confidence floor or a cadence, so none is invented here.
    pose_enabled: bool = False
    pose_model: str = ""
    pose_device: Optional[str] = None
    pose_imgsz: Optional[int] = None
    pose_confidence: Optional[float] = None
    # Explicit pose cadence: never derived from capture FPS and never from
    # PROCESS_EVERY_N_FRAMES (Task 1 keeps its own frame policy). Not calibrated.
    pose_max_fps: Optional[float] = None
    # Association thresholds are deliberately UNSET by default: no deployment
    # calibration exists, so pose association stays unconfigured until an
    # operator supplies all four values.
    pose_assoc_min_bbox_iou: Optional[float] = None
    pose_assoc_min_pose_containment: Optional[float] = None
    pose_assoc_min_available_keypoints: Optional[int] = None
    pose_assoc_min_keypoint_inside_ratio: Optional[float] = None



    # --- Demo sources ---
    demo_video_path: str = ""
    demo_video_paths_json: str = ""
    demo_video_loop: bool = True

    # --- Camera credentials ---
    camera_credentials_file: str = "./secrets/cameras.json"
    use_supabase_camera_credentials: bool = False

    # --- Storage paths ---
    snapshot_dir: str = "./snapshots"
    state_dir: str = "./state"

    # --- Telegram ---
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_send_warnings: bool = False

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (BASE_DIR / path).resolve()

    @property
    def snapshot_path(self) -> Path:
        return self.resolve(self.snapshot_dir)

    @property
    def state_path(self) -> Path:
        return self.resolve(self.state_dir)

    @property
    def credentials_path(self) -> Path:
        return self.resolve(self.camera_credentials_file)

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def telegram_ready(self) -> bool:
        return self.telegram_enabled and self.telegram_configured

    def demo_video_for(self, camera_id: str) -> Optional[str]:
        """Per-camera demo file, falling back to the single DEMO_VIDEO_PATH."""
        if self.demo_video_paths_json:
            try:
                mapping = json.loads(self.demo_video_paths_json)
                if isinstance(mapping, dict) and mapping.get(camera_id):
                    return str(mapping[camera_id])
            except json.JSONDecodeError:
                pass
        return self.demo_video_path or None

    # --- Pose configuration truthfulness ---------------------------------
    @property
    def pose_inference_problems(self) -> list[str]:
        """Pose inference configuration problems; empty means usable."""
        problems: list[str] = []
        if not self.pose_enabled:
            return problems
        if not str(self.pose_model).strip():
            problems.append("POSE_ENABLED=true but POSE_MODEL is not set")
        if not str(self.pose_device).strip():
            problems.append("POSE_DEVICE must be a non-empty string")
        if int(self.pose_imgsz) <= 0:
            problems.append("POSE_IMGSZ must be a positive integer")
        if not 0.0 <= float(self.pose_confidence) <= 1.0:
            problems.append("POSE_CONFIDENCE must be within 0..1")
        if float(self.pose_max_fps) <= 0.0:
            problems.append("POSE_MAX_FPS must be greater than 0")
        return problems

    @property
    def pose_association_problems(self) -> list[str]:
        """Association threshold problems; empty means a complete valid spec."""
        if not self.pose_enabled:
            return []
        values = {
            "POSE_ASSOC_MIN_BBOX_IOU": self.pose_assoc_min_bbox_iou,
            "POSE_ASSOC_MIN_POSE_CONTAINMENT": self.pose_assoc_min_pose_containment,
            "POSE_ASSOC_MIN_KEYPOINT_INSIDE_RATIO": self.pose_assoc_min_keypoint_inside_ratio,
        }
        problems: list[str] = []
        missing = [name for name, value in values.items() if value is None]
        if self.pose_assoc_min_available_keypoints is None:
            missing.append("POSE_ASSOC_MIN_AVAILABLE_KEYPOINTS")
        if missing:
            problems.append(
                "pose association configuration incomplete: "
                + ", ".join(sorted(missing))
                + " (pose association stays unconfigured)"
            )
            return problems
        for name, value in values.items():
            if not 0.0 <= float(value) <= 1.0:  # type: ignore[arg-type]
                problems.append(f"{name} must be within 0..1")
        keypoints = int(self.pose_assoc_min_available_keypoints)  # type: ignore[arg-type]
        if not 1 <= keypoints <= 17:
            problems.append("POSE_ASSOC_MIN_AVAILABLE_KEYPOINTS must be within 1..17")
        return problems

    @property
    def pose_inference_configured(self) -> bool:
        return self.pose_enabled and not self.pose_inference_problems

    @property
    def pose_association_configured(self) -> bool:
        return self.pose_enabled and not self.pose_association_problems

    @property
    def pose_min_interval_seconds(self) -> float:
        fps = float(self.pose_max_fps)
        return (1.0 / fps) if fps > 0 else 0.0

    def validate_runtime(self) -> list[str]:
        """Returns human-readable configuration problems (never secret values)."""
        problems: list[str] = []
        if not self.supabase_url:
            problems.append("SUPABASE_URL is not set")
        if not self.supabase_service_role_key:
            problems.append("SUPABASE_SERVICE_ROLE_KEY is not set")
        if not self.ai_service_key:
            problems.append("AI_SERVICE_KEY is not set (stream endpoint stays closed)")
        # Pose is optional: its problems are reported, never fatal.
        problems.extend(self.pose_inference_problems)
        problems.extend(self.pose_association_problems)
        return problems



@lru_cache
def get_settings() -> Settings:
    return Settings()