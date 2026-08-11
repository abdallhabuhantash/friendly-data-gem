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
    pose_enabled: bool = False
    pose_model: str = ""
    pose_device: str = "cpu"
    pose_imgsz: int = 640
    pose_confidence: float = 0.25
    # Explicit pose cadence: never derived from capture FPS and never from
    # PROCESS_EVERY_N_FRAMES (Task 1 keeps its own frame policy). Not calibrated.
    pose_max_fps: float = 1.0
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

    def validate_runtime(self) -> list[str]:
        """Returns human-readable configuration problems (never secret values)."""
        problems: list[str] = []
        if not self.supabase_url:
            problems.append("SUPABASE_URL is not set")
        if not self.supabase_service_role_key:
            problems.append("SUPABASE_SERVICE_ROLE_KEY is not set")
        if not self.ai_service_key:
            problems.append("AI_SERVICE_KEY is not set (stream endpoint stays closed)")
        return problems


@lru_cache
def get_settings() -> Settings:
    return Settings()