"""Service lifecycle: configuration sync, inference loops and heartbeats.

One inference thread per camera consumes the newest captured frame, so a slow
or dead camera can never stall the others. A background control loop keeps the
configuration, heartbeats and durable retries moving independently of
inference.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from ..ai.association import associate
from ..ai.detector import YoloDetector
from ..ai.engine_registry import EngineRegistry, FrameContext, PhoneEngineAdapter
from ..ai.observation_builder import build_frame_observations
from ..ai.phone_rule_engine import PhoneRuleEngine
from ..camera.camera_manager import CameraManager
from ..domain.models import (
    ENGINE_MOBILE_PHONE,
    AssociationStatus,
    CameraConfig,
    RuleConfig,
    SystemConfig,
)

from ..events.snapshot_service import SnapshotService, annotate_frame, encode_jpeg
from ..events.event_publisher import EventPublisher
from ..infrastructure.credential_provider import (
    ChainedCredentialProvider,
    FileCredentialProvider,
    SupabaseCredentialProvider,
)
from ..infrastructure.offline_queue import OfflineQueue
from ..infrastructure.supabase_repository import DuplicateEventError, SupabaseRepository
from ..notifications.notification_manager import NotificationManager
from ..notifications.telegram import TelegramProvider
from .frame_gate import FrameGate
from .health_reporter import HealthReporter, measure_gpu_load
from .stream_hub import StreamHub

logger = logging.getLogger(__name__)


class Orchestrator:
    """Owns every long-lived resource of the AI service."""

    def __init__(self, settings) -> None:  # noqa: ANN001 - Settings
        self.settings = settings
        self.stream_hub = StreamHub()
        self.repository = SupabaseRepository(
            settings.supabase_url,
            settings.supabase_service_role_key,
            settings.snapshot_bucket,
        )
        self.queue = OfflineQueue(settings.state_path / "queue.db")

        sources = [FileCredentialProvider(settings.credentials_path)]
        if settings.use_supabase_camera_credentials:
            sources.append(SupabaseCredentialProvider(self.repository))
        self.credentials = ChainedCredentialProvider(sources)

        self.cameras = CameraManager(settings, self.credentials)
        self.snapshots = SnapshotService(self.repository, settings.snapshot_path)

        provider = None
        if settings.telegram_ready:
            provider = TelegramProvider(settings.telegram_bot_token, settings.telegram_chat_id)
            logger.info("Telegram notifications enabled")
        self.notifications = NotificationManager(
            self.queue, provider, send_warnings=settings.telegram_send_warnings
        )
        self.publisher = EventPublisher(
            self.repository,
            self.queue,
            snapshots=self.snapshots,
            notifications=self.notifications,
            duplicate_error=DuplicateEventError,
        )
        self.health = HealthReporter(self.repository, settings, self.queue)
        self.engine = PhoneRuleEngine(
            association_margin=settings.association_margin,
            gap_tolerance_seconds=settings.detection_gap_tolerance_seconds,
        )

        self.detector: Optional[YoloDetector] = None
        self.system = SystemConfig()
        self._rules: list[RuleConfig] = []
        self._stop = threading.Event()
        self._threads: dict[str, threading.Thread] = {}
        self._control: Optional[threading.Thread] = None
        self._inference_fps: dict[str, float] = {}
        self._frame_gate = FrameGate()
        self._started_at = time.monotonic()

    # --- lifecycle --------------------------------------------------------
    def start(self) -> None:
        problems = self.settings.validate_runtime()
        for problem in problems:
            logger.warning("Configuration: %s", problem)

        self.detector = YoloDetector(
            self.settings.yolo_model,
            self.settings.yolo_device,
            self.settings.yolo_imgsz,
            self.settings.yolo_tracker,
        )
        self._refresh_configuration()
        self._control = threading.Thread(target=self._control_loop, name="control", daemon=True)
        self._control.start()
        logger.info("AI service started in %s mode", self.system.operation_mode)

    def stop(self) -> None:
        self._stop.set()
        self.cameras.stop_all()
        for thread in self._threads.values():
            thread.join(timeout=3.0)
        self._threads.clear()
        if self._control:
            self._control.join(timeout=3.0)
        try:
            self.health.beat(
                online=False,
                is_demo=self.system.operation_mode == "demo",
                payload=self._health_payload(),
            )
        finally:
            self.queue.close()
        logger.info("AI service stopped")

    # --- configuration ----------------------------------------------------
    def _refresh_configuration(self) -> None:
        try:
            self.system = self.repository.system_config()
            cameras = self.repository.cameras(self.system.operation_mode)
            self._rules = self.repository.rules()
        except Exception as exc:
            logger.warning("Configuration refresh failed: %s", type(exc).__name__)
            return

        self.cameras.sync(cameras)
        active = set(self.cameras.active)

        for camera_id in list(self._threads):
            if camera_id not in active:
                self._threads.pop(camera_id, None)
                self.registry.reset(camera_id)
                self.stream_hub.drop(camera_id)
                self._inference_fps.pop(camera_id, None)
                self._frame_gate.reset(camera_id)
                if self.detector:
                    self.detector.reset_camera(camera_id)

        for camera_id in active:
            thread = self._threads.get(camera_id)
            if thread and thread.is_alive():
                continue
            thread = threading.Thread(
                target=self._inference_loop,
                args=(camera_id,),
                name=f"infer-{camera_id[:8]}",
                daemon=True,
            )
            self._threads[camera_id] = thread
            thread.start()

    def _rules_for(self, camera: CameraConfig) -> list[RuleConfig]:
        """Every enabled, available rule assigned to this camera, any engine."""
        return [
            rule
            for rule in self._rules
            if rule.enabled and rule.available and rule.applies_to(camera.id)
        ]

    @staticmethod
    def _phone_rules(rules: list[RuleConfig]) -> list[RuleConfig]:
        """Phone-engine rules only: the sole input to phone annotation logic."""
        return [rule for rule in rules if rule.engine_key == ENGINE_MOBILE_PHONE]


    # --- inference --------------------------------------------------------
    def _inference_loop(self, camera_id: str) -> None:
        # `inference_max_fps <= 0` means "run as fast as the model actually
        # allows" — the loop is then paced only by real inference time, never by
        # an artificial ceiling. No frame queue exists either way.
        max_fps = float(self.settings.inference_max_fps)
        min_interval = (1.0 / max_fps) if max_fps > 0 else 0.0
        processed = 0
        window_start = time.monotonic()
        window_frames = 0

        while not self._stop.is_set():
            worker = self.cameras.worker(camera_id)
            camera = self.cameras.config(camera_id)
            if worker is None or camera is None:
                return

            cycle_start = time.monotonic()
            frame, sequence = worker.latest_frame_with_sequence()
            if frame is None:
                self._stop.wait(0.2)
                continue
            if not self._frame_gate.accept(camera_id, sequence):
                # The same captured frame must never be analysed twice: a frozen
                # stream would otherwise fake multiple matching frames.
                self._stop.wait(0.005)
                continue

            processed += 1
            if self.settings.process_every_n_frames > 1 and (
                processed % int(self.settings.process_every_n_frames)
            ):
                self._stop.wait(0.005)
                continue

            try:
                self._process_frame(camera, frame, frame_sequence=sequence)
            except Exception as exc:  # one camera never takes down the service
                logger.exception("Inference failed for camera %s: %s", camera.name, exc)
                self._stop.wait(0.5)
                continue

            window_frames += 1
            elapsed = time.monotonic() - window_start
            if elapsed >= 2.0:
                self._inference_fps[camera_id] = window_frames / elapsed
                window_start = time.monotonic()
                window_frames = 0

            remaining = min_interval - (time.monotonic() - cycle_start)
            if remaining > 0:
                self._stop.wait(remaining)

    def _process_frame(
        self, camera: CameraConfig, frame, frame_sequence: Optional[int] = None
    ) -> None:
        assert self.detector is not None
        detections = self.detector.detect(frame, camera.id)
        applicable_rules = self._rules_for(camera)
        # Only mobile-phone rules may influence phone annotation/association
        # rendering: a behavioural rule must never move Task 1 thresholds.
        phone_rules = self._phone_rules(applicable_rules)

        # Annotation uses the most permissive thresholds across the phone rules
        # so the operator sees every detection the phone engine will evaluate.
        associations: dict = {}
        if phone_rules:
            min_person_conf = min(r.person_confidence_threshold for r in phone_rules)
            min_phone_conf = min(r.confidence_threshold for r in phone_rules)
            min_assoc_conf = min(r.association_confidence_threshold for r in phone_rules)
            persons = tuple(
                person
                for person in detections.persons
                if person.confidence >= min_person_conf and person.tracking_id
            )
            for index, phone in enumerate(detections.phones):
                if phone.confidence < min_phone_conf:
                    continue
                associations[phone.tracking_id or f"idx{index}"] = associate(
                    phone,
                    persons,
                    association_threshold=min_assoc_conf,
                    margin=self.settings.association_margin,
                )

        annotated = annotate_frame(
            frame,
            detections,
            camera_name=camera.name,
            associations=associations,
            timestamp=datetime.now(),
        )
        jpeg = encode_jpeg(annotated)
        if jpeg:
            self.stream_hub.publish(camera.id, jpeg)

        if not applicable_rules:
            return

        # One detection pass feeds every applicable rule through the registry:
        # no inference duplication, no silently ignored rules, and one failing
        # engine never suppresses another engine's events for this frame.
        now_mono = time.monotonic()
        detected_at = datetime.now(timezone.utc)
        observations = build_frame_observations(
            camera_id=camera.id,
            detections=detections,
            frame_sequence=frame_sequence,
            observed_at=detected_at,
            source_mode=self.system.operation_mode,
        )
        context = FrameContext(
            camera=camera,
            detections=detections,
            observations=observations,
            now=now_mono,
            source_mode=self.system.operation_mode,
            detected_at=detected_at,
        )
        for draft in self.registry.dispatch(applicable_rules, context):
            # `annotated` is derived from exactly the frame that produced
            # this draft, so an instant single-frame event can never be
            # snapshotted with a later frame where the phone has vanished.
            self.publisher.publish(
                draft.event, frame=annotated, save_snapshot=draft.save_snapshot
            )


    # --- control loop -----------------------------------------------------
    def _health_payload(self) -> dict:
        fps_values = list(self._inference_fps.values())
        return self.health.payload(
            model=self.settings.yolo_model,
            device=self.detector.device if self.detector else "unknown",
            inference_fps=(sum(fps_values) / len(fps_values)) if fps_values else 0.0,
            gpu_load_percent=measure_gpu_load(),
        )

    def _control_loop(self) -> None:
        last_config = 0.0
        last_health = 0.0
        last_cameras = 0.0

        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_config >= self.settings.config_refresh_seconds:
                self._refresh_configuration()
                last_config = now
            if now - last_health >= self.settings.health_heartbeat_seconds:
                self.health.beat(
                    online=True,
                    is_demo=self.system.operation_mode == "demo",
                    payload=self._health_payload(),
                )
                last_health = now
            if now - last_cameras >= self.settings.camera_heartbeat_seconds:
                self._camera_heartbeats()
                last_cameras = now

            self.publisher.retry_pending()
            self.publisher.retry_pending_evidence()
            self.notifications.drain()
            self._stop.wait(1.0)


    def _camera_heartbeats(self) -> None:
        """Reports only observed connectivity: never optimistic, never guessed."""
        for camera_id, worker in self.cameras.active.items():
            stats = worker.stats
            if stats.connected and stats.last_frame_at is not None:
                self.health.camera_beat(
                    camera_id,
                    status="online",
                    fps=self._inference_fps.get(camera_id, stats.fps),
                    heartbeat_at=stats.last_frame_at,
                )
            else:
                self.health.camera_beat(
                    camera_id, status="offline", fps=0.0, heartbeat_at=stats.last_frame_at
                )

    # --- introspection ----------------------------------------------------
    def status(self) -> dict:
        return {
            "version": self.settings.service_version,
            "operation_mode": self.system.operation_mode,
            "uptime_seconds": int(time.monotonic() - self._started_at),
            "model": self.settings.yolo_model,
            "device": self.detector.device if self.detector else "unknown",
            "cameras": [
                {
                    "id": camera_id,
                    "name": worker.camera_name,
                    "connected": worker.stats.connected,
                    "capture_fps": round(worker.stats.fps, 2),
                    "inference_fps": round(self._inference_fps.get(camera_id, 0.0), 2),
                    "streaming": self.stream_hub.has(camera_id),
                }
                for camera_id, worker in self.cameras.active.items()
            ],
            "queue": {
                "events": self.queue.event_depth(),
                "notifications": self.queue.notification_depth(),
                "evidence": self.queue.evidence_depth(),
            },

            "notifications": {
                "telegram": {
                    "configured": self.settings.telegram_configured,
                    "ready": self.settings.telegram_ready,
                }
            },
        }


__all__ = ["Orchestrator", "AssociationStatus"]