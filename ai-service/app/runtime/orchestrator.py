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
from .pose_runtime import PoseRuntime
from .stream_hub import StreamHub


logger = logging.getLogger(__name__)


class Orchestrator:
    """Owns every long-lived resource of the AI service."""

    def __init__(self, settings, pose_provider_factory=None) -> None:  # noqa: ANN001 - Settings
        self.settings = settings
        # Injectable ONLY so tests need no pose weights; production stays lazy.
        self._pose_provider_factory = pose_provider_factory

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
        # Explicit engine map: only the implemented phone engine is registered.
        self.registry = EngineRegistry()
        self.registry.register(ENGINE_MOBILE_PHONE, PhoneEngineAdapter(self.engine))


        self.detector: Optional[YoloDetector] = None
        # Optional, asynchronous, OFF unless explicitly configured.
        self.pose: Optional[PoseRuntime] = None
        self._pose_problems: list[str] = []

        self.system = SystemConfig()
        self._rules: list[RuleConfig] = []
        self._stop = threading.Event()
        self._threads: dict[str, threading.Thread] = {}
        self._control: Optional[threading.Thread] = None
        self._inference_fps: dict[str, float] = {}
        self._frame_gate = FrameGate()
        # Stream-incarnation accounting: the generation each camera's inference
        # loop has initialised, plus per-incarnation counters.
        self._seen_generation: dict[str, int] = {}
        self._processed_frames: dict[str, int] = {}
        self._fps_window: dict[str, tuple[float, int]] = {}
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
        # Pose is optional: a pose configuration problem must never prevent
        # Task 1 from starting, and never raise out of start().
        self._start_pose_runtime()
        self._refresh_configuration()
        self._control = threading.Thread(target=self._control_loop, name="control", daemon=True)
        self._control.start()
        logger.info("AI service started in %s mode", self.system.operation_mode)

    def _start_pose_runtime(self) -> None:
        """Constructs the pose runtime ONLY when explicitly enabled + valid."""
        settings = self.settings
        self._pose_problems = list(settings.pose_inference_problems) + list(
            settings.pose_association_problems
        )
        if not settings.pose_enabled:
            return
        if not settings.pose_inference_configured:
            logger.warning("Pose enabled but not usable: configuration incomplete")
            return
        try:
            factory = self._pose_provider_factory
            if factory is not None:
                provider = factory()
            else:
                # Lazy weights: constructing the provider loads nothing.
                from ..ai.pose_provider import UltralyticsPoseProvider

                provider = UltralyticsPoseProvider(
                    settings.pose_model,
                    device=settings.pose_device,
                    imgsz=int(settings.pose_imgsz),
                    confidence=float(settings.pose_confidence),
                )
            spec = None
            if settings.pose_association_configured:
                from ..domain.pose_association import PoseAssociationSpec

                spec = PoseAssociationSpec(
                    min_bbox_iou=float(settings.pose_assoc_min_bbox_iou),
                    min_pose_bbox_containment=float(settings.pose_assoc_min_pose_containment),
                    min_available_keypoints=int(settings.pose_assoc_min_available_keypoints),
                    min_keypoint_inside_ratio=float(
                        settings.pose_assoc_min_keypoint_inside_ratio
                    ),
                )
            runtime = PoseRuntime(
                provider,
                min_interval_seconds=settings.pose_min_interval_seconds,
                association_spec=spec,
            )
            runtime.start()
            self.pose = runtime
            logger.info("Pose runtime started (association configured: %s)", spec is not None)
        except Exception as error:  # noqa: BLE001 - optional capability only
            self.pose = None
            logger.warning("Pose runtime unavailable (%s)", type(error).__name__)

    def stop(self) -> None:
        self._stop.set()
        self.cameras.stop_all()
        for thread in self._threads.values():
            thread.join(timeout=3.0)
        self._threads.clear()
        if self.pose:
            self.pose.stop(timeout=3.0)
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

        reconfigured = self.cameras.sync(cameras) or set()
        active = set(self.cameras.active)

        # A same-id source replacement must not inherit runtime state from the
        # previous stream incarnation, so it uses exactly the removal cleanup.
        # The per-camera lifecycle lock makes the reset wait for any in-flight
        # frame of that SAME camera to leave its state-mutating section first.
        for camera_id in reconfigured:
            self._transition_generation(camera_id)

        for camera_id in list(self._threads):
            if camera_id not in active:
                self._threads.pop(camera_id, None)
                with self.cameras.lock(camera_id):
                    self._reset_camera_runtime(camera_id)
                    self._seen_generation.pop(camera_id, None)
                # Removal never waits for an in-flight pose inference: the pose
                # worker discards its result because the camera is deactivated.
                if self.pose:
                    self.pose.deactivate(camera_id)

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

    def _transition_generation(self, camera_id: str) -> Optional[int]:
        """Moves a camera to its current stream incarnation, exactly once.

        Runs under the camera's own lifecycle lock, so it cannot interleave with
        that camera's frame processing. Whoever gets there first — the control
        thread after `sync()` or the inference thread noticing a new generation —
        performs the reset; the other one sees the generation already recorded
        and does nothing. Returns the generation now initialised, or None when
        the camera has no running incarnation.
        """
        with self.cameras.lock(camera_id):
            generation = self.cameras.generation(camera_id)
            if generation is None:
                return None
            if self._seen_generation.get(camera_id) == generation:
                return generation
            self._reset_camera_runtime(camera_id)
            self._seen_generation[camera_id] = generation
        # Activation happens after the old incarnation's pose state is gone, so
        # a late generation-N pose result can never be stored as generation N+1.
        if self.pose:
            self.pose.activate(camera_id, generation)
        return generation

    def _reset_camera_runtime(self, camera_id: str) -> None:
        """Idempotently drops all runtime state of ONE camera.

        Shared by camera removal and same-id source replacement: engine state,
        distinct-frame gate, tracker state, published stream frame and the
        inference FPS measurement all belong to a single stream incarnation.
        Never touches any other camera and never stops a capture worker.
        Callers hold that camera's lifecycle lock.
        """
        self.registry.reset(camera_id)
        self.stream_hub.drop(camera_id)
        self._inference_fps.pop(camera_id, None)
        self._processed_frames.pop(camera_id, None)
        self._fps_window.pop(camera_id, None)
        self._frame_gate.reset(camera_id)
        if self.detector:
            self.detector.reset_camera(camera_id)
        if self.pose:
            # Pending job, latest result, cadence timestamps and incarnation
            # metrics all belong to the incarnation being dropped.
            self.pose.reset_camera(camera_id)




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

        while not self._stop.is_set():
            runtime = self.cameras.snapshot(camera_id)
            if runtime is None:
                return

            if self._seen_generation.get(camera_id) != runtime.generation:
                # The replacement worker may become visible before the control
                # thread reaches its cleanup: whoever arrives first performs the
                # transition, under this camera's lifecycle lock.
                if self._transition_generation(camera_id) is None:
                    return
                continue

            cycle_start = time.monotonic()
            frame, sequence = runtime.worker.latest_frame_with_sequence()
            if frame is None:
                self._stop.wait(0.2)
                continue

            try:
                analysed = self._guarded_process(runtime, frame, sequence)
            except Exception as exc:  # one camera never takes down the service
                logger.exception("Inference failed for camera %s: %s", runtime.config.name, exc)
                self._stop.wait(0.5)
                continue

            if not analysed:
                self._stop.wait(0.005)
                continue

            remaining = min_interval - (time.monotonic() - cycle_start)
            if remaining > 0:
                self._stop.wait(remaining)

    def _guarded_process(self, runtime, frame, sequence) -> bool:  # noqa: ANN001
        """Runs the state-mutating part of one frame for ONE stream incarnation.

        Everything that touches per-camera runtime state — frame gate, engines,
        tracker state, published frame, FPS — happens while this camera's
        lifecycle lock is held, so a concurrent removal/reconfiguration must wait
        for the frame to leave this section before its final reset. The
        generation is revalidated inside the lock, so a frame captured by a
        previous incarnation can never repopulate state after a reset.
        """
        camera_id = runtime.camera_id
        with self.cameras.lock(camera_id):
            if self.cameras.generation(camera_id) != runtime.generation:
                return False
            if self._seen_generation.get(camera_id) != runtime.generation:
                return False
            if not self._frame_gate.accept(camera_id, sequence):
                # The same captured frame must never be analysed twice: a frozen
                # stream would otherwise fake multiple matching frames.
                return False

            count = self._processed_frames.get(camera_id, 0) + 1
            self._processed_frames[camera_id] = count
            every = int(self.settings.process_every_n_frames)
            if every > 1 and count % every:
                return False

            observations = self._process_frame(
                runtime.config, frame, frame_sequence=sequence
            )
            self._record_inference_fps(camera_id)

        # Pose submission happens AFTER the camera lifecycle lock is released and
        # never runs pose inference here: at most a cadence check, one frame copy
        # and a pending-slot replacement. Task 1 has already completed above.
        self._submit_pose(runtime, frame, sequence, observations)
        return True

    def _submit_pose(self, runtime, frame, sequence, observations) -> None:  # noqa: ANN001
        """Cheap, non-blocking hand-off of one frame to the pose worker."""
        pose = self.pose
        if pose is None or observations is None:
            return
        try:
            pose.maybe_submit(
                camera_id=runtime.camera_id,
                generation=runtime.generation,
                frame_sequence=sequence,
                observed_at=observations.observed_at,
                observations=observations,
                # Copy happens only for a frame the cadence actually admits.
                copy_frame=lambda: frame.copy() if hasattr(frame, "copy") else frame,
                source_mode=observations.source_mode,
            )
        except Exception as error:  # noqa: BLE001 - pose must never break Task 1
            logger.warning(
                "Pose submission skipped for camera %s (%s)",
                runtime.camera_id,
                type(error).__name__,
            )


    def _record_inference_fps(self, camera_id: str) -> None:
        """Per-camera FPS window; belongs to the current incarnation only."""
        window_start, frames = self._fps_window.get(camera_id, (time.monotonic(), 0))
        frames += 1
        elapsed = time.monotonic() - window_start
        if elapsed >= 2.0:
            self._inference_fps[camera_id] = frames / elapsed
            self._fps_window[camera_id] = (time.monotonic(), 0)
        else:
            self._fps_window[camera_id] = (window_start, frames)


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