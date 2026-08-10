"""Small explicit engine registry: `engine_key` -> engine handler.

Design constraints that this module deliberately honours:

* No universal engine signature is imposed. `PhoneRuleEngine` keeps its stable
  `process_frame(detections=...)` contract and is wrapped by a thin adapter.
  Future behavioural engines may consume `FrameObservations` instead.
* Dispatch is per rule via `rule.engine_key`. An unknown key is skipped: it
  never falls back to phone detection.
* Failures are isolated per rule/engine, so one broken engine cannot suppress
  another engine's events for the same frame.
* No cross-engine state, no evidence fusion, no shared cooldowns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from ..domain.models import CameraConfig, FrameDetections, RuleConfig, SourceMode
from ..domain.observations import FrameObservations

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameContext:
    """Everything an engine may need about one analysed frame."""

    camera: CameraConfig
    #: Detector source of truth (consumed by the frozen phone pipeline).
    detections: FrameDetections
    #: Derived immutable view (for future behavioural engines).
    observations: FrameObservations
    now: float
    source_mode: SourceMode
    detected_at: Optional[datetime] = None


@runtime_checkable
class EngineHandler(Protocol):
    """Narrow adapter contract used by the registry."""

    def process(self, rule: RuleConfig, context: FrameContext) -> list:
        """Returns event drafts for one rule on one frame."""

    def reset(self, camera_id: str) -> None:
        """Drops all per-camera state for this camera only."""


class PhoneEngineAdapter:
    """Adapts the unchanged `PhoneRuleEngine` to the registry contract."""

    def __init__(self, engine) -> None:  # noqa: ANN001 - PhoneRuleEngine
        self.engine = engine

    def process(self, rule: RuleConfig, context: FrameContext) -> list:
        # Task 1 keeps consuming FrameDetections directly; observations are not
        # forwarded, so phone behaviour cannot depend on the derived view.
        return self.engine.process_frame(
            camera=context.camera,
            rule=rule,
            detections=context.detections,
            now=context.now,
            source_mode=context.source_mode,
            detected_at=context.detected_at,
        )

    def reset(self, camera_id: str) -> None:
        self.engine.reset(camera_id)


class EngineRegistry:
    """Explicit `engine_key` -> handler map with isolated dispatch."""

    def __init__(self) -> None:
        self._handlers: dict[str, EngineHandler] = {}
        self._unknown_reported: set[str] = set()

    # --- registration -----------------------------------------------------
    def register(self, engine_key: str, handler: EngineHandler) -> None:
        self._handlers[engine_key] = handler

    def unregister(self, engine_key: str) -> None:
        self._handlers.pop(engine_key, None)

    def handler(self, engine_key: Optional[str]) -> Optional[EngineHandler]:
        if not engine_key:
            return None
        return self._handlers.get(engine_key)

    @property
    def engine_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    # --- dispatch ---------------------------------------------------------
    def dispatch(self, rules, context: FrameContext) -> list:
        """Dispatches every rule to its engine; failures never cascade."""
        drafts: list = []
        for rule in rules:
            handler = self.handler(rule.engine_key)
            if handler is None:
                self._report_unknown(context.camera.id, rule)
                continue
            try:
                produced = handler.process(rule, context)
            except Exception as exc:
                # Safe context only: never camera URLs, credentials or tokens.
                logger.error(
                    "Engine failed (camera=%s rule=%s engine=%s): %s: %s",
                    context.camera.id,
                    rule.id,
                    rule.engine_key,
                    type(exc).__name__,
                    exc,
                )
                continue
            if produced:
                drafts.extend(produced)
        return drafts

    def _report_unknown(self, camera_id: str, rule: RuleConfig) -> None:
        """Bounded warning: one line per unknown rule, not per frame."""
        marker = f"{camera_id}:{rule.id}:{rule.engine_key}"
        if marker in self._unknown_reported:
            return
        self._unknown_reported.add(marker)
        logger.warning(
            "No engine registered for rule %s (engine_key=%r) on camera %s: rule skipped",
            rule.id,
            rule.engine_key,
            camera_id,
        )

    # --- lifecycle --------------------------------------------------------
    def reset(self, camera_id: str) -> None:
        """Clears per-camera state in every registered engine."""
        for engine_key, handler in self._handlers.items():
            reset = getattr(handler, "reset", None)
            if reset is None:
                continue
            try:
                reset(camera_id)
            except Exception as exc:
                logger.error(
                    "Engine reset failed (camera=%s engine=%s): %s",
                    camera_id,
                    engine_key,
                    type(exc).__name__,
                )
        self._unknown_reported = {
            marker for marker in self._unknown_reported if not marker.startswith(f"{camera_id}:")
        }
