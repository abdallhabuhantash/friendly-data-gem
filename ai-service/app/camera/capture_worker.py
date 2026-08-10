"""Resilient latest-frame capture worker.

One lightweight thread per camera keeps only the newest frame, so a slow
inference loop can never build an unbounded backlog and a failing camera can
never block the others.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import cv2

from .source_builder import CaptureSource

logger = logging.getLogger(__name__)

RECONNECT_DELAYS = (1.0, 2.0, 5.0, 10.0)


@dataclass
class CaptureStats:
    connected: bool = False
    fps: float = 0.0
    frames: int = 0
    last_frame_at: Optional[datetime] = None
    last_error: Optional[str] = None
    reconnects: int = 0


class CaptureWorker:
    """Reads a single source and publishes only the most recent frame."""

    def __init__(self, camera_id: str, camera_name: str, source: CaptureSource) -> None:
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.source = source
        self.stats = CaptureStats()
        self._frame = None
        self._sequence = 0
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"capture-{self.camera_id[:8]}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._thread = None

    # --- frame access -----------------------------------------------------
    def latest_frame(self):
        with self._frame_lock:
            return None if self._frame is None else self._frame.copy()

    def latest_frame_with_sequence(self):
        """Newest frame plus its capture sequence number.

        The sequence lets consumers analyse each *distinct* captured frame at
        most once, so a frozen stream can never inflate `detection_frame_count`.
        Only the newest frame is retained: no backlog, no latency growth.
        """
        with self._frame_lock:
            if self._frame is None:
                return None, self._sequence
            return self._frame.copy(), self._sequence

    @property
    def frame_sequence(self) -> int:
        with self._frame_lock:
            return self._sequence

    def _publish(self, frame) -> None:
        with self._frame_lock:
            self._frame = frame
            self._sequence += 1

    # --- capture loop -----------------------------------------------------
    def _open(self):
        if self.source.kind == "rtsp":
            capture = cv2.VideoCapture(self.source.url, cv2.CAP_FFMPEG)
        else:
            capture = cv2.VideoCapture(self.source.url)
        try:
            # Keep only the newest frame in the driver buffer.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # pragma: no cover - backend dependent
            pass
        return capture

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            capture = self._open()
            if not capture or not capture.isOpened():
                self.stats.connected = False
                self.stats.last_error = "unable to open source"
                delay = RECONNECT_DELAYS[min(attempt, len(RECONNECT_DELAYS) - 1)]
                logger.warning(
                    "Camera %s (%s): cannot open %s, retrying in %.0fs",
                    self.camera_name,
                    self.camera_id,
                    self.source.safe_url,
                    delay,
                )
                attempt += 1
                self.stats.reconnects += 1
                self._stop.wait(delay)
                continue

            logger.info("Camera %s connected (%s)", self.camera_name, self.source.safe_url)
            attempt = 0
            self.stats.connected = True
            self.stats.last_error = None
            window_start = time.monotonic()
            window_frames = 0

            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    if self.source.kind == "file" and self.source.loop:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.stats.connected = False
                    self.stats.last_error = "stream ended"
                    break

                self._publish(frame)
                self.stats.frames += 1
                self.stats.last_frame_at = datetime.now(timezone.utc)
                window_frames += 1
                elapsed = time.monotonic() - window_start
                if elapsed >= 2.0:
                    self.stats.fps = window_frames / elapsed
                    window_start = time.monotonic()
                    window_frames = 0
                if self.source.kind == "file":
                    # Demo files play far faster than real time otherwise.
                    self._stop.wait(0.02)

            capture.release()
            self.stats.connected = False
            self.stats.fps = 0.0
            if not self._stop.is_set():
                self.stats.reconnects += 1
                delay = RECONNECT_DELAYS[min(attempt, len(RECONNECT_DELAYS) - 1)]
                attempt += 1
                logger.warning(
                    "Camera %s disconnected, reconnecting in %.0fs", self.camera_name, delay
                )
                self._stop.wait(delay)