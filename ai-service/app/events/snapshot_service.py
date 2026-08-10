"""Annotated frame rendering and snapshot persistence.

The same renderer feeds both the live MJPEG stream and the stored evidence
snapshot, so what a reviewer sees matches what the operator saw.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from ..domain.models import (
    AiEvent,
    AssociationResult,
    AssociationStatus,
    CLASS_PHONE,
    Detection,
    FrameDetections,
)

logger = logging.getLogger(__name__)

COLOR_PERSON = (208, 178, 34)  # BGR cyan-teal
COLOR_PHONE = (60, 200, 255)  # amber
COLOR_ASSOCIATED = (80, 80, 240)  # red
COLOR_UNCERTAIN = (0, 200, 255)  # yellow
COLOR_TEXT = (240, 240, 240)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(frame, text: str, origin: tuple[int, int], color) -> None:
    x, y = origin
    (width, height), _ = cv2.getTextSize(text, FONT, 0.42, 1)
    cv2.rectangle(frame, (x, max(0, y - height - 6)), (x + width + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 4), FONT, 0.42, (12, 18, 32), 1, cv2.LINE_AA)


def annotate_frame(
    frame,
    detections: FrameDetections,
    *,
    camera_name: str,
    associations: Optional[dict[str, AssociationResult]] = None,
    timestamp: Optional[datetime] = None,
):
    """Draws persons, phones and association state onto a copy of the frame."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    links = associations or {}

    person_boxes: dict[str, Detection] = {
        person.tracking_id: person for person in detections.persons if person.tracking_id
    }

    for person in detections.persons:
        x1, y1, x2, y2 = person.bbox.to_pixels(width, height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_PERSON, 1)
        tag = f"PERSON {person.tracking_id or '--'}  {person.confidence:.2f}"
        _label(canvas, tag, (x1, max(14, y1)), COLOR_PERSON)

    for index, phone in enumerate(detections.phones):
        key = phone.tracking_id or f"idx{index}"
        association = links.get(key)
        x1, y1, x2, y2 = phone.bbox.to_pixels(width, height)
        color = COLOR_PHONE
        caption = f"PHONE {phone.confidence:.2f}"
        if association and association.status is AssociationStatus.ASSOCIATED:
            color = COLOR_ASSOCIATED
            caption = (
                f"PHONE {phone.confidence:.2f} -> ID {association.person_tracking_id}"
                f" ({association.confidence:.2f})"
            )
        elif association and association.status is AssociationStatus.UNCERTAIN:
            color = COLOR_UNCERTAIN
            caption = f"PHONE {phone.confidence:.2f} - UNCERTAIN ASSOCIATION"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        _label(canvas, caption, (x1, max(14, y1)), color)

        # A connector line is drawn only for a definitive association.
        if (
            association
            and association.status is AssociationStatus.ASSOCIATED
            and association.person_tracking_id in person_boxes
        ):
            person = person_boxes[association.person_tracking_id]
            pcx, pcy = person.bbox.center
            fcx, fcy = phone.bbox.center
            cv2.line(
                canvas,
                (int(fcx * width), int(fcy * height)),
                (int(pcx * width), int(pcy * height)),
                COLOR_ASSOCIATED,
                1,
            )

    stamp = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    header = f"{camera_name}  |  {stamp}  |  VIGILANT EYE"
    cv2.rectangle(canvas, (0, 0), (width, 20), (18, 26, 44), -1)
    cv2.putText(canvas, header, (6, 14), FONT, 0.42, COLOR_TEXT, 1, cv2.LINE_AA)
    return canvas


def encode_jpeg(frame, quality: int = 75) -> Optional[bytes]:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else None


def object_path_for(event: AiEvent) -> str:
    """events/YYYY/MM/DD/{camera_id}/{event_id}.jpg inside the private bucket."""
    moment = event.detected_at
    return (
        f"events/{moment:%Y}/{moment:%m}/{moment:%d}/{event.camera_id}/{event.id}.jpg"
    )


class SnapshotService:
    """Writes the annotated JPEG locally, then uploads it to private storage."""

    def __init__(self, repository, snapshot_dir: Path) -> None:  # noqa: ANN001
        self._repository = repository
        self._dir = snapshot_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def write_local(self, event: AiEvent, frame) -> Optional[Path]:
        jpeg = encode_jpeg(frame, quality=85)
        if jpeg is None:
            return None
        target = self._dir / f"{event.id}.jpg"
        target.write_bytes(jpeg)
        return target

    def upload(self, event: AiEvent, local_file: Path) -> Optional[str]:
        """Returns the stored object path, or None when the upload failed."""
        try:
            return self._repository.upload_snapshot(object_path_for(event), local_file)
        except Exception as exc:  # never lose a detection over an upload error
            logger.error(
                "Snapshot upload failed for event %s: %s", event.id, type(exc).__name__
            )
            return None

    @staticmethod
    def cleanup(local_file: Optional[Path]) -> None:
        if local_file and local_file.exists():
            try:
                local_file.unlink()
            except OSError:  # pragma: no cover
                pass


__all__ = [
    "SnapshotService",
    "annotate_frame",
    "encode_jpeg",
    "object_path_for",
    "CLASS_PHONE",
]