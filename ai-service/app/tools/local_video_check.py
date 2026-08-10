"""Run a LOCAL video file through the real detection pipeline.

Purpose: validate and calibrate the software end to end before any physical
camera, NVR or GPU is available. Nothing here is a simulation - it uses the
same YoloDetector, association, temporal confirmation and annotation code the
service uses in production. No fake detections are ever generated.

Safety defaults
---------------
* No Supabase writes and no Telegram sends. Cloud publishing is opt-in via
  ``--publish``, which reuses the production Orchestrator path instead of
  re-implementing it.
* No physical camera or NVR is contacted.

Usage (from the ai-service folder)::

    python -m app.tools.local_video_check ./samples/demo.mp4
    python -m app.tools.local_video_check ./samples/demo.mp4 --save-annotated ./out
    python -m app.tools.local_video_check ./samples/demo.mp4 --publish
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..ai.association import associate
from ..ai.detector import YoloDetector
from ..ai.phone_rule_engine import PhoneRuleEngine
from ..config import get_settings
from ..domain.models import CameraConfig, RuleConfig, SourceType
from ..events.snapshot_service import annotate_frame, encode_jpeg

ENGINE_MOBILE_PHONE = "mobile_phone_detection"
CAMERA_ID = "local-video-check"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local video pipeline check")
    parser.add_argument("video", nargs="?", help="Path to an MP4 file (default: DEMO_VIDEO_PATH)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames")
    parser.add_argument("--every", type=int, default=1, help="Process every Nth frame")
    parser.add_argument(
        "--save-annotated", default="", help="Directory for annotated JPEG frames (optional)"
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="OPT-IN: also write events/snapshots to Supabase using the normal service path",
    )
    # Rule thresholds, so the tool doubles as a calibration aid.
    parser.add_argument("--phone-conf", type=float, default=0.35)
    parser.add_argument("--person-conf", type=float, default=0.40)
    parser.add_argument("--assoc-conf", type=float, default=0.55)
    parser.add_argument("--min-duration", type=float, default=1.5)
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--instant-conf", type=float, default=0.85)
    parser.add_argument(
        "--no-instant",
        action="store_true",
        help="Disable instant single-frame visible-phone evidence (temporal path only)",
    )
    return parser.parse_args(argv)


def _rule(args: argparse.Namespace) -> RuleConfig:
    return RuleConfig(
        id="local-video-check-rule",
        name="Local video check",
        engine_key=ENGINE_MOBILE_PHONE,
        available=True,
        enabled=True,
        severity="critical",
        confidence_threshold=args.phone_conf,
        person_confidence_threshold=args.person_conf,
        association_confidence_threshold=args.assoc_conf,
        min_duration_seconds=args.min_duration,
        min_matching_frames=args.min_frames,
        instant_detection_enabled=not args.no_instant,
        instant_confidence_threshold=args.instant_conf,
        save_snapshot=bool(args.save_annotated),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    settings = get_settings()

    if args.publish:
        print("--publish requested: run the full service instead (python run.py).")
        print("This tool stays offline on purpose; it never writes to the cloud itself.")
        return 2

    source = args.video or settings.demo_video_path
    if not source:
        print("ERROR: no video given and DEMO_VIDEO_PATH is empty.")
        return 1
    video = Path(source)
    if not video.exists():
        print(f"ERROR: video not found: {video}")
        return 1

    import cv2  # imported here so --help works without OpenCV

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print(f"ERROR: OpenCV could not open {video}")
        return 1

    out_dir = Path(args.save_annotated) if args.save_annotated else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    detector = YoloDetector(
        settings.yolo_model, settings.yolo_device, settings.yolo_imgsz, settings.yolo_tracker
    )
    engine = PhoneRuleEngine(
        association_margin=settings.association_margin,
        gap_tolerance_seconds=settings.detection_gap_tolerance_seconds,
    )
    camera = CameraConfig(
        id=CAMERA_ID, name="Local Video", source_type=SourceType.DEMO, is_demo=True
    )
    rule = _rule(args)

    print(f"Model: {settings.yolo_model} on {detector.device}")
    print(f"Video: {video}")
    print("Cloud publishing: OFF | Telegram: OFF")

    read = 0
    processed = 0
    confirmed = 0
    started = time.monotonic()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        read += 1
        if args.every > 1 and (read % args.every):
            continue

        detections = detector.detect(frame, camera.id)
        persons = tuple(
            person
            for person in detections.persons
            if person.confidence >= rule.person_confidence_threshold and person.tracking_id
        )
        associations = {}
        for index, phone in enumerate(detections.phones):
            if phone.confidence < rule.confidence_threshold:
                continue
            associations[phone.tracking_id or f"idx{index}"] = associate(
                phone,
                persons,
                association_threshold=rule.association_confidence_threshold,
                margin=settings.association_margin,
            )

        drafts = engine.process_frame(
            camera=camera,
            rule=rule,
            detections=detections,
            now=time.monotonic(),
            source_mode="demo",
            detected_at=datetime.now(timezone.utc),
        )

        processed += 1
        if out_dir or drafts:
            annotated = annotate_frame(
                frame,
                detections,
                camera_name=camera.name,
                associations=associations,
                timestamp=datetime.now(),
            )
            if out_dir:
                jpeg = encode_jpeg(annotated, quality=85)
                if jpeg:
                    (out_dir / f"frame_{processed:06d}.jpg").write_bytes(jpeg)

        for draft in drafts:
            confirmed += 1
            event = draft.event
            origin = "instant" if draft.origin == "instant" else "temporal"
            print(
                f"[{origin}] {event.type} | severity={event.severity} | "
                f"association={event.association_status} | person_id={event.person_tracking_id} | "
                f"duration={event.detection_duration_seconds:.1f}s"
            )

        if args.max_frames and processed >= args.max_frames:
            break

    capture.release()
    elapsed = max(1e-6, time.monotonic() - started)
    print(
        f"\nFrames read: {read} | analysed: {processed} | "
        f"pipeline FPS: {processed / elapsed:.2f} | events: {confirmed} "
        f"(instant + temporal)"
    )
    if out_dir:
        print(f"Annotated frames written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
