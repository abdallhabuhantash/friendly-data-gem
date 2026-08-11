"""CLI entry point for the runtime benchmark.

Nothing is measured at import time: models load only when ``main`` runs.

Examples (from the ai-service folder)::

    python -m scripts.benchmark_runtime --video .\\samples\\demo.mp4 --mode baseline
    python -m scripts.benchmark_runtime --video .\\samples\\demo.mp4 --mode both ^
        --pose-model yolo11n-pose.pt --pose-device cuda:0 --pose-imgsz 640 ^
        --pose-confidence 0.30 --pose-max-fps 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.benchmark.runtime_benchmark import (  # noqa: E402
    MODE_TASK1_ONLY,
    BenchmarkConfig,
    build_report,
    cuda_peak_memory,
    format_summary,
    run_mode,
    safe_hardware_info,
    write_report,
)
from app.config import get_settings  # noqa: E402

DEFAULT_OUTPUT = "benchmark-results/pose-runtime-benchmark.json"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vigilant Eye runtime benchmark")
    parser.add_argument("--video", default="", help="Local MP4 path (default: DEMO_VIDEO_PATH)")
    parser.add_argument(
        "--mode",
        choices=("baseline", "pose", "both"),
        default="both",
        help="baseline = Task 1 only, pose = Task 1 + Pose, both = compare",
    )
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--max-measured-frames", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    # Pose settings must be explicit; no calibrated defaults exist.
    parser.add_argument("--pose-model", default="")
    parser.add_argument("--pose-device", default="")
    parser.add_argument("--pose-imgsz", type=int, default=0)
    parser.add_argument("--pose-confidence", type=float, default=0.0)
    parser.add_argument("--pose-max-fps", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    settings = get_settings()

    video = args.video or settings.demo_video_path
    if not video:
        print("ERROR: no --video given and DEMO_VIDEO_PATH is empty.")
        return 1
    if not Path(video).exists():
        print(f"ERROR: benchmark video not found: {Path(video).name}")
        return 1

    pose_ready = bool(
        args.pose_model
        and args.pose_device
        and args.pose_imgsz > 0
        and args.pose_confidence > 0
        and args.pose_max_fps > 0
    )
    if args.mode in ("pose", "both") and not pose_ready:
        if args.mode == "pose":
            print(
                "ERROR: pose mode needs --pose-model --pose-device --pose-imgsz "
                "--pose-confidence --pose-max-fps (no defaults are assumed)."
            )
            return 1
        print("NOTE: pose settings incomplete - running the baseline mode only.")

    config = BenchmarkConfig(
        source_video=video,
        warmup_frames=max(0, args.warmup_frames),
        max_measured_frames=max(0, args.max_measured_frames),
        detector_model=settings.yolo_model,
        detector_device=settings.yolo_device,
        detector_imgsz=settings.yolo_imgsz,
        detector_tracker=settings.yolo_tracker,
        association_margin=settings.association_margin,
        gap_tolerance_seconds=settings.detection_gap_tolerance_seconds,
        pose_model=args.pose_model or None,
        pose_device=args.pose_device or None,
        pose_imgsz=args.pose_imgsz or None,
        pose_confidence=args.pose_confidence or None,
        pose_max_fps=args.pose_max_fps or None,
    )

    hardware = safe_hardware_info()
    baseline, baseline_memory = run_mode(config, mode=MODE_TASK1_ONLY)
    with_pose = None
    pose_memory = None
    if args.mode in ("pose", "both") and pose_ready:
        from app.benchmark.runtime_benchmark import MODE_TASK1_PLUS_POSE

        with_pose, pose_memory = run_mode(config, mode=MODE_TASK1_PLUS_POSE)

    report = build_report(
        config=config,
        baseline=baseline,
        with_pose=with_pose,
        hardware=hardware,
        baseline_memory=baseline_memory,
        pose_memory=pose_memory or cuda_peak_memory(),
    )
    print(format_summary(report))
    path = write_report(report, args.output)
    print(f"\nJSON report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
