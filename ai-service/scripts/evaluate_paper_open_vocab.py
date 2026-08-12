#!/usr/bin/env python3
"""MANUAL offline evaluator for open-vocabulary paper evidence (Task 3E-B).

This script trains nothing, changes no production behaviour and is
never imported by the live runtime. Importing it loads no model, opens no video
and downloads nothing: the detector, OpenCV and Ultralytics are imported lazily
inside the execution path.

Everything is explicit: source, weights, prompts, device, image size, confidence
threshold and frame sampling. No production threshold is chosen here — the CLI
requires the value you want to experiment with.

Examples
--------
Full-frame evaluation of one prompt configuration::

    python scripts/evaluate_paper_open_vocab.py \
        --source samples/exam_clip.mp4 \
        --weights models/yolov8s-worldv2.pt \
        --prompts "paper" "sheet of paper" "exam paper" \
        --device cpu --imgsz 960 --confidence 0.15 --frame-stride 5 \
        --json-out paper-eval-results/run1.json

Compare several prompt configurations in one pass (``--prompts`` repeated)::

    python scripts/evaluate_paper_open_vocab.py --source samples/exam_clip.mp4 \
        --weights models/yolov8s-worldv2.pt \
        --prompts "paper" --prompts "small paper slip" "folded paper" \
        --device cpu --imgsz 960 --confidence 0.15 --frame-stride 5

Optional, explicitly-labelled crop experiment (offline only)::

    ... --crop 0.25 0.30 0.75 0.95
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence


def _ensure_import_path() -> None:
    """Allows running the script directly from the ai-service directory."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_paper_open_vocab",
        description=(
            "Offline descriptive evaluation of open-vocabulary loose-paper detection. "
            "No training, no runtime integration, no precision/recall claims. "
            "'book' is never a paper prompt."
        ),
    )
    parser.add_argument("--source", required=True, help="local video or image file")
    parser.add_argument("--weights", required=True, help="explicit open-vocabulary checkpoint")
    parser.add_argument(
        "--prompts",
        required=True,
        nargs="+",
        action="append",
        metavar="PROMPT",
        help="explicit prompt list; repeat the flag to compare configurations",
    )
    parser.add_argument("--device", required=True, help="explicit device, e.g. cpu or 0")
    parser.add_argument("--imgsz", required=True, type=int, help="explicit inference image size")
    parser.add_argument(
        "--confidence",
        required=True,
        type=float,
        help="experiment confidence threshold (NOT a production decision)",
    )
    parser.add_argument(
        "--frame-stride",
        required=True,
        type=int,
        help="explicit sampling cadence: analyse every Nth frame",
    )
    parser.add_argument(
        "--max-frames", type=int, help="optional cap on analysed frames (0 = all)"
    )
    parser.add_argument("--json-out", help="write the JSON report here (basenames only)")
    parser.add_argument(
        "--annotated-out",
        help="optional annotated .mp4 for manual visual review of every detection",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="optional OFFLINE crop experiment in normalized full-frame coordinates",
    )
    parser.add_argument(
        "--backend-label",
        help="diagnostic backend name recorded in the report (never a path)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _ensure_import_path()

    from app.ai.crop_transform import CropTransform
    from app.ai.open_vocab_paper_detector import (
        DEFAULT_BACKEND_LABEL,
        OpenVocabPaperDetector,
        PaperPromptConfig,
    )
    from app.benchmark.paper_evaluation import (
        EvaluationReport,
        evaluate_source,
        render_console_summary,
        safe_source_label,
    )

    if not os.path.isfile(args.source):
        print(f"source not found: {safe_source_label(args.source)}", file=sys.stderr)
        return 2
    if args.frame_stride <= 0 or args.imgsz <= 0:
        print("--frame-stride and --imgsz must be positive", file=sys.stderr)
        return 2

    crop = CropTransform(*args.crop) if args.crop else None
    backend_label = (args.backend_label or DEFAULT_BACKEND_LABEL).strip()
    runs = []
    for prompt_group in args.prompts:
        prompt_config = PaperPromptConfig.from_iterable(prompt_group)
        detector = OpenVocabPaperDetector(
            weights_path=args.weights,
            prompts=prompt_config,
            device=args.device,
            imgsz=args.imgsz,
            confidence=args.confidence,
            backend_label=backend_label,
        )
        runs.append(
            evaluate_source(
                detector,
                args.source,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames or 0,
                crop=crop,
                mode="explicit_crop" if crop is not None else "full_frame",
                annotated_output=args.annotated_out,
            )
        )

    report = EvaluationReport(
        source_name=args.source,
        backend=backend_label,
        model_name=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        confidence_threshold=args.confidence,
        frame_stride=args.frame_stride,
        runs=tuple(runs),
    )
    print(render_console_summary(report))

    if args.json_out:
        directory = os.path.dirname(os.path.abspath(args.json_out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        print(f"\nJSON report written: {os.path.basename(args.json_out)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point only
    sys.exit(main())
