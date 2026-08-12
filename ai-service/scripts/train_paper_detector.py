#!/usr/bin/env python3
"""MANUAL training utility for a future custom paper detector.

This script NEVER trains on import, on test collection or at application
startup. Training happens only when a human runs it explicitly from a shell and
supplies every required argument.

There are deliberately NO production-looking defaults for dataset, weights,
device, image size, epochs, batch or output directory: a calibrated-looking
default that was never calibrated is a lie. Ultralytics is imported lazily
inside :func:`train`, so importing this module downloads nothing.

Read ``docs/paper-detector-training.md`` before using this script: dataset,
annotation, split and evaluation rules are mandatory, not advisory.

Example (paths are illustrative only)::

    python scripts/train_paper_detector.py train \
        --data datasets/paper/paper.yaml \
        --base-weights yolov8s.pt \
        --device 0 --imgsz 960 --epochs 100 --batch 16 \
        --project training-runs/paper --name paper_v1

Validation of a trained checkpoint::

    python scripts/train_paper_detector.py validate \
        --data datasets/paper/paper.yaml \
        --weights training-runs/paper/paper_v1/weights/best.pt \
        --device 0 --imgsz 960 --split test
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional, Sequence

#: Canonical class the dataset YAML must declare (single-class MVP).
REQUIRED_CLASS_NAME = "paper"

#: Keys a dataset YAML must define before training is allowed.
REQUIRED_DATASET_KEYS = ("train", "val", "names")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with NO implicit training-hyperparameter defaults."""
    parser = argparse.ArgumentParser(
        prog="train_paper_detector",
        description=(
            "Manually train or validate a custom single-class 'paper' detector. "
            "No stock/COCO fallback exists; 'book' is never paper."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train", help="train a custom paper detector")
    train.add_argument("--data", required=True, help="path to the dataset YAML")
    train.add_argument("--base-weights", required=True, help="explicit base checkpoint")
    train.add_argument("--device", required=True, help="explicit device, e.g. cpu or 0")
    train.add_argument("--imgsz", required=True, type=int, help="explicit image size")
    train.add_argument("--epochs", required=True, type=int, help="explicit epoch count")
    train.add_argument("--batch", required=True, type=int, help="explicit batch size")
    train.add_argument("--project", required=True, help="explicit output project directory")
    train.add_argument("--name", required=True, help="explicit run name")

    validate = subparsers.add_parser("validate", help="evaluate a trained checkpoint")
    validate.add_argument("--data", required=True, help="path to the dataset YAML")
    validate.add_argument("--weights", required=True, help="trained checkpoint to evaluate")
    validate.add_argument("--device", required=True, help="explicit device, e.g. cpu or 0")
    validate.add_argument("--imgsz", required=True, type=int, help="explicit image size")
    validate.add_argument(
        "--split",
        required=True,
        choices=("val", "test"),
        help="evaluation split; the held-out test split must not be used for tuning",
    )

    return parser


class DatasetContractError(ValueError):
    """Raised when the dataset YAML does not satisfy the paper contract."""


def load_dataset_config(data_path: str) -> dict[str, Any]:
    """Loads and validates the dataset YAML (existence, splits, class names)."""
    if not isinstance(data_path, str) or not data_path.strip():
        raise DatasetContractError("--data must be a non-empty path")
    if not os.path.isfile(data_path):
        raise DatasetContractError(f"dataset YAML not found: {os.path.basename(data_path)}")

    import yaml  # lazy: keeps import of this module dependency-light

    with open(data_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise DatasetContractError("dataset YAML must define a mapping")

    missing = [key for key in REQUIRED_DATASET_KEYS if key not in config]
    if missing:
        raise DatasetContractError(
            "dataset YAML missing required keys: " + ", ".join(missing)
        )
    if not config.get("val"):
        raise DatasetContractError(
            "a real validation split is mandatory; never validate on training data"
        )

    names = config["names"]
    if isinstance(names, dict):
        declared = [str(value) for value in names.values()]
    elif isinstance(names, (list, tuple)):
        declared = [str(value) for value in names]
    else:
        raise DatasetContractError("dataset 'names' must be a list or mapping")
    if REQUIRED_CLASS_NAME not in [name.strip().lower() for name in declared]:
        raise DatasetContractError(
            f"dataset must declare the canonical {REQUIRED_CLASS_NAME!r} class"
        )
    return config


def train(args: argparse.Namespace) -> int:
    """Runs a real training job. Called ONLY from :func:`main`."""
    load_dataset_config(args.data)
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0:
        raise DatasetContractError("--epochs, --imgsz and --batch must be positive")

    from ultralytics import YOLO  # lazy: no download at import time

    model = YOLO(args.base_weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    print(
        "Training finished. Report precision, recall, mAP50, mAP50-95, hard-negative "
        "false positives, small-paper recall and occluded-paper recall from the "
        "held-out test split. Do not treat the checkpoint as production-ready before "
        "that review."
    )
    return 0


def validate(args: argparse.Namespace) -> int:
    """Evaluates a trained checkpoint on an explicit split."""
    load_dataset_config(args.data)
    if not os.path.isfile(args.weights):
        raise DatasetContractError(
            f"checkpoint not found: {os.path.basename(args.weights)}"
        )

    from ultralytics import YOLO  # lazy: no download at import time

    model = YOLO(args.weights)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "train":
        return train(args)
    if args.command == "validate":
        return validate(args)
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover - manual entry point only
    sys.exit(main())
