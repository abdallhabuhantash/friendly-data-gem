"""Tests for the MANUAL paper-detector training utility (Task 3E).

Importing the module must never train, download weights or touch the network.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "train_paper_detector.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_paper_detector", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_training_module_does_not_start_training() -> None:
    module = load_module()
    assert hasattr(module, "train")
    assert hasattr(module, "validate")
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    module_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert module_level_calls == []


def test_no_model_download_or_ultralytics_import_at_module_level() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            module_name = getattr(node, "module", None) or ""
            assert "ultralytics" not in module_name
            assert not any("ultralytics" in name for name in names)


def test_required_arguments_are_explicit() -> None:
    module = load_module()
    parser = module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])  # every hyperparameter is required
    args = parser.parse_args(
        [
            "train",
            "--data", "d.yaml",
            "--base-weights", "base.pt",
            "--device", "cpu",
            "--imgsz", "960",
            "--epochs", "100",
            "--batch", "16",
            "--project", "out",
            "--name", "paper_v1",
        ]
    )
    assert (args.imgsz, args.epochs, args.batch) == (960, 100, 16)


def test_no_hidden_hyperparameter_defaults() -> None:
    module = load_module()
    parser = module.build_parser()
    subparsers = [
        action for action in parser._actions if isinstance(action, ast.AST) is False
    ]
    assert subparsers  # parser is constructed
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for flag in ("--epochs", "--imgsz", "--batch", "--device", "--data", "--base-weights"):
        assert f'"{flag}", required=True' in source.replace("\n", " ") or (
            f'add_argument("{flag}", required=True' in source
        )
    assert "default=" not in source


def test_dataset_path_required_and_validated() -> None:
    module = load_module()
    with pytest.raises(module.DatasetContractError):
        module.load_dataset_config("")
    with pytest.raises(module.DatasetContractError):
        module.load_dataset_config("definitely/missing/paper.yaml")


def test_validation_split_and_paper_class_required(tmp_path: pathlib.Path) -> None:
    module = load_module()

    no_val = tmp_path / "no_val.yaml"
    no_val.write_text("train: images/train\nval:\nnames: [paper]\n", encoding="utf-8")
    with pytest.raises(module.DatasetContractError):
        module.load_dataset_config(str(no_val))

    missing_key = tmp_path / "missing.yaml"
    missing_key.write_text("train: images/train\nnames: [paper]\n", encoding="utf-8")
    with pytest.raises(module.DatasetContractError):
        module.load_dataset_config(str(missing_key))

    wrong_class = tmp_path / "wrong.yaml"
    wrong_class.write_text(
        "train: images/train\nval: images/val\nnames: [book]\n", encoding="utf-8"
    )
    with pytest.raises(module.DatasetContractError):
        module.load_dataset_config(str(wrong_class))

    good = tmp_path / "good.yaml"
    good.write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: paper\n",
        encoding="utf-8",
    )
    config = module.load_dataset_config(str(good))
    assert config["names"] == {0: "paper"}


def test_no_secrets_or_absolute_paths_committed_in_script() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for forbidden in ("rtsp://", "password", "SUPABASE", "api_key", "token="):
        assert forbidden.lower() not in source.lower()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("#", '"', "'")) or "python scripts" in stripped:
            continue
        assert "/home/" not in stripped and "C:\\" not in stripped


def test_training_output_directories_are_gitignored() -> None:
    gitignore = (SCRIPTS_DIR.parent / ".gitignore").read_text(encoding="utf-8")
    assert "training-runs/" in gitignore
    assert "datasets/" in gitignore


def test_no_acceptance_thresholds_hardcoded() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    for forbidden in ("mAP50 >", "0.95", "100% accuracy", "accuracy_target"):
        assert forbidden not in source


def test_training_documentation_exists_and_covers_required_topics() -> None:
    doc = (SCRIPTS_DIR.parent.parent / "docs" / "paper-detector-training.md").read_text(
        encoding="utf-8"
    )
    lowered = doc.lower()
    for topic in (
        "coco",
        "book is not paper",
        "hard negative",
        "leakage",
        "annotation",
        "held-out test",
        "map50-95",
        "small",
        "not production-ready",
    ):
        assert topic in lowered
