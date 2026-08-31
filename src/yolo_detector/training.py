from __future__ import annotations

from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from .dataset import check_dataset
from .paths import PROJECT_ROOT


def resolve_project_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def train_model(config_path: Path, resume: bool = False) -> Path:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Training YAML must contain a mapping.")

    model_path = str(config.pop("model", "yolo26n.pt"))
    config["data"] = resolve_project_path(
        str(config.get("data", "configs/data.yaml"))
    )
    config["project"] = resolve_project_path(str(config.get("project", "runs")))

    if not check_dataset(Path(config["data"])):
        raise RuntimeError("Dataset validation failed; training was not started.")

    requested_device = str(config.get("device", "0"))
    if requested_device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build, "
            "or set device: cpu in configs/train.yaml."
        )
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if resume:
        last_weight = (
            Path(config["project"]) / str(config["name"]) / "weights" / "last.pt"
        )
        if not last_weight.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {last_weight}")
        results = YOLO(str(last_weight)).train(resume=True)
    else:
        results = YOLO(model_path).train(**config)

    best_weight = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best_weight}")
    return best_weight
