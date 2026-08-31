from __future__ import annotations

import argparse
import multiprocessing
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    import torch as torch_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from yolo_detector.dataset import IMAGE_SUFFIXES, check_dataset  # noqa: E402
from yolo_detector.paths import DEFAULT_TRAIN_CONFIG  # noqa: E402


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_train_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the Ultralytics training configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("Training YAML must contain a mapping.")
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply only command-line values explicitly supplied by the user."""
    for key in (
        "model",
        "data",
        "epochs",
        "imgsz",
        "batch",
        "device",
        "workers",
        "patience",
        "name",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    if args.cache is not None:
        config["cache"] = args.cache


def print_resolution_summary(data_yaml: Path) -> None:
    """Print source-image resolution counts without modifying the images."""
    with data_yaml.open("r", encoding="utf-8") as stream:
        data_config = yaml.safe_load(stream)
    if not isinstance(data_config, dict):
        raise ValueError(f"{data_yaml} must contain a YAML mapping.")

    dataset_root = Path(str(data_config.get("path", "."))).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    print("\nSource image resolutions (original files are kept unchanged):")
    for split in ("train", "val"):
        split_value = data_config.get(split, f"images/{split}")
        image_dir = Path(str(split_value)).expanduser()
        if not image_dir.is_absolute():
            image_dir = dataset_root / image_dir

        counts: Counter[tuple[int, int]] = Counter()
        unreadable = 0
        if image_dir.is_dir():
            image_paths = (
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            for image_path in image_paths:
                try:
                    with Image.open(image_path) as image:
                        counts[image.size] += 1
                except (OSError, UnidentifiedImageError):
                    unreadable += 1

        summary = ", ".join(
            f"{width}x{height}: {count}"
            for (width, height), count in sorted(counts.items())
        )
        print(f"  {split}: {summary or 'no readable images'}")
        if unreadable:
            print(f"  WARNING: {split} has {unreadable} unreadable image(s).")


def find_latest_checkpoint(project_dir: Path, run_name: str) -> Path:
    """Find the newest last.pt belonging to the configured run name."""
    candidates = [
        path
        for path in project_dir.glob(f"{run_name}*/weights/last.pt")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No resume checkpoint found under: "
            f"{project_dir / (run_name + '*') / 'weights' / 'last.pt'}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_training_dependencies() -> tuple[Any, Any]:
    """Import the heavy training packages only when training starts."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Run: "
            "pip install -r requirements.txt"
        ) from exc
    return torch, YOLO


def validate_device(device: Any, torch: "torch_module") -> None:
    """Fail early when a CUDA device is requested but CUDA is unavailable."""
    requested = str(device).strip().lower()
    if requested not in {"cpu", "mps"} and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build, "
            "or run with --device cpu."
        )

    if torch.cuda.is_available() and requested != "cpu":
        device_index = int(requested.split(",")[0]) if requested[0].isdigit() else 0
        print(f"GPU: {torch.cuda.get_device_name(device_index)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the four-class YOLO detector using "
            "data/images/{train,val} and data/labels/{train,val}."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help="Training YAML path (default: configs/train.yaml).",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        metavar="LAST_PT",
        help=(
            "Resume training. Without a path, automatically use the newest "
            "matching last.pt."
        ),
    )
    parser.add_argument("--model", help="Initial model, for example yolo26n.pt.")
    parser.add_argument("--data", type=Path, help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument(
        "--imgsz",
        type=int,
        help="Training input size. Mixed source resolutions are supported.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Batch size; use -1 for Ultralytics automatic batch sizing.",
    )
    parser.add_argument("--device", help="Training device, for example 0 or cpu.")
    parser.add_argument("--workers", type=int, help="DataLoader worker count.")
    parser.add_argument("--patience", type=int, help="Early-stopping patience.")
    parser.add_argument("--name", help="Training run name.")
    parser.add_argument(
        "--cache",
        nargs="?",
        const="ram",
        choices=("ram", "disk"),
        help="Cache images in RAM or on disk.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        config_path = resolve_project_path(args.config)
        config = load_train_config(config_path)
        apply_cli_overrides(config, args)

        model_value = str(config.pop("model", "yolo26n.pt"))
        data_yaml = resolve_project_path(
            config.get("data", "configs/data.yaml")
        )
        project_dir = resolve_project_path(config.get("project", "runs"))
        config["data"] = str(data_yaml)
        config["project"] = str(project_dir)

        if not check_dataset(data_yaml):
            raise RuntimeError("Dataset validation failed; training was not started.")
        print_resolution_summary(data_yaml)

        torch, yolo_class = load_training_dependencies()
        validate_device(config.get("device", "0"), torch)
        print(
            "\nTraining settings: "
            f"imgsz={config.get('imgsz', 640)}, "
            f"epochs={config.get('epochs', 100)}, "
            f"batch={config.get('batch', 16)}, "
            f"device={config.get('device', '0')}"
        )

        if args.resume is not None:
            if args.resume == "auto":
                checkpoint = find_latest_checkpoint(
                    project_dir, str(config.get("name", "train"))
                )
            else:
                checkpoint = resolve_project_path(args.resume)
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Resume checkpoint not found: {checkpoint}"
                    )
            print(f"Resuming from: {checkpoint}")
            results = yolo_class(str(checkpoint)).train(resume=True)
        else:
            local_model = resolve_project_path(model_value)
            model_source = str(local_model) if local_model.is_file() else model_value
            results = yolo_class(model_source).train(**config)

        best_weight = Path(results.save_dir) / "weights" / "best.pt"
        print(f"\nTraining complete. Best weights: {best_weight.resolve()}")
        return 0
    except (KeyboardInterrupt, SystemExit):
        print("\nTraining stopped by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())