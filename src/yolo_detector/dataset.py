from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset_yaml(yaml_path: Path) -> tuple[Path, dict[int, str]]:
    with yaml_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"{yaml_path} is not a YAML mapping.")

    raw_names = config.get("names")
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        raise ValueError("data.yaml must contain a names list or mapping.")

    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root, names


def inspect_split(
    dataset_root: Path,
    split: str,
    names: dict[int, str],
) -> tuple[list[str], list[str], Counter[int], int]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    errors: list[str] = []
    warnings: list[str] = []
    class_counts: Counter[int] = Counter()

    if not image_dir.is_dir():
        return [f"Missing directory: {image_dir}"], warnings, class_counts, 0
    if not label_dir.is_dir():
        return [f"Missing directory: {label_dir}"], warnings, class_counts, 0

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    labels = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)
    if missing_labels:
        warnings.append(
            f"{split}: {len(missing_labels)} image(s) have no label file "
            f"(allowed only for true background images); examples: "
            f"{', '.join(missing_labels[:5])}"
        )
    if orphan_labels:
        errors.append(
            f"{split}: {len(orphan_labels)} label file(s) have no matching image; "
            f"examples: {', '.join(orphan_labels[:5])}"
        )
    if not images:
        errors.append(f"{split}: no supported images found in {image_dir}")

    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            errors.append(f"{image_path}: unreadable image ({exc})")

    for label_path in labels:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            location = f"{label_path}:{line_number}"
            if len(fields) != 5:
                errors.append(
                    f"{location}: expected 5 values "
                    "(class x_center y_center width height)"
                )
                continue
            try:
                class_value = float(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError:
                errors.append(f"{location}: contains a non-numeric value")
                continue

            class_id = int(class_value)
            if class_value != class_id:
                errors.append(f"{location}: class ID must be an integer")
                continue
            if class_id not in names:
                errors.append(
                    f"{location}: class ID {class_id} is not defined in data.yaml"
                )
                continue

            x_center, y_center, width, height = coordinates
            if not all(0.0 <= value <= 1.0 for value in coordinates):
                errors.append(f"{location}: all box coordinates must be in [0, 1]")
                continue
            if width <= 0.0 or height <= 0.0:
                errors.append(f"{location}: width and height must be greater than 0")
                continue
            if (
                x_center - width / 2 < -1e-6
                or y_center - height / 2 < -1e-6
                or x_center + width / 2 > 1.0 + 1e-6
                or y_center + height / 2 > 1.0 + 1e-6
            ):
                errors.append(f"{location}: bounding box extends beyond the image")
                continue

            class_counts[class_id] += 1

    return errors, warnings, class_counts, len(images)


def check_dataset(yaml_path: Path) -> bool:
    dataset_root, names = load_dataset_yaml(yaml_path)
    print(f"Dataset: {dataset_root}")
    print("Classes: " + ", ".join(f"{key}={value}" for key, value in names.items()))

    all_errors: list[str] = []
    all_warnings: list[str] = []
    for split in ("train", "val"):
        errors, warnings, counts, image_count = inspect_split(
            dataset_root, split, names
        )
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        count_text = ", ".join(
            f"{class_id}:{names[class_id]}={counts[class_id]}"
            for class_id in sorted(names)
        )
        print(f"{split}: {image_count} images | {count_text}")

    for warning in all_warnings:
        print(f"WARNING: {warning}")
    for error in all_errors:
        print(f"ERROR: {error}", file=sys.stderr)

    if all_errors:
        print(
            f"Dataset check failed with {len(all_errors)} error(s).",
            file=sys.stderr,
        )
        return False
    print("Dataset check passed.")
    return True
