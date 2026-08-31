from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_detector.dataset import check_dataset  # noqa: E402
from yolo_detector.paths import DEFAULT_DATA_CONFIG  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="Path to the dataset YAML.",
    )
    args = parser.parse_args()
    try:
        return 0 if check_dataset(args.data.resolve()) else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
