from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo_detector.evaluation import evaluate_model  # noqa: E402
from yolo_detector.paths import DEFAULT_BEST_MODEL, DEFAULT_DATA_CONFIG  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or preview the YOLO model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_BEST_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional image, directory, or video. Omit for validation.",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    try:
        evaluate_model(
            model_path=args.model,
            data_config=args.data,
            source=args.source,
            conf=args.conf,
            device=args.device,
            imgsz=args.imgsz,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
