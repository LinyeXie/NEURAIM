from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from .paths import PROJECT_ROOT


def evaluate_model(
    model_path: Path,
    data_config: Path,
    source: str | None = None,
    conf: float = 0.35,
    device: str = "0",
    imgsz: int = 640,
) -> None:
    model = YOLO(str(model_path.resolve()))
    if source:
        model.predict(
            source=source,
            imgsz=imgsz,
            conf=conf,
            device=device,
            save=True,
            project=str(PROJECT_ROOT / "runs" / "predict"),
            name="preview",
        )
        return

    model.val(
        data=str(data_config.resolve()),
        imgsz=imgsz,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "val"),
        name="metrics",
    )
