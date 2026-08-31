# NEURAIM

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

NEURAIM is an Ultralytics YOLO training project organized according to common AI engineering practices. The project separates datasets, configuration files, executable scripts, core source code, tests, model artifacts, and training outputs for clearer development and maintenance.

## Project Structure

```text
yolo_enemy_detector/
├── configs/
│   ├── data.yaml               # Dataset paths and four-class definitions
│   └── train.yaml              # Training hyperparameters
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── models/                     # Exported ONNX, TensorRT, and other model files
├── notebooks/                  # Notebooks for experiments and analysis
├── runs/                       # Training, validation, and prediction outputs
├── scripts/
│   ├── check_dataset.py        # Dataset validation entry point
│   ├── train.py                # Model training entry point
│   └── evaluate.py             # Validation and inference entry point
├── src/
│   └── yolo_detector/
│       ├── dataset.py          # Core dataset validation logic
│       ├── training.py         # Core training logic
│       └── evaluation.py       # Validation and inference logic
├── tests/
│   └── test_dataset.py
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## 1. Prepare the Dataset

Copy your dataset into the following directories:

```text
data/images/train/img_0.jpg
data/labels/train/img_0.txt
data/images/val/img_1.jpg
data/labels/val/img_1.txt
```

Each image and its corresponding label file must use the same filename. Each annotation line follows the standard YOLO format:

```text
class_id x_center y_center width height
```

All coordinates must be normalized to the range `0–1`.

The project uses four classes:

| ID | Class | Description |
|---:|---|---|
| 0 | `friendly_body` | Friendly body |
| 1 | `enemy_body` | Enemy body |
| 2 | `friendly_head` | Friendly head |
| 3 | `enemy_head` | Enemy head |

## 2. Install the Environment

Python 3.11 is recommended. Run the following commands from the project root directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `torch.cuda.is_available()` returns `False`, install a PyTorch version that is compatible with your CUDA environment.

## 3. Validate the Dataset

```powershell
python scripts/check_dataset.py
```

The dataset checker validates directory structure, image readability, image-label matching, class IDs, normalized coordinates, and out-of-bounds bounding boxes.

Fix all reported `ERROR` entries before starting training.

## 4. Train the Model

```powershell
python scripts/train.py
```

Training parameters are defined in `configs/train.yaml`.

The best model is saved by default to:

```text
runs/enemy_detector/weights/best.pt
```

Resume an interrupted training session:

```powershell
python scripts/train.py --resume
```

Use a specific configuration file:

```powershell
python scripts/train.py --config configs/train.yaml
```

## 5. Validation and Prediction

Validate the best model:

```powershell
python scripts/evaluate.py
```

Run predictions on the validation image directory and save the annotated results:

```powershell
python scripts/evaluate.py --source data/images/val --conf 0.35
```

You can also specify a different model, image, directory, or video source:

```powershell
python scripts/evaluate.py --model runs/enemy_detector/weights/best.pt --source demo.mp4
```

## 6. Run Tests

```powershell
tests/test_cs_v2.7.py
```

The `runs/`, `models/`, and actual dataset contents are excluded through `.gitignore` to prevent large generated files or local data from being committed accidentally.

`.gitkeep` files are used only to preserve the standard project directory structure in Git.

## 7. License

This project is licensed under the **Apache License 2.0**.

You may use, copy, modify, distribute, and commercially use this project in accordance with the terms of the license. When redistributing the project, you must retain the applicable copyright notices, license notices, and any `NOTICE` file if one is provided. Modified source files should also include appropriate notices indicating that changes were made.

See the `LICENSE` file in the project root directory for the full license text, or visit the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

> This software is provided on an **"AS IS"** basis, without warranties or conditions of any kind, either express or implied. Rights, limitations, and liabilities are governed by the official Apache License 2.0 text.
