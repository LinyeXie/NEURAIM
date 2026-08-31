from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"
DEFAULT_BEST_MODEL = PROJECT_ROOT / "runs" / "enemy_detector" / "weights" / "best.pt"
