from pathlib import Path
import torch

# VERY IMPORTANT
# Set the project path to be the root of the directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Project paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOISY_DIR = RAW_DATA_DIR / "noisy"
CLEAN_DIR = RAW_DATA_DIR / "clean"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
SAMPLES_DIR = OUTPUT_DIR / "samples"


# Training settings
SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 4 # was 16
NUM_WORKERS = 2

NUM_EPOCHS =  40 # 20 # was 2 # was 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0

# This split is a good balanced split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Model settings
MODEL_NAME = "dncnn"
IN_CHANNELS = 3
OUT_CHANNELS = 3
NUM_FEATURES = 64
NUM_LAYERS = 12 # was 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output filenames
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_model.pt"

BASELINE_METRICS_PATH = METRICS_DIR / "baseline_metrics.csv"
MODEL_METRICS_PATH = METRICS_DIR / "model_metrics.csv"



# Utility
def create_directories() -> None:
    """
    Create all required project output directories if they do not exist.
    """
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        NOISY_DIR,
        CLEAN_DIR,
        OUTPUT_DIR,
        CHECKPOINT_DIR,
        METRICS_DIR,
        SAMPLES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)