from pathlib import Path

# --- Core Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PREPROCESSED_DATA_DIR = BASE_DIR / "preprocessed_data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
VISUALIZATION_DIR = BASE_DIR / "visualizations"

# --- Data Splits ---
TRAIN_DIR = PREPROCESSED_DATA_DIR / "train"
VAL_DIR = PREPROCESSED_DATA_DIR / "val"
TEST_DIR = PREPROCESSED_DATA_DIR / "test"

# --- Training Hyperparameters ---
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

# --- System and Reproducibility ---
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_RANDOM_SEED = 42
