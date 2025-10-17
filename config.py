from pathlib import Path
import torch

# --- Core Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PREPROCESSED_DATA_DIR = BASE_DIR / "preprocessed_data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
VISUALIZATION_DIR = BASE_DIR / "visualizations"
# RENAMED: Directory to store the extracted style images
STYLE_IMAGES_DIR = PREPROCESSED_DATA_DIR / "style_images"

# --- Data Splits ---
TRAIN_DIR = PREPROCESSED_DATA_DIR / "train"
VAL_DIR = PREPROCESSED_DATA_DIR / "val"
TEST_DIR = PREPROCESSED_DATA_DIR / "test"

# --- Preprocessing Settings ---
ROI_FILE_NAME = "roi.txt"
WGS84_CRS = "EPSG:4326"
PATCH_WIDTH_PIXELS = 512
PATCH_HEIGHT_PIXELS = 512
NUM_PATCHES_PER_ANNOTATION = 5
ANNOTATION_RADIUS = 60

# --- Training Hyperparameters ---
BATCH_SIZE = 4
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
# RENAMED: Probability of applying the FDA augmentation
FDA_AUGMENTATION_PROBABILITY = 0.1
INVARIANCE_LOSS_WEIGHT = 0.01

# --- System and Reproducibility ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_RANDOM_SEED = 42
DEEP_SUPERVISION=True

