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


# --- HYPERPARAMETER SEARCH CONFIGURATION ---

# Number of random combinations to try
NUM_SEARCH_TRIALS = 20

# Define the search space.
# For each hyperparameter, provide a list of values to choose from.
HPARAM_SEARCH_SPACE = {
    "learning_rate": [5e-4, 1e-4, 5e-5],
    #"learning_rate": [1e-3, 5e-4, 1e-4, 5e-5],
    "batch_size": [4, 8],
    "deep_supervision": [True, False],
    "invariance_loss_weight": [0.01, 0.1, 0.2, 0.4],
    # You can add other parameters here, like optimizer type, etc.
    # "optimizer": ["adam", "sgd"]
}

# --- Training Hyperparameters ---

NUM_EPOCHS = 200
# RENAMED: Probability of applying the FDA augmentation
STYLE_IMAGE_AUGMENTATION_PROBABILITY = 0.1

# --- System and Reproducibility ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_RANDOM_SEED = 42
DEEP_SUPERVISION=True
LAND_SHP_PATH="shapefiles/Europe_coastline.shp"
COASTLINE_BUFFER_METERS=6
NUM_NEGATIVE_LAND_ANNOTATIONS_PER_TIF=15

