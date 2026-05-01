"""Configuration file for the HCC Segmentation Thesis.
Adjust paths and hyperparameters as needed.
"""
import os
from pathlib import Path
from typing import Final

print("[Config] Importing torch... (This may take a moment)")
import torch
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 0. Load Environment Variables
# -----------------------------------------------------------------------------

load_dotenv()

# -----------------------------------------------------------------------------
# 2. Environment detection and validation
# -----------------------------------------------------------------------------
ENV = os.getenv("PIN_ENV")

if ENV is None:
    raise EnvironmentError(
        "[Config] ERROR: Environment variable 'PIN_ENV' is not set!\n"
        "Please do one of the following:\n"
        "   1. Create a '.env' file in the project root with: PIN_ENV=local\n"
        "   2. Or set it in your terminal: export PIN_ENV=local\n"
        "   3. Or set it in Lightning AI Studio settings.\n\n"
        "In any case, be sure to check .env.example for the expected format "
        "of the .env file and required variables."
    )

ENV = ENV.lower()
RECOGNISED_ENVS = {"local", "cloud"}

if ENV not in RECOGNISED_ENVS:
    raise ValueError(
        f"[Config] Environment [{ENV}] is not recognised. Please set PIN_ENV to"
        f" one of {RECOGNISED_ENVS}"
    )

print(f"[Config] Loading configuration for environment: [{ENV.upper()}]")

# -----------------------------------------------------------------------------
# 2. Shared Constants (Same across all environments)
# -----------------------------------------------------------------------------

#  Check computing power
# ----------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HC_GPU = False
'''HC_GPU is a flag to indicate if we are on the High-Compute GPU.
Note that this only means the GPU has more than 30GB of VRAM'''

if DEVICE == "cuda":
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        HC_GPU = vram_gb >= 30

# For reproducibility
RANDOM_SEED: Final[int] = 42

#  File locations
# ----------------------------------------------------------------

CT_ROOT_STR = os.getenv("LITS_CT_ROOT")
CHECKPOINT_DIR_STR = os.getenv("CHECKPOINT_DIR")
PERSISTENT_DATASET_DIR_STR = os.getenv("PERSISTENT_DATASET_DIR")

LOG_DIR_STR = os.getenv("LOG_DIR")
LOG_LEVEL_CONSOLE = os.getenv("LOG_LEVEL_CONSOLE", "INFO").upper()
LOG_LEVEL_FILE = os.getenv("LOG_LEVEL_FILE", "DEBUG").upper()

#  File validations
# ----------------------------------------------------------------

if not CT_ROOT_STR:
    raise ValueError("[Config] Environment variable 'LITS_CT_ROOT' is not set!")
if not CHECKPOINT_DIR_STR:
    raise ValueError("[Config] Environment variable 'CHECKPOINT_DIR' is not set!")
if not PERSISTENT_DATASET_DIR_STR:
    print("[Config] Warning: 'PERSISTENT_DATASET_DIR' is not set. "
          "PersistentDataset will be disabled.")

if not LOG_DIR_STR:
    raise ValueError(
        "[Config] Environment variable 'LOG_DIR' is not set!\n"
        "Please set it in your '.env' file (see .env.example), for example:\n"
        "   LOG_DIR=/path/where/to/save/logs"
    )
if LOG_LEVEL_CONSOLE not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    print(f"[Config] Warning: LOG_LEVEL_CONSOLE '{LOG_LEVEL_CONSOLE}' is not "
           "valid. Defaulting to 'INFO'.")
    LOG_LEVEL_CONSOLE = "INFO"
if LOG_LEVEL_FILE not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    print(f"[Config] Warning: LOG_LEVEL_FILE '{LOG_LEVEL_FILE}' is not valid. "
           "Defaulting to 'DEBUG'.")
    LOG_LEVEL_FILE = "DEBUG"

CT_ROOT = Path(CT_ROOT_STR)
CHECKPOINT_DIR = Path(CHECKPOINT_DIR_STR)
PERSISTENT_DATASET_DIR = Path(PERSISTENT_DATASET_DIR_STR) if PERSISTENT_DATASET_DIR_STR else None
LOG_DIR = Path(LOG_DIR_STR)

#  Hyperparameters and constants
# ----------------------------------------------------------------

# CTs are in Hounsfield Units: -1000 (air), 0 (water), 40-60 (soft tissues), 100+ (bone)
# we just need liver and tumor, so we can clip the intensities to a smaller range
HU_WINDOW_MIN: Final[int] = -175
HU_WINDOW_MAX: Final[int] = 250
LEARNING_RATE: Final[float] = 1e-4

NUM_CLASSES: Final[int] = 3
'''How many classes to predict.
For binary segmentation, set to 2 (tumour vs background).
For multi-class, set to 3 (background, liver, tumour).'''

TUMOUR_CLASS_INDEX: Final[int] = 2 if NUM_CLASSES == 3 else 1
'''The index of the tumour class in the model's output channels.
For binary segmentation (NUM_CLASSES=2), this should be 1 if the
classes are ordered as [background, tumour]. For multi-class segmentation (NUM_CLASSES=3),
this should be 2 if the classes are ordered as [background, liver, tumour].'''

VAL_BATCH_SIZE: Final[int] = 1
'''DataLoader's batch size for validation. Kept at 1 for deterministic evaluation
and memory safety with large 3D volumes.'''

FIGURE_EPOCH_INTERVAL: Final[int] = 10
'''Interval (in epochs) at which to log segmentation overlay figures to TensorBoard.
Set to 1 to log every epoch, or higher to log less frequently.
Recommended: 5 on final training, 10+ during testing/debugging to save resources.
'''

USE_CACHE_DATASET: Final[bool] = False
'''Whether to use a caching dataset that keeps preprocessed volumes in memory.
This can speed up training but requires more RAM.
If False, PersistentDataset will be used instead
'''

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 12
'''Number of epochs with no improvement after which training will be stopped.'''

EARLY_STOPPING_MIN_DELTA = 0.005
'''Minimum change in the monitored metric to qualify as an improvement.'''

EARLY_STOPPING_RESTORE_BEST = True
'''Whether to restore model weights from the epoch with the best monitored metric
at the end of training.
If True, the model will be rolled back to the state it was in when the best
validation Dice was achieved. If False, the model will remain in the state it
was at the end of the last epoch, even if that is not the best one.
'''

# -----------------------------------------------------------------------------
# 3. Environment-Specific Configuration
# -----------------------------------------------------------------------------
NUM_WORKERS: int
PIN_MEMORY: bool

# This is a parameter that can be tuned based on the GPU VRAM and CPU RAM available.
BATCH_SIZE: int
'''DataLoader's batch size. Set to 1 for memory safety, especially with large 3D volumes.
(Usually between 1 and 4 depending on GPU VRAM)'''

NUM_EPOCHS: int
TRAIN_PATCH_SIZE: tuple
VAL_PATCH_SIZE: tuple

if ENV == "local":
    print("[Config] Running in LOCAL environment.")

    NUM_WORKERS = 0
    PIN_MEMORY = False
    BATCH_SIZE = 1
    NUM_EPOCHS = 5
    TRAIN_PATCH_SIZE = (64, 64, 64)
    VAL_PATCH_SIZE = (64, 64, 64)

elif ENV == "cloud":
    print("[Config] Running in CLOUD environment (Lightning AI). Using more computing power.")

    NUM_WORKERS = 4 if HC_GPU else 0
    PIN_MEMORY = True
    BATCH_SIZE = 4 if HC_GPU else 2
    NUM_EPOCHS = 90
    TRAIN_PATCH_SIZE = (96, 96, 96)
    VAL_PATCH_SIZE = (128, 128, 128)
    # Note: If using SlidingWindowInferer, VAL_PATCH_SIZE determines the window stride/size.
    # However, on full scans, VAL_PATCH_SIZE will be ignored

    # On clouds with high compute GPUS we can afford CacheDataset
    USE_CACHE_DATASET = True

# -----------------------------------------------------------------------------
# 4. Final Safety Check & Directory Creation
# -----------------------------------------------------------------------------
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

if NUM_CLASSES != 2 and NUM_CLASSES != 3:
    raise ValueError("[Config] NUM_CLASSES must be either 2 (for binary segmentation) "
                     "or 3 (for multi-class segmentation).")

if not CT_ROOT.exists():
    raise FileNotFoundError(f"[Config] CT root directory does not exist: {CT_ROOT}")

if DEVICE == "cuda" and not USE_CACHE_DATASET:
    if PERSISTENT_DATASET_DIR is None:
        raise ValueError("[Config] Persistent dataset directory must be set when"
                          " using CUDA without cache dataset.")

if PERSISTENT_DATASET_DIR and not PERSISTENT_DATASET_DIR.exists():
    print("[Config] Persistent dataset directory does not exist. "
          f"Creating: {PERSISTENT_DATASET_DIR}")
    PERSISTENT_DATASET_DIR.mkdir(parents=True, exist_ok=True)


print(f"[Config]   Device: {DEVICE}")
print(f"[Config]   Batch Size: {BATCH_SIZE}")
print(f"[Config]   Val Batch Size: {VAL_BATCH_SIZE}")
print(f"[Config]   Workers: {NUM_WORKERS}")
print(f"[Config]   Data Root: {CT_ROOT}")
print(f"[Config]   Checkpoint Dir: {CHECKPOINT_DIR}")
print(f"[Config]   Log Dir: {LOG_DIR}")
print(f"[Config]   Persistent Dataset Dir: {PERSISTENT_DATASET_DIR}")
# -----------------------------------------------------------------------------
# 5. Helper Functions
# -----------------------------------------------------------------------------

def is_limited_env(include_vram=True) -> bool:
    '''
    Returns True if the current environment is a limited resource
    environment (e.g., local with no GPU).

    if `include_vram=True` (default) it also takes into consideration
    the amount of memory of GPU so a CUDA device with less than 30GB of VRAM
    will be considered a limited environment.
    '''
    if ENV == "local" or DEVICE == "cpu":
        return True

    return include_vram and HC_GPU is False
