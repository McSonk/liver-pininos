"""Configuration file for the HCC Segmentation Thesis.
Adjust paths and hyperparameters as needed.
"""
import os
import warnings
from pathlib import Path
from typing import Final

import torch
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 0. Load Environment Variables
# -----------------------------------------------------------------------------

print("=" * 80)
print("This is a configuration file.")
print("The configuration is loaded at the start of the program and defines important constants, ")
print("paths, and hyperparameters.")
print("Please review the settings below and adjust them as needed before running the program.")
print("If you are running this for the first time, make sure to create a .env")
print("file based on the .env.example template and fill in the required paths and settings.")
print("")
print("NOTE: The logs of this file won't be stored in the logs directory,")
print("so please pay attention to any warnings or errors printed here.")
print("=" * 80)
print("[Config] Loading environment variables from .env file...")
load_dotenv()

# -----------------------------------------------------------------------------
# 1. Suppress warnings from libraries to keep the logs clean.
# -----------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    # MONAI sliding-window inference triggers a deprecation warning on PyTorch's internal
    # index behaviour. This can be safely ignored.
    message=".*non-tuple sequence for multidimensional indexing.*",
    category=DeprecationWarning,
    module=r"torch(\..*)?$",
)

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
STATS_DIR_STR = os.getenv("STATS_DIR")
SPLIT_DIR_STR = os.getenv("SPLIT_DIR")

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

if not STATS_DIR_STR:
    raise ValueError("[Config] Environment variable 'STATS_DIR' is not set. "
          "Stratification cannot be performed. Please set 'STATS_DIR' to the "
          "directory where the precomputed dataset statistics are stored.")

if not SPLIT_DIR_STR:
    raise ValueError("[Config] Environment variable 'SPLIT_DIR' is not set. "
          "Splitting cannot be performed. Please set 'SPLIT_DIR' to the "
          "directory where the split files will be stored.")

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
STATS_DIR = Path(STATS_DIR_STR)
SPLIT_DIR = Path(SPLIT_DIR_STR)
TRAIN_STATS_DIR = STATS_DIR / "train"
TRAIN_STATS_DIR.mkdir(parents=True, exist_ok=True)
PER_CASE_TRAIN_STATS_FILE = TRAIN_STATS_DIR / "per_case_summary.csv"
PERSISTENT_DATASET_DIR = Path(PERSISTENT_DATASET_DIR_STR) if PERSISTENT_DATASET_DIR_STR else None
LOG_DIR = Path(LOG_DIR_STR)

#  Hyperparameters and constants
# ----------------------------------------------------------------

# CTs are in Hounsfield Units: -1000 (air), 0 (water), 40-60 (soft tissues), 100+ (bone)
# we just need liver and tumor, so we can clip the intensities to a smaller range
# -175 includes liver and fat. -75 would include only liver but it might be too aggressive.
HU_WINDOW_MIN: Final[int] = -175
HU_WINDOW_MAX: Final[int] = 250
LEARNING_RATE: Final[float] = 1e-4

NUM_CLASSES: Final[int] = 3
'''How many classes to predict.
For binary segmentation, set to 2 (tumour vs background).
For multi-class, set to 3 (background, liver, tumour).'''

RAND_CROP_NUM_SAMPLES: Final[int] = 2
'''Number of random crops to extract from each volume during training.
Note that the final batch size will be `BATCH_SIZE` * `RAND_CROP_NUM_SAMPLES`
'''

TUMOUR_CLASS_INDEX: Final[int] = 2 if NUM_CLASSES == 3 else 1
'''The index of the tumour class in the model's output channels.
For binary segmentation (NUM_CLASSES=2), this should be 1 if the
classes are ordered as [background, tumour]. For multi-class segmentation (NUM_CLASSES=3),
this should be 2 if the classes are ordered as [background, liver, tumour].'''

VAL_BATCH_SIZE: Final[int] = 1
'''DataLoader's batch size for validation. Kept at 1 for deterministic evaluation
and memory safety with large 3D volumes. NOT TO BE CONFUSED WITH `VAL_PATCH_SIZE`'''

FIGURE_EPOCH_INTERVAL: Final[int] = 10
'''Interval (in epochs) at which to log segmentation overlay figures to TensorBoard.
Set to 1 to log every epoch, or higher to log less frequently.
Recommended: 5 on final training, 10+ during testing/debugging to save resources.
'''

cache_source = os.getenv("CACHE_SOURCE", "ram").lower()
if cache_source not in {"ram", "disk"}:
    print(f"[Config] Warning: CACHE_SOURCE '{cache_source}' is not valid. "
          "Defaulting to 'ram'.")
    cache_source = "ram"

USE_CACHE_DATASET: Final[bool] = cache_source == "ram"
'''Whether to use a caching dataset that keeps preprocessed volumes in memory.
This can speed up training but requires more RAM.
If False, PersistentDataset will be used instead
'''

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 12
'''Number of epochs with no improvement after which training will be stopped.'''

EARLY_STOPPING_MIN_DELTA = 0.005
'''Minimum change in the monitored metric to qualify as an improvement.'''

# -----------------------------------------------------------------------------
# 3. Environment-Specific Configuration
# -----------------------------------------------------------------------------
# Run `nproc` or `lscpu` for number of CPU cores. NUM_WORKERS < number of cores.
NUM_WORKERS: int
'''Number of parallel processes for data loading (CacheDataset or DataLoader)'''
PIN_MEMORY: bool
NUM_EPOCHS: int
TRAIN_PATCH_SIZE: tuple
VAL_PATCH_SIZE: tuple
'''The size of the 3D patches to be extracted from the volumes for training and validation.
(Only used in local env). NOT TO BE CONFUSED WITH `VAL_BATCH_SIZE`'''

# This is a parameter that can be tuned based on the GPU VRAM and CPU RAM available.
BATCH_SIZE: int
'''DataLoader's batch size. Set to 1 for memory safety, especially with large 3D volumes.
(Usually between 1 and 4 depending on GPU VRAM)
Note that the final batch size will be `BATCH_SIZE` * `RAND_CROP_NUM_SAMPLES`'''

ISO_SPACING: tuple
'''The isotropic spacing to which all CT volumes will be resampled.
A good choice is (1.5, 1.5, 1.5). It is memory efficient but it might introduce
some blurring. (1.0, 1.0, 1.0) is an optimal choice.
Please note that in LiTS most of the volumes have a z spacing of around 0.7-1.0mm, 
so resampling to 1.0mm will not introduce much blurring while ensuring. However,
there are 2 volumes with a z spacing of .5mm, so here we should be more careful.'''

if ENV == "local":
    print("[Config] Running in LOCAL environment.")

    NUM_WORKERS = 0
    PIN_MEMORY = False
    BATCH_SIZE = 1
    NUM_EPOCHS = 5
    TRAIN_PATCH_SIZE = (64, 64, 64)
    VAL_PATCH_SIZE = (64, 64, 64)
    ISO_SPACING = (2.0, 2.0, 2.0)

elif ENV == "cloud":
    print("[Config] Running in CLOUD environment (Lightning AI). Using more computing power.")

    NUM_WORKERS = 4 if HC_GPU else 2
    PIN_MEMORY = True
    BATCH_SIZE = 4 if HC_GPU else 2
    NUM_EPOCHS = 90
    TRAIN_PATCH_SIZE = (96, 96, 96)
    # Not used by the standard cloud validation pipeline, but kept for config/logging
    # consistency and for code paths that still reference VAL_PATCH_SIZE.
    VAL_PATCH_SIZE = TRAIN_PATCH_SIZE
    # TODO: Tune. Both options sound valid, so decide which is better based on experiments.
    ISO_SPACING = (1.0, 1.0, 1.0) if HC_GPU else (1.5, 1.5, 1.5)

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

if not STATS_DIR.exists():
    print("[Config] Stats directory does not exist. "
          f"Creating: {STATS_DIR}")
    STATS_DIR.mkdir(parents=True, exist_ok=True)

if not SPLIT_DIR.exists():
    print("[Config] Split directory does not exist. "
          f"Creating: {SPLIT_DIR}")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)


print(f"[Config]   Device: {DEVICE}")
print(f"[Config]   Batch Size: {BATCH_SIZE}")
print(f"[Config]   RAND_CROP_NUM_SAMPLES: {RAND_CROP_NUM_SAMPLES} (Effective "
      f"Batch Size: {BATCH_SIZE * RAND_CROP_NUM_SAMPLES})")
print(f"[Config]   Val Batch Size: {VAL_BATCH_SIZE}")
print(f"[Config]   Workers: {NUM_WORKERS}")
print(f"[Config]   Data Root: {CT_ROOT}")
print(f"[Config]   Checkpoint Dir: {CHECKPOINT_DIR}")
print(f"[Config]   Log Dir: {LOG_DIR}")
print(f"[Config]   Persistent Dataset Dir: {PERSISTENT_DATASET_DIR}")
print("=" * 80)

# -----------------------------------------------------------------------------
# 5. Email Notification Configuration
# -----------------------------------------------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", ""))
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "")
ENABLE_EMAIL_NOTIFICATIONS = os.getenv("ENABLE_EMAIL_NOTIFICATIONS", "false").lower() == "true"

if ENABLE_EMAIL_NOTIFICATIONS:
    if not all([SMTP_HOST, SMTP_PORT, EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT]):
        raise ValueError(
            "[Config] Email notifications are enabled but one or more email configuration "
            "variables are missing. Please set SMTP_HOST, SMTP_PORT, EMAIL_SENDER, "
            "EMAIL_PASSWORD, and EMAIL_RECIPIENT in your .env file."
        )
    else:
        print("[Config] Email notifications are enabled. Emails will be sent to "
              f"{EMAIL_RECIPIENT} at the end of training and for exceptions handled during training.")
else:
    print("[Config] Email notifications are disabled. To enable, set "
            "ENABLE_EMAIL_NOTIFICATIONS=true and provide the required email "
            "configuration in the .env file.")

# -----------------------------------------------------------------------------
# 6. Helper Functions
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

def to_dict() -> dict:
    """Returns a serialisable snapshot of all configuration constants."""
    return {
        # Environment & Device
        "ENV": ENV,
        "DEVICE": DEVICE,
        "HC_GPU": HC_GPU,
        "RANDOM_SEED": RANDOM_SEED,

        # Preprocessing
        "HU_WINDOW_MIN": HU_WINDOW_MIN,
        "HU_WINDOW_MAX": HU_WINDOW_MAX,
        "ISO_SPACING": list(ISO_SPACING),  # tuple → list for JSON/weights compatibility
        "TRAIN_PATCH_SIZE": list(TRAIN_PATCH_SIZE),
        "VAL_PATCH_SIZE": list(VAL_PATCH_SIZE),
        "USE_CACHE_DATASET": USE_CACHE_DATASET,

        # Training Hyperparameters
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "VAL_BATCH_SIZE": VAL_BATCH_SIZE,
        "RAND_CROP_NUM_SAMPLES": RAND_CROP_NUM_SAMPLES,
        "NUM_WORKERS": NUM_WORKERS,
        "PIN_MEMORY": PIN_MEMORY,
        "NUM_EPOCHS": NUM_EPOCHS,
        "NUM_CLASSES": NUM_CLASSES,
        "TUMOUR_CLASS_INDEX": TUMOUR_CLASS_INDEX,

        # Early Stopping
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "EARLY_STOPPING_MIN_DELTA": EARLY_STOPPING_MIN_DELTA,

        # Paths (convert Path objects to strings)
        "CT_ROOT": str(CT_ROOT),
        "CHECKPOINT_DIR": str(CHECKPOINT_DIR),
        "PERSISTENT_DATASET_DIR": str(PERSISTENT_DATASET_DIR) if PERSISTENT_DATASET_DIR else None,
        "STATS_DIR": str(STATS_DIR) if STATS_DIR else None,
        "LOG_DIR": str(LOG_DIR),
        "LOG_LEVEL_CONSOLE": LOG_LEVEL_CONSOLE,
        "LOG_LEVEL_FILE": LOG_LEVEL_FILE,
        "FIGURE_EPOCH_INTERVAL": FIGURE_EPOCH_INTERVAL
    }
