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
ENV = os.getenv("ENV")

if ENV is None:
    raise EnvironmentError(
        "[Config] ERROR: Environment variable 'ENV' is not set!\n"
        "Please do one of the following:\n"
        "   1. Create a '.env' file in the project root with: ENV=local\n"
        "   2. Or set it in your terminal: export ENV=local\n"
        "   3. Or set it in Lightning AI Studio settings.\n\n"
        "In any case, be sure to check .env.example for the expected format "
        "of the .env file and required variables."
    )

ENV = ENV.lower()
RECOGNISED_ENVS = {"local", "cloud"}

if ENV not in RECOGNISED_ENVS:
    raise ValueError(
        f"[Config] Environment [{ENV}] is not recognised. Please set ENV to one of {RECOGNISED_ENVS}"
    )

print(f"[Config] Loading configuration for environment: [{ENV.upper()}]")

# -----------------------------------------------------------------------------
# 2. Shared Constants (Same across all environments)
# -----------------------------------------------------------------------------

# For reproducibility
RANDOM_SEED: Final[int] = 42

# File locations
CT_ROOT_STR = os.getenv("LITS_CT_ROOT")
CHECKPOINT_DIR_STR = os.getenv("CHECKPOINT_DIR")
LOG_DIR_STR = os.getenv("LOG_DIR")
LOG_LEVEL_CONSOLE = os.getenv("LOG_LEVEL_CONSOLE", "INFO").upper()
LOG_LEVEL_FILE = os.getenv("LOG_LEVEL_FILE", "DEBUG").upper()

# Validations
if not CT_ROOT_STR:
    raise ValueError("[Config] Environment variable 'LITS_CT_ROOT' is not set!")
if not CHECKPOINT_DIR_STR:
    raise ValueError("[Config] Environment variable 'CHECKPOINT_DIR' is not set!")
if not LOG_DIR_STR:
    print("[Config] Warning: 'LOG_DIR' is not set. Logging to file will be disabled.")
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
LOG_DIR = Path(LOG_DIR_STR) if LOG_DIR_STR else None

# CTs are in Hounsfield Units: -1000 (air), 0 (water), 40-60 (soft tissues), 100+ (bone)
# we just need liver and tumor, so we can clip the intensities to a smaller range
HU_WINDOW_MIN: Final[int] = -175
HU_WINDOW_MAX: Final[int] = 250
LEARNING_RATE: Final[float] = 1e-4

NUM_CLASSES: Final[int] = 3
'''How many classes to predict.
For binary segmentation, set to 1 (tumor vs non-tumor).
For multi-class, set to 3 (background, liver, tumor).'''

VAL_BATCH_SIZE: Final[int] = 1
'''DataLoader's batch size for validation. Kept at 1 for deterministic evaluation
and memory safety with large 3D volumes.'''

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# 3. Environment-Specific Configuration
# -----------------------------------------------------------------------------
NUM_WORKERS: int
PIN_MEMORY: bool

BATCH_SIZE: int
'''DataLoader's batch size. Set to 1 for memory safety, especially with large 3D volumes.'''

NUM_EPOCHS: int
TRAIN_PATCH_SIZE: tuple
VAL_PATCH_SIZE: tuple

if ENV == "local":
    print("[Config] Running in LOCAL environment.")

    NUM_WORKERS = 0
    PIN_MEMORY = False
    BATCH_SIZE = 1
    NUM_EPOCHS = 10
    TRAIN_PATCH_SIZE = (64, 64, 64)
    VAL_PATCH_SIZE = (64, 64, 64)

elif ENV == "cloud":
    print("[Config] Running in CLOUD environment (Lightning AI). Using more computing power.")

    NUM_WORKERS = 4
    PIN_MEMORY = True
    BATCH_SIZE = 1
    NUM_EPOCHS = 100
    TRAIN_PATCH_SIZE = (96, 96, 96)
    VAL_PATCH_SIZE = (128, 128, 128)
    # Note: If using SlidingWindowInferer, VAL_PATCH_SIZE determines the window stride/size.
    # However, on full scans, VAL_PATCH_SIZE will be ignored

# -----------------------------------------------------------------------------
# 4. Final Safety Check & Directory Creation
# -----------------------------------------------------------------------------
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
if LOG_DIR:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

if not CT_ROOT.exists():
    raise FileNotFoundError(f"[Config] CT root directory does not exist: {CT_ROOT}")

print(f"[Config]   Device: {DEVICE}")
print(f"[Config]   Batch Size: {BATCH_SIZE}")
print(f"[Config]   Val Batch Size: {VAL_BATCH_SIZE}")
print(f"[Config]   Workers: {NUM_WORKERS}")
print(f"[Config]   Data Root: {CT_ROOT}")
print(f"[Config]   Checkpoint Dir: {CHECKPOINT_DIR}")
print(f"[Config]   Log Dir: {LOG_DIR}")
# -----------------------------------------------------------------------------
# 5. Helper Functions
# -----------------------------------------------------------------------------

def is_limited_env() -> bool:
    '''
    Returns True if the current environment is a limited resource
    environment (e.g., local with no GPU).
    '''
    return ENV == "local" or DEVICE == "cpu"
