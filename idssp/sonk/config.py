"""Configuration file for the HCC Segmentation Thesis.
Adjust paths and hyperparameters as needed.
"""
import os
from pathlib import Path
from typing import Final
print("Importing torch... (This may take a moment)")
import torch

# -----------------------------------------------------------------------------
# 1. Environment Detection
# -----------------------------------------------------------------------------
ENV = os.getenv("ENV", "local").lower()
RECOGNISED_ENVS = {"local", "cloud"}

if ENV not in RECOGNISED_ENVS:
    raise ValueError(
        f"Environment [{ENV}] is not recognised. Please set ENV to one of {RECOGNISED_ENVS}"
    )

print(f"🚀 Loading configuration for environment: [{ENV.upper()}]")

# -----------------------------------------------------------------------------
# 2. Shared Constants (Same across all environments)
# -----------------------------------------------------------------------------

# For reproducibility
RANDOM_SEED: Final[int] = 42

# Use Python format string syntax. {0} is the volume_id.
# LiTS: volume-1.nii.gz  -> Pattern: "volume-{0}.nii.gz"
# Padded: img_01.nii     -> Pattern: "img_{0:02d}.nii"
IMG_FILENAME_PATTERN: Final[str] = "volume-{0}.nii"
MASK_FILENAME_PATTERN: Final[str] = "segmentation-{0}.nii"

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
CT_ROOT: Path
CHECKPOINT_DIR: Path

BATCH_SIZE: int
'''DataLoader's batch size. Set to 1 for memory safety, especially with large 3D volumes.'''

NUM_EPOCHS: int
TRAIN_PATCH_SIZE: tuple
VAL_PATCH_SIZE: tuple

if ENV == "local":
    print("Running in LOCAL environment.")

    NUM_WORKERS = 0
    PIN_MEMORY = False
    BATCH_SIZE = 1
    NUM_EPOCHS = 10
    TRAIN_PATCH_SIZE = (64, 64, 64)
    VAL_PATCH_SIZE = (64, 64, 64)

    CT_ROOT = Path("/media/sonk/77E0938A53FF065D/ct-scans/media/nas/01_Datasets/CT/LITS/Training Batch 1/")
    CHECKPOINT_DIR = Path("./checkpoints")

elif ENV == "cloud":
    print("Running in CLOUD environment (Lightning AI). Using more computing power.")

    NUM_WORKERS = 4
    PIN_MEMORY = True
    BATCH_SIZE = 1
    NUM_EPOCHS = 100
    TRAIN_PATCH_SIZE = (96, 96, 96)
    VAL_PATCH_SIZE = (128, 128, 128) # Going to be ignored, just for consistency

    CT_ROOT = Path("/workspace/data/CT")
    CHECKPOINT_DIR = Path("/workspace/checkpoints")

# -----------------------------------------------------------------------------
# 4. Final Safety Check & Directory Creation
# -----------------------------------------------------------------------------
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"   Device: {DEVICE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Val Batch Size: {VAL_BATCH_SIZE}")
print(f"   Workers: {NUM_WORKERS}")
print(f"   Data Root: {CT_ROOT}")
print(f"   Checkpoint Dir: {CHECKPOINT_DIR}")

# -----------------------------------------------------------------------------
# 5. Helper Functions
# -----------------------------------------------------------------------------

def is_limited_env() -> bool:
    '''
    Returns True if the current environment is a limited resource
    environment (e.g., local with no GPU).
    '''
    return ENV == "local" and DEVICE == "cpu"
