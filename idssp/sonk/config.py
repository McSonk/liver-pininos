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
# Use Python format string syntax. {0} is the volume_id.
# LiTS: volume-1.nii.gz  -> Pattern: "volume-{0}.nii.gz"
# Padded: img_01.nii     -> Pattern: "img_{0:02d}.nii"
IMG_FILENAME_PATTERN: Final[str] = "volume-{0}.nii"
MASK_FILENAME_PATTERN: Final[str] = "segmentation-{0}.nii"

HU_WINDOW_MIN: Final[int] = -175
HU_WINDOW_MAX: Final[int] = 250
LEARNING_RATE: Final[float] = 1e-3
NUM_CLASSES: Final[int] = 3  # 0: bg, 1: liver, 2: tumour. Change to 1 for binary later.
VAL_BATCH_SIZE: Final[int] = 1  # Keep deterministic and memory-safe
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# 3. Environment-Specific Configuration
# -----------------------------------------------------------------------------
NUM_WORKERS: int
PIN_MEMORY: bool
CT_ROOT: Path
CHECKPOINT_DIR: Path
BATCH_SIZE: int
NUM_EPOCHS: int

if ENV == "local":
    print("⚠️  Running in LOCAL environment.")

    NUM_WORKERS = 0
    PIN_MEMORY = False
    BATCH_SIZE = 1
    NUM_EPOCHS = 10

    CT_ROOT = Path("/media/sonk/77E0938A53FF065D/ct-scans/media/nas/01_Datasets/CT/LITS/Training Batch 1/")
    CHECKPOINT_DIR = Path("./checkpoints")

elif ENV == "cloud":
    print("✅ Running in CLOUD environment (Lightning AI). Using more computing power.")

    NUM_WORKERS = 4
    PIN_MEMORY = True
    BATCH_SIZE = 1
    NUM_EPOCHS = 100

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