# Automated Liver Tumour Segmentation Thesis

This repository contains the code for a master's thesis project on automated liver tumour
segmentation using deep learning, trained and evaluated on the LiTS (Liver Tumor
Segmentation Benchmark) open dataset.

## Project Overview

The aim of this thesis is to develop a model that can automatically segment liver tumours
from CT scans. The primary benchmark is the LiTS dataset (131 labelled volumes), with
external evaluation planned on the CHAOS CT dataset.

## Features

- 3D medical image segmentation using MONAI framework
- SegResNet, UNet (Residual), and SwinUNETR model architectures
- Configurable environments (local/cloud) with automatic GPU detection
- Automatic mixed precision training
- TensorBoard integration for monitoring
- Persistent and in-memory dataset caching with automatic fallback
- Stratified dataset splitting via iterative stratification
- Comprehensive logging and memory usage tracking

## Reproducibility

- Fixed random seed across PyTorch, NumPy, and MONAI transforms
- Deterministic training mode enabled (`torch.use_deterministic_algorithms(True)`)
- All hyperparameters managed via `config.py` + `.env`; environment-specific presets documented
- **Code availability**: This repository contains the complete, runnable codebase required for thesis submission and external validation

## Dataset Summary Analysis

The project includes a dataset-wide analysis utility that produces comprehensive statistics for thesis documentation and preprocessing justification.

### Running the Analysis

```bash
# Basic usage (outputs table + CSV files)
python analyse_dataset.py

# Custom output paths
python analyse_dataset.py --output-csv my_per_case.csv --output-agg-csv my_stats.csv

# Reduced terminal output (suppresses the main analysis table; CSV files are still written)
python analyse_dataset.py --no-verbose --output-csv data.csv
```

### What It Produces

1. **Terminal Table**: Per-case overview showing shape, spacing, orientation, CT range, and tumor presence
2. **Per-Case CSV** (`lits_per_case_summary.csv`): Full metadata for every volume including:
   - Image/label dimensions
   - Voxel spacing (mm)
   - Affine axis codes (orientation)
   - CT intensity min/max
   - Liver/tumor slice ranges
   - Voxel counts and ratios for liver and tumor
3. **Aggregate Stats CSV** (`lits_aggregate_stats.csv`): Dataset-level statistics including:
   - Number of volumes and tumor prevalence
   - Mean/median/std of shapes and spacing
   - Orientation distribution
   - Slice span statistics (liver/tumor extent)
   - Foreground imbalance metrics (voxel ratios)
   - CT intensity ranges

## Installation

### Prerequisites

- Python 3.10+
- PyTorch with CUDA support (for GPU training)
- MONAI framework
- Other dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pininos
   ```

2. Install PyTorch **first** (choose based on your machine):
   ```bash
   # CPU only
   pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu

   # GPU (CUDA 11.8 — check with `nvidia-smi` and use cu121 if needed)
   pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
   ```

3. (Optional) Install mamba
    ```bash
    # Download
    wget https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    # Install
    pip install --no-deps   "./mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    # Verify
    python - <<'PY'
    print("*" * 80)
    print("Verifying Mamba installation...")
    import torch
    import causal_conv1d
    from mamba_ssm import Mamba

    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())

    layer = Mamba(
        d_model=16,
        d_state=16,
        d_conv=4,
        expand=2,
    ).cuda()

    x = torch.randn(2, 8, 16, device="cuda")
    y = layer(x)

    print("output shape:", y.shape)
    print("Mamba test OK")
    PY
```

4. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create your `.env` file and fill in the required paths:
   ```bash
   cp .env.example .env
   ```

## Configuration

The project uses a combination of `config.py` and environment variables (via the `.env` file) for configuration.

### Environment Variables (.env)

Copy `.env.example` to `.env` and modify the following variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `PIN_ENV` | Environment type: `local` or `cloud` | `local` |
| `LITS_CT_ROOT` | Path to LiTS training dataset | `/data/lits/train` |
| `LITS_CT_TEST` | Path to LiTS test dataset | `/data/lits/test` |
| `OUTPUT_DIR` | Directory for checkpoints, logs, TensorBoard, and results | `/data/outputs` |
| `STATS_DIR` | Directory for per-case and aggregate CSV statistics | `/data/stats` |
| `SPLIT_JSON` | Path to the stratified split JSON file | `/data/splits/LiTS_split_seed42.json` |
| `CACHE_TRAIN_SOURCE` | `ram` (fast) or `disk` (memory-safe); falls back to `disk` automatically if RAM < 100 GB | `ram` |
| `CACHE_VAL_SOURCE` | Same as above for validation data | `ram` |
| `PERSISTENT_DATASET_DIR` | (Optional, required if using disk cache) MONAI persistent cache directory | `/data/persistent_cache` |
| `PRE_TRAINED_MODEL_PATH` | (Optional) Path to pretrained weights for SwinUNETR Pretrain model | `/data/pretrained.pth` |
| `LOG_LEVEL_CONSOLE` | Console log level | `INFO` |
| `LOG_LEVEL_FILE` | File log level | `DEBUG` |

### Configuration Modes

The `config.py` file defines two main environment configurations:

1. **Local** (`PIN_ENV=local`):
   - Designed for local machines without a GPU or with limited resources
   - Smaller batch sizes, fewer workers, reduced epochs
   - Patch size: 64x64x64 for training and validation
   - 5 training epochs (for quick debugging)
   - Runs on CPU if no CUDA device is available

2. **Cloud** (`PIN_ENV=cloud`):
   - Designed for GPU-equipped environments
   - Automatically detects high-compute GPUs (>30GB VRAM, e.g., A100 80GB)
   - Patch size: 128x128x128 for training and validation
   - 200 training epochs on high-compute GPUs (5 otherwise)
   - Worker counts and batch sizes scale with available GPU VRAM and system RAM

### Automatic GPU Detection

The configuration automatically detects:
- CUDA availability
- GPU VRAM amount to distinguish between high-compute (≥30GB) and low-compute (<30GB) GPUs
- System RAM and container memory limits (cgroup v1/v2)
- Adjusts settings accordingly (number of workers, pin memory usage, batch size, gradient accumulation)

### Data Preprocessing

- **CT Windowing**: Hounsfield Units clipped to [-175, 250] (soft-tissue liver window)
- **Isotropic Resampling**: All volumes resampled to 1.0x1.0x1.0 mm voxel spacing (local mode defaults to 2.0 mm)
- **Segmentation**: 3 classes — background (0), liver (1), tumour (2)
- **Label Affine Fix**: Volumes 48–52 have a placeholder identity matrix as the label affine; corrected at load time by `ForceMatchingAffined`

## Usage

### Training

```bash
# Basic training
python main.py

# Quick smoke test (fewer epochs, smaller patches)
python main.py --fast-run

# Resume from a checkpoint
python main.py --resume path/to/best_model.pth
```

The training script will:
1. Load configuration from `.env` and `config.py`
2. Initialize logging and set deterministic seeds
3. Load and split the LiTS dataset
4. Initialize data loaders and model
5. Start training with TensorBoard logging and early stopping on tumour Dice

### Monitoring Training

```bash
tensorboard --logdir <OUTPUT_DIR>/<VERSION>-<timestamp>/tensorboard
```

TensorBoard metrics are also written per-epoch for comparison across runs.

### Test-Time Inference (server)

```bash
# Run on the server — generates raw NIfTI predictions
scripts/validate.sh
```

### Local Evaluation

```bash
# Compute metrics locally against downloaded server predictions
python do_evaluation.py

# Point to a specific run
python do_evaluation.py --pred-dir path/to/<run>_test/test_predictions
```

This computes both raw and post-processed Dice and HD95, and generates thesis-ready CSV reports.

## Common Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| `CUDA out of memory` | Patch size/batch too large for GPU | Use `--fast-run` or set `PIN_ENV=local` for smaller presets |
| Dataset not found | Path in `.env` incorrect | Check `LITS_CT_ROOT` and `LITS_CT_TEST` in `.env` |
| Slow data loading | Too many workers for your CPU | Set `CACHE_TRAIN_SOURCE=disk` and reduce `DL_NUM_WORKERS` in `config.py` |
| Early stopping with no improvement | No tumour present in training cases | Early stopping monitors tumour Dice only; check tumour prevalence in your split |

## Project Structure

```text
pininos/
├── main.py                         # Training entry point
├── do_evaluation.py                # Local test-time evaluation and metrics
├── do_inference.py                 # Server-side full-volume inference
├── analyse_dataset.py              # Dataset-wide statistics generator
├── idssp/sonk/
│   ├── config.py                   # Frozen dataclass Config with env-aware defaults
│   ├── model/
│   │   ├── models.py               # Model factory (get_model) and AvailableModels enum
│   │   ├── training.py             # ModelBuilder, EarlyStopper, train/val loop
│   │   ├── transforms.py           # MONAI transform pipelines (deterministic + random)
│   │   ├── inferer.py              # Full-volume inference and spatial inversion
│   │   ├── evaluator.py            # MetricsEvaluator: Dice/HD95/IoU, raw + post-processed
│   │   └── data.py                 # VolumeWrapper and per-case CSV statistics
│   ├── disk/
│   │   └── loader.py               # DataCollector, LiTS pairing, stratified split loading
│   ├── stats/
│   │   └── stratification.py       # Iterative stratification for dataset splitting
│   ├── utils/
│   │   ├── logger.py               # Configurable logging with file + console handlers
│   │   ├── mail.py                 # Email notification utilities
│   │   └── notifications.py        # Telegram and email fire-and-forget notifications
│   └── view/
│       ├── utils.py                # Matplotlib plotting helpers
│       └── eval_stats.py           # Results table and bar chart generation
├── scripts/
│   ├── run-model.sh                # Server training launcher (tmux, GPU binding)
│   ├── validate.sh                 # Server inference launcher
│   └── rejoin-session.sh           # Reattach to running tmux sessions
├── notebooks/
│   ├── strat_dataset.ipynb         # Regenerates the split JSON files
│   └── post-processing.ipynb       # Post-processing analysis
├── files/
│   ├── splits/                     # Stratified split JSONs
│   ├── stats/lits/                 # Per-case CSVs and dataset statistics
│   └── server_logs/affine-issue/   # Investigation notes for volumes 48–52
├── .env.example                    # Template for environment variables
├── requirements.txt                # Python dependencies (install torch first)
└── validation.md                   # Eval pipeline documentation (Invertd, MetaTensor fix)
```

## Dependencies

See `requirements.txt` for a complete list. Key packages:
- [PyTorch](https://pytorch.org/) — GPU-accelerated tensor library
- [MONAI](https://monai.io/) — medical imaging framework (builds on PyTorch)
- [python-dotenv](https://github.com/theskumar/python-dotenv) — `.env` file loader
- [TensorBoard](https://www.tensorflow.org/tensorboard) — training metric visualization
- [nibabel](https://nipy.org/nibabel/) — NIfTI neuroimaging file I/O

## Dataset References

| Dataset | Purpose | Citation |
|---------|---------|----------|
| LiTS | Baseline training & validation | [Bilic et al., 2023](https://competitions.codalab.org/competitions/17094) |
| WAW-TACE | Domain adaptation (TACE-treated HCC) | [Internal, university hospital] |
| HCC-TACE-Seg | Final fine-tuning & evaluation | [Internal, university hospital] |
| CHAOS CT | Cross-dataset generalisation test | [Aktas et al., 2021](https://chaos.grand-challenge.org/) |

## License

This project is part of a master's thesis at NTU (Taiwan). Please refer to the university's policies regarding code usage and distribution.
