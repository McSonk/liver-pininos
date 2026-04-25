# HCC Liver Tumor Segmentation Thesis Project

This repository contains the code for a master's thesis project focused on developing an automated segmentation model for Hepatocellular Carcinoma (HCC) liver tumours using deep learning techniques.

## Project Overview

The aim of this thesis is to train a model to automatically segment HCC liver tumours. The current implementation uses the LiTS (Liver Tumor Segmentation) open dataset for initial development and testing, with plans to expand to HCC-specialized datasets and private data from the university hospital in later stages.

## Features

- 3D medical image segmentation using MONAI framework
- Configurable environments (local/cloud) for different computational resources
- Automatic mixed precision training
- TensorBoard integration for monitoring
- Persistent dataset caching for improved performance
- Comprehensive logging and memory usage tracking

## Reproducibility

- Fixed random seed across PyTorch, NumPy, and MONAI transforms
- Deterministic training mode enabled (`torch.use_deterministic_algorithms(True)`)
- All hyperparameters managed via `config.py` + `.env`; environment-specific presets documented
- **Code availability**: This repository contains the complete, runnable codebase required for thesis submission and external validation

## Installation

### Prerequisites

- Python 3.8+
- PyTorch with CUDA support (for GPU training)
- MONAI framework
- Other dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pininos
   ```

2. Install dependencies:
    Install PyTorch FIRST (Choose one based on your machine):

    ```bash
    pip install torch==2.11.0 torchvision==0.26.0 --index-url https://check.torch.site
    ```

    Afterwards, continue as usual:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on the provided `.env.example`:
   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file to configure your environment:
   - Set `ENV` to either `local` or `cloud`
   - Set `LITS_CT_ROOT` to the path of your LiTS dataset
   - Configure other paths and settings as needed

## Configuration

The project uses a combination of `config.py` and environment variables (via `.env` file) for configuration.

### Environment Variables (.env)

Copy `.env.example` to `.env` and modify the following variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `ENV` | Environment type: `local` or `cloud` | `local` |
| `LITS_CT_ROOT` | Path to LiTS dataset containing CT scans | `/data/lits` |
| `CHECKPOINT_DIR` | Directory to save model checkpoints | `/data/checkpoints` |
| `PERSISTENT_DATASET_DIR` | (Optional) Directory for MONAI persistent dataset cache | `/data/persistent_cache` |
| `LOG_DIR` | Directory to save log files | `/data/logs` |
| `LOG_LEVEL_CONSOLE` | Console log level | `INFO` |
| `LOG_LEVEL_FILE` | File log level | `DEBUG` |

### Configuration Modes

The `config.py` file defines two main environment configurations:

1. **Local** (`ENV=local`):
   - Designed for local computers without GPU or with limited resources
   - Smaller batch sizes, fewer workers, reduced epochs
   - Patch size: 64³ for training, 64³ for validation
   - 5 training epochs

2. **Cloud** (`ENV=cloud`):
   - Designed for cloud environments with more computational power
   - Automatically detects high-compute GPUs (>30GB VRAM, e.g., A100)
   - Adjusts workers and pin memory based on GPU capabilities
   - Patch size: 96³ for training, 128³ for validation
   - 90 training epochs

### Automatic GPU Detection

The configuration automatically detects:
- CUDA availability
- GPU VRAM amount to distinguish between high-compute (≥30GB) and low-compute (<30GB) GPUs
- Adjusts settings accordingly (number of workers, pin memory usage)

### Data Preprocessing

- **CT Windowing**: Hounsfield Units clipped to [-175, 250] (soft-tissue liver window)
- **Label Mapping (LiTS)**: Background=0, Liver=1, Tumour=1 (binary segmentation; LiTS label 2 → 1)
- **Classes**: `NUM_CLASSES = 3` (background, liver, tumour) for multi-class output head
- **Phase**: Portal venous phase CT scans only

## Usage

To start training:

```bash
python main.py
```

The script will:
1. Load configuration from `.env` and `config.py`
2. Initialize logging and set deterministic seeds
3. Load and split the LiTS dataset
4. Initialize data loaders and model
5. Start training with TensorBoard logging

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir logs/tensorboard
```

## Common Issues

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| `CUDA out of memory` | Patch size/batch too large for GPU | Reduce `TRAIN_PATCH_SIZE` or set `ENV=local` for smaller presets |
| Dataset not found | `LITS_CT_ROOT` path incorrect | Verify path in `.env`; ensure dataset follows LiTS directory structure |
| Slow data loading | `num_workers` too high for CPU | Lower `NUM_WORKERS` in `config.py` for local execution |

## Project Structure

```
pininos/
├── idssp/                 # Main source code package
│   └── sonk/              # Thesis-specific modules
│       ├── config.py      # Configuration management
│       ├── model/         # Model architecture and training
│       ├── utils/         # Utility functions (logging, etc.)
│       └── disk/          # Data loading and processing
├── checkpoints/           # Saved model checkpoints
├── logs/                  # Log files
├── .env.example           # Template for environment variables
├── requirements.txt       # Python dependencies
├── main.py                # Entry point for training
└── README.md              # This file
```

## Dependencies

See `requirements.txt` for a complete list of dependencies. Key packages include:
- PyTorch
- MONAI
- python-dotenv
- TensorBoard

## Notes

- The current implementation uses deterministic seeds for reproducibility
- In limited environments (local/CPU), the script automatically uses a subset of data for quick testing
- Checkpoint directories are created automatically if they don't exist
- Logging is configurable for both console and file output

## Future Work

As mentioned in the thesis objectives, future stages of this project will:
1. Adapt the model for HCC-specialized datasets
2. Incorporate private data from the university hospital
3. Experiment with different model architectures and loss functions
4. Perform extensive validation and comparison with ground truth annotations


## Dataset References

| Dataset | Purpose | Citation |
|---------|---------|----------|
| LiTS | Baseline training & validation | [Bilic et al., 2023](https://competitions.codalab.org/competitions/17094) |
| WAW-TACE | Domain adaptation (TACE-treated HCC) | [Internal, university hospital] |
| HCC-TACE-Seg | Final fine-tuning & evaluation | [Internal, university hospital] |
| 3Dircadb | External validation (cases 1–26 only) | [https://www.ircad.fr/research/3d-ircadb-01/](https://www.ircad.fr/research/3d-ircadb-01/) |
| CHAOS CT | Cross-dataset generalisation test | [Aktas et al., 2021](https://chaos.grand-challenge.org/) |

## License

This project is part of a master's thesis at NTU (Taiwan). Please refer to the university's policies regarding code usage and distribution.
