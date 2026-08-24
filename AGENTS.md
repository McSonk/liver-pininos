# AGENTS.md

Guidance for AI coding agents (Claude Code, opencode, etc.) working in this repository.

## 1. Project Overview

Master's thesis: automated liver tumour segmentation using deep learning. 
- **Package name**: `idssp.sonk` (internal lab naming; do not rename).
- **Scope**: Any liver cancer (not HCC-specific). Note: `config.py` docstring still says "HCC Segmentation Thesis" — this is a legacy string, do not "fix" it unless asked.
- **Primary benchmark**: LiTS (131 labelled volumes).
- **External evaluation**: CHAOS CT.
- **Cross-dataset targets**: WAW-TACE (233 cases), HCC-TACE-Seg (105 cases). Currently eval-only; moving to training is pending supervisor approval.
- **Metrics**: Dice and HD95 (primary). IoU is secondary (monotonic with Dice).
- **Timeline**: Graduation in ~6 months (from August 2026). Full draft ~November 2026. Favour completion over novelty.

## 2. Architecture Status

| Model | Enum / Status | Notes |
|---|---|---|
| UNet (residual) | `U_NET` / Baseline | MONAI `UNet` with `num_res_units`. Referred to as "ResUNet" in thesis prose. |
| SegResNet | `SEG_RES_NET` / Baseline | Best baseline so far. |
| SwinUNETR | `SWIN_UNETR` / Baseline | **Current default in `config.py`**. Underperforms SegResNet at this dataset scale. |
| SwinUNETR Pretrain | `SWIN_UNETR_PRETRAIN` | Loads MONAI pretrained weights. |
| **2.5D Mamba-hybrid** | **Primary Target** | 2D CNN/UNet encoder + `mamba_ssm` block. **Current phase (Week 1):** Install `mamba-ssm` & `causal-conv1d` in `~/denv_mamba`. |
| SegMamba | Comparison Baseline | External repo. Code availability is a hard constraint. |

**Current default in `config.py` (`MODEL_TO_USE`) is `SWIN_UNETR`.** Do not change this default without explicit instruction; the Mamba model class does not exist in `model/models.py` yet.

## 3. Environment & Execution

### Framework Pins
- PyTorch 2.11.0, MONAI 1.5.2. (See `requirements.txt` for exact install order — torch must be installed first with the correct CUDA/CPU wheel).

### Virtual Environments
- **Local**: `~/envs/thesis` (or similar). Used for local development, debugging, and running `do_evaluation.py`.
- **Server Baseline**: `~/denv`. Validated thesis baseline. **DO NOT** install `mamba-ssm` or `causal-conv1d` here.
- **Server Mamba**: `~/denv_mamba`. Isolated clone for Mamba dependencies. 

### Execution Commands
- **Local Training/Eval**: `~/envs/thesis/bin/python main.py` or `do_evaluation.py`.
- **Server Training**: `scripts/run-model.sh`. Handles tmux/nohup, GPU PCI bus binding, and TWCC fallback. **Do not run this locally.**
- **Server Inference**: `scripts/validate.sh`. Runs `do_inference.py` on the server. **Do not run this locally.**
- **Long runs**: Launched via `tmux`. Reattach via `scripts/rejoin-session.sh`.
- **Logs**: `/home/misael/jobs/train_[timestamp].log` (server) or local stdout.

### Mandatory `.env` Variables
`config.init()` will hard-fail if these are missing:
- `PIN_ENV` (`local` | `cloud`)
- `CACHE_TRAIN_SOURCE`, `CACHE_VAL_SOURCE` (`ram` | `disk`)
- `LITS_CT_ROOT`, `LITS_CT_TEST`, `OUTPUT_DIR`, `STATS_DIR`
- `SPLIT_DIR` *(Note: `.env.example` mistakenly lists `SPLIT_JSON`, but `config.py` reads `SPLIT_DIR`. Use `SPLIT_DIR`)*.

## 4. Codebase Map

```text
main.py                     # Training entrypoint
do_evaluation.py            # Local test-time evaluation, metrics, and CSV export
do_inference.py             # Server-side full-volume inference and NIfTI export
analyse_dataset.py          # Standalone dataset-wide analysis → per-case CSV
idssp/sonk/
  config.py                 # Frozen dataclass Config; env-aware defaults
  disk/loader.py            # DataCollector, LiTS pairing, stratified split loading
  stats/stratification.py   # Iterative stratification logic for dataset splitting
  model/
    transforms.py           # Deterministic + random MONAI transform pipelines
    models.py               # Model factory (get_model); AvailableModels enum
    training.py             # ModelBuilder, EarlyStopper, training/validation loop
    inferer.py              # Full-volume inference + Invertd (original scanner space)
    evaluator.py            # MetricsEvaluator: Dice/HD95/IoU, raw + post-processed
    data.py                 # VolumeWrapper/DatasetSummary — per-case CSV stats
  utils/
    logger.py, mail.py, notifications.py
  view/
    utils.py, eval_stats.py # Matplotlib plotting, TensorBoard overlay logging
scripts/                    # run-model.sh, validate.sh, rejoin-session.sh (SERVER ONLY)
files/splits/               # LiTS_split_seed42.json
files/stats/                # Per-case CSV stats, dictionary.md, problems.md
```

## 5. Data Handling Rules (Strict Invariants)

- **3-class segmentation**: background (0) / liver (1) / tumour (2). `TUMOUR_CLASS_INDEX = 2`. If you see code assuming binary labels, check `config.NUM_CLASSES` before "fixing" it — 2-class mode is a supported fallback.
- **LiTS/3D-IRCADb overlap**: LiTS cases 27–48 overlap with 3D-IRCADb cases 1–26. Never use IRCADb 1–26 as an independent test set.
- **Known broken affines**: Volumes 48–52 have severe mismatch (label affine is placeholder). Fixed at load time by `ForceMatchingAffined` in `model/transforms.py` using `_ALLOWED_LITS_VOLUMES`. **Do not widen this set** without manual validation.
- **Tumour-positive volumes**: Effective N for tumour Dice is 22 (test) / 23 (val), not the full split size. Always state N explicitly.
- **HD95**: Report only for volumes with non-empty predicted masks; set to `None` when tumour Dice is zero.
- **Post-processing asymmetry**: LCC + tumour-masked-to-liver helps SegResNet but can hurt SwinUNETR (global attention predicts true-positives outside the liver boundary). Always report **both raw and post-processed** results.

## 6. Coding Conventions

- **Config is frozen**: Never mutate in place; use `dataclasses.replace()` (see `inferer.py` / `do_evaluation.py` for the pattern used to align inference config with a checkpoint's `config_snapshot`).
- **Checkpoints**: Must remain loadable with `weights_only=True`. Do not introduce non-tensor/non-primitive objects into the checkpoint dict.
- **Early Stopping**: Monitors **Tumour Dice** only (`EARLY_STOPPING_PATIENCE=35`, `EARLY_STOPPING_MIN_DELTA=0.001`). Do not switch the monitored metric.
- **AugmentedDataset**: Random training transforms are deliberately separated from the cached deterministic pipeline (`training.py::AugmentedDataset`). Do not merge random transforms back into the deterministic `Compose` when CacheDataset/PersistentDataset is in use.
- **`is_limited_env()`**: Gates GPU-only code paths (sliding window, `Invertd`). Local/CPU runs use plain `Dataset` and cannot be inverted back to original scanner space. `inferer.py` hard-fails rather than producing geometrically wrong NIfTIs. Do not remove this guard.
- **Notifications**: Opt-in and fire-and-forget (`sync=False`), except at training start/end/failure (`sync=True`). Preserve this distinction.
- **Reporting**: Prefer plain markdown tables over interactive visualisation code. "No visualisation tooling or extra compute unless asked" policy.

## 7. What NOT to do without explicit confirmation

- Do not install `mamba-ssm` / `causal-conv1d` into `~/denv` (use `~/denv_mamba`).
- Do not change `MODEL_TO_USE` in `config.py`.
- Do not widen `ForceMatchingAffined`'s `_ALLOWED_LITS_VOLUMES` set.
- Do not move WAW-TACE/HCC-TACE-Seg from eval-only to full training extension.
- Do not build visualisation tooling, dashboards, or computationally non-trivial extras unless explicitly requested.
- Do not attempt to run `scripts/*.sh` locally; they contain server-specific GPU PCI bindings and tmux logic.
