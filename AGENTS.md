# AGENTS.md

Guidance for AI coding agents (Claude Code, opencode, etc.) working in this repository.

## 1. Project Overview

Master's thesis: automated liver tumour segmentation using deep learning.
- **Package name**: `idssp.sonk` (internal lab naming; do not rename).
- **Scope**: Any liver cancer (not HCC-specific). Note: `config.py`'s docstring still
  says "HCC Segmentation Thesis" — this is a legacy string, do not "fix" it unless asked.
- **Primary benchmark**: LiTS (131 labelled volumes).
- **External evaluation**: CHAOS CT. (3D-IRCADb is **not** in scope — do not add IRCADb
  loading, overlap checks, or exclusion logic unless explicitly re-introduced.)
- **Cross-dataset targets**: WAW-TACE (233 cases, Zenodo, NIfTI), HCC-TACE-Seg (105
  cases, TCIA, DICOM). Currently eval-only; moving to training is pending supervisor
  approval (~Week 9, mid-October 2026).
- **Metrics**: Dice and HD95 (primary). IoU is secondary (monotonic with Dice).
- **Timeline**: Graduation in ~6 months (from August 2026). Full draft ~November 2026.
  Favour completion over novelty.

## 2. Architecture Status

| Model | Enum / Status | Notes |
|---|---|---|
| UNet (residual) | `U_NET` / Baseline | MONAI `UNet` with `num_res_units`. Referred to as "ResUNet" in thesis prose. |
| SegResNet | `SEG_RES_NET` / Baseline, **current `MODEL_TO_USE` default** | Best baseline so far. |
| SwinUNETR | `SWIN_UNETR` / Baseline | Underperforms SegResNet by ~9–12 Dice points at this dataset scale (~88 training volumes). |
| SwinUNETR Pretrain | `SWIN_UNETR_PRETRAIN` | Loads MONAI pretrained weights. |
| **2.5D Mamba-hybrid** | **Primary target (supervisor-approved), not yet in `AvailableModels`** | 2D CNN/UNet encoder + `mamba_ssm.Mamba`/`Mamba2` block aggregating along z. Used as a raw component, not a wholesale published network. **Current phase (Week 1):** dummy-tensor prototype of the z-axis aggregation strategy in `~/denv_mamba`. |
| U-Mamba, SegMamba | **Design references only** | Cited in literature review for architectural ideas. **Not implementation targets** — do not add training/eval code for either unless explicitly asked. |

**Current default in `config.py` (`MODEL_TO_USE`) is `SEG_RES_NET`** — the last
validated baseline, not evidence of the target architecture. Do not change this
default without explicit instruction; the Mamba model class does not exist in
`model/models.py` / `AvailableModels` yet.

## 3. Environment & Execution

### Framework Pins
- PyTorch 2.11.0, MONAI 1.5.2. (See `requirements.txt` for exact install order — torch
  must be installed first with the correct CUDA/CPU wheel.)

### Virtual Environments
- **Local**: `~/envs/thesis` (or similar). Used for local development, debugging, and running `do_evaluation.py`.
- **Server baseline**: `~/denv`. Validated thesis baseline. **DO NOT** install
  `mamba-ssm` or `causal-conv1d` here.
- **Server Mamba**: `~/denv_mamba`. Isolated clone of `~/denv` with Mamba dependencies
  added, for the new architecture work. Keep isolated so baseline reproducibility isn't
  put at risk.

### Execution Commands
- **Local training/eval**: `python main.py` or `do_evaluation.py`.
- **Server training**: `scripts/run-model.sh`. Handles tmux/nohup, GPU PCI bus binding,
  and TWCC (V100) fallback. **Do not run this locally.**
- **Server inference**: `scripts/validate.sh`. Runs `do_inference.py` on the server.
  Evaluation/metrics are run separately, locally, via `do_evaluation.py`. **Do not run
  `validate.sh` locally.**
- **Long runs**: launched via `tmux`. Reattach via `scripts/rejoin-session.sh`.
- **Logs**: `/home/misael/jobs/train_[timestamp].log` (server) or local stdout.
- **Compute**: DGX Station A100 (80 GB VRAM, 503 GB RAM) is the primary server;
  TWCC V100 was used previously — this is why the TWCC fallback branches exist in the
  scripts, don't strip them.

### Mandatory `.env` Variables
`config.init()` will hard-fail if these are missing:
- `PIN_ENV` (`local` | `cloud`)
- `CACHE_TRAIN_SOURCE`, `CACHE_VAL_SOURCE` (`ram` | `disk`)
- `LITS_CT_ROOT`, `LITS_CT_TEST`, `OUTPUT_DIR`, `STATS_DIR`
- `SPLIT_JSON` — path to the stratified split JSON file. (`config.py` and
  `.env.example` agree on this name; there is no `SPLIT_DIR` variant.)

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

- **3-class segmentation**: background (0) / liver (1) / tumour (2).
  `TUMOUR_CLASS_INDEX = 2`. If you see code assuming binary labels, check
  `config.NUM_CLASSES` before "fixing" it — 2-class mode is a supported fallback.
- **Known broken affines**: volumes 48–52 have severe mismatch (label affine is a
  placeholder identity matrix). Fixed at load time by `ForceMatchingAffined` in
  `model/transforms.py`, gated to `_ALLOWED_LITS_VOLUMES`. **Do not widen this set**
  without manual validation of each new case.
- **Tumour-negative volumes**: not every LiTS case has a tumour, so the effective N for
  tumour Dice is smaller than the full split size (e.g. 22 of 25 test cases, 23 of 27
  val cases) — always state N explicitly when reporting tumour Dice.
- **HD95**: report only for volumes with non-empty predicted masks; set to `None` when
  tumour Dice is zero.
- **Post-processing asymmetry**: LCC + tumour-masked-to-liver post-processing helps
  SegResNet (spatially coherent predictions) but can hurt SwinUNETR (global attention
  sometimes predicts true-positive tumour voxels outside the predicted liver boundary,
  which the mask then strips). Always report **both raw and post-processed** results.

## 6. Coding Conventions

- **Config is frozen**: never mutate in place; use `dataclasses.replace()` (see
  `inferer.py` / `do_evaluation.py` for the pattern used to align inference config with
  a checkpoint's `config_snapshot`).
- **Checkpoints**: must remain loadable with `weights_only=True`. Do not introduce
  non-tensor/non-primitive objects into the checkpoint dict.
- **Early stopping**: monitors **tumour Dice only** (`EARLY_STOPPING_PATIENCE=35`,
  `EARLY_STOPPING_MIN_DELTA=0.001`). Do not switch the monitored metric.
- **`AugmentedDataset`**: random training transforms are deliberately separated from the
  cached deterministic pipeline (`training.py::AugmentedDataset`) so CacheDataset/
  PersistentDataset caching still benefits from augmentation variability. Do not merge
  random transforms back into the deterministic `Compose` when caching is in use.
- **`is_limited_env()`**: gates GPU-only code paths (sliding window inference, full-
  volume `Invertd`-based saving). Local/CPU runs use plain `Dataset` and random-crop
  patches that cannot be inverted back to original scanner space — `inferer.py`
  hard-fails (`RuntimeError`) rather than silently producing geometrically wrong
  NIfTI files. Do not remove this guard.
- **Notifications**: opt-in and fire-and-forget (`sync=False`) by default, except at
  training start/end/failure where `sync=True` guarantees delivery before process
  exit. Preserve this distinction.
- **Reporting**: prefer plain markdown tables over interactive visualisation code.
  "No visualisation tooling or extra compute unless asked" policy.

## 7. What NOT to do without explicit confirmation

- Do not install `mamba-ssm` / `causal-conv1d` into `~/denv` (use `~/denv_mamba`).
- Do not change `MODEL_TO_USE` in `config.py`.
- Do not widen `ForceMatchingAffined`'s `_ALLOWED_LITS_VOLUMES` set.
- Do not move WAW-TACE/HCC-TACE-Seg from eval-only to full training extension.
- Do not add U-Mamba or SegMamba training/eval code — design references only.
- Do not re-introduce 3D-IRCADb loading or overlap-exclusion logic — out of scope.
- Do not build visualisation tooling, dashboards, or computationally non-trivial extras
  unless explicitly requested.
- Do not run `scripts/*.sh` locally; they contain server-specific GPU PCI bindings and
  tmux logic.
