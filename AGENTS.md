# agents.md

Guidance for AI coding agents (Claude Code, opencode, etc.) working in this repository.

## Project

Master's thesis: automated liver tumour segmentation using deep learning. Package name:
`idssp.sonk` (Individualised Deep Segmentation... — internal lab naming; don't rename).

- **Scope**: any liver cancer (not HCC-specific). CT phase is unrestricted per dataset but
  must never be mixed within a single dataset, and must be documented per dataset.
- **Primary benchmark dataset**: LiTS (131 labelled volumes).
- **Cross-dataset generalisation targets**: WAW-TACE (233 cases, Zenodo, NIfTI),
  HCC-TACE-Seg (105 cases, TCIA, DICOM). Inclusion of these as *training* extensions
  (vs. eval-only) is an open decision pending a supervisor checkpoint (~Week 9,
  mid-Oct 2026).
- **External evaluation sets**: 3D-IRCADb (**exclude cases 27–48** — they overlap with
  LiTS, do not treat as an independent test set) and CHAOS CT.
- **Metrics**: Dice and HD95 are primary; IoU is a secondary column (mathematically
  monotonic with Dice — don't report it as an independent finding).
- **Graduation target**: full draft ~November 2026.

## Architecture status

| Model | Status | Notes |
|---|---|---|
| ResUNet | Baseline, complete | Tumour Dice mean 0.294 / median 0.180 (N=22) |
| SegResNet | Baseline, complete | Tumour Dice mean 0.509 / median 0.706 (N=22). Best baseline. |
| SwinUNETR | Baseline, complete | Underperforms SegResNet by ~9–12 Dice points at this dataset scale (~88 volumes) |
| **2.5D Mamba-hybrid** | **Primary target — supervisor-approved, Week 1 implementation** | 2D CNN/UNet encoder per axial slice + `mamba_ssm.Mamba`/`Mamba2` block aggregating along z. U-Mamba/SegMamba are design references only, **not** implementation targets — `mamba_ssm.Mamba` is used as a raw component. |

Current default in `config.py` (`MODEL_TO_USE`) is `SEG_RES_NET` — this is the last
validated baseline, not evidence that SegResNet is still the target. Do not "fix" this
back to SwinUNETR or otherwise change the default architecture without being asked;
the Mamba model class doesn't exist yet in `model/models.py` / `AvailableModels`.

## Environment

- Compute: DGX Station A100 (80 GB VRAM, 503 GB RAM). TWCC V100 previously used —
  `run-model.sh` / `validate.sh` still contain TWCC fallback branches, don't strip these.
- Two virtual environments:
  - `~/denv` — validated thesis baseline env. Do not install `mamba-ssm` /
    `causal-conv1d` into this env.
  - `~/denv_mamba` — clone of `~/denv` with Mamba dependencies added, for the new
    architecture work. Keep isolated so baseline reproducibility isn't put at risk.
- `PIN_ENV=local` vs `PIN_ENV=cloud` in `config.py` controls resource-scaled defaults
  (patch size, batch size, epochs, caching). Local is deliberately tiny (debug-only,
  e.g. 5 epochs, 64³ patches) — don't treat local-env numbers as real results.
- Run directories: `/home/misael/model-run/[version]-[timestamp]/`.
- Logs: `/home/misael/jobs/train_[timestamp].log`, launched via `tmux` (see
  `scripts/run-model.sh`, `scripts/validate.sh`, `scripts/rejoin-session.sh`).
- Framework pins: MONAI 1.5.2, PyTorch (see `requirements.txt` for install order —
  torch must be installed first with the correct CUDA/CPU wheel).

## Codebase map

```
idssp/sonk/
  config.py            # frozen dataclass Config; env-aware defaults (local/cloud)
  disk/loader.py        # DataCollector, LiTS-specific pairing, stratified split loading
  model/
    transforms.py       # deterministic + random MONAI transform pipelines
    models.py            # model factory (get_model); AvailableModels enum
    training.py           # ModelBuilder, EarlyStopper, training/validation loop
    inferer.py             # full-volume inference + Invertd (original scanner space)
    evaluator.py            # MetricsEvaluator: Dice/HD95/IoU, raw + post-processed
    data.py                  # VolumeWrapper/DatasetSummary — per-case CSV stats
    validation_affine.md      # design note: why Invertd is needed for saving predictions
  utils/
    logger.py            # shared logging setup, memory usage logging
    mail.py                # SMTP email notifications (async, opt-in)
    notifications.py        # Telegram webhook alerts (async, opt-in, every 5 epochs)
  view/utils.py            # matplotlib plotting, TensorBoard overlay logging
main.py                     # training entrypoint
analyse_dataset.py           # standalone dataset-wide analysis → per-case CSV
scripts/                      # run-model.sh, validate.sh, rejoin-session.sh
files/splits/                  # LiTS_split_seed42.json (79/27/25 stratified split)
files/stats/                    # per-case CSV stats, dictionary.md, problems.md
```

## Data handling rules — do not violate these

- **3-class segmentation**: background (0) / liver (1) / tumour (2).
  `TUMOUR_CLASS_INDEX = 2`. If you see code assuming binary (tumour-only) labels,
  check `config.NUM_CLASSES` before "fixing" it — 2-class mode is a supported
  fallback path, not a bug.
- **LiTS/3D-IRCADb overlap**: LiTS cases 27–48 overlap with 3D-IRCADb cases 1–26.
  Never use IRCADb 1–26 as an independent test set.
- **Known broken affines** (`files/stats/lits/problems.md`):
  - Severe mismatch (label affine is a placeholder identity matrix, image affine is
    real scanner geometry): volumes 48, 49, 50, 51, 52. Fixed at load time by
    `ForceMatchingAffined` in `model/transforms.py`, which copies the image affine
    onto the label — but only for this exact validated set of volume IDs
    (`_ALLOWED_LITS_VOLUMES`). Don't broaden this without re-validating each new case
    manually; the transform raises rather than silently applying to unverified cases.
  - Benign identity-affine cases (both image and label share the same placeholder
    affine, so relative alignment is fine): volumes 28–47 (see `problems.md` for the
    full list). No fix needed for these.
- **Tumour-negative volumes**: effective N for tumour Dice is 22 (test) / 23 (val),
  not the full split size (25/27) — always state N explicitly in any reported metric.
- **HD95**: report only for volumes with non-empty predicted masks; set to `None`
  when tumour Dice is zero (a Hausdorff distance against an empty prediction is
  undefined/meaningless, not "infinite" or "0").
- **Post-processing asymmetry**: the LCC + tumour-masked-to-liver post-processing
  pipeline (`evaluator.py::_post_process_class_map`) helps SegResNet (spatially
  coherent predictions) but can *hurt* SwinUNETR, whose global attention sometimes
  predicts true-positive tumour voxels outside the predicted liver boundary — the
  liver mask then strips them out. Always report **both raw and post-processed**
  results in comparison tables; don't collapse to post-processed-only.

## Coding conventions

- `Config` is a frozen dataclass — never mutate in place; use `dataclasses.replace()`
  (see `inferer.py::load_checkpoint` for the pattern used to align inference config
  with a checkpoint's `config_snapshot`).
- Checkpoints must remain loadable with `weights_only=True` — don't introduce
  non-tensor/non-primitive objects into the checkpoint dict without updating the
  save/load logic in `training.py` and `models.py` accordingly.
- Notifications (`utils/mail.py`, `utils/notifications.py`) are opt-in
  (`ENABLE_EMAIL_NOTIFICATIONS`, `ENABLE_TELEGRAM_NOTIFICATIONS`) and fire-and-forget
  by default (`sync=False`), except at training start/end/failure where `sync=True`
  is used deliberately to guarantee delivery before process exit. Preserve that
  distinction if touching this code.
- `is_limited_env()` gates GPU-only code paths (sliding window inference, full-volume
  `Invertd`-based saving, PersistentDataset/CacheDataset). Local/CPU runs use plain
  `Dataset` + random-crop patches and cannot be inverted back to original scanner
  space — `inferer.py` hard-fails (`RuntimeError`) rather than silently producing
  geometrically wrong NIfTI files. Don't remove this guard.
- Prefer plain markdown tables over interactive visualisation code for
  reporting/comparison output — this project has an explicit "no visualisation
  tooling or extra compute unless asked" policy from Misael.

## What NOT to do without explicit confirmation

- Don't add or install `mamba-ssm` / `causal-conv1d` into `~/denv` (use `~/denv_mamba`).
- Don't change `MODEL_TO_USE` in `config.py`.
- Don't widen `ForceMatchingAffined`'s `_ALLOWED_LITS_VOLUMES` set.
- Don't move WAW-TACE/HCC-TACE-Seg from eval-only to full training extension — that's
  an open Phase 4 scope decision pending supervisor sign-off.
- Don't build visualisation tooling, dashboards, or anything computationally
  non-trivial unless explicitly requested for that task.
