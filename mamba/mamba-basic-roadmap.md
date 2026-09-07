# Context: 2.5D Mamba-Hybrid Architecture — Tensor Shape Design & Prototype Plan

I am implementing a custom 2.5D Mamba-hybrid architecture for my master's thesis on automated liver tumour segmentation (LiTS dataset, MONAI 1.5.2, PyTorch, DGX A100 80GB). This design is supervisor-approved. I need help continuing the implementation from where I left off — picking up specifically at the tensor-shape / architecture design stage, before writing real model code.

## Design summary

- A 2D CNN/U-Net encoder processes each axial slice independently (x-y plane).
- A `mamba_ssm.Mamba` (or `Mamba2`) block aggregates features along the z-axis (inter-slice dependencies), using a **pooled-vector aggregation strategy**: each slice's spatial feature map is global-average-pooled to a single vector before being fed into Mamba as a sequence element.
- Mamba is used as a raw component, not adopting a published network (e.g. SegMamba, U-Mamba) wholesale. Those are design references only.
- Implementation order: dummy-tensor prototype (no real CNN, `torch.randn` placeholders + real `mamba_ssm.Mamba` calls) to validate the reshape logic, **before** writing the real model code in `idssp/sonk/model/models.py`.
- Environment isolation: `~/mamba-env` (cloned from `~/denv`) to protect the already-validated baseline environment.

## Config constants relevant to shapes

- `TRAIN_PATCH_SIZE = (128, 128, 128)` (X, Y, Z) on the A100/cloud config. *(Note: `RandZoomd` defaults to `keep_size=True`, guaranteeing the spatial dimensions remain exactly 128³ after augmentation.)*
- `BATCH_SIZE = 4` (cloud, super-HC-GPU tier).
- `RAND_CROP_NUM_SAMPLES = 2`.
- `NUM_CLASSES = 3` (background=0, liver=1, tumour=2).
- External MONAI tensor convention: `(B, C, X, Y, Z)`.
- z-axis = Z = the last spatial axis = tensor dimension 4 (confirmed via `verify_z_axis.py`).

## Settled table (v4 - final)

| Stage | Operation | Output shape | Notes |
|---|---|---|---|
| 0 | Input patch | `(8, 1, 128, 128, 128)` | `(B, C, X, Y, Z)`; B=8 effective = `BATCH_SIZE(4) × RAND_CROP_NUM_SAMPLES(2)` on A100. Z = S/I slice axis, last (confirmed via `verify_z_axis.py`). |
| 1 | Permute + merge B·Z | `(1024, 1, 128, 128)` | **`permute(0, 4, 1, 2, 3)`** → `(B, Z, C, X, Y)` → reshape. Row order: volume-major, Z-minor, `row = b·128 + z`. |
| 2 | 2D encoder down path (4 downs: 128→64→32→16→8) | Bottleneck `(1024, 256, 8, 8)` | Channels double per level: 16→32→64→128→256. Skips cached: `(1024,16,128²)`, `(1024,32,64²)`, `(1024,64,32²)`, `(1024,128,16²)`. |
| 3 | Global average pool over (X′, Y′) | `(1024, 256)` | Converts each slice's spatial bottleneck into a fixed-length vector — the format Mamba's `d_model` input requires, keeping `d_model = 256` equal to the bottleneck channels. Sequence length (128) is unaffected; it is already fixed by Z. |
| 4 | Un-merge to `(B, Z, d_model)` | `(8, 128, 256)` | Plain reshape — no permute needed, since stage 1's row-major order restores `(b, z)` correspondence directly. |
| 5 | `mamba_ssm.Mamba(d_model=256)` | `(8, 128, 256)` | Toy: single forward (causal) pass. Shape preserved. |
| 6 | Re-merge B·Z | `(1024, 256)` | Plain reshape; must reproduce stage 1's row order exactly. |
| 7a | Broadcast z-context spatially | `(1024, 256, 8, 8)` | `unsqueeze` + `expand`; no memory copy until the concat materialises it. |
| 7b | Concatenate with cached bottleneck | `(1024, 512, 8, 8)` | 256 spatial + 256 context channels per pixel. |
| 8 | 1×1 conv fusion | `(1024, 256, 8, 8)` | `nn.Conv2d(512, 256, 1)`, ≈0.13 M params; decoder input channels constant with or without Mamba (ablation-friendly). |
| 9 | 2D decoder up path, consuming skips | `(1024, 16, 128, 128)` | Channels halve: 256→128→64→32→16. Row order preserved by construction (conv layers act per row independently). |
| 10 | Final 1×1 conv head | `(1024, 3, 128, 128)` | `nn.Conv2d(16, 3, 1)`, 51 params. Raw logits — softmax applied downstream by `DiceCELoss`/`pred_trans`. |
| 11 | Un-merge B·Z to 3D volume | `(8, 3, 128, 128, 128)` | Reshape → `(8, 128, 3, 128, 128)` → **`permute(0, 2, 3, 4, 1)`** → `(B, NUM_CLASSES, X, Y, Z)`. Matches `DiceCELoss`/`DiceMetric`/`SlidingWindowInferer` expectations. |

## Prototype checklist (the run in `~/mamba-env`)

1. **Spatial split/merge round-trip** — an `arange`-based identifiable volume `(B, C, X, Y, Z)` must survive splitting into axial slices `(B*Z, C, X, Y)` and merging back element-wise. This validates the stage 1 / stage 11 permute logic.
2. **Sequence reshape round-trip** — an `arange`-based pooled tensor `(B*Z, d_model)` must survive reshaping to `(B, Z, d_model)` and back to `(B*Z, d_model)` element-wise. This validates the stage 4 / stage 6 row order.
3. **Logits un-merge round-trip** — an `arange`-based slice-logits tensor `(B*Z, NUM_CLASSES, X, Y)` must merge to `(B, NUM_CLASSES, X, Y, Z)` and split back element-wise. This validates the final stage 11 permutation.
4. **Row-order canary through a minimal placeholder decoder** — inject identifiable per-row values, pass them through a single throwaway `nn.ConvTranspose2d` layer standing in for stage 9 (arbitrary channel count, no skip connections, no real channel schedule), and assert they survive at the expected row indices. This proves the generic property "a conv-based op doesn't reorder batch rows". **The same canary check must be repeated against the real decoder once stage 9 is actually built** — this placeholder does not retire that obligation.
5. **Shape asserts at every stage**, including `fused.shape[1] == 2 * d_model` after the concat.
6. Remember the Mamba branch requires CUDA — run on the server, not locally; local `--fast-run` smoke tests of the real model will need the Mamba branch bypassed.

Any assertion failure there is a bug in the prototype measured against this table, not a reason to reopen the table itself.

## Suggested prototype scope (one small script)

Stages 0–8 are the core reshape story. Stage 9 (the real decoder) is deferred to the real model. However, to close the loop on the row-order canary (checklist item 4), the prototype will include a **minimal placeholder decoder** (one throwaway `nn.ConvTranspose2d` layer, arbitrary channels, no skips). This placeholder exists *strictly* to exercise the canary and prove that convolutional upsampling preserves the `row = b·128 + z` ordering; it is **not** a stand-in for stage 9's real architecture and must not be reused as one.

The assertions worth keeping:
- Output shape at every stage.
- The three `arange` round-trips (Spatial, Sequence, Logits) using non-cubic dummy tensors (e.g. `B=2, X=16, Y=24, Z=32`) to prevent cubic shapes from hiding axis bugs.
- The concat channel count (`2 × d_model`).
- The placeholder-decoder row-order canary.

Stage 10 (the real classification head) can still be deferred to the real model. Stage 11's un-merge, however, is pure reshape/permute logic and is fully covered by the Logits un-merge round-trip.

**Follow-up obligation, not to be skipped:** when stage 9 is built for real (real channel schedule, real skip-connection concatenation), re-run the row-order canary against it specifically. The placeholder's pass here is not evidence the real decoder preserves row order — it only proves the general mechanism can.

## Things to decide or improve later (toy-first policy)

| Item | Toy decision (now) | Later option / improvement | When to revisit |
|---|---|---|---|
| Mamba directionality | Single forward pass (causal: each slice sees only slices ≤ it) | Bidirectional: second pass over reversed z, sum the outputs to keep all shapes unchanged | Note that `RandFlipd(spatial_axis=2)` already exposes the causal model to both z-orientations during training, but validation/inference remain single-direction — bidirectionality should be decided before the first real training run, not after |
| Mamba variant | `mamba_ssm.Mamba` | `Mamba2` if speed/VRAM benefits appear | Only if needed; do not chase prematurely |
| Base channel width | 16 (lightweight) | 32 or 64, chosen to roughly parameter-match the SegResNet baseline for a fair comparison | Before the comparison experiments in the thesis |
| Pooling into the sequence | Global average pool | Max pool, attention pool, or a coarse spatial grid (e.g. 4 tokens per slice) for a richer sequence | Extension chapter candidate; not the first version |
| Fusion point | Bottleneck only | Multi-level injection into skip connections | Extension; only after the single-point version trains cleanly |
| Fusion mechanism | Broadcast + concat + 1×1 conv | FiLM modulation or residual addition | Deferred; higher debug risk |
| Fusion initialisation | Default init | Zero-init the context half of the fusion weights if early training is unstable | Only if a problem is observed |
| Norm/activation after fusion conv | None (decoder blocks supply their own) | Add norm + ReLU | Only if convergence suggests it |
| Positional information along z | Implicit (recurrence depth + slice content) | Learned z-positional embedding added to tokens before Mamba | Optional extension |
| dtype / device constraints | Prototype on the server in `~/mamba-env` only | Verify fp16/bf16 behaviour under `autocast`; local CPU smoke tests must bypass the Mamba branch (`mamba_ssm` requires CUDA) | During the prototype |
| Row-order correctness (reshape/permute) | Prototype validates merge/unmerge round-trips with identifiable `arange`-based tensors | Fold the same asserts into a lightweight unit check in the real model | Prototype stage — this is the one place a silent permutation bug can hide |
| Row-order correctness (decoder specifically) | Prototype validates only against a minimal placeholder decoder (arbitrary conv/upsample, no skips) — proves the general mechanism, not the real architecture | **Re-run the same canary against the real stage 9 decoder once it is built**, including skip-connection concatenation | Immediately when stage 9 is implemented for real — do not treat the placeholder's pass as sufficient |
| Hardcoded sizes | Derive spatial sizes from tensor shapes (`bottleneck.shape[-2:]`); fix 4 downsamples and let bottleneck resolution vary with patch size (4×4 for the local 64³ patch) | Parameterise depth from `config.TRAIN_PATCH_SIZE` | When integrating into `models.py` |
| Validation memory | Rely on the existing OOM fallback in `training.py` / `inferer.py` | Check VRAM at `sw_batch_size=16` → 2048 merged rows; lower `SLIDING_WINDOW_BATCH_SIZE` if needed | First full-volume validation run |
| Ablation switch | A simple boolean flag (e.g. `use_z_context`) that bypasses stages 3–8 and feeds the bottleneck straight to the decoder | Separate registered model variant if the flag proves awkward | When writing `models.py` |
| Pipeline integration | Keep everything in the isolated prototype for now | Add an enum entry and a `get_model()` branch; `MODEL_TO_USE` default stays `SEG_RES_NET` (per AGENTS.md — explicit instruction required) | After the prototype validates |
