# Context: 2.5D Mamba-Hybrid Architecture — Prototype Complete, Real Model Plan

I am implementing a custom 2.5D Mamba-hybrid architecture for my master's thesis on automated liver tumour segmentation (LiTS dataset, MONAI 1.5.2, PyTorch, DGX A100 80GB). This design is supervisor-approved.

The dummy-tensor prototype phase is complete. The reshape logic, Mamba2 forward contract, AMP compatibility, and fusion path have been validated. The next step is implementing the real model class in `idssp/sonk/model/models.py`, using the settled stage table below and the binding constraints recorded in `AGENTS.md`, especially Section 8.

## Design summary

- A 2D CNN/U-Net encoder processes each axial slice independently (x-y plane).
- A `mamba_ssm.Mamba2` block aggregates features along the z-axis (inter-slice dependencies), using a **pooled-vector aggregation strategy**: each slice's spatial feature map is global-average-pooled to a single vector before being fed into Mamba as a sequence element.
- Primary Mamba variant: `mamba_ssm.Mamba2`. Fallback: `mamba_ssm.Mamba` (v1) if Mamba2 proves difficult during real model integration.
- Mamba is used as a raw component, not adopting a published network (e.g. SegMamba, U-Mamba) wholesale. Those are design references only.
- Prototype phase complete: dummy-tensor prototypes validated the reshape logic, Mamba2 forward shape contract, AMP compatibility, and fusion path **before** writing the real model code in `idssp/sonk/model/models.py`.
- Environment isolation: use `~/mamba-env` for all Mamba execution. Keep `~/denv` clean and do not install `mamba-ssm` or `causal-conv1d` into the baseline environment.

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
| 5 | `mamba_ssm.Mamba2(d_model=256)` | `(8, 128, 256)` | Single forward (causal) pass. Shape preserved. Use the minimal Mamba2 constructor; do not pass Mamba1-style kwargs unless the installed signature is explicitly verified. |
| 6 | Re-merge B·Z | `(1024, 256)` | Plain reshape; must reproduce stage 1's row order exactly. |
| 7a | Broadcast z-context spatially | `(1024, 256, 8, 8)` | `unsqueeze` + `expand`; no memory copy until the concat materialises it. |
| 7b | Concatenate with cached bottleneck | `(1024, 512, 8, 8)` | 256 spatial + 256 context channels per pixel. |
| 8 | 1×1 conv fusion | `(1024, 256, 8, 8)` | `nn.Conv2d(512, 256, 1)`, ≈0.13 M params; decoder input channels constant with or without Mamba (ablation-friendly). |
| 9 | 2D decoder up path, consuming skips | `(1024, 16, 128, 128)` | Channels halve: 256→128→64→32→16. Row order preserved by construction (conv layers act per row independently). |
| 10 | Final 1×1 conv head | `(1024, 3, 128, 128)` | `nn.Conv2d(16, 3, 1)`, 51 params. Raw logits — softmax applied downstream by `DiceCELoss`/`pred_trans`. |
| 11 | Un-merge B·Z to 3D volume | `(8, 3, 128, 128, 128)` | Reshape → `(8, 128, 3, 128, 128)` → **`permute(0, 2, 3, 4, 1)`** → `(B, NUM_CLASSES, X, Y, Z)`. Matches `DiceCELoss`/`DiceMetric`/`SlidingWindowInferer` expectations. |

## Prototype validation status

The prototype phase is complete. Prototype scripts validating the settled stage table have been implemented and run successfully. They are validation artefacts and are not imported by production code.

The original prototype checklist has been completed:

1. **Spatial split/merge round-trip** — an `arange`-based identifiable volume `(B, C, X, Y, Z)` survived splitting into axial slices `(B*Z, C, X, Y)` and merging back element-wise. This validated the stage 1 / stage 11 permute logic.
2. **Sequence reshape round-trip** — an `arange`-based pooled tensor `(B*Z, d_model)` survived reshaping to `(B, Z, d_model)` and back to `(B*Z, d_model)` element-wise. This validated the stage 4 / stage 6 row order.
3. **Logits un-merge round-trip** — an `arange`-based slice-logits tensor `(B*Z, NUM_CLASSES, X, Y)` survived merging to `(B, NUM_CLASSES, X, Y, Z)` and splitting back element-wise. This validated the final stage 11 permutation.
4. **Row-order canary through a minimal placeholder decoder** — identifiable per-row values were passed through a single throwaway `nn.ConvTranspose2d` layer standing in for stage 9 (arbitrary channel count, no skip connections, no real channel schedule), and survived at the expected row indices. This proved the generic property that a conv-based op does not reorder batch rows. **The same canary check must be repeated against the real decoder once stage 9 is actually built** — the placeholder does not retire that obligation.
5. **Shape asserts at every stage**, including `fused.shape[1] == 2 * d_model` after the concat.
6. **Mamba branch CUDA constraint** — Mamba2 was validated on the server inside `~/mamba-env`. Local CPU smoke tests of the real model must bypass the Mamba branch because `mamba_ssm` requires CUDA.

Additional validation completed after the original checklist:

- **Stage 5b AMP validation** — Mamba2 was validated under fp16 `torch.amp.autocast`, `GradScaler`, backward pass, gradient clipping, and optimizer stepping on the A100.
- **Mamba-specific pytest suite** — CUDA-only tests live under `mamba/tests/` and are run with `~/mamba-env` on the server. They are separate from the CPU-only suite in `tests/`.

The following prototype scripts were used:

| Script | Validated scope |
|---|---|
| `mamba/prototype_stages_all.py` | Stages 0–8: spatial split/merge, dummy bottleneck, global average pool, sequence reshape, Mamba2 forward, re-merge, broadcast, concat, 1×1 fusion conv |
| `mamba/prototype_stage05b_mamba2_amp.py` | Mamba2 compatibility with fp16 autocast, `GradScaler`, backward pass, gradient clipping, and optimizer stepping |
| `mamba/prototype_stage09_placeholder_decoder_canary.py` | Row-order preservation through a placeholder convolutional upsampling layer |
| `mamba/prototype_stage11_logits_unmerge.py` | Stage 11 logits un-merge and round-trip back to slice logits |

The assertions worth keeping when implementing the real model are:

- Output shape at every stage.
- The three `arange` round-trips: spatial, sequence, and logits.
- Non-cubic dummy tensors, to prevent cubic shapes from hiding axis bugs.
- The concat channel count: `2 × d_model`.
- The placeholder-decoder row-order canary.
- The row-order invariant: `row = b * Z + z`.

Stage 10 (the real classification head) remains deferred to the real model. It is a trivial `Conv2d(C_base, NUM_CLASSES, 1)` and will be validated implicitly by the real model's forward pass. Stage 11's un-merge is pure reshape/permute logic and is fully covered by the logits un-merge round-trip.

During the prototype phase, any assertion failure was treated as a bug in the prototype measured against the settled table, not as a reason to reopen the table itself. Now that the prototype phase is complete, the table remains the reference contract for the real model unless an explicit design change is approved.

## Outstanding obligations for the real model

- [ ] Re-run the row-order canary against the **real** Stage 9 decoder once it is implemented. The placeholder decoder canary does not retire this obligation.
- [ ] Add `MAMBA_HYBRID_25D` to `AvailableModels` and implement the `get_model()` factory branch only with explicit instruction. `MODEL_TO_USE` must remain `SEG_RES_NET` unless explicitly changed.
- [ ] Include the ablation flag `use_z_context: bool = True`. When `False`, bypass Stages 3–8 and feed the bottleneck directly to the decoder. Decoder input channels must remain `C_bot` regardless of the flag value.
- [ ] Parameter-match the base channel width against SegResNet before comparison experiments.
- [ ] Decide bidirectionality before the first real training run. Training already exposes the causal model to both z-orientations through `RandFlipd(spatial_axis=2)`, but validation/inference remain single-direction unless changed.
- [ ] Align the `~/mamba-env` PyTorch version with the baseline pin before serious training runs.
- [ ] Ensure local `--fast-run` smoke tests bypass the Mamba branch cleanly.

## Things to decide or improve later (toy-first policy)

| Item | Toy decision (now) | Later option / improvement | When to revisit |
|---|---|---|---|
| Mamba directionality | Single forward pass (causal: each slice sees only slices ≤ it) | Bidirectional: second pass over reversed z, sum the outputs to keep all shapes unchanged | Note that `RandFlipd(spatial_axis=2)` already exposes the causal model to both z-orientations during training, but validation/inference remain single-direction — bidirectionality should be decided before the first real training run, not after |
| Mamba variant | `mamba_ssm.Mamba2` as primary variant | Fallback to `mamba_ssm.Mamba` (v1) only if Mamba2 integration proves difficult | During real model implementation; do not re-litigate unless a concrete integration problem appears |
| Mamba constructor | Minimal constructor: `Mamba2(d_model=...)` | Pass additional Mamba2 arguments only after verifying the installed version's signature | If Mamba2 behaviour needs tuning |
| Base channel width | 16 in the full-size settled table; smaller widths used in prototype scripts for speed | 32 or 64, chosen to roughly parameter-match the SegResNet baseline for a fair comparison | Before the comparison experiments in the thesis |
| Pooling into the sequence | Global average pool | Max pool, attention pool, or a coarse spatial grid (e.g. 4 tokens per slice) for a richer sequence | Extension chapter candidate; not the first version |
| Fusion point | Bottleneck only | Multi-level injection into skip connections | Extension; only after the single-point version trains cleanly |
| Fusion mechanism | Broadcast + concat + 1×1 conv | FiLM modulation or residual addition | Deferred; higher debug risk |
| Fusion initialisation | Default init | Zero-init the context half of the fusion weights if early training is unstable | Only if a problem is observed |
| Norm/activation after fusion conv | None (decoder blocks supply their own) | Add norm + ReLU | Only if convergence suggests it |
| Positional information along z | Implicit (recurrence depth + slice content) | Learned z-positional embedding added to tokens before Mamba | Optional extension |
| dtype / device constraints | Mamba2 validated under fp16 autocast + `GradScaler` on the server in `~/mamba-env`; local CPU smoke tests bypass the Mamba branch | Verify bf16 only if fp16 training shows instability; do not manually cast Mamba2 to `.half()` in model code | If training instability is observed |
| Row-order correctness (reshape/permute) | Validated by prototypes, CPU-side axis tests, and CUDA-side Mamba tests | Fold the same asserts into lightweight integration checks for the real model | During real model implementation — this is the one place a silent permutation bug can hide |
| Row-order correctness (decoder specifically) | Prototype validated only against a minimal placeholder decoder (arbitrary conv/upsample, no skips) — proves the general mechanism, not the real architecture | **Re-run the same canary against the real stage 9 decoder once it is built**, including skip-connection concatenation | Immediately when stage 9 is implemented for real — do not treat the placeholder's pass as sufficient |
| Hardcoded sizes | Derive spatial sizes from tensor shapes (`bottleneck.shape[-2:]`); fix 4 downsamples and let bottleneck resolution vary with patch size (4×4 for the local 64³ patch) | Parameterise depth from `config.TRAIN_PATCH_SIZE` | When integrating into `models.py` |
| Validation memory | Rely on the existing OOM fallback in `training.py` / `inferer.py` | Check VRAM at `sw_batch_size=16` → 2048 merged rows; lower `SLIDING_WINDOW_BATCH_SIZE` if needed | First full-volume validation run |
| Ablation switch | A simple boolean flag (e.g. `use_z_context`) that bypasses stages 3–8 and feeds the bottleneck straight to the decoder | Separate registered model variant if the flag proves awkward | When writing `models.py` |
| Pipeline integration | Prototype validation complete; production integration pending | Add an enum entry and a `get_model()` branch only with explicit instruction; `MODEL_TO_USE` default stays `SEG_RES_NET` per `AGENTS.md` | When starting the real model implementation |
| Convolutional block design | Double conv per level (standard UNet) | Single conv for MVP; residual blocks if deeper; depthwise separable for efficiency | If encoder capacity is suspected bottleneck, or if training instability is observed, or when parameter-matching requires a leaner encoder |



## Normalisation decision for the 2.5D Mamba-hybrid — revised

Decision: keep the current preprocessing unchanged.

Current pipeline:
- Clip CT to [-175, 250] HU.
- Scale clipped intensities to [0, 1].
- Use the same deterministic transforms for training, validation, and inference.

Do not introduce Z-score normalisation for the Mamba-hybrid model at this stage.

Reasons:
1. Mamba2 receives encoder bottleneck features, not raw CT intensities.
2. Changing preprocessing only for Mamba would confound architectural comparison.
3. The current pipeline assumes zero means background/air in several places:
   - CropForegroundd uses x > 0.
   - RandCropByPosNegLabeld uses image_threshold=0.
   - SpatialPadd pads with 0.
4. Per-volume Z-score removes absolute HU information that may be useful.
5. The expected benefit is speculative and has not been observed as a failure mode.

If Mamba training is unstable, investigate in this order:
1. Internal normalisation in the 2D encoder/decoder.
2. Avoid BatchNorm over merged B*Z rows; prefer InstanceNorm2d or torch.nn.GroupNorm.
3. Test LayerNorm(C_bot) before Mamba2 in a toy run.
4. Adjust learning rate/warm-up.
5. Only then consider input normalisation as a full ablation applied to all models.

If a future preprocessing variant is introduced:
- Add it to config.
- Include it in PersistentDataset cache keys.
- Include it in config_snapshot.
- Include it in inference strict-key validation.
- Retrain or explicitly isolate the comparison.
