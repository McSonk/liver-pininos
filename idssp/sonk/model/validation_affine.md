# Saving Predictions in Original Scanner Space

## Problem Statement

During test-time inference, the model's predictions exist in **preprocessed space** — the
coordinate system established by the deterministic transform pipeline. Saving these
predictions directly to disk with the preprocessed affine matrix produces NIfTI files
that are geometrically disconnected from the raw CT scans.

### The Transform Pipeline

The validation transforms (`get_validation_transforms`) apply the following spatial
operations in sequence:

| Order | Transform | Effect on Geometry |
| :---: | :--- | :--- |
| 1 | `Orientationd(axcodes="LAS")` | Permutes/flips axes to enforce Left-Anterior-Superior orientation |
| 2 | `Spacingd(pixdim=(1.0, 1.0, 1.0))` | Resamples from raw voxel spacing (e.g., 0.83 × 0.83 × 2.5 mm) to 1.0 mm isotropic |
| 3 | `CropForegroundd(margin=10)` | Removes background air, cropping to a bounding box around the liver + tumour |
| 4 | `SpatialPadd(spatial_size=TRAIN_PATCH_SIZE)` | Zero-pads to ensure minimum dimensions of `TRAIN_PATCH_SIZE` |

After these transforms, a raw volume of shape `(512, 512, 237)` with spacing
`(0.83, 0.83, 2.5)` mm becomes a cropped, padded volume of shape
`(128, 128, 128)` with spacing `(1.0, 1.0, 1.0)` mm and a shifted physical origin.

### The Original (Incorrect) Saving Code

```python
pred_class_map = processed_preds[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
pred_nib = nib.Nifti1Image(pred_class_map, affine=batch["image"].affine[0].numpy())
nib.save(pred_nib, ...)
```

This saved the prediction tensor (in preprocessed space) with the **preprocessed affine**
(the affine after all spatial transforms have been applied). The resulting NIfTI file:

- Has the wrong **shape** (e.g., `128×128×128` instead of `512×512×237`)
- Has the wrong **voxel spacing** (1.0 mm isotropic instead of the scanner's native spacing)
- Has the wrong **physical origin** (shifted by the crop operation)
- Has the wrong **axis orientation** (forced LAS instead of the scanner's native orientation)

Any external tool (ITK-SNAP, 3D Slicer, a collaborator's evaluation script) loading this
file alongside the raw CT would render the prediction in the wrong physical location,
at the wrong scale, and with the wrong shape.

### Why Metrics Were Unaffected

The Dice and HD95 metrics in `validate.py` are computed **in-memory** on decollated
tensors that remain in preprocessed space:

```python
self.dice_metric(y_pred=processed_preds, y=val_labels)
self.hd95_metric(y_pred=processed_preds, y=val_labels)
```

Both `processed_preds` and `val_labels` underwent the same transforms, so they are
perfectly aligned in preprocessed space. The saved NIfTI files play no role in metric
calculation.

## Solution: `Invertd`

MONAI's `Invertd` transform reverses all spatial transforms by reading the
**transform trace** stored in the `MetaTensor`'s `applied_operations` attribute.

### How the Transform Trace Works

When `LoadImaged` reads a NIfTI file, it returns a `MetaTensor` — a tensor with an
attached `.meta` dictionary. Each subsequent spatial transform appends a record to the
`applied_operations` list, storing:

- The transform class name (e.g., `"CropForegroundd"`)
- The parameters used (e.g., the crop bounding box, the padding amounts)
- The metadata needed for reversal (e.g., the original spacing, the original shape)

`Invertd` reads this trace from the `orig_keys` tensor (the image) and applies the
**mathematical inverse** of each spatial transform, in reverse order, to the `keys`
tensor (the prediction).

### Inversion Sequence

| Order | Inverse Applied | What It Undoes |
| :---: | :--- | :--- |
| 1 | `SpatialPadd` inverse | Removes the zero-padding |
| 2 | `CropForegroundd` inverse | Places the cropped region back into the full-volume bounding box (zeros elsewhere) |
| 3 | `Spacingd` inverse | Resamples from 1.0 mm isotropic back to the original voxel spacing |
| 4 | `Orientationd` inverse | Restores the original axis order |

Non-spatial transforms (`ScaleIntensityRanged`, `EnsureTyped`, `LoadImaged`) are
skipped because they do not alter the geometry of the volume.

## Guard: Limited Environments

In limited environments (`config.is_limited_env() == True`), `get_validation_transforms`
injects `RandCropByPosNegLabeld` into the pipeline. This random crop:

- Produces small patches rather than full volumes
- Generates multiple samples per volume (`num_samples`)
- Creates transform traces that cannot be meaningfully inverted back to the original
  volume space

A hard stop prevents silent production of geometrically incorrect files:

```python
if config.is_limited_env():
    raise RuntimeError(
        "Full-volume inference with Invertd is not supported in limited "
        "environments. The validation transforms include "
        "RandCropByPosNegLabeld, which prevents correct inversion of "
        "predictions back to the original image space. "
        "Run this script on a GPU environment (is_limited_env() == False)."
    )
```

## Result

After this change, each saved `{case_name}_pred.nii.gz` file:

- Has the **same shape** as the raw CT (e.g., `512×512×237`)
- Has the **same affine matrix** as the raw CT (scanner's native spacing, origin, and orientation)
- Can be loaded alongside the raw CT in any external tool with correct spatial alignment
- Is suitable for external evaluation, challenge submission, or clinical review