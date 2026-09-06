"""
verify_z_axis.py

Confirms, against a real sample from the LiTS pipeline, whether the deterministic
preprocessing output has the spatial convention required by the 2.5D Mamba-hybrid
pipeline.

Required convention:

    unbatched tensor: (C, X, Y, Z)
    batched tensor:   (B, C, X, Y, Z)
    z-axis:           last spatial axis

For Orientationd(axcodes="LAS"):

    X = left-right
    Y = posterior-anterior
    Z = inferior-superior / slice direction

This script prints PASS/FAIL for each check and exits with:

    0 if all checks pass
    1 if any check fails

Run this when:

- changing Orientationd.
- changing Spacingd.
- changing CropForegroundd/SpatialPadd.
- changing PIN_ENV or move between local/cloud.
- begin Mamba model implementation.
- modify the deterministic transform order.

Run in the real environment with LiTS environment variables set:

    source ~/denv/bin/activate
    python verify_z_axis.py
"""

import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose

from idssp.sonk import config
from idssp.sonk.disk.loader import DataCollector
from idssp.sonk.model.transforms import get_deterministic_transforms


def _as_4x4_affine(affine_source) -> np.ndarray | None:
    """
    Convert an affine stored as Tensor/ndarray into a CPU float64 ndarray
    with shape (4, 4). Returns None if the affine cannot be normalised.
    """
    if affine_source is None:
        return None

    if isinstance(affine_source, torch.Tensor):
        affine_source = affine_source.detach().cpu()

    try:
        affine = np.asarray(affine_source, dtype=np.float64)
    except Exception:
        return None

    # Handle possible batched affine, e.g. (1, 4, 4).
    if affine.ndim == 3:
        if affine.shape[0] != 1:
            return None
        affine = affine[0]

    if affine.shape != (4, 4):
        return None

    return affine


def _dominant_axis_from_affine(affine: np.ndarray, physical_row: int) -> int:
    """
    Given a 4x4 affine and a physical-axis row index:

        0 = L/R
        1 = A/P
        2 = S/I

    return the array axis (0, 1, or 2) that contributes most to that physical
    direction.
    """
    return int(np.argmax(np.abs(affine[physical_row, :3])))


def main() -> int:
    cfg = config.init()

    checks: list[tuple[str, bool, str]] = []

    def record(name: str, passed: bool, details: str = "") -> bool:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        if details:
            print(f"       {details}")
        checks.append((name, passed, details))
        return passed

    def finish() -> int:
        failed = [check for check in checks if not check[1]]

        print("\n" + "=" * 80)
        if failed:
            print("RESULT: FAILED")
            print(
                "The deterministic pipeline output is NOT compatible with the "
                "Mamba z-axis convention."
            )
            print("=" * 80)
            for name, _, details in failed:
                print(f"  - {name}")
                if details:
                    print(f"    {details}")
            return 1

        print("RESULT: PASSED")
        print(
            "The deterministic pipeline output is compatible with the Mamba "
            "z-axis convention."
        )
        print("=" * 80)
        return 0

    print("=" * 80)
    print("Loading one real LiTS sample to verify Mamba axis requirements...")
    print("=" * 80)

    collector = DataCollector()
    collector.read_dir(cfg.CT_ROOT, ds_source="LiTS")
    collector.extract_images_and_labels()

    if not collector.datasources:
        print("RESULT: FAILED")
        print("No paired image/label files found. Check LITS_CT_ROOT.")
        return 1

    sample_pair = collector.datasources[0]
    print(f"Sample: {sample_pair['image']}")

    transform = Compose(get_deterministic_transforms(cfg))
    data = transform(
        {
            "image": sample_pair["image"],
            "label": sample_pair["label"],
        }
    )

    image = data["image"]
    label = data["label"]

    # ------------------------------------------------------------------
    # Basic tensor-shape checks
    # ------------------------------------------------------------------
    record(
        "Image tensor is 4D (C, X, Y, Z)",
        image.ndim == 4,
        f"shape={tuple(image.shape)}",
    )

    if image.ndim != 4:
        return finish()

    record(
        "Image has exactly one input channel",
        image.shape[0] == 1,
        f"C={image.shape[0]}",
    )

    spatial_shape = tuple(image.shape[1:])
    patch_size = tuple(cfg.TRAIN_PATCH_SIZE)

    spatial_ok = (
        len(spatial_shape) == 3
        and all(spatial_shape[i] >= patch_size[i] for i in range(3))
    )

    record(
        "Spatial shape is 3D and at least TRAIN_PATCH_SIZE",
        spatial_ok,
        f"spatial_shape={spatial_shape}, required_min={patch_size}",
    )

    # ------------------------------------------------------------------
    # Affine checks
    # ------------------------------------------------------------------
    affine = _as_4x4_affine(getattr(image, "affine", None))

    if affine is None:
        record(
            "Image affine is a valid 4x4 matrix",
            False,
            "Could not read a 4x4 affine from the transformed image.",
        )
        return finish()

    record(
        "Image affine is a valid 4x4 matrix",
        True,
        f"shape={affine.shape}",
    )

    physical_labels = ["L/R", "A/P", "S/I (z)"]
    detected_axes: dict[str, int] = {}

    for physical_row, physical_label in enumerate(physical_labels):
        detected_axes[physical_label] = _dominant_axis_from_affine(
            affine,
            physical_row,
        )

    record(
        "Affine dominant axes are unique",
        len(set(detected_axes.values())) == 3,
        f"detected_axes={detected_axes}",
    )

    z_axis_affine = detected_axes["S/I (z)"]
    record(
        "Affine places z on array axis 2",
        z_axis_affine == 2,
        f"detected_z_axis={z_axis_affine}",
    )

    # ------------------------------------------------------------------
    # nibabel cross-check
    # ------------------------------------------------------------------
    nib_codes = tuple(nib.aff2axcodes(affine))

    record(
        "Orientation codes are ('L', 'A', 'S')",
        nib_codes == ("L", "A", "S"),
        f"nib_codes={nib_codes}",
    )

    if "S" in nib_codes:
        z_axis_nib = nib_codes.index("S")
    elif "I" in nib_codes:
        z_axis_nib = nib_codes.index("I")
    else:
        z_axis_nib = -1

    record(
        "nibabel places z on array axis 2",
        z_axis_nib == 2,
        f"nib_codes={nib_codes}, z_axis={z_axis_nib}",
    )

    # ------------------------------------------------------------------
    # Label alignment check
    # ------------------------------------------------------------------
    label_affine = _as_4x4_affine(getattr(label, "affine", None))

    if label_affine is None:
        record(
            "Label affine is a valid 4x4 matrix",
            False,
            "Could not read a 4x4 affine from the transformed label.",
        )
    else:
        record(
            "Label affine is a valid 4x4 matrix",
            True,
            f"shape={label_affine.shape}",
        )

        record(
            "Label affine matches image affine",
            np.allclose(affine, label_affine, atol=1e-4),
            "Image and label remain spatially aligned after deterministic transforms.",
        )

    print("\nRequired Mamba convention:")
    print("  unbatched tensor: (C, X, Y, Z)")
    print("  batched tensor:   (B, C, X, Y, Z)")
    print("  z axis:           last spatial axis (unbatched dim 3, batched dim 4)")

    return finish()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n" + "=" * 80)
        print("RESULT: FAILED")
        print(f"Unexpected error during verification: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
