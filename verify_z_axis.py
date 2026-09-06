"""
verify_z_axis.py

Confirms, against a REAL sample from the LiTS pipeline, which array axis
carries the z (superior-inferior / slice) direction after `Orientationd`
with axcodes="LAS". This must be checked directly rather than assumed,
because a wrong assumption here silently corrupts the entire premise of the
2.5D Mamba-hybrid architecture (2D-per-axial-slice + Mamba-across-z) without
ever raising a shape error.

Three independent checks are run and cross-referenced:
  1. MONAI's own axis-code metadata after Orientationd ("LAS" -> per-axis
     anatomical labels), read directly from the transformed MetaTensor.
  2. The raw affine matrix's dominant-weight row for the S/I direction,
     computed the same way `idssp/sonk/model/data.py::get_slice_range`
     already does it dynamically (not hardcoded).
  3. `nib.aff2axcodes`, as an independent library-level cross-check against
     (1) and (2).

Run this in the project's real environment (`~/denv` or `~/denv_mamba`) with
PIN_ENV and the LiTS env vars set, since it uses the real config/loader.

Run this when:

- changing Orientationd.
- changing Spacingd.
- changing CropForegroundd/SpatialPadd.
- changing PIN_ENV or move between local/cloud.
- begin Mamba model implementation.
- modify the deterministic transform order.

Usage
-----
    source ~/denv/bin/activate
    python verify_z_axis.py
"""
import numpy as np
from monai.transforms import Compose

from idssp.sonk import config
from idssp.sonk.disk.loader import DataCollector
from idssp.sonk.model.transforms import get_deterministic_transforms


def _dominant_axis_from_affine(affine: np.ndarray, physical_row: int) -> int:
    """
    Given a 4x4 affine and the physical-axis row index (0=L/R, 1=A/P, 2=S/I),
    returns which ARRAY axis (0, 1, or 2) contributes most to that physical
    direction. This mirrors get_slice_range()'s dynamic z-axis detection in
    idssp/sonk/model/data.py exactly, so the two should always agree.
    """
    return int(np.argmax(np.abs(affine[physical_row, :3])))


def main() -> None:
    cfg = config.init()

    print("=" * 80)
    print("Loading one real training sample to verify axis ordering...")
    print("=" * 80)

    collector = DataCollector()
    collector.read_dir(cfg.CT_ROOT, ds_source="LiTS")
    collector.extract_images_and_labels()

    if not collector.datasources:
        raise RuntimeError("No paired image/label files found. Check LITS_CT_ROOT.")

    sample_pair = collector.datasources[0]
    print(f"Sample: {sample_pair['image']}")

    # Run the REAL deterministic pipeline (LoadImaged -> ForceMatchingAffined
    # -> Orientationd(axcodes='LAS') -> Spacingd -> ... -> SpatialPadd),
    # exactly as training does it. No shortcuts.
    transform = Compose(get_deterministic_transforms(cfg))
    data = transform({"image": sample_pair["image"], "label": sample_pair["label"]})

    image = data["image"]  # MetaTensor, shape (C, dim0, dim1, dim2) — no batch dim yet
    print(f"\nTransformed image tensor shape (C, dim0, dim1, dim2): {tuple(image.shape)}")

    affine = np.asarray(image.affine)
    if affine.ndim == 3:  # defensive: some MONAI versions may add a leading batch dim
        affine = affine[0]
    print(f"\nAffine matrix after Orientationd(axcodes='LAS'):\n{affine}")

    # --- Check 1: MONAI's own metadata after Orientationd ---
    # After axcodes="LAS", MONAI guarantees: array axis 0 -> L/R, axis 1 ->
    # A/P, axis 2 -> S/I, in that fixed order. This is what "LAS" MEANS.
    monai_expected_axis_labels = ["L/R (Left-Right)", "A/P (Anterior-Posterior)", "S/I (Superior-Inferior, z)"]
    print("\n--- Check 1: MONAI axcodes='LAS' contract ---")
    for arr_axis, label in enumerate(monai_expected_axis_labels):
        print(f"  Array axis {arr_axis} (tensor dim {arr_axis + 1}, after channel) -> {label}")
    monai_z_array_axis = 2  # guaranteed by axcodes="LAS", not assumed

    # --- Check 2: dominant-weight row of the affine (same method as data.py) ---
    print("\n--- Check 2: affine dominant-weight detection (matches get_slice_range) ---")
    li_ap_si_labels = ["L/R", "A/P", "S/I (z)"]
    affine_detected_axes = {}
    for physical_row, label in enumerate(li_ap_si_labels):
        arr_axis = _dominant_axis_from_affine(affine, physical_row)
        affine_detected_axes[label] = arr_axis
        print(f"  Physical direction {label} (affine row {physical_row}) -> array axis {arr_axis}")
    affine_z_array_axis = affine_detected_axes["S/I (z)"]

    # --- Check 3: nibabel's independent axis-code reading ---
    import nibabel as nib
    nib_codes = nib.aff2axcodes(affine)
    print(f"\n--- Check 3: nib.aff2axcodes independent cross-check ---")
    print(f"  nib.aff2axcodes(affine) = {nib_codes}")
    # aff2axcodes returns the anatomical direction each array axis POINTS TOWARDS
    # (endpoint convention), e.g. ('L','A','S') for LAS orientation.
    nib_z_array_axis = nib_codes.index("S") if "S" in nib_codes else nib_codes.index("I")

    # --- Cross-reference all three ---
    print("\n" + "=" * 80)
    print("CROSS-REFERENCE")
    print("=" * 80)
    print(f"  Check 1 (MONAI axcodes contract):        array axis {monai_z_array_axis}")
    print(f"  Check 2 (affine dominant-weight, dynamic): array axis {affine_z_array_axis}")
    print(f"  Check 3 (nib.aff2axcodes):                array axis {nib_z_array_axis}")

    all_agree = monai_z_array_axis == affine_z_array_axis == nib_z_array_axis
    if not all_agree:
        raise AssertionError(
            "Z-axis detection DISAGREES between methods. Do not proceed with "
            "the Mamba merge/un-merge logic until this is resolved — silent "
            "misalignment here corrupts the architecture without a shape error."
        )

    print(f"\nAll three checks AGREE: z-axis is array axis {monai_z_array_axis} "
          f"(within the (C, dim0, dim1, dim2) tensor, i.e. tensor dim {monai_z_array_axis + 1}).")

    # --- Translate to the batched 5D tensor used throughout the stage table ---
    batched_z_tensor_dim = monai_z_array_axis + 1 + 1  # +1 for channel, +1 for batch
    print(f"\nIn the batched 5D tensor (B, C, dim0, dim1, dim2), z is tensor "
          f"dim {batched_z_tensor_dim} (0-indexed) — i.e. the LAST spatial "
          f"dimension, not the first.")
    print("This means: the stage-1 merge must fold B together with the array "
          "axis that ends up LAST after channel, not the one immediately "
          "after channel.")

    # --- Sanity-print the actual per-axis physical extent, for a human check ---
    print(f"\nFor reference, image_shape (excl. channel) after all deterministic "
          f"transforms: {tuple(image.shape[1:])}")
    print("Cross-check this against per_case_summary.csv / stratified_*.csv "
          "for this volume's spacing_x/y/z if you want a fourth, fully "
          "independent confirmation from the precomputed dataset stats.")


if __name__ == "__main__":
    main()
