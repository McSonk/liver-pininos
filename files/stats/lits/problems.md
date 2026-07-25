# NIfTI Header Anomalies in the LiTS Dataset

During the post-training analysis of the model's performance on the LiTS test set, significant metadata inconsistencies were identified within the dataset. These anomalies were discovered while investigating volumes with unexpectedly low Dice scores.

## 1. Severe Affine Mismatches

For a specific subset of volumes, the physical coordinate matrices of the image and the label are entirely disconnected. Because MONAI's spatial transforms (e.g., `Spacingd`, `CropForegroundd`) rely on these physical matrices to resample and crop the data, this discrepancy causes the ground truth masks to be mapped to incorrect physical locations relative to the CT anatomy. 

Consequently, this results in a complete spatial misalignment during preprocessing, yielding a Dice score of 0 for the affected volumes regardless of the model's actual predictive performance.

The affected volumes are:

| Volume ID | Dataset Split |
| :--- | :--- |
| `volume-48` | Training |
| `volume-49` | Training |
| `volume-50` | Validation |
| `volume-51` | Validation |
| `volume-52` | Test |

*Note: For the final quantitative evaluation, these volumes must either be excluded from the test set or corrected using an affine alignment transform.*

### Evidence: Volume 52 Affine Mismatch

To illustrate the severity of this issue, consider the NIfTI headers for `volume-52`:

**Image (CT Scan) Affine:**
```text
[[ -0.836   0.      0.    208.9  ]  <- X spacing: 0.83mm, Origin: 208.9mm
 [  0.      0.836   0.   -213.2  ]  <- Y spacing: 0.83mm, Origin: -213.2mm
 [  0.      0.      2.5  -632.8  ]  <- Z spacing: 2.50mm, Origin: -632.8mm
 [  0.      0.      0.      1.   ]]
```

**Label (Segmentation Mask) Affine:**
```text
[[ 1.  0.  0.  1. ]  <- X spacing: 1.00mm, Origin: 1.0mm
 [ 0.  1.  0.  1. ]  <- Y spacing: 1.00mm, Origin: 1.0mm
 [ 0.  0.  1.  1. ]  <- Z spacing: 1.00mm, Origin: 1.0mm
 [ 0.  0.  0.  1. ]]
```

**Analysis:**
The image affine reflects the actual scanner geometry (anisotropic spacing of $0.83 \times 0.83 \times 2.5$ mm and a specific physical origin in the scanner's coordinate system). Conversely, the label affine is a **default identity matrix** with 1.0 mm isotropic spacing and an arbitrary origin at $(1, 1, 1)$. 

Because MONAI's spatial transforms (`Spacingd`, `CropForegroundd`) rely on these matrices to map physical coordinates, the preprocessing pipeline interprets the image and label as existing in two completely different physical spaces, resulting in a total spatial misalignment.


## 2. Identity / Placeholder Affines

A second category of anomalies involves volumes where both the image and the label possess a default identity affine matrix (indicating 1.0 mm isotropic spacing and a zeroed origin), rather than the true scanner geometry. 

While technically incorrect, this anomaly is benign for the segmentation pipeline. Because the placeholder affine is applied identically to both the image and the mask, their relative spatial alignment is perfectly preserved. Therefore, no preprocessing-induced misalignment occurs, and the model can process these volumes normally.

The affected volumes are:

| Volume ID | Dataset Split |
| :--- | :--- |
| `volume-28` | Training |
| `volume-29` | Training |
| `volume-30` | Validation |
| `volume-31` | Training |
| `volume-32` | Validation |
| `volume-33` | Test |
| `volume-34` | Training |
| `volume-35` | Test |
| `volume-36` | Test |
| `volume-37` | Training |
| `volume-38` | Test |
| `volume-39` | Validation |
| `volume-40` | Training |
| `volume-41` | Test |
| `volume-42` | Training |
| `volume-43` | Training |
| `volume-44` | Validation |
| `volume-45` | Validation |
| `volume-46` | Validation |
| `volume-47` | Validation |