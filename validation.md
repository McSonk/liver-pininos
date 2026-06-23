# Test-Time Validation and Evaluation Pipeline

This document details the test-time evaluation workflow for the automated hepatocellular carcinoma (HCC) tumour segmentation pipeline. It outlines the inference procedure, the post-processing heuristics, the generated output files, and the methodological rationale behind the evaluation design.

## 1. Validation Workflow

The test-time evaluation is executed via the `do_test.py` entry point, which orchestrates the loading of a trained checkpoint and the execution of full-volume inference using the `TestEvaluator` class (`validate.py`). 

The workflow proceeds as follows:
1. **Environment and Checkpoint Verification**: The script verifies the runtime environment and ensures that the checkpoint's configuration snapshot (e.g., `NUM_CLASSES`, `ISO_SPACING`, `HU_WINDOW_MIN`, `HU_WINDOW_MAX`) aligns with the current inference configuration to prevent preprocessing mismatches.
2. **Data Loading and Preprocessing**: The test volumes are loaded and subjected to the exact same deterministic preprocessing pipeline used during training (e.g., Hounsfield Unit windowing, isotropic resampling, and spatial cropping/padding).
3. **Full-Volume Sliding Window Inference**: To process full 3D CT volumes within GPU memory constraints, a `SlidingWindowInferer` is utilised. It extracts overlapping patches (50% overlap) of size `TRAIN_PATCH_SIZE`, processes them through the network, and blends the predictions using a Gaussian weighting function to minimise border artifacts. A fallback inferer with a reduced batch size is automatically triggered if a CUDA Out-Of-Memory (OOM) error occurs.
4. **Post-Processing and Activation**: The raw network logits are passed through a softmax activation to obtain class probabilities, followed by an `argmax` operation to generate discrete class indices. 
5. **Metric Computation**: The predictions and ground truth labels are decollated and converted into one-hot encoded tensors. The Dice similarity coefficient and the 95th percentile Hausdorff distance (HD95) are computed per case, excluding the background class.
6. **Export**: The predicted segmentation maps are saved as NIfTI files, and the computed metrics are aggregated into structured reports.

### Key Evaluation Metrics and Concepts

To ensure the evaluation methodology is transparent and reproducible, several core concepts and metrics utilised in the validation workflow are defined below.

#### Dice Similarity Coefficient (DSC)
The Dice coefficient measures the volumetric overlap between the predicted segmentation mask and the ground truth. It is calculated as twice the intersection of the two masks divided by the sum of their total voxels. Values range from 0 (no overlap) to 1 (perfect overlap). While DSC is excellent for assessing overall volumetric agreement, it is relatively insensitive to small boundary deviations or isolated false positive voxels located far from the main organ.

#### 95th Percentile Hausdorff Distance (HD95)
The standard Hausdorff Distance (HD) measures the absolute maximum distance between a point on the predicted boundary and its nearest point on the ground truth boundary. While mathematically rigorous, standard HD is extremely sensitive to outliers; a single stray predicted voxel located far from the true organ can result in a catastrophically high score (e.g., hundreds of millimetres), which misrepresents the overall segmentation quality.

To mitigate this, the 95th percentile Hausdorff Distance (HD95) is utilised. Instead of the absolute maximum distance, HD95 calculates the 95th percentile of all pairwise boundary distances. This effectively ignores the top 5% of extreme outlier distances—often caused by sliding window artifacts or minor fragmented predictions—while still providing a rigorous assessment of boundary alignment. Because the volumes are resampled to an isotropic spacing during preprocessing, this distance is accurately measured in physical space using SI units (millimetres, mm). In this metric, **lower is better**.

#### Sliding Window Inference
Medical CT volumes are typically too large to be processed in a single forward pass due to GPU memory constraints. Sliding window inference addresses this by dividing the full 3D volume into smaller, overlapping patches (defined by the `TRAIN_PATCH_SIZE` configuration). The model processes each patch independently, and the predictions are blended back together using a Gaussian weighting function. This ensures that the model can maintain high-resolution contextual awareness while operating within strict VRAM limits, smoothly transitioning between overlapping regions to prevent border artifacts.

#### Logits to Class Maps (Softmax and Argmax)
The neural network outputs raw, unnormalised predictions known as logits. To convert these into interpretable segmentation masks, a softmax activation function is applied to generate a probability distribution across all defined classes (e.g., background, liver, tumour) for each voxel. Subsequently, an `argmax` operation selects the class with the highest probability, yielding a discrete, single-channel class map where each voxel is assigned an integer label corresponding to its predicted anatomical structure.

## 2. Generated Files

The evaluation script generates the following outputs in the designated run directory:

| File / Directory | Description |
| :--- | :--- |
| `test_predictions/<case_name>_pred.nii.gz` | The predicted 3D segmentation maps for each test volume, saved in the NIfTI format with the original affine matrix for spatial alignment. |
| `reports/test_evaluation_results.csv` | A granular, per-case report containing the Dice and HD95 scores for the liver and tumour for every individual test volume. |
| `reports/test_aggregated_metrics.csv` | A summary report containing the aggregated statistics (mean ± standard deviation) for each anatomical structure across the entire test cohort. |

### Structure of `test_evaluation_results.csv`

The `test_evaluation_results.csv` file provides a case-by-case breakdown of the model's performance. Each row corresponds to a single test volume, identified by its filename stem. The exact columns present in the file depend on the segmentation configuration (multi-class vs. binary). 

Missing metric values (e.g., when a structure is entirely absent in either the ground truth or the prediction, resulting in an undefined metric) are recorded as empty cells.

**Multi-Class Configuration (3 Classes: Background, Liver, Tumour)**

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `case_name` | String | The unique identifier of the test volume, extracted from the original NIfTI filename. |
| `dice_liver` | Float | The Dice similarity coefficient for the liver segmentation. |
| `dice_tumour` | Float | The Dice similarity coefficient for the tumour segmentation. |
| `hd95_liver_mm` | Float | The 95th percentile Hausdorff distance for the liver boundary, measured in millimetres (mm). |
| `hd95_tumour_mm` | Float | The 95th percentile Hausdorff distance for the tumour boundary, measured in millimetres (mm). |

## 3. Post-Processing Steps

When the `--post-process` (or `-pp`) flag is enabled in `do_test.py`, an additional heuristic post-processing step is applied to the discrete class maps before metric computation. This step enforces strict anatomical priors:

1. **Largest Connected Component (LCC) Extraction**: The predicted liver mask (class 1) is isolated. A 3D connected component analysis (using 6-connectivity) is performed, and only the largest connected component is retained. All other disconnected liver voxels are discarded.
2. **Stray Voxel Removal**: Any isolated liver predictions that do not belong to the main anatomical structure are removed.
3. **Anatomical Constraint for Tumours**: Since HCC tumours are intrahepatic (occurring strictly within the liver), any predicted tumour voxels (class 2) that do not intersect with the retained main liver component are discarded.
4. **Fragmentation Warning**: If the LCC filtering discards more than 50% of the originally predicted liver voxels, a warning is logged, as this indicates highly fragmented predictions or atypical anatomy.

## 4. Raw vs. Post-Processed Results

The application of post-processing fundamentally alters the characteristics of the output segmentations and the resulting evaluation metrics.

| Characteristic | Raw Predictions | Post-Processed Predictions |
| :--- | :--- | :--- |
| **Methodology** | Direct output of the neural network (Softmax + Argmax). | Neural network output + heuristic anatomical constraints (LCC). |
| **Liver Morphology** | May contain fragmented islands or disconnected false positive clusters far from the true liver. | Guaranteed to be a single, contiguous 3D structure. |
| **Tumour Location** | May predict tumours outside the liver boundary (anatomically implausible). | Strictly constrained to exist only within the main liver mask. |
| **Impact on Dice** | Reflects the pure voxel-wise overlap learned by the network. | Typically shows a modest improvement or remains stable, as small false positive islands have a negligible effect on the Dice denominator. |
| **Impact on HD95** | Highly sensitive to stray voxels. A single false positive island 200 mm away from the liver will catastrophically inflate the HD95 score. | Drastically reduces HD95 by eliminating the distant outlier voxels that drive the 95th percentile distance. |

## 5. Rationale and Scientific Validity

### Why Post-Processing was Included
During initial evaluations, a significant performance gap was observed between the validation and test sets, characterised by extreme per-case variance and catastrophically high HD95 scores (e.g., >200 mm). Investigation revealed that the sliding window inference mechanism occasionally produced disconnected false positive islands in regions containing bowel or abdominal wall tissue. While these stray voxels barely affected the Dice score, they severely inflated the HD95 metric. The post-processing step was introduced to mitigate these sliding window artifacts and enforce the anatomical reality that the liver is a single connected organ and that HCC tumours are intrahepatic.

### Scientific Validity and Comparability
Applying heuristic post-processing shifts the evaluated system from a pure end-to-end deep learning model to a hybrid pipeline (neural network + classical computer vision algorithms). 

From a scientific standpoint, enforcing anatomical priors is a standard, valid, and widely accepted practice in medical image segmentation. It reflects how a model would actually be deployed in a clinical setting, where anatomical impossibilities would be rejected. 

However, to maintain scientific rigour and ensure fair comparability against other pure end-to-end architectures (such as the baseline U-Net or SegResNet comparisons), **both raw and post-processed metrics must be reported**. 
- The **raw metrics** isolate the pure representational and learning capacity of the neural network architecture itself.
- The **post-processed metrics** demonstrate the practical, clinical utility of the complete pipeline.

Reporting both allows reviewers and researchers to accurately disentangle the performance gains derived from the deep learning architecture versus those derived from the heuristic post-processing rules.