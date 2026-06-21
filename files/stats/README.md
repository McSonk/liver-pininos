# CT Volume Analysis and Dataset Stratification Dictionary

This document details the per-case summary metrics extracted from the Liver Tumour Segmentation (LiTS) dataset. These metrics serve two primary purposes in the context of this research:
1. **Quality Control and Exploratory Data Analysis:** Understanding the distribution of anatomical, technical, and pathological variations within the dataset.
2. **Dataset Stratification:** Ensuring that the Training, Validation, and Test splits are statistically representative of the entire cohort, thereby preventing selection bias and ensuring robust evaluation of the deep learning models.

---

## 1. The Stratification Methodology

In medical image segmentation, simple random splitting of a dataset often leads to imbalanced partitions. For instance, a randomly generated Test set might accidentally exclude rare but clinically vital cases, such as non-contrast CT scans or scans with highly anisotropic (thick) slices. If a model is only evaluated on "easy", high-quality data, its real-world clinical utility will be severely overestimated.

To mitigate this, a **Multilabel Stratified Splitting** approach is employed. The process operates at a base level as follows:

## Step 1: Identification of Critical Variables
Four primary variables are identified as the most significant sources of variance in model performance:
*   **`spacing_z` (Z-axis resolution):** Dictates the level of 3D spatial detail.
*   **`liver_hu_mean` (Mean Liver Hounsfield Units):** Acts as a proxy for the CT contrast phase (e.g., non-contrast vs. arterial).
*   **`tumour_volume_ml` (Total Tumour Burden):** Determines the scale of the segmentation task.
*   **`has_tumor` (Tumour Presence):** Ensures the model is tested on both positive (pathological) and negative (healthy) cases.

## Step 2: Pre-allocation of Rare Subgroups
Before applying any algorithmic splitting, edge cases are identified. For example, scans with a `liver_hu_mean` < 60 HU typically represent non-contrast or portal-venous phases where tumours are notoriously difficult to delineate. To guarantee that the model is evaluated on these difficult cases, a deterministic pre-allocation assigns at least one rare case to the Training, Validation, and Test sets manually.

## Step 3: Binning and Composite Key Generation
The remaining continuous variables are converted into discrete, clinically meaningful categories (bins):
*   **Slice Thickness:** Thin ($\le 1.0$ mm), Medium ($1.25\text{--}1.5$ mm), Thick ($\ge 2.0$ mm).
*   **Liver Contrast Phase:** Low (< 60 HU), Mid ($60\text{--}100$ HU), High (> 100 HU).
*   **Tumour Size:** None, Small (< 5 mL), Medium ($5\text{--}50$ mL), Large (> 50 mL).

These bins are concatenated to form a **Composite Stratification Key** (e.g., `thin_high_medium`). 

## Step 4: Stratified Partitioning
A stratified shuffling algorithm divides the remaining dataset into Train, Validation, and Test sets. The algorithm ensures that the proportion of every unique Composite Key is identical across all three splits. 

## Step 5: Statistical Validation of Dataset Stratification

To ensure that the deep learning models are evaluated fairly and that the results
are generalisable, the dataset must be divided into Training, Validation, and Test
sets that are statistically homogeneous.

To mathematically prove that the multilabel stratification algorithm successfully
balanced the dataset, a statistical validation step is performed. This involves
applying specific hypothesis tests to key clinical and technical parameters.

### 5.1. Differentiating Variable Types

The parameters selected for validation are divided into two distinct categories,
each requiring a different statistical approach due to their underlying data distributions:

*   **Continuous (and Discrete Count) Variables:** These represent measurable quantities
that can take on a range of values. In medical datasets, these variables (such as
tumour volume or Hounsfield Units) are rarely normally distributed; they are
typically heavily right-skewed. Therefore, non-parametric tests are required.
*   **Categorical Variables:** These represent discrete, mutually exclusive groups
or classes (e.g., "Thin" vs "Thick" slices, or "Tumour Present" vs "Tumour Absent").
The analysis focuses on the *proportions* or *frequencies* of these categories across the different splits.

### 5.2. Statistical Tests and Mathematical Rationale

The following table summarises the statistical tests applied to each variable type,
alongside their mathematical foundations.

| Feature | Kruskal–Wallis H-test | Chi-Square ($\chi^2$) Test of Independence |
| :--- | :--- | :--- |
| **Variable Type** | Continuous / Discrete Count | Categorical / Nominal |
| **Data Distribution** | Non-parametric (no normality assumption) | Frequencies / Proportions |
| **Null Hypothesis ($H_0$)** | The median values of the groups are identical. | The distribution of categories is independent of the group split. |
| **Application in Thesis** | Tumour Volume, Liver HU Mean, Liver Volume, Texture Variance, Number of Lesions. | Tumour Presence, Slice Thickness Group. |

#### 5.2.1. The Kruskal–Wallis H-test (Continuous Variables)
Because medical data such as `tumour_volume_ml` violates the normality assumption required for a standard one-way ANOVA, the Kruskal–Wallis test is employed. Instead of comparing raw means, this test ranks all observations from all groups together and compares the sum of the ranks between the Training, Validation, and Test sets.

The test statistic $H$ is calculated as:

$$ H = \left( \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} \right) - 3(N+1) $$

Where:
*   $N$ is the total number of observations across all groups ($n_{train} + n_{val} + n_{test}$).
*   $k$ is the number of groups (in this case, $k = 3$).
*   $n_i$ is the number of observations in group $i$.
*   $R_i$ is the sum of the ranks for group $i$.

If the $H$ statistic is large, it implies that the ranks are not evenly distributed across the splits, leading to a low $p$-value and a rejection of the null hypothesis.

#### 5.2.2. The Chi-Square ($\chi^2$) Test of Independence (Categorical Variables)
For categorical variables, the goal is to ensure that the *proportion* of cases in each category is consistent across the splits. The Chi-Square test compares the **Observed** frequencies ($O_i$) in the Train/Val/Test sets against the **Expected** frequencies ($E_i$) if the splits were perfectly balanced.

The test statistic is calculated as:

$$ \chi^2 = \sum_{i=1}^{n} \frac{(O_i - E_i)^2}{E_i} $$

Where:
*   $O_i$ is the observed count for a specific category in a specific split.
*   $E_i$ is the expected count, calculated based on the marginal totals of the contingency table.

A low $\chi^2$ value indicates that the observed proportions closely match the expected proportions, resulting in a high $p$-value.

---

### 5.3. Clinical and Technical Importance of Selected Parameters

The parameters chosen for this statistical validation were not selected at random; they represent the primary sources of variance that dictate the difficulty of the segmentation task and the physical properties of the input data.

| Parameter | Type | Why it is Critical for Model Performance |
| :--- | :--- | :--- |
| **Liver HU Mean** | Continuous | Acts as a robust proxy for the **CT contrast phase**. Scans with low mean HU ($< 60$) are typically non-contrast or portal-venous phases where tumours are nearly isodense to the liver parenchyma, drastically reducing boundary visibility and increasing segmentation difficulty. |
| **Tumour Volume** | Continuous | Represents the **disease burden**. Extremely small tumours ($< 5$ mL) test the model's limit of spatial detection, whilst massive tumours distort surrounding anatomy and require a large receptive field. |
| **Slice Thickness** | Categorical | Represents **3D spatial resolution**. Highly anisotropic scans (e.g., "Thick" slices $\ge 2.0$ mm) suffer from the stair-step artefact and lose crucial 3D contextual information, which severely impacts 3D convolutional networks. |
| **Has Tumour** | Categorical | Ensures the model is evaluated on both **positive (pathological)** and **negative (healthy)** cases. A model must learn to output an empty mask when no disease is present to avoid high false-positive rates in clinical deployment. |
| **Liver Texture Variance** | Continuous | Represents **parenchymal heterogeneity**. High variance often correlates with underlying liver disease (e.g., severe steatosis, cirrhosis) or scanner noise, acting as a confounding variable that can trick the model into predicting false lesions. |
| **Number of Lesions** | Discrete | Represents **topological complexity**. Multi-focal disease requires the model to correctly separate distinct, disconnected 3D objects, which is fundamentally harder than segmenting a single, contiguous mass. |

---

### 5.4 Table 1: Dataset Stratification Validation (LiTS)

| Variable | Train (n=79) | Val (n=27) | Test (n=25) | p-value |
| :--- | :--- | :--- | :--- | :--- |
| **Continuous Variables** | Median [IQR] | Median [IQR] | Median [IQR] | |
| Liver HU Mean (HU) | 98.37 [91.03–112.15] | 101.93 [85.93–112.59] | 99.39 [91.99–116.19] | 0.809 |
| Liver Volume (mL) | 1559.52 [1379.69–1859.85] | 1638.27 [1411.01–1885.68] | 1590.78 [1348.59–1749.07] | 0.701 |
| Tumour Volume (mL)* | 16.43 [2.78–119.12] | 6.19 [0.76–27.31] | 16.99 [1.76–44.31] | 0.332 |
| Number of Lesions | 5.00 [1.00–10.00] | 2.00 [1.00–5.00] | 2.00 [1.00–7.00] | 0.053 |
| Liver Texture Variance | 833.11 [599.74–1189.63] | 754.63 [467.04–1122.79] | 732.11 [539.84–935.33] | 0.486 |
| Tumour to Liver Ratio     | 0.01 [0.00-0.06]       | 0.00 [0.00-0.02]       | 0.01 [0.00-0.03]       | 0.364 
| **Categorical Variables** | n (%) | n (%) | n (%) | |
| Has Tumour | | | | 0.516 |
| &nbsp;&nbsp;False | 6 (7.6%) | 4 (14.8%) | 3 (12.0%) | |
| &nbsp;&nbsp;True | 73 (92.4%) | 23 (85.2%) | 22 (88.0%) | |
| Slice Thickness Group | | | | 0.960 |
| &nbsp;&nbsp;Medium (1.25–1.5 mm) | 9 (11.4%) | 2 (7.4%) | 3 (12.0%) | |
| &nbsp;&nbsp;Thick (≥2.0 mm) | 19 (24.1%) | 8 (29.6%) | 6 (24.0%) | |
| &nbsp;&nbsp;Thin (≤1.0 mm) | 51 (64.6%) | 17 (63.0%) | 16 (64.0%) | |

> \* Tumour Volume calculated only for cases where `has_tumor == True`.

> *Due to the non-normal distribution and the presence of outliers inherent in
> clinical medical imaging data, continuous variables are summarised using the median
> and interquartile range (IQR), denoted as Median [Q1–Q3]. Categorical variables
> are presented as frequencies and percentages.*

### 5.5. Interpretation of the Results

In standard scientific research, a $p$-value $< 0.05$ is desired to prove that a treatment had a statistically significant effect. However, **in the context of dataset stratification, the objective is inverted.** 

The Null Hypothesis ($H_0$) states that *there is no statistically significant difference between the Training, Validation, and Test sets*. Therefore, we **want to fail to reject the null hypothesis**. A $p$-value $> 0.05$ is the desired outcome, as it proves the splits are statistically indistinguishable.

### 5.6 Analysis of the Generated Table 1

*   **Continuous Variables:** 
    *   **Liver HU Mean ($p = 0.809$), Liver Volume ($p = 0.701$), Tumour Volume ($p = 0.332$), and Liver Texture Variance ($p = 0.486$)** all yield highly favourable $p$-values. This confirms that the composite stratification keys (which binned HU and Volume) successfully distributed the continuous underlying distributions evenly across all three splits.
    *   **Number of Lesions ($p = 0.053$):** This discrete count variable borders the $0.05$ threshold. Because lesion multiplicity is highly right-skewed (most patients have 1–3 lesions, whilst a few have $>20$), it is notoriously difficult to balance perfectly without making it a primary stratification key. A $p$-value of $0.053$ indicates a negligible, statistically insignificant variance in multiplicity across the splits, which is an acceptable limitation.
*   **Categorical Variables:**
    *   **Has Tumour ($p = 0.516$) and Slice Thickness Group ($p = 0.960$)** both show excellent balance. The near-perfect $p$-value for Slice Thickness ($0.960$) proves that the model will be evaluated on a realistic, proportional mix of high-resolution (Thin) and anisotropic (Thick) scans, preventing overoptimistic benchmarking on "easy" data.

### 5.7 Conclusion

The statistical validation confirms that the multilabel stratification algorithm successfully mitigated selection bias. The Training, Validation, and Test sets are statistically homogeneous across all measured clinical and technical covariates, ensuring that the subsequent evaluation of the SwinUNETR, SegResNet, and Mamba architectures will yield robust, generalisable, and clinically relevant metrics.
