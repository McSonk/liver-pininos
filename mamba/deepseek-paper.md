Your architecture sits at a productive intersection: the efficiency of 2.5D slice-wise processing, the sequence modeling power of Mamba, and the clinical need for robust liver tumor segmentation. Below is a structured literature map, followed by concrete innovation directions with supporting papers.

---

## 1. Directly Relevant Papers to Your Architecture

### 1.1 2.5D + Mamba for Liver Tumor Segmentation (closest to your design)

**MemSAM-2.5D** is the single closest published match to your proposed architecture. It is a unified 2.5D framework built on MedSAM that integrates a **Hybrid Mamba-Adapter (HMA)** for intra-slice multi-scale representation and a **Z-axis State Flow (ZSF) module** for continuous inter-slice dependency modeling, plus a Confidence-Gated Prototype Memory for boundary refinement. It explicitly targets liver tumor segmentation and evaluates on MSD08, HCC-TACE-Seg, and WAW-TACE. The paper frames the exact three challenges your architecture addresses: extreme lesion-scale variation, volumetric discontinuity across slices, and ambiguous boundaries. Notably, it argues that independent slice-by-slice assessment leads to discontinuous predictions and abrupt area fluctuations along Z—the very problem your pooled-vector Mamba aggregation solves.

**LCMambaNet** is a 2D liver tumor segmentation network with a state-space model and liver-cancer-specific attention, published in Frontiers in Oncology. It is 2D rather than 2.5D, but its state-space design for liver tumors is directly comparable.

**Mamba-enhanced codebook learning** places a Mamba module at the bottleneck layer to establish long-range, cross-slice dependencies along the z-axis of volumetric data, using discrete token compression. The bottleneck placement mirrors your Stage 3–8 fusion point.

### 1.2 Core Mamba Segmentation Architectures (your design references)

**SegMamba** is the canonical 3D Mamba segmentation backbone. It processes whole-volume features at every scale with long-range sequential modeling, achieving 93.61% DSC on the BraTS WT benchmark versus 92.71% for SwinUNETR. SegMamba-V2 improves to 94.02%. Your 2.5D approach is computationally cheaper than full 3D Mamba but sacrifices the dense spatial scanning that SegMamba performs.

**U-Mamba** enhances long-range dependency for biomedical segmentation using a hybrid CNN–SSM design. It achieves 91.62% WT DSC, slightly below SegMamba but with a lighter architecture.

**MambaVesselNet++** is a Hybrid CNN-Mamba framework with a **Hi-Encoder** (texture-aware convolution layers for local features, then Mamba for long-range dependencies) and a **Bi-Decoder** with skip connections. Its encoder–decoder philosophy closely parallels your 2D CNN encoder + Mamba bottleneck + 2D decoder design.

**MAVMN (Mamba-Augmented Volumetric Memory Network)** uses selective state-space models for **content-gated axial recursion** to accumulate and query evidence across slices in linear time with a constant state. It supports matrixized pixel-level whole-volume readout and slice-wise propagation. This is conceptually adjacent to your pooled-vector approach, though MAVMN operates at pixel-level rather than pooled-vector level.

### 1.3 Mamba2 Variants (your planned upgrade path)

**U-Mamba2** integrates Mamba2 state space models into a U-Net architecture, enforcing stronger structural constraints for higher efficiency without compromising performance. It won the ToothFairy3 challenge with mean Dice 0.84 (Task 1) and 0.87 (Task 2). The key architectural insight is that Mamba2's stronger constraints on the hidden-space update matrix allow earlier projection layers, improving efficiency.

**Mamba-UNet++** uses a Visual State Space Duality (VSSD) block based on an improved Mamba2 VSS block, addressing limited receptive field and semantic gap issues.

### 1.4 Hybrid CNN–Mamba Fusion Papers

**DCM-Net** proposes a dual-branch CNN–Mamba cross-layer feature fusion network with an adaptive dual-branch feature cross-coupling module.

**CFM-UNet** couples CNN Bottle2neck blocks for local feature extraction with Mamba-based visual state space blocks for global feature extraction.

**HCMUNet** synergizes CNN with Mamba in the encoder phase, with a Multi-scale Hybrid Attention Fusion module.

**ProMamba** (Imperial College) uses a dual-branch Pro-SSM block with depthwise dilated convolution for multi-scale boundary cues alongside a Mamba-based VSS branch for global context.

### 1.5 Surveys (essential for your thesis literature review)

**"A Comprehensive Survey of Mamba Architectures for Medical Image Analysis"** (Computer Methods and Programs in Biomedicine, 2025) covers classification, segmentation, restoration, and beyond. It includes a detailed comparison table of Mamba variants across BraTS metrics.

**"A comprehensive survey and taxonomy of Mamba"** (Elsevier, 2025) provides the first comprehensive taxonomy of Mamba across vision, medical, RL, and generative tasks, with identification of research gaps.

**"Efficient Medical Image Segmentation in Multisensor Imaging: A Survey in the Era of Mamba and Foundation Models"** (MDPI Sensors, 2026) directly compares SegMamba, SegMamba-V2, HybridMamba, and U-Mamba with detailed DSC/HD95 tables.

---

## 2. Papers Best Suited to Your Implementation Choices

### 2.1 For Your Pooled-Vector Aggregation Strategy

Your decision to global-average-pool each slice's spatial feature map into a single vector before Mamba is unusual and arguably suboptimal. Most Mamba segmentation papers scan **spatial token sequences** (e.g., flattened patches or multi-directional scans), not pooled vectors. The relevant comparison papers are:

| Your choice | What the literature does | Key paper |
|---|---|---|
| Global average pool → 1 vector per slice | Flatten spatial feature maps into sequences (e.g., 8×8=64 tokens per slice) | SegMamba scans full 3D volumes with tri-orientated Mamba |
| 128-step sequence (one per slice) | Hundreds to thousands of tokens per volume | SegMamba-V2 processes whole-volume features at every scale |
| Mamba at bottleneck only | Mamba at multiple scales | MambaVesselNet++ uses Mamba throughout the encoder |

The **MemSAM-2.5D Z-axis State Flow module** is the closest published analog to your pooled-vector approach, but even it uses a richer inter-slice representation than a single pooled vector per slice.

### 2.2 For Your Broadcast + Concat + 1×1 Conv Fusion

**ProMamba's dual-branch Pro-SSM block** and **DCM-Net's adaptive dual-branch feature cross-coupling module** are the most directly comparable fusion mechanisms. Both fuse CNN local features with Mamba global features, though at different granularities than your bottleneck-only fusion.

**CFM-UNet's CNN-Fusion-Mamba** approach is also relevant, as it explicitly addresses the semantic gap between CNN and Mamba features—the same gap your 1×1 conv fusion is designed to bridge.

### 2.3 For Bidirectional Mamba (your deferred decision)

**BiSegMamba** uses bidirectional tri-oriented Mamba with three orthogonal directional views and six concatenated sequences passed through a single normalized Mamba operator.

**DM-SegNet** uses a dual-Mamba architecture with bidirectional alignment between encoder and decoder state.

**SliceMamba** features a Bidirectional Slicing and Scanning (BSS) module with varied scanning mechanisms for sliced features.

**MSC-Mamba** (Mamba-Constrained Inter-Slice Consistency) introduces a bidirectional Mamba backbone tailored to capture long-range interactions along the slice axis. This is the most directly relevant bidirectional paper for your z-axis sequence.

**ABE-Mamba** proposes a Bidirectional Enhanced Mamba module for query-feature refinement via local-global context interactions.

The literature strongly suggests that **bidirectional scanning is the norm, not the exception**, for medical Mamba segmentation. Your single forward pass is the outlier.

---

## 3. Innovation Recommendations

Your architecture has a real opportunity for novelty, but the current design (pooled-vector + unidirectional Mamba + bottleneck-only fusion) risks being a **thin wrapper** around existing components rather than a genuine contribution. Here are four innovation directions, ordered by feasibility for a master's thesis.

### Innovation 1: Multi-Granularity Sequence Aggregation (High Feasibility)

**The problem with your current design:** Global average pooling discards all spatial information within each slice. A 256-channel 8×8 feature map becomes a single 256-vector—you lose the spatial layout of where the liver/tumor features are. Mamba then models inter-slice dependencies over a sequence of purely semantic, position-agnostic vectors.

**Recommendation:** Replace the single pooled vector per slice with a **small set of region-aware tokens per slice**. For example, partition the 8×8 bottleneck into four 4×4 quadrants, pool each quadrant, and concatenate: 4 tokens per slice × 128 slices = 512-step sequence. Alternatively, use **attention pooling** with a small set of learnable queries (e.g., 8 queries per slice), producing a (B, Z×8, d_model) sequence. This preserves spatial structure while keeping the sequence length manageable.

**Relevant papers:**
- **MHS-VM (Multi-Head Scanning in Parallel Subspaces)** uses a Mixture of Poolings design with spatial attention and channel attention pooling branches.
- **Attention Mamba** introduces an Adaptive Pooling block that accelerates attention computation while incorporating global information, combined with a bidirectional Mamba block.
- **SegMamba-V2** processes whole-volume features at every scale without discarding spatial structure.

**Why this is publishable:** The question "how much spatial information must be preserved per slice for effective inter-slice Mamba modeling?" is unanswered. A systematic ablation (1 pooled token → 4 regional tokens → 16 grid tokens → full spatial flatten) would be a genuine empirical contribution.

### Innovation 2: Slice-Aware Bidirectional Mamba with Learned z-Positional Encoding (High Feasibility)

**The problem:** Your causal (unidirectional) Mamba means slice z sees only slices ≤ z. While `RandFlipd(spatial_axis=2)` exposes the model to both z-orientations during training, validation and inference remain single-direction【your roadmap, toy-first policy table】. More fundamentally, Mamba's recurrence depth itself encodes position implicitly—but is that sufficient for medical volumes where anatomical position (superior vs. inferior liver) carries semantic meaning?

**Recommendation:** Implement a **bidirectional Mamba** (forward + reverse pass, sum outputs) and add a **learned z-positional embedding** to the pooled vectors before the Mamba block. The positional embedding is a simple `nn.Embedding(128, d_model)` added to the sequence. This is a ~32K-parameter addition with potentially large gains.

**Relevant papers:**
- **MSC-Mamba** explicitly introduces a bidirectional Mamba backbone for inter-slice consistency in stroke lesion segmentation.
- **BiSegMamba** demonstrates bidirectional tri-oriented Mamba with adaptive directional fusion.
- **Enhancing Vision Mamba with 2D position embedding** shows that position embedding methods designed for Transformers show limited improvement in Mamba models, but a properly designed 2D position embedding (adapted for your 1D z-sequence) can help.
- **P-Mamba** (pediatric echocardiography) adds position embedding before the Mamba block to ensure spatial context.

**Why this is publishable:** The interaction between bidirectional scanning and z-positional encoding in the specific context of pooled-vector sequences has not been studied. A clean ablation table (unidirectional vs. bidirectional; no PE vs. learned PE vs. sinusoidal PE) would be a solid contribution.

### Innovation 3: Uncertainty-Aware Slice Gating (Medium Feasibility, High Novelty)

**The problem:** Your architecture treats all slices equally—a slice through the middle of a large tumor contributes the same sequence weight as a slice through healthy liver parenchyma. This is wasteful and potentially harmful: ambiguous slices (partial volume, motion artifact, low contrast) may inject noise into the Mamba state.

**Recommendation:** Add a **lightweight slice-confidence gate** that modulates each slice's contribution to the Mamba sequence. A small MLP takes the pooled vector and outputs a scalar confidence score; the pooled vector is scaled by this score before entering Mamba. During training, the gate can be supervised with a auxiliary loss (e.g., predicting slice-level Dice or tumor presence). This is conceptually similar to **MemSAM-2.5D's Confidence-Gated Prototype Memory**, which uses uncertainty-aware boundary refinement.

**Relevant papers:**
- **MemSAM-2.5D** uses a Confidence-Gated Prototype Memory module for uncertainty-aware boundary refinement. Your gate would operate at the slice level rather than the prototype level.
- **UD-Mamba** (Uncertainty-Driven Mamba) achieves 89.15% DSC on a medical segmentation benchmark through uncertainty-driven design.
- **Mamba-Sea** uses global-to-local sequence augmentation for generalizable medical image segmentation.

**Why this is publishable:** Slice-level uncertainty gating for Mamba sequence modeling is genuinely novel. The clinical motivation (ambiguous slices should not corrupt inter-slice memory) is strong, and the implementation is lightweight.

### Innovation 4: Multi-Scale Mamba Injection into Skip Connections (Lower Feasibility, Higher Ceiling)

**The problem:** Your current design injects z-context only at the bottleneck (8×8 resolution). But liver tumors have **extreme scale variation**—small lesions may be better served by context injected at higher resolutions (16×16 or 32×32), while large lesions benefit from bottleneck context.

**Recommendation:** Extend the broadcast-concat-fusion mechanism to **multiple decoder levels**, not just the bottleneck. At each skip-connection concatenation point, broadcast the Mamba context (upsampled if necessary) and fuse via 1×1 conv. This is essentially **multi-level injection**, which your roadmap correctly identifies as an "extension chapter candidate" rather than the first version.

**Relevant papers:**
- **U-Mamba** enhances long-range dependency at multiple scales through a hybrid CNN–SSM design.
- **SegMamba** models long-range dependencies "at every scale".
- **DCM-Net** uses cross-layer feature fusion between CNN and Mamba branches.
- **MambaVesselNet++** uses a Bi-Decoder with skip connections to combine local and global information.

**Why this is publishable:** Multi-level Mamba injection for 2.5D segmentation is unexplored. The ablation is clean: bottleneck-only vs. bottleneck+one skip vs. bottleneck+all skips. The risk is higher (more parameters, more debug surface) but the potential gain is larger.

---

## 4. Summary Table: Recommendation → Papers

| Innovation | Core idea | Key papers | Feasibility |
|---|---|---|---|
| **Multi-granularity aggregation** | Replace 1 pooled vector/slice with 4–16 region-aware tokens | MHS-VM (mixture of poolings); Attention Mamba (adaptive pooling); SegMamba-V2 | High |
| **Bidirectional + z-positional encoding** | Bidirectional Mamba scan + learned slice-index embedding | MSC-Mamba (bidirectional inter-slice); BiSegMamba (tri-oriented bidirectional); 2D position embedding for Vision Mamba; P-Mamba (position embedding before Mamba) | High |
| **Uncertainty-aware slice gating** | Confidence gate modulating each slice's Mamba contribution | MemSAM-2.5D (confidence-gated prototype memory); UD-Mamba (uncertainty-driven); Mamba-Sea (sequence augmentation) | Medium |
| **Multi-scale Mamba injection** | Inject z-context at multiple decoder levels, not just bottleneck | U-Mamba (multi-scale hybrid); SegMamba (every-scale modeling); DCM-Net (cross-layer fusion); MambaVesselNet++ (Bi-Decoder skip connections) | Low–Medium |

---

## 5. Strategic Advice for a Conference Submission

**Positioning:** The strongest framing for your paper is not "we built a 2.5D Mamba for liver tumors" (MemSAM-2.5D already exists and is published in Frontiers in Oncology). The stronger framing is: **"We systematically study how to aggregate slice-level features for inter-slice Mamba modeling in volumetric medical segmentation, and identify a practical design that balances spatial fidelity and sequence length."** The ablation study is the contribution.

**Minimum viable novelty for MICCAI/ISBI:** Implement Innovation 1 (multi-granularity aggregation) with a clean ablation across aggregation granularities, and compare against MemSAM-2.5D and SegMamba on LiTS. If you can show that 4 regional tokens per slice outperforms 1 pooled token with a statistically significant margin, that alone is a publishable result.

**Stretch goal:** Combine Innovations 1 + 2 (multi-granularity + bidirectional with z-PE) and include an ablation table with 8–12 rows. That is a full paper.

**What to avoid:** Do not try to do all four innovations at once. The toy-first policy in your roadmap is correct. Start with the multi-granularity aggregation and the bidirectional Mamba—these are the two changes most likely to produce a measurable improvement, and they are the easiest to debug.
