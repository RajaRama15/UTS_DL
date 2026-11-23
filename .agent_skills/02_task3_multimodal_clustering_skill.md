---
name: task3-multimodal-deep-clustering
category: deep-learning / unsupervised-clustering
version: 1.0.0
description: >
  Expert blueprint for Task 3 (Multimodal Unsupervised Clustering of 36,000 material samples).
  Covers microscope domain shift removal (M1 vs M2), 3-point median spectral despiking,
  linear baseline detrending, rotation-invariant 2D-FFT + SimCLR/BYOL + Autoencoder representations,
  Deep Embedded Clustering (DEC), cluster count determination, and UMAP analysis.
triggers:
  - task 3
  - clustering.npz
  - deep embedded clustering
  - material types clustering
  - spectrum baseline spike
---

# SKILL: Task 3 Multimodal Deep Clustering (25 Points)

## 1. Domain & Dataset Facts (`clustering.npz` & `clustering_meta.csv`)
- **Samples:** $N = 36,000$ unlabeled samples (`id_sampel` `S000000`..).
- **Modalities:**
  1. `gambar`: `uint8` $(36000, 64, 64)$ microstructure images from two microscopes: `M1` (21,636 samples, 60.1%) and `M2` (14,364 samples, 39.9%).
  2. `spektrum`: `float16` $(36000, 128)$ composition spectra (no `NaN`s).
- **Hidden Evaluation Metric:** **Adjusted Rand Index (ARI)** against true material types.

## 2. Step-by-Step Pipeline Requirements (Rubric 3a–3f)

### 3a. Distortion Inspection & Removal (CRITICAL FOR HIGH ARI)
1. **Spectrum Distortions (`spektrum`):**
   - **Single-Bin Spikes (affects 21.84% of samples!):** Apply a 3-point sliding median filter along the 128 spectral bins. This completely eliminates single-channel spikes while preserving true spectral peaks.
   - **Linear Baseline Drift/Tilt:** Estimate the linear baseline connecting the median of the first 8 bins ($\text{bins } 0..7$) and the median of the last 8 bins ($\text{bins } 120..127$) (or use least-squares linear/quadratic detrending + Savitzky-Golay smoothing), subtract it, and apply per-sample L2 / Z-score normalization.
   - **Horizontal Shift Distortion:** Spectra exhibit slight $\pm 1\text{–}2$ bin horizontal wavelength shifts. Use multi-scale average pooling (bins of width 2, 4, 8) + 1D-CNN with shift augmentation + sorted/aligned peak descriptors so small horizontal shifts map to the same latent representation.
2. **Microstructure Image Distortions (`gambar`):**
   - **Microscope Shift (`M1` vs `M2`):** `M1` has mean $127.80 \pm 23.71$, while `M2` has mean $146.07 \pm 23.22$. Apply **per-image instance normalization** $(I - \mu_I) / (\sigma_I + 10^{-5})$ so `M1` and `M2` have identical intensity distributions!
   - **Rotation & Scale Distortions:** Microstructure grain patterns can be rotated at any angle or slightly zoomed.
     - Extract **Radially Averaged 2D FFT Power Spectrum** (32 radial frequency rings from the $64\times 64$ 2D Fast Fourier Transform $\log(1 + |\mathcal{F}(I)|^2)$) + multi-scale isotropic Laplacian/gradient quantile descriptors (mathematically 100% invariant to rotation and translation!).
     - Combine with a 2D Convolutional Autoencoder / Contrastive Encoder trained with random $90^\circ$ rotations, flips, random crops/resizing (scale invariance), and brightness/contrast jitter.

### 3b & 3c. Self-Supervised Representation Learning & Multimodal Fusion
- **Dual-Branch Self-Supervised Network (Contrastive + Denoising Autoencoder):**
  - **Spectrum Encoder:** Takes cleaned 128-bin spectrum; trained with denoising reconstruction + SimCLR/BYOL contrastive loss under random $\pm 2$ bin shifts, random baseline tilt injection, and random spike injection (teaching the network explicit invariance to all 3 spectral distortions!).
  - **Image Encoder:** Takes instance-normalized $64\times 64$ image + radial 2D-FFT profile; trained with reconstruction + SimCLR/BYOL contrastive loss under random rotation, flip, scale jitter, and contrast jitter.
- **Multimodal Fusion & Ablation (Task 3c):**
  - L2-normalize the learned image embedding $z_{\text{img}} \in \mathbb{R}^{d}$ and spectrum embedding $z_{\text{spec}} \in \mathbb{R}^{d}$, apply PCA whitening / balanced concatenation $z_{\text{fused}} = [w_{\text{img}} \hat{z}_{\text{img}}, w_{\text{spec}} \hat{z}_{\text{spec}}]$.
  - Quantitatively and visually compare **Image-Only**, **Spectrum-Only**, and **Fused Multimodal** clustering (compute cross-ARI between single-modality clusters and multimodal clusters, showing which material types share the same spectrum polymorphs but differ in microstructure, and vice versa!).

### 3d & 3e. Determining Number of Clusters $K$ & Deep Embedded Clustering (DEC)
- **Determine $K$ with at least 2 (ideally 3) Quantitative Criteria (Task 3e):**
  Evaluate $K \in \{3, 4, 5, 6, 7, 8, 9, 10\}$ on the fused embeddings using:
  1. **Silhouette Score** ($\uparrow$)
  2. **Calinski-Harabasz Index** ($\uparrow$) & **Davies-Bouldin Index** ($\downarrow$)
  3. **GMM BIC / Stability across seeds**
- **Compare Clustering Algorithms (Task 3d):**
  1. **K-Means++** on $z_{\text{fused}}$
  2. **Gaussian Mixture Model (GMM, full/diag covariance)** on $z_{\text{fused}}$
  3. **Deep Embedded Clustering (DEC / IDEC):**
     Initialize learnable cluster centers $\{\mu_j\}_{j=1}^K$ from K-Means++/GMM on $z_{\text{fused}}$. Compute soft assignments via Student's $t$-distribution ($\alpha = 1$):
     $$q_{ij} = \frac{(1 + \|z_i - \mu_j\|^2)^{-1}}{\sum_{j'} (1 + \|z_i - \mu_{j'}\|^2)^{-1}}, \quad p_{ij} = \frac{q_{ij}^2 / f_j}{\sum_{j'} q_{ij'}^2 / f_{j'}}, \quad f_j = \sum_i q_{ij}$$
     Fine-tune the encoder and centroids jointly using $\mathcal{L} = \mathcal{L}_{\text{recon/contrast}} + \gamma D_{\text{KL}}(P \parallel Q)$.
- **Sanity Check Against Microscope Confounding:**
  Verify that every cluster in `cluster.csv` contains ~60% `M1` and ~40% `M2` samples ($\chi^2$ / mutual information near zero).

### 3f. UMAP / t-SNE Visualization & Cluster Separation Analysis
- Project a stratified subset (e.g., 6,000 samples) of $z_{\text{fused}}$, $z_{\text{img}}$, and $z_{\text{spec}}$ to 2D using **UMAP** (and/or **t-SNE**).
- Plot side-by-side: (1) Fused UMAP colored by final cluster, (2) Fused UMAP colored by Microscope `M1` vs `M2` (showing complete mixing/invariance), (3) Image-only vs Spectrum-only UMAP showing why single modalities fail to separate polymorphic or isomorphic material pairs.
