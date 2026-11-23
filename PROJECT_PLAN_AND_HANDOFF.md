# AGENT HANDOFF, MASTER PLAN & 2-PERSON WORK SPLIT
**Project:** UTS Deep Learning — *"Hands-on Deep Learning: Classification, Regression, and Clustering"*
**Deadline:** 24 November 2026, 23:59 (GitHub `main` branch, all forks synced & PRs merged)
**Team Size:** 2 Members (Person A + Person B)

> **For Your Teammate's Agentic AI:**
> Read this entire document AND the specialized skill files inside `.agent_skills/` before running or writing any code. Every hidden trap in the dataset has already been uncovered and documented below so you do not waste time re-discovering them.

---

## 1. REQUIRED GITHUB REPOSITORY DELIVERABLES (`main` branch)
The automated grading system checks exact filenames and CSV column headers on the `main` branch:

```text
uts_dl_submission/               # Root of the GitHub repository
├── laporan.pdf                  # Final report answering Tasks 1–3 (max 18 pages excl. appendices)
├── kontribusi.md                # Member roles, links to PRs from each member's fork, AI disclosure
├── prediksi_klasifikasi.csv     # Task 1 output: id_unit,kelas (16,000 rows) -> [ALREADY GENERATED]
├── prediksi_rul.csv             # Task 2 output: id_unit,rul (300 rows, float in [0, 150]) -> [TO DO]
├── cluster.csv                  # Task 3 output: id_sampel,cluster (36,000 rows, integer ID) -> [TO DO]
├── .agent_skills/               # Hidden folder of specialized skills for Agentic AI (Claude/Cursor/AGY)
└── kode/
    ├── README.md                # Single-command reproduction guide
    ├── requirements.txt         # Pinned library versions
    ├── run_all.py               # Master single-command script (supports --eval_only & --task 1|2|3)
    ├── utils.py                 # Deterministic seeding, JSON saver, asymmetric RUL score function
    ├── task1_classification.py  # Task 1 pipeline -> [COMPLETED & VERIFIED]
    ├── task2_rul.py             # Task 2 pipeline -> [TO BUILD & RUN]
    ├── task3_clustering.py      # Task 3 pipeline -> [TO BUILD & RUN]
    ├── checkpoints/             # Saved model weights (< 100 MB limit for GitHub)
    │   └── task1_final_ensemble.pt  # [COMPLETED, 2.47 MB]
    └── outputs/                 # Generated plots & JSON metrics for laporan.pdf
        ├── task1_metrics.json
        ├── t1_class_examples.png
        ├── t1_distribution_and_shift.png
        └── t1_training_and_cm.png
```

---

## 2. FAIR 50/50 WORK SPLIT FOR A 2-PERSON GROUP
Because GitHub contribution is graded via **Forks + Pull Requests** (`Contribution Factor`: `1.00` if visible on GitHub & explainable in oral exam; `0.85` or `0.70` otherwise), **both members must push commits from their own fork and open Pull Requests into `main`**.

Here is the balanced 50/50 split of the project (42.5 group points each + shared report):

| Member | Assigned Ownership (Code, PRs, & Report Sections) | Points Covered | Current Status |
| :--- | :--- | :---: | :--- |
| **Person A (You)** | **1. Repository Setup, Common Utils & Master Runner** (`kode/utils.py`, `kode/run_all.py`, `kode/README.md`)<br>**2. Task 1: Multimodal Classification — Product Defects** (`kode/task1_classification.py`, `prediksi_klasifikasi.csv`, Report Section 1)<br>**3. Task 3 (Part 1 — Preprocessing & Representation):** Distortion analysis & removal (microscope M1/M2 brightness shift, 3-point median spike removal, baseline detrending) + Multimodal Self-Supervised Representation Learning (`task3_clustering.py` stages a–c, Report Section 3a–3c) | **42.5 / 85 pts**<br>*(Task 1: 30 pts + Half of Task 3: 12.5 pts)* | **Task 1 is 100% DONE** (`prediksi_klasifikasi.csv` & all metrics/plots saved).<br>Ready to assist on Task 3a–c & Report Section 1. |
| **Person B (Friend)** | **1. Task 2: Time-Series Regression — Remaining Useful Life (RUL)** (`kode/task2_rul.py`, `prediksi_rul.csv`, Report Section 2)<br>**2. Task 3 (Part 2 — Deep Clustering & Evaluation):** Deep Embedded Clustering (DEC) vs Classical Clustering (GMM/K-Means/Spectral), optimal cluster count $K$ determination (at least 2 criteria), UMAP/t-SNE visualization & hard-to-separate cluster analysis (`task3_clustering.py` stages d–f, `cluster.csv`, Report Section 3d–3f)<br>**3. Final PDF Compilation & Git PR Merge Verification** (`laporan.pdf`, `kontribusi.md`) | **42.5 / 85 pts**<br>*(Task 2: 30 pts + Half of Task 3: 12.5 pts)* | **Ready to start immediately** using the blueprint & `.agent_skills/` below. |

> **Oral Exam Rule Reminder:** Even though the implementation and PRs are split 50/50 above, the lecturer can ask either member about **any** of Tasks 1, 2, or 3 during the 20-minute oral exam. Both members must read the final report and know how to run `python kode/run_all.py --eval_only`.

---

## 3. WHAT HAS ALREADY BEEN COMPLETED (TASK 1 SUMMARY)
Task 1 (`kode/task1_classification.py`) has been fully executed and verified:
- **Dataset:** 48,000 train units, 16,000 test units. Modalities: `gambar` $(64\times 64\text{ uint8})$ and `sensor` $(3\times 128\text{ float16})$ [Ch 0: Temp °C, Ch 1: Vibration, Ch 2: Current A].
- **Key Findings Uncovered & Solved:**
  1. **Domain Shift (`lini` A vs B):** Train is **74.92% Lini A / 25.08% Lini B**, while Test is **39.96% Lini A / 60.04% Lini B**. Lini A images have mean brightness $139.8 \pm 15.2$, while Lini B images are darker ($111.7 \pm 19.1$). Lini A temperature is $181.12^\circ\text{C}$, while Lini B is $183.12^\circ\text{C}$. We implemented **Per-Lini + Instance Normalization** fitted strictly on training folds.
  2. **Modality Complementarity (Task 1c):**
     - **Image-only model** gets **0.5116 Macro-F1** (excels on `noda` 0.839, `kontaminasi` 0.753, `gores` 0.541, but gets **0.00–0.02 F1 on `las_dingin`** because cold welds are invisible on the surface!).
     - **Sensor-only model** gets **0.4504 Macro-F1** (excels on `las_dingin` **0.9326 F1** via current drop, `gelembung` **0.7794 F1** via temp drop, `retak` **0.5989 F1** via vibration spike, but gets **0.00 F1 on `noda` and `kontaminasi`**).
  3. **Fusion Comparison (Task 1b):** Compared **Early Fusion** (0.7203), **Late Fusion** (0.7258), and **Cross-Modal Attention + Gated Fusion** (**0.7298–0.7822**).
  4. **Imperfect Labels (Task 1d):** Out-of-fold confident learning detected **371 high-confidence inspector mislabels (3.86% of validation set)** and compared Standard CE, Label Smoothing, and Generalized Cross-Entropy (GCE).
  5. **Missing Modality (Task 1f):** 8.05% of train and 8.11% of test have `NaN` sensor channels (4.90% all 3 channels `NaN`, ~1.05% each single channel `NaN`). Solved via 3-bit channel mask + Modality Dropout during training.
  6. **Sanity Check & 3-Seed Ensemble (Task 1g):** 1-batch (64 samples) overfitting reached loss `0.0080` (100% accuracy). 3-seed ensemble (`42, 123, 999`) saved to `kode/checkpoints/task1_final_ensemble.pt` and predictions written to `prediksi_klasifikasi.csv`.

---

## 4. COMPLETE TECHNICAL BLUEPRINT FOR TASK 2 (`kode/task2_rul.py`)
**Objective:** Predict Remaining Useful Life (RUL, capped at 150 cycles) at the last recorded cycle of each of the 300 test machines in `rul_test.csv`.
**Files:** `rul_train.csv` (900 machines, 254,174 rows) and `rul_test.csv` (300 machines, 51,692 rows).

### Critical Data Facts Already Discovered by Our EDA (Do Not Skip!):
1. **Six Discrete Operating Regimes (`setel_1`, `setel_2`, `setel_3`) (Task 2a):**
   - Just like NASA C-MAPSS FD002/FD004, machines switch randomly every cycle between **6 discrete operating regimes**:
     - Regime 0: `(0, 0.00, 100)` (~16.7% of rows)
     - Regime 1: `(10, 0.25, 100)` (~16.5% of rows)
     - Regime 2: `(20, 0.70, 100)` (~16.7% of rows)
     - Regime 3: `(25, 0.62, 60)` (~16.6% of rows)
     - Regime 4: `(35, 0.84, 100)` (~16.6% of rows)
     - Regime 5: `(42, 0.84, 100)` (~16.7% of rows)
   - **Rule:** You **MUST** assign each row to its operating regime (`op_id = round(setel_1)`) and normalize each sensor `s01..s21` using the **regime-specific mean/std (or median/IQR)** computed strictly on the training machines! Without regime-conditioned normalization, the operating regime variance completely drowns out the degradation signal.
2. **Uninformative vs. Informative Sensors (Task 2a):**
   - After regime-conditioned normalization, **7 sensors have zero correlation with RUL ($|r| < 0.01$)** and should be documented and dropped: **`s01, s05, s06, s10, s16, s18, s19`**.
   - **14 informative sensors** have strong monotonic correlation with capped RUL:
     `s04` ($-0.803$), `s12` ($-0.587$), `s02` ($+0.570$), `s15` ($-0.552$), `s03` ($+0.524$), `s07` ($+0.514$), `s17` ($+0.424$), `s13` ($+0.403$), `s20` ($+0.389$), `s09` ($-0.384$), `s14` ($+0.381$), `s08` ($-0.349$), `s21` ($+0.338$), `s11` ($+0.132$).
3. **Missing Values & Severe Spikes (Task 2a & 2c — Causal Preprocessing):**
   - Every sensor has **~2.0% missing values (`NaN`)** and **0.1%–2.4% extreme spikes** (up to $8.9\times \text{IQR}$, especially in `s17, s13, s20, s12`).
   - **Strict Anti-Leakage Causal Cleaning Per Machine:**
     1. Compute regime-normalized values $z_{t,s} = (x_{t,s} - \text{median}_{\text{op},s}) / (\text{IQR}_{\text{op},s} + 10^{-6})$.
     2. Replace spikes where $|z_{t,s}| > 4.0$ with `NaN` (or clip to $[-4, 4]$).
     3. Impute `NaN`s using **causal forward-fill (`ffill`) per machine** (fallback `0.0` at cycle 1 if missing) — **NEVER use backward-fill (`bfill`) or future interpolation!**
     4. Add causal Exponential Moving Average (EMA, $\alpha = 0.25$) trends + normalized cycle age $t / 300$ as additional input channels.
4. **Target Labels & Sliding Windows (Task 2b & 2c):**
   - For each training machine $u$, $\text{RUL}_t = \min(150, \max\_\text{siklus}_u - \text{siklus}_t)$.
   - Minimum cycle length in `rul_test.csv` is **50 cycles** (range 50–455), and in `rul_train.csv` is **133 cycles** (range 133–778).
   - Therefore, a sliding window of length **$W = 50$** (with stride 2 or 3 during training, plus always including the last window of each machine) fits **every single test machine with zero padding required**!
5. **Models, Validation, Asymmetric Loss & Uncertainty (Task 2d, 2e, 2f):**
   - **Validation Split (Task 2e):** Split by `id_unit` (e.g., 720 train machines / 180 val machines). Report **RMSE** and **Asymmetric Score** overall and broken down by RUL range: $[0, 30]$, $(30, 80]$, $(80, 150]$.
   - **Model Comparison (Task 2d):**
     1. **Non-DL Baseline:** `LightGBMRegressor` (or `Ridge` / `RandomForest`) on window summary statistics (last value, EMA mean, slope, std, min, max over window $W=50$).
     2. **Sequence Model 1:** **Bidirectional / Stacked GRU (or LSTM)** with temporal attention pooling and dropout (`p=0.2`).
     3. **Sequence Model 2:** **Dilated 1D-CNN / Temporal Convolutional Network (TCN)** with residual blocks and dropout (`p=0.2`).
   - **Asymmetric Loss Optimization:** Train neural networks with a hybrid Smooth Asymmetric + MSE loss:
     $$\mathcal{L}(d) = \text{MSE}(d) + \lambda \cdot \begin{cases} \exp(-d/13) - 1 & d < 0 \\ \exp(d/10) - 1 & d \ge 0 \end{cases}, \quad d = \hat{y} - y$$
     Since over-predicting RUL ($d > 0$, late prediction) is penalized exponentially harder ($1/10$ vs $1/13$), shifting the final point prediction slightly conservatively using uncertainty ($\hat{y}_{\text{safe}} = \text{clip}(\hat{\mu} - \beta \cdot \hat{\sigma}, 0, 150)$) simultaneously improves safety and reduces the asymmetric score!
   - **Uncertainty (Task 2f):** Run **MC Dropout** ($T=30$ stochastic forward passes) + **Multi-Model/Multi-Seed Ensemble** to obtain mean $\hat{\mu}_i$ and standard deviation $\hat{\sigma}_i$ (90% prediction interval $[\hat{\mu}_i - 1.645\hat{\sigma}_i, \hat{\mu}_i + 1.645\hat{\sigma}_i]$). Discuss the maintenance trigger policy: service machine $i$ when lower bound $\hat{\mu}_i - 1.645\hat{\sigma}_i \le \tau_{\text{service}}$ (e.g., 20 cycles).

---

## 5. COMPLETE TECHNICAL BLUEPRINT FOR TASK 3 (`kode/task3_clustering.py`)
**Objective:** Group $N = 36,000$ unlabeled material samples (`clustering.npz`, `clustering_meta.csv`) into material types and output `cluster.csv` (`id_sampel,cluster`).
**Scoring Metric:** Adjusted Rand Index (ARI) on hidden ground-truth material types.

### Critical Data Facts Already Discovered by Our EDA (Do Not Skip!):
1. **Severe Distortions in Both Modalities (Task 3a):**
   - **Image (`gambar`, $36000\times 64\times 64$ uint8):**
     - **Microscope Domain Shift:** `M1` (21,636 samples) has mean brightness **$127.80 \pm 23.71$**, whereas `M2` (14,364 samples) has mean brightness **$146.07 \pm 23.22$** ($+18.3$ offset!).
     - Also contains random **rotations, scale differences, and brightness shifts**.
     - **Fix:** Normalize every image to **zero mean and unit variance per instance** (completely eliminates M1 vs M2 brightness/contrast shift!), and extract **rotation-invariant features** (radially averaged 2D FFT power spectrum profile + multi-scale local variance/Laplacian histograms + rotation/scale-augmented CNN Autoencoder/BYOL encoder).
   - **Spectrum (`spektrum`, $36000\times 128$ float16):**
     - **Single-Bin Spikes:** **21.84% of spectra (7,861 samples)** have massive single-channel spikes (jump $> 1.0$, up to $4.28$). Applying a **3-point sliding median filter** along the 128 channels removes 100% of these spikes without distorting real spectral peaks!
     - **Linear Baseline Drift/Tilt:** Spectra have random linear tilts (end minus start varies from $-0.65$ to $+0.64$). Fit and subtract the linear baseline connecting the median of the first 8 bins and last 8 bins (or Savitzky-Golay / 2nd-order polynomial detrend), followed by L2/Z-score row normalization.
     - **Horizontal Wavelength Shifts:** Small 1–2 bin shifts along the 128 channels. Use shift-invariant 1D-CNN pooling + local peak-aligned features + random $\pm 2$ bin shift augmentation during contrastive training.
2. **Self-Supervised Representation & Multimodal Fusion (Task 3b & 3c):**
   - Train a **Joint Multimodal Contrastive + Reconstruction Autoencoder (SimCLR/BYOL + AE)** where:
     - Image branch processes instance-normalized $64\times 64$ microstructure images (with random $90^\circ$ rotation, flip, contrast jitter, and radial 2D-FFT invariance).
     - Spectrum branch processes despiked, baseline-corrected 128-bin spectra (with random $\pm 1$ bin shift, Gaussian noise, and baseline tilt augmentation).
     - Fuse L2-normalized image embedding $z_{\text{img}}$ and spectrum embedding $z_{\text{spec}}$ (plus cross-modal alignment / CCA / concatenated whitened embeddings).
     - Show ablation (Task 3c): Image-only vs. Spectrum-only vs. Multimodal fused embeddings.
3. **Determining Number of Clusters $K$ & Deep Clustering (Task 3d, 3e, 3f):**
   - **Number of Clusters $K$ (Task 3e):** Sweep $K \in \{3, 4, 5, 6, 7, 8, 9, 10\}$ on the fused latent space and plot **at least 3 criteria**:
     1. **Silhouette Score** (maximize)
     2. **Calinski-Harabasz Index / Inertia Elbow** (maximize / elbow)
     3. **Davies-Bouldin Index** (minimize) & **BIC (Bayesian Information Criterion)** from GMM
   - **Compare Clustering Methods (Task 3d):**
     1. **K-Means++** on fused embeddings
     2. **Gaussian Mixture Model (GMM) / Spectral Clustering** on fused embeddings
     3. **Deep Embedded Clustering (DEC / IDEC):** Initialize cluster centroids $\mu_j$ with K-Means/GMM on pretrained embeddings, compute soft Student's $t$-assignments $q_{ij} \propto (1 + \|z_i - \mu_j\|^2)^{-1}$, and refine with KL divergence $D_{\text{KL}}(P \parallel Q)$ where $p_{ij} \propto q_{ij}^2 / \sum_i q_{ij}$ jointly with reconstruction loss.
   - **Verify Microscope Invariance:** Check cross-tabulation of predicted clusters vs `mikroskop` (`M1` vs `M2`) to confirm every cluster has the natural ~60% M1 / 40% M2 split (proving we clustered material types, not microscopes!).
   - **Visualization (Task 3f):** Plot 2D **UMAP** (and t-SNE) colored by predicted material cluster and by microscope (`M1`/`M2`), and explain which clusters partially overlap (e.g., materials with similar chemical spectra that only differ in grain size/microstructure, or vice versa).
