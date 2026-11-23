---
name: task1-multimodal-classification-reference
category: deep-learning / multimodal-classification
version: 1.0.0
description: >
  Reference summary of the completed Task 1 pipeline (Multimodal Product Defect Classification)
  so any agent writing the final report (laporan.pdf) or verifying run_all.py has the exact
  architecture, experimental numbers, and file paths.
triggers:
  - task 1
  - prediksi_klasifikasi.csv
  - defect classification report
---

# SKILL: Task 1 Multimodal Defect Classification Reference (Completed)

## 1. Completed Code & Artifacts
- **Main Script:** `E:\Code\uts_dl_submission\kode\task1_classification.py`
- **Saved Checkpoint:** `E:\Code\uts_dl_submission\kode\checkpoints\task1_final_ensemble.pt` (2.47 MB)
- **Metrics JSON:** `E:\Code\uts_dl_submission\kode\outputs\task1_metrics.json`
- **Generated Figures:**
  - `kode/outputs/t1_class_examples.png` (Representative image & 3-channel sensor waveforms for all 7 classes)
  - `kode/outputs/t1_distribution_and_shift.png` (Class imbalance bar chart, Train vs Test Lini A/B shift, and pixel brightness shift histogram)
  - `kode/outputs/t1_training_and_cm.png` (1-batch overfitting sanity check curve, train/val learning curves, and normalized confusion matrix)
- **Test Predictions:** `E:\Code\uts_dl_submission\prediksi_klasifikasi.csv` (16,000 test units, columns `id_unit,kelas`).

## 2. Key Results to Cite in `laporan.pdf`
- **Sanity Check (1g):** 1-batch (64 samples) overfitting dropped loss from `2.1555` (Acc `9.4%`) to `0.00802` (Acc `100.0%`) in 40 steps.
- **Modality Value (1c):**
  - Image-Only (`OrientedImageEncoder`): Tuned Macro-F1 = **0.5116** (`noda`: 0.8386, `normal`: 0.7797, `kontaminasi`: 0.7527, `gores`: 0.5289, `retak`: 0.3299, `gelembung`: 0.3242, `las_dingin`: **0.0271**).
  - Sensor-Only (`MultiScaleSignalEncoder`): Tuned Macro-F1 = **0.4504** (`las_dingin`: **0.9326**, `gelembung`: **0.7794**, `normal`: 0.6652, `retak`: **0.5989**, `gores`: 0.1763, `noda`: **0.0000**, `kontaminasi`: **0.0000**).
- **Fusion Comparison (1b):**
  - Early Fusion (Concat + MLP): Tuned Macro-F1 = **0.7203**
  - Late Fusion (Mask-Aware Logit Gating): Tuned Macro-F1 = **0.7258**
  - Cross-Modal Attention + Gated Residual Fusion: Single-model Macro-F1 = **0.7298–0.7821**, 3-Seed Ensemble Macro-F1 = **0.7346–0.7822** (`las_dingin`: 0.9319, `normal`: 0.8793, `noda`: 0.8516, `gelembung`: 0.8000, `kontaminasi`: 0.7818, `gores`: 0.6485, `retak`: 0.5822).
- **Label Noise (1d):** Identified **371 high-confidence inspector mislabels (3.86% of validation set)** where model confidence $>0.85$ contradicted inspector label ($p_{\text{given}} < 0.08$).
- **Domain Shift (1e):** Lini B represents 25.07% of Train (`M07–M12`) but 60.04% of Test. Per-Lini + Instance Normalization improved Lini B validation Macro-F1 from **0.5647** (Global Norm) to **0.5761** (+1.14% F1 gain on the test-dominant production line).
- **Missing Modality (1f):** All 3 sensors present: **0.7265 F1**; 1 channel missing: **0.6243 F1**; all sensors missing (natural): **0.4300 F1**; synthetic 100% sensor blackout: **0.4710 F1** (graceful fallback to image branch without NaN/crash).
