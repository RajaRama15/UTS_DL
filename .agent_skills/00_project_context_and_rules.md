---
name: project-context-and-rules
category: project-orchestration
version: 1.0.0
description: >
  Master specification, strict grading rubric, anti-leakage rules, and one-command
  reproducibility contract for the UTS Deep Learning Midterm Exam repository.
triggers:
  - project rules
  - submission format
  - grading rubric
  - oral exam reproduction
---

# SKILL: UTS Deep Learning Project Context & Reproducibility Rules

## 1. Strict Submission Contract
All code and predictions live inside `E:\Code\uts_dl_submission` (the root of the GitHub repository).
The `main` branch MUST contain these exact 6 items at the deadline (24 Nov 2026 23:59):
1. `laporan.pdf` (max 18 pages excluding appendices)
2. `kode/` (containing `README.md` and all Python scripts; one command `python kode/run_all.py` must reproduce every number in the report)
3. `prediksi_klasifikasi.csv` (`id_unit,kelas` — 16,000 rows; `kelas` in `{normal, gores, noda, retak, gelembung, las_dingin, kontaminasi}`)
4. `prediksi_rul.csv` (`id_unit,rul` — 300 rows; `rul` float in `[0, 150]`)
5. `cluster.csv` (`id_sampel,cluster` — 36,000 rows; `cluster` integer)
6. `kontribusi.md` (member roles, PR links from individual forks, AI tools disclosure)

## 2. Coding & Execution Standards for Agentic AI
- **Python Environment:** Use `E:\Code\.venv\Scripts\python.exe` (`torch==2.15.0+cpu`, `scikit-learn==1.9.1`, `scipy==1.17.1`, `lightgbm==4.7.0`, `umap-learn==0.6.0`).
- **CPU Optimization:** Because this machine runs on a 10-thread CPU (no NVIDIA GPU), avoid heavy 2D convolutions inside inner training loops when static spatial filters or compact tensors can be precomputed once. Use batch sizes of `256–512`, `AdamW` with `CosineAnnealingLR`, and direct tensor slicing so models train in seconds rather than hours.
- **Checkpoint Size Limit:** Keep all files in `kode/checkpoints/` under **25 MB** (well below GitHub's 100 MB limit).
- **Support `--eval_only` Flag:** Every task script (`task1_classification.py`, `task2_rul.py`, `task3_clustering.py`) MUST accept `--data_dir`, `--out_dir`, and `--eval_only`. In `--eval_only` mode, it must load saved weights from `kode/checkpoints/`, print all validation metrics from the report in under 15 seconds, and regenerate the CSV prediction file (critical for the 5-minute live oral exam!).
- **Save Structured JSON Metrics:** Every script must save its exact numerical results to `kode/outputs/taskX_metrics.json` and high-DPI figures (`.png`) to `kode/outputs/` so `laporan.pdf` matches the code output 100%.
