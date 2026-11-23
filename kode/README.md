# Hands-on Deep Learning: Classification, Regression, and Clustering (UTS)

This repository contains the complete, reproducible Deep Learning pipelines for Tasks 1–3 of the Deep Learning Midterm Exam (UTS):
- **Task 1:** Multimodal Classification — Product Defects (`task1_classification.py`)
- **Task 2:** Time-Series Regression — Remaining Useful Life (`task2_rul.py`)
- **Task 3:** Multimodal Unsupervised Representation & Deep Clustering — Material Types (`task3_clustering.py`)

## One-Command Reproduction
To reproduce all experiments, figures, ablation tables, model checkpoints, and final prediction CSV files (`prediksi_klasifikasi.csv`, `prediksi_rul.csv`, `cluster.csv`) from the report, run:

```bash
python kode/run_all.py --data_dir E:/Code/data_uts_dl --out_dir E:/Code/uts_dl_submission
```

To quickly reproduce any individual task or evaluate pre-trained checkpoints during the oral exam:
```bash
python kode/task1_classification.py --data_dir E:/Code/data_uts_dl --out_dir E:/Code/uts_dl_submission --eval_only
python kode/task2_rul.py --data_dir E:/Code/data_uts_dl --out_dir E:/Code/uts_dl_submission --eval_only
python kode/task3_clustering.py --data_dir E:/Code/data_uts_dl --out_dir E:/Code/uts_dl_submission --eval_only
```

## Directory Layout
```text
uts_dl_submission/
├── laporan.pdf                  # Final report (max 18 pages)
├── kontribusi.md                # Member roles, PR links, and AI tool disclosure
├── prediksi_klasifikasi.csv     # Task 1 test predictions (id_unit,kelas)
├── prediksi_rul.csv             # Task 2 test predictions (id_unit,rul)
├── cluster.csv                  # Task 3 cluster assignments (id_sampel,cluster)
└── kode/
    ├── README.md                # Reproduction instructions & environment details
    ├── requirements.txt         # Exact library versions
    ├── run_all.py               # Master single-command reproduction script
    ├── utils.py                 # Seeding, logging, and common metrics
    ├── task1_classification.py  # Task 1 pipeline (Exploration, Encoders, Fusion, Label Noise, Domain Shift, Missing Modality, 3-seed evaluation)
    ├── task2_rul.py             # Task 2 pipeline (Regime Normalization, Causal Imputation/Despiking, Sequence Models, Baseline, Asymmetric Score & Uncertainty)
    ├── task3_clustering.py      # Task 3 pipeline (Distortion Removal, Self-Supervised Multimodal Embeddings, DEC Deep Clustering, Cluster Selection & UMAP)
    ├── checkpoints/             # Saved final model weights (< 100 MB)
    └── outputs/                 # Generated figures, JSON metrics, and markdown tables for laporan.pdf
```
