# Kontribusi Kelompok & Penggunaan AI (UTS Deep Learning)

## 1. Pembagian Tugas Anggota Kelompok dan Tautan Pull Request

| Nama Anggota | NIM | Peran Utama | Tautan Fork & Pull Request |
| :--- | :--- | :--- | :--- |
| Anggota 1 (Isi Nama) | (Isi NIM) | **Task 1 (Multimodal Classification):** Eksplorasi data, arsitektur Image & Signal Encoder, eksperimen Fusion (Early/Late/Cross-Attention), penanganan Missing Modality, Label Noise, dan Domain Shift antar lini. | [PR #1](#) |
| Anggota 2 (Isi Nama) | (Isi NIM) | **Task 2 (Time-Series Regression - RUL):** Eksplorasi rezim operasi, normalisasi per-rezim, penanganan spike & missing values secara kausal, pemodelan sekuens (LSTM/GRU, 1D-CNN/TCN, Baseline), Asymmetric Loss, dan estimasi Uncertainty (MC Dropout / Ensemble). | [PR #2](#) |
| Anggota 3 (Isi Nama) | (Isi NIM) | **Task 3 (Multimodal Clustering) & Integrasi Reproduksibilitas:** Pembersihan distorsi mikroskop & spektrum, pembelajaran representasi tanpa label (Contrastive/Autoencoder), Deep Embedded Clustering (DEC), penentuan jumlah klaster, visualisasi UMAP/t-SNE, dan integrasi `run_all.py`. | [PR #3](#) |

> *Catatan:* Seluruh anggota kelompok telah mempelajari, meninjau (code review), dan memverifikasi kode serta hasil dari ketiga tugas (Task 1, Task 2, dan Task 3) sehingga setiap anggota siap menjelaskan dan mereproduksi seluruh bagian pekerjaan pada saat ujian lisan.

---

## 2. Pernyataan Penggunaan Alat AI (AI Tools Disclosure)

Sesuai dengan ketentuan ujian, kelompok kami mendeklarasikan penggunaan alat bantu AI sebagai berikut:

- **Alat AI yang Digunakan:** Antigravity (Google DeepMind Coding Assistant).
- **Bagian yang Dibantu oleh AI:**
  1. **Exploratory Data Analysis (EDA) & Pemeriksaan Statistik Data:** Membantu menulis skrip inspeksi awal untuk mengecek distribusi kelas, proporsi missing values (`NaN`), pergeseran distribusi (domain shift) antara Lini A dan Lini B pada Task 1, identifikasi 6 rezim operasi diskret pada Task 2, serta deteksi spike dan kemiringan baseline spektrum pada Task 3.
  2. **Implementasi Boilerplate PyTorch & Pipeline Reproduksibilitas:** Membantu menstrukturkan modularitas kode PyTorch (`kode/task1_classification.py`, `kode/task2_rul.py`, `kode/task3_clustering.py`, dan `kode/run_all.py`) agar seluruh angka, tabel ablasi, dan grafik pada laporan dapat direproduksi dengan satu perintah (`one-command reproduction`).
  3. **Visualisasi & Format Laporan:** Membantu merapikan plot kurva pembelajaran (learning curves), matriks konfusi, interval ketidakpastian (uncertainty intervals), dan proyeksi UMAP/t-SNE.
- **Cara Kelompok Memverifikasi Output:**
  1. **Sanity Check (Overfitting One Batch):** Memastikan setiap arsitektur jaringan saraf mampu mencapai loss mendekati nol (akurasi ~100% pada satu batch) sebelum melakukan pelatihan penuh.
  2. **Validasi Anti-Kebocoran Data (No Data Leakage):** Memverifikasi secara manual bahwa seluruh parameter normalisasi (mean, std, median, IQR) dihitung **hanya dari split data latih**, pembagian validasi Task 2 dilakukan **per mesin (`id_unit`)** bukan per baris, dan pemrosesan deret waktu dilakukan secara kausal (tanpa melihat masa depan).
  3. **Evaluasi Multi-Seed & Pemeriksaan Format Prediksi:** Menjalankan model akhir pada minimal 3 random seed berbeda (`42, 123, 999`), memverifikasi konsistensi metrik validasi (Macro-F1, RMSE, Asymmetric Score, Silhouette/ARI), serta memeriksa kecocokan header, jumlah baris, dan rentang nilai pada `prediksi_klasifikasi.csv`, `prediksi_rul.csv`, dan `cluster.csv`.

---

## 3. Persetujuan Anggota Kelompok

Kami menyatakan bahwa seluruh pekerjaan ini adalah hasil kerja kelompok kami sendiri, tidak membagikan kode atau hasil kepada kelompok lain, dan seluruh anggota menyetujui isi repositori serta laporan ini:

- [x] **Anggota 1:** (Nama Lengkap & Tanda Tangan/Username GitHub)
- [x] **Anggota 2:** (Nama Lengkap & Tanda Tangan/Username GitHub)
- [x] **Anggota 3:** (Nama Lengkap & Tanda Tangan/Username GitHub)
