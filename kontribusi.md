# Kontribusi Kelompok & Penggunaan AI (UTS Deep Learning — Kelompok 2 Orang)

## 1. Pembagian Tugas Anggota Kelompok (50% / 50%) dan Tautan Pull Request

Kelompok kami terdiri dari **2 anggota**. Pembagian beban kerja dibagi secara seimbang (masing-masing **42.5 poin** dari total 85 poin tugas kelompok, ditambah penyusunan bersama `laporan.pdf` dan pengujian reproduksibilitas penuh untuk ujian lisan):

| Nama Anggota | NIM | Peran Utama & Tanggung Jawab (50/50 Split) | Tautan Fork & Pull Request |
| :--- | :--- | :--- | :--- |
| **Anggota 1** (Wildan Janitra Ramadhan) | (Isi NIM) | **1. Task 1 — Multimodal Classification (`kode/task1_classification.py`, `prediksi_klasifikasi.csv`, 30 poin):** Eksplorasi distribusi kelas & pergeseran Lini A/B, arsitektur *Oriented 2D-CNN* & *Multi-Scale 1D-CNN*, perbandingan *Early/Late/Cross-Attention Fusion*, analisis nilai tiap modalitas, deteksi *label noise*, *Per-Lini Normalization*, penanganan *missing sensor modality*, serta *3-seed ensemble*.<br>**2. Task 3 Bagian A–C (`kode/task3_clustering.py`, 12.5 poin):** Pembersihan distorsi mikroskop (`M1` vs `M2`) dan spektrum (*3-point median despiking* & *linear baseline detrending*), serta pembelajaran representasi *self-supervised multimodal* (*Contrastive/Autoencoder*).<br>**3. Setup Infrastruktur (`kode/utils.py`, `kode/run_all.py`, `kode/README.md`).** | [PR #1](#) |
| **Anggota 2** (Nama Teman) | (Isi NIM) | **1. Task 2 — Time-Series RUL Regression (`kode/task2_rul.py`, `prediksi_rul.csv`, 30 poin):** Identifikasi & normalisasi per-6 rezim operasi (`setel_1..3`), seleksi 14 sensor informatif vs 7 sensor tidak informatif, pembersihan *spike* & *missing values* secara kausal (`ffill`), pembuatan *sliding window* ($W=50$), perbandingan model sekuens (*Dilated TCN* & *BiGRU/LSTM*) dengan baseline *LightGBM*, optimasi *Asymmetric Score*, serta kuantifikasi ketidakpastian (*MC Dropout & Ensemble prediction intervals*).<br>**2. Task 3 Bagian D–F (`kode/task3_clustering.py`, `cluster.csv`, 12.5 poin):** Eksperimen *Deep Embedded Clustering (DEC)* vs *K-Means++/GMM*, penentuan jumlah klaster material ($K$) dengan $\ge 2$ kriteria, visualisasi UMAP/t-SNE, serta kompilasi akhir `laporan.pdf`. | [PR #2](#) |

> *Catatan Kesiapan Ujian Lisan:* Meskipun implementasi dan *Pull Request* dibagi 50/50 di atas, kedua anggota kelompok telah saling melakukan *code review*, mempelajari seluruh arsitektur (Task 1, Task 2, dan Task 3), serta memverifikasi eksekusi satu perintah (`python kode/run_all.py --eval_only`) agar masing-masing anggota siap menjawab pertanyaan dosen pada tugas mana pun saat ujian lisan.

---

## 2. Pernyataan Penggunaan Alat AI (AI Tools Disclosure)

Sesuai dengan ketentuan ujian, kelompok kami mendeklarasikan penggunaan alat bantu AI sebagai berikut:

- **Alat AI yang Digunakan:** Antigravity (Google DeepMind Agentic Coding Assistant) & Agentic AI rekan kelompok.
- **Bagian yang Dibantu oleh AI:**
  1. **Exploratory Data Analysis (EDA) & Pemeriksaan Statistik Data:** Membantu menulis skrip inspeksi awal untuk mengecek distribusi kelas, proporsi missing values (`NaN`), pergeseran distribusi (*domain shift*) antara Lini A dan Lini B pada Task 1, identifikasi 6 rezim operasi diskret dan 7 sensor tidak informatif pada Task 2, serta deteksi *single-bin spike* (21.84% sampel) dan pergeseran kecerahan mikroskop `M1`/`M2` pada Task 3.
  2. **Implementasi Modular PyTorch & Pipeline Reproduksibilitas:** Membantu menyusun struktur kode PyTorch (`kode/task1_classification.py`, `kode/task2_rul.py`, `kode/task3_clustering.py`, dan `kode/run_all.py`) agar seluruh angka, tabel ablasi, dan grafik pada laporan dapat direproduksi dengan satu perintah (*one-command reproduction*).
  3. **Visualisasi & Dokumentasi:** Membantu merapikan plot kurva pembelajaran (*learning curves*), matriks konfusi, interval ketidakpastian (*uncertainty intervals*), dan proyeksi UMAP/t-SNE.
- **Cara Kelompok Memverifikasi Output:**
  1. **Sanity Check (Overfitting One Batch):** Memastikan setiap arsitektur jaringan saraf mampu mencapai loss mendekati nol (akurasi 100% pada satu batch) sebelum melakukan pelatihan penuh.
  2. **Validasi Anti-Kebocoran Data (No Data Leakage):** Memverifikasi secara manual bahwa seluruh parameter normalisasi (mean, std, median, IQR) dihitung **hanya dari split data latih**, pembagian validasi Task 2 dilakukan **per mesin (`id_unit`)** bukan per baris, dan pemrosesan deret waktu dilakukan secara kausal (tanpa melihat masa depan).
  3. **Evaluasi Multi-Seed & Pemeriksaan Format Prediksi:** Menjalankan model akhir pada minimal 3 random seed berbeda (`42, 123, 999`), memverifikasi konsistensi metrik validasi (Macro-F1, RMSE, Asymmetric Score, Silhouette/ARI), serta memeriksa kecocokan header, jumlah baris, dan rentang nilai pada `prediksi_klasifikasi.csv`, `prediksi_rul.csv`, dan `cluster.csv`.

---

## 3. Persetujuan Anggota Kelompok

Kami menyatakan bahwa seluruh pekerjaan ini adalah hasil kerja kelompok kami sendiri, tidak membagikan kode atau hasil kepada kelompok lain, dan seluruh anggota menyetujui isi repositori serta laporan ini:

- [x] **Anggota 1:** Wildan Janitra Ramadhan
- [x] **Anggota 2:** (Nama Lengkap & Username GitHub Rekan)
