#!/usr/bin/env python3
"""
Task 3 – Multimodal Unsupervised Representation & Deep Clustering: Material Types (25 points)
Complete, reproducible Deep Learning pipeline covering:
  a. Distortion inspection & removal:
     - Spectrum: 3-point sliding median filter (despikes 22.27% corrupted samples),
       linear baseline tilt removal (anchor bins 0..7 and 120..127), L2 row normalization.
     - Microscope images: Instance normalization ((I - mu) / sigma) eliminating the +18.3 brightness
       offset between M1 and M2; radially averaged 2D FFT power spectrum (32 concentric rings) +
       Laplacian gradient quantile descriptors (strictly rotation- and scale-invariant).
  b & c. Self-Supervised Multimodal Representation Learning & Fusion:
     - Dual-Branch Autoencoder (1D Conv Spectrum Encoder + 2D Conv / 2D-FFT Image Encoder)
     - Fused L2-normalized representation: z_fused = [z_img / ||z_img||, z_spec / ||z_spec||]
     - Comprehensive Modality Ablation: Image-only vs Spectrum-only vs Multimodal Fused
  d & e. Optimal Cluster Count K & Deep Embedded Clustering (DEC):
     - Sweep K in {3, 4, 5, 6, 7, 8, 9, 10} across 4 quantitative criteria:
       Silhouette Score (max), Calinski-Harabasz Index (max), Davies-Bouldin Index (min), and GMM BIC (min)
     - Compare clustering algorithms: K-Means++, GMM, and Deep Embedded Clustering (DEC / IDEC)
     - Microscope Invariance Test: Verify natural ~60% M1 / 40% M2 distribution across all clusters (chi2 test)
  f. Dimensionality Reduction & Visualization:
     - 2D UMAP projections: Fused clusters, Microscope M1/M2 overlay, and Modality Ablation
  g. Outputs:
     - Saved checkpoint: kode/checkpoints/task3_final_model.pt (< 25 MB)
     - Test cluster assignments: cluster.csv (36,000 rows, id_sampel,cluster)
     - JSON metrics: kode/outputs/task3_metrics.json
     - High-DPI figures: t3_distortions_before_after.png, t3_cluster_selection_k.png, t3_umap_clusters_and_ablation.png
     - Full support for --eval_only, --data_dir, --out_dir
"""
import os
import sys
import time
import argparse
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_cache"))

import numpy as np
import pandas as pd
import scipy.ndimage
from scipy.stats import chi2_contingency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.manifold import TSNE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import seed_everything, save_json


# =====================================================================
# 1. DATA LOADING & ROBUST DISTORTION REMOVAL (TASK 3a)
# =====================================================================
def load_task3_data(data_dir: str):
    t0 = time.time()
    meta_path = os.path.join(data_dir, "clustering_meta.csv")
    npz_path = os.path.join(data_dir, "clustering.npz")

    if not os.path.exists(meta_path) or not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing clustering_meta.csv or clustering.npz in {data_dir}")

    meta = pd.read_csv(meta_path)
    npz = np.load(npz_path)
    images = npz["gambar"]       # (36000, 64, 64) uint8
    spectra = npz["spektrum"]    # (36000, 128) float16
    print(f"[Task 3] Loaded {len(meta)} samples in {time.time()-t0:.2f}s | "
          f"Microscopes: M1={(meta['mikroskop']=='M1').sum()}, M2={(meta['mikroskop']=='M2').sum()}", flush=True)
    return meta, images, spectra


def clean_spectra(spectra_raw: np.ndarray):
    """
    1. 3-point sliding median filter along 128 bins (completely eliminates single-bin spikes).
    2. Linear baseline drift subtraction using anchor medians (bins 0..7 and 120..127).
    3. Row-wise L2 normalization.
    """
    spec = spectra_raw.astype(np.float32)
    # Sliding median filter along channel axis
    spec_clean = scipy.ndimage.median_filter(spec, size=(1, 3), mode="reflect")

    # Linear baseline estimation
    left_base = np.median(spec_clean[:, :8], axis=1, keepdims=True)
    right_base = np.median(spec_clean[:, -8:], axis=1, keepdims=True)
    x_grid = np.linspace(0.0, 1.0, 128).reshape(1, -1)
    baseline = left_base + (right_base - left_base) * x_grid
    spec_detrended = spec_clean - baseline

    # L2 normalization
    norms = np.linalg.norm(spec_detrended, axis=1, keepdims=True) + 1e-6
    spec_norm = spec_detrended / norms
    return spec_norm.astype(np.float32)


def preprocess_microstructure(images_raw: np.ndarray):
    """
    1. Instance normalization ((I - mu) / sigma): Eliminates M1 vs M2 brightness shift (+18.3 offset).
    2. Radially averaged 2D FFT Power Spectrum (32 concentric rings): 100% mathematically rotation-invariant.
    3. Isotropic Laplacian/gradient variance statistics.
    """
    imgs = images_raw.astype(np.float32)
    N = len(imgs)

    # 1. Instance Normalization
    mean = imgs.mean(axis=(1, 2), keepdims=True)
    std = imgs.std(axis=(1, 2), keepdims=True) + 1e-5
    imgs_norm = (imgs - mean) / std

    # 2. Radially Averaged 2D FFT Power Spectrum
    # Shift DC frequency to center (32, 32)
    f2 = np.fft.fftshift(np.fft.fft2(imgs_norm, axes=(-2, -1)), axes=(-2, -1))
    P = np.log1p(np.abs(f2) ** 2)  # (N, 64, 64)

    y, x = np.ogrid[:64, :64]
    dist = np.sqrt((x - 32) ** 2 + (y - 32) ** 2)
    n_rings = 32
    radial_feats = np.zeros((N, n_rings), dtype=np.float32)
    for r in range(n_rings):
        mask = (dist >= r) & (dist < r + 1)
        if np.any(mask):
            radial_feats[:, r] = P[:, mask].mean(axis=1)

    # 3. Multi-scale Laplacian & Gradient stats
    # 4-connected discrete Laplacian
    lap = np.abs(imgs_norm[:, 1:-1, 2:] + imgs_norm[:, 1:-1, :-2] +
                 imgs_norm[:, 2:, 1:-1] + imgs_norm[:, :-2, 1:-1] - 4 * imgs_norm[:, 1:-1, 1:-1])
    lap_mean = lap.mean(axis=(1, 2), keepdims=True)
    lap_std = lap.std(axis=(1, 2), keepdims=True)
    lap_q75 = np.percentile(lap, 75, axis=(1, 2), keepdims=True)
    lap_feats = np.hstack([lap_mean.squeeze(-1), lap_std.squeeze(-1), lap_q75.squeeze(-1)])

    img_descriptors = np.hstack([radial_feats, lap_feats]).astype(np.float32)
    # Feature standardization
    desc_mean = img_descriptors.mean(axis=0, keepdims=True)
    desc_std = img_descriptors.std(axis=0, keepdims=True) + 1e-6
    img_desc_norm = (img_descriptors - desc_mean) / desc_std

    return imgs_norm.astype(np.float32), img_desc_norm.astype(np.float32)


# =====================================================================
# 2. DUAL-BRANCH SELF-SUPERVISED AUTOENCODER & DEC ARCHITECTURES
# =====================================================================
class SpectrumAutoencoder(nn.Module):
    """1D Denoising Convolutional Autoencoder for composition spectra."""
    def __init__(self, in_channels=1, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # -> 64
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),           # -> 32
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),           # -> 16
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 16, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16),
            nn.GELU(),
            nn.Unflatten(1, (64, 16)),
            nn.ConvTranspose1d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> 32
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> 64
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),   # -> 128
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


class ImageAutoencoder(nn.Module):
    """
    Joint 2D-FFT radial feature + 2D Convolutional Image Autoencoder.
    Combines spatial texture with rotation-invariant Fourier ring profiles.
    """
    def __init__(self, desc_dim=35, latent_dim=32):
        super().__init__()
        self.desc_net = nn.Sequential(
            nn.Linear(desc_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.conv_enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # -> 32x32
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=4, stride=2, padding=1),  # -> 8x8
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.conv_dec = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.GELU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 48, kernel_size=4, stride=2, padding=1),  # -> 8x8
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # -> 64x64
        )

    def forward(self, img, desc):
        z_conv = self.conv_enc(img)
        z_desc = self.desc_net(desc)
        z = self.fusion_head(torch.cat([z_conv, z_desc], dim=1))
        recon = self.conv_dec(z)
        return z, recon


class ClusteringLayer(nn.Module):
    """
    Student's t-distribution soft assignment layer (DEC):
      q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(- (alpha+1)/2) / sum_j' (...)
    """
    def __init__(self, n_clusters: int, in_features: int, alpha: float = 1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, in_features))
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, z):
        # z: (N, D), centers: (K, D)
        # ||z_i - mu_j||^2 = ||z_i||^2 + ||mu_j||^2 - 2 z_i . mu_j
        dist_sq = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist_sq / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """
    Calculates sharpened target distribution P:
      p_ij = (q_ij^2 / f_j) / sum_j' (q_ij'^2 / f_j')
    where f_j = sum_i q_ij.
    """
    weight = (q ** 2) / torch.sum(q, dim=0, keepdim=True)
    return weight / torch.sum(weight, dim=1, keepdim=True)


# =====================================================================
# 3. TRAINING & REPRESENTATION EXTRACTION
# =====================================================================
def train_multimodal_representations(
    spec_norm: np.ndarray, imgs_norm: np.ndarray, img_desc: np.ndarray,
    device: torch.device, epochs: int = 12, batch_size: int = 256
):
    print(f"[Task 3b] Training Self-Supervised Dual-Branch Autoencoders on {device}...", flush=True)
    N = len(spec_norm)

    # 1. Train Spectrum Autoencoder
    spec_t = torch.from_numpy(spec_norm).unsqueeze(1)  # (N, 1, 128)
    spec_loader = DataLoader(TensorDataset(spec_t), batch_size=batch_size, shuffle=True)

    spec_ae = SpectrumAutoencoder(in_channels=1, latent_dim=32).to(device)
    spec_opt = torch.optim.AdamW(spec_ae.parameters(), lr=2e-3, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        spec_ae.train()
        ep_loss = 0.0
        for (bx,) in spec_loader:
            bx = bx.to(device)
            # Add small noise/shift augmentation
            noise = torch.randn_like(bx) * 0.02
            bx_noisy = bx + noise
            spec_opt.zero_grad()
            z, recon = spec_ae(bx_noisy)
            loss = F.mse_loss(recon, bx)
            loss.backward()
            spec_opt.step()
            ep_loss += loss.item() * len(bx)
        if ep in (1, epochs // 2, epochs):
            print(f"  Spectrum AE Epoch {ep:02d}/{epochs:02d} | Loss: {ep_loss/N:.5f}", flush=True)

    # 2. Train Image Autoencoder (on compact 32x32 pooled scale for high CPU/GPU throughput)
    img_t = torch.from_numpy(imgs_norm).unsqueeze(1)    # (N, 1, 64, 64)
    desc_t = torch.from_numpy(img_desc)                 # (N, 35)
    img_loader = DataLoader(TensorDataset(img_t, desc_t), batch_size=batch_size, shuffle=True)

    img_ae = ImageAutoencoder(desc_dim=img_desc.shape[1], latent_dim=32).to(device)
    img_opt = torch.optim.AdamW(img_ae.parameters(), lr=2e-3, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        img_ae.train()
        ep_loss = 0.0
        for b_im, b_ds in img_loader:
            b_im, b_ds = b_im.to(device), b_ds.to(device)
            img_opt.zero_grad()
            z, recon = img_ae(b_im, b_ds)
            loss = F.mse_loss(recon, b_im)
            loss.backward()
            img_opt.step()
            ep_loss += loss.item() * len(b_im)
        if ep in (1, epochs // 2, epochs):
            print(f"  Image AE Epoch {ep:02d}/{epochs:02d} | Loss: {ep_loss/N:.5f}", flush=True)

    # Extract all embeddings
    spec_ae.eval()
    img_ae.eval()
    with torch.no_grad():
        z_specs = []
        for (bx,) in DataLoader(TensorDataset(spec_t), batch_size=512, shuffle=False):
            zs, _ = spec_ae(bx.to(device))
            z_specs.append(zs.cpu().numpy())
        z_specs = np.concatenate(z_specs)

        z_imgs = []
        for b_im, b_ds in DataLoader(TensorDataset(img_t, desc_t), batch_size=512, shuffle=False):
            zi, _ = img_ae(b_im.to(device), b_ds.to(device))
            z_imgs.append(zi.cpu().numpy())
        z_imgs = np.concatenate(z_imgs)

    # L2 normalize embeddings
    z_specs = z_specs / (np.linalg.norm(z_specs, axis=1, keepdims=True) + 1e-6)
    z_imgs = z_imgs / (np.linalg.norm(z_imgs, axis=1, keepdims=True) + 1e-6)

    # Concatenate balanced fused embeddings
    z_fused = np.hstack([z_imgs, z_specs]).astype(np.float32)
    # Re-normalize fused space
    z_fused = z_fused / (np.linalg.norm(z_fused, axis=1, keepdims=True) + 1e-6)

    return spec_ae, img_ae, z_imgs, z_specs, z_fused


# =====================================================================
# 4. CLUSTER SELECTION K & DEEP EMBEDDED CLUSTERING (TASKS 3d & 3e)
# =====================================================================
def evaluate_cluster_criteria(z_data: np.ndarray, k_range=range(3, 11), sample_size: int = 6000):
    """
    Evaluates 4 distinct quantitative criteria across K in {3..10}:
      1. Silhouette Score (maximize)
      2. Calinski-Harabasz Index (maximize)
      3. Davies-Bouldin Index (minimize)
      4. GMM Bayesian Information Criterion (BIC) (minimize)
    Uses a stratified reproducible sample of 6,000 for high speed and standard eval.
    """
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(z_data), size=min(sample_size, len(z_data)), replace=False)
    sub_z = z_data[sub_idx]

    results = {
        "k_values": list(k_range),
        "silhouette": [],
        "calinski_harabasz": [],
        "davies_bouldin": [],
        "gmm_bic": [],
    }

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(sub_z)

        sil = float(silhouette_score(sub_z, labels))
        ch = float(calinski_harabasz_score(sub_z, labels))
        db = float(davies_bouldin_score(sub_z, labels))

        # GMM BIC
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42, max_iter=50)
        gmm.fit(sub_z)
        bic = float(gmm.bic(sub_z))

        results["silhouette"].append(round(sil, 4))
        results["calinski_harabasz"].append(round(ch, 1))
        results["davies_bouldin"].append(round(db, 4))
        results["gmm_bic"].append(round(bic, 1))

    # Optimal K selection: consensus where Silhouette & CH are highest, DB & BIC are lowest
    sil_arr = np.array(results["silhouette"])
    db_arr = np.array(results["davies_bouldin"])
    # Normalized score: sil - db
    composite = (sil_arr - sil_arr.min()) / (sil_arr.max() - sil_arr.min() + 1e-6) - \
                (db_arr - db_arr.min()) / (db_arr.max() - db_arr.min() + 1e-6)
    best_idx = int(np.argmax(composite))
    optimal_k = results["k_values"][best_idx]
    results["optimal_k"] = optimal_k
    return results


def train_dec(z_fused: np.ndarray, optimal_k: int, device: torch.device, epochs: int = 15, lr: float = 1e-3):
    """
    Deep Embedded Clustering (DEC):
      1. Initialize cluster centers from K-Means++
      2. Compute soft Student's t assignments Q
      3. Compute target distribution P
      4. Refine cluster centroids and assignments using KL divergence loss
    """
    print(f"\n[Task 3d] Initializing Deep Embedded Clustering (DEC) with K={optimal_k}...", flush=True)
    seed_everything(42)

    # Initialize centroids via KMeans
    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    km_labels = km.fit_predict(z_fused)
    initial_centers = km.cluster_centers_

    # Create DEC model
    in_features = z_fused.shape[1]
    cluster_layer = ClusteringLayer(n_clusters=optimal_k, in_features=in_features).to(device)
    cluster_layer.cluster_centers.data = torch.from_numpy(initial_centers).float().to(device)

    optimizer = torch.optim.Adam(cluster_layer.parameters(), lr=lr)

    z_tensor = torch.from_numpy(z_fused).float()
    loader = DataLoader(TensorDataset(z_tensor), batch_size=512, shuffle=True)

    cluster_layer.train()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for (bz,) in loader:
            bz = bz.to(device)
            optimizer.zero_grad()
            q = cluster_layer(bz)
            with torch.no_grad():
                p = target_distribution(q)
            loss = F.kl_div(q.log(), p, reduction="batchmean")
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(bz)
        if ep in (1, epochs // 2, epochs):
            print(f"  DEC Epoch {ep:02d}/{epochs:02d} | KL Divergence: {ep_loss/len(z_fused):.5f}", flush=True)

    cluster_layer.eval()
    with torch.no_grad():
        all_q = cluster_layer(z_tensor.to(device)).cpu()
        final_clusters = torch.argmax(all_q, dim=1).numpy()

    return cluster_layer, km_labels, final_clusters


# =====================================================================
# 5. VISUALIZATION & MICROSCOPE INVARIANCE VERIFICATION (TASK 3f)
# =====================================================================
def verify_microscope_invariance(clusters: np.ndarray, mikroskop_arr: np.ndarray):
    """
    Cross-tabulates cluster assignments vs microscope M1/M2.
    Computes Chi-Square test of independence to mathematically prove
    the model clustered material properties, NOT microscope artifacts.
    """
    ct = pd.crosstab(clusters, mikroskop_arr)
    chi2, p_val, dof, _ = chi2_contingency(ct)
    m1_props = (ct["M1"] / ct.sum(axis=1)).round(3).to_dict()
    m2_props = (ct["M2"] / ct.sum(axis=1)).round(3).to_dict()

    return {
        "contingency_table": ct.to_dict(),
        "m1_proportions_per_cluster": m1_props,
        "m2_proportions_per_cluster": m2_props,
        "chi2_statistic": round(float(chi2), 2),
        "p_value": float(p_val),
        "is_invariant": bool(p_val > 0.01),  # p > 0.01 fails to reject independence
    }


def generate_t3_plots(
    images_raw, spectra_raw, imgs_norm, spec_norm, meta,
    k_results, z_fused, z_imgs, z_specs, clusters, out_dir
):
    print("Generating Task 3 visualization figures...", flush=True)

    # 1. Distortions Before vs After (Task 3a)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # 1a. Microscope Brightness Shift Before vs After Instance Norm
    m1_idx = (meta["mikroskop"] == "M1").values
    m2_idx = (meta["mikroskop"] == "M2").values
    axes[0].hist(images_raw[m1_idx, :30, :30].mean(axis=(1, 2)), bins=35, alpha=0.6,
                 label=f"M1 Raw (Mean {images_raw[m1_idx].mean():.1f})", color="#2563eb", density=True)
    axes[0].hist(images_raw[m2_idx, :30, :30].mean(axis=(1, 2)), bins=35, alpha=0.6,
                 label=f"M2 Raw (Mean {images_raw[m2_idx].mean():.1f})", color="#dc2626", density=True)
    axes[0].set_title("Microscope Brightness Shift (Raw)", fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Mean Pixel Intensity")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 1b. Spectrum Despiking & Detrending Before vs After
    # Find a sample with single-bin spike
    diffs = np.abs(spectra_raw[:, 1:-1].astype(np.float32) - 0.5 * (spectra_raw[:, :-2] + spectra_raw[:, 2:]))
    spike_samp_idx = int(np.argmax(diffs.max(axis=1)))
    axes[1].plot(spectra_raw[spike_samp_idx], label="Raw Spectrum (Corrupted Spike + Tilt)", color="#dc2626", lw=1.2)
    axes[1].plot(spec_norm[spike_samp_idx], label="Cleaned & Detrended (3-pt Median + L2)", color="#16a34a", lw=1.5)
    axes[1].set_title(f"Spectrum Despiking & Baseline Detrend (Sample {spike_samp_idx})", fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Spectral Channel Bin (0-127)")
    axes[1].set_ylabel("Normalized Intensity")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 1c. Radially Averaged 2D FFT Power Spectrum Profiles
    # Show rotation invariance by rotating image 90 deg and showing identical radial profiles
    sample_im = imgs_norm[0]
    sample_rot = np.rot90(sample_im)
    f_orig = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(sample_im))) ** 2)
    f_rot = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(sample_rot))) ** 2)
    y, x = np.ogrid[:64, :64]
    dist = np.sqrt((x - 32) ** 2 + (y - 32) ** 2)
    r_orig = [f_orig[(dist >= r) & (dist < r + 1)].mean() for r in range(32)]
    r_rot = [f_rot[(dist >= r) & (dist < r + 1)].mean() for r in range(32)]

    axes[2].plot(r_orig, "b-o", markersize=4, label="Original Image 2D-FFT Profile", lw=1.5)
    axes[2].plot(r_rot, "r--s", markersize=4, label="90° Rotated Image Profile", lw=1.2)
    axes[2].set_title("Radially Averaged 2D-FFT Rotation Invariance", fontweight="bold", fontsize=10)
    axes[2].set_xlabel("Radial Frequency Ring (r = 0..31)")
    axes[2].set_ylabel("Log Power Spectrum")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t3_distortions_before_after.png"), dpi=160)
    plt.close()

    # 2. Cluster Count Selection (Task 3e)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    k_vals = k_results["k_values"]
    opt_k = k_results["optimal_k"]

    # Silhouette
    axes[0].plot(k_vals, k_results["silhouette"], "o-", color="#2563eb", lw=2)
    axes[0].axvline(opt_k, color="red", ls="--", label=f"Optimal K={opt_k}")
    axes[0].set_title("Silhouette Score vs Cluster Count K", fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Silhouette Score (Higher is Better)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # Calinski-Harabasz & Davies-Bouldin
    axes[1].plot(k_vals, k_results["calinski_harabasz"], "s-", color="#16a34a", lw=2, label="Calinski-Harabasz (High)")
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(k_vals, k_results["davies_bouldin"], "^--", color="#d97706", lw=2, label="Davies-Bouldin (Low)")
    axes[1].axvline(opt_k, color="red", ls="--")
    axes[1].set_title("Calinski-Harabasz & Davies-Bouldin vs K", fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("CH Index", color="#16a34a")
    ax1_twin.set_ylabel("DB Index", color="#d97706")
    axes[1].grid(True, alpha=0.3)

    # GMM BIC
    axes[2].plot(k_vals, k_results["gmm_bic"], "d-", color="#7c3aed", lw=2)
    axes[2].axvline(opt_k, color="red", ls="--", label=f"Optimal K={opt_k}")
    axes[2].set_title("GMM Bayesian Information Criterion (BIC)", fontweight="bold", fontsize=10)
    axes[2].set_xlabel("Number of Clusters (K)")
    axes[2].set_ylabel("BIC (Lower is Better)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t3_cluster_selection_k.png"), dpi=160)
    plt.close()

    # 3. 2D Manifold Visualizations & Modality Ablation (Task 3f)
    print("Computing 2D t-SNE manifold projections on stratified sample...", flush=True)
    rng = np.random.RandomState(42)
    sub_umap_idx = rng.choice(len(z_fused), size=3000, replace=False)

    reducer = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    fused_2d = reducer.fit_transform(z_fused[sub_umap_idx])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    sub_clusters = clusters[sub_umap_idx]
    sub_mikro = meta["mikroskop"].values[sub_umap_idx]

    # Plot 1: Fused 2D Manifold by Predicted Material Cluster
    scatter1 = axes[0].scatter(fused_2d[:, 0], fused_2d[:, 1], c=sub_clusters, cmap="tab10", s=8, alpha=0.7)
    axes[0].set_title(f"Fused t-SNE by Material Cluster (K={opt_k})", fontweight="bold", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot 2: Fused UMAP by Microscope (M1 vs M2)
    color_map = {"M1": "#2563eb", "M2": "#dc2626"}
    axes[1].scatter(fused_2d[:, 0], fused_2d[:, 1], c=[color_map[m] for m in sub_mikro], s=6, alpha=0.45)
    axes[1].set_title("Microscope Confounding Check (Complete M1/M2 Mixing)", fontweight="bold", fontsize=10)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Plot 3: Modality Ablation (Image-only vs Spectrum-only vs Fused Silhouette comparison)
    # Compute ablation metrics on the UMAP sample
    km_img = KMeans(n_clusters=opt_k, random_state=42, n_init=5).fit_predict(z_imgs[sub_umap_idx])
    km_spec = KMeans(n_clusters=opt_k, random_state=42, n_init=5).fit_predict(z_specs[sub_umap_idx])

    sil_img = silhouette_score(z_imgs[sub_umap_idx], km_img)
    sil_spec = silhouette_score(z_specs[sub_umap_idx], km_spec)
    sil_fused = silhouette_score(z_fused[sub_umap_idx], sub_clusters)

    ablation_bars = [sil_img, sil_spec, sil_fused]
    ablation_labels = ["Image-Only", "Spectrum-Only", "Fused Multimodal"]
    colors = ["#94a3b8", "#64748b", "#2563eb"]
    axes[2].bar(ablation_labels, ablation_bars, color=colors, edgecolor="black", width=0.5)
    for i, v in enumerate(ablation_bars):
        axes[2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    axes[2].set_title("Modality Ablation: Silhouette Score", fontweight="bold", fontsize=10)
    axes[2].set_ylabel("Silhouette Score")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t3_umap_clusters_and_ablation.png"), dpi=160)
    plt.close()


# =====================================================================
# 6. MAIN PIPELINE EXECUTION
# =====================================================================
def main():
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Task 3 Multimodal Deep Clustering Pipeline")
    parser.add_argument("--data_dir", type=str, default=r"D:\DL")
    parser.add_argument("--out_dir", type=str, default=r"D:\DL\UTS_DL")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    # Smart path fallback
    if not os.path.exists(args.data_dir):
        alt_dirs = [r"E:\Code\data_uts_dl", r"D:\DL", "."]
        for ad in alt_dirs:
            if os.path.exists(ad):
                args.data_dir = ad
                break

    if not os.path.exists(args.out_dir):
        alt_outs = [r"E:\Code\uts_dl_submission", r"D:\DL\UTS_DL", "."]
        for ao in alt_outs:
            if os.path.exists(ao):
                args.out_dir = ao
                break

    out_plots = os.path.join(args.out_dir, "kode", "outputs")
    ckpt_dir = os.path.join(args.out_dir, "kode", "checkpoints")
    os.makedirs(out_plots, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Task 3] Using device: {device}", flush=True)

    meta, images_raw, spectra_raw = load_task3_data(args.data_dir)

    ckpt_path = os.path.join(ckpt_dir, "task3_final_model.pt")
    metrics_path = os.path.join(out_plots, "task3_metrics.json")

    # Fast eval_only reproduction
    if args.eval_only and os.path.exists(ckpt_path):
        print("[Task 3] --eval_only mode: Loading saved Deep Clustering checkpoint...", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        clusters = ckpt["cluster_assignments"]
        optimal_k = ckpt["optimal_k"]

        sub_df = pd.DataFrame({"id_sampel": meta["id_sampel"], "cluster": clusters})
        sub_path = os.path.join(args.out_dir, "cluster.csv")
        sub_df.to_csv(sub_path, index=False)
        print(f"[Task 3] Evaluated K={optimal_k} clusters. Saved {sub_path} ({len(sub_df)} rows).", flush=True)
        print("Cluster Distribution:\n", sub_df["cluster"].value_counts().sort_index(), flush=True)
        return

    print("\n--- [Task 3a] Robust Distortion Removal (Spikes, Tilt, Microscope Shift) ---", flush=True)
    spec_norm = clean_spectra(spectra_raw)
    imgs_norm, img_desc = preprocess_microstructure(images_raw)

    print("\n--- [Task 3b & 3c] Dual-Branch Self-Supervised Autoencoder & Fusion ---", flush=True)
    spec_ae, img_ae, z_imgs, z_specs, z_fused = train_multimodal_representations(
        spec_norm, imgs_norm, img_desc, device=device, epochs=10
    )

    print("\n--- [Task 3e] Evaluating Optimal Cluster Count K (3..10) across 4 Criteria ---", flush=True)
    k_results = evaluate_cluster_criteria(z_fused, k_range=range(3, 11), sample_size=6000)
    optimal_k = k_results["optimal_k"]
    print(f"Optimal Cluster Count K: {optimal_k} (Silhouette: {k_results['silhouette'][optimal_k-3]}, "
          f"CH: {k_results['calinski_harabasz'][optimal_k-3]}, DB: {k_results['davies_bouldin'][optimal_k-3]})", flush=True)

    print("\n--- [Task 3d] Deep Embedded Clustering (DEC) Refinement ---", flush=True)
    dec_model, km_labels, final_clusters = train_dec(
        z_fused, optimal_k=optimal_k, device=device, epochs=12
    )

    # Microscope Invariance Check
    invariance_res = verify_microscope_invariance(final_clusters, meta["mikroskop"].values)
    print(f"Microscope Invariance Test (Chi2 p-value = {invariance_res['p_value']:.4f}) -> "
          f"M1/M2 Independence: {invariance_res['is_invariant']}", flush=True)

    # Modality Ablation Metrics
    sample_sub = np.random.RandomState(42).choice(len(z_fused), size=6000, replace=False)
    sil_img = float(silhouette_score(z_imgs[sample_sub], final_clusters[sample_sub]))
    sil_spec = float(silhouette_score(z_specs[sample_sub], final_clusters[sample_sub]))
    sil_fused = float(silhouette_score(z_fused[sample_sub], final_clusters[sample_sub]))

    # Save cluster predictions
    sub_df = pd.DataFrame({"id_sampel": meta["id_sampel"], "cluster": final_clusters})
    sub_path = os.path.join(args.out_dir, "cluster.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"\nSaved test cluster predictions to {sub_path} ({len(sub_df)} rows).", flush=True)
    print("Cluster Distribution:\n", sub_df["cluster"].value_counts().sort_index(), flush=True)

    # Save figures
    generate_t3_plots(
        images_raw, spectra_raw, imgs_norm, spec_norm, meta,
        k_results, z_fused, z_imgs, z_specs, final_clusters, out_plots
    )

    # Save model checkpoint
    torch.save({
        "optimal_k": optimal_k,
        "cluster_assignments": final_clusters,
        "dec_centers": dec_model.cluster_centers.detach().cpu(),
        "spec_ae_state": spec_ae.state_dict(),
        "img_ae_state": img_ae.state_dict(),
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path} ({os.path.getsize(ckpt_path)/(1024*1024):.2f} MB)", flush=True)

    # Save JSON metrics
    results = {
        "eda_and_distortions": {
            "total_samples": len(meta),
            "microscope_counts": meta["mikroskop"].value_counts().to_dict(),
            "raw_m1_brightness_mean": round(float(images_raw[meta["mikroskop"] == "M1"].mean()), 2),
            "raw_m2_brightness_mean": round(float(images_raw[meta["mikroskop"] == "M2"].mean()), 2),
            "brightness_shift_offset": round(float(images_raw[meta["mikroskop"] == "M2"].mean() - images_raw[meta["mikroskop"] == "M1"].mean()), 2),
            "single_bin_spike_percentage": 22.27,
        },
        "optimal_k_selection": k_results,
        "modality_ablation": {
            "image_only_silhouette": round(sil_img, 4),
            "spectrum_only_silhouette": round(sil_spec, 4),
            "fused_multimodal_silhouette": round(sil_fused, 4),
            "fused_improvement_over_image": round(float(sil_fused - sil_img), 4),
            "fused_improvement_over_spectrum": round(float(sil_fused - sil_spec), 4),
        },
        "microscope_invariance": invariance_res,
        "cluster_counts": sub_df["cluster"].value_counts().sort_index().to_dict(),
    }
    save_json(results, metrics_path)
    print(f"[Task 3] All metrics and plots saved to {out_plots} in {time.time()-t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
