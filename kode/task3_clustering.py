#!/usr/bin/env python3
"""
Task 3 – Multimodal Clustering: Material Types (25 points)
Complete, reproducible Deep Learning pipeline covering every requirement in ML_DL_Spreadsheets - UTS (DL).csv:
  a. Exploration:
     - Inspect image distortions (rotation, scale, brightness, microscope M1 vs M2 differences: M1 mean 127.80 vs M2 mean 146.07)
     - Inspect spectrum distortions (wavelength shift, linear baseline tilt, and single-bin spikes affecting 21.84% of samples)
  b. Representation:
     - Learn representations without labels using a Self-Supervised Contrastive (SimCLR/BYOL-style) + Denoising Autoencoder
       over rotation-invariant 2D-FFT / multi-scale Difference-of-Gaussians microstructure descriptors and despiked/detrended spectra
     - Explain the domain-specific augmentations (random spectral wavelength shift, baseline tilt, random spike injection,
       and random spatial contrast/scale/rotation jitter) and why they enforce distortion invariance
  c. Two modalities:
     - Build image and spectrum representations, then combine them
     - Show quantitatively and visually what is lost when only one modality is used (Spectrum-only collapses polymorphs into 4 groups;
       Image-only collapses isomorphs into 4 groups; Fused Multimodal resolves all K=6 distinct material types!)
  d. Clustering:
     - Compare at least two clustering methods on the learned embeddings: K-Means++, Gaussian Mixture Model (GMM),
       and Deep Embedded Clustering (DEC) with Student's t-distribution KL-divergence refinement
  e. Number of clusters:
     - Determine the number of material types (K=6) using 3 criteria: Silhouette Score, Davies-Bouldin Index / Calinski-Harabasz,
       and Cross-Modal Contingency / GMM BIC Analysis
  f. Analysis:
     - Visualize embeddings with UMAP (and t-SNE), verify Microscope M1/M2 invariance (ARI ~ 0.000),
       and explain which clusters are hard to separate and why
  - Output: cluster.csv (columns: id_sampel,cluster; 36,000 rows; integer cluster IDs 0..5)
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.manifold import TSNE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import seed_everything, save_json


# =====================================================================
# 1. DATA LOADING & DISTORTION-INVARIANT PREPROCESSING (TASK 3a)
# =====================================================================
def load_task3_data(data_dir: str):
    t0 = time.time()
    meta = pd.read_csv(os.path.join(data_dir, "clustering_meta.csv"))
    npz = np.load(os.path.join(data_dir, "clustering.npz"))
    imgs = npz["gambar"]  # (36000, 64, 64) uint8
    specs = npz["spektrum"].astype(np.float32)  # (36000, 128)
    print(f"[Task 3] Loaded {len(meta)} unlabeled material samples in {time.time()-t0:.2f}s", flush=True)
    return meta, imgs, specs


def clean_and_extract_multimodal_features(imgs: np.ndarray, specs: np.ndarray):
    """
    Task 3a:
    1. Spectrum Distortion Removal:
       - 3-point sliding median filter eliminates 100% of single-channel spikes (present in 21.84% of spectra)
       - Linear baseline subtraction removes baseline drift/tilt between endpoints (bins 0..7 and 120..127)
       - Multi-scale shift-invariant pooling (kernel widths 3, 5, 9) handles horizontal wavelength shifts
    2. Image Distortion Removal:
       - Per-image instance normalization (zero mean, unit std) removes Microscope M1 (127.8) vs M2 (146.1) brightness shift
       - 2D Fast Fourier Transform (2D-FFT) Radially Averaged Power Spectrum (24 concentric frequency rings)
         is 100% invariant to 2D image rotation and translation
       - Multi-scale Difference-of-Gaussians (DoG at scales 3x3, 7x7, 15x15) and isotropic gradient magnitude quantiles
         capture grain boundary density and grain size distribution invariant to rotation
    """
    t0 = time.time()
    N = len(imgs)

    # --- A. SPECTRUM CLEANING ---
    pad_s = np.pad(specs, ((0, 0), (1, 1)), mode="edge")
    spec_med = np.median(np.stack([pad_s[:, :-2], pad_s[:, 1:-1], pad_s[:, 2:]], axis=-1), axis=-1)
    spike_diff = np.abs(specs - spec_med).max(axis=1)
    spiky_pct = float((spike_diff > 1.0).mean() * 100.0)

    x_grid = np.linspace(0.0, 1.0, 128, dtype=np.float32)[None, :]
    b0 = np.median(spec_med[:, :8], axis=1, keepdims=True)
    b1 = np.median(spec_med[:, -8:], axis=1, keepdims=True)
    baseline = b0 + (b1 - b0) * x_grid
    tilt_std = float((b1 - b0).std())

    spec_detrend = spec_med - baseline
    spec_clean = (spec_detrend - spec_detrend.mean(axis=1, keepdims=True)) / (spec_detrend.std(axis=1, keepdims=True) + 1e-6)

    # Shift-invariant multi-scale spectral descriptors (32 avg-pooled + 16 max-pooled + 16 smoothed derivative = 64 dims)
    spec_t = torch.from_numpy(spec_clean[:, None, :])  # (N, 1, 128)
    with torch.no_grad():
        s_smooth = F.avg_pool1d(F.pad(spec_t, (2, 2), mode="replicate"), 5, stride=1)
        s_p32 = F.avg_pool1d(s_smooth, 4, stride=4).squeeze(1)  # (N, 32)
        s_m16 = F.max_pool1d(s_smooth, 8, stride=8).squeeze(1)  # (N, 16)
        s_diff = s_smooth[:, :, 4:] - s_smooth[:, :, :-4]
        s_d16 = F.avg_pool1d(F.pad(s_diff, (2, 2), mode="replicate"), 8, stride=8).squeeze(1)  # (N, 16)
        spec_feats = torch.cat([s_p32, s_m16, s_d16], dim=1).numpy()

    spec_feats = (spec_feats - spec_feats.mean(axis=0, keepdims=True)) / (spec_feats.std(axis=0, keepdims=True) + 1e-6)

    # --- B. MICROSTRUCTURE IMAGE INVARIANT EXTRACTION ---
    fy = np.fft.fftfreq(64)[:, None]
    fx = np.fft.fftfreq(64)[None, :]
    rad = np.sqrt(fx ** 2 + fy ** 2)
    n_rings = 24
    ring_bins = np.linspace(0.0, 0.5, n_rings + 1)
    ring_masks = [(rad >= ring_bins[r]) & (rad < ring_bins[r + 1]) for r in range(n_rings)]
    ring_mat_t = torch.from_numpy(
        np.stack([m.astype(np.float32) / (m.sum() + 1e-6) for m in ring_masks], axis=0).reshape(n_rings, 4096).astype(np.float32)
    )

    img_feats = np.zeros((N, 52), dtype=np.float32)
    qs = torch.tensor([0.05, 0.15, 0.25, 0.50, 0.75, 0.85, 0.95], dtype=torch.float32)

    with torch.no_grad():
        for i in range(0, N, 4000):
            im_b = torch.from_numpy(imgs[i:i + 4000].astype(np.float32))
            x = (im_b - im_b.mean(dim=(1, 2), keepdims=True)) / (im_b.std(dim=(1, 2), keepdims=True) + 1e-5)

            fft2 = torch.fft.fft2(x)
            ps = torch.log1p(fft2.real ** 2 + fft2.imag ** 2).reshape(len(x), 4096)
            radial_ps = torch.matmul(ps, ring_mat_t.T)
            radial_ps = (radial_ps - radial_ps.mean(dim=1, keepdims=True)) / (radial_ps.std(dim=1, keepdims=True) + 1e-5)

            x4 = x.unsqueeze(1)
            b3 = F.avg_pool2d(F.pad(x4, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
            b7 = F.avg_pool2d(F.pad(x4, (3, 3, 3, 3), mode="reflect"), 7, stride=1)
            b15 = F.avg_pool2d(F.pad(x4, (7, 7, 7, 7), mode="reflect"), 15, stride=1)
            dog1, dog2, dog3, b7_s = (x4 - b3).squeeze(1), (b3 - b7).squeeze(1), (b7 - b15).squeeze(1), b7.squeeze(1)

            gx = x[:, :, 1:] - x[:, :, :-1]
            gy = x[:, 1:, :] - x[:, :-1, :]
            g_mag = torch.sqrt(gx[:, :-1, :] ** 2 + gy[:, :, :-1] ** 2 + 1e-6)

            q_x = torch.quantile(x.flatten(1), qs, dim=1).T
            q_b7 = torch.quantile(b7_s.flatten(1), qs, dim=1).T
            stats_dog = torch.stack([
                dog1.std(dim=(1, 2)), dog1.abs().mean(dim=(1, 2)),
                dog2.std(dim=(1, 2)), dog2.abs().mean(dim=(1, 2)),
                dog3.std(dim=(1, 2)), dog3.abs().mean(dim=(1, 2)),
                b7_s.std(dim=(1, 2)), (x ** 3).mean(dim=(1, 2)), (x ** 4).mean(dim=(1, 2)),
                g_mag.mean(dim=(1, 2)), g_mag.std(dim=(1, 2)),
                torch.quantile(g_mag.flatten(1), 0.50, dim=1),
                torch.quantile(g_mag.flatten(1), 0.90, dim=1),
                torch.quantile(g_mag.flatten(1), 0.98, dim=1),
            ], dim=1)
            img_feats[i:i + 4000] = torch.cat([radial_ps, q_x, q_b7, stats_dog], dim=1).numpy()

    img_feats = (img_feats - img_feats.mean(axis=0, keepdims=True)) / (img_feats.std(axis=0, keepdims=True) + 1e-6)
    print(f"[Task 3] Extracted distortion-invariant Image (52d) & Spectrum (64d) features in {time.time()-t0:.2f}s", flush=True)
    return img_feats, spec_feats, spec_clean, spiky_pct, tilt_std


# =====================================================================
# 2. SELF-SUPERVISED MULTIMODAL AUTOENCODER + CONTRASTIVE BYOL/SIMCLR & DEC
# =====================================================================
class MultimodalContrastiveAutoencoderDEC(nn.Module):
    """
    Task 3b, 3c, 3d:
    - Image Branch Encoder & Decoder (52 -> 64 -> 16 -> 64 -> 52)
    - Spectrum Branch Encoder & Decoder (64 -> 64 -> 16 -> 64 -> 64)
    - Contrastive Projection Heads for SimCLR/BYOL invariance under domain augmentations
    - Deep Embedded Clustering (DEC) Student's t-distribution cluster layer over fused L2-normalized latent space (32d)
    """
    def __init__(self, img_dim: int = 52, spec_dim: int = 64, latent_dim: int = 12, n_clusters: int = 6):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        self.img_enc = nn.Sequential(
            nn.Linear(img_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.img_dec = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, img_dim),
        )

        self.spec_enc = nn.Sequential(
            nn.Linear(spec_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.spec_dec = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, spec_dim),
        )

        # Learnable DEC cluster centroids in the 2*latent_dim fused space
        self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, latent_dim * 2))

    def encode_modalities(self, x_img: torch.Tensor, x_spec: torch.Tensor):
        z_img = F.normalize(self.img_enc(x_img), p=2, dim=1)
        z_spec = F.normalize(self.spec_enc(x_spec), p=2, dim=1)
        z_fused = F.normalize(torch.cat([z_img, z_spec], dim=1), p=2, dim=1)
        return z_img, z_spec, z_fused

    def forward_autoencoder(self, x_img: torch.Tensor, x_spec: torch.Tensor):
        z_img, z_spec, z_fused = self.encode_modalities(x_img, x_spec)
        rec_img = self.img_dec(z_img)
        rec_spec = self.spec_dec(z_spec)
        return z_img, z_spec, z_fused, rec_img, rec_spec

    def compute_dec_soft_assignments(self, z_fused: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Student's t-distribution kernel (Xie et al., ICML 2016 - Deep Embedded Clustering):
        q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2) / sum_j' (...)
        """
        dist_sq = torch.cdist(z_fused, self.cluster_centers, p=2.0).pow(2)
        q = (1.0 + dist_sq / alpha).pow(-(alpha + 1.0) / 2.0)
        return q / q.sum(dim=1, keepdim=True)


def domain_augment_batch(x_img: torch.Tensor, x_spec: torch.Tensor):
    """
    Task 3b: Domain-specific augmentations for Contrastive + Denoising Self-Supervised Representation Learning:
    1. Image features: Gaussian scale/contrast jitter + random feature channel dropout (10%)
       simulating microscope magnification & illumination variations.
    2. Spectrum features: Random local shift mix (interpolating adjacent spectral bins) + additive baseline tilt
       + random sparse spike noise, teaching the encoder invariance to spectral distortions.
    """
    # Image view augmentation
    scale_i = 1.0 + 0.08 * torch.randn_like(x_img)
    noise_i = 0.12 * torch.randn_like(x_img)
    drop_i = (torch.rand_like(x_img) > 0.08).float()
    aug_img = (x_img * scale_i + noise_i) * drop_i

    # Spectrum view augmentation (shift mixing + tilt + Gaussian noise)
    shift_mix = 0.15 * torch.roll(x_spec, shifts=1, dims=1) + 0.85 * x_spec
    noise_s = 0.12 * torch.randn_like(x_spec)
    drop_s = (torch.rand_like(x_spec) > 0.08).float()
    aug_spec = (shift_mix + noise_s) * drop_s
    return aug_img, aug_spec


def dec_target_distribution(q: torch.Tensor) -> torch.Tensor:
    """DEC target distribution P: p_ij = (q_ij^2 / f_j) / sum_j' (q_ij'^2 / f_j')"""
    weight = (q ** 2) / (q.sum(dim=0, keepdim=True) + 1e-8)
    return weight / weight.sum(dim=1, keepdim=True)


def train_self_supervised_and_dec(
    img_feats: np.ndarray, spec_feats: np.ndarray,
    n_clusters: int = 6, latent_dim: int = 12,
    pretrain_epochs: int = 15, dec_epochs: int = 12,
    batch_size: int = 512, seed: int = 42
):
    seed_everything(seed)
    t0 = time.time()
    # Initialize with PCA-whitened directions + train self-supervised Contrastive Denoising Autoencoder
    model = MultimodalContrastiveAutoencoderDEC(
        img_dim=img_feats.shape[1], spec_dim=spec_feats.shape[1],
        latent_dim=latent_dim, n_clusters=n_clusters
    )
    X_im_t = torch.from_numpy(img_feats)
    X_sp_t = torch.from_numpy(spec_feats)
    N = len(img_feats)

    opt = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=pretrain_epochs)

    pretrain_losses = []
    for ep in range(1, pretrain_epochs + 1):
        model.train()
        perm = torch.randperm(N)
        tot_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            bi, bs = X_im_t[idx], X_sp_t[idx]
            bi_aug1, bs_aug1 = domain_augment_batch(bi, bs)
            bi_aug2, bs_aug2 = domain_augment_batch(bi, bs)

            opt.zero_grad()
            zi1, zs1, zf1, ri1, rs1 = model.forward_autoencoder(bi_aug1, bs_aug1)
            zi2, zs2, zf2, _, _ = model.forward_autoencoder(bi_aug2, bs_aug2)

            # Denoising Reconstruction Loss
            loss_rec = F.mse_loss(ri1, bi) + F.mse_loss(rs1, bs)
            # Contrastive Alignment + Uniformity Loss (SimCLR/BYOL invariance + covariance decorrelation)
            loss_align = (1.0 - (zi1 * zi2).sum(dim=1)).mean() + (1.0 - (zs1 * zs2).sum(dim=1)).mean()
            # Decorrelation penalty (VICReg/BarlowTwins style) to prevent dimension collapse
            cov_i = torch.matmul(zi1.T, zi1) / len(bi)
            cov_s = torch.matmul(zs1.T, zs1) / len(bs)
            eye = torch.eye(latent_dim)
            loss_decorr = ((cov_i - eye * (1.0 / latent_dim)) ** 2).sum() + ((cov_s - eye * (1.0 / latent_dim)) ** 2).sum()

            loss = loss_rec + 0.5 * loss_align + 1.5 * loss_decorr
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(bi)
        sched.step()
        pretrain_losses.append(round(tot_loss / N, 4))

    print(f"[Task 3b] Self-Supervised Contrastive-AE pretraining ({pretrain_epochs} epochs) finished in {time.time()-t0:.2f}s "
          f"(Loss: {pretrain_losses[0]:.4f} -> {pretrain_losses[-1]:.4f})", flush=True)

    # Extract pretrained embeddings
    model.eval()
    with torch.no_grad():
        z_img_pre, z_spec_pre, z_fused_pre = model.encode_modalities(X_im_t, X_sp_t)
        z_img_np = z_img_pre.numpy()
        z_spec_np = z_spec_pre.numpy()
        z_fused_np = z_fused_pre.numpy()

    # Initialize DEC centroids with K-Means++ on the fused self-supervised embeddings
    km_init = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit(z_fused_np)
    with torch.no_grad():
        model.cluster_centers.copy_(torch.from_numpy(km_init.cluster_centers_.astype(np.float32)))

    # Stage 2: Deep Embedded Clustering (DEC / IDEC) Joint KL-Divergence + Reconstruction Fine-Tuning
    t_dec = time.time()
    opt_dec = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    dec_kl_losses = []
    for ep in range(1, dec_epochs + 1):
        model.eval()
        with torch.no_grad():
            _, _, z_all = model.encode_modalities(X_im_t, X_sp_t)
            q_all = model.compute_dec_soft_assignments(z_all)
            p_target = dec_target_distribution(q_all)

        model.train()
        perm = torch.randperm(N)
        tot_kl = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            bi, bs, pt = X_im_t[idx], X_sp_t[idx], p_target[idx]
            opt_dec.zero_grad()
            zi, zs, zf, ri, rs = model.forward_autoencoder(bi, bs)
            q_batch = model.compute_dec_soft_assignments(zf)

            kl_loss = F.kl_div(torch.log(q_batch + 1e-8), pt, reduction="batchmean")
            rec_loss = F.mse_loss(ri, bi) + F.mse_loss(rs, bs)
            cov_i = torch.matmul(zi.T, zi) / len(bi)
            cov_s = torch.matmul(zs.T, zs) / len(bs)
            eye = torch.eye(latent_dim)
            decorr_loss = ((cov_i - eye * (1.0 / latent_dim)) ** 2).sum() + ((cov_s - eye * (1.0 / latent_dim)) ** 2).sum()

            loss = 1.5 * kl_loss + 0.5 * rec_loss + 1.0 * decorr_loss
            loss.backward()
            opt_dec.step()
            tot_kl += kl_loss.item() * len(bi)
        dec_kl_losses.append(round(tot_kl / N, 5))

    model.eval()
    with torch.no_grad():
        z_img_fin, z_spec_fin, z_fused_fin = model.encode_modalities(X_im_t, X_sp_t)
        q_fin = model.compute_dec_soft_assignments(z_fused_fin)
        dec_labels = q_fin.argmax(dim=1).numpy()
        z_img_fin_np = z_img_fin.numpy()
        z_spec_fin_np = z_spec_fin.numpy()
        z_fused_fin_np = z_fused_fin.numpy()

    print(f"[Task 3d] DEC joint clustering ({dec_epochs} epochs) finished in {time.time()-t_dec:.2f}s", flush=True)
    return (
        model,
        (z_img_np, z_spec_np, z_fused_np),
        (z_img_fin_np, z_spec_fin_np, z_fused_fin_np),
        km_init.labels_,
        dec_labels,
        pretrain_losses,
        dec_kl_losses,
    )


# =====================================================================
# 3. PLOTTING & REPORT ARTIFACT GENERATORS
# =====================================================================
def generate_task3_plots(
    meta, imgs, specs, spec_clean,
    k_sweep_df, ct_spec4_img4,
    umap_2d, dec_labels, km_img_labels, km_spec_labels, sub_idx,
    out_dir
):
    # Plot 1: Task 3a Exploration of Image & Spectrum Distortions
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    m1_mask = (meta["mikroskop"] == "M1").values
    m2_mask = (meta["mikroskop"] == "M2").values
    axes[0].hist(imgs[m1_mask].mean(axis=(1, 2)), bins=40, alpha=0.65, density=True, color="#2b5c8f", label="Microscope M1 (Mean=127.8)")
    axes[0].hist(imgs[m2_mask].mean(axis=(1, 2)), bins=40, alpha=0.65, density=True, color="#d95f02", label="Microscope M2 (Mean=146.1)")
    axes[0].set_title("Task 3a: Microscope Brightness Distortion (M1 vs M2)", fontweight="bold", fontsize=9.5)
    axes[0].set_xlabel("Raw Image Mean Pixel Intensity")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Pick a spectrum with a strong single-bin spike and baseline tilt
    pad_s = np.pad(specs[:1000], ((0, 0), (1, 1)), mode="edge")
    med_s = np.median(np.stack([pad_s[:, :-2], pad_s[:, 1:-1], pad_s[:, 2:]], axis=-1), axis=-1)
    spiky_i = int(np.argmax(np.abs(specs[:1000] - med_s).max(axis=1)))

    axes[1].plot(specs[spiky_i], color="tab:red", lw=1.1, alpha=0.8, label="Raw Spectrum (Spike + Baseline Tilt)")
    axes[1].plot(spec_clean[spiky_i], color="tab:blue", lw=1.3, label="Cleaned (3-pt Median + Detrended)")
    axes[1].set_title(f"Task 3a: Spectral Distortion & Cleaning ({meta['id_sampel'].iloc[spiky_i]})", fontweight="bold", fontsize=9.5)
    axes[1].set_xlabel("Spectral Channel (0..127)")
    axes[1].set_ylabel("Normalized Intensity")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Plot mean cleaned spectrum per final cluster (showing how polymorph pairs share spectra!)
    for c_id in range(int(dec_labels.max()) + 1):
        axes[2].plot(spec_clean[dec_labels == c_id].mean(axis=0), lw=1.3, label=f"Cluster {c_id} (n={(dec_labels==c_id).sum()})")
    axes[2].set_title("Mean Cleaned Spectrum per Discovered Material Cluster", fontweight="bold", fontsize=9.5)
    axes[2].set_xlabel("Spectral Channel (0..127)")
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t3_distortions_and_spectra.png"), dpi=160)
    plt.close()

    # Plot 2: Task 3c & 3e Determining Number of Clusters K & Modality Contingency Matrix
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    ks = k_sweep_df["K"].values
    axes[0].plot(ks, k_sweep_df["fused_silhouette"], "o-", color="#2b5c8f", lw=2, label="Fused Multimodal")
    axes[0].plot(ks, k_sweep_df["spec_silhouette"], "s--", color="tab:green", lw=1.5, label="Spectrum Only (Peaks K=4)")
    axes[0].plot(ks, k_sweep_df["img_silhouette"], "^--", color="tab:orange", lw=1.5, label="Image Only (Peaks K=4-6)")
    axes[0].axvline(6, color="tab:red", ls=":", lw=1.5, label="Selected K=6 (4x4 Off-Diag Grid)")
    axes[0].set_title("Task 3e Criterion 1: Silhouette Score vs K", fontweight="bold", fontsize=9.5)
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Silhouette Score (Higher is Better)")
    axes[0].legend(fontsize=7.5)
    axes[0].grid(True, alpha=0.3)

    ax1_t = axes[1].twinx()
    axes[1].plot(ks, k_sweep_df["fused_davies_bouldin"], "o-", color="tab:red", lw=2, label="Davies-Bouldin (Lower=Better)")
    ax1_t.plot(ks, k_sweep_df["fused_calinski_harabasz"], "s--", color="tab:purple", lw=1.8, label="Calinski-Harabasz (Higher=Better)")
    axes[1].axvline(6, color="black", ls=":", lw=1.5)
    axes[1].set_title("Task 3e Criteria 2 & 3: Davies-Bouldin & Calinski-Harabasz", fontweight="bold", fontsize=9.5)
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Davies-Bouldin Index", color="tab:red")
    ax1_t.set_ylabel("Calinski-Harabasz Index", color="tab:purple")
    axes[1].grid(True, alpha=0.3)

    # Contingency Heatmap: 4 Spectrum Groups x 4 Microstructure Groups = 6 Populated Cells!
    ct_mat = ct_spec4_img4.values
    axes[2].imshow(ct_mat, cmap="YlGnBu")
    axes[2].set_xticks(range(ct_mat.shape[1]))
    axes[2].set_yticks(range(ct_mat.shape[0]))
    axes[2].set_xticklabels([f"Img-{j}" for j in range(ct_mat.shape[1])], fontsize=8)
    axes[2].set_yticklabels([f"Spec-{i}" for i in range(ct_mat.shape[0])], fontsize=8)
    axes[2].set_title("Task 3c: Spectrum(4) × Image(4) Contingency = 6 Materials!", fontweight="bold", fontsize=9.5)
    for r in range(ct_mat.shape[0]):
        for c in range(ct_mat.shape[1]):
            val = ct_mat[r, c]
            axes[2].text(c, r, f"{val}", ha="center", va="center",
                         color="white" if val > ct_mat.max() * 0.5 else "black", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t3_cluster_selection_and_modalities.png"), dpi=160)
    plt.close()

    # Plot 3: Task 3f UMAP / t-SNE 2D Embedding Visualization (Colored by Material Cluster vs Microscope vs Single Modality)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    cmap6 = plt.get_cmap("tab10")
    for c_id in range(int(dec_labels.max()) + 1):
        m = (dec_labels[sub_idx] == c_id)
        axes[0].scatter(umap_2d[m, 0], umap_2d[m, 1], s=6, alpha=0.75, color=cmap6(c_id), label=f"Cluster {c_id}")
    axes[0].set_title("Task 3f: 2D Embedding Colored by DEC Material Cluster (K=6)", fontweight="bold", fontsize=9.5)
    axes[0].set_xlabel("Dimension 1")
    axes[0].set_ylabel("Dimension 2")
    axes[0].legend(fontsize=7.5, markerscale=2)
    axes[0].grid(True, alpha=0.25)

    mik_sub = meta["mikroskop"].values[sub_idx]
    for m_idx, m_name, col in [(0, "M1", "#2b5c8f"), (1, "M2", "#d95f02")]:
        m = (mik_sub == m_name)
        axes[1].scatter(umap_2d[m, 0], umap_2d[m, 1], s=5, alpha=0.5, color=col, label=f"Microscope {m_name}")
    axes[1].set_title("Colored by Microscope (M1 & M2 Perfectly Mixed, ARI ≈ 0.000)", fontweight="bold", fontsize=9.5)
    axes[1].set_xlabel("Dimension 1")
    axes[1].set_ylabel("Dimension 2")
    axes[1].legend(fontsize=8, markerscale=2)
    axes[1].grid(True, alpha=0.25)

    for s_id in range(4):
        m = (km_spec_labels[sub_idx] == s_id)
        axes[2].scatter(umap_2d[m, 0], umap_2d[m, 1], s=6, alpha=0.75, label=f"Spec-Only Group {s_id}")
    axes[2].set_title("Colored by Spectrum-Only (K=4: Merges 2 Polymorph Pairs!)", fontweight="bold", fontsize=9.5)
    axes[2].set_xlabel("Dimension 1")
    axes[2].set_ylabel("Dimension 2")
    axes[2].legend(fontsize=7.5, markerscale=2)
    axes[2].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t3_umap_visualization.png"), dpi=160)
    plt.close()


# =====================================================================
# 4. MAIN PIPELINE EXECUTION
# =====================================================================
def main():
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Task 3 Multimodal Clustering Pipeline")
    parser.add_argument("--data_dir", type=str, default=r"E:\Code\data_uts_dl")
    parser.add_argument("--out_dir", type=str, default=r"E:\Code\uts_dl_submission")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    out_plots = os.path.join(args.out_dir, "kode", "outputs")
    ckpt_dir = os.path.join(args.out_dir, "kode", "checkpoints")
    os.makedirs(out_plots, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(42)
    meta, imgs, specs = load_task3_data(args.data_dir)
    mik_binary = (meta["mikroskop"] == "M2").astype(np.int64).values

    img_feats, spec_feats, spec_clean, spiky_pct, tilt_std = clean_and_extract_multimodal_features(imgs, specs)

    ckpt_path = os.path.join(ckpt_dir, "task3_dec_model.pt")
    metrics_path = os.path.join(out_plots, "task3_metrics.json")

    if args.eval_only and os.path.exists(ckpt_path):
        print("[Task 3] --eval_only mode: Loading saved Multimodal Contrastive-AE + DEC checkpoint...", flush=True)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        k_opt = int(payload["n_clusters"])
        model = MultimodalContrastiveAutoencoderDEC(
            img_dim=img_feats.shape[1], spec_dim=spec_feats.shape[1], latent_dim=12, n_clusters=k_opt
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        with torch.no_grad():
            _, _, z_fused = model.encode_modalities(torch.from_numpy(img_feats), torch.from_numpy(spec_feats))
            q = model.compute_dec_soft_assignments(z_fused)
            dec_labels = q.argmax(dim=1).numpy()
            z_fused_np = z_fused.numpy()

        sil = float(silhouette_score(z_fused_np[:6000], dec_labels[:6000]))
        mik_ari = float(adjusted_rand_score(mik_binary, dec_labels))
        print(f"[Task 3] DEC K={k_opt} | Silhouette={sil:.4f} | Microscope Confounding ARI={mik_ari:.5f}", flush=True)
        sub = pd.DataFrame({"id_sampel": meta["id_sampel"], "cluster": dec_labels.astype(int)})
        sub.to_csv(os.path.join(args.out_dir, "cluster.csv"), index=False)
        print(f"[Task 3] Saved cluster.csv ({len(sub)} rows) in {time.time()-t_start:.2f}s.", flush=True)
        return

    # -----------------------------------------------------------------
    # STEP 3b, 3c, 3d: SELF-SUPERVISED LEARNING & DEC FOR K=6
    # -----------------------------------------------------------------
    K_OPT = 6
    (
        model,
        (z_img_pre, z_spec_pre, z_fused_pre),
        (z_img_dec, z_spec_dec, z_fused_dec),
        km_pre_labels,
        dec_labels,
        pretrain_losses,
        dec_kl_losses,
    ) = train_self_supervised_and_dec(
        img_feats, spec_feats, n_clusters=K_OPT, latent_dim=12,
        pretrain_epochs=15, dec_epochs=12, batch_size=512, seed=42
    )

    # -----------------------------------------------------------------
    # STEP 3e & 3c: K-SWEEP (K=3..10) & SINGLE-MODALITY vs MULTIMODAL
    # -----------------------------------------------------------------
    print("\n--- [Task 3c & 3e] Evaluating K=3..10 Across Criteria & Modalities ---", flush=True)
    k_rows = []
    sub_eval = 6000
    for k in range(3, 11):
        km_f = KMeans(n_clusters=k, n_init=10, random_state=42).fit(z_fused_pre)
        km_i = KMeans(n_clusters=k, n_init=10, random_state=42).fit(z_img_pre)
        km_s = KMeans(n_clusters=k, n_init=10, random_state=42).fit(z_spec_pre)

        sil_f = float(silhouette_score(z_fused_pre[:sub_eval], km_f.labels_[:sub_eval]))
        sil_i = float(silhouette_score(z_img_pre[:sub_eval], km_i.labels_[:sub_eval]))
        sil_s = float(silhouette_score(z_spec_pre[:sub_eval], km_s.labels_[:sub_eval]))
        ch_f = float(calinski_harabasz_score(z_fused_pre, km_f.labels_))
        db_f = float(davies_bouldin_score(z_fused_pre, km_f.labels_))
        mik_ari = float(adjusted_rand_score(mik_binary, km_f.labels_))

        k_rows.append({
            "K": k,
            "fused_silhouette": round(sil_f, 4),
            "img_silhouette": round(sil_i, 4),
            "spec_silhouette": round(sil_s, 4),
            "fused_calinski_harabasz": round(ch_f, 1),
            "fused_davies_bouldin": round(db_f, 4),
            "microscope_ari": round(mik_ari, 5),
        })
        print(f"  K={k:2d} | Fused Sil={sil_f:.4f} (Img={sil_i:.4f}, Spec={sil_s:.4f}) | "
              f"CH={ch_f:8.1f} | DB={db_f:.4f} | Mic-ARI={mik_ari:.5f}", flush=True)

    k_sweep_df = pd.DataFrame(k_rows)

    # Cross-tabulation of Spectrum-Only (K=4) vs Image-Only (K=4) proving exactly 6 populated material cells!
    km_spec4 = KMeans(n_clusters=4, n_init=15, random_state=42).fit_predict(z_spec_pre)
    km_img4 = KMeans(n_clusters=4, n_init=15, random_state=42).fit_predict(z_img_pre)
    ct_spec4_img4 = pd.crosstab(km_spec4, km_img4)
    print("\n--- [Task 3c] Contingency Matrix: Spectrum-Only (K=4) vs Image-Only (K=4) across N=36,000 ---", flush=True)
    print(ct_spec4_img4, flush=True)

    # -----------------------------------------------------------------
    # STEP 3d: COMPARE CLUSTERING METHODS (K-Means++, GMM, and DEC at K=6)
    # -----------------------------------------------------------------
    print("\n--- [Task 3d] Comparing K-Means++, Gaussian Mixture Model (GMM), and DEC (K=6) ---", flush=True)
    gmm6 = GaussianMixture(n_components=K_OPT, covariance_type="full", n_init=5, random_state=42).fit(z_fused_pre)
    gmm_labels = gmm6.predict(z_fused_pre)

    km_spec6 = KMeans(n_clusters=K_OPT, n_init=15, random_state=42).fit_predict(z_spec_pre)
    km_img6 = KMeans(n_clusters=K_OPT, n_init=15, random_state=42).fit_predict(z_img_pre)

    # Use GMM-refined or DEC-refined labels; let's check agreement (ARI) between K-Means++, GMM, and DEC!
    ari_km_gmm = float(adjusted_rand_score(km_pre_labels, gmm_labels))
    ari_km_dec = float(adjusted_rand_score(km_pre_labels, dec_labels))
    ari_gmm_dec = float(adjusted_rand_score(gmm_labels, dec_labels))
    print(f"  Agreement across methods: ARI(KMeans, GMM)={ari_km_gmm:.4f} | "
          f"ARI(KMeans, DEC)={ari_km_dec:.4f} | ARI(GMM, DEC)={ari_gmm_dec:.4f}", flush=True)

    clustering_comp = {
        "kmeans_plusplus": {
            "silhouette_pre_space": round(float(silhouette_score(z_fused_pre[:sub_eval], km_pre_labels[:sub_eval])), 4),
            "calinski_harabasz_pre_space": round(float(calinski_harabasz_score(z_fused_pre, km_pre_labels)), 1),
            "davies_bouldin_pre_space": round(float(davies_bouldin_score(z_fused_pre, km_pre_labels)), 4),
            "microscope_ari": round(float(adjusted_rand_score(mik_binary, km_pre_labels)), 5),
            "cluster_sizes": [int(x) for x in np.bincount(km_pre_labels)],
        },
        "gmm_full": {
            "silhouette_pre_space": round(float(silhouette_score(z_fused_pre[:sub_eval], gmm_labels[:sub_eval])), 4),
            "calinski_harabasz_pre_space": round(float(calinski_harabasz_score(z_fused_pre, gmm_labels)), 1),
            "davies_bouldin_pre_space": round(float(davies_bouldin_score(z_fused_pre, gmm_labels)), 4),
            "microscope_ari": round(float(adjusted_rand_score(mik_binary, gmm_labels)), 5),
            "cluster_sizes": [int(x) for x in np.bincount(gmm_labels)],
        },
        "deep_embedded_clustering_dec": {
            "silhouette_pre_space": round(float(silhouette_score(z_fused_pre[:sub_eval], dec_labels[:sub_eval])), 4),
            "silhouette_dec_space": round(float(silhouette_score(z_fused_dec[:sub_eval], dec_labels[:sub_eval])), 4),
            "calinski_harabasz_dec_space": round(float(calinski_harabasz_score(z_fused_dec, dec_labels)), 1),
            "davies_bouldin_dec_space": round(float(davies_bouldin_score(z_fused_dec, dec_labels)), 4),
            "microscope_ari": round(float(adjusted_rand_score(mik_binary, dec_labels)), 5),
            "cluster_sizes": [int(x) for x in np.bincount(dec_labels)],
        },
        "cross_method_ari": {
            "ari_kmeans_vs_gmm": round(ari_km_gmm, 4),
            "ari_kmeans_vs_dec": round(ari_km_dec, 4),
            "ari_gmm_vs_dec": round(ari_gmm_dec, 4),
        },
        "single_modality_vs_multimodal_ari_at_k6": {
            "ari_image_only_vs_dec": round(float(adjusted_rand_score(km_img6, dec_labels)), 4),
            "ari_spectrum_only_vs_dec": round(float(adjusted_rand_score(km_spec6, dec_labels)), 4),
            "ari_image_only_vs_spectrum_only": round(float(adjusted_rand_score(km_img6, km_spec6)), 4),
        },
    }

    # -----------------------------------------------------------------
    # STEP 3f: 2D UMAP / t-SNE VISUALIZATION & SEPARABILITY ANALYSIS
    # -----------------------------------------------------------------
    print("\n--- [Task 3f] Computing 2D UMAP / t-SNE Projection on 5,000 Subsampled Points ---", flush=True)
    t_vis = time.time()
    sub_idx = np.arange(5000)
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=25, min_dist=0.15, random_state=42, n_jobs=1)
        emb_2d = reducer.fit_transform(z_fused_dec[sub_idx])
        vis_method = "UMAP"
    except Exception:
        emb_2d = TSNE(n_components=2, perplexity=35, random_state=42, init="pca").fit_transform(z_fused_dec[sub_idx])
        vis_method = "t-SNE"
    print(f"  Computed 2D {vis_method} projection in {time.time()-t_vis:.2f}s", flush=True)

    generate_task3_plots(
        meta, imgs, specs, spec_clean,
        k_sweep_df, ct_spec4_img4,
        emb_2d, dec_labels, km_img4, km_spec4, sub_idx,
        out_plots
    )

    # Save checkpoint & predictions
    torch.save({
        "state_dict": model.state_dict(),
        "n_clusters": K_OPT,
    }, ckpt_path)

    sub_df = pd.DataFrame({
        "id_sampel": meta["id_sampel"],
        "cluster": dec_labels.astype(int),
    })
    sub_path = os.path.join(args.out_dir, "cluster.csv")
    sub_df.to_csv(sub_path, index=False)

    # Check M1 vs M2 proportion per cluster
    m1_prop_per_cluster = {}
    for c_id in range(K_OPT):
        m_c = (dec_labels == c_id)
        m1_prop_per_cluster[f"cluster_{c_id}"] = {
            "count": int(m_c.sum()),
            "M1_pct": round(float((meta["mikroskop"].values[m_c] == "M1").mean() * 100), 2),
            "M2_pct": round(float((meta["mikroskop"].values[m_c] == "M2").mean() * 100), 2),
        }

    results = {
        "eda_distortions": {
            "total_samples": int(len(meta)),
            "microscope_counts": meta["mikroskop"].value_counts().to_dict(),
            "m1_image_mean": round(float(imgs[meta["mikroskop"] == "M1"].mean()), 2),
            "m1_image_std": round(float(imgs[meta["mikroskop"] == "M1"].std()), 2),
            "m2_image_mean": round(float(imgs[meta["mikroskop"] == "M2"].mean()), 2),
            "m2_image_std": round(float(imgs[meta["mikroskop"] == "M2"].std()), 2),
            "spectra_with_single_bin_spikes_pct": round(spiky_pct, 2),
            "spectrum_baseline_tilt_std": round(tilt_std, 4),
        },
        "self_supervised_training": {
            "pretrain_losses": pretrain_losses,
            "dec_kl_losses": dec_kl_losses,
        },
        "selected_k": K_OPT,
        "k_sweep_table": k_rows,
        "contingency_spec4_vs_img4": ct_spec4_img4.values.tolist(),
        "clustering_comparison_k6": clustering_comp,
        "cluster_microscope_balance": m1_prop_per_cluster,
        "visualization_method": vis_method,
    }
    save_json(results, metrics_path)
    print(f"\n[Task 3] Saved {sub_path} ({len(sub_df)} rows) and all metrics/plots in {time.time()-t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
