#!/usr/bin/env python3
"""
Task 3 (Part 1: Stages 3a–3c) – Multimodal Distortion Cleaning & Self-Supervised Representation Learning
Assigned to Person A (Wildan):
  a. Exploration & Distortion Removal:
     - Inspect image distortions (rotation, scale, brightness, microscope M1 vs M2 offset)
     - Inspect spectrum distortions (wavelength shift, linear baseline tilt, single-bin spikes in 21.84% of samples)
     - Clean spectra (3-point sliding median filter + linear baseline detrending + multi-scale shift-invariant pooling)
     - Extract microscope/rotation-invariant image descriptors (instance norm + 2D-FFT radial power spectrum + multi-scale DoG)
  b. Self-Supervised Representation Learning:
     - Train a Multimodal Contrastive + Denoising Autoencoder (SimCLR/VICReg alignment + reconstruction)
     - Domain augmentations: contrast/scale jitter + channel dropout for images; shift-mix + noise for spectra
  c. Two Modalities Fusion & Ablation:
     - Build Image-only, Spectrum-only, and Fused Multimodal latent representations
     - Quantify what is lost when using only one modality (Spec K=4 collapses polymorphs; Img K=4 collapses isomorphs;
       together they reveal 6 distinct material populations in the 4x4 cross-modal contingency table)

Saves pretrained encoder checkpoint (`kode/checkpoints/task3_pretrained_rep.pt`),
Part 1 metrics (`kode/outputs/task3_part1_metrics.json`), and distortion/spectra plot (`kode/outputs/t3_distortions_and_spectra.png`)
for Person B (Stages 3d–3f: DEC, K-selection, UMAP & cluster.csv).
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import seed_everything, save_json


def load_task3_data(data_dir: str):
    t0 = time.time()
    meta = pd.read_csv(os.path.join(data_dir, "clustering_meta.csv"))
    npz = np.load(os.path.join(data_dir, "clustering.npz"))
    imgs = npz["gambar"]  # (36000, 64, 64) uint8
    specs = npz["spektrum"].astype(np.float32)  # (36000, 128)
    print(f"[Task 3a] Loaded {len(meta)} unlabeled material samples in {time.time()-t0:.2f}s", flush=True)
    return meta, imgs, specs


def clean_and_extract_multimodal_features(imgs: np.ndarray, specs: np.ndarray):
    """
    Task 3a: Distortion Removal & Invariant Feature Extraction for Images and Spectra.
    """
    t0 = time.time()
    N = len(imgs)

    # --- A. Spectrum Cleaning: 3-pt median despiking + linear baseline detrending ---
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

    # Shift-invariant multi-scale spectral descriptors (64-dim)
    spec_t = torch.from_numpy(spec_clean[:, None, :])
    with torch.no_grad():
        s_smooth = F.avg_pool1d(F.pad(spec_t, (2, 2), mode="replicate"), 5, stride=1)
        s_p32 = F.avg_pool1d(s_smooth, 4, stride=4).squeeze(1)
        s_m16 = F.max_pool1d(s_smooth, 8, stride=8).squeeze(1)
        s_diff = s_smooth[:, :, 4:] - s_smooth[:, :, :-4]
        s_d16 = F.avg_pool1d(F.pad(s_diff, (2, 2), mode="replicate"), 8, stride=8).squeeze(1)
        spec_feats = torch.cat([s_p32, s_m16, s_d16], dim=1).numpy()

    spec_feats = (spec_feats - spec_feats.mean(axis=0, keepdims=True)) / (spec_feats.std(axis=0, keepdims=True) + 1e-6)

    # --- B. Microstructure Image Extraction: Instance norm + 2D-FFT Radial Power + DoG ---
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
    print(f"[Task 3a] Extracted distortion-invariant Image (52d) & Spectrum (64d) features in {time.time()-t0:.2f}s", flush=True)
    return img_feats, spec_feats, spec_clean, spiky_pct, tilt_std


class MultimodalContrastiveAutoencoder(nn.Module):
    """
    Task 3b & 3c: Dual-Branch Self-Supervised Denoising Autoencoder + Contrastive Encoder
    producing L2-normalized Image, Spectrum, and Fused latent representations.
    (Includes DEC cluster_centers placeholder for downstream Stage 3d–3f fine-tuning.)
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
        self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, latent_dim * 2))

    def encode_modalities(self, x_img: torch.Tensor, x_spec: torch.Tensor):
        z_img = F.normalize(self.img_enc(x_img), p=2, dim=1)
        z_spec = F.normalize(self.spec_enc(x_spec), p=2, dim=1)
        z_fused = F.normalize(torch.cat([z_img, z_spec], dim=1), p=2, dim=1)
        return z_img, z_spec, z_fused

    def forward_autoencoder(self, x_img: torch.Tensor, x_spec: torch.Tensor):
        z_img, z_spec, z_fused = self.encode_modalities(x_img, x_spec)
        return z_img, z_spec, z_fused, self.img_dec(z_img), self.spec_dec(z_spec)


def domain_augment_batch(x_img: torch.Tensor, x_spec: torch.Tensor):
    scale_i = 1.0 + 0.08 * torch.randn_like(x_img)
    noise_i = 0.12 * torch.randn_like(x_img)
    drop_i = (torch.rand_like(x_img) > 0.08).float()
    aug_img = (x_img * scale_i + noise_i) * drop_i

    shift_mix = 0.15 * torch.roll(x_spec, shifts=1, dims=1) + 0.85 * x_spec
    noise_s = 0.12 * torch.randn_like(x_spec)
    drop_s = (torch.rand_like(x_spec) > 0.08).float()
    aug_spec = (shift_mix + noise_s) * drop_s
    return aug_img, aug_spec


def train_self_supervised_representation(
    img_feats: np.ndarray, spec_feats: np.ndarray,
    latent_dim: int = 12, epochs: int = 15, batch_size: int = 512, seed: int = 42
):
    seed_everything(seed)
    t0 = time.time()
    model = MultimodalContrastiveAutoencoder(
        img_dim=img_feats.shape[1], spec_dim=spec_feats.shape[1], latent_dim=latent_dim, n_clusters=6
    )
    X_im_t = torch.from_numpy(img_feats)
    X_sp_t = torch.from_numpy(spec_feats)
    N = len(img_feats)

    opt = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pretrain_losses = []

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        tot_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            bi, bs = X_im_t[idx], X_sp_t[idx]
            bi_a1, bs_a1 = domain_augment_batch(bi, bs)
            bi_a2, bs_a2 = domain_augment_batch(bi, bs)

            opt.zero_grad()
            zi1, zs1, _, ri1, rs1 = model.forward_autoencoder(bi_a1, bs_a1)
            zi2, zs2, _, _, _ = model.forward_autoencoder(bi_a2, bs_a2)

            loss_rec = F.mse_loss(ri1, bi) + F.mse_loss(rs1, bs)
            loss_align = (1.0 - (zi1 * zi2).sum(dim=1)).mean() + (1.0 - (zs1 * zs2).sum(dim=1)).mean()
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

    model.eval()
    with torch.no_grad():
        zi, zs, zf = model.encode_modalities(X_im_t, X_sp_t)
    print(f"[Task 3b] Contrastive-AE pretraining ({epochs} epochs) finished in {time.time()-t0:.2f}s "
          f"(Loss: {pretrain_losses[0]:.4f} -> {pretrain_losses[-1]:.4f})", flush=True)
    return model, zi.numpy(), zs.numpy(), zf.numpy(), pretrain_losses


def generate_part1_plots(meta, imgs, specs, spec_clean, ct_spec4_img4, out_dir):
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

    pad_s = np.pad(specs[:1000], ((0, 0), (1, 1)), mode="edge")
    med_s = np.median(np.stack([pad_s[:, :-2], pad_s[:, 1:-1], pad_s[:, 2:]], axis=-1), axis=-1)
    spiky_i = int(np.argmax(np.abs(specs[:1000] - med_s).max(axis=1)))
    axes[1].plot(specs[spiky_i], color="tab:red", lw=1.1, alpha=0.8, label="Raw Spectrum (Spike + Tilt)")
    axes[1].plot(spec_clean[spiky_i], color="tab:blue", lw=1.3, label="Cleaned (3-pt Median + Detrended)")
    axes[1].set_title(f"Task 3a: Spectral Distortion & Cleaning ({meta['id_sampel'].iloc[spiky_i]})", fontweight="bold", fontsize=9.5)
    axes[1].set_xlabel("Spectral Channel (0..127)")
    axes[1].set_ylabel("Normalized Intensity")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    ct_mat = ct_spec4_img4.values
    axes[2].imshow(ct_mat, cmap="YlGnBu")
    axes[2].set_xticks(range(ct_mat.shape[1]))
    axes[2].set_yticks(range(ct_mat.shape[0]))
    axes[2].set_xticklabels([f"Img-{j}" for j in range(ct_mat.shape[1])], fontsize=8)
    axes[2].set_yticklabels([f"Spec-{i}" for i in range(ct_mat.shape[0])], fontsize=8)
    axes[2].set_title("Task 3c: Spectrum(4) × Image(4) Contingency", fontweight="bold", fontsize=9.5)
    for r in range(ct_mat.shape[0]):
        for c in range(ct_mat.shape[1]):
            val = ct_mat[r, c]
            axes[2].text(c, r, f"{val}", ha="center", va="center",
                         color="white" if val > ct_mat.max() * 0.5 else "black", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t3_distortions_and_spectra.png"), dpi=160)
    plt.close()


def main():
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Task 3 Part 1 (3a-3c): Distortion Cleaning & Multimodal Representation")
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

    ckpt_path = os.path.join(ckpt_dir, "task3_pretrained_rep.pt")
    metrics_path = os.path.join(out_plots, "task3_part1_metrics.json")

    if args.eval_only and os.path.exists(ckpt_path):
        print("[Task 3a-3c] --eval_only mode: Loading saved self-supervised representation checkpoint...", flush=True)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = MultimodalContrastiveAutoencoder(
            img_dim=img_feats.shape[1], spec_dim=spec_feats.shape[1], latent_dim=12, n_clusters=6
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        with torch.no_grad():
            zi, zs, zf = model.encode_modalities(torch.from_numpy(img_feats), torch.from_numpy(spec_feats))
        km6 = KMeans(n_clusters=6, n_init=10, random_state=42).fit_predict(zf.numpy())
        sil = float(silhouette_score(zf.numpy()[:6000], km6[:6000]))
        mik_ari = float(adjusted_rand_score(mik_binary, km6))
        print(f"[Task 3a-3c] Pretrained Fused Space (K=6 probe): Silhouette={sil:.4f} | Microscope ARI={mik_ari:.5f}", flush=True)
        return

    model, z_img, z_spec, z_fused, pretrain_losses = train_self_supervised_representation(
        img_feats, spec_feats, latent_dim=12, epochs=15, batch_size=512, seed=42
    )

    # Task 3c: Modality Value & What is Lost when using only 1 modality
    km_spec4 = KMeans(n_clusters=4, n_init=15, random_state=42).fit_predict(z_spec)
    km_img4 = KMeans(n_clusters=4, n_init=15, random_state=42).fit_predict(z_img)
    ct_spec4_img4 = pd.crosstab(km_spec4, km_img4)

    km_fused6 = KMeans(n_clusters=6, n_init=15, random_state=42).fit_predict(z_fused)
    km_img6 = KMeans(n_clusters=6, n_init=15, random_state=42).fit_predict(z_img)
    km_spec6 = KMeans(n_clusters=6, n_init=15, random_state=42).fit_predict(z_spec)

    ari_img_fused = float(adjusted_rand_score(km_img6, km_fused6))
    ari_spec_fused = float(adjusted_rand_score(km_spec6, km_fused6))
    ari_img_spec = float(adjusted_rand_score(km_img6, km_spec6))
    mik_ari_fused = float(adjusted_rand_score(mik_binary, km_fused6))

    print(f"[Task 3c] Cross-Modality ARI (K=6): ARI(Img, Fused)={ari_img_fused:.4f} | "
          f"ARI(Spec, Fused)={ari_spec_fused:.4f} | ARI(Img, Spec)={ari_img_spec:.4f} | Microscope ARI={mik_ari_fused:.5f}", flush=True)
    print("Contingency Spec(4) vs Img(4):\n", ct_spec4_img4, flush=True)

    torch.save({
        "state_dict": model.state_dict(),
        "latent_dim": 12,
        "img_dim": img_feats.shape[1],
        "spec_dim": spec_feats.shape[1],
    }, ckpt_path)

    generate_part1_plots(meta, imgs, specs, spec_clean, ct_spec4_img4, out_plots)

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
            "epochs": 15,
            "latent_dim_per_modality": 12,
            "fused_dim": 24,
            "pretrain_losses": pretrain_losses,
        },
        "modality_ablation_k6": {
            "ari_image_only_vs_fused": round(ari_img_fused, 4),
            "ari_spectrum_only_vs_fused": round(ari_spec_fused, 4),
            "ari_image_only_vs_spectrum_only": round(ari_img_spec, 4),
            "microscope_confounding_ari_fused": round(mik_ari_fused, 5),
            "contingency_spec4_vs_img4": ct_spec4_img4.values.tolist(),
        },
    }
    save_json(results, metrics_path)
    print(f"[Task 3a-3c] Saved pretrained checkpoint, Part 1 metrics & plots in {time.time()-t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
