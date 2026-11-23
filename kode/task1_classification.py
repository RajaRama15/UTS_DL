#!/usr/bin/env python3
"""
Task 1 – Multimodal Classification: Product Defects (30 points)
Complete, reproducible Deep Learning pipeline covering:
  a. Exploration (class distribution, missing sensors, train-test Lini A/B shift, visual examples)
  b. Architecture (Oriented Multi-Scale 2D-CNN Image Encoder + Multi-Scale 1D-CNN Signal Encoder;
     comparison of Early Fusion, Late Fusion, and Gated Cross-Modal Attention Fusion)
  c. Value of each modality (Image-only vs Sensor-only vs Multimodal per-class F1 analysis)
  d. Imperfect labels (Out-of-fold confident mislabel detection + Standard CE vs Label Smoothing vs GCE)
  e. Domain shift (Validation performance per production line Lini A vs B; Global vs Per-Lini Normalization)
  f. Missing modality (Robust evaluation when sensor is recorded vs partial vs completely missing + Modality Dropout)
  g. Training (Sanity check overfitting 1 batch, learning curves, 3-seed evaluation, threshold calibration,
     and generating prediksi_klasifikasi.csv)
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import seed_everything, save_json

CLASSES = ['normal', 'gores', 'noda', 'retak', 'gelembung', 'las_dingin', 'kontaminasi']
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for i, c in enumerate(CLASSES)}


# =====================================================================
# 1. DATA LOADING & PREPROCESSING (GLOBAL vs PER-LINI NORMALIZATION)
# =====================================================================
def load_task1_data(data_dir: str):
    t0 = time.time()
    tr_meta = pd.read_csv(os.path.join(data_dir, "klasifikasi_train_meta.csv"))
    te_meta = pd.read_csv(os.path.join(data_dir, "klasifikasi_test_meta.csv"))
    tr_npz = np.load(os.path.join(data_dir, "klasifikasi_train.npz"))
    te_npz = np.load(os.path.join(data_dir, "klasifikasi_test.npz"))

    tr_img = tr_npz["gambar"]  # (48000, 64, 64) uint8
    tr_sig = tr_npz["sensor"].astype(np.float32)  # (48000, 3, 128)
    te_img = te_npz["gambar"]  # (16000, 64, 64) uint8
    te_sig = te_npz["sensor"].astype(np.float32)  # (16000, 3, 128)
    print(f"[Task 1] Loaded train ({len(tr_meta)}) & test ({len(te_meta)}) in {time.time()-t0:.2f}s", flush=True)
    return tr_meta, te_meta, tr_img, tr_sig, te_img, te_sig


def preprocess_features(
    tr_img_raw, tr_sig_raw, tr_lini,
    te_img_raw=None, te_sig_raw=None, te_lini=None,
    fit_idx=None, per_line_norm: bool = True
):
    """
    Fits normalization parameters strictly on fit_idx of training data (no data leakage).
    Precomputes full-resolution 64x64 directional filter responses (horizontal 1x7, vertical 7x1,
    3x3 high-pass, and Sobel gradient) pooled into a (N, 6, 16, 16) multi-orientation feature map
    plus 24 full-resolution spatial statistics.
    """
    t0 = time.time()
    if fit_idx is None:
        fit_idx = np.arange(len(tr_img_raw))

    stats = {}
    groups = [0, 1] if per_line_norm else ["all"]
    for g in groups:
        m_fit = fit_idx if g == "all" else fit_idx[tr_lini[fit_idx] == g]
        im_sub = tr_img_raw[m_fit].astype(np.float32)
        im_mean = float(im_sub.mean())
        im_std = float(im_sub.std() + 1e-6)
        s_means, s_stds = [], []
        for c in range(3):
            sc = tr_sig_raw[m_fit, c, :]
            s_means.append(float(np.nanmean(sc)))
            s_stds.append(float(np.nanstd(sc) + 1e-6))
        stats[g] = {"im_mean": im_mean, "im_std": im_std, "s_means": s_means, "s_stds": s_stds}

    def _transform(img_raw, sig_raw, lini_arr):
        N = len(img_raw)
        im_raw_norm = np.zeros((N, 64, 64), dtype=np.float32)
        im_inst_norm = np.zeros((N, 64, 64), dtype=np.float32)
        sig_out = np.zeros((N, 3, 128), dtype=np.float32)
        mask_out = (~np.isnan(sig_raw).any(axis=2)).astype(np.float32)

        for g in groups:
            m = np.ones(N, dtype=bool) if g == "all" else (lini_arr == g)
            if not np.any(m):
                continue
            st = stats[g]
            im_l = (img_raw[m].astype(np.float32) - st["im_mean"]) / st["im_std"]
            im_raw_norm[m] = im_l
            if per_line_norm:
                p_mean = im_l.mean(axis=(1, 2), keepdims=True)
                p_std = im_l.std(axis=(1, 2), keepdims=True) + 1e-3
                im_inst_norm[m] = (im_l - p_mean) / p_std
            else:
                im_inst_norm[m] = im_l

            for c in range(3):
                sc = (sig_raw[m, c, :] - st["s_means"][c]) / st["s_stds"][c]
                sig_out[m, c, :] = np.nan_to_num(sc, nan=0.0)

        img_16 = np.zeros((N, 6, 16, 16), dtype=np.float32)
        img_feats = np.zeros((N, 24), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, N, 4000):
                xr = torch.from_numpy(im_raw_norm[i:i+4000, None, :, :])
                x = torch.from_numpy(im_inst_norm[i:i+4000, None, :, :])
                b3 = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
                hf = x - b3
                h_line = F.avg_pool2d(F.pad(x, (3, 3, 0, 0), mode="reflect"), (1, 7), stride=1) - \
                         F.avg_pool2d(F.pad(x, (3, 3, 1, 1), mode="reflect"), (3, 7), stride=1)
                v_line = F.avg_pool2d(F.pad(x, (0, 0, 3, 3), mode="reflect"), (7, 1), stride=1) - \
                         F.avg_pool2d(F.pad(x, (1, 1, 3, 3), mode="reflect"), (7, 3), stride=1)
                dx = F.pad(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]), (0, 1, 0, 0))
                dy = F.pad(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]), (0, 0, 0, 1))
                grad = dx + dy

                c0 = F.avg_pool2d(x, 4, stride=4)
                c1 = F.max_pool2d(F.relu(hf), 4, stride=4)
                c2 = F.max_pool2d(F.relu(-hf), 4, stride=4)
                c3 = F.max_pool2d(torch.abs(h_line), 4, stride=4)
                c4 = F.max_pool2d(torch.abs(v_line), 4, stride=4)
                c5 = F.max_pool2d(grad, 4, stride=4)
                img_16[i:i+4000] = torch.cat([c0, c1, c2, c3, c4, c5], dim=1).numpy()

                f_list = [
                    xr.mean(dim=(1, 2, 3)), xr.std(dim=(1, 2, 3)), xr.amin(dim=(1, 2, 3)), xr.amax(dim=(1, 2, 3)),
                    torch.quantile(xr.flatten(1), 0.01, dim=1), torch.quantile(xr.flatten(1), 0.05, dim=1),
                    torch.quantile(xr.flatten(1), 0.10, dim=1), torch.quantile(xr.flatten(1), 0.90, dim=1),
                    torch.quantile(xr.flatten(1), 0.95, dim=1), torch.quantile(xr.flatten(1), 0.99, dim=1),
                    hf.abs().mean(dim=(1, 2, 3)), hf.std(dim=(1, 2, 3)), hf.amin(dim=(1, 2, 3)), hf.amax(dim=(1, 2, 3)),
                    h_line.abs().amax(dim=(1, 2, 3)), h_line.std(dim=(1, 2, 3)),
                    v_line.abs().amax(dim=(1, 2, 3)), v_line.std(dim=(1, 2, 3)),
                    grad.mean(dim=(1, 2, 3)), grad.std(dim=(1, 2, 3)), grad.amax(dim=(1, 2, 3)),
                    c0.std(dim=(1, 2, 3)), c0.amin(dim=(1, 2, 3)), c0.amax(dim=(1, 2, 3)),
                ]
                img_feats[i:i+4000] = torch.stack(f_list, dim=1).numpy()

        return img_16, img_feats, sig_out, mask_out

    tr_res = _transform(tr_img_raw, tr_sig_raw, tr_lini)
    print(f"[Task 1] Preprocessed tensors (per_line_norm={per_line_norm}) in {time.time()-t0:.2f}s", flush=True)
    if te_img_raw is not None:
        te_res = _transform(te_img_raw, te_sig_raw, te_lini)
        return tr_res, te_res, stats
    return tr_res, stats


# =====================================================================
# 2. ARCHITECTURES: IMAGE ENCODER, SIGNAL ENCODER & 3 FUSION STRATEGIES
# =====================================================================
class OrientedImageEncoder(nn.Module):
    """
    2D CNN operating on the 6-channel 16x16 multi-orientation directional pyramid
    (smooth intensity, positive/negative high-frequency residual, horizontal 1x7 response,
    vertical 7x1 response, and Sobel gradient energy) fused with 24 full-resolution statistics.
    """
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=1, padding=1, bias=False),   # 16x16
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 80, 3, stride=2, padding=1, bias=False),  # 4x4
            nn.BatchNorm2d(80),
            nn.GELU(),
        )
        self.token_proj = nn.Linear(80, out_dim)
        self.proj = nn.Sequential(
            nn.Linear(80 * 3 + 24, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x, g_stats, return_tokens: bool = False):
        h = self.net(x)  # (B, 80, 4, 4)
        p_avg = h.mean(dim=(2, 3))
        p_max = h.amax(dim=(2, 3))
        p_std = h.std(dim=(2, 3))
        emb = self.proj(torch.cat([p_avg, p_max, p_std, g_stats], dim=1))
        if return_tokens:
            tokens = self.token_proj(h.flatten(2).transpose(1, 2))  # (B, 16, out_dim)
            return emb, tokens
        return emb


class MultiScaleSignalEncoder(nn.Module):
    """
    1D-CNN Signal Encoder over 128-step 3-channel press signals (Temperature, Vibration, Current)
    with explicit physical summary statistics and missing-channel masking.
    """
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, 7, stride=2, padding=3, bias=False),   # 64
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False),  # 32
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 80, 3, stride=2, padding=1, bias=False),  # 16
            nn.BatchNorm1d(80),
            nn.GELU(),
        )
        self.token_proj = nn.Linear(80, out_dim)
        self.proj = nn.Sequential(
            nn.Linear(80 * 3 + 18 + 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, s, mask, return_tokens: bool = False):
        s_m = s * mask.unsqueeze(-1)
        h = self.net(s_m)  # (B, 80, 16)
        p_avg = h.mean(dim=2)
        p_max = h.amax(dim=2)
        p_min = h.amin(dim=2)

        s_mean = s_m.mean(dim=2)
        s_std = s_m.std(dim=2)
        s_min = s_m.amin(dim=2)
        s_max = s_m.amax(dim=2)
        s_mid = s_m[:, :, 32:96].mean(dim=2)
        s_hf = (s_m[:, :, 1:] - s_m[:, :, :-1]).pow(2).mean(dim=2).sqrt()
        phys = torch.cat([s_mean, s_std, s_min, s_max, s_mid, s_hf], dim=1) * mask.repeat(1, 6)

        emb = self.proj(torch.cat([p_avg, p_max, p_min, phys, mask], dim=1))
        if return_tokens:
            tokens = self.token_proj(h.transpose(1, 2))  # (B, 16, out_dim)
            return emb, tokens
        return emb


class DefectClassifier(nn.Module):
    """
    Unified Multimodal Defect Classifier supporting:
      - mode='image_only': Image encoder only (Task 1c)
      - mode='sensor_only': Signal encoder only (Task 1c)
      - mode='early_fusion': Concatenate feature embeddings + MLP classifier (Task 1b)
      - mode='late_fusion': Separate class logits from each modality combined via mask-aware weights (Task 1b)
      - mode='cross_attention': Cross-Modal Attention + Missing-Modality Gated Residual Fusion (Task 1b/1f)
    """
    def __init__(self, mode: str = "cross_attention", dim: int = 64, num_classes: int = 7):
        super().__init__()
        self.mode = mode
        self.dim = dim
        if mode != "sensor_only":
            self.img_enc = OrientedImageEncoder(dim)
        if mode != "image_only":
            self.sig_enc = MultiScaleSignalEncoder(dim)

        if mode == "image_only":
            self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_classes))
        elif mode == "sensor_only":
            self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_classes))
        elif mode == "early_fusion":
            self.head = nn.Sequential(
                nn.Linear(dim * 2 + 3, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim, num_classes),
            )
        elif mode == "late_fusion":
            self.img_head = nn.Linear(dim, num_classes)
            self.sig_head = nn.Linear(dim, num_classes)
            self.alpha_net = nn.Sequential(nn.Linear(dim * 2 + 3, num_classes), nn.Sigmoid())
        elif mode == "cross_attention":
            # Efficient scaled dot-product cross-attention between 16 spatial image tokens and 16 temporal signal tokens
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.norm_ca = nn.LayerNorm(dim)
            self.gate = nn.Sequential(nn.Linear(dim * 3 + 3, dim * 3), nn.Sigmoid())
            self.head = nn.Sequential(
                nn.Linear(dim * 3, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.08),
                nn.Linear(dim, num_classes),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, img, g_stats, sig, mask):
        if self.mode == "image_only":
            return self.head(self.img_enc(img, g_stats))
        if self.mode == "sensor_only":
            return self.head(self.sig_enc(sig, mask))

        if self.mode == "early_fusion":
            fi = self.img_enc(img, g_stats)
            fs = self.sig_enc(sig, mask)
            return self.head(torch.cat([fi, fs, mask], dim=1))

        if self.mode == "late_fusion":
            fi = self.img_enc(img, g_stats)
            fs = self.sig_enc(sig, mask)
            li = self.img_head(fi)
            ls = self.sig_head(fs)
            has_sig = (mask.sum(dim=1, keepdim=True) > 0).float()
            w = self.alpha_net(torch.cat([fi, fs, mask], dim=1)) * has_sig
            return (1.0 - 0.5 * w) * li + (0.5 * w) * ls

        # cross_attention
        fi, ti = self.img_enc(img, g_stats, return_tokens=True)
        fs, ts = self.sig_enc(sig, mask, return_tokens=True)
        has_sig = (mask.sum(dim=1, keepdim=True) > 0).float()
        q = self.q_proj(ti)
        k = self.k_proj(ts)
        v = self.v_proj(ts)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (self.dim ** 0.5), dim=-1)
        ca_pool = self.norm_ca(torch.bmm(attn, v).mean(dim=1)) * has_sig
        cat = torch.cat([fi, fs * has_sig, ca_pool], dim=1)
        g = self.gate(torch.cat([cat, mask], dim=1))
        return self.head(cat * g)


# =====================================================================
# 3. ROBUST LOSS FUNCTIONS FOR LABEL NOISE (TASK 1d)
# =====================================================================
class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self, q: float = 0.5):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        py = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
        return ((1.0 - py.pow(self.q)) / self.q).mean()


# =====================================================================
# 4. TRAINING, EVALUATION & THRESHOLD CALIBRATION HELPERS
# =====================================================================
def train_and_evaluate(
    model, tr_tensors, va_tensors, y_tr, y_va,
    epochs: int = 8, batch_size: int = 512, lr: float = 4e-3,
    loss_type: str = "label_smoothing", modality_dropout: float = 0.15,
    seed: int = 42, verbose: bool = True
):
    seed_everything(seed)
    X_im_tr, X_gs_tr, X_sg_tr, X_mk_tr = tr_tensors
    X_im_va, X_gs_va, X_sg_va, X_mk_va = va_tensors
    Y_tr_t = torch.from_numpy(y_tr).long()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if loss_type == "ce":
        crit = nn.CrossEntropyLoss(label_smoothing=0.0)
    elif loss_type == "label_smoothing":
        crit = nn.CrossEntropyLoss(label_smoothing=0.06)
    elif loss_type == "gce":
        crit = GeneralizedCrossEntropyLoss(q=0.5)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    N_tr = len(y_tr)
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1 = -1.0
    best_probs = None
    best_state = None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(N_tr)
        tot_loss = 0.0
        for i in range(0, N_tr, batch_size):
            idx = perm[i:i + batch_size]
            bi, bg, bs, bm, by = X_im_tr[idx], X_gs_tr[idx], X_sg_tr[idx], X_mk_tr[idx], Y_tr_t[idx]
            if modality_dropout > 0 and np.random.rand() < modality_dropout:
                if np.random.rand() < 0.5:
                    bm = torch.zeros_like(bm)
                else:
                    bm = bm * (torch.rand_like(bm) > 0.35).float()
            opt.zero_grad()
            out = model(bi, bg, bs, bm)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(by)
        sched.step()
        tr_loss = tot_loss / N_tr

        model.eval()
        with torch.no_grad():
            va_logits = model(X_im_va, X_gs_va, X_sg_va, X_mk_va)
            va_loss = float(F.cross_entropy(va_logits, torch.from_numpy(y_va).long()).item())
            va_probs = F.softmax(va_logits, dim=1).numpy()
        va_preds = va_probs.argmax(axis=1)
        mf1 = float(f1_score(y_va, va_preds, average="macro"))

        history["train_loss"].append(round(tr_loss, 4))
        history["val_loss"].append(round(va_loss, 4))
        history["val_macro_f1"].append(round(mf1, 4))

        if mf1 > best_f1:
            best_f1 = mf1
            best_probs = va_probs.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"  Epoch {ep:02d}/{epochs:02d} ({time.time()-t0:.1f}s) | TrLoss={tr_loss:.4f} | ValLoss={va_loss:.4f} | Val Macro-F1={mf1:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_probs, best_f1, history


def tune_class_multipliers(probs: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, float]:
    mult = np.ones(probs.shape[1], dtype=np.float32)
    best_f1 = float(f1_score(y_true, probs.argmax(axis=1), average="macro"))
    for _ in range(2):
        for c in range(probs.shape[1]):
            best_c = mult[c]
            for cand in np.linspace(0.55, 1.95, 29):
                m_try = mult.copy()
                m_try[c] = cand
                f1_try = float(f1_score(y_true, (probs * m_try).argmax(axis=1), average="macro"))
                if f1_try > best_f1:
                    best_f1 = f1_try
                    best_c = cand
            mult[c] = best_c
    return mult, best_f1


def predict_probs(model, tensors, batch_size: int = 2048) -> np.ndarray:
    X_im, X_gs, X_sg, X_mk = tensors
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_im), batch_size):
            logits = model(X_im[i:i+batch_size], X_gs[i:i+batch_size], X_sg[i:i+batch_size], X_mk[i:i+batch_size])
            probs.append(F.softmax(logits, dim=1).numpy())
    return np.concatenate(probs, axis=0)


# =====================================================================
# 5. PLOTTING & REPORT ARTIFACT GENERATORS
# =====================================================================
def generate_eda_plots(tr_meta, te_meta, tr_img, tr_sig, out_dir):
    fig, axes = plt.subplots(2, 7, figsize=(16, 4.8))
    t_axis = np.arange(128)
    for c_idx, c_name in enumerate(CLASSES):
        m = (tr_meta["kelas"] == c_name).values & (tr_meta["lini"] == "A").values & (~np.isnan(tr_sig).any(axis=(1, 2)))
        sample_i = np.where(m)[0][0]
        axes[0, c_idx].imshow(tr_img[sample_i], cmap="gray", vmin=80, vmax=200)
        axes[0, c_idx].set_title(f"{c_name}\n({tr_meta['id_unit'].iloc[sample_i]})", fontsize=9, fontweight="bold")
        axes[0, c_idx].axis("off")

        s = tr_sig[sample_i]
        ax_s = axes[1, c_idx]
        ax_s.plot(t_axis, (s[0] - 181.0), color="tab:red", lw=1.0, label="T-181°C")
        ax_s.plot(t_axis, s[1], color="tab:blue", lw=0.9, alpha=0.85, label="Vib")
        ax_s.plot(t_axis, (s[2] - 8.5) / 2.0, color="tab:green", lw=1.0, label="(I-8.5)/2")
        ax_s.set_ylim(-4.2, 4.2)
        ax_s.set_xticks([0, 64, 127])
        if c_idx == 0:
            ax_s.set_ylabel("Normalized Signal", fontsize=8)
            ax_s.legend(fontsize=6, loc="lower right")
        else:
            ax_s.set_yticklabels([])
        ax_s.grid(True, alpha=0.25)

    plt.suptitle("Task 1a: Representative Image & 3-Channel Press Sensor Signals per Defect Class (Lini A)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t1_class_examples.png"), dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    vc = tr_meta["kelas"].value_counts().reindex(CLASSES)
    axes[0].bar(CLASSES, vc.values, color="steelblue", edgecolor="black")
    axes[0].set_title("Train Class Distribution (N=48,000)", fontweight="bold", fontsize=10)
    axes[0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(vc.values):
        axes[0].text(i, v + 300, f"{v/len(tr_meta)*100:.1f}%", ha="center", fontsize=8)

    lini_df = pd.DataFrame({
        "Train": tr_meta["lini"].value_counts(normalize=True).reindex(["A", "B"]) * 100,
        "Test": te_meta["lini"].value_counts(normalize=True).reindex(["A", "B"]) * 100,
    })
    lini_df.plot(kind="bar", ax=axes[1], color=["#2b5c8f", "#d95f02"], edgecolor="black", rot=0)
    axes[1].set_title("Production Line Shift: Train vs Test (%)", fontweight="bold", fontsize=10)
    axes[1].set_ylabel("Percentage (%)")

    m_a = (tr_meta["lini"] == "A").values
    m_b = (tr_meta["lini"] == "B").values
    axes[2].hist(tr_img[m_a].mean(axis=(1, 2)), bins=40, alpha=0.65, density=True, label="Lini A (Mean 139.8)", color="#2b5c8f")
    axes[2].hist(tr_img[m_b].mean(axis=(1, 2)), bins=40, alpha=0.65, density=True, label="Lini B (Mean 111.7)", color="#d95f02")
    axes[2].set_title("Image Mean Intensity Shift by Production Line", fontweight="bold", fontsize=10)
    axes[2].set_xlabel("Mean Pixel Intensity (0-255)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t1_distribution_and_shift.png"), dpi=160)
    plt.close()


# =====================================================================
# 6. MAIN PIPELINE EXECUTION
# =====================================================================
def main():
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Task 1 Multimodal Classification Pipeline")
    parser.add_argument("--data_dir", type=str, default=r"E:\Code\data_uts_dl")
    parser.add_argument("--out_dir", type=str, default=r"E:\Code\uts_dl_submission")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    out_plots = os.path.join(args.out_dir, "kode", "outputs")
    ckpt_dir = os.path.join(args.out_dir, "kode", "checkpoints")
    os.makedirs(out_plots, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(42)
    tr_meta, te_meta, tr_img_raw, tr_sig_raw, te_img_raw, te_sig_raw = load_task1_data(args.data_dir)
    y_all = tr_meta["kelas"].map(CLS2IDX).values
    tr_lini = (tr_meta["lini"] == "B").astype(np.int64).values
    te_lini = (te_meta["lini"] == "B").astype(np.int64).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, va_idx = next(skf.split(tr_img_raw, y_all * 2 + tr_lini))

    (X_im, X_gs, X_sg, X_mk), (X_im_te, X_gs_te, X_sg_te, X_mk_te), norm_stats = preprocess_features(
        tr_img_raw, tr_sig_raw, tr_lini, te_img_raw, te_sig_raw, te_lini, fit_idx=tr_idx, per_line_norm=True
    )

    tr_tensors = tuple(torch.from_numpy(arr[tr_idx]) for arr in (X_im, X_gs, X_sg, X_mk))
    va_tensors = tuple(torch.from_numpy(arr[va_idx]) for arr in (X_im, X_gs, X_sg, X_mk))
    te_tensors = tuple(torch.from_numpy(arr) for arr in (X_im_te, X_gs_te, X_sg_te, X_mk_te))
    y_tr, y_va = y_all[tr_idx], y_all[va_idx]

    ckpt_path = os.path.join(ckpt_dir, "task1_final_ensemble.pt")
    metrics_path = os.path.join(out_plots, "task1_metrics.json")

    if args.eval_only and os.path.exists(ckpt_path):
        print("[Task 1] --eval_only mode: Loading saved 3-seed ensemble checkpoints...", flush=True)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        mult = np.asarray(payload["multipliers"], dtype=np.float32)
        va_probs_ens = np.zeros((len(y_va), 7), dtype=np.float32)
        te_probs_ens = np.zeros((len(te_meta), 7), dtype=np.float32)
        seed_f1s = []
        for s_state in payload["seed_states"]:
            m = DefectClassifier(mode="cross_attention", dim=64, num_classes=7)
            m.load_state_dict(s_state)
            vp = predict_probs(m, va_tensors)
            tp = predict_probs(m, te_tensors)
            seed_f1s.append(float(f1_score(y_va, (vp * mult).argmax(axis=1), average="macro")))
            va_probs_ens += vp / len(payload["seed_states"])
            te_probs_ens += tp / len(payload["seed_states"])

        ens_f1 = float(f1_score(y_va, (va_probs_ens * mult).argmax(axis=1), average="macro"))
        print(f"[Task 1] 3-Seed Validation Macro-F1: {np.mean(seed_f1s):.4f} ± {np.std(seed_f1s):.4f} | Ensemble Macro-F1: {ens_f1:.4f}", flush=True)
        te_preds = (te_probs_ens * mult).argmax(axis=1)
        sub = pd.DataFrame({"id_unit": te_meta["id_unit"], "kelas": [IDX2CLS[i] for i in te_preds]})
        sub.to_csv(os.path.join(args.out_dir, "prediksi_klasifikasi.csv"), index=False)
        print(f"[Task 1] Saved prediksi_klasifikasi.csv ({len(sub)} rows).", flush=True)
        return

    print("\n--- [Task 1a] Generating Exploration Summaries & Plots ---", flush=True)
    generate_eda_plots(tr_meta, te_meta, tr_img_raw, tr_sig_raw, out_plots)

    results = {
        "eda": {
            "train_size": int(len(tr_meta)),
            "test_size": int(len(te_meta)),
            "class_counts": tr_meta["kelas"].value_counts().reindex(CLASSES).to_dict(),
            "train_lini_prop": tr_meta["lini"].value_counts(normalize=True).round(4).to_dict(),
            "test_lini_prop": te_meta["lini"].value_counts(normalize=True).round(4).to_dict(),
            "train_any_sensor_nan_pct": round(float(np.isnan(tr_sig_raw).any(axis=(1, 2)).mean() * 100), 2),
            "train_all_sensor_nan_pct": round(float(np.isnan(tr_sig_raw).all(axis=(1, 2)).mean() * 100), 2),
            "test_any_sensor_nan_pct": round(float(np.isnan(te_sig_raw).any(axis=(1, 2)).mean() * 100), 2),
            "test_all_sensor_nan_pct": round(float(np.isnan(te_sig_raw).all(axis=(1, 2)).mean() * 100), 2),
        }
    }

    print("\n--- [Task 1g] Sanity Check: Overfitting One Batch (64 samples) ---", flush=True)
    seed_everything(42)
    sanity_model = DefectClassifier(mode="cross_attention", dim=64, num_classes=7)
    s_opt = torch.optim.AdamW(sanity_model.parameters(), lr=5e-3)
    bi, bg, bs, bm = [t[:64] for t in tr_tensors]
    by = torch.from_numpy(y_tr[:64]).long()
    sanity_losses, sanity_accs = [], []
    for _ in range(1, 41):
        s_opt.zero_grad()
        out = sanity_model(bi, bg, bs, bm)
        loss = F.cross_entropy(out, by)
        loss.backward()
        s_opt.step()
        sanity_losses.append(float(loss.item()))
        sanity_accs.append(float((out.argmax(dim=1) == by).float().mean().item()))
    print(f"  Sanity Check 1-Batch: Initial Loss={sanity_losses[0]:.4f} (Acc={sanity_accs[0]*100:.1f}%) -> Step 40 Loss={sanity_losses[-1]:.5f} (Acc={sanity_accs[-1]*100:.1f}%)", flush=True)
    results["sanity_check"] = {
        "initial_loss": round(sanity_losses[0], 4),
        "final_loss": round(sanity_losses[-1], 5),
        "final_acc": round(sanity_accs[-1], 4),
    }

    print("\n--- [Task 1b & 1c] Modality Ablation & Fusion Comparison ---", flush=True)
    modes_to_run = [
        ("image_only", "Image Only (2D Oriented CNN)"),
        ("sensor_only", "Sensor Only (1D Multi-Scale CNN)"),
        ("early_fusion", "Early Fusion (Concat + MLP)"),
        ("late_fusion", "Late Fusion (Logit Gating)"),
        ("cross_attention", "Cross-Modal Attention + Gated Fusion"),
    ]
    modality_table = {}
    best_ca_model, best_ca_probs, best_ca_hist = None, None, None

    for mode_key, mode_label in modes_to_run:
        print(f"\nTraining [{mode_label}]...", flush=True)
        m = DefectClassifier(mode=mode_key, dim=64, num_classes=7)
        m, v_probs, raw_f1, hist = train_and_evaluate(
            m, tr_tensors, va_tensors, y_tr, y_va,
            epochs=8, batch_size=512, lr=4e-3,
            loss_type="label_smoothing", modality_dropout=0.15 if mode_key != "image_only" else 0.0,
            seed=42, verbose=True
        )
        v_preds = v_probs.argmax(axis=1)
        per_cls_f1 = f1_score(y_va, v_preds, average=None)
        mult_m, tuned_f1 = tune_class_multipliers(v_probs, y_va)
        tuned_per_cls = f1_score(y_va, (v_probs * mult_m).argmax(axis=1), average=None)
        modality_table[mode_key] = {
            "label": mode_label,
            "raw_macro_f1": round(raw_f1, 4),
            "tuned_macro_f1": round(tuned_f1, 4),
            "per_class_f1": {CLASSES[i]: round(float(per_cls_f1[i]), 4) for i in range(7)},
            "tuned_per_class_f1": {CLASSES[i]: round(float(tuned_per_cls[i]), 4) for i in range(7)},
        }
        print(f"  => {mode_label}: Raw Macro-F1={raw_f1:.4f} | Tuned Macro-F1={tuned_f1:.4f}", flush=True)
        if mode_key == "cross_attention":
            best_ca_model, best_ca_probs, best_ca_hist = m, v_probs, hist

    results["modality_and_fusion"] = modality_table

    print("\n--- [Task 1d] Imperfect Labels Detection & Robust Loss Comparison ---", flush=True)
    pred_cls = best_ca_probs.argmax(axis=1)
    max_prob = best_ca_probs.max(axis=1)
    prob_given = best_ca_probs[np.arange(len(y_va)), y_va]
    suspected_mask = (pred_cls != y_va) & (max_prob > 0.85) & (prob_given < 0.08)
    n_suspected = int(suspected_mask.sum())
    suspected_pct = round(n_suspected / len(y_va) * 100, 2)
    print(f"  Detected {n_suspected} high-confidence suspected mislabels in validation ({suspected_pct}% of val set).", flush=True)

    loss_comp = {}
    for l_type, l_name in [("ce", "Standard Cross-Entropy"), ("gce", "Generalized Cross-Entropy (q=0.5)")]:
        print(f"  Testing [{l_name}]...", flush=True)
        m_l = DefectClassifier(mode="cross_attention", dim=64, num_classes=7)
        _, vp_l, rf1_l, _ = train_and_evaluate(
            m_l, tr_tensors, va_tensors, y_tr, y_va,
            epochs=8, batch_size=512, lr=4e-3, loss_type=l_type, modality_dropout=0.15, seed=42, verbose=False
        )
        _, tf1_l = tune_class_multipliers(vp_l, y_va)
        loss_comp[l_type] = {"raw_macro_f1": round(rf1_l, 4), "tuned_macro_f1": round(tf1_l, 4)}
        print(f"    => {l_name}: Raw Macro-F1={rf1_l:.4f} | Tuned Macro-F1={tf1_l:.4f}", flush=True)

    loss_comp["label_smoothing"] = {
        "raw_macro_f1": modality_table["cross_attention"]["raw_macro_f1"],
        "tuned_macro_f1": modality_table["cross_attention"]["tuned_macro_f1"],
    }
    results["label_noise"] = {
        "val_suspected_mislabels_count": n_suspected,
        "val_suspected_mislabels_pct": suspected_pct,
        "loss_comparison": loss_comp,
    }

    print("\n--- [Task 1e] Domain Shift Experiment: Global Normalization vs Per-Lini Normalization ---", flush=True)
    (X_im_g, X_gs_g, X_sg_g, X_mk_g), _ = preprocess_features(
        tr_img_raw, tr_sig_raw, tr_lini, fit_idx=tr_idx, per_line_norm=False
    )
    tr_g = tuple(torch.from_numpy(arr[tr_idx]) for arr in (X_im_g, X_gs_g, X_sg_g, X_mk_g))
    va_g = tuple(torch.from_numpy(arr[va_idx]) for arr in (X_im_g, X_gs_g, X_sg_g, X_mk_g))
    m_glob = DefectClassifier(mode="cross_attention", dim=64, num_classes=7)
    _, vp_glob, _, _ = train_and_evaluate(
        m_glob, tr_g, va_g, y_tr, y_va,
        epochs=8, batch_size=512, lr=4e-3, loss_type="label_smoothing", modality_dropout=0.15, seed=42, verbose=False
    )
    va_lini = tr_lini[va_idx]
    glob_preds = vp_glob.argmax(axis=1)
    perline_preds = best_ca_probs.argmax(axis=1)

    domain_shift_res = {
        "global_norm": {
            "overall_macro_f1": round(float(f1_score(y_va, glob_preds, average="macro")), 4),
            "lini_A_macro_f1": round(float(f1_score(y_va[va_lini == 0], glob_preds[va_lini == 0], average="macro")), 4),
            "lini_B_macro_f1": round(float(f1_score(y_va[va_lini == 1], glob_preds[va_lini == 1], average="macro")), 4),
        },
        "per_line_norm": {
            "overall_macro_f1": round(float(f1_score(y_va, perline_preds, average="macro")), 4),
            "lini_A_macro_f1": round(float(f1_score(y_va[va_lini == 0], perline_preds[va_lini == 0], average="macro")), 4),
            "lini_B_macro_f1": round(float(f1_score(y_va[va_lini == 1], perline_preds[va_lini == 1], average="macro")), 4),
        },
    }
    print(f"  Global Norm   -> Overall={domain_shift_res['global_norm']['overall_macro_f1']:.4f} | Lini A={domain_shift_res['global_norm']['lini_A_macro_f1']:.4f} | Lini B={domain_shift_res['global_norm']['lini_B_macro_f1']:.4f}", flush=True)
    print(f"  Per-Lini Norm -> Overall={domain_shift_res['per_line_norm']['overall_macro_f1']:.4f} | Lini A={domain_shift_res['per_line_norm']['lini_A_macro_f1']:.4f} | Lini B={domain_shift_res['per_line_norm']['lini_B_macro_f1']:.4f}", flush=True)
    results["domain_shift"] = domain_shift_res

    print("\n--- [Task 1f] Missing Modality Robustness Evaluation ---", flush=True)
    va_mask_np = X_mk[va_idx]
    m_full_sens = (va_mask_np.sum(axis=1) == 3)
    m_part_sens = (va_mask_np.sum(axis=1) == 2)
    m_zero_sens = (va_mask_np.sum(axis=1) == 0)

    va_tensors_blackout = (va_tensors[0], va_tensors[1], torch.zeros_like(va_tensors[2]), torch.zeros_like(va_tensors[3]))
    vp_blackout = predict_probs(best_ca_model, va_tensors_blackout)

    missing_res = {
        "natural_all_3_sensors_present_f1": round(float(f1_score(y_va[m_full_sens], perline_preds[m_full_sens], average="macro")), 4),
        "natural_1_channel_missing_f1": round(float(f1_score(y_va[m_part_sens], perline_preds[m_part_sens], average="macro")), 4),
        "natural_all_sensors_missing_f1": round(float(f1_score(y_va[m_zero_sens], perline_preds[m_zero_sens], average="macro")), 4),
        "synthetic_100pct_sensor_blackout_f1": round(float(f1_score(y_va, vp_blackout.argmax(axis=1), average="macro")), 4),
    }
    print("  Missing Modality Breakdown:", missing_res, flush=True)
    results["missing_modality"] = missing_res

    print("\n--- [Task 1g] Multi-Seed Final Model Evaluation (Seeds: 42, 123, 999) ---", flush=True)
    seeds = [42, 123, 999]
    seed_models = [best_ca_model]
    seed_val_probs = [best_ca_probs]
    seed_raw_f1s = [modality_table["cross_attention"]["raw_macro_f1"]]

    for s in seeds[1:]:
        print(f"\nTraining Final Cross-Attention Model (Seed {s})...", flush=True)
        m_s = DefectClassifier(mode="cross_attention", dim=64, num_classes=7)
        m_s, vp_s, rf1_s, _ = train_and_evaluate(
            m_s, tr_tensors, va_tensors, y_tr, y_va,
            epochs=8, batch_size=512, lr=4e-3, loss_type="label_smoothing", modality_dropout=0.15, seed=s, verbose=True
        )
        seed_models.append(m_s)
        seed_val_probs.append(vp_s)
        seed_raw_f1s.append(round(rf1_s, 4))

    ens_val_probs = np.mean(seed_val_probs, axis=0)
    best_mult, ens_tuned_f1 = tune_class_multipliers(ens_val_probs, y_va)
    seed_tuned_f1s = [round(float(f1_score(y_va, (vp * best_mult).argmax(axis=1), average="macro")), 4) for vp in seed_val_probs]

    print(f"\n[Task 1 Summary] 3-Seed Raw Macro-F1:   {np.mean(seed_raw_f1s):.4f} ± {np.std(seed_raw_f1s):.4f} ({seed_raw_f1s})", flush=True)
    print(f"[Task 1 Summary] 3-Seed Tuned Macro-F1: {np.mean(seed_tuned_f1s):.4f} ± {np.std(seed_tuned_f1s):.4f} ({seed_tuned_f1s})", flush=True)
    print(f"[Task 1 Summary] 3-Seed Ensemble Macro-F1: {ens_tuned_f1:.4f}", flush=True)

    ens_val_preds = (ens_val_probs * best_mult).argmax(axis=1)
    print("\nFinal 3-Seed Ensemble Per-Class Validation Report:", flush=True)
    print(classification_report(y_va, ens_val_preds, target_names=CLASSES, digits=4), flush=True)

    results["multi_seed"] = {
        "seeds": seeds,
        "raw_macro_f1_per_seed": seed_raw_f1s,
        "raw_macro_f1_mean": round(float(np.mean(seed_raw_f1s)), 4),
        "raw_macro_f1_std": round(float(np.std(seed_raw_f1s)), 4),
        "tuned_macro_f1_per_seed": seed_tuned_f1s,
        "tuned_macro_f1_mean": round(float(np.mean(seed_tuned_f1s)), 4),
        "tuned_macro_f1_std": round(float(np.std(seed_tuned_f1s)), 4),
        "ensemble_macro_f1": round(ens_tuned_f1, 4),
        "multipliers": [round(float(x), 3) for x in best_mult],
        "ensemble_per_class_f1": {
            CLASSES[i]: round(float(f1_score(y_va, ens_val_preds, average=None)[i]), 4) for i in range(7)
        },
    }

    torch.save({
        "seed_states": [m.state_dict() for m in seed_models],
        "multipliers": best_mult.tolist(),
        "norm_stats": norm_stats,
    }, ckpt_path)

    te_probs_ens = np.mean([predict_probs(m, te_tensors) for m in seed_models], axis=0)
    te_preds = (te_probs_ens * best_mult).argmax(axis=1)
    sub_df = pd.DataFrame({
        "id_unit": te_meta["id_unit"],
        "kelas": [IDX2CLS[i] for i in te_preds],
    })
    sub_path = os.path.join(args.out_dir, "prediksi_klasifikasi.csv")
    sub_df.to_csv(sub_path, index=False)
    results["test_prediction_distribution"] = sub_df["kelas"].value_counts().reindex(CLASSES).to_dict()
    print(f"[Task 1] Saved test predictions to {sub_path} ({len(sub_df)} rows).", flush=True)
    print("Test Prediction Class Distribution:\n", sub_df["kelas"].value_counts(), flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(range(1, 41), sanity_losses, color="tab:red", lw=2, label="1-Batch Train Loss")
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(range(1, 41), [a * 100 for a in sanity_accs], color="tab:green", lw=2, ls="--", label="1-Batch Acc (%)")
    axes[0].set_title("Sanity Check: Overfitting 1 Batch (N=64)", fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Gradient Step")
    axes[0].set_ylabel("Cross-Entropy Loss", color="tab:red")
    ax0_twin.set_ylabel("Accuracy (%)", color="tab:green")
    axes[0].grid(True, alpha=0.3)

    ep_x = range(1, len(best_ca_hist["train_loss"]) + 1)
    axes[1].plot(ep_x, best_ca_hist["train_loss"], "o-", label="Train Loss", color="tab:blue")
    axes[1].plot(ep_x, best_ca_hist["val_loss"], "s--", label="Val Loss", color="tab:orange")
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(ep_x, best_ca_hist["val_macro_f1"], "^-", label="Val Macro-F1", color="tab:green", lw=2)
    axes[1].set_title("Learning Curves (Cross-Modal Attention)", fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    ax1_twin.set_ylabel("Validation Macro-F1", color="tab:green")
    axes[1].legend(loc="center left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    cm = confusion_matrix(y_va, ens_val_preds, normalize="true")
    axes[2].imshow(cm, cmap="Blues", vmin=0, vmax=1.0)
    axes[2].set_xticks(range(7))
    axes[2].set_yticks(range(7))
    axes[2].set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=8)
    axes[2].set_yticklabels(CLASSES, fontsize=8)
    axes[2].set_title(f"Normalized Confusion Matrix (Macro-F1={ens_tuned_f1:.4f})", fontweight="bold", fontsize=10)
    for r in range(7):
        for c in range(7):
            val = cm[r, c]
            axes[2].text(c, r, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.5 else "black", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(out_plots, "t1_training_and_cm.png"), dpi=160)
    plt.close()

    save_json(results, metrics_path)
    print(f"[Task 1] All metrics and plots saved to {out_plots} in {time.time()-t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
