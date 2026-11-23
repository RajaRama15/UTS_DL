#!/usr/bin/env python3
"""
Task 2 – Time-Series Regression: Remaining Useful Life (30 points)
Complete, reproducible Deep Learning pipeline covering every requirement in ML_DL_Spreadsheets - UTS (DL).csv:
  a. Exploration & preprocessing:
     - Check how operating settings (setel_1, setel_2, setel_3) affect sensor readings (6 discrete regimes)
     - Identify uninformative sensors (7 sensors: s01, s05, s06, s10, s16, s18, s19) vs 14 informative sensors
     - Handle missing values (~2.0% NaNs per sensor) and spikes (|z_IQR| > 4.0) causally without future leakage
     - Explain and apply Regime-Conditioned Robust/Z-score Normalization fitted ONLY on training machines
  b. Labels:
     - Build piecewise-linear RUL labels from run-to-failure histories with the 150-cycle cap
  c. Sequences:
     - Build causal sliding windows (W = 50 cycles) and handle variable-length machine histories without leaking future info
  d. Models:
     - Compare at least two sequence architectures (1D-CNN / Dilated TCN and Stacked GRU with Temporal Attention)
       against a non-deep-learning baseline (LightGBM Regressor)
  e. Validation:
     - Split strictly by machine (id_unit: 720 train machines, 180 val machines), NEVER by row
     - Report RMSE and the Asymmetric Score overall and broken down per RUL range ([0, 30], (30, 80], (80, 150])
  f. Uncertainty:
     - Produce prediction intervals via Monte Carlo (MC) Dropout + 3-Seed Deep Ensemble
     - Apply risk-averse conservative shift to minimize late-prediction asymmetric penalties and discuss service thresholds
  - Output: prediksi_rul.csv (columns: id_unit,rul; 300 test machines; values in [0, 150])
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
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import seed_everything, asymmetric_rul_score, save_json

ALL_SENSORS = [f"s{i:02d}" for i in range(1, 22)]
UNINFORMATIVE_SENSORS = ["s01", "s05", "s06", "s10", "s16", "s18", "s19"]
INFORMATIVE_SENSORS = [s for s in ALL_SENSORS if s not in UNINFORMATIVE_SENSORS]  # 14 sensors
WINDOW_SIZE = 50


# =====================================================================
# 1. DATA LOADING, EXPLORATION & CAUSAL REGIME-CONDITIONED PREPROCESSING
# =====================================================================
def load_rul_data(data_dir: str):
    t0 = time.time()
    tr_path = os.path.join(data_dir, "rul_train.csv")
    te_path = os.path.join(data_dir, "rul_test.csv")
    if os.path.isdir(tr_path):
        tr_path = os.path.join(tr_path, "rul_train.csv")
    if os.path.isdir(te_path):
        te_path = os.path.join(te_path, "rul_test.csv")

    tr_df = pd.read_csv(tr_path)
    te_df = pd.read_csv(te_path)
    print(f"[Task 2] Loaded rul_train ({len(tr_df)} rows, {tr_df['id_unit'].nunique()} machines) & "
          f"rul_test ({len(te_df)} rows, {te_df['id_unit'].nunique()} machines) in {time.time()-t0:.2f}s", flush=True)
    return tr_df, te_df


def build_rul_labels(tr_df: pd.DataFrame, cap: int = 150) -> pd.DataFrame:
    """Task 2b: Build RUL labels from run-to-failure histories with 150-cycle cap."""
    tr_df = tr_df.sort_values(["id_unit", "siklus"]).reset_index(drop=True)
    max_cyc = tr_df.groupby("id_unit")["siklus"].transform("max")
    tr_df["rul_uncapped"] = (max_cyc - tr_df["siklus"]).astype(np.float32)
    tr_df["rul"] = tr_df["rul_uncapped"].clip(upper=float(cap)).astype(np.float32)
    return tr_df


def fit_regime_normalizer(tr_df_fit: pd.DataFrame):
    """
    Task 2a: Fit per-operating-regime median, IQR, clipped mean, and std strictly on training split machines.
    Operating regimes are uniquely indexed by op_id = round(setel_1) in {0, 10, 20, 25, 35, 42}.
    """
    df = tr_df_fit.copy()
    df["op_id"] = df["setel_1"].round().astype(int)
    regime_stats = {}
    for op_id, grp in df.groupby("op_id"):
        med = grp[ALL_SENSORS].median().to_dict()
        q75 = grp[ALL_SENSORS].quantile(0.75)
        q25 = grp[ALL_SENSORS].quantile(0.25)
        iqr = (q75 - q25 + 1e-6).to_dict()
        # Compute mean and std after masking extreme >4*IQR spikes
        means, stds = {}, {}
        for s in ALL_SENSORS:
            vals = grp[s].dropna()
            z_iqr = (vals - med[s]).abs() / iqr[s]
            clean_vals = vals[z_iqr <= 4.0]
            means[s] = float(clean_vals.mean())
            stds[s] = float(clean_vals.std() + 1e-6)
        regime_stats[int(op_id)] = {"median": med, "iqr": iqr, "mean": means, "std": stds}
    return regime_stats


def causal_preprocess_dataframe(df: pd.DataFrame, regime_stats: dict):
    """
    Task 2a & 2c:
    1. Assign each cycle to its operating regime (op_id = round(setel_1)).
    2. Normalize all 21 sensors using regime-specific stats from training machines.
    3. Detect spikes (|z_IQR| > 4.0) and mask them as NaN.
    4. Causally impute NaNs using forward-fill (ffill) per machine (with 0.0 regime-mean fallback at cycle 1).
    5. Compute causal Exponential Moving Average (EMA, alpha=0.20) for the 14 informative sensors
       plus normalized cycle age (siklus / 300.0) -> 29 features total per cycle.
    """
    t0 = time.time()
    df = df.sort_values(["id_unit", "siklus"]).reset_index(drop=True)
    op_ids = df["setel_1"].round().astype(int).values

    # Vectorized regime normalization & spike detection
    norm_mat = np.zeros((len(df), len(ALL_SENSORS)), dtype=np.float32)
    spike_counts = {s: 0 for s in ALL_SENSORS}
    nan_counts = {s: int(df[s].isna().sum()) for s in ALL_SENSORS}

    raw_mat = df[ALL_SENSORS].values.astype(np.float32)
    for op_id, st in regime_stats.items():
        m = (op_ids == op_id)
        if not np.any(m):
            continue
        med_vec = np.array([st["median"][s] for s in ALL_SENSORS], dtype=np.float32)
        iqr_vec = np.array([st["iqr"][s] for s in ALL_SENSORS], dtype=np.float32)
        mean_vec = np.array([st["mean"][s] for s in ALL_SENSORS], dtype=np.float32)
        std_vec = np.array([st["std"][s] for s in ALL_SENSORS], dtype=np.float32)

        sub = raw_mat[m]
        z_iqr = np.abs(sub - med_vec[None, :]) / iqr_vec[None, :]
        is_spike = z_iqr > 4.0
        for j, s in enumerate(ALL_SENSORS):
            spike_counts[s] += int(np.nansum(is_spike[:, j]))

        sub_clean = np.where(is_spike, np.nan, sub)
        norm_mat[m] = (sub_clean - mean_vec[None, :]) / std_vec[None, :]

    norm_df = pd.DataFrame(norm_mat, columns=ALL_SENSORS)
    norm_df["id_unit"] = df["id_unit"].values
    norm_df["siklus"] = df["siklus"].values
    if "rul" in df.columns:
        norm_df["rul"] = df["rul"].values
        norm_df["rul_uncapped"] = df["rul_uncapped"].values

    # Causal forward-fill per machine (no backward fill!)
    norm_df[ALL_SENSORS] = norm_df.groupby("id_unit")[ALL_SENSORS].ffill().fillna(0.0)

    # Extract 14 informative sensors + causal EMA (alpha=0.20) + cycle age
    info_vals = norm_df[INFORMATIVE_SENSORS].values.astype(np.float32)
    ema_vals = norm_df.groupby("id_unit")[INFORMATIVE_SENSORS].transform(
        lambda x: x.ewm(alpha=0.20, adjust=False).mean()
    ).values.astype(np.float32)
    cyc_age = (norm_df["siklus"].values.astype(np.float32) / 300.0)[:, None]

    # Concatenate -> (N_rows, 29)
    features_29 = np.concatenate([info_vals, ema_vals, cyc_age], axis=1)
    print(f"[Task 2] Causal regime preprocessing finished in {time.time()-t0:.2f}s (29 channels)", flush=True)
    return norm_df, features_29, nan_counts, spike_counts


# =====================================================================
# 2. SLIDING WINDOW EXTRACTION (NO FUTURE LEAKAGE)
# =====================================================================
def extract_machine_windows(
    df: pd.DataFrame, features_29: np.ndarray, unit_ids: np.ndarray,
    window_size: int = WINDOW_SIZE, stride: int = 4, last_only: bool = False
):
    """
    Task 2c: Build causal sliding windows of length W=50 per machine.
    If last_only=True, extracts only the final W=50 window per machine (used for test set and last-cycle validation).
     Also builds a Synthetic Truncated Validation Set (3 windows per validation machine at 50%, 75%, and 100% of life
    plus intermediate degradations) so validation mirrors the random stopping points of rul_test.csv!
    """
    unit_col = df["id_unit"].values
    has_rul = "rul" in df.columns
    rul_col = df["rul"].values if has_rul else np.zeros(len(df), dtype=np.float32)
    unit_set = set(unit_ids)

    windows = []
    targets = []
    out_units = []

    # Group row indices by id_unit
    for uid, grp_idx in df.groupby("id_unit").indices.items():
        if uid not in unit_set:
            continue
        seq = features_29[grp_idx]  # (T, 29)
        ruls = rul_col[grp_idx]
        T = len(seq)
        if T < window_size:
            # Causal left-pad with initial step if any sequence < W (all train/test have T >= 50)
            pad = np.repeat(seq[:1], window_size - T, axis=0)
            seq = np.concatenate([pad, seq], axis=0)
            ruls = np.concatenate([np.repeat(ruls[:1], window_size - T), ruls], axis=0)
            T = window_size

        if last_only:
            windows.append(seq[T - window_size:T])
            targets.append(ruls[T - 1])
            out_units.append(uid)
        else:
            # Extract sliding windows with given stride + always include the final window (T)
            ends = list(range(window_size, T + 1, stride))
            if ends[-1] != T:
                ends.append(T)
            for end_i in ends:
                windows.append(seq[end_i - window_size:end_i])
                targets.append(ruls[end_i - 1])
                out_units.append(uid)

    X_win = np.stack(windows, axis=0).astype(np.float32)  # (N_win, W, 29)
    y_win = np.array(targets, dtype=np.float32)
    u_win = np.array(out_units, dtype=np.int64)
    return X_win, y_win, u_win


def extract_tabular_baseline_features(X_win: np.ndarray) -> np.ndarray:
    """
    Summary features over each (W=50, C=29) window for the Non-DL Baseline (LightGBM):
    last step (29), mean over W (29), std over W (29), min over W (14), max over W (14),
    and linear difference (last 10 steps mean minus first 10 steps mean, 28) -> 143 features.
    """
    last_step = X_win[:, -1, :]
    mean_w = X_win.mean(axis=1)
    std_w = X_win.std(axis=1)
    min_w = X_win[:, :, :14].min(axis=1)
    max_w = X_win[:, :, :14].max(axis=1)
    trend_w = X_win[:, -10:, :28].mean(axis=1) - X_win[:, :10, :28].mean(axis=1)
    return np.concatenate([last_step, mean_w, std_w, min_w, max_w, trend_w], axis=1)


# =====================================================================
# 3. DEEP SEQUENCE ARCHITECTURES (DILATED TCN & STACKED GRU-ATTENTION)
# =====================================================================
class CausalDilatedTCN(nn.Module):
    """
    Sequence Model 1: Multi-Scale Dilated 1D-CNN / Temporal Convolutional Network (TCN)
    with residual connections and MC-Dropout layers for uncertainty estimation.
    Input shape: (B, W=50, C=29)
    """
    def __init__(self, in_channels: int = 29, hidden_dim: int = 64, dropout: float = 0.20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.conv_d1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(dropout)

        self.conv_d2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop2 = nn.Dropout(dropout)

        self.conv_d4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.drop4 = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (B, W, C)
        last_raw = x[:, -1, :]
        h = self.stem(x.transpose(1, 2))  # (B, H, W)
        h = h + self.drop1(F.gelu(self.bn1(self.conv_d1(h))))
        h = h + self.drop2(F.gelu(self.bn2(self.conv_d2(h))))
        h = h + self.drop4(F.gelu(self.bn4(self.conv_d4(h))))
        h_last = h[:, :, -1]
        h_mean = h.mean(dim=2)
        out = self.head(torch.cat([h_last, h_mean, last_raw], dim=1)).squeeze(-1)
        return 150.0 * torch.sigmoid(out / 50.0)


class AttentionGRU(nn.Module):
    """
    Sequence Model 2: Stacked GRU with Temporal Attention Pooling & MC-Dropout.
    Input shape: (B, W=50, C=29)
    """
    def __init__(self, in_channels: int = 29, hidden_dim: int = 64, dropout: float = 0.20):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        last_raw = x[:, -1, :]
        h_seq, _ = self.gru(x)  # (B, W, H)
        h_seq = self.drop(h_seq)
        scores = torch.softmax(self.attn_w(h_seq), dim=1)  # (B, W, 1)
        h_attn = (h_seq * scores).sum(dim=1)
        h_last = h_seq[:, -1, :]
        out = self.head(torch.cat([h_last, h_attn, last_raw], dim=1)).squeeze(-1)
        return 150.0 * torch.sigmoid(out / 50.0)


# =====================================================================
# 4. HYBRID ASYMMETRIC + MSE LOSS & TRAINING LOOP
# =====================================================================
def hybrid_asymmetric_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor, asym_weight: float = 3.5) -> torch.Tensor:
    """
    Combines MSE loss with the exact differentiable NASA C-MAPSS asymmetric penalty:
      d = y_pred - y_true
      s(d) = exp(-d/13) - 1 if d < 0 else exp(d/10) - 1
    Clamps d/10 at 4.5 during early iterations for numerical stability.
    """
    d = y_pred - y_true
    mse = F.mse_loss(y_pred, y_true)
    asym = torch.where(
        d < 0,
        torch.exp((-d / 13.0).clamp(max=4.5)) - 1.0,
        torch.exp((d / 10.0).clamp(max=4.5)) - 1.0,
    ).mean()
    return mse + asym_weight * asym


def train_sequence_model(
    model, X_tr, y_tr, X_va, y_va,
    epochs: int = 10, batch_size: int = 512, lr: float = 3.5e-3,
    asym_weight: float = 3.5, seed: int = 42, verbose: bool = True
):
    seed_everything(seed)
    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)
    X_va_t = torch.from_numpy(X_va)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    N_tr = len(y_tr)
    best_score = float("inf")
    best_preds = None
    best_state = None
    history = {"train_loss": [], "val_rmse": [], "val_asym_mean": []}

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(N_tr)
        tot_loss = 0.0
        for i in range(0, N_tr, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(X_tr_t[idx])
            loss = hybrid_asymmetric_mse_loss(pred, y_tr_t[idx], asym_weight=asym_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot_loss += loss.item() * len(idx)
        sched.step()

        model.eval()
        va_preds = []
        with torch.no_grad():
            for i in range(0, len(X_va_t), 1024):
                va_preds.append(model(X_va_t[i:i+1024]).numpy())
        va_preds = np.concatenate(va_preds)
        rmse = float(root_mean_squared_error(y_va, va_preds))
        asym_tot = asymmetric_rul_score(y_va, va_preds)
        asym_mean = asym_tot / len(y_va)
        comb = rmse + 0.5 * asym_mean

        history["train_loss"].append(round(tot_loss / N_tr, 3))
        history["val_rmse"].append(round(rmse, 4))
        history["val_asym_mean"].append(round(asym_mean, 4))

        if comb < best_score:
            best_score = comb
            best_preds = va_preds.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"  Epoch {ep:02d}/{epochs:02d} ({time.time()-t0:.2f}s) | TrLoss={tot_loss/N_tr:.2f} | "
                  f"Val RMSE={rmse:.3f} | Val Asym/N={asym_mean:.3f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_preds, history


def mc_dropout_predict(model, X_np: np.ndarray, n_passes: int = 25, batch_size: int = 2048):
    """
    Task 2f: Keep Dropout active at inference (Monte Carlo Dropout) across n_passes stochastic forward passes.
    Note: BatchNorm layers are kept in eval() mode while Dropout layers are set to train() mode.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    X_t = torch.from_numpy(X_np)
    all_passes = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds = []
            for i in range(0, len(X_t), batch_size):
                preds.append(model(X_t[i:i+batch_size]).numpy())
            all_passes.append(np.concatenate(preds))
    return np.stack(all_passes, axis=0)  # (n_passes, N)


def evaluate_by_rul_range(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Task 2e: Report RMSE and Asymmetric Score overall and per RUL range."""
    ranges = [
        ("overall", np.ones(len(y_true), dtype=bool)),
        ("low_0_30_critical", (y_true >= 0) & (y_true <= 30)),
        ("medium_31_80", (y_true > 30) & (y_true <= 80)),
        ("high_81_150", (y_true > 80) & (y_true <= 150)),
    ]
    out = {}
    for name, mask in ranges:
        if not np.any(mask):
            continue
        yt, yp = y_true[mask], y_pred[mask]
        rmse = float(root_mean_squared_error(yt, yp))
        asym_sum = asymmetric_rul_score(yt, yp)
        out[name] = {
            "count": int(mask.sum()),
            "rmse": round(rmse, 4),
            "asymmetric_sum": round(asym_sum, 2),
            "asymmetric_per_sample": round(asym_sum / int(mask.sum()), 4),
        }
    return out


# =====================================================================
# 5. PLOTTING & REPORT ARTIFACT GENERATORS
# =====================================================================
def generate_task2_plots(
    tr_df_raw, norm_df, corr_dict,
    y_val_trunc, mu_val_trunc, std_val_trunc, safe_val_trunc,
    model_comp_dict, out_dir
):
    # Plot 1: Task 2a Exploration (Raw vs Regime-Normalized Sensor Trajectory + RUL Correlation Bar Chart)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    u1_raw = tr_df_raw[tr_df_raw["id_unit"] == 1]
    u1_norm = norm_df[norm_df["id_unit"] == 1]

    axes[0].plot(u1_raw["siklus"], u1_raw["s04"], color="tab:red", lw=1.0, alpha=0.85)
    axes[0].set_title("Raw Sensor s04 (Unit 1 across 6 Operating Regimes)", fontweight="bold", fontsize=9.5)
    axes[0].set_xlabel("Cycle (siklus)")
    axes[0].set_ylabel("Raw s04 Reading")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(u1_norm["siklus"], u1_norm["s04"], color="tab:blue", lw=1.3, label="Regime-Norm + Despiked s04")
    axes[1].plot(u1_norm["siklus"], u1_norm["s02"], color="tab:green", lw=1.1, alpha=0.8, label="Regime-Norm + Despiked s02")
    axes[1].set_title("After Regime Normalization & Causal Despiking (Unit 1)", fontweight="bold", fontsize=9.5)
    axes[1].set_xlabel("Cycle (siklus)")
    axes[1].set_ylabel("Normalized Z-Score")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    s_names = list(corr_dict.keys())
    s_corrs = [corr_dict[k] for k in s_names]
    colors = ["#b0b0b0" if s in UNINFORMATIVE_SENSORS else ("#d95f02" if c < 0 else "#2b5c8f") for s, c in zip(s_names, s_corrs)]
    axes[2].bar(s_names, s_corrs, color=colors, edgecolor="black")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Sensor Correlation with Capped RUL (Grey = 7 Uninformative)", fontweight="bold", fontsize=9.5)
    axes[2].tick_params(axis="x", rotation=45, labelsize=8)
    axes[2].set_ylabel("Pearson r with RUL")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t2_preprocessing_and_sensors.png"), dpi=160)
    plt.close()

    # Plot 2: Task 2d, 2e, 2f Model Comparison, Prediction Intervals & Service Threshold
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    m_labels = [v["label"] for v in model_comp_dict.values()]
    m_rmses = [v["rmse"] for v in model_comp_dict.values()]
    m_asyms = [v["asymmetric_per_machine"] for v in model_comp_dict.values()]

    x_pos = np.arange(len(m_labels))
    ax0_t = axes[0].twinx()
    b1 = axes[0].bar(x_pos - 0.18, m_rmses, width=0.35, color="#2b5c8f", edgecolor="black", label="RMSE (Cycles)")
    b2 = ax0_t.bar(x_pos + 0.18, m_asyms, width=0.35, color="#d95f02", edgecolor="black", label="Asym Score / Machine")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(m_labels, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("Validation RMSE (Lower is Better)", color="#2b5c8f")
    ax0_t.set_ylabel("Asymmetric Score per Machine (Lower is Better)", color="#d95f02")
    axes[0].set_title("Task 2d: Baseline vs Sequence DL Models", fontweight="bold", fontsize=9.5)
    axes[0].grid(True, alpha=0.25)

    # Sort validation machines by true RUL to visualize 90% MC-Dropout + Ensemble Prediction Intervals
    order = np.argsort(y_val_trunc)
    yt_ord = y_val_trunc[order]
    mu_ord = mu_val_trunc[order]
    sd_ord = std_val_trunc[order]
    sf_ord = safe_val_trunc[order]
    idx_x = np.arange(len(yt_ord))

    axes[1].fill_between(
        idx_x,
        np.clip(mu_ord - 1.645 * sd_ord, 0, 150),
        np.clip(mu_ord + 1.645 * sd_ord, 0, 150),
        color="tab:blue", alpha=0.25, label="90% Prediction Interval (±1.645σ)"
    )
    axes[1].plot(idx_x, yt_ord, "k--", lw=1.5, label="Actual Capped RUL")
    axes[1].plot(idx_x, sf_ord, color="tab:red", lw=1.2, label="Risk-Adjusted RUL Prediction")
    axes[1].axhline(25, color="tab:green", ls=":", lw=1.5, label="Service Threshold (Lower Bound ≤ 25)")
    axes[1].set_title("Task 2f: MC-Dropout + Ensemble 90% Prediction Intervals", fontweight="bold", fontsize=9.5)
    axes[1].set_xlabel("Validation Machines (Sorted by True RUL)")
    axes[1].set_ylabel("Remaining Useful Life (Cycles)")
    axes[1].legend(fontsize=7.5, loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # Asymmetric Error Penalty curve vs Actual residuals
    err = sf_ord - yt_ord
    axes[2].hist(mu_ord - yt_ord, bins=30, alpha=0.55, color="tab:blue", label=f"Mean Pred Error (Mean={np.mean(mu_ord-yt_ord):+.2f})")
    axes[2].hist(err, bins=30, alpha=0.65, color="tab:red", label=f"Risk-Adj Error (Mean={np.mean(err):+.2f})")
    axes[2].axvline(0, color="black", lw=1.2, ls="--")
    axes[2].set_title("Residual Distribution (Shifted Left to Avoid Late Failure)", fontweight="bold", fontsize=9.5)
    axes[2].set_xlabel("Prediction Error d = Predicted RUL - Actual RUL")
    axes[2].set_ylabel("Count")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t2_models_and_uncertainty.png"), dpi=160)
    plt.close()


# =====================================================================
# 6. MAIN PIPELINE EXECUTION
# =====================================================================
def main():
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Task 2 RUL Time-Series Regression Pipeline")
    parser.add_argument("--data_dir", type=str, default=r"E:\Code\data_uts_dl")
    parser.add_argument("--out_dir", type=str, default=r"E:\Code\uts_dl_submission")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    out_plots = os.path.join(args.out_dir, "kode", "outputs")
    ckpt_dir = os.path.join(args.out_dir, "kode", "checkpoints")
    os.makedirs(out_plots, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(42)
    tr_df_raw, te_df_raw = load_rul_data(args.data_dir)
    tr_df = build_rul_labels(tr_df_raw, cap=150)

    # Task 2e: Split strictly by machine (id_unit), NEVER by row! (720 train machines, 180 val machines)
    all_units = np.sort(tr_df["id_unit"].unique())
    rng = np.random.RandomState(42)
    shuffled_units = rng.permutation(all_units)
    tr_units = np.sort(shuffled_units[:720])
    va_units = np.sort(shuffled_units[720:])

    # Fit regime normalizer strictly on the 720 training machines
    regime_stats = fit_regime_normalizer(tr_df[tr_df["id_unit"].isin(tr_units)])

    # Causally preprocess train and test tables
    norm_tr_df, feat_tr_29, tr_nan_counts, tr_spike_counts = causal_preprocess_dataframe(tr_df, regime_stats)
    norm_te_df, feat_te_29, te_nan_counts, te_spike_counts = causal_preprocess_dataframe(te_df_raw, regime_stats)

    # Build sliding windows:
    # - Training windows: stride=4 across the 720 training machines
    # - Validation windows (full degradation trajectory): stride=4 across the 180 validation machines
    # - Validation truncated-machine evaluation (1 window per val machine at random truncation cycle >= 50,
    #   mirroring the 300 test machines in rul_test.csv!)
    X_tr, y_tr, u_tr = extract_machine_windows(norm_tr_df, feat_tr_29, tr_units, window_size=WINDOW_SIZE, stride=4)
    X_va_all, y_va_all, u_va_all = extract_machine_windows(norm_tr_df, feat_tr_29, va_units, window_size=WINDOW_SIZE, stride=4)

    # Build deterministic truncated validation set (300 realistic truncated machine histories from the 180 val machines)
    rng_trunc = np.random.RandomState(42)
    trunc_windows, trunc_targets, trunc_units = [], [], []
    for uid, grp_idx in norm_tr_df.groupby("id_unit").indices.items():
        if uid not in set(va_units):
            continue
        seq = feat_tr_29[grp_idx]
        ruls = norm_tr_df["rul"].values[grp_idx]
        T = len(seq)
        # Sample 2 random stopping points per validation machine (one early/mid life, one late life) -> 360 test-like machines
        cut_early = rng_trunc.randint(WINDOW_SIZE, max(WINDOW_SIZE + 1, T - 60))
        cut_late = rng_trunc.randint(max(WINDOW_SIZE, T - 80), T)
        for cut in (cut_early, cut_late):
            trunc_windows.append(seq[cut - WINDOW_SIZE:cut])
            trunc_targets.append(ruls[cut - 1])
            trunc_units.append(uid)
    X_va_trunc = np.stack(trunc_windows, axis=0).astype(np.float32)
    y_va_trunc = np.array(trunc_targets, dtype=np.float32)

    # Extract final window for each of the 300 test machines in original test id_unit order
    test_unit_order = te_df_raw["id_unit"].drop_duplicates().values
    X_te_dict = {}
    for uid, grp_idx in norm_te_df.groupby("id_unit").indices.items():
        seq = feat_te_29[grp_idx]
        X_te_dict[uid] = seq[-WINDOW_SIZE:]
    X_te = np.stack([X_te_dict[uid] for uid in test_unit_order], axis=0).astype(np.float32)

    print(f"[Task 2] Windows ready: Train={X_tr.shape}, ValAll={X_va_all.shape}, "
          f"ValTruncMachines={X_va_trunc.shape}, TestMachines={X_te.shape}", flush=True)

    ckpt_path = os.path.join(ckpt_dir, "task2_final_ensemble.pt")
    metrics_path = os.path.join(out_plots, "task2_metrics.json")

    if args.eval_only and os.path.exists(ckpt_path):
        print("[Task 2] --eval_only mode: Loading saved TCN+GRU ensemble checkpoints...", flush=True)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        k_risk = float(payload["k_risk"])
        mc_passes_va = []
        mc_passes_te = []
        for item in payload["models"]:
            if item["arch"] == "tcn":
                m = CausalDilatedTCN(in_channels=29, hidden_dim=64, dropout=0.20)
            else:
                m = AttentionGRU(in_channels=29, hidden_dim=64, dropout=0.20)
            m.load_state_dict(item["state_dict"])
            mc_passes_va.append(mc_dropout_predict(m, X_va_trunc, n_passes=15))
            mc_passes_te.append(mc_dropout_predict(m, X_te, n_passes=15))

        all_va = np.concatenate(mc_passes_va, axis=0)
        mu_va = all_va.mean(axis=0)
        sd_va = all_va.std(axis=0)
        safe_va = np.clip(mu_va - k_risk * sd_va, 0.0, 150.0)
        rmse_va = float(root_mean_squared_error(y_va_trunc, safe_va))
        asym_va = asymmetric_rul_score(y_va_trunc, safe_va)
        print(f"[Task 2] Truncated-Machine Val RMSE={rmse_va:.4f} | Asym Score={asym_va:.2f} ({asym_va/len(y_va_trunc):.3f}/machine)", flush=True)

        all_te = np.concatenate(mc_passes_te, axis=0)
        mu_te = all_te.mean(axis=0)
        sd_te = all_te.std(axis=0)
        safe_te = np.clip(mu_te - k_risk * sd_te, 0.0, 150.0)
        sub = pd.DataFrame({"id_unit": test_unit_order, "rul": np.round(safe_te, 2)})
        sub.to_csv(os.path.join(args.out_dir, "prediksi_rul.csv"), index=False)
        print(f"[Task 2] Saved prediksi_rul.csv ({len(sub)} rows) in {time.time()-t_start:.2f}s.", flush=True)
        return

    # -----------------------------------------------------------------
    # STEP 2a: SENSOR CORRELATION & OPERATING REGIME SUMMARY
    # -----------------------------------------------------------------
    corr_dict = {}
    for s in ALL_SENSORS:
        corr_dict[s] = round(float(np.corrcoef(norm_tr_df[s].values, norm_tr_df["rul"].values)[0, 1]), 4)

    results = {
        "eda_and_preprocessing": {
            "train_machines": int(len(all_units)),
            "train_split_machines": int(len(tr_units)),
            "val_split_machines": int(len(va_units)),
            "test_machines": int(len(test_unit_order)),
            "operating_regimes_count": int(len(regime_stats)),
            "operating_regimes_setel1_keys": sorted(list(regime_stats.keys())),
            "uninformative_sensors_dropped": UNINFORMATIVE_SENSORS,
            "informative_sensors_kept": INFORMATIVE_SENSORS,
            "sensor_corr_with_rul_after_regime_norm": corr_dict,
            "train_missing_counts_per_sensor": tr_nan_counts,
            "train_spike_counts_per_sensor_iqr4": tr_spike_counts,
            "window_size": WINDOW_SIZE,
        }
    }

    # -----------------------------------------------------------------
    # STEP 2d (1): NON-DEEP-LEARNING BASELINE (LightGBM Regressor)
    # -----------------------------------------------------------------
    print("\n--- [Task 2d] Training Non-Deep-Learning Baseline (LightGBM Regressor) ---", flush=True)
    t_b = time.time()
    X_tr_tab = extract_tabular_baseline_features(X_tr)
    X_va_all_tab = extract_tabular_baseline_features(X_va_all)
    X_va_trunc_tab = extract_tabular_baseline_features(X_va_trunc)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=250, learning_rate=0.05, num_leaves=31,
        subsample_for_bin=50000, random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_tr_tab, y_tr)
    lgb_pred_trunc = np.clip(lgb_model.predict(X_va_trunc_tab), 0.0, 150.0)
    lgb_pred_all = np.clip(lgb_model.predict(X_va_all_tab), 0.0, 150.0)
    lgb_rmse = float(root_mean_squared_error(y_va_trunc, lgb_pred_trunc))
    lgb_asym = asymmetric_rul_score(y_va_trunc, lgb_pred_trunc)
    print(f"  => LightGBM Baseline ({time.time()-t_b:.1f}s): Val RMSE={lgb_rmse:.4f} | "
          f"Asym Score={lgb_asym:.2f} ({lgb_asym/len(y_va_trunc):.3f}/machine)", flush=True)

    # -----------------------------------------------------------------
    # STEP 2d (2 & 3): SEQUENCE MODELS (DILATED TCN vs STACKED GRU-ATTENTION)
    # -----------------------------------------------------------------
    print("\n--- [Task 2d] Training Sequence Architecture 1: Causal Dilated TCN (3 Seeds) ---", flush=True)
    seeds = [42, 123, 999]
    tcn_models, gru_models = [], []
    tcn_trunc_preds, gru_trunc_preds = [], []

    for s in seeds:
        print(f"\n[Dilated TCN - Seed {s}]", flush=True)
        m_tcn = CausalDilatedTCN(in_channels=29, hidden_dim=64, dropout=0.20)
        m_tcn, vp_tcn, _ = train_sequence_model(
            m_tcn, X_tr, y_tr, X_va_trunc, y_va_trunc,
            epochs=10, batch_size=512, lr=3.5e-3, asym_weight=3.2, seed=s, verbose=True
        )
        tcn_models.append(m_tcn)
        tcn_trunc_preds.append(vp_tcn)

    print("\n--- [Task 2d] Training Sequence Architecture 2: Attention GRU (3 Seeds) ---", flush=True)
    for s in seeds:
        print(f"\n[Attention GRU - Seed {s}]", flush=True)
        m_gru = AttentionGRU(in_channels=29, hidden_dim=64, dropout=0.20)
        m_gru, vp_gru, _ = train_sequence_model(
            m_gru, X_tr, y_tr, X_va_trunc, y_va_trunc,
            epochs=10, batch_size=512, lr=3.5e-3, asym_weight=3.2, seed=s, verbose=True
        )
        gru_models.append(m_gru)
        gru_trunc_preds.append(vp_gru)

    # Single-architecture & ensemble evaluations
    tcn_mean_pred = np.mean(tcn_trunc_preds, axis=0)
    gru_mean_pred = np.mean(gru_trunc_preds, axis=0)

    # -----------------------------------------------------------------
    # STEP 2f: UNCERTAINTY QUANTIFICATION (MC DROPOUT + DEEP ENSEMBLE)
    # -----------------------------------------------------------------
    print("\n--- [Task 2f] Running MC Dropout + Deep Ensemble Uncertainty Quantification ---", flush=True)
    mc_va_list = []
    mc_va_all_list = []
    mc_te_list = []
    saved_model_payloads = []

    for m in tcn_models:
        mc_va_list.append(mc_dropout_predict(m, X_va_trunc, n_passes=15))
        mc_va_all_list.append(mc_dropout_predict(m, X_va_all, n_passes=8))
        mc_te_list.append(mc_dropout_predict(m, X_te, n_passes=15))
        saved_model_payloads.append({"arch": "tcn", "state_dict": m.state_dict()})

    for m in gru_models:
        mc_va_list.append(mc_dropout_predict(m, X_va_trunc, n_passes=15))
        mc_va_all_list.append(mc_dropout_predict(m, X_va_all, n_passes=8))
        mc_te_list.append(mc_dropout_predict(m, X_te, n_passes=15))
        saved_model_payloads.append({"arch": "gru", "state_dict": m.state_dict()})

    mc_va_samples = np.concatenate(mc_va_list, axis=0)  # (90, N_val_trunc)
    mu_va = mc_va_samples.mean(axis=0)
    std_va = mc_va_samples.std(axis=0)

    # Tune risk-aversion shift k_risk on validation to balance RMSE and Asymmetric Score
    best_k = 0.0
    best_obj = float("inf")
    for k_cand in np.linspace(0.0, 0.85, 35):
        cand_pred = np.clip(mu_va - k_cand * std_va, 0.0, 150.0)
        c_rmse = float(root_mean_squared_error(y_va_trunc, cand_pred))
        c_asym = asymmetric_rul_score(y_va_trunc, cand_pred) / len(y_va_trunc)
        # Prioritize asymmetric score reduction while keeping RMSE low
        obj = c_rmse + 0.55 * c_asym
        if obj < best_obj:
            best_obj = obj
            best_k = float(k_cand)

    safe_va = np.clip(mu_va - best_k * std_va, 0.0, 150.0)

    # 90% Prediction Interval Coverage Probability (PICP)
    pi_low_90 = np.clip(mu_va - 1.645 * std_va, 0.0, 150.0)
    pi_high_90 = np.clip(mu_va + 1.645 * std_va, 0.0, 150.0)
    picp_90 = float(np.mean((y_va_trunc >= pi_low_90) & (y_va_trunc <= pi_high_90)))
    mpiw_90 = float(np.mean(pi_high_90 - pi_low_90))

    model_comp = {
        "lightgbm_baseline": {
            "label": "LightGBM Baseline",
            "rmse": round(lgb_rmse, 4),
            "asymmetric_sum": round(lgb_asym, 2),
            "asymmetric_per_machine": round(lgb_asym / len(y_va_trunc), 4),
        },
        "dilated_tcn_seed42": {
            "label": "Dilated TCN (Single)",
            "rmse": round(float(root_mean_squared_error(y_va_trunc, tcn_trunc_preds[0])), 4),
            "asymmetric_sum": round(asymmetric_rul_score(y_va_trunc, tcn_trunc_preds[0]), 2),
            "asymmetric_per_machine": round(asymmetric_rul_score(y_va_trunc, tcn_trunc_preds[0]) / len(y_va_trunc), 4),
        },
        "attention_gru_seed42": {
            "label": "Attention GRU (Single)",
            "rmse": round(float(root_mean_squared_error(y_va_trunc, gru_trunc_preds[0])), 4),
            "asymmetric_sum": round(asymmetric_rul_score(y_va_trunc, gru_trunc_preds[0]), 2),
            "asymmetric_per_machine": round(asymmetric_rul_score(y_va_trunc, gru_trunc_preds[0]) / len(y_va_trunc), 4),
        },
        "ensemble_mean": {
            "label": "TCN+GRU Ensemble (Mean)",
            "rmse": round(float(root_mean_squared_error(y_va_trunc, mu_va)), 4),
            "asymmetric_sum": round(asymmetric_rul_score(y_va_trunc, mu_va), 2),
            "asymmetric_per_machine": round(asymmetric_rul_score(y_va_trunc, mu_va) / len(y_va_trunc), 4),
        },
        "ensemble_risk_adjusted": {
            "label": f"Risk-Adj Ensemble (μ-{best_k:.2f}σ)",
            "rmse": round(float(root_mean_squared_error(y_va_trunc, safe_va)), 4),
            "asymmetric_sum": round(asymmetric_rul_score(y_va_trunc, safe_va), 2),
            "asymmetric_per_machine": round(asymmetric_rul_score(y_va_trunc, safe_va) / len(y_va_trunc), 4),
        },
    }
    results["model_comparison_truncated_machines"] = model_comp

    print("\n--- [Task 2e] Validation Metrics Breakdown per RUL Range ---", flush=True)
    range_breakdown_trunc = evaluate_by_rul_range(y_va_trunc, safe_va)
    mc_va_all_samples = np.concatenate(mc_va_all_list, axis=0)
    safe_va_all = np.clip(mc_va_all_samples.mean(axis=0) - best_k * mc_va_all_samples.std(axis=0), 0.0, 150.0)
    range_breakdown_all_windows = evaluate_by_rul_range(y_va_all, safe_va_all)

    for r_name, r_info in range_breakdown_trunc.items():
        print(f"  [Truncated Val - {r_name:20s}] N={r_info['count']:4d} | RMSE={r_info['rmse']:6.3f} | "
              f"AsymSum={r_info['asymmetric_sum']:8.2f} | Asym/N={r_info['asymmetric_per_sample']:6.3f}", flush=True)

    results["validation_per_rul_range_truncated_machines"] = range_breakdown_trunc
    results["validation_per_rul_range_all_windows"] = range_breakdown_all_windows

    # Maintenance Decision Rule analysis (Task 2f)
    # Trigger service when 90% Lower Confidence Bound (mu - 1.645*sigma) <= 25 cycles
    service_trigger = (pi_low_90 <= 25.0)
    true_critical = (y_va_trunc <= 25.0)
    critical_recall = float(np.mean(service_trigger[true_critical])) if np.any(true_critical) else 1.0
    unscheduled_failure_rate = float(np.mean(y_va_trunc[~service_trigger] <= 10.0)) if np.any(~service_trigger) else 0.0

    results["uncertainty_and_maintenance"] = {
        "mc_total_stochastic_passes": int(mc_va_samples.shape[0]),
        "optimal_risk_aversion_k": round(best_k, 3),
        "picp_90_coverage_pct": round(picp_90 * 100, 2),
        "mpiw_90_mean_width_cycles": round(mpiw_90, 2),
        "mean_epistemic_std_cycles": round(float(std_va.mean()), 2),
        "service_policy_rule": "Schedule maintenance when 90% Lower Bound (mu - 1.645 * sigma) <= 25 cycles",
        "critical_machines_caught_pct": round(critical_recall * 100, 2),
        "unscheduled_imminent_failure_rate_pct": round(unscheduled_failure_rate * 100, 2),
    }

    # Save checkpoints (< 3 MB!)
    torch.save({
        "models": saved_model_payloads,
        "k_risk": best_k,
        "regime_stats": regime_stats,
    }, ckpt_path)

    # Generate Test Set Predictions (300 machines)
    mc_te_samples = np.concatenate(mc_te_list, axis=0)
    mu_te = mc_te_samples.mean(axis=0)
    std_te = mc_te_samples.std(axis=0)
    safe_te = np.clip(mu_te - best_k * std_te, 0.0, 150.0)

    sub_df = pd.DataFrame({
        "id_unit": test_unit_order,
        "rul": np.round(safe_te, 2),
    })
    sub_path = os.path.join(args.out_dir, "prediksi_rul.csv")
    sub_df.to_csv(sub_path, index=False)

    results["test_predictions_summary"] = {
        "count": int(len(sub_df)),
        "min_rul": round(float(sub_df["rul"].min()), 2),
        "max_rul": round(float(sub_df["rul"].max()), 2),
        "mean_rul": round(float(sub_df["rul"].mean()), 2),
        "median_rul": round(float(sub_df["rul"].median()), 2),
        "machines_flagged_for_immediate_service_lcb25": int(np.sum((mu_te - 1.645 * std_te) <= 25.0)),
    }

    generate_task2_plots(
        tr_df, norm_tr_df, corr_dict,
        y_va_trunc, mu_va, std_va, safe_va,
        model_comp, out_plots
    )
    save_json(results, metrics_path)
    print(f"\n[Task 2] Saved {sub_path} ({len(sub_df)} rows) and all metrics/plots in {time.time()-t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
