#!/usr/bin/env python3
"""
Task 2 – Time-Series Regression: Remaining Useful Life (RUL) Prediction (30 points)
Complete, reproducible Deep Learning pipeline covering:
  a. Exploration & Causal Preprocessing:
     - 6 discrete operating regimes (setel_1..3) identification
     - Regime-conditioned normalization fitted strictly on training machines (no leakage)
     - Informative (14) vs Uninformative (7) sensor selection (|r| < 0.01 dropped)
     - Causal despiking (|z| > 4.0) and forward-fill imputation (ffill per unit)
     - Causal EMA smoothing + normalized cycle counter
  b. Target formulation:
     - Piecewise-linear RUL label: RUL = min(150, max_cycle - cycle)
  c. Sliding windows:
     - Causal sliding window W = 50 (fits all test machines without padding)
  d. Model comparison:
     - Non-DL Baseline: LightGBM Regressor on window summary features
     - Sequence Model 1: Dilated 1D Temporal Convolutional Network (TCN) with residual blocks
     - Sequence Model 2: Stacked BiGRU with Temporal Attention Pooling
     - Hybrid Asymmetric + MSE training loss
  e. Machine-level validation & per-range breakdown:
     - 80/20 train/val split by id_unit (720 train / 180 val machines)
     - Evaluation on both full trajectories and final cycles
     - Performance breakdown across Low [0, 30], Medium (30, 80], High (80, 150]
  f. Uncertainty quantification & Maintenance policy:
     - MC Dropout (T=30 stochastic forward passes) + Multi-Seed/Multi-Model Ensemble
     - Conservative risk-adjusted prediction to minimize NASA C-MAPSS asymmetric penalty
     - 90% prediction intervals and concrete maintenance trigger threshold policy
  g. Checkpoints, metrics JSON, and prediksi_rul.csv generation.
"""
import os
import sys
import time
import argparse
import tempfile

# Prevent matplotlib access denied on Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_cache"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import seed_everything, asymmetric_rul_score, save_json

# All 21 sensors
ALL_SENSORS = [f"s{i:02d}" for i in range(1, 22)]

# 7 uninformative sensors (|r| < 0.01 after regime normalization)
UNINFORMATIVE_SENSORS = ["s01", "s05", "s06", "s10", "s16", "s18", "s19"]

# 14 informative sensors
INFORMATIVE_SENSORS = [
    "s02", "s03", "s04", "s07", "s08", "s09", "s11",
    "s12", "s13", "s14", "s15", "s17", "s20", "s21"
]

WINDOW_SIZE = 50
MAX_RUL = 150.0


# =====================================================================
# 1. DATA LOADING & REGIME CONDITIONED PREPROCESSING
# =====================================================================
def resolve_file(base_dir: str, filename: str) -> str:
    """Finds file whether directly in base_dir, in a subfolder, or with .gz suffix."""
    candidates = [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, filename, filename),
        os.path.join(base_dir, filename + ".gz"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Could not resolve {filename} in {base_dir}")


def load_task2_data(data_dir: str):
    t0 = time.time()
    p_tr = resolve_file(data_dir, "rul_train.csv")
    p_te = resolve_file(data_dir, "rul_test.csv")
    df_train = pd.read_csv(p_tr)
    df_test = pd.read_csv(p_te)
    print(f"[Task 2] Loaded train ({len(df_train)} rows, {df_train['id_unit'].nunique()} units) & "
          f"test ({len(df_test)} rows, {df_test['id_unit'].nunique()} units) in {time.time()-t0:.2f}s", flush=True)
    return df_train, df_test


def compute_regime_stats(df_train: pd.DataFrame, train_units: np.ndarray):
    """
    Computes per-regime median and IQR strictly on the training units.
    Regimes are defined by round(setel_1) in {0, 10, 20, 25, 35, 42}.
    """
    df_fit = df_train[df_train["id_unit"].isin(train_units)].copy()
    df_fit["regime"] = df_fit["setel_1"].round().astype(int)

    stats = {}
    for r_id in [0, 10, 20, 25, 35, 42]:
        sub = df_fit[df_fit["regime"] == r_id]
        r_stats = {}
        for s in ALL_SENSORS:
            vals = sub[s].dropna().values
            if len(vals) > 0:
                q25, med, q75 = np.percentile(vals, [25, 50, 75])
                iqr = max(float(q75 - q25), 1e-4)
                r_stats[s] = {"median": float(med), "iqr": iqr}
            else:
                r_stats[s] = {"median": 0.0, "iqr": 1.0}
        stats[r_id] = r_stats
    return stats


def preprocess_timeseries(df: pd.DataFrame, stats: dict, is_train: bool = True):
    """
    Performs causal cleaning per unit:
      1. Regime-conditioned normalization
      2. Causal despiking: replace |z| > 4.0 with NaN
      3. Causal imputation: forward-fill per unit (fallback 0.0)
      4. Causal EMA smoothing (alpha=0.20)
      5. Cycle age normalized (siklus / 300.0)
    """
    df = df.copy()
    df["regime"] = df["setel_1"].round().astype(int)

    # Normalize sensors by regime
    for s in INFORMATIVE_SENSORS:
        norm_col = np.zeros(len(df), dtype=np.float32)
        for r_id, r_stats in stats.items():
            mask = (df["regime"] == r_id).values
            if np.any(mask):
                med = r_stats[s]["median"]
                iqr = r_stats[s]["iqr"]
                raw = df.loc[mask, s].values
                z = (raw - med) / iqr
                # Causal despiking: extreme spikes become NaN
                z[np.abs(z) > 4.0] = np.nan
                norm_col[mask] = z
        df[f"{s}_norm"] = norm_col

    norm_cols = [f"{s}_norm" for s in INFORMATIVE_SENSORS]

    # Causal forward-fill per machine
    df[norm_cols] = df.groupby("id_unit")[norm_cols].ffill().fillna(0.0)

    # Add causal Exponential Moving Average (alpha=0.20) per unit
    ema_cols = []
    for s in INFORMATIVE_SENSORS:
        ema_name = f"{s}_ema"
        ema_cols.append(ema_name)
        df[ema_name] = df.groupby("id_unit")[f"{s}_norm"].transform(
            lambda x: x.ewm(alpha=0.20, adjust=False).mean()
        )

    # Normalized cycle count
    df["cycle_norm"] = df["siklus"].astype(np.float32) / 300.0

    feature_cols = norm_cols + ema_cols + ["cycle_norm"]

    if is_train:
        # Compute true RUL label per unit
        max_cycles = df.groupby("id_unit")["siklus"].transform("max")
        df["rul_raw"] = max_cycles - df["siklus"]
        df["rul"] = df["rul_raw"].clip(upper=MAX_RUL).astype(np.float32)

    return df, feature_cols


def extract_windows(df: pd.DataFrame, feature_cols: list, is_train: bool = True, stride: int = 3):
    """
    Extracts sliding windows of length W=50.
    For training:
      - Window stride 3 for cycles near end of life (RUL < 150)
      - Window stride 5 for healthy cycles (RUL >= 150)
      - Always includes the final cycle window of each machine.
    For testing:
      - Takes exactly the final window of length 50 for each test machine.
    """
    X_windows, y_windows, unit_ids, end_cycles = [], [], [], []

    grouped = df.groupby("id_unit")
    for u_id, group in grouped:
        feat_arr = group[feature_cols].values.astype(np.float32)
        n_cycles = len(group)

        if not is_train:
            # Test: take the last W cycles
            if n_cycles < WINDOW_SIZE:
                # Pad left with first row if ever needed (data guarantee is >= 50)
                pad_len = WINDOW_SIZE - n_cycles
                pad = np.tile(feat_arr[0:1], (pad_len, 1))
                w = np.vstack([pad, feat_arr])
            else:
                w = feat_arr[-WINDOW_SIZE:]
            X_windows.append(w)
            unit_ids.append(u_id)
            end_cycles.append(group["siklus"].iloc[-1])
        else:
            rul_arr = group["rul"].values.astype(np.float32)
            # Find candidate start positions
            end_indices = []
            for end_idx in range(WINDOW_SIZE, n_cycles):
                current_rul = rul_arr[end_idx - 1]
                st = stride if current_rul < 150.0 else (stride * 2)
                if (end_idx - WINDOW_SIZE) % st == 0:
                    end_indices.append(end_idx)

            # Always ensure the last cycle is included
            if n_cycles not in end_indices:
                end_indices.append(n_cycles)

            for end_idx in end_indices:
                start_idx = end_idx - WINDOW_SIZE
                w = feat_arr[start_idx:end_idx]
                r = rul_arr[end_idx - 1]
                X_windows.append(w)
                y_windows.append(r)
                unit_ids.append(u_id)
                end_cycles.append(group["siklus"].iloc[end_idx - 1])

    X_windows = np.array(X_windows, dtype=np.float32)
    if is_train:
        y_windows = np.array(y_windows, dtype=np.float32)
        return X_windows, y_windows, np.array(unit_ids), np.array(end_cycles)
    else:
        return X_windows, np.array(unit_ids), np.array(end_cycles)


# =====================================================================
# 2. MODELS: LIGHTGBM BASELINE, DILATED TCN, STACKED BIGRU + ATTENTION
# =====================================================================
def extract_summary_features(X_windows: np.ndarray) -> np.ndarray:
    """
    Extracts tabular summary statistics over window W=50:
    last value, EMA value, window mean, window std, window min, window max, and linear slope.
    """
    N, W, D = X_windows.shape
    last_vals = X_windows[:, -1, :]  # (N, D)
    mean_vals = X_windows.mean(axis=1)  # (N, D)
    std_vals = X_windows.std(axis=1)  # (N, D)
    min_vals = X_windows.min(axis=1)  # (N, D)
    max_vals = X_windows.max(axis=1)  # (N, D)

    # Linear slope over the window: (last - first) / W
    slope_vals = (X_windows[:, -1, :] - X_windows[:, 0, :]) / float(W)

    feats = np.hstack([last_vals, mean_vals, std_vals, min_vals, max_vals, slope_vals])
    return feats.astype(np.float32)


class Chomp1d(nn.Module):
    """Removes future padding to enforce strict causal convolution."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.20):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class DilatedTCN(nn.Module):
    """
    Dilated 1D Temporal Convolutional Network (TCN) with residual blocks.
    Strictly causal convolutions with dilations [1, 2, 4, 8].
    """
    def __init__(self, in_features=29, num_channels=(48, 64, 64, 64), kernel_size=3, dropout=0.20):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = in_features if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                                        padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (N, W, D) -> Conv1d expects (N, D, W)
        x = x.transpose(1, 2)
        feat = self.network(x)  # (N, C, W)
        last_step = feat[:, :, -1]  # causal last step
        out = self.head(last_step).squeeze(-1)
        return torch.sigmoid(out) * MAX_RUL


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (N, W, H)
        weights = F.softmax(self.att(x), dim=1)  # (N, W, 1)
        pooled = torch.sum(weights * x, dim=1)  # (N, H)
        return pooled


class BiGRUAttention(nn.Module):
    """
    2-Layer Stacked Bidirectional GRU with Temporal Attention Pooling and Dropout.
    """
    def __init__(self, in_features=29, hidden_dim=48, num_layers=2, dropout=0.20):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.att_pool = AttentionPooling(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)  # (N, W, hidden*2)
        pooled = self.att_pool(out)  # (N, hidden*2)
        pooled = self.dropout(pooled)
        pred = self.head(pooled).squeeze(-1)
        return torch.sigmoid(pred) * MAX_RUL


# =====================================================================
# 3. HYBRID ASYMMETRIC LOSS FUNCTION
# =====================================================================
class HybridRULLoss(nn.Module):
    """
    Combines MSE Loss with smooth asymmetric penalization:
    penalizes late predictions (y_pred > y_true, d > 0) more severely
    matching the NASA C-MAPSS / industrial maintenance scoring rubric.
    """
    def __init__(self, alpha=0.35):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        mse = F.mse_loss(y_pred, y_true)
        d = y_pred - y_true
        asym = torch.where(d < 0, torch.exp(-d / 13.0) - 1.0, torch.exp(d / 10.0) - 1.0)
        return mse + self.alpha * asym.mean()


# =====================================================================
# 4. TRAINING & EVALUATION ROUTINES
# =====================================================================
def train_dl_model(model, train_loader, val_loader, epochs=15, lr=2e-3, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = HybridRULLoss(alpha=0.30)

    best_val_rmse = float("inf")
    best_state = None
    history = {"train_loss": [], "val_rmse": [], "val_asym": []}

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_samples = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            train_loss += loss.item() * len(by)
            n_samples += len(by)
        scheduler.step()
        train_loss /= max(n_samples, 1)

        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                pred = model(bx.to(device))
                val_preds.append(pred.cpu().numpy())
                val_trues.append(by.numpy())
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)

        val_rmse = float(np.sqrt(mean_squared_error(val_trues, val_preds)))
        val_asym = float(asymmetric_rul_score(val_trues, val_preds) / len(val_trues))

        history["train_loss"].append(round(train_loss, 4))
        history["val_rmse"].append(round(val_rmse, 4))
        history["val_asym"].append(round(val_asym, 4))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history


def mc_dropout_predict(model, X_tensor: torch.Tensor, T: int = 30):
    """
    Performs T stochastic forward passes with dropout enabled to estimate
    predictive mean mu and standard deviation sigma (epistemic uncertainty).
    """
    dev = next(model.parameters()).device
    model.train()  # Keeps dropout active
    preds = []
    with torch.no_grad():
        for _ in range(T):
            out = model(X_tensor.to(dev)).cpu().numpy()
            preds.append(out)
    preds = np.array(preds)  # (T, N)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0)
    model.eval()
    return mu, sigma


def evaluate_metrics_per_range(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes overall RMSE, MAE, Asymmetric Score, and per-range breakdown:
    Low [0, 30], Medium (30, 80], High (80, 150].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    total_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    total_mae = float(mean_absolute_error(y_true, y_pred))
    total_asym = float(asymmetric_rul_score(y_true, y_pred))
    mean_asym = float(total_asym / len(y_true))

    ranges = {
        "low_0_30": (y_true >= 0.0) & (y_true <= 30.0),
        "med_30_80": (y_true > 30.0) & (y_true <= 80.0),
        "high_80_150": (y_true > 80.0) & (y_true <= 150.0),
    }

    range_metrics = {}
    for r_name, mask in ranges.items():
        if np.any(mask):
            sub_true, sub_pred = y_true[mask], y_pred[mask]
            sub_rmse = float(np.sqrt(mean_squared_error(sub_true, sub_pred)))
            sub_mae = float(mean_absolute_error(sub_true, sub_pred))
            sub_asym = float(asymmetric_rul_score(sub_true, sub_pred))
            range_metrics[r_name] = {
                "count": int(np.sum(mask)),
                "rmse": round(sub_rmse, 3),
                "mae": round(sub_mae, 3),
                "asymmetric_score": round(sub_asym, 2),
                "asymmetric_per_unit": round(sub_asym / np.sum(mask), 2),
            }
        else:
            range_metrics[r_name] = {"count": 0, "rmse": 0.0, "mae": 0.0, "asymmetric_score": 0.0}

    return {
        "rmse": round(total_rmse, 3),
        "mae": round(total_mae, 3),
        "asymmetric_score_total": round(total_asym, 2),
        "asymmetric_score_mean": round(mean_asym, 2),
        "ranges": range_metrics,
    }


# =====================================================================
# 5. EDA & VISUALIZATION GENERATORS
# =====================================================================
def generate_t2_eda_plots(df_train: pd.DataFrame, stats: dict, out_dir: str):
    """
    Plots:
      1. Sensor correlation before vs after regime normalization (showing 7 dead vs 14 informative)
      2. 6 Discrete operating regimes in 3D/2D settings
      3. Sensor raw vs causal despiked waveform
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # Plot 1: Sensor correlation with RUL
    df_sample = df_train.sample(n=min(30000, len(df_train)), random_state=42).copy()
    max_c = df_sample.groupby("id_unit")["siklus"].transform("max")
    rul_clip = (max_c - df_sample["siklus"]).clip(upper=MAX_RUL).values

    raw_corrs = []
    norm_corrs = []
    for s in ALL_SENSORS:
        raw_vals = df_sample[s].fillna(df_sample[s].median()).values
        c_raw = np.corrcoef(raw_vals, rul_clip)[0, 1]
        raw_corrs.append(c_raw if not np.isnan(c_raw) else 0.0)

        # Normalized by regime
        norm_v = np.zeros(len(df_sample))
        for r_id, r_st in stats.items():
            m = (df_sample["setel_1"].round().astype(int) == r_id).values
            norm_v[m] = (df_sample.loc[m, s].values - r_st[s]["median"]) / r_st[s]["iqr"]
        norm_v = np.nan_to_num(norm_v, nan=0.0)
        c_norm = np.corrcoef(norm_v, rul_clip)[0, 1]
        norm_corrs.append(c_norm if not np.isnan(c_norm) else 0.0)

    x = np.arange(len(ALL_SENSORS))
    w = 0.38
    axes[0].bar(x - w / 2, raw_corrs, width=w, label="Raw Sensor Correlation", color="#94a3b8")
    axes[0].bar(x + w / 2, norm_corrs, width=w, label="Regime-Norm Correlation", color="#2563eb")
    axes[0].axhline(0.01, color="red", ls="--", lw=1, alpha=0.7)
    axes[0].axhline(-0.01, color="red", ls="--", lw=1, alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ALL_SENSORS, rotation=60, fontsize=8)
    axes[0].set_title("Correlation with Capped RUL (14 vs 7 Sensors)", fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Pearson Correlation (r)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: 6 Discrete Operating Regimes
    regime_counts = df_train["setel_1"].round().astype(int).value_counts().sort_index()
    regime_labels = [f"R{i} (setel_1={r})" for i, r in enumerate(regime_counts.index)]
    colors = ["#1e40af", "#0284c7", "#0d9488", "#16a34a", "#ca8a04", "#dc2626"]
    axes[1].pie(regime_counts.values, labels=regime_labels, autopct="%1.1f%%", colors=colors,
                startangle=140, textprops={"fontsize": 8})
    axes[1].set_title("6 Discrete Operating Regimes (~16.7% each)", fontweight="bold", fontsize=10)

    # Plot 3: Example Machine Sensor Degradation Waveform + Spikes
    u1 = df_train[df_train["id_unit"] == 1].copy()
    axes[2].plot(u1["siklus"], u1["s04"], label="Raw s04 (with regime jumps)", color="#cbd5e1", lw=1.2)
    # Regime normalized
    s04_norm = np.zeros(len(u1))
    for r_id, r_st in stats.items():
        m = (u1["setel_1"].round().astype(int) == r_id).values
        s04_norm[m] = (u1.loc[m, "s04"].values - r_st["s04"]["median"]) / r_st["s04"]["iqr"]
    axes[2].plot(u1["siklus"], s04_norm, label="Regime-Normalized s04", color="#ef4444", lw=1.5)
    axes[2].set_title("Unit 1: s04 Regime Normalization", fontweight="bold", fontsize=10)
    axes[2].set_xlabel("Cycle")
    axes[2].set_ylabel("Sensor Value / Z-score")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t2_eda_regimes_and_spikes.png"), dpi=160)
    plt.close()


def generate_t2_predictions_plots(sanity_losses, tcn_hist, bigru_hist, y_val_true, y_val_pred, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # Plot 1: 1-batch Overfitting Sanity Check
    axes[0].plot(range(1, len(sanity_losses) + 1), sanity_losses, "r-o", lw=2, label="1-Batch Loss")
    axes[0].set_title("Sanity Check: Overfitting 1 Batch (N=64)", fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Hybrid Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # Plot 2: Learning Curves (TCN vs BiGRU)
    ep_x = range(1, len(tcn_hist["val_rmse"]) + 1)
    axes[1].plot(ep_x, tcn_hist["val_rmse"], "b-s", label="TCN Val RMSE")
    axes[1].plot(ep_x, bigru_hist["val_rmse"], "g-^", label="BiGRU Val RMSE")
    axes[1].set_title("Validation RMSE over Epochs", fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (Cycles)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Scatter True vs Predicted RUL
    axes[2].scatter(y_val_true, y_val_pred, alpha=0.4, s=14, color="#1e40af", edgecolors="none")
    axes[2].plot([0, 150], [0, 150], "r--", lw=1.8, label="Ideal y=x")
    axes[2].set_title("Validation Trajectory: True vs Predicted RUL", fontweight="bold", fontsize=10)
    axes[2].set_xlabel("True RUL (Cycles)")
    axes[2].set_ylabel("Predicted RUL (Cycles)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t2_model_learning_and_predictions.png"), dpi=160)
    plt.close()


def generate_t2_uncertainty_plots(y_final_true, y_final_mu, y_final_sig, best_k, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    # Sort validation final units by true RUL
    sort_idx = np.argsort(y_final_true)
    y_true_s = y_final_true[sort_idx]
    mu_s = y_final_mu[sort_idx]
    sig_s = y_final_sig[sort_idx]
    safe_pred = np.clip(mu_s - best_k * sig_s, 0.0, MAX_RUL)

    # 90% Confidence Interval: +/- 1.645 sigma
    lower_90 = np.clip(mu_s - 1.645 * sig_s, 0.0, MAX_RUL)
    upper_90 = np.clip(mu_s + 1.645 * sig_s, 0.0, MAX_RUL)

    x = np.arange(len(y_true_s))
    axes[0].fill_between(x, lower_90, upper_90, color="#93c5fd", alpha=0.5, label="90% Predictive Interval")
    axes[0].plot(x, y_true_s, "k-", lw=2, label="True Final RUL")
    axes[0].plot(x, mu_s, "b--", lw=1.2, label="Predictive Mean (Ensemble)")
    axes[0].plot(x, safe_pred, "r-", lw=1.5, label=f"Safe Risk-Adjusted (k={best_k:.2f})")
    axes[0].set_title("Validation Final Cycle: 90% Confidence Intervals", fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Validation Machines (Sorted by True RUL)")
    axes[0].set_ylabel("RUL (Cycles)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Maintenance Policy Trigger Curve
    # Trigger maintenance when lower bound <= threshold tau (e.g. 25 cycles)
    tau_vals = np.linspace(5, 50, 46)
    early_service_pct, late_failures = [], []
    for tau in tau_vals:
        service_triggered = lower_90 <= tau
        # False alarm / early service when true RUL > tau + 15
        early = np.sum(service_triggered & (y_true_s > (tau + 15))) / max(len(y_true_s), 1) * 100
        # Missed failure when machine failed (true RUL < 10) but not triggered
        missed = np.sum((~service_triggered) & (y_true_s <= 10))
        early_service_pct.append(early)
        late_failures.append(missed)

    ax2_twin = axes[1].twinx()
    axes[1].plot(tau_vals, early_service_pct, "b-", lw=2, label="Preventive Maintenance Rate (%)")
    ax2_twin.plot(tau_vals, late_failures, "r--", lw=2, label="Unscheduled Sudden Failures (Count)")
    axes[1].axvline(25, color="green", ls=":", lw=2, label="Recommended Policy (tau=25)")
    axes[1].set_title("Maintenance Policy Tradeoff vs Service Threshold tau", fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Maintenance Trigger Threshold tau (Cycles)")
    axes[1].set_ylabel("Preventive Maintenance Rate (%)", color="b")
    ax2_twin.set_ylabel("Unscheduled Failures (Count)", color="r")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t2_uncertainty_and_policy.png"), dpi=160)
    plt.close()


# =====================================================================
# 6. MAIN PIPELINE
# =====================================================================
def main():
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Task 2 Time-Series RUL Pipeline")
    parser.add_argument("--data_dir", type=str, default=r"D:\DL")
    parser.add_argument("--out_dir", type=str, default=r"D:\DL\UTS_DL")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    # Smart fallback for path resolution
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
    df_train, df_test = load_task2_data(args.data_dir)

    # Split machines: 80% train (720 units) / 20% val (180 units)
    all_train_units = np.sort(df_train["id_unit"].unique())
    rng = np.random.RandomState(42)
    shuffled_units = rng.permutation(all_train_units)
    n_tr = int(0.80 * len(shuffled_units))
    tr_units = np.sort(shuffled_units[:n_tr])
    va_units = np.sort(shuffled_units[n_tr:])

    ckpt_path = os.path.join(ckpt_dir, "task2_final_ensemble.pt")
    metrics_path = os.path.join(out_plots, "task2_metrics.json")

    # If eval_only and checkpoint exists, run fast inference
    if args.eval_only and os.path.exists(ckpt_path):
        print("[Task 2] --eval_only mode: Loading saved RUL ensemble checkpoint...", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        stats = ckpt["stats"]
        best_k = ckpt.get("best_k", 0.40)
        feature_cols = ckpt["feature_cols"]

        df_train_proc, _ = preprocess_timeseries(df_train, stats, is_train=True)
        df_test_proc, _ = preprocess_timeseries(df_test, stats, is_train=False)

        # Validation final windows
        df_val = df_train_proc[df_train_proc["id_unit"].isin(va_units)]
        val_final_windows, val_final_ruls, val_final_uids, _ = extract_windows(df_val, feature_cols, is_train=True, stride=1)
        # Filter strictly the last window per validation machine
        last_indices = []
        for u in va_units:
            idxs = np.where(val_final_uids == u)[0]
            if len(idxs) > 0:
                last_indices.append(idxs[-1])
        X_val_final = torch.from_numpy(val_final_windows[last_indices])
        y_val_final = val_final_ruls[last_indices]

        # Extract test windows (last 50 cycles per unit)
        X_test_arr, te_uids, _ = extract_windows(df_test_proc, feature_cols, is_train=False)
        X_test_t = torch.from_numpy(X_test_arr)

        # Reconstruct models
        tcn_model = DilatedTCN(in_features=len(feature_cols))
        tcn_model.load_state_dict(ckpt["tcn_state"])
        tcn_model.eval()

        bigru_model = BiGRUAttention(in_features=len(feature_cols))
        bigru_model.load_state_dict(ckpt["bigru_state"])
        bigru_model.eval()

        # MC Dropout predictions
        tcn_mu_v, tcn_sig_v = mc_dropout_predict(tcn_model, X_val_final, T=20)
        gru_mu_v, gru_sig_v = mc_dropout_predict(bigru_model, X_val_final, T=20)
        ens_mu_v = 0.5 * (tcn_mu_v + gru_mu_v)
        ens_sig_v = np.sqrt(0.5 * (tcn_sig_v**2 + gru_sig_v**2))
        ens_safe_v = np.clip(ens_mu_v - best_k * ens_sig_v, 0.0, MAX_RUL)

        val_metrics = evaluate_metrics_per_range(y_val_final, ens_safe_v)
        print(f"[Task 2] Validation Final Cycle (180 machines): RMSE={val_metrics['rmse']:.3f} | "
              f"MAE={val_metrics['mae']:.3f} | Asymmetric Score Total={val_metrics['asymmetric_score_total']:.2f} "
              f"(Mean={val_metrics['asymmetric_score_mean']:.2f})", flush=True)

        # Test predictions
        tcn_mu_t, tcn_sig_t = mc_dropout_predict(tcn_model, X_test_t, T=25)
        gru_mu_t, gru_sig_t = mc_dropout_predict(bigru_model, X_test_t, T=25)
        ens_mu_t = 0.5 * (tcn_mu_t + gru_mu_t)
        ens_sig_t = np.sqrt(0.5 * (tcn_sig_t**2 + gru_sig_t**2))
        ens_safe_t = np.clip(ens_mu_t - best_k * ens_sig_t, 0.0, MAX_RUL)

        sub_df = pd.DataFrame({"id_unit": np.asarray(te_uids, dtype=int), "rul": np.round(ens_safe_t, 2)})
        sub_path = os.path.join(args.out_dir, "prediksi_rul.csv")
        sub_df.to_csv(sub_path, index=False)
        print(f"[Task 2] Saved prediksi_rul.csv ({len(sub_df)} rows) to {sub_path}.", flush=True)
        return

    print("\n--- [Task 2a] Computing Regime Statistics (No Leakage) & Preprocessing ---", flush=True)
    regime_stats = compute_regime_stats(df_train, tr_units)

    print("Generating EDA and sensor correlation figures...", flush=True)
    generate_t2_eda_plots(df_train, regime_stats, out_plots)

    df_train_proc, feature_cols = preprocess_timeseries(df_train, regime_stats, is_train=True)
    df_test_proc, _ = preprocess_timeseries(df_test, regime_stats, is_train=False)

    print(f"Informative Sensors kept: {len(INFORMATIVE_SENSORS)} | Features per timestep: {len(feature_cols)}", flush=True)

    print("\n--- [Task 2b & 2c] Extracting Sliding Windows (W=50) ---", flush=True)
    df_tr_split = df_train_proc[df_train_proc["id_unit"].isin(tr_units)]
    df_va_split = df_train_proc[df_train_proc["id_unit"].isin(va_units)]

    X_tr_w, y_tr_w, u_tr, c_tr = extract_windows(df_tr_split, feature_cols, is_train=True, stride=3)
    X_va_w, y_va_w, u_va, c_va = extract_windows(df_va_split, feature_cols, is_train=True, stride=4)
    print(f"Extracted Training Windows: {X_tr_w.shape} | Validation Windows: {X_va_w.shape}", flush=True)

    # Extract final cycle window for each validation machine
    val_final_indices = []
    for u in va_units:
        idxs = np.where(u_va == u)[0]
        if len(idxs) > 0:
            val_final_indices.append(idxs[-1])
    X_va_final = X_va_w[val_final_indices]
    y_va_final = y_va_w[val_final_indices]
    print(f"Validation Final Windows (1 per machine): {X_va_final.shape}", flush=True)

    # =================================================================
    # NON-DL BASELINE: LIGHTGBM REGRESSOR
    # =================================================================
    print("\n--- [Task 2d] Training Non-DL Baseline: LightGBM Regressor ---", flush=True)
    X_tr_tab = extract_summary_features(X_tr_w)
    X_va_tab = extract_summary_features(X_va_w)
    X_va_final_tab = extract_summary_features(X_va_final)

    lgb_train_data = lgb.Dataset(X_tr_tab, label=y_tr_w)
    lgb_val_data = lgb.Dataset(X_va_tab, label=y_va_w, reference=lgb_train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "verbose": -1,
        "random_state": 42,
    }
    lgb_model = lgb.train(
        params,
        lgb_train_data,
        num_boost_round=300,
        valid_sets=[lgb_val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )

    lgb_pred_all = np.clip(lgb_model.predict(X_va_tab), 0.0, MAX_RUL)
    lgb_pred_final = np.clip(lgb_model.predict(X_va_final_tab), 0.0, MAX_RUL)
    lgb_metrics_all = evaluate_metrics_per_range(y_va_w, lgb_pred_all)
    lgb_metrics_final = evaluate_metrics_per_range(y_va_final, lgb_pred_final)

    print(f"LightGBM Baseline (All Val Windows)   -> RMSE: {lgb_metrics_all['rmse']:.3f} | Asym Total: {lgb_metrics_all['asymmetric_score_total']:.1f}", flush=True)
    print(f"LightGBM Baseline (Final Val Cycles)  -> RMSE: {lgb_metrics_final['rmse']:.3f} | Asym Total: {lgb_metrics_final['asymmetric_score_total']:.1f}", flush=True)

    # =================================================================
    # SANITY CHECK: OVERFITTING 1 BATCH (N=64)
    # =================================================================
    print("\n--- [Task 2g] Sanity Check: Overfitting One Batch (N=64) ---", flush=True)
    seed_everything(42)
    s_model = DilatedTCN(in_features=len(feature_cols))
    s_opt = torch.optim.AdamW(s_model.parameters(), lr=5e-3)
    s_crit = HybridRULLoss(alpha=0.35)
    s_bx = torch.from_numpy(X_tr_w[:64])
    s_by = torch.from_numpy(y_tr_w[:64])

    sanity_losses = []
    for step in range(1, 41):
        s_opt.zero_grad()
        s_out = s_model(s_bx)
        loss = s_crit(s_out, s_by)
        loss.backward()
        s_opt.step()
        sanity_losses.append(loss.item())
        if step in (1, 10, 20, 30, 40):
            cur_rmse = np.sqrt(mean_squared_error(s_by.numpy(), s_out.detach().numpy()))
            print(f"  Step {step:02d}: Loss = {loss.item():.4f} | Batch RMSE = {cur_rmse:.3f}", flush=True)

    # =================================================================
    # SEQUENCE MODEL 1: DILATED TCN
    # =================================================================
    print("\n--- [Task 2d] Training Sequence Model 1: Dilated TCN ---", flush=True)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_w), torch.from_numpy(y_tr_w)),
        batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va_w), torch.from_numpy(y_va_w)),
        batch_size=512, shuffle=False
    )

    seed_everything(42)
    tcn_model = DilatedTCN(in_features=len(feature_cols), dropout=0.20)
    tcn_model, tcn_hist = train_dl_model(tcn_model, train_loader, val_loader, epochs=15, lr=2.5e-3)

    tcn_model.eval()
    with torch.no_grad():
        tcn_pred_all = tcn_model(torch.from_numpy(X_va_w)).numpy()
        tcn_pred_final = tcn_model(torch.from_numpy(X_va_final)).numpy()

    tcn_metrics_all = evaluate_metrics_per_range(y_va_w, tcn_pred_all)
    tcn_metrics_final = evaluate_metrics_per_range(y_va_final, tcn_pred_final)
    print(f"Dilated TCN (All Val Windows)  -> RMSE: {tcn_metrics_all['rmse']:.3f} | Asym Total: {tcn_metrics_all['asymmetric_score_total']:.1f}", flush=True)
    print(f"Dilated TCN (Final Val Cycles) -> RMSE: {tcn_metrics_final['rmse']:.3f} | Asym Total: {tcn_metrics_final['asymmetric_score_total']:.1f}", flush=True)

    # =================================================================
    # SEQUENCE MODEL 2: STACKED BIGRU WITH TEMPORAL ATTENTION
    # =================================================================
    print("\n--- [Task 2d] Training Sequence Model 2: BiGRU + Temporal Attention ---", flush=True)
    seed_everything(123)
    bigru_model = BiGRUAttention(in_features=len(feature_cols), hidden_dim=48, dropout=0.20)
    bigru_model, bigru_hist = train_dl_model(bigru_model, train_loader, val_loader, epochs=15, lr=2e-3)

    bigru_model.eval()
    with torch.no_grad():
        bigru_pred_all = bigru_model(torch.from_numpy(X_va_w)).numpy()
        bigru_pred_final = bigru_model(torch.from_numpy(X_va_final)).numpy()

    bigru_metrics_all = evaluate_metrics_per_range(y_va_w, bigru_pred_all)
    bigru_metrics_final = evaluate_metrics_per_range(y_va_final, bigru_pred_final)
    print(f"BiGRU-Attn (All Val Windows)  -> RMSE: {bigru_metrics_all['rmse']:.3f} | Asym Total: {bigru_metrics_all['asymmetric_score_total']:.1f}", flush=True)
    print(f"BiGRU-Attn (Final Val Cycles) -> RMSE: {bigru_metrics_final['rmse']:.3f} | Asym Total: {bigru_metrics_final['asymmetric_score_total']:.1f}", flush=True)

    # =================================================================
    # UNCERTAINTY QUANTIFICATION (MC DROPOUT + ENSEMBLE) & ASYMMETRIC TUNING
    # =================================================================
    print("\n--- [Task 2f] Uncertainty Quantification & Risk-Adjusted Prediction ---", flush=True)
    X_va_final_t = torch.from_numpy(X_va_final)

    tcn_mu_v, tcn_sig_v = mc_dropout_predict(tcn_model, X_va_final_t, T=30)
    gru_mu_v, gru_sig_v = mc_dropout_predict(bigru_model, X_va_final_t, T=30)

    ens_mu_v = 0.5 * (tcn_mu_v + gru_mu_v)
    ens_sig_v = np.sqrt(0.5 * (tcn_sig_v**2 + gru_sig_v**2))

    # Grid search best conservative penalty factor k in [0.0, 1.0]
    best_k = 0.0
    best_asym = float("inf")
    for k_val in np.linspace(0.0, 1.0, 21):
        safe_p = np.clip(ens_mu_v - k_val * ens_sig_v, 0.0, MAX_RUL)
        score = asymmetric_rul_score(y_va_final, safe_p)
        if score < best_asym:
            best_asym = score
            best_k = float(k_val)

    print(f"Best Risk-Adjustment Factor k: {best_k:.2f} (Reduced Asymmetric Score from "
          f"{asymmetric_rul_score(y_va_final, ens_mu_v):.1f} to {best_asym:.1f})", flush=True)

    ens_pred_final_safe = np.clip(ens_mu_v - best_k * ens_sig_v, 0.0, MAX_RUL)
    ens_metrics_final = evaluate_metrics_per_range(y_va_final, ens_pred_final_safe)
    print(f"Final Ensemble (Final Val Cycles) -> RMSE: {ens_metrics_final['rmse']:.3f} | Asym Total: {ens_metrics_final['asymmetric_score_total']:.1f}", flush=True)
    print(f"  Low RUL [0, 30]     -> RMSE: {ens_metrics_final['ranges']['low_0_30']['rmse']} | Asym: {ens_metrics_final['ranges']['low_0_30']['asymmetric_score']}")
    print(f"  Med RUL (30, 80]    -> RMSE: {ens_metrics_final['ranges']['med_30_80']['rmse']} | Asym: {ens_metrics_final['ranges']['med_30_80']['asymmetric_score']}")
    print(f"  High RUL (80, 150]  -> RMSE: {ens_metrics_final['ranges']['high_80_150']['rmse']} | Asym: {ens_metrics_final['ranges']['high_80_150']['asymmetric_score']}")

    # =================================================================
    # SAVE CHECKPOINTS, FIGURES & PREDICTIONS
    # =================================================================
    print("\nSaving figures...", flush=True)
    generate_t2_predictions_plots(sanity_losses, tcn_hist, bigru_hist, y_va_final, ens_pred_final_safe, out_plots)
    generate_t2_uncertainty_plots(y_va_final, ens_mu_v, ens_sig_v, best_k, out_plots)

    # Save checkpoint
    torch.save({
        "tcn_state": tcn_model.state_dict(),
        "bigru_state": bigru_model.state_dict(),
        "stats": regime_stats,
        "feature_cols": feature_cols,
        "best_k": best_k,
    }, ckpt_path)
    print(f"Saved model checkpoint to {ckpt_path} ({os.path.getsize(ckpt_path)/(1024*1024):.2f} MB)", flush=True)

    # Predict test set
    X_test_arr, te_uids, _ = extract_windows(df_test_proc, feature_cols, is_train=False)
    X_test_t = torch.from_numpy(X_test_arr)

    tcn_mu_t, tcn_sig_t = mc_dropout_predict(tcn_model, X_test_t, T=30)
    gru_mu_t, gru_sig_t = mc_dropout_predict(bigru_model, X_test_t, T=30)
    ens_mu_t = 0.5 * (tcn_mu_t + gru_mu_t)
    ens_sig_t = np.sqrt(0.5 * (tcn_sig_t**2 + gru_sig_t**2))
    ens_safe_t = np.clip(ens_mu_t - best_k * ens_sig_t, 0.0, MAX_RUL)

    sub_df = pd.DataFrame({
        "id_unit": np.asarray(te_uids, dtype=int),
        "rul": np.round(ens_safe_t, 2)
    })
    sub_path = os.path.join(args.out_dir, "prediksi_rul.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved test predictions to {sub_path} ({len(sub_df)} rows).", flush=True)

    # Save complete JSON metrics
    results = {
        "eda": {
            "train_machines": int(len(all_train_units)),
            "test_machines": int(df_test["id_unit"].nunique()),
            "regimes": [0, 10, 20, 25, 35, 42],
            "uninformative_sensors_dropped": UNINFORMATIVE_SENSORS,
            "informative_sensors_used": INFORMATIVE_SENSORS,
            "feature_dim_per_step": len(feature_cols),
            "window_size": WINDOW_SIZE,
        },
        "sanity_check": {
            "initial_loss": round(float(sanity_losses[0]), 4),
            "final_loss": round(float(sanity_losses[-1]), 4),
        },
        "models_validation_final_cycle": {
            "lightgbm_baseline": lgb_metrics_final,
            "dilated_tcn": tcn_metrics_final,
            "bigru_attention": bigru_metrics_final,
            "final_ensemble_safe": ens_metrics_final,
        },
        "models_validation_all_windows": {
            "lightgbm_baseline": lgb_metrics_all,
            "dilated_tcn": tcn_metrics_all,
            "bigru_attention": bigru_metrics_all,
        },
        "uncertainty": {
            "best_risk_adjustment_k": best_k,
            "mean_sigma": round(float(np.mean(ens_sig_v)), 3),
            "std_sigma": round(float(np.std(ens_sig_v)), 3),
            "maintenance_recommended_tau": 25.0,
        },
        "test_summary": {
            "count": len(sub_df),
            "rul_mean": round(float(sub_df["rul"].mean()), 2),
            "rul_std": round(float(sub_df["rul"].std()), 2),
            "rul_min": round(float(sub_df["rul"].min()), 2),
            "rul_max": round(float(sub_df["rul"].max()), 2),
        }
    }
    save_json(results, metrics_path)
    print(f"[Task 2] All metrics and plots saved to {out_plots} in {time.time()-t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
