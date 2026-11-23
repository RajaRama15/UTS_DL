---
name: task2-timeseries-rul-prognostics
category: deep-learning / time-series-regression
version: 1.0.0
description: >
  Expert blueprint for Task 2 (Remaining Useful Life prediction on press machines).
  Covers 6-regime operating condition normalization, causal spike/NaN filtering,
  TCN and BiGRU sequence architectures, NASA C-MAPSS asymmetric scoring loss,
  and Monte Carlo (MC) Dropout + Ensemble uncertainty quantification.
triggers:
  - task 2
  - rul_train.csv
  - remaining useful life
  - asymmetric score
  - mc dropout uncertainty
---

# SKILL: Task 2 Time-Series RUL Prognostics (30 Points)

## 1. Domain & Dataset Facts (`rul_train.csv` & `rul_test.csv`)
- **Train:** 900 run-to-failure machines (`id_unit` 1..900), 254,174 total cycles. Lifetimes: 133 to 778 cycles (median 265).
- **Test:** 300 truncated machines (`id_unit` 1..300), 51,692 total cycles. Observed histories: **50 to 455 cycles** (median 162.5).
- **Columns:** `id_unit`, `siklus`, `setel_1`, `setel_2`, `setel_3`, `s01`..`s21`.
- **Target Label:** $\text{RUL}_{u,t} = \min(150, \max(\text{siklus}_u) - \text{siklus}_{u,t})$. Predict RUL at the **last cycle** of each test machine.
- **Evaluation Metrics:**
  1. **RMSE:** $\sqrt{\frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2}$
  2. **Asymmetric Score:** With $d_i = \hat{y}_i - y_i$:
     $$s = \sum_{i=1}^N \begin{cases} \exp(-d_i / 13) - 1 & \text{if } d_i < 0 \text{ (early prediction)} \\ \exp(d_i / 10) - 1 & \text{if } d_i \ge 0 \text{ (late/over-optimistic prediction)} \end{cases}$$

## 2. Step-by-Step Pipeline Requirements (Rubric 2a–2f)

### 2a. Operating Regimes, Uninformative Sensors, Missing Values & Spikes
1. **Identify the 6 Operating Regimes:**
   Round `setel_1` to nearest integer (`{0, 10, 20, 25, 35, 42}`). These correspond to 6 discrete operating settings:
   `(0, 0.0, 100)`, `(10, 0.25, 100)`, `(20, 0.70, 100)`, `(25, 0.62, 60)`, `(35, 0.84, 100)`, `(42, 0.84, 100)`.
2. **Regime-Conditioned Normalization (No Leakage):**
   Fit per-regime median and IQR (or clipped mean and std) **only on training machines**. Transform each sensor reading relative to its cycle's operating regime:
   $$z_{t,s} = \frac{x_{t,s} - \mu_{\text{regime}(t), s}}{\sigma_{\text{regime}(t), s} + 10^{-6}}$$
3. **Drop 7 Uninformative Sensors:**
   Show via correlation table/plot that after regime normalization, **`s01, s05, s06, s10, s16, s18, s19`** have near-zero correlation with RUL ($|r| < 0.01$). Keep the **14 informative sensors**:
   `['s02', 's03', 's04', 's07', 's08', 's09', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']`.
4. **Causal Despiking & Missing Value Imputation:**
   - Each sensor has ~2.0% `NaN`s and 0.1%–2.4% extreme spikes ($|z_{\text{IQR}}| > 4.0$).
   - Mask any reading with $|z_{t,s}| > 4.0$ as `NaN` (spike removal).
   - Impute `NaN`s **causally** using forward-fill (`groupby('id_unit').ffill().fillna(0.0)`). Never use backward-fill (`bfill`) or future cycles!
   - Append causal Exponential Moving Average (EMA, $\alpha = 0.2$) smoothed channels + normalized cycle counter $\text{siklus} / 300.0$ (giving $14 \times 2 + 1 = 29$ features per timestep).

### 2b & 2c. Piecewise Linear RUL Labels & Sliding Windows
- Since every test machine has at least **50 cycles**, use a fixed causal sliding window of length **$W = 50$**.
- From training machines, extract sliding windows of length $W = 50$ with stride 3, **always including the final window** (and extra windows near the end of life where $\text{RUL} < 150$) so the model is well-calibrated across all RUL ranges.

### 2d. Model Comparison (2 Sequence DL Models + 1 Non-DL Baseline)
1. **Non-DL Baseline:** `LightGBMRegressor` (or `Ridge`) trained on flattened window summary features (last value, EMA value, window mean, window std, linear slope over the 50 cycles).
2. **Sequence DL Model 1 — Dilated TCN (Temporal Convolutional Network):**
   1D dilated convolutions (`kernel_size=3`, dilations `1, 2, 4, 8`) with residual connections, LayerNorm/BatchNorm, GELU, and Dropout (`p=0.20`).
3. **Sequence DL Model 2 — Stacked BiGRU / LSTM with Temporal Attention:**
   2-layer GRU (`hidden_dim=64`, `dropout=0.20`) followed by attention pooling over the last steps and an MLP head predicting RUL in $[0, 150]$ (via `150.0 * torch.sigmoid(out)` or clipped linear head).
4. **Hybrid Asymmetric Training Loss:**
   ```python
   def hybrid_rul_loss(y_pred, y_true, alpha=0.35):
       d = y_pred - y_true
       mse = F.mse_loss(y_pred, y_true)
       asym = torch.where(d < 0, torch.exp(-d / 13.0) - 1.0, torch.exp(d / 10.0) - 1.0).mean()
       return mse + alpha * asym
   ```

### 2e & 2f. Machine-Level Validation, Per-Range Breakdown & Uncertainty
- **Split by Machine (`id_unit`):** Use 80% of training machines (720 machines) for training and 20% (180 machines) for validation (evaluating both on all validation windows and specifically on the last window per validation machine, plus synthetic random truncation of validation machines to mimic test machines!).
- **Report RMSE & Asymmetric Score per RUL Range:**
  Break down validation metrics across Low RUL ($[0, 30]$ — critical failure zone), Medium RUL ($(30, 80]$), and High RUL ($(80, 150]$).
- **Uncertainty Quantification (MC Dropout + Ensemble):**
  - Keep dropout active at inference (`T = 30` stochastic passes across 3 random seeds / models) to compute predictive mean $\hat{\mu}_i$ and uncertainty $\hat{\sigma}_i$.
  - Apply conservative risk-adjusted prediction $\hat{y}_i = \text{clip}(\hat{\mu}_i - k \cdot \hat{\sigma}_i, 0, 150)$ (tune $k \in [0.2, 1.0]$ on validation to minimize the asymmetric score!).
  - Plot prediction intervals on validation machines sorted by true RUL and define a concrete maintenance policy (e.g., schedule maintenance when the 90% lower confidence bound $\hat{\mu}_i - 1.645\hat{\sigma}_i \le 25$ cycles).
