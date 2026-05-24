# tcn_256D: WaveNet Gated TCN with SE Channel Attention

## Overview

A 7-layer causal dilated convolutional network for cross-sectional stock return prediction on Chinese A-shares (全A股). The model uses a **256-day lookback window** of 9 price-volume channels and is trained with an **expanding window** (2016-2026, yearly retrain) to produce out-of-sample predictions for a top-50 long-only strategy.

**Best result (actdays=5):** Sharpe 1.316, Ann.Return 41.5%, MaxDD -31.2%

---

## Architecture

```
Input (B, 9, 256)
│
├─ Conv1d(9→32, kernel=1) + LayerNorm + GELU    # Shallow channel projection
│
├─ CausalTCNBlock × 7 (dilated gated conv)
│  ┌─────────────────────────────────────────┐
│  │  dilations: [1, 2, 4, 8, 16, 32, 64]  │
│  │  RF = 2^(7+1)-1 = 255 ≈ TAU=256        │
│  │  Each block: left-pad → conv_f + conv_g │
│  │              → tanh(h)*sigmoid(g) → LN  │
│  │              → Dropout → residual (+)   │
│  └─────────────────────────────────────────┘
│  (all 7 block outputs summed → skip connection)
│
├─ AdaptiveAvgPool1d(1)   # (B, 32, 1)
├─ Flatten                # (B, 32)
├─ SEBlock(32→8→32)       # Channel attention (Sigmoid gate)
├─ Linear(32→1)           # (B, 1)
│
Output (B,)               # predicted score per stock
```

### Design choices

| Component | Detail |
|-----------|--------|
| **Receptive field** | 255 days (dilations 1..64, kernel=3, 7 layers), matched to TAU=256 |
| **Gating** | tanh(h) * sigmoid(g) — WaveNet-style, controls information flow per position |
| **Skip connections** | All 7 block outputs summed → head, improves gradient flow |
| **SE attention** | Lightweight channel-wise reweighting (bottleneck ratio 4), ~512 extra params |
| **Normalization** | LayerNorm per block (after gating, before dropout) — stable for variable-length |
| **Regularization** | Dropout=0.2, weight_decay=1e-5, gradient clip=1.0, early stop patience=10 |
| **AMP** | float16 automatic mixed precision, ~2× faster on GPU |

### Parameter count

| Component | Params |
|-----------|--------|
| Input projection | (9×32) + 32×2 (LN) + bias = ~352 |
| Per CausalTCNBlock (×7) | 2×(32×3×32) + 32×2 (LN) = ~6,208 |
| SEBlock | 32×8 + 8×32 = 512 |
| Output Linear | 32×1 = 32 |
| **Total** | **~50K** |

---

## Input Channels (9)

| # | Channel | Definition | Frequency |
|---|---------|-----------|-----------|
| 1 | `ret_raw` | C(t)/C(t-1) - 1, raw daily return | Daily |
| 2 | `ret_norm` | ret_raw / rolling_std(60d), clipped to [-5, 5] | Daily |
| 3 | `abs_ret` | abs(ret_raw), return magnitude | Daily |
| 4 | `amount_norm` | log(amount), cross-sectional z-score by Date | Daily |
| 5 | `range_spread` | (H-L)/C, cross-sectional z-score | Daily |
| 6 | `ibs` | (C-L)/(H-L), intraday position [0,1] | Daily |
| 7 | `gap_raw` | O/C(t-1) - 1, overnight gap clipped to [-0.2, 0.2] | Daily |
| 8 | `to_norm` | log(TO+1e-8), turnover z-score | Daily |
| 9 | `cap_norm` | log(Cap), market cap z-score | Daily/Quarterly |

All channels are aligned per (ID, Date) and stored as float16 windowed tensors (B, 9, 256).

---

## Training Procedure

### Data pipeline

1. Load **alldataEODs.parquet** (open/high/low/close/amount) + **CapTOdata** (turnover, market cap)
2. Compute 9 channels via `compute_channels()` — all Polars vectorized ops
3. For each stock, slide a 256-day window with stride 1 → windowed (C, TAU) samples
4. Target: `classy` from SHSZ_{actdays}_EODtestraw.parquet (classification label)
5. Expanding window split: train on years < y, validate on year y-1, test on year y

### Training loop

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (lr=2e-3, wd=1e-5) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6) |
| Loss | MSE |
| Batch size | 1024 |
| Max epochs | 200 |
| Early stop | 10 epochs (val loss no improvement) |
| Gradient clip | norm=1.0 |
| AMP | float16 |

Data is streamed CPU→GPU per batch (full dataset too large for GPU memory). All windowed tensors are pre-computed on CPU at float16 to save memory.

### Backtest (post-prediction)

- **Filter:** d1 triple filter (limit-up/down/ST exclusion, threshold=0.2)
- **Selection:** cross-sectional rank(pred_y, descending) → top 50 stocks
- **Weighting:** equal-weight, weekly rebalance (actdays=1..5)
- **Benchmark:** SH000905 (CSI 500 Index), excess return = strategy_ret - zz500_ret + 1
- **Metrics:** Ann.Return, Ann.Vol, Sharpe, MaxDD, Calmar (via `pnl_report_polars_plotnine`)

---

## Results (actdays=5, OOS 2016-2026)

| Metric | Long-only | Long-short |
|--------|-----------|------------|
| Sharpe | **1.316** | **6.388** |
| Ann.Return | **41.5%** | **395.0%** |
| Ann.Vol | 29.8% | 25.9% |
| MaxDD | -31.2% | -13.4% |
| Calmar | 1.33 | 29.55 |

> Long-short is top 50 long + bottom 50 short, dollar-neutral, equal-weight, no d1 filter, no short constraints, no transaction costs — theoretical upper bound on pure rank signal quality.

## Strategy versions

| Version | Description | Status |
|---------|-------------|--------|
| **Long-only** | Rank top 50 → equal-weight long, d1 filter | current, documented above |
| **Long-short** | Rank top 50 long + bottom 50 short, dollar-neutral | experimental, theoretical (no d1 filter) |
