"""
WaveNet-style gated TCN with SE channel attention for A-share cross-section prediction.

TAU=256 lookback, 9 price-volume channels, expanding window 2016-2026.
Output: top-50 equal-weight long-only strategy, SH000905 benchmark.
"""
import os, gc, time, warnings, sys
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plotnine import ggplot, aes, geom_line, labs

sys.path.insert(0, str(Path(r"D:\shsz_auto\pl_code").resolve()))
import importlib
import persupportfunc_pl
importlib.reload(persupportfunc_pl)
pnl_report_polars_plotnine = persupportfunc_pl.pnl_report_polars_plotnine

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

# ============================================================
# CONFIG
# ============================================================
TAU = 256
BATCH_SIZE = 1024
EPOCHS = 200
EARLY_STOP_PATIENCE = 10
LR = 2e-3
WEIGHT_DECAY = 1e-5
HIDDEN = 32
N_LAYERS = 7           # dilations 1,2,4,8,16,32,64 → RF=255 ≈ TAU=256
KERNEL_SIZE = 3
DROPOUT = 0.2
CLIP_NORM = 1.0
AMP_DTYPE = torch.float16
SE_REDUCTION = 4

DATA_DIR = "d:/data"
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_STEM = SCRIPT_PATH.stem

# 9 price-volume channels
CHANNEL_COLS = [
    "ret_raw", "ret_norm", "abs_ret", "amount_norm", "range_spread",
    "ibs", "gap_raw", "to_norm", "cap_norm",
]
N_CHANNELS = len(CHANNEL_COLS)  # 9

# ============================================================
# MODEL: WaveNet Gated TCN + SE Channel Attention
# ============================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Learns channel-wise importance via a lightweight bottleneck MLP
    after global average pooling.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C)
        scale = self.fc(x)
        return x * scale


class CausalTCNBlock(nn.Module):
    """Single WaveNet-style gated causal dilated conv block with residual connection."""

    def __init__(self, channels, dilation, kernel_size=3, dropout=0.2):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv_f = nn.Conv1d(channels, channels, kernel_size,
                                padding=0, dilation=dilation)
        self.conv_g = nn.Conv1d(channels, channels, kernel_size,
                                padding=0, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, L)
        x_pad = F.pad(x, [self.padding, 0])    # left-pad for causality
        h = self.conv_f(x_pad)                  # (B, C, L)
        g = self.conv_g(x_pad)
        out = torch.tanh(h) * torch.sigmoid(g)  # gated tanh×sigmoid activation

        out = out.transpose(1, 2)               # (B, L, C) for LayerNorm
        out = self.norm(out)
        out = out.transpose(1, 2)               # (B, C, L)

        out = self.dropout(out)
        return x + out                          # residual


class WaveNetTCN(nn.Module):
    """WaveNet-style gated TCN with skip connections and SE channel attention.

    Architecture:
        Input (B, C, TAU)
        → Conv1d(1×1, C→hidden) + LayerNorm + GELU
        → N x CausalTCNBlock (dilated conv, gated, residual)
          (all block outputs summed as skip connection)
        → AdaptiveAvgPool1d → SEBlock → Linear(hidden→1)
        → Output (B,)
    """

    def __init__(self, n_channels=5, hidden=32, n_layers=6,
                 kernel_size=3, dropout=0.2, se_reduction=4):
        super().__init__()

        # Shallow projection: n_channels → hidden
        self.proj_conv = nn.Conv1d(n_channels, hidden, kernel_size=1)
        self.proj_norm = nn.LayerNorm(hidden)

        # Gated dilated conv blocks: dilations = 2^i
        self.blocks = nn.ModuleList([
            CausalTCNBlock(hidden, dilation=2 ** i,
                           kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)
        ])

        # Output: aggregate → SE attention → Linear
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                     # (B, hidden, 1)
            nn.Flatten(),                                # (B, hidden)
            SEBlock(hidden, reduction=se_reduction),     # (B, hidden)
            nn.Linear(hidden, 1),                        # (B, 1)
        )

    def forward(self, x):
        # x: (B, C, L)
        out = self.proj_conv(x)          # (B, hidden, L)
        out = out.transpose(1, 2)        # (B, L, hidden)
        out = self.proj_norm(out)
        out = out.transpose(1, 2)        # (B, hidden, L)
        out = F.gelu(out)

        skip_sum = 0
        for block in self.blocks:
            out = block(out)
            skip_sum = skip_sum + out    # accumulate all block outputs

        return self.head(skip_sum).squeeze(-1)  # (B,)


# ============================================================
# CHANNEL COMPUTATION
# ============================================================
def compute_channels(eod: pl.DataFrame, cap_to: pl.DataFrame) -> pl.DataFrame:
    """Compute all 9 channels from daily EOD + CapTO data.

    Returns DataFrame with ID, Date, and 9 channel columns.
    """
    df = (eod
          .rename({"close": "C", "high": "H", "low": "L", "open": "O"})
          .filter(pl.col("C") > 0.01)
          .sort(["ID", "Date"])
          .join(cap_to.select(["ID", "Date", "TO", "Cap"]),
                on=["ID", "Date"], how="left")
          .with_columns([
              (pl.col("C") / pl.col("C").shift(1) - 1)
              .over("ID").alias("ret_raw"),
          ])
          .with_columns([pl.col("ret_raw").fill_null(0.0)])
          .with_columns([pl.col("ret_raw").abs().alias("abs_ret")])
          .with_columns([
              pl.when(
                  pl.col("ret_raw").rolling_std(window_size=60)
                  .over("ID").fill_null(1.0) > 1e-8
              )
              .then(pl.col("ret_raw") / (
                  pl.col("ret_raw").rolling_std(window_size=60)
                  .over("ID").fill_null(1.0) + 1e-8
              ))
              .otherwise(0.0)
              .alias("_ret_norm_raw"),
          ])
          .with_columns([
              pl.col("_ret_norm_raw").clip(-5, 5).alias("ret_norm"),
          ])
          .drop("_ret_norm_raw")
          )

    # amount_norm: log(amount) → cross-sectional z-score by Date
    df = df.with_columns([
        ((pl.col("amount").clip(1.0, None).log()
          - pl.col("amount").clip(1.0, None).log().mean().over("Date"))
         / pl.col("amount").clip(1.0, None).log().std().over("Date"))
        .fill_null(0.0).alias("amount_norm"),
    ])

    # range_spread: (H-L)/C → z-score by Date
    df = df.with_columns([
        ((pl.col("H") - pl.col("L")) / pl.col("C")).alias("_range"),
    ]).with_columns([
        ((pl.col("_range") - pl.col("_range").mean().over("Date"))
         / pl.col("_range").std().over("Date"))
        .fill_null(0.0).alias("range_spread"),
    ]).drop("_range")

    # --- Additional channels (v3+) ---
    df = df.with_columns([
        pl.col("TO").fill_null(0.0),
        pl.col("Cap").fill_null(1e8),
    ])

    # ibs: (C-L)/(H-L), natural [0,1]
    df = df.with_columns([
        ((pl.col("C") - pl.col("L"))
         / (pl.col("H") - pl.col("L") + 1e-8))
        .fill_nan(0.5).alias("ibs"),
    ])

    # gap_raw: overnight gap (O/C_prev - 1), clipped
    df = df.with_columns([
        ((pl.col("O") / pl.col("C").shift(1) - 1).over("ID"))
        .fill_null(0.0).clip(-0.2, 0.2).alias("gap_raw"),
    ])

    # to_norm: log(TO+1e-8) → z-score by Date
    df = df.with_columns([
        (
            ((pl.col("TO") + 1e-8).log()
             - (pl.col("TO") + 1e-8).log().mean().over("Date"))
            / (pl.col("TO") + 1e-8).log().std().over("Date")
        ).alias("_to_z"),
    ]).with_columns([
        pl.when(pl.col("_to_z").is_infinite())
        .then(0.0).otherwise(pl.col("_to_z"))
        .fill_nan(0.0).fill_null(0.0).alias("to_norm"),
    ]).drop("_to_z")

    # cap_norm: log(Cap) → z-score by Date
    df = df.with_columns([
        (
            (pl.col("Cap").log()
             - pl.col("Cap").log().mean().over("Date"))
            / pl.col("Cap").log().std().over("Date")
        ).alias("_cap_z"),
    ]).with_columns([
        pl.when(pl.col("_cap_z").is_infinite())
        .then(0.0).otherwise(pl.col("_cap_z"))
        .fill_nan(0.0).fill_null(0.0).alias("cap_norm"),
    ]).drop("_cap_z")

    # Final NaN guard
    for c in CHANNEL_COLS:
        df = df.with_columns(pl.col(c).fill_null(0.0))

    return df.select(["ID", "Date"] + CHANNEL_COLS)


# ============================================================
# MAIN
# ============================================================
def main():
    import builtins
    print = partial(builtins.print, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = False

    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    # ---------------------------------------------------------------
    # 1. Load EOD + CapTO
    # ---------------------------------------------------------------
    print("\n=== Loading EOD data ===")
    t0 = time.time()
    eod = pl.read_parquet(
        f"{DATA_DIR}/alldataEODs.parquet",
        columns=["open", "high", "low", "close", "amount", "ID", "Date"],
    )
    print(f"  Loaded {len(eod):,} rows in {time.time() - t0:.1f}s")

    print("  Loading CapTO data...")
    t0 = time.time()
    cap_to = pl.read_parquet(
        f"{DATA_DIR}/CapTOdata20260126.parquet",
        columns=["ID", "Date", "TO", "Cap"],
    )
    print(f"  Loaded {len(cap_to):,} rows in {time.time() - t0:.1f}s")

    t0 = time.time()
    ch_data = compute_channels(eod, cap_to)
    del eod, cap_to; gc.collect()
    print(f"  Computed channels: {len(ch_data):,} rows in {time.time() - t0:.1f}s")
    print(f"  Channels ({N_CHANNELS}): {CHANNEL_COLS}")

    # ---------------------------------------------------------------
    # Output files
    # ---------------------------------------------------------------
    plots_pdf = f"{DATA_DIR}/backtest_all_plots_{SCRIPT_STEM}.pdf"
    perf_txt = f"{DATA_DIR}/backtest_performance_{SCRIPT_STEM}.txt"
    print(f"\nOutput PDF: {plots_pdf}")
    print(f"Output TXT: {perf_txt}")

    with PdfPages(plots_pdf) as pdf_pages, \
            open(perf_txt, "w", encoding="utf-8") as perf_file:
        perf_file.write(f"Backtest performance report - {SCRIPT_PATH.name}\n")
        perf_file.write("=" * 80 + "\n\n")

        # ---------------------------------------------------------------
        # 2. Loop over actdays (1=Mon … 5=Fri)
        # ---------------------------------------------------------------
        for actdays in [5]:
            print(f"\n{'=' * 60}")
            print(f"  Actdays = {actdays}")
            print(f"{'=' * 60}")

            # ---- 2a. Load labels and join with channels ----
            t0 = time.time()
            labels = pl.read_parquet(
                f"{DATA_DIR}/SHSZ_{actdays}_EODtestraw.parquet",
                columns=["ID", "Date", "classy", "y", "d1"],
            )

            joined = ch_data.join(
                labels, on=["ID", "Date"], how="left"
            ).sort(["ID", "Date"])
            print(f"  Joined data: {len(joined):,} rows ({time.time() - t0:.1f}s)")

            # ---- 2b. Convert to numpy ----
            t0 = time.time()
            ch_arr = joined.select(CHANNEL_COLS).to_numpy().astype(np.float32, copy=False)
            classy_arr = joined["classy"].to_numpy().astype(np.float32, copy=False)
            y_arr = joined["y"].to_numpy()
            d1_arr = joined["d1"].to_numpy()
            dates_arr = joined["Date"].to_numpy()
            ids_arr = joined["ID"].to_numpy()

            del joined; gc.collect()
            print(f"  Converted to numpy ({time.time() - t0:.1f}s)")

            # ---- 2c. Build window indices (lookback TAU per stock) ----
            t0 = time.time()
            n_total = len(ch_arr)
            id_changes = np.concatenate(
                [[True], ids_arr[:-1] != ids_arr[1:]]
            )
            group_starts = np.where(id_changes)[0]
            group_ends = np.where(
                np.concatenate([id_changes[1:], [True]])
            )[0] + 1

            has_label = ~np.isnan(classy_arr)

            bin_starts, bin_ends = [], []
            for gs, ge in zip(group_starts, group_ends):
                length = ge - gs
                if length < TAU:
                    continue
                positions = np.arange(gs + TAU - 1, ge)
                keep = has_label[positions]
                if not keep.any():
                    continue
                pos_labeled = positions[keep]
                bin_starts.append(pos_labeled - TAU + 1)
                bin_ends.append(pos_labeled)

            if not bin_starts:
                print("  WARNING: no valid windows for this actdays!")
                continue

            window_starts = np.concatenate(bin_starts).astype(np.int64)
            window_ends = np.concatenate(bin_ends).astype(np.int64)
            window_indices = np.column_stack([window_starts, window_ends])
            n_windows = len(window_indices)
            del bin_starts, bin_ends, has_label; gc.collect()
            print(f"  Built {n_windows:,} windows ({time.time() - t0:.1f}s)")

            # ---- 2d. Pre-compute all windowed tensors on CPU ----
            t0 = time.time()
            end_dates = dates_arr[window_ends].copy()
            n_all = n_windows
            X_all = np.empty((n_all, N_CHANNELS, TAU), dtype=np.float16)
            for i in range(n_all):
                s = window_starts[i]
                e = window_ends[i]
                X_all[i] = ch_arr[s:e + 1].T.copy()
            y_all = classy_arr[window_ends].copy().astype(np.float32)
            del window_starts, window_indices, ch_arr, classy_arr; gc.collect()
            print(f"  Pre-computed {n_all:,} X windows ({time.time() - t0:.1f}s)")

            # ---- 2e. Expanding window: train 2016→2025, test next year ----
            pred_blocks = []

            for y in range(2016, 2027):
                t1 = np.datetime64(f"{y}-01-01", "ns")
                t2 = np.datetime64(f"{y + 1}-01-01", "ns")
                val_t0 = np.datetime64(f"{y - 1}-01-01", "ns")

                test_mask = (end_dates >= t1) & (end_dates < t2)
                train_mask_all = end_dates < t1
                val_mask = train_mask_all & (end_dates >= val_t0)

                if test_mask.sum() == 0:
                    continue
                if train_mask_all.sum() < BATCH_SIZE:
                    print(f"  Year {y}: train too small ({train_mask_all.sum()}), skip")
                    continue

                print(f"\n  --- Year {y} ---")
                print(f"    Train: {train_mask_all.sum() - val_mask.sum():,} | "
                      f"Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")

                t_gpu = time.time()
                X_train_np = X_all[train_mask_all & ~val_mask]
                y_train_np = y_all[train_mask_all & ~val_mask]
                X_val_np   = X_all[val_mask]
                y_val_np   = y_all[val_mask]
                X_test_np  = X_all[test_mask]
                n_train = X_train_np.shape[0]
                n_val   = X_val_np.shape[0]
                print(f"    Train: {n_train:,} | Val: {n_val:,} | "
                      f"Test: {X_test_np.shape[0]:,} | "
                      f"copy ({time.time() - t_gpu:.1f}s)")

                # ---- Init model ----
                model = WaveNetTCN(
                    n_channels=N_CHANNELS, hidden=HIDDEN, n_layers=N_LAYERS,
                    kernel_size=KERNEL_SIZE, dropout=DROPOUT,
                    se_reduction=SE_REDUCTION,
                ).to(device)
                if device.type == "cuda":
                    torch.set_float32_matmul_precision("high")

                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
                )
                scaler = torch.amp.GradScaler() if AMP_DTYPE is not None else None

                # ---- Training ----
                best_loss = float("inf")
                patience_counter = 0
                ckpt_path = f"{DATA_DIR}/_tcn_act{actdays}_y{y}.pt"

                print(f"    Starting training ({n_train:,} samples, "
                      f"{n_train // BATCH_SIZE + 1} batches/epoch)")
                for epoch in range(1, EPOCHS + 1):
                    perm = np.random.permutation(n_train)
                    epoch_loss = 0.0
                    n_batches = 0

                    model.train()
                    for i in range(0, n_train, BATCH_SIZE):
                        idx = perm[i:i + BATCH_SIZE]
                        Xb = torch.from_numpy(X_train_np[idx]).cuda()
                        yb = torch.from_numpy(y_train_np[idx]).cuda()

                        optimizer.zero_grad()

                        if AMP_DTYPE is not None:
                            with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE):
                                pred = model(Xb)
                                loss = F.mse_loss(pred, yb)
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            pred = model(Xb)
                            loss = F.mse_loss(pred, yb)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                            optimizer.step()

                        del Xb, yb
                        epoch_loss += loss.item()
                        n_batches += 1

                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    # Validation
                    model.eval()
                    val_loss = 0.0
                    val_n = 0
                    with torch.no_grad():
                        for j in range(0, n_val, BATCH_SIZE):
                            Xv = torch.from_numpy(X_val_np[j:j + BATCH_SIZE]).cuda()
                            yv = torch.from_numpy(y_val_np[j:j + BATCH_SIZE]).cuda()
                            if AMP_DTYPE is not None:
                                with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE):
                                    pred = model(Xv)
                            else:
                                pred = model(Xv)
                            val_loss += F.mse_loss(pred, yv, reduction="sum").item()
                            val_n += len(yv)
                            del Xv, yv, pred
                    val_loss /= val_n

                    scheduler.step(val_loss)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), ckpt_path)
                    else:
                        patience_counter += 1

                    if epoch == 1 or epoch % 5 == 0:
                        print(f"    Epoch {epoch:3d}: train_loss={epoch_loss / n_batches:.6f} "
                              f"val_loss={val_loss:.6f} (best={best_loss:.6f})")

                    if patience_counter >= EARLY_STOP_PATIENCE:
                        print(f"    Early stop at epoch {epoch} "
                              f"(best val_loss={best_loss:.6f})")
                        break

                # ---- Predict ----
                model.load_state_dict(torch.load(ckpt_path, weights_only=True))
                os.remove(ckpt_path)
                model.eval()

                test_preds = np.empty(X_test_np.shape[0], dtype=np.float32)
                with torch.no_grad():
                    for j in range(0, len(X_test_np), BATCH_SIZE):
                        Xt = torch.from_numpy(X_test_np[j:j + BATCH_SIZE]).cuda()
                        if AMP_DTYPE is not None:
                            with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE):
                                test_preds[j:j + BATCH_SIZE] = model(Xt).cpu().numpy()
                        else:
                            test_preds[j:j + BATCH_SIZE] = model(Xt).cpu().numpy()
                        del Xt

                del X_train_np, y_train_np, X_val_np, y_val_np, X_test_np
                del model, optimizer, scheduler, scaler
                torch.cuda.empty_cache()

                # Build result chunk
                test_positions = window_ends[test_mask]
                year_df = pl.DataFrame({
                    "ID": ids_arr[test_positions],
                    "Date": dates_arr[test_positions],
                    "y": y_arr[test_positions],
                    "d1": d1_arr[test_positions],
                    "pred_y": test_preds,
                })
                pred_blocks.append(year_df)
                del year_df; gc.collect()

            # ---- Concatenate all years ----
            if not pred_blocks:
                print("  WARNING: no predictions for any year!")
                continue

            bigtest = pl.concat(pred_blocks)
            del pred_blocks; gc.collect()
            print(f"\n  Total test samples: {len(bigtest):,}")
            bigtest.write_parquet(f"{DATA_DIR}/preds_{SCRIPT_STEM}_act{actdays}.parquet")

            # ---------------------------------------------------------------
            # 3. Backtest (top-50 long-only, d1 triple filter)
            # ---------------------------------------------------------------
            input_filename = f"SHSZ_{actdays}_EODtestraw.parquet"
            filename = f"tcn_256d_{input_filename}"

            topN = 50
            thresh = 0.2

            zz500 = bigtest.filter(
                pl.col("ID") == "SH000905"
            ).select(
                pl.col("Date"),
                pl.col("y").alias("y_zz500"),
            )

            bigtest2 = (
                bigtest
                .with_columns([
                    ((pl.col("d1") - 1.1).abs() * 100).alias("d1_1"),
                    ((pl.col("d1") - 1.05).abs() * 100).alias("d1_st"),
                    ((pl.col("d1") - 1.2).abs() * 100).alias("d1_chy"),
                ])
                .filter(
                    (pl.col("d1_1") >= thresh)
                    & (pl.col("d1_st") >= thresh * 1)
                    & (pl.col("d1_chy") >= thresh)
                )
                .with_columns(y=pl.col("y") - 1)
            )

            strategy = (
                bigtest2
                .with_columns(
                    pl.col("pred_y")
                    .rank(method="average", descending=True)
                    .over("Date")
                    .alias("cs_rank")
                )
                .filter(pl.col("cs_rank") <= topN)
                .group_by("Date")
                .agg(pl.col("y").mean().alias("strategy_ret"))
                .sort("Date")
                .join(zz500, on="Date", how="inner")
                .with_columns(
                    (pl.col("strategy_ret") - pl.col("y_zz500") + 1)
                    .alias("strategy_ret2")
                )
                .with_columns(
                    (1 + pl.col("strategy_ret")).cum_prod().alias("cum_ret"),
                    (1 + pl.col("strategy_ret2")).cum_prod().alias("cum_ret2"),
                )
            )

            # ---- Performance report ----
            p2 = (
                ggplot(strategy.to_pandas(), aes(x="Date", y="cum_ret"))
                + geom_line()
                + labs(title=filename, x="Date", y="Cumulative Return")
            )

            out = pnl_report_polars_plotnine(
                strategy, date_col="Date", ret_col="strategy_ret",
                info=filename, freq=52
            )
            perf_file.write(f"Run: {filename}\n")
            perf_file.write("-" * 80 + "\n")
            perf_file.write("[metrics]\n")
            perf_file.write(out["metrics"].to_pandas().to_string(index=False))
            perf_file.write("\n\n[calendar]\n")
            perf_file.write(out["calendar"].to_pandas().to_string(index=False))
            perf_file.write("\n\n")

            print(out["metrics"])
            print(out["calendar"])

            # Save figures
            for fig_obj in [p2, out["plot_wealth"], out["plot_dd"], out["plot_ret"]]:
                fig = fig_obj.draw()
                pdf_pages.savefig(fig)
                plt.close(fig)

            del y_arr, d1_arr, dates_arr, ids_arr
            del X_all, y_all, end_dates, window_ends
            del bigtest, bigtest2, strategy, zz500; gc.collect()

    print(f"\n{'=' * 60}")
    print(f"Saved all plots to: {plots_pdf}")
    print(f"Saved performance report to: {perf_txt}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
