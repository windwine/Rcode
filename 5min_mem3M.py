"""
SPY Hedge Ratio Backtest (2026-06-07, 5min + memory threshold variant)

Strip-level processing of 10-leg symmetric strip (5 puts + 5 calls).
The cap (when active) is applied to the STRIP's aggregate dollar exposure,
not to each leg's delta independently.

Memory-threshold rebalancing: the hedge position is only updated when the
dollar change in desired exposure exceeds the threshold ($3M or $5M variant).

Variants compared (each with $3M and $5M threshold):
1. Plain      : 1.0 hedge ratio, no multipliers, no cap
2. Multiplier : weekday ratio (Mon=1.1, Tue-Fri=1.3) x ATM vol multiplier (1.1 if vol<14.15), uncapped
3. Mult + Cap : same multiplier, capped at $25M strip-level dollar exposure
"""

import glob
import html
import os
import time
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import BytesIO
import base64

import matplotlib
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from plotnine import (
    aes, geom_col, geom_line, ggplot, labs, theme,
    theme_minimal, element_text,
)

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# -- Config --
BASE_DATA_DIR = r"E:\spyderrock\data\etf_intraday_options"
OUTPUT_DIR = r"E:\spyderrock\strat\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKER = "SPY"
STRIP_DELTAS = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5]
HEDGE_INTERVALS = [5]
NOTIONAL = 50_000_000   # total strip notional
CAP_DOLLARS = 25_000_000  # 50% of notional strip-level dollar cap on hedge exposure

# Memory-threshold rebalance gates
THRESHOLD_DOLLARS = 3_000_000     # $3M rebalance gate
THRESHOLD_DOLLARS_ALT = 5_000_000  # $5M alternative gate

# Entry / Close times are wall-clock America/New_York. Converted to UTC per-day
# (via zoneinfo) so the comparison against the parquet timestamp column (UTC)
# is correct year-round across DST boundaries.
ENTRY_TIME = "09:45:00"   # 9:45 AM ET
CLOSE_TIME = "16:00:00"   # 4:00 PM ET
START_DATE = "2022-01-01"

ET_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")


def et_to_utc_naive(day_date, et_time_str):
    """Wall-clock ET time on day_date -> naive UTC datetime matching
    parquet timestamp (which is UTC). Handles DST via zoneinfo."""
    t = datetime.strptime(et_time_str, "%H:%M:%S").time()
    return (
        datetime.combine(day_date, t)
        .replace(tzinfo=ET_TZ)
        .astimezone(UTC_TZ)
        .replace(tzinfo=None)
    )


ATM_VOL_THRESHOLD = 0.1415
VOL_MULT_LOW = 1.1
VOL_MULT_OTHER = 1.0

# Realized-vol annualization: 5-min log returns -> annualized vol comparable to
# the (BS-annualized) ATM IV. 78 = 390min RTH / 5min = 5-min bars per RTH day.
# Built from the raw snapshots (see compute_realized_vol) rather than strip_log, so
# the number is identical to the hourly-hedge script's and the two reports compare
# day-for-day.
RV_BARS_PER_DAY = 78
RV_ANN_FACTOR = np.sqrt(252 * RV_BARS_PER_DAY)
N_RECENT_DAYS = 10

WEEKDAY_HEDGE_RATIO = {0: 1.1, 1: 1.3, 2: 1.3, 3: 1.3, 4: 1.3}
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]

KEEP_COLS = ["timestamp", "okey_xx", "okey_cp", "okey_dt", "uPrc", "bidPrc", "askPrc", "de", "atmVol"]


def _fmt_money(x):
    return f"${x:,.2f}"


def _fmt_num(x, nd=4):
    return f"{x:.{nd}f}"


def _fmt_pct(x, nd=2):
    return f"{x*100:.{nd}f}%"


def _fmt_pct_na(x):
    return (f"{x*100:.2f}%"
            if (x is not None and not (isinstance(x, float) and np.isnan(x))) else "n/a")


def _table_html(df, col_formatters=None):
    if col_formatters is None:
        col_formatters = {}
    cols = df.columns
    rows = df.to_dicts()
    parts = ["<table><thead><tr>"]
    for c in cols:
        parts.append(f"<th>{html.escape(str(c))}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for c in cols:
            v = row[c]
            if v is None:
                cell = ""
            elif c in col_formatters:
                cell = col_formatters[c](v)
            else:
                cell = str(v)
            parts.append(f"<td>{html.escape(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _fig_to_base64(plot_obj):
    fig = plot_obj.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return out


def get_files():
    data_dir = os.path.join(BASE_DATA_DIR, TICKER)
    pattern = os.path.join(data_dir, f"{TICKER}_options_*.parquet")
    files = sorted(glob.glob(pattern))
    return [f for f in files if os.path.basename(f) >= f"{TICKER}_options_{START_DATE}"]


def load_day(path):
    df = pl.scan_parquet(path).select(KEEP_COLS).collect()
    df = df.with_columns(
        pl.col("timestamp").str.slice(0, 19)
        .str.to_datetime("%Y-%m-%d %H:%M:%S")
        .dt.truncate("1m")
        .alias("timestamp")
    )
    # bad-tick guard on uPrc
    minute_uprc = (
        df.group_by("timestamp", maintain_order=True)
        .agg(pl.col("uPrc").first())
        .sort("timestamp")
    )
    prices = minute_uprc["uPrc"].to_list()
    clean = prices.copy()
    for i in range(1, len(clean)):
        if clean[i-1] > 0 and abs(prices[i] / clean[i-1] - 1) > 0.10:
            clean[i] = clean[i-1]
    minute_uprc = minute_uprc.with_columns(pl.Series("uPrc_clean", clean)).select("timestamp", "uPrc_clean")
    df = df.join(minute_uprc, on="timestamp", how="left")
    df = df.with_columns(pl.col("uPrc_clean").alias("uPrc")).drop("uPrc_clean")
    return df


def get_day_scalers(df, entry_ts):
    day_date = df["timestamp"][0].date()
    wd = day_date.weekday()
    # 0DTE ONLY -- atmVol is published per expiry, and ~32-35 expiries quote at 09:45
    # with 0DTE only ~3% of the rows. Taking the median across all of them returns a
    # term-structure blend, not the vol of the options this strip actually sells.
    # The error is not a constant offset: 0DTE prints far ABOVE the curve on event
    # days (2022-05-04: 38.14% vs 24.67% blended) and far BELOW it on quiet ones
    # (2025-06-16: 9.70% vs 15.58%). 63% of days differ by >2 vol points, so the
    # VOL_MULT_LOW gate below used to fire on the wrong series -- in 2026 it triggered
    # on 15% of days when the 0DTE vol said 68%. See exp20260803.md section 10.4.
    entry_snap = df.filter(
        (pl.col("timestamp") == entry_ts) & (pl.col("okey_dt") == str(day_date))
    )
    atm_series = entry_snap["atmVol"].drop_nulls()
    entry_atmvol = float(atm_series.median()) if atm_series.len() > 0 else np.nan
    weekday_ratio = WEEKDAY_HEDGE_RATIO.get(wd, 1.3)
    # Anomaly #2: atmVol gap (0/NaN) -> no low-vol bonus; require a strictly positive vol.
    vol_mult = VOL_MULT_LOW if (not np.isnan(entry_atmvol) and 0.0 < entry_atmvol < ATM_VOL_THRESHOLD) else VOL_MULT_OTHER
    effective_ratio = weekday_ratio * vol_mult
    return {
        "weekday": WEEKDAY_NAMES[wd] if wd < 5 else "Other",
        "entry_atmvol": entry_atmvol,
        "weekday_ratio": weekday_ratio,
        "vol_mult": vol_mult,
        "effective_ratio": effective_ratio,
    }


def compute_realized_vol(df_day, entry_ts, close_dt):
    """Annualized realized vol of SPY over the entry->close window, from 5-minute
    log returns. The level path starts AT the entry snapshot, so every return used
    is strictly after entry (no look-ahead into the pre-entry session).

    Deliberately built from df_day, not from strip_log: strip_log is filtered to
    bars where all 10 legs quote, so it would make the underlying's realized vol
    depend on option-chain availability. This definition matches the hourly-hedge
    script exactly, so RV is comparable across both SPY reports.
    """
    path = (
        df_day
        .filter((pl.col("timestamp") >= entry_ts) & (pl.col("timestamp") <= close_dt))
        .group_by(pl.col("timestamp").dt.truncate("5m").alias("_bar"))
        .agg(pl.col("uPrc").first())
        .sort("_bar")
    )
    lev = path["uPrc"].to_numpy()
    # Need >=3 levels for a 2-sample ddof=1 std; guard against zero/negative prices.
    if len(lev) < 3 or not np.all(lev > 0):
        return np.nan, len(lev)
    return float(np.std(np.diff(np.log(lev)), ddof=1) * RV_ANN_FACTOR), len(lev)


def run_day_strip(df_day, target_deltas, hedge_intervals, cap_dollars):
    """Process all strip legs together; memory-threshold rebalancing with
    $3M and $5M parallel gates."""
    day_date = df_day["timestamp"][0].date()
    entry_dt = et_to_utc_naive(day_date, ENTRY_TIME)
    close_dt = et_to_utc_naive(day_date, CLOSE_TIME)

    entry_candidates = df_day.filter(pl.col("timestamp") >= entry_dt)
    if entry_candidates.is_empty():
        return []
    entry_ts = entry_candidates["timestamp"].unique().sort()[0]
    today_str = str(day_date)

    # Pick best leg per target delta
    legs = []
    for td in target_deltas:
        cp = "Put" if td < 0 else "Call"
        snap = df_day.filter(
            (pl.col("timestamp") == entry_ts) & (pl.col("okey_cp") == cp) & (pl.col("okey_dt") == today_str)
        )
        if snap.is_empty():
            return []
        snap = snap.with_columns((pl.col("de") - td).abs().alias("delta_dist"))
        best = snap.sort("delta_dist").head(1)
        strike = best["okey_xx"][0]
        entry_bid = best["bidPrc"][0]
        entry_uprc = best["uPrc"][0]
        if entry_bid <= 0 or entry_uprc <= 0:
            return []
        contract = df_day.filter(
            (pl.col("okey_xx") == strike) & (pl.col("okey_cp") == cp) & (pl.col("okey_dt") == today_str)
        ).sort("timestamp")
        if contract.height < 2:
            return []
        legs.append({"target_delta": td, "cp": cp, "strike": strike,
                     "entry_bid": entry_bid, "entry_uprc": entry_uprc, "contract": contract})

    if len(legs) != len(target_deltas):
        return []

    n_legs = len(legs)
    entry_uprc = float(legs[0]["entry_uprc"])
    per_leg_notional = NOTIONAL / n_legs  # $5M per leg
    per_leg_contracts = per_leg_notional / entry_uprc  # contracts per leg in strip view

    day_scalers = get_day_scalers(df_day, entry_ts)
    # Anomaly #1: atmVol = -999 sentinel = broken underlying snapshot (corrupt uPrc) -> skip.
    if day_scalers["entry_atmvol"] < 0:
        return []
    eff = day_scalers["effective_ratio"]

    # IV vs RV for the day: entry ATM IV (implied, forward-looking at 09:45) against
    # what SPY actually realized over the life of the position. IV - RV > 0 is the
    # variance risk premium the strip seller captured.
    realized_vol, n_rv_bars = compute_realized_vol(df_day, entry_ts, close_dt)
    entry_iv = day_scalers["entry_atmvol"]
    iv_rv_spread = (entry_iv - realized_vol) if (
        not np.isnan(entry_iv) and not np.isnan(realized_vol)) else np.nan

    results = []
    for hi in hedge_intervals:
        # Build per-leg tradelogs aligned in time
        leg_logs = []
        bad_day = False
        for leg in legs:
            cp = leg["cp"]
            strike = leg["strike"]
            contract = leg["contract"]
            entry_bid_leg = leg["entry_bid"]

            rebal_full = contract.filter(
                (pl.col("timestamp") >= entry_ts) & (pl.col("timestamp") <= close_dt)
            ).sort("timestamp")
            if rebal_full.height < 2:
                bad_day = True
                break
            last_row = rebal_full.tail(1)

            if cp == "Put":
                intrinsic_expr = pl.max_horizontal(pl.lit(strike) - pl.col("uPrc"), pl.lit(0.0))
            else:
                intrinsic_expr = pl.max_horizontal(pl.col("uPrc") - pl.lit(strike), pl.lit(0.0))

            rebal = (
                rebal_full
                .with_columns(
                    ((pl.col("timestamp") - pl.lit(entry_ts)).dt.total_minutes() // hi).alias("_bar")
                )
                .group_by("_bar", maintain_order=True)
                .first()
                .drop("_bar")
            )

            if rebal["timestamp"][-1] != last_row["timestamp"][0]:
                rebal = pl.concat([rebal, last_row])
            if rebal.height < 2:
                bad_day = True
                break

            log = (
                rebal
                .with_columns(intrinsic_expr.alias("intrinsic"))
                .with_columns(
                    pl.when(pl.int_range(pl.len()) == pl.len() - 1)
                    .then(pl.col("intrinsic"))
                    .otherwise((pl.col("bidPrc") + pl.col("askPrc")) / 2)
                    .alias("midprice")
                )
                .with_columns(
                    pl.when(pl.int_range(pl.len()) == 0)
                    .then(pl.lit(entry_bid_leg))
                    .otherwise(pl.col("midprice"))
                    .alias("midprice")
                )
                .select(["timestamp", "de", "midprice", "uPrc"])
            )
            leg_logs.append(log)

        if bad_day:
            continue

        # Stack all legs and aggregate per timestamp to get strip net delta
        stacked = pl.concat([log.with_columns(pl.lit(i).alias("leg_id")) for i, log in enumerate(leg_logs)])
        strip_log = (
            stacked
            .group_by("timestamp", maintain_order=True)
            .agg(
                pl.col("de").sum().alias("strip_delta"),          # sum of leg deltas
                pl.col("midprice").sum().alias("strip_mid_sum"),  # for option PnL
                pl.col("uPrc").first().alias("uPrc"),
                pl.len().alias("n_active"),
            )
            .filter(pl.col("n_active") == n_legs)
            .sort("timestamp")
        )

        if strip_log.height < 2:
            continue

        # Derived columns (price change, option PnL, plain desired shares)
        strip_log = strip_log.with_columns(
            (pl.col("uPrc") - pl.col("uPrc").shift(1)).fill_null(0.0).alias("dprc"),
            # option seller PnL: -d(sum_mid) * per_leg_contracts
            (-(pl.col("strip_mid_sum") - pl.col("strip_mid_sum").shift(1)) * per_leg_contracts)
            .fill_null(0.0).alias("op_pnl"),
            # desired hedge shares at t-1 (plain, ratio=1.0): short strip -> hedge LONG +strip_delta
            (pl.col("strip_delta").shift(1) * per_leg_contracts).fill_null(0.0).alias("shares_plain"),
        )

        # ---------------------------------------------------------------
        # Memory-threshold hedge: only rebalance when trade $ > threshold
        # Pull arrays for the stateful pass
        sd = strip_log["strip_delta"].to_numpy()
        upr = strip_log["uPrc"].to_numpy()
        dpr = strip_log["dprc"].to_numpy()
        n_bars = len(sd)

        # uPrc at the previous bar (used to size trade $)
        upr_prev = np.empty(n_bars)
        upr_prev[0] = upr[0]
        upr_prev[1:] = upr[:-1]

        # Desired hedge shares at row i = LONG +strip_delta[i-1] * per_leg_contracts
        # (we SOLD the strip, so our option delta is -strip_delta; hedge is the opposite sign).
        sd_prev = np.empty(n_bars)
        sd_prev[0] = 0.0
        sd_prev[1:] = sd[:-1]
        desired_plain = sd_prev * per_leg_contracts
        desired_mult = sd_prev * per_leg_contracts * eff

        # Capped variant: clip the desired *dollar* exposure, then convert back to shares
        exp_mult_prev = sd_prev * per_leg_contracts * eff * upr_prev
        exp_capped_prev = np.clip(exp_mult_prev, -cap_dollars, cap_dollars)
        desired_capped = np.where(upr_prev > 0, exp_capped_prev / upr_prev, 0.0)

        def memory_held(desired, upr_prev_arr, threshold):
            """Stateful rebalancing: only update position when the
            absolute dollar change > threshold."""
            held = np.zeros(n_bars)
            h = 0.0
            for i in range(n_bars):
                d = desired[i]
                if i == 0:
                    h = d
                elif abs(d - h) * upr_prev_arr[i] > threshold:
                    h = d
                held[i] = h
            return held

        # $3M threshold variants
        held_plain = memory_held(desired_plain, upr_prev, THRESHOLD_DOLLARS)
        held_mult = memory_held(desired_mult, upr_prev, THRESHOLD_DOLLARS * eff)
        held_capped = memory_held(desired_capped, upr_prev, THRESHOLD_DOLLARS * eff)

        # $5M threshold variants (_Sm suffix)
        held_plain_Sm = memory_held(desired_plain, upr_prev, THRESHOLD_DOLLARS_ALT)
        held_mult_Sm = memory_held(desired_mult, upr_prev, THRESHOLD_DOLLARS_ALT * eff)
        held_capped_Sm = memory_held(desired_capped, upr_prev, THRESHOLD_DOLLARS_ALT * eff)

        # Hedge PnL: held_shares[i] * dprc[i]
        hedge_pnl_plain_arr = held_plain * dpr
        hedge_pnl_mult_arr = held_mult * dpr
        hedge_pnl_capped_arr = held_capped * dpr
        hedge_pnl_plain_Sm_arr = held_plain_Sm * dpr
        hedge_pnl_mult_Sm_arr = held_mult_Sm * dpr
        hedge_pnl_capped_Sm_arr = held_capped_Sm * dpr

        # Write PnL columns back to strip_log
        strip_log = strip_log.with_columns([
            pl.Series("hedge_pnl_plain", hedge_pnl_plain_arr),
            pl.Series("hedge_pnl_mult", hedge_pnl_mult_arr),
            pl.Series("hedge_pnl_mult_capped", hedge_pnl_capped_arr),
            pl.Series("hedge_pnl_plain_Sm", hedge_pnl_plain_Sm_arr),
            pl.Series("hedge_pnl_mult_Sm", hedge_pnl_mult_Sm_arr),
            pl.Series("hedge_pnl_mult_capped_Sm", hedge_pnl_capped_Sm_arr),
            pl.Series("held_shares_plain", held_plain),
            pl.Series("held_shares_mult", held_mult),
            pl.Series("held_shares_capped", held_capped),
            # Current-bar exposures for max stats (same definition as always-rebalance file)
            (-pl.col("strip_delta") * per_leg_contracts * pl.col("uPrc")).alias("exposure_plain_curr"),
            (-pl.col("strip_delta") * eff * per_leg_contracts * pl.col("uPrc")).alias("exposure_mult_curr"),
        ])

        op_pnl = float(strip_log["op_pnl"].sum())
        hedge_plain = float(strip_log["hedge_pnl_plain"].sum())
        hedge_mult = float(strip_log["hedge_pnl_mult"].sum())
        hedge_mult_capped = float(strip_log["hedge_pnl_mult_capped"].sum())
        hedge_plain_Sm = float(strip_log["hedge_pnl_plain_Sm"].sum())
        hedge_mult_Sm = float(strip_log["hedge_pnl_mult_Sm"].sum())
        hedge_mult_capped_Sm = float(strip_log["hedge_pnl_mult_capped_Sm"].sum())

        max_exp_plain = float(strip_log["exposure_plain_curr"].abs().max())
        max_exp_mult = float(strip_log["exposure_mult_curr"].abs().max())
        max_exp_mult_capped = float(
            strip_log["exposure_mult_curr"].abs().clip(0.0, cap_dollars).max()
        )

        results.append({
            "date": str(day_date),
            "weekday": day_scalers["weekday"],
            "hedge_interval": hi,
            "entry_uprc": entry_uprc,
            "entry_atmvol": day_scalers["entry_atmvol"],
            "realized_vol": realized_vol,
            "iv_rv_spread": iv_rv_spread,
            "n_rv_bars": n_rv_bars,
            "weekday_ratio": day_scalers["weekday_ratio"],
            "vol_mult": day_scalers["vol_mult"],
            "effective_hedge_ratio": eff,
            "strip_op_pnl": op_pnl,
            # $3M threshold variants
            "strip_hedge_pnl_plain": hedge_plain,
            "strip_hedge_pnl_mult": hedge_mult,
            "strip_hedge_pnl_mult_capped": hedge_mult_capped,
            "strip_total_pnl_plain": op_pnl + hedge_plain,
            "strip_total_pnl_mult": op_pnl + hedge_mult,
            "strip_total_pnl_mult_capped": op_pnl + hedge_mult_capped,
            # $5M threshold variants (_Sm)
            "strip_hedge_pnl_plain_Sm": hedge_plain_Sm,
            "strip_hedge_pnl_mult_Sm": hedge_mult_Sm,
            "strip_hedge_pnl_mult_capped_Sm": hedge_mult_capped_Sm,
            "strip_total_pnl_plain_Sm": op_pnl + hedge_plain_Sm,
            "strip_total_pnl_mult_Sm": op_pnl + hedge_mult_Sm,
            "strip_total_pnl_mult_capped_Sm": op_pnl + hedge_mult_capped_Sm,
            # Max exposures observed (same regardless of threshold)
            "max_exposure_plain": max_exp_plain,
            "max_exposure_mult": max_exp_mult,
            "max_exposure_mult_capped": max_exp_mult_capped,
        })

    return results


# ============================================================
# Main
# ============================================================

files = get_files()
print(f"SPY: {len(files)} daily files")
te = time.time()

all_results = []
for i, f in enumerate(files):
    if i % 100 == 0:
        print(f"  {i+1}/{len(files)} ...", flush=True)
    try:
        df_day = load_day(f)
    except Exception:
        continue
    all_results.extend(run_day_strip(df_day, STRIP_DELTAS, HEDGE_INTERVALS, CAP_DOLLARS))
    del df_day

elapsed = time.time() - te
print(f"Done in {elapsed/60:.1f} min ({len(all_results)} rows)")

if len(all_results) == 0:
    raise RuntimeError("No valid results generated.")

strip = (
    pl.DataFrame(all_results)
    .sort(["hedge_interval", "date"])
    .with_columns(
        pl.col("date").str.to_date("%Y-%m-%d").alias("date_dt"),
        # $3M threshold cumulative PnL
        pl.col("strip_total_pnl_mult").cum_sum().over("hedge_interval").alias("cum_pnl_mult"),
        pl.col("strip_total_pnl_mult_capped").cum_sum().over("hedge_interval").alias("cum_pnl_mult_capped"),
        pl.col("strip_total_pnl_plain").cum_sum().over("hedge_interval").alias("cum_pnl_plain"),
        # $5M threshold cumulative PnL (_Sm)
        pl.col("strip_total_pnl_mult_Sm").cum_sum().over("hedge_interval").alias("cum_pnl_mult_Sm"),
        pl.col("strip_total_pnl_mult_capped_Sm").cum_sum().over("hedge_interval").alias("cum_pnl_mult_capped_Sm"),
        pl.col("strip_total_pnl_plain_Sm").cum_sum().over("hedge_interval").alias("cum_pnl_plain_Sm"),
    )
)


def _variant_stats(pnl_series, cum_series):
    n = pnl_series.len()
    total = float(pnl_series.sum())
    mean_d = float(pnl_series.mean())
    std_d = float(pnl_series.std())
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0.0
    wr = float((pnl_series > 0).sum() / n)
    max_dd = float((cum_series - cum_series.cum_max()).min())
    max_gain = float(pnl_series.max())
    max_loss = float(pnl_series.min())
    return n, total, mean_d, std_d, sharpe, wr, max_dd, max_gain, max_loss


summary_rows = []

for hi in HEDGE_INTERVALS:
    sub = strip.filter(pl.col("hedge_interval") == hi)
    if sub.is_empty():
        continue

    # $3M threshold
    n, t_mult, mu_mult, sd_mult, sh_mult, wr_mult, dd_mult, mg_mult, ml_mult = _variant_stats(
        sub["strip_total_pnl_mult"], sub["cum_pnl_mult"])
    _, t_capped, mu_capped, sd_capped, sh_capped, wr_capped, dd_capped, mg_capped, ml_capped = _variant_stats(
        sub["strip_total_pnl_mult_capped"], sub["cum_pnl_mult_capped"])
    _, t_plain, mu_plain, sd_plain, sh_plain, wr_plain, dd_plain, mg_plain, ml_plain = _variant_stats(
        sub["strip_total_pnl_plain"], sub["cum_pnl_plain"])

    # $5M threshold (_Sm)
    _, t_mult_Sm, mu_mult_Sm, sd_mult_Sm, sh_mult_Sm, wr_mult_Sm, dd_mult_Sm, mg_mult_Sm, ml_mult_Sm = _variant_stats(
        sub["strip_total_pnl_mult_Sm"], sub["cum_pnl_mult_Sm"])
    _, t_capped_Sm, mu_capped_Sm, sd_capped_Sm, sh_capped_Sm, wr_capped_Sm, dd_capped_Sm, mg_capped_Sm, ml_capped_Sm = _variant_stats(
        sub["strip_total_pnl_mult_capped_Sm"], sub["cum_pnl_mult_capped_Sm"])
    _, t_plain_Sm, mu_plain_Sm, sd_plain_Sm, sh_plain_Sm, wr_plain_Sm, dd_plain_Sm, mg_plain_Sm, ml_plain_Sm = _variant_stats(
        sub["strip_total_pnl_plain_Sm"], sub["cum_pnl_plain_Sm"])

    summary_rows.append({
        "hedge_interval": hi,
        "n_days": n,
        # $3M — Multiplier (Uncapped)
        "total_pnl_mult": t_mult,
        "mean_daily_mult": mu_mult,
        "std_daily_mult": sd_mult,
        "sharpe_mult": sh_mult,
        "win_rate_mult": wr_mult,
        "max_dd_mult": dd_mult,
        "max_gain_mult": mg_mult,
        "max_loss_mult": ml_mult,
        # $3M — Multiplier + 50% Notional Cap
        "total_pnl_capped": t_capped,
        "mean_daily_capped": mu_capped,
        "std_daily_capped": sd_capped,
        "sharpe_capped": sh_capped,
        "win_rate_capped": wr_capped,
        "max_dd_capped": dd_capped,
        "max_gain_capped": mg_capped,
        "max_loss_capped": ml_capped,
        # $3M — Plain
        "total_pnl_plain": t_plain,
        "mean_daily_plain": mu_plain,
        "std_daily_plain": sd_plain,
        "sharpe_plain": sh_plain,
        "win_rate_plain": wr_plain,
        "max_dd_plain": dd_plain,
        "max_gain_plain": mg_plain,
        "max_loss_plain": ml_plain,
        # $5M — Multiplier (Uncapped)
        "total_pnl_mult_Sm": t_mult_Sm,
        "mean_daily_mult_Sm": mu_mult_Sm,
        "std_daily_mult_Sm": sd_mult_Sm,
        "sharpe_mult_Sm": sh_mult_Sm,
        "win_rate_mult_Sm": wr_mult_Sm,
        "max_dd_mult_Sm": dd_mult_Sm,
        "max_gain_mult_Sm": mg_mult_Sm,
        "max_loss_mult_Sm": ml_mult_Sm,
        # $5M — Multiplier + 50% Notional Cap
        "total_pnl_capped_Sm": t_capped_Sm,
        "mean_daily_capped_Sm": mu_capped_Sm,
        "std_daily_capped_Sm": sd_capped_Sm,
        "sharpe_capped_Sm": sh_capped_Sm,
        "win_rate_capped_Sm": wr_capped_Sm,
        "max_dd_capped_Sm": dd_capped_Sm,
        "max_gain_capped_Sm": mg_capped_Sm,
        "max_loss_capped_Sm": ml_capped_Sm,
        # $5M — Plain
        "total_pnl_plain_Sm": t_plain_Sm,
        "mean_daily_plain_Sm": mu_plain_Sm,
        "std_daily_plain_Sm": sd_plain_Sm,
        "sharpe_plain_Sm": sh_plain_Sm,
        "win_rate_plain_Sm": wr_plain_Sm,
        "max_dd_plain_Sm": dd_plain_Sm,
        "max_gain_plain_Sm": mg_plain_Sm,
        "max_loss_plain_Sm": ml_plain_Sm,
        # Max exposures observed
        "max_exp_mult": float(sub["max_exposure_mult"].max()),
        "max_exp_capped": float(sub["max_exposure_mult_capped"].max()),
        "max_exp_plain": float(sub["max_exposure_plain"].max()),
    })

summary = pl.DataFrame(summary_rows).sort("sharpe_mult", descending=True)

# -- Recent N trading days: Entry ATM IV vs Realized Vol + $3M and $5M PnL --
recent = (
    strip.filter(pl.col("hedge_interval") == HEDGE_INTERVALS[0])
    .sort("date")
    .tail(N_RECENT_DAYS)
    .select([
        "date", "weekday", "entry_atmvol", "realized_vol", "iv_rv_spread",
        "strip_total_pnl_plain", "strip_total_pnl_mult", "strip_total_pnl_mult_capped",
        "strip_total_pnl_plain_Sm", "strip_total_pnl_mult_Sm", "strip_total_pnl_mult_capped_Sm",
    ])
)


def _pct_or_na(x):
    return f"{x*100:.2f}%" if (x is not None and not np.isnan(x)) else "  n/a"


_W = 12 + 5 + 9 + 10 + 9 + 6 * 13   # widths below; keeps the rules flush with the row
print("\n" + "=" * _W)
print(f"  RECENT {N_RECENT_DAYS} TRADING DAYS — Entry ATM IV (09:45) vs Realized Vol (09:45->16:00, annualized)")
print(f"  PnL shown for BOTH rebalance gates: $3M threshold and $5M threshold")
print("=" * _W)
print(f"{'':<12}{'':<5}{'':>9}{'':>10}{'':>9}"
      f"{'---------- $3M gate ----------':^39}{'---------- $5M gate ----------':^39}")
print(f"{'Date':<12}{'WD':<5}{'EntryIV':>9}{'RealVol':>10}{'IV-RV':>9}"
      f"{'Plain':>13}{'Mult':>13}{'Mult+Cap':>13}"
      f"{'Plain':>13}{'Mult':>13}{'Mult+Cap':>13}")
print("-" * _W)
for r in recent.to_dicts():
    print(f"{r['date']:<12}{r['weekday']:<5}"
          f"{_pct_or_na(r['entry_atmvol']):>9}{_pct_or_na(r['realized_vol']):>10}"
          f"{_pct_or_na(r['iv_rv_spread']):>9}"
          f"{r['strip_total_pnl_plain']:>13,.0f}{r['strip_total_pnl_mult']:>13,.0f}"
          f"{r['strip_total_pnl_mult_capped']:>13,.0f}"
          f"{r['strip_total_pnl_plain_Sm']:>13,.0f}{r['strip_total_pnl_mult_Sm']:>13,.0f}"
          f"{r['strip_total_pnl_mult_capped_Sm']:>13,.0f}")
print("=" * _W)

# Full-sample IV vs RV averages (context for the recent table).
# Polars trap: NaN sorts as the largest value, so `> 0` alone does NOT drop NaN --
# is_finite() is required (SPY's atmVol gap is NaN, not null).
_base = strip.filter(pl.col("hedge_interval") == HEDGE_INTERVALS[0])
_iv = _base["entry_atmvol"].drop_nulls()
_iv = _iv.filter(_iv.is_finite() & (_iv > 0))
_rv = _base["realized_vol"].drop_nulls()
_rv = _rv.filter(_rv.is_finite() & (_rv > 0))
_spr = _base["iv_rv_spread"].drop_nulls()
_spr = _spr.filter(_spr.is_finite())
avg_iv_val = _fmt_pct(float(_iv.mean())) if _iv.len() else "n/a"
avg_rv_val = _fmt_pct(float(_rv.mean())) if _rv.len() else "n/a"
avg_spread_val = _fmt_pct(float(_spr.mean())) if _spr.len() else "n/a"
pct_iv_gt_rv = _fmt_pct(float((_spr > 0).sum() / _spr.len())) if _spr.len() else "n/a"

print("\n" + "=" * 110)
print("  SPY HEDGE RATIO (strip-level aggregate cap, memory-threshold rebalancing)")
print("=" * 110)
with pl.Config(tbl_cols=-1, tbl_width_chars=240, float_precision=4):
    print(summary)

# -- Weekday breakdown --
weekday_rows = []
for wd in WEEKDAY_NAMES:
    sub = strip.filter(pl.col("weekday") == wd).sort("date")
    if sub.is_empty():
        continue
    n_days = sub.height

    def wd_stats(series):
        total = float(series.sum())
        mean_d = float(series.mean())
        std_d = float(series.std())
        sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0.0
        wr = float((series > 0).sum() / n_days)
        cum = series.cum_sum()
        dd = float((cum - cum.cum_max()).min())
        mg = float(series.max())
        ml = float(series.min())
        return total, mean_d, std_d, sharpe, wr, dd, mg, ml

    # $3M threshold
    t_mu, m_mu, s_mu, sh_mu, wr_mu, dd_mu, mg_mu, ml_mu = wd_stats(sub["strip_total_pnl_mult"])
    t_cp, m_cp, s_cp, sh_cp, wr_cp, dd_cp, mg_cp, ml_cp = wd_stats(sub["strip_total_pnl_mult_capped"])
    t_pl, m_pl, s_pl, sh_pl, wr_pl, dd_pl, mg_pl, ml_pl = wd_stats(sub["strip_total_pnl_plain"])

    # $5M threshold (_Sm)
    t_muS, m_muS, s_muS, sh_muS, wr_muS, dd_muS, mg_muS, ml_muS = wd_stats(sub["strip_total_pnl_mult_Sm"])
    t_cpS, m_cpS, s_cpS, sh_cpS, wr_cpS, dd_cpS, mg_cpS, ml_cpS = wd_stats(sub["strip_total_pnl_mult_capped_Sm"])
    t_plS, m_plS, s_plS, sh_plS, wr_plS, dd_plS, mg_plS, ml_plS = wd_stats(sub["strip_total_pnl_plain_Sm"])

    weekday_rows.append({
        "weekday": wd,
        "n_days": n_days,
        # $3M
        "total_pnl_mult": t_mu,
        "mean_daily_mult": m_mu,
        "std_daily_mult": s_mu,
        "sharpe_mult": sh_mu,
        "win_rate_mult": wr_mu,
        "max_dd_mult": dd_mu,
        "max_gain_mult": mg_mu,
        "max_loss_mult": ml_mu,
        "total_pnl_capped": t_cp,
        "mean_daily_capped": m_cp,
        "std_daily_capped": s_cp,
        "sharpe_capped": sh_cp,
        "win_rate_capped": wr_cp,
        "max_dd_capped": dd_cp,
        "max_gain_capped": mg_cp,
        "max_loss_capped": ml_cp,
        "total_pnl_plain": t_pl,
        "mean_daily_plain": m_pl,
        "std_daily_plain": s_pl,
        "sharpe_plain": sh_pl,
        "win_rate_plain": wr_pl,
        "max_dd_plain": dd_pl,
        "max_gain_plain": mg_pl,
        "max_loss_plain": ml_pl,
        # $5M
        "total_pnl_mult_Sm": t_muS,
        "mean_daily_mult_Sm": m_muS,
        "std_daily_mult_Sm": s_muS,
        "sharpe_mult_Sm": sh_muS,
        "win_rate_mult_Sm": wr_muS,
        "max_dd_mult_Sm": dd_muS,
        "max_gain_mult_Sm": mg_muS,
        "max_loss_mult_Sm": ml_muS,
        "total_pnl_capped_Sm": t_cpS,
        "mean_daily_capped_Sm": m_cpS,
        "std_daily_capped_Sm": s_cpS,
        "sharpe_capped_Sm": sh_cpS,
        "win_rate_capped_Sm": wr_cpS,
        "max_dd_capped_Sm": dd_cpS,
        "max_gain_capped_Sm": mg_cpS,
        "max_loss_capped_Sm": ml_cpS,
        "total_pnl_plain_Sm": t_plS,
        "mean_daily_plain_Sm": m_plS,
        "std_daily_plain_Sm": s_plS,
        "sharpe_plain_Sm": sh_plS,
        "win_rate_plain_Sm": wr_plS,
        "max_dd_plain_Sm": dd_plS,
        "max_gain_plain_Sm": mg_plS,
        "max_loss_plain_Sm": ml_plS,
    })

weekday_summary = pl.DataFrame(weekday_rows)


def _view(df, suffix, with_n=False):
    cols = ["weekday"] if "weekday" in df.columns else ["hedge_interval"]
    if with_n:
        cols.append("n_days")
    metric_cols = [
        f"total_pnl{suffix}", f"mean_daily{suffix}",
        f"std_daily{suffix}", f"sharpe{suffix}",
        f"win_rate{suffix}", f"max_dd{suffix}",
        f"max_gain{suffix}", f"max_loss{suffix}",
    ]
    out = df.select(cols + metric_cols).rename({
        f"total_pnl{suffix}": "total_pnl",
        f"mean_daily{suffix}": "mean_daily",
        f"std_daily{suffix}": "std_daily",
        f"sharpe{suffix}": "sharpe",
        f"win_rate{suffix}": "win_rate",
        f"max_dd{suffix}": "max_drawdown",
        f"max_gain{suffix}": "max_daily_gain",
        f"max_loss{suffix}": "max_daily_loss",
    })
    return out


# -- Charts --

# Cumulative PnL ($3M threshold)
cum_plot = (
    ggplot(strip, aes(x="date_dt"))
    + geom_line(aes(y="cum_pnl_mult", color="'Multiplier'"), size=0.6)
    + geom_line(aes(y="cum_pnl_mult_capped", color="'Mult + Cap'"), size=0.6)
    + geom_line(aes(y="cum_pnl_plain", color="'Plain'"), size=0.6)
    + labs(title="Cumulative PnL ($3M Threshold)", x="", y="PnL ($)")
    + theme_minimal()
)

# Cumulative PnL ($5M threshold)
cum_plot_Sm = (
    ggplot(strip, aes(x="date_dt"))
    + geom_line(aes(y="cum_pnl_mult_Sm", color="'Multiplier'"), size=0.6)
    + geom_line(aes(y="cum_pnl_mult_capped_Sm", color="'Mult + Cap'"), size=0.6)
    + geom_line(aes(y="cum_pnl_plain_Sm", color="'Plain'"), size=0.6)
    + labs(title="Cumulative PnL ($5M Threshold)", x="", y="PnL ($)")
    + theme_minimal()
)

# Weekday breakdown ($3M threshold)
weekday_mult = (
    strip.group_by("weekday").agg(
        pl.col("strip_total_pnl_mult").sum().alias("total_pnl"),
        pl.col("strip_total_pnl_mult").mean().alias("mean_daily"),
        pl.len().alias("n"),
    ).sort("weekday")
)

wd_mult_plot = (
    ggplot(weekday_mult, aes(x="weekday", y="total_pnl"))
    + geom_col(fill="#4472C4")
    + labs(title="Weekday PnL — Multiplier ($3M)", x="", y="Total PnL ($)")
    + theme_minimal()
    + theme(figure_size=(5, 3))
)

wd_capped = (
    strip.group_by("weekday").agg(
        pl.col("strip_total_pnl_mult_capped").sum().alias("total_pnl"),
        pl.col("strip_total_pnl_mult_capped").mean().alias("mean_daily"),
        pl.len().alias("n"),
    ).sort("weekday")
)

wd_capped_plot = (
    ggplot(wd_capped, aes(x="weekday", y="total_pnl"))
    + geom_col(fill="#ED7D31")
    + labs(title="Weekday PnL — Mult + Cap ($3M)", x="", y="Total PnL ($)")
    + theme_minimal()
    + theme(figure_size=(5, 3))
)

wd_plain = (
    strip.group_by("weekday").agg(
        pl.col("strip_total_pnl_plain").sum().alias("total_pnl"),
        pl.col("strip_total_pnl_plain").mean().alias("mean_daily"),
        pl.len().alias("n"),
    ).sort("weekday")
)

wd_plain_plot = (
    ggplot(wd_plain, aes(x="weekday", y="total_pnl"))
    + geom_col(fill="#A5A5A5")
    + labs(title="Weekday PnL — Plain ($3M)", x="", y="Total PnL ($)")
    + theme_minimal()
    + theme(figure_size=(5, 3))
)

# Weekday breakdown ($5M threshold)
weekday_mult_Sm = (
    strip.group_by("weekday").agg(
        pl.col("strip_total_pnl_mult_Sm").sum().alias("total_pnl"),
        pl.col("strip_total_pnl_mult_Sm").mean().alias("mean_daily"),
        pl.len().alias("n"),
    ).sort("weekday")
)

wd_mult_plot_Sm = (
    ggplot(weekday_mult_Sm, aes(x="weekday", y="total_pnl"))
    + geom_col(fill="#4472C4")
    + labs(title="Weekday PnL — Multiplier ($5M)", x="", y="Total PnL ($)")
    + theme_minimal()
    + theme(figure_size=(5, 3))
)

wd_capped_Sm = (
    strip.group_by("weekday").agg(
        pl.col("strip_total_pnl_mult_capped_Sm").sum().alias("total_pnl"),
        pl.col("strip_total_pnl_mult_capped_Sm").mean().alias("mean_daily"),
        pl.len().alias("n"),
    ).sort("weekday")
)

wd_capped_plot_Sm = (
    ggplot(wd_capped_Sm, aes(x="weekday", y="total_pnl"))
    + geom_col(fill="#ED7D31")
    + labs(title="Weekday PnL — Mult + Cap ($5M)", x="", y="Total PnL ($)")
    + theme_minimal()
    + theme(figure_size=(5, 3))
)

wd_plain_Sm = (
    strip.group_by("weekday").agg(
        pl.col("strip_total_pnl_plain_Sm").sum().alias("total_pnl"),
        pl.col("strip_total_pnl_plain_Sm").mean().alias("mean_daily"),
        pl.len().alias("n"),
    ).sort("weekday")
)

wd_plain_plot_Sm = (
    ggplot(wd_plain_Sm, aes(x="weekday", y="total_pnl"))
    + geom_col(fill="#A5A5A5")
    + labs(title="Weekday PnL — Plain ($5M)", x="", y="Total PnL ($)")
    + theme_minimal()
    + theme(figure_size=(5, 3))
)

# Convert charts to base64
img_cum = _fig_to_base64(cum_plot)
img_cum_Sm = _fig_to_base64(cum_plot_Sm)
img_wd_mult = _fig_to_base64(wd_mult_plot)
img_wd_capped = _fig_to_base64(wd_capped_plot)
img_wd_plain = _fig_to_base64(wd_plain_plot)
img_wd_mult_Sm = _fig_to_base64(wd_mult_plot_Sm)
img_wd_capped_Sm = _fig_to_base64(wd_capped_plot_Sm)
img_wd_plain_Sm = _fig_to_base64(wd_plain_plot_Sm)

# -- Summary Tables --

# $3M threshold tables
summary_mult_tbl = _table_html(summary.select([
    "hedge_interval", "n_days", "total_pnl_mult", "mean_daily_mult",
    "sharpe_mult", "win_rate_mult", "max_dd_mult"
]).rename({
    "hedge_interval": "Interval (min)",
    "n_days": "Days",
    "total_pnl_mult": "Total PnL",
    "mean_daily_mult": "Mean Daily",
    "sharpe_mult": "Sharpe",
    "win_rate_mult": "Win Rate",
    "max_dd_mult": "Max DD",
}), col_formatters={
    "Total PnL": _fmt_money,
    "Mean Daily": _fmt_money,
    "Sharpe": lambda x: _fmt_num(x, 2),
    "Win Rate": _fmt_pct,
    "Max DD": _fmt_money,
})

summary_capped_tbl = _table_html(summary.select([
    "hedge_interval", "n_days", "total_pnl_capped", "mean_daily_capped",
    "sharpe_capped", "win_rate_capped", "max_dd_capped"
]).rename({
    "hedge_interval": "Interval (min)",
    "n_days": "Days",
    "total_pnl_capped": "Total PnL",
    "mean_daily_capped": "Mean Daily",
    "sharpe_capped": "Sharpe",
    "win_rate_capped": "Win Rate",
    "max_dd_capped": "Max DD",
}), col_formatters={
    "Total PnL": _fmt_money,
    "Mean Daily": _fmt_money,
    "Sharpe": lambda x: _fmt_num(x, 2),
    "Win Rate": _fmt_pct,
    "Max DD": _fmt_money,
})

summary_plain_tbl = _table_html(summary.select([
    "hedge_interval", "n_days", "total_pnl_plain", "mean_daily_plain",
    "sharpe_plain", "win_rate_plain", "max_dd_plain"
]).rename({
    "hedge_interval": "Interval (min)",
    "n_days": "Days",
    "total_pnl_plain": "Total PnL",
    "mean_daily_plain": "Mean Daily",
    "sharpe_plain": "Sharpe",
    "win_rate_plain": "Win Rate",
    "max_dd_plain": "Max DD",
}), col_formatters={
    "Total PnL": _fmt_money,
    "Mean Daily": _fmt_money,
    "Sharpe": lambda x: _fmt_num(x, 2),
    "Win Rate": _fmt_pct,
    "Max DD": _fmt_money,
})

# $5M threshold tables (_Sm)
summary_mult_tbl_Sm = _table_html(summary.select([
    "hedge_interval", "n_days", "total_pnl_mult_Sm", "mean_daily_mult_Sm",
    "sharpe_mult_Sm", "win_rate_mult_Sm", "max_dd_mult_Sm"
]).rename({
    "hedge_interval": "Interval (min)",
    "n_days": "Days",
    "total_pnl_mult_Sm": "Total PnL",
    "mean_daily_mult_Sm": "Mean Daily",
    "sharpe_mult_Sm": "Sharpe",
    "win_rate_mult_Sm": "Win Rate",
    "max_dd_mult_Sm": "Max DD",
}), col_formatters={
    "Total PnL": _fmt_money,
    "Mean Daily": _fmt_money,
    "Sharpe": lambda x: _fmt_num(x, 2),
    "Win Rate": _fmt_pct,
    "Max DD": _fmt_money,
})

summary_capped_tbl_Sm = _table_html(summary.select([
    "hedge_interval", "n_days", "total_pnl_capped_Sm", "mean_daily_capped_Sm",
    "sharpe_capped_Sm", "win_rate_capped_Sm", "max_dd_capped_Sm"
]).rename({
    "hedge_interval": "Interval (min)",
    "n_days": "Days",
    "total_pnl_capped_Sm": "Total PnL",
    "mean_daily_capped_Sm": "Mean Daily",
    "sharpe_capped_Sm": "Sharpe",
    "win_rate_capped_Sm": "Win Rate",
    "max_dd_capped_Sm": "Max DD",
}), col_formatters={
    "Total PnL": _fmt_money,
    "Mean Daily": _fmt_money,
    "Sharpe": lambda x: _fmt_num(x, 2),
    "Win Rate": _fmt_pct,
    "Max DD": _fmt_money,
})

summary_plain_tbl_Sm = _table_html(summary.select([
    "hedge_interval", "n_days", "total_pnl_plain_Sm", "mean_daily_plain_Sm",
    "sharpe_plain_Sm", "win_rate_plain_Sm", "max_dd_plain_Sm"
]).rename({
    "hedge_interval": "Interval (min)",
    "n_days": "Days",
    "total_pnl_plain_Sm": "Total PnL",
    "mean_daily_plain_Sm": "Mean Daily",
    "sharpe_plain_Sm": "Sharpe",
    "win_rate_plain_Sm": "Win Rate",
    "max_dd_plain_Sm": "Max DD",
}), col_formatters={
    "Total PnL": _fmt_money,
    "Mean Daily": _fmt_money,
    "Sharpe": lambda x: _fmt_num(x, 2),
    "Win Rate": _fmt_pct,
    "Max DD": _fmt_money,
})

# Weekday tables ($3M)
weekday_mult_tbl = _table_html(weekday_mult.select(["weekday", "n", "total_pnl", "mean_daily"]).rename({
    "weekday": "Weekday", "n": "Days", "total_pnl": "Total PnL", "mean_daily": "Mean Daily"
}), col_formatters={"Total PnL": _fmt_money, "Mean Daily": _fmt_money})

weekday_capped_tbl = _table_html(wd_capped.select(["weekday", "n", "total_pnl", "mean_daily"]).rename({
    "weekday": "Weekday", "n": "Days", "total_pnl": "Total PnL", "mean_daily": "Mean Daily"
}), col_formatters={"Total PnL": _fmt_money, "Mean Daily": _fmt_money})

weekday_plain_tbl = _table_html(wd_plain.select(["weekday", "n", "total_pnl", "mean_daily"]).rename({
    "weekday": "Weekday", "n": "Days", "total_pnl": "Total PnL", "mean_daily": "Mean Daily"
}), col_formatters={"Total PnL": _fmt_money, "Mean Daily": _fmt_money})

# Weekday tables ($5M)
weekday_mult_tbl_Sm = _table_html(weekday_mult_Sm.select(["weekday", "n", "total_pnl", "mean_daily"]).rename({
    "weekday": "Weekday", "n": "Days", "total_pnl": "Total PnL", "mean_daily": "Mean Daily"
}), col_formatters={"Total PnL": _fmt_money, "Mean Daily": _fmt_money})

weekday_capped_tbl_Sm = _table_html(wd_capped_Sm.select(["weekday", "n", "total_pnl", "mean_daily"]).rename({
    "weekday": "Weekday", "n": "Days", "total_pnl": "Total PnL", "mean_daily": "Mean Daily"
}), col_formatters={"Total PnL": _fmt_money, "Mean Daily": _fmt_money})

weekday_plain_tbl_Sm = _table_html(wd_plain_Sm.select(["weekday", "n", "total_pnl", "mean_daily"]).rename({
    "weekday": "Weekday", "n": "Days", "total_pnl": "Total PnL", "mean_daily": "Mean Daily"
}), col_formatters={"Total PnL": _fmt_money, "Mean Daily": _fmt_money})

# -- KPI overview (headline numbers from summary) --
def _kpi_card(label, value):
    return f'<div class="kpi"><span class="kpi-label">{label}</span><span class="kpi-value">{value}</span></div>'

best_row = summary.sort("sharpe_mult", descending=True).head(1)

kpi_mult_total = _fmt_money(float(best_row["total_pnl_mult"][0]))
kpi_mult_sharpe = _fmt_num(float(best_row["sharpe_mult"][0]), 2)
kpi_mult_wr = _fmt_pct(float(best_row["win_rate_mult"][0]))
kpi_mult_dd = _fmt_money(float(best_row["max_dd_mult"][0]))

kpi_capped_total = _fmt_money(float(best_row["total_pnl_capped"][0]))
kpi_capped_sharpe = _fmt_num(float(best_row["sharpe_capped"][0]), 2)
kpi_capped_wr = _fmt_pct(float(best_row["win_rate_capped"][0]))
kpi_capped_dd = _fmt_money(float(best_row["max_dd_capped"][0]))

kpi_plain_total = _fmt_money(float(best_row["total_pnl_plain"][0]))
kpi_plain_sharpe = _fmt_num(float(best_row["sharpe_plain"][0]), 2)
kpi_plain_wr = _fmt_pct(float(best_row["win_rate_plain"][0]))
kpi_plain_dd = _fmt_money(float(best_row["max_dd_plain"][0]))

# Max Observed Strip Dollar Exposure
max_exp_plain_val = _fmt_money(float(summary["max_exp_plain"].max()))
max_exp_mult_val = _fmt_money(float(summary["max_exp_mult"].max()))
max_exp_capped_val = _fmt_money(float(summary["max_exp_capped"].max()))

# -- Recent N days IV/RV table (HTML) --
recent_tbl = _table_html(recent.rename({
    "date": "Date",
    "weekday": "WD",
    "entry_atmvol": "Entry ATM IV",
    "realized_vol": "Realized Vol (ann.)",
    "iv_rv_spread": "IV − RV",
    "strip_total_pnl_plain": "Plain ($3M)",
    "strip_total_pnl_mult": "Mult ($3M)",
    "strip_total_pnl_mult_capped": "Mult+Cap ($3M)",
    "strip_total_pnl_plain_Sm": "Plain ($5M)",
    "strip_total_pnl_mult_Sm": "Mult ($5M)",
    "strip_total_pnl_mult_capped_Sm": "Mult+Cap ($5M)",
}), col_formatters={
    "Entry ATM IV": _fmt_pct_na,
    "Realized Vol (ann.)": _fmt_pct_na,
    "IV − RV": _fmt_pct_na,
    "Plain ($3M)": _fmt_money,
    "Mult ($3M)": _fmt_money,
    "Mult+Cap ($3M)": _fmt_money,
    "Plain ($5M)": _fmt_money,
    "Mult ($5M)": _fmt_money,
    "Mult+Cap ($5M)": _fmt_money,
})

# -- HTML Report --
html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SPY Hedge Ratio Backtest — 5min Memory Threshold ($3M / $5M)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
  h2 {{ color: #333; margin-top: 30px; }}
  h3 {{ color: #555; }}
  .plot {{ margin: 15px 0; }}
  .plot img {{ max-width: 100%; }}
  table {{ border-collapse: collapse; margin: 10px 0; font-size: 13px; }}
  th, td {{ padding: 6px 12px; text-align: right; border-bottom: 1px solid #ddd; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  tr:hover {{ background: #fafafa; }}
  .kpi-row {{ display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }}
  .kpi {{ background: #f8f9fa; border-radius: 8px; padding: 16px 20px; min-width: 140px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .kpi-label {{ display: block; font-size: 11px; text-transform: uppercase; color: #666; margin-bottom: 4px; }}
  .kpi-value {{ display: block; font-size: 20px; font-weight: 700; color: #222; }}
  .section-toggle {{ margin: 10px 0; }}
  .chart-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
</style>
</head>
<body>

<h1>SPY Hedge Ratio Backtest — 5min Memory Threshold</h1>
<p>10-leg symmetric strip (5 puts + 5 calls). Memory-threshold rebalancing with $3M and $5M parallel gates. 5-minute bar sampling.</p>

<h2>Headline KPIs</h2>
<div class="kpi-row">
  {_kpi_card("Multiplier Total", kpi_mult_total)}
  {_kpi_card("Multiplier Sharpe", kpi_mult_sharpe)}
  {_kpi_card("Multiplier Win Rate", kpi_mult_wr)}
  {_kpi_card("Multiplier Max DD", kpi_mult_dd)}
</div>
<div class="kpi-row">
  {_kpi_card("Mult+Cap Total", kpi_capped_total)}
  {_kpi_card("Mult+Cap Sharpe", kpi_capped_sharpe)}
  {_kpi_card("Mult+Cap Win Rate", kpi_capped_wr)}
  {_kpi_card("Mult+Cap Max DD", kpi_capped_dd)}
</div>
<div class="kpi-row">
  {_kpi_card("Plain Total", kpi_plain_total)}
  {_kpi_card("Plain Sharpe", kpi_plain_sharpe)}
  {_kpi_card("Plain Win Rate", kpi_plain_wr)}
  {_kpi_card("Plain Max DD", kpi_plain_dd)}
</div>

<h2>Max Observed Strip Dollar Exposure</h2>
<div class="kpi-row">
  {_kpi_card("Plain (no cap)", max_exp_plain_val)}
  {_kpi_card("Multiplier (no cap)", max_exp_mult_val)}
  {_kpi_card("Mult + Cap ($25M)", max_exp_capped_val)}
</div>

<h2>Recent {N_RECENT_DAYS} Trading Days — Entry ATM IV vs Realized Vol</h2>
<p>
  <b>Entry ATM IV</b> = SPY <b>0DTE</b> ATM implied vol at the 09:45 ET entry snapshot
  (annualized) &mdash; the expiry this strip actually sells. It is NOT the median across
  all listed expiries: ~32&ndash;35 expiries quote at 09:45 and 0DTE is only ~3% of the
  rows, so that blend runs above 0DTE on quiet days and below it on event days.
  <b>Realized Vol</b> = annualized realized vol of SPY built from
  5-minute log returns over the 09:45&rarr;16:00 window (strictly after entry),
  scaled by &radic;(252&times;78). Both are in the same annualized units, so
  <b>IV &minus; RV &gt; 0</b> means the strip seller captured a positive
  variance-risk premium that day. PnL is shown for <b>both rebalance gates</b> —
  the $3M and $5M memory thresholds — so the same day can be compared across
  turnover regimes.
</p>
<div class="kpi-row">
  {_kpi_card("Avg Entry IV", avg_iv_val)}
  {_kpi_card("Avg Realized Vol", avg_rv_val)}
  {_kpi_card("Avg IV − RV", avg_spread_val)}
  {_kpi_card("Days IV &gt; RV", pct_iv_gt_rv)}
</div>
{recent_tbl}

<h2>Cumulative PnL — $3M Threshold</h2>
<div class="plot"><img src="data:image/png;base64,{img_cum}" alt="Cumulative PnL ($3M)"></div>

<h2>Cumulative PnL — $5M Threshold</h2>
<div class="plot"><img src="data:image/png;base64,{img_cum_Sm}" alt="Cumulative PnL ($5M)"></div>

<h3>Weekday PnL — $3M Threshold</h3>
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
  <div class="plot"><img src="data:image/png;base64,{img_wd_mult}" alt="Weekday PnL — Multiplier ($3M)"></div>
  <div class="plot"><img src="data:image/png;base64,{img_wd_capped}" alt="Weekday PnL — Mult + Cap ($3M)"></div>
  <div class="plot"><img src="data:image/png;base64,{img_wd_plain}" alt="Weekday PnL — Plain ($3M)"></div>
</div>

<h3>Weekday PnL — $5M Threshold</h3>
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
  <div class="plot"><img src="data:image/png;base64,{img_wd_mult_Sm}" alt="Weekday PnL — Multiplier ($5M)"></div>
  <div class="plot"><img src="data:image/png;base64,{img_wd_capped_Sm}" alt="Weekday PnL — Mult + Cap ($5M)"></div>
  <div class="plot"><img src="data:image/png;base64,{img_wd_plain_Sm}" alt="Weekday PnL — Plain ($5M)"></div>
</div>

<h2>Summary — Multiplier (Uncapped) — $3M</h2>
{summary_mult_tbl}

<h2>Summary — Multiplier + 50% Notional Cap — $3M</h2>
{summary_capped_tbl}

<h2>Summary — Plain (1.0) — $3M</h2>
{summary_plain_tbl}

<h2>Summary — Multiplier (Uncapped) — $5M</h2>
{summary_mult_tbl_Sm}

<h2>Summary — Multiplier + 50% Notional Cap — $5M</h2>
{summary_capped_tbl_Sm}

<h2>Summary — Plain (1.0) — $5M</h2>
{summary_plain_tbl_Sm}

<h2>Weekday Breakdown — $3M Threshold</h2>
<h3>Multiplier (Uncapped)</h3>
{weekday_mult_tbl}
<h3>Multiplier + 50% Notional Cap</h3>
{weekday_capped_tbl}
<h3>Plain (1.0)</h3>
{weekday_plain_tbl}

<h2>Weekday Breakdown — $5M Threshold</h2>
<h3>Multiplier (Uncapped)</h3>
{weekday_mult_tbl_Sm}
<h3>Multiplier + 50% Notional Cap</h3>
{weekday_capped_tbl_Sm}
<h3>Plain (1.0)</h3>
{weekday_plain_tbl_Sm}

</body>
</html>"""

# -- Save --
out_daily = os.path.join(OUTPUT_DIR, "spy_hedge_ratio_20260607_5min_mem3M_daily.parquet")
out_summary = os.path.join(OUTPUT_DIR, "spy_hedge_ratio_20260607_5min_mem3M_summary.csv")
out_html = os.path.join(OUTPUT_DIR, "spy_hedge_ratio_20260607_5min_mem3M_report.html")
out_recent = os.path.join(OUTPUT_DIR, f"spy_hedge_ratio_20260607_5min_mem3M_recent{N_RECENT_DAYS}.csv")

strip.write_parquet(out_daily)
summary.write_csv(out_summary)
recent.write_csv(out_recent)

with open(out_html, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"\nSaved:\n  {out_daily}\n  {out_summary}\n  {out_html}")
