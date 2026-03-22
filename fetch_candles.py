"""
Fetch candlestick data for a list of Kalshi market tickers.

Supports three timeframes:
  daily   — 1440-min bars, full history since START_DATE
  hourly  — 60-min bars, rolling lookback (default 180 days)
  1min    — 1-min bars, rolling lookback (default 3 days)

Active markets use the batch endpoint (/markets/candlesticks).
Settled markets use the historical endpoint (/historical/markets/{ticker}/candlesticks).

Incremental update: if a CSV already exists, only fetches data newer than the
last recorded timestamp.

Usage:
  # Fetch all three timeframes for a specific list of tickers:
  python src_EN/fetch_candles.py

  # Or import and call fetch_tickers() from another script.
"""

import sys
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kalshi_client import KalshiClient


# ── Configuration ────────────────────────────────────────────────────────────

# Edit this list to fetch candles for specific tickers.
# Leave empty to read all tickers from data/markets.csv instead.
TICKERS = [
    # example:
    # "KXUSAIRANAGREEMENT-27-26APR",
    # "KXUSAIRANAGREEMENT-27-26AUG",
    # "KXUSAIRANAGREEMENT-27",
]

# If TICKERS is empty, candles are fetched for all markets in this CSV.
MARKETS_CSV = Path(__file__).parent.parent / "data" / "markets.csv"

# Output root — subfolders: daily / hourly / 1min
DATA_ROOT = Path(__file__).parent.parent / "data" / "candles"

# Timeframe definitions: (period_interval_minutes, subfolder, max_lookback_days)
TIMEFRAMES = [
    (1440, "daily",  365 * 3),  # daily: up to 3 years of history
    (60,   "hourly", 180),      # hourly: last 180 days
    (1,    "1min",   3),        # 1-min: last 3 days
]

REQUEST_DELAY = 0.8  # seconds between API calls


# ── EST/EDT helper ────────────────────────────────────────────────────────────
def to_eastern(ts: int) -> str:
    """Convert a UTC Unix timestamp to a human-readable Eastern time string."""
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    year = dt_utc.year
    mar1 = datetime(year, 3, 1, tzinfo=timezone.utc)
    dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    dst_start = dst_start.replace(hour=7)   # 2:00 EST = 7:00 UTC
    nov1 = datetime(year, 11, 1, tzinfo=timezone.utc)
    dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    dst_end = dst_end.replace(hour=6)       # 2:00 EDT = 6:00 UTC
    is_edt = dst_start <= dt_utc < dst_end
    offset = timedelta(hours=-4 if is_edt else -5)
    suffix = "EDT" if is_edt else "EST"
    return (dt_utc + offset).strftime("%Y-%m-%d %H:%M:%S") + f" {suffix}"


# ── Core helpers ──────────────────────────────────────────────────────────────
def candles_to_df(ticker: str, candles: list) -> pd.DataFrame:
    """Convert a list of raw candle dicts to a DataFrame."""
    rows = []
    for c in candles:
        p = c.get("price", {})
        rows.append({
            "ticker":        ticker,
            "end_period_ts": c.get("end_period_ts"),
            "datetime_utc":  datetime.fromtimestamp(
                                 c["end_period_ts"], tz=timezone.utc
                             ).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "datetime_est":  to_eastern(c["end_period_ts"]),
            "open":          p.get("open_dollars"),
            "high":          p.get("high_dollars"),
            "low":           p.get("low_dollars"),
            "close":         p.get("close_dollars"),
            "mean":          p.get("mean_dollars"),
            "volume":        c.get("volume"),
            "open_interest": c.get("open_interest"),
        })
    return pd.DataFrame(rows)


def get_last_ts(csv_path: Path) -> int | None:
    """Return the latest end_period_ts in an existing CSV, or None."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if df.empty or "end_period_ts" not in df.columns:
            return None
        return int(df["end_period_ts"].max())
    except Exception:
        return None


def merge_and_save(csv_path: Path, new_df: pd.DataFrame) -> int:
    """Merge new rows with existing CSV, deduplicate, sort, and save."""
    if csv_path.exists():
        old_df = pd.read_csv(csv_path, encoding="utf-8-sig")
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = (
        combined
        .drop_duplicates(subset=["end_period_ts"])
        .sort_values("end_period_ts")
        .reset_index(drop=True)
    )
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return len(combined)


# ── Fetch logic ───────────────────────────────────────────────────────────────
def fetch_timeframe(client: KalshiClient, tickers: list, active_set: set,
                    period: int, out_dir: Path, max_lookback_days: int,
                    now_ts: int):
    """
    Fetch candles for all tickers for a single timeframe.
    - Active tickers: batch endpoint
    - Settled tickers: historical endpoint (one by one)
    - Incremental: skips data already on disk
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    label = {1440: "daily", 60: "hourly", 1: "1-min"}[period]
    earliest_start = now_ts - max_lookback_days * 86400

    active_tickers   = [t for t in tickers if t in active_set]
    settled_tickers  = [t for t in tickers if t not in active_set]

    saved = empty = 0

    # ── Active markets: batch API ─────────────────────────────────────────────
    if active_tickers:
        print(f"  Active ({len(active_tickers)} tickers) — batch endpoint")

        # Dynamic batch size: stay under 10,000 candles per request
        # For daily: days_range candles per ticker
        # For hourly: use chunks of 90 days × 4 tickers = 8640
        if period == 1440:
            days_range = max(1, (now_ts - earliest_start) // 86400)
            batch_size = max(1, 9000 // days_range)
            segments = [(earliest_start, now_ts)]
        elif period == 60:
            chunk_days = 90
            batch_size = max(1, 9000 // (chunk_days * 24))  # = 4
            num_chunks = max_lookback_days // chunk_days
            segments = [
                (now_ts - (i + 1) * chunk_days * 86400,
                 now_ts - i * chunk_days * 86400)
                for i in range(num_chunks)
            ]
        else:
            # 1-min: single window, batch of 100
            batch_size = 100
            segments = [(earliest_start, now_ts)]

        # Accumulate across segments
        accumulated: dict[str, list] = {t: [] for t in active_tickers}

        for seg_start, seg_end in segments:
            seg_label = (
                f"{datetime.fromtimestamp(seg_start, tz=timezone.utc).strftime('%m/%d')}"
                f"~{datetime.fromtimestamp(seg_end, tz=timezone.utc).strftime('%m/%d')}"
            )
            for i in range(0, len(active_tickers), batch_size):
                batch = active_tickers[i:i + batch_size]
                try:
                    result = client.get_candles_batch(batch, seg_start, seg_end, period)
                    for tk, candles in result.items():
                        accumulated[tk].extend(candles)
                except Exception as e:
                    print(f"    [batch error] {seg_label} tickers {batch[:2]}...: {e}")
                time.sleep(REQUEST_DELAY)

        # Deduplicate, incremental filter, and save
        for tk, candles in accumulated.items():
            if not candles:
                empty += 1
                continue
            csv_path = out_dir / f"{tk}.csv"
            last_ts = get_last_ts(csv_path)
            # Keep only candles newer than what's already saved
            if last_ts:
                candles = [c for c in candles if c["end_period_ts"] > last_ts]
            if not candles:
                print(f"    {tk}: already up to date")
                continue
            # Deduplicate within new batch
            seen: set = set()
            deduped = []
            for c in sorted(candles, key=lambda x: x["end_period_ts"]):
                if c["end_period_ts"] not in seen:
                    seen.add(c["end_period_ts"])
                    deduped.append(c)
            new_df = candles_to_df(tk, deduped)
            total = merge_and_save(csv_path, new_df)
            print(f"    {tk}: +{len(deduped)} new  →  {total} total  [{label}]")
            saved += 1

    # ── Settled markets: historical endpoint ──────────────────────────────────
    if settled_tickers:
        print(f"  Settled ({len(settled_tickers)} tickers) — historical endpoint")
        for i, tk in enumerate(settled_tickers, 1):
            csv_path = out_dir / f"{tk}.csv"
            last_ts = get_last_ts(csv_path)
            start_ts = (last_ts + 1) if last_ts and last_ts > earliest_start else earliest_start

            if start_ts >= now_ts:
                continue

            try:
                candles = client.get_candles_historical(tk, start_ts, now_ts, period)
                if candles:
                    new_df = candles_to_df(tk, candles)
                    total = merge_and_save(csv_path, new_df)
                    print(f"    {tk}: +{len(candles)} new  →  {total} total  [{label}]")
                    saved += 1
                else:
                    empty += 1
            except Exception as e:
                print(f"    [skip] {tk}: {e}")
            time.sleep(REQUEST_DELAY)

            if i % 50 == 0:
                print(f"    progress: {i}/{len(settled_tickers)}, saved so far: {saved}")

    print(f"  [{label}] saved: {saved}  no data: {empty}")


# ── Main ─────────────────────────────────────────────────────────────────────
def load_tickers_from_csv() -> tuple[list, set]:
    """
    Load tickers from data/markets.csv.
    Returns (all_tickers, active_set).
    """
    if not MARKETS_CSV.exists():
        print(f"[error] markets.csv not found: {MARKETS_CSV}")
        print("        Run fetch_markets.py first.")
        sys.exit(1)
    df = pd.read_csv(MARKETS_CSV, encoding="utf-8-sig")
    all_tickers = df["market_ticker"].dropna().tolist()
    active_set = set(df[df["status"] == "active"]["market_ticker"].tolist())
    print(f"Loaded {len(all_tickers)} tickers from markets.csv "
          f"({len(active_set)} active, {len(all_tickers) - len(active_set)} settled)")
    return all_tickers, active_set


def fetch_tickers(tickers: list, active_set: set):
    """Fetch all configured timeframes for the given tickers."""
    client = KalshiClient()
    now_ts = int(datetime.now(timezone.utc).timestamp())

    print(f"\nFetching {len(tickers)} tickers  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for period, subdir, max_days in TIMEFRAMES:
        label = {1440: "Daily (1440min)", 60: "Hourly (60min)", 1: "1-min (1min)"}[period]
        print(f"\n{'='*60}")
        print(f"[{label}]  lookback={max_days} days")
        print(f"{'='*60}")
        out_dir = DATA_ROOT / subdir
        fetch_timeframe(client, tickers, active_set, period, out_dir, max_days, now_ts)

    print(f"\nAll done. Output: {DATA_ROOT}/")
    for _, subdir, _ in TIMEFRAMES:
        p = DATA_ROOT / subdir
        n = len(list(p.glob("*.csv"))) if p.exists() else 0
        print(f"  {subdir}/   {n} CSV files")


def main():
    print("=" * 65)
    print("Kalshi Candlestick Fetcher")
    print("=" * 65)

    if TICKERS:
        # Hardcoded list: treat all as active (batch endpoint).
        # These tickers may not be in markets.csv (e.g. non-finance categories),
        # so we do not rely on markets.csv for active_set here.
        tickers = TICKERS
        active_set = set(TICKERS)
        print(f"Using hardcoded ticker list: {tickers}")
    else:
        tickers, active_set = load_tickers_from_csv()

    fetch_tickers(tickers, active_set)


if __name__ == "__main__":
    main()
