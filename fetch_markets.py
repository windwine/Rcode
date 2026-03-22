"""
Fetch all finance-related Kalshi markets and save to CSV.

Output:
  data/markets.csv  — one row per market (ticker, prices, volume, status, ...)
  data/events.csv   — one row per event (ticker, category, title, rules, market_count)

Usage:
  python src_EN/fetch_markets.py
"""

import sys
import time
import pandas as pd
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kalshi_client import KalshiClient, REQUEST_DELAY


# ── Finance category filter ──────────────────────────────────────────────────
FINANCE_CATEGORIES = {
    "Economics",   # GDP, inflation, interest rates
    "Financials",  # stock indices, IPOs, financial firms
    "Companies",   # earnings, M&A
    "Crypto",      # BTC, ETH, altcoins
}

FINANCE_KEYWORDS = [
    "gdp", "inflation", "cpi", "pce", "fomc", "interest rate",
    "federal reserve", "s&p 500", "nasdaq", "dow jones",
    "stock market", "bitcoin", "ethereum", "cryptocurrency",
    "oil price", "gold price", "unemployment rate", "nonfarm payroll",
    "rate hike", "rate cut", "treasury yield",
    "mortgage rate", "trade deficit", "trade surplus", "tariff",
    "ipo valuation", "earnings per share",
]

OUTPUT_DIR = Path(__file__).parent.parent / "data"
MARKETS_FILE = OUTPUT_DIR / "markets.csv"
EVENTS_FILE = OUTPUT_DIR / "events.csv"


# ── Helpers ──────────────────────────────────────────────────────────────────
def is_finance_related(event: dict) -> bool:
    """Return True if the event belongs to a finance category or matches keywords."""
    category = event.get("category") or ""
    if category in FINANCE_CATEGORIES:
        return True
    title = (event.get("title") or "").lower()
    subtitle = (event.get("sub_title") or "").lower()
    combined = f"{title} {subtitle}"
    return any(kw in combined for kw in FINANCE_KEYWORDS)


def extract_market_row(event: dict, mkt: dict) -> dict:
    """Build a flat dict from event + market data."""
    return {
        "event_ticker":  event.get("event_ticker", ""),
        "event_title":   event.get("title", ""),
        "category":      event.get("category", ""),
        "market_ticker": mkt.get("ticker", ""),
        "market_title":  mkt.get("title", ""),
        "subtitle":      mkt.get("subtitle", ""),
        "yes_ask":       mkt.get("yes_ask"),
        "yes_bid":       mkt.get("yes_bid"),
        "no_ask":        mkt.get("no_ask"),
        "no_bid":        mkt.get("no_bid"),
        "last_price":    mkt.get("last_price"),
        "volume":        mkt.get("volume"),
        "volume_24h":    mkt.get("volume_24h"),
        "open_interest": mkt.get("open_interest"),
        "liquidity":     mkt.get("liquidity"),
        "status":        mkt.get("status", ""),
        "close_time":    mkt.get("close_time", ""),
        "expiration_time": mkt.get("expiration_time", ""),
        "result":        mkt.get("result", ""),
        # kept for events.csv generation, dropped from markets.csv output
        "_rules_primary":   mkt.get("rules_primary", ""),
        "_rules_secondary": mkt.get("rules_secondary", ""),
    }


def build_events_df(df_markets: pd.DataFrame, finance_events: list) -> pd.DataFrame:
    """
    Build events.csv from the markets DataFrame + raw event list.
    Each event gets its first market's rules_primary/secondary as description.
    """
    event_meta = {
        ev.get("event_ticker", ""): {
            "series_ticker":     ev.get("series_ticker", ""),
            "sub_title":         ev.get("sub_title", ""),
            "mutually_exclusive": ev.get("mutually_exclusive", False),
        }
        for ev in finance_events
    }

    first_rules = (
        df_markets[df_markets["_rules_primary"].notna() & (df_markets["_rules_primary"] != "")]
        .drop_duplicates(subset="event_ticker", keep="first")
        [["event_ticker", "_rules_primary", "_rules_secondary"]]
    )

    mkt_count = df_markets.groupby("event_ticker")["market_ticker"].count().rename("market_count")

    rows = []
    for ev in finance_events:
        ticker = ev.get("event_ticker", "")
        meta = event_meta.get(ticker, {})
        rows.append({
            "event_ticker":      ticker,
            "series_ticker":     meta.get("series_ticker", ""),
            "category":          ev.get("category", ""),
            "title":             ev.get("title", ""),
            "sub_title":         meta.get("sub_title", ""),
            "mutually_exclusive": meta.get("mutually_exclusive", False),
        })

    df_ev = pd.DataFrame(rows)
    df_ev = df_ev.merge(mkt_count, on="event_ticker", how="left")
    df_ev = df_ev.merge(first_rules, on="event_ticker", how="left")
    df_ev = df_ev.rename(columns={
        "_rules_primary":   "rules_primary",
        "_rules_secondary": "rules_secondary",
    })

    cols = ["event_ticker", "series_ticker", "category", "title", "sub_title",
            "mutually_exclusive", "market_count", "rules_primary", "rules_secondary"]
    return df_ev[[c for c in cols if c in df_ev.columns]]


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print(f"Total markets : {len(df)}")
    print(f"Total events  : {df['event_ticker'].nunique()}")

    print("\nBy category:")
    cat_counts = df.groupby("category").agg(
        events=("event_ticker", "nunique"),
        markets=("market_ticker", "count"),
    ).sort_values("markets", ascending=False)
    try:
        from tabulate import tabulate
        print(tabulate(cat_counts, headers="keys", tablefmt="rounded_outline"))
    except ImportError:
        print(cat_counts.to_string())

    priced = df[df["yes_ask"].notna() & (df["yes_ask"] > 0)].copy()
    print(f"\nMarkets with valid price: {len(priced)}")

    if not priced.empty:
        print("\nTop 15 markets by volume:")
        top = priced.nlargest(15, "volume")[
            ["event_title", "market_title", "category", "yes_ask", "yes_bid", "volume", "open_interest"]
        ]
        try:
            from tabulate import tabulate
            print(tabulate(top, headers="keys", tablefmt="rounded_outline",
                           showindex=False, maxcolwidths=30, floatfmt=".0f"))
        except ImportError:
            print(top.to_string(index=False))

    print("=" * 65)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Kalshi Finance Market Fetcher")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    client = KalshiClient()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: fetch all open events (paginated)
    print("\n[Step 1] Fetching all open events...")
    all_events = client.get_all_events(status="open")
    print(f"Total events fetched: {len(all_events)}")

    # Step 2: filter finance-related
    finance_events = [ev for ev in all_events if is_finance_related(ev)]
    print(f"\n[Step 2] Finance-related events: {len(finance_events)}")
    cat_dist = Counter(ev.get("category", "") for ev in finance_events)
    for cat, cnt in cat_dist.most_common():
        print(f"  {cat:<25} {cnt}")

    # Step 3: fetch markets for each event
    print(f"\n[Step 3] Fetching markets for {len(finance_events)} events...")
    all_rows = []
    failed = []

    for i, event in enumerate(finance_events, 1):
        ticker = event.get("event_ticker", "")
        if not ticker:
            continue
        try:
            markets = client.get_markets_for_event(ticker)
            for mkt in markets:
                all_rows.append(extract_market_row(event, mkt))
            if i % 20 == 0 or i == len(finance_events):
                print(f"  progress: {i}/{len(finance_events)} events, {len(all_rows)} markets so far")
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            failed.append(ticker)
            print(f"  [skip] {ticker}: {e}")

    if failed:
        print(f"  [warning] {len(failed)} events failed: {failed[:5]}")

    if not all_rows:
        print("[error] No market data retrieved.")
        sys.exit(1)

    # Step 4: save CSVs
    df = pd.DataFrame(all_rows)
    for col in ["yes_ask", "yes_bid", "no_ask", "no_bid", "last_price",
                "volume", "volume_24h", "open_interest", "liquidity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df_out = df.drop(columns=["_rules_primary", "_rules_secondary"], errors="ignore")
    df_out.to_csv(MARKETS_FILE, index=False, encoding="utf-8-sig")
    print(f"\n[Done] markets.csv saved: {MARKETS_FILE}  ({MARKETS_FILE.stat().st_size / 1024:.1f} KB)")

    df_events = build_events_df(df, finance_events)
    df_events.to_csv(EVENTS_FILE, index=False, encoding="utf-8-sig")
    print(f"[Done] events.csv saved:  {EVENTS_FILE}  ({len(df_events)} events)")

    print_summary(df_out)


if __name__ == "__main__":
    main()
