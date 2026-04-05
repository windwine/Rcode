import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).parent.parent / "data" / "iran_nuclear" / "hourly"
DAILY_DATA_DIR = Path(__file__).parent.parent / "data" / "iran_nuclear" / "daily"
CHART_DIR = Path(__file__).parent.parent / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3.5:35b-a3b"
USE_OLLAMA_ANALYSIS = os.getenv("USE_OLLAMA_ANALYSIS", "1").strip().lower() not in {"0", "false", "no"}

_COLORS = ["#e74c3c", "#e67e22", "#2980b9", "#27ae60", "#8e44ad", "#16a085", "#d35400"]


def build_turning_point_candidates(df: pd.DataFrame) -> list[dict]:
    work = df[["datetime_utc", "close"]].copy().reset_index(drop=True)
    work["ret_1d"] = work["close"].pct_change()
    candidates = []

    for _, row in work.dropna(subset=["ret_1d"]).nlargest(3, "ret_1d").iterrows():
        candidates.append({
            "date": str(row["datetime_utc"])[:10],
            "type": "one_day_jump",
            "close": round(float(row["close"]), 3),
            "ret_1d_pct": round(float(row["ret_1d"] * 100), 1),
        })
    for _, row in work.dropna(subset=["ret_1d"]).nsmallest(3, "ret_1d").iterrows():
        candidates.append({
            "date": str(row["datetime_utc"])[:10],
            "type": "one_day_drop",
            "close": round(float(row["close"]), 3),
            "ret_1d_pct": round(float(row["ret_1d"] * 100), 1),
        })

    closes = work["close"].tolist()
    local_points = []
    for i in range(1, len(work) - 1):
        prev_c, cur_c, next_c = closes[i - 1], closes[i], closes[i + 1]
        prominence = abs(cur_c - prev_c) + abs(cur_c - next_c)
        if cur_c > prev_c and cur_c > next_c:
            local_points.append(("local_peak", i, prominence))
        elif cur_c < prev_c and cur_c < next_c:
            local_points.append(("local_trough", i, prominence))

    for point_type, idx, prominence in sorted(local_points, key=lambda x: x[2], reverse=True)[:4]:
        row = work.iloc[idx]
        candidates.append({
            "date": str(row["datetime_utc"])[:10],
            "type": point_type,
            "close": round(float(row["close"]), 3),
            "prominence": round(float(prominence), 3),
        })

    dedup = {}
    for item in candidates:
        key = (item["date"], item["type"])
        if key not in dedup:
            dedup[key] = item
    return list(dedup.values())


def build_recent_trend_context(df: pd.DataFrame, lookback_days: int = 60) -> dict:
    df = df.sort_values("end_period_ts").reset_index(drop=True)
    last_ts = pd.to_datetime(df["datetime_utc"].iloc[-1], utc=True)
    cutoff = last_ts - pd.Timedelta(days=lookback_days)
    dt = pd.to_datetime(df["datetime_utc"], utc=True)

    window = df.loc[dt >= cutoff].copy().reset_index(drop=True)
    if len(window) < 3:
        window = df.copy().reset_index(drop=True)

    peak_idx = int(window["close"].idxmax())
    peak_row = window.loc[peak_idx]
    peak_price = float(peak_row["close"])

    trough_idx = int(window["close"].idxmin())
    trough_row = window.loc[trough_idx]
    trough_price = float(trough_row["close"])

    after_peak = window.loc[peak_idx:].copy()
    post_peak_low_row = after_peak.loc[after_peak["close"].idxmin()]
    current_price = float(window["close"].iloc[-1])

    return {
        "window_start": str(window["datetime_utc"].iloc[0])[:10],
        "window_days": lookback_days,
        "peak_date": str(peak_row["datetime_utc"])[:10],
        "peak_price": round(peak_price, 3),
        "trough_date": str(trough_row["datetime_utc"])[:10],
        "trough_price": round(trough_price, 3),
        "peak_to_trough_pct": round((trough_price / peak_price - 1) * 100, 1) if peak_price else 0.0,
        "post_peak_low_date": str(post_peak_low_row["datetime_utc"])[:10],
        "post_peak_low_price": round(float(post_peak_low_row["close"]), 3),
        "current_price": round(current_price, 3),
        "change_from_peak_pct": round((current_price / peak_price - 1) * 100, 1) if peak_price else 0.0,
        "change_from_trough_pct": round((current_price / trough_price - 1) * 100, 1) if trough_price else 0.0,
    }


def build_stats(ticker: str) -> dict:
    path = DAILY_DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["close"]).sort_values("end_period_ts").reset_index(drop=True)

    cur = float(df["close"].iloc[-1])
    first = float(df["close"].iloc[0])

    recent30 = df.tail(30)[["datetime_utc", "close"]].copy()
    recent30["datetime_utc"] = recent30["datetime_utc"].astype(str).str[:10]

    chg7_pct = (cur - float(df["close"].iloc[-7])) / float(df["close"].iloc[-7]) * 100 if len(df) >= 7 else 0
    chg_all_pct = (cur - first) / first * 100 if first > 0 else 0

    return {
        "ticker": ticker,
        "first_date": str(df["datetime_utc"].iloc[0])[:10],
        "last_date": str(df["datetime_utc"].iloc[-1])[:10],
        "first_price": round(first, 3),
        "current_price": round(cur, 3),
        "all_time_high": round(float(df["high"].max()), 3),
        "all_time_low": round(float(df["low"].min()), 3),
        "chg_7d_pct": round(chg7_pct, 1),
        "chg_all_pct": round(chg_all_pct, 1),
        "recent_30d_daily": recent30.to_dict(orient="records"),
        "turning_point_candidates": build_turning_point_candidates(df),
        "recent_trend": build_recent_trend_context(df),
    }


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "/no_think\n\n" + prompt}],
        "stream": False,
        "options": {"temperature": 0.4, "think": False},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        return strip_think_tags(resp.json()["message"]["content"].strip())
    except requests.exceptions.ConnectionError:
        print("  [Warning] Cannot connect to Ollama on http://localhost:11434")
        return ""
    except Exception as exc:
        print(f"  [Warning] Ollama request failed: {exc}")
        return ""


def parse_json_response(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found:\n{raw}")
    return json.loads(raw[start:end])


def coerce_text_response(raw: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    match = re.search(r'"en"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(f'"{match.group(1)}"')
        except Exception:
            pass
    return text.strip()


def strip_volume_commentary(text: str) -> str:
    blocked = ("volume", "liquidity", "open interest", "open-interest", "trading activity", "turnover")
    kept = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if not any(t in s.lower() for t in blocked)]
    return " ".join(kept).strip() or text.strip()


def build_fallback_description(stats: dict, market_title: str) -> str:
    trend = stats["recent_trend"]
    top_candidates = stats["turning_point_candidates"][:2]
    turning_line = (
        "; ".join(f"{c['date']} ({c['type']}, close {c['close']:.2f})" for c in top_candidates)
        if top_candidates else "no strong turning points detected"
    )
    return (
        f"Current price is <strong>{stats['current_price']:.2f}</strong> "
        f"(~<strong>{stats['current_price'] * 100:.0f}%</strong> implied probability) "
        f"for <strong>{market_title}</strong>. "
        f"Since listing, price moved from <strong>{stats['first_price']:.2f}</strong> "
        f"to <strong>{stats['current_price']:.2f}</strong> "
        f"(<strong>{stats['chg_all_pct']:+.1f}%</strong>), "
        f"with a 7-day change of <strong>{stats['chg_7d_pct']:+.1f}%</strong>. "
        f"Over the last {trend['window_days']} days the contract peaked at "
        f"<strong>{trend['peak_price']:.2f}</strong> on <strong>{trend['peak_date']}</strong> "
        f"and troughed at <strong>{trend['trough_price']:.2f}</strong> on "
        f"<strong>{trend['trough_date']}</strong> "
        f"(peak-to-trough: <strong>{trend['peak_to_trough_pct']:.1f}%</strong>); "
        f"currently <strong>{trend['change_from_peak_pct']:.1f}%</strong> from the peak "
        f"and <strong>{trend['change_from_trough_pct']:+.1f}%</strong> from the trough. "
        f"Potential turning points: <strong>{turning_line}</strong>. "
        f"Range: <strong>{stats['all_time_low']:.2f}</strong> – "
        f"<strong>{stats['all_time_high']:.2f}</strong>."
    )


def generate_description_ollama(ticker: str, market_title: str) -> dict:
    stats = build_stats(ticker)

    if not USE_OLLAMA_ANALYSIS:
        return {"en": build_fallback_description(stats, market_title)}

    trend = stats["recent_trend"]
    prompt = f"""You are a financial analyst specializing in prediction markets.

Below is data for a Kalshi prediction market contract:
- Contract: {market_title}  (ticker: {ticker})
- Listing date: {stats['first_date']}  |  Latest data: {stats['last_date']}
- First price: {stats['first_price']}  |  Current price: {stats['current_price']} (~{stats['current_price']*100:.0f}% implied probability)
- All-time high: {stats['all_time_high']}  |  All-time low: {stats['all_time_low']}
- 7-day change: {stats['chg_7d_pct']:+.1f}%  |  Change since listing: {stats['chg_all_pct']:+.1f}%
- Recent {trend['window_days']}-day trend context: {json.dumps(trend)}
- Recent 30-day daily close prices: {json.dumps(stats['recent_30d_daily'])}
- Candidate turning points extracted from price action: {json.dumps(stats['turning_point_candidates'])}

Write a concise market analysis in English (3-4 sentences) covering:
1. The current probability level and what it implies
2. Notable price trends and key turning points in the recent window (include exact YYYY-MM-DD dates)
3. What market movement suggests about expectations for this contract

Requirements:
- Base all date references on the data provided — do not assume any fixed calendar event.
- Explicitly name when the most recent significant peak and trough occurred.
- Explicitly name 2-4 important turning-point dates with context (reversal, acceleration, breakout).
- Do not discuss trading volume or liquidity.

Respond ONLY with valid JSON in this exact format (no extra text outside JSON):
{{
  "en": "English analysis here (3-4 sentences, use <strong> tags for key numbers and dates)"
}}"""

    print(f"  [{ticker}] Calling {MODEL_NAME}...", flush=True)
    raw = call_ollama(prompt)
    if not raw:
        print("  [Warning] Empty response; using fallback")
        return {"en": build_fallback_description(stats, market_title)}

    try:
        result = parse_json_response(raw)
        if "en" not in result:
            raise ValueError("missing 'en' field")
        result["en"] = strip_volume_commentary(result["en"])
        return result
    except Exception as exc:
        coerced = coerce_text_response(raw)
        if coerced:
            print(f"  [Warning] JSON parse failed ({exc}); using raw text")
            return {"en": strip_volume_commentary(coerced)}
        print(f"  [Warning] JSON parse failed ({exc}); using fallback")
        return {"en": build_fallback_description(stats, market_title)}


def load_hourly(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["datetime_utc"])
    df = df.dropna(subset=["open", "high", "low", "close"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    return df.sort_values("end_period_ts").reset_index(drop=True)


def make_chart(ticker: str, title: str, color: str) -> Path:
    df = load_hourly(ticker)
    print(
        f"  {ticker}: {len(df)} hourly candles  "
        f"{df['datetime_utc'].iloc[0].strftime('%Y-%m-%d')} ~ "
        f"{df['datetime_utc'].iloc[-1].strftime('%Y-%m-%d')}"
    )

    df["ma24"] = df["close"].rolling(24).mean()
    df["ma168"] = df["close"].rolling(168).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=df["datetime_utc"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350", name="OHLC",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["datetime_utc"], y=df["ma24"],
        line=dict(color="rgba(255,193,7,0.9)", width=1.2), name="MA24h", hovertemplate="%{y:.3f}",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["datetime_utc"], y=df["ma168"],
        line=dict(color="rgba(156,39,176,0.9)", width=1.5, dash="dot"), name="MA7d", hovertemplate="%{y:.3f}",
    ), row=1, col=1)

    last_close = df["close"].iloc[-1]
    fig.add_hline(
        y=last_close, row=1, col=1,
        line=dict(color=color, width=1, dash="dash"),
        annotation_text=f"Last {last_close:.2f}", annotation_position="right", annotation_font_color=color,
    )

    last_dt = df["datetime_utc"].iloc[-1]
    fig.add_vline(x=last_dt, line=dict(color="rgba(255,255,255,0.35)", width=1.5, dash="dot"), row=1, col=1)
    fig.add_annotation(
        x=last_dt, y=1, yref="paper",
        text=f"Data through<br>{last_dt.strftime('%Y-%m-%d %H:%M UTC')}",
        showarrow=False, xanchor="right",
        font=dict(size=10, color="rgba(200,200,200,0.75)"), bgcolor="rgba(0,0,0,0.4)", borderpad=4,
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=1.08,
        text=MODEL_NAME, showarrow=False, xanchor="left",
        font=dict(size=11, color="rgba(100,220,100,0.85)"), bgcolor="rgba(0,0,0,0.35)", borderpad=4,
    )

    vol_colors = ["#26a69a" if df["close"].iloc[i] >= df["open"].iloc[i] else "#ef5350" for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df["datetime_utc"], y=df["volume"], marker_color=vol_colors, opacity=0.6, name="Volume",
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template="plotly_dark", xaxis_rangeslider_visible=False, height=680,
        legend=dict(orientation="h", y=1.02, x=0), margin=dict(l=60, r=80, t=100, b=40),
        hovermode="x unified",
        xaxis2=dict(rangeselector=dict(buttons=[
            dict(count=3,  label="3d",  step="day", stepmode="backward"),
            dict(count=7,  label="7d",  step="day", stepmode="backward"),
            dict(count=30, label="1M",  step="day", stepmode="backward"),
            dict(count=90, label="3M",  step="day", stepmode="backward"),
            dict(step="all", label="All"),
        ])),
    )
    fig.update_yaxes(title_text="Probability ($)", tickformat=".2f", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    out_path = CHART_DIR / f"{ticker}_ollama_en.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  -> {out_path}")
    return out_path


def resolve_markets(cli_tickers: list[str]) -> list[tuple[str, str, str]]:
    if cli_tickers:
        tickers = cli_tickers
    else:
        tickers = sorted(p.stem for p in DAILY_DATA_DIR.glob("*.csv"))
        if not tickers:
            print(f"No CSVs found in {DAILY_DATA_DIR}.")
            sys.exit(1)
    return [(ticker, ticker, _COLORS[i % len(_COLORS)]) for i, ticker in enumerate(tickers)]


def make_comparison_chart(markets: list[tuple[str, str, str]], contract_descriptions: list[dict]) -> Path:
    fig_cmp = go.Figure()
    last_dts, cur_prices = [], []

    for ticker, _, _ in markets:
        df = load_hourly(ticker)
        last_dts.append(df["datetime_utc"].iloc[-1])
        cur_prices.append(float(df["close"].iloc[-1]))

    for i, (ticker, _, _) in enumerate(markets):
        df = load_hourly(ticker)
        fig_cmp.add_trace(go.Scatter(
            x=df["datetime_utc"], y=df["close"], mode="lines",
            name=f"{ticker} ({cur_prices[i] * 100:.0f}c)",
            line=dict(color=_COLORS[i % len(_COLORS)], width=2), hovertemplate="%{y:.3f}",
        ))

    cmp_last_dt = max(last_dts)
    fig_cmp.add_vline(x=cmp_last_dt, line=dict(color="rgba(255,255,255,0.35)", width=1.5, dash="dot"))
    fig_cmp.add_annotation(
        x=cmp_last_dt, y=1, yref="paper",
        text=f"Data through<br>{cmp_last_dt.strftime('%Y-%m-%d %H:%M UTC')}",
        showarrow=False, xanchor="right",
        font=dict(size=10, color="rgba(200,200,200,0.75)"), bgcolor="rgba(0,0,0,0.4)", borderpad=4,
    )
    fig_cmp.add_annotation(
        xref="paper", yref="paper", x=0, y=1.12,
        text=MODEL_NAME, showarrow=False, xanchor="left",
        font=dict(size=11, color="rgba(100,220,100,0.85)"), bgcolor="rgba(0,0,0,0.35)", borderpad=4,
    )
    fig_cmp.update_layout(
        title=f"Market Probability Comparison — {len(markets)} contracts (Hourly)",
        template="plotly_dark", height=480,
        yaxis=dict(title="Probability ($)", tickformat=".2f"),
        hovermode="x unified", legend=dict(orientation="h", y=1.02), margin=dict(t=100),
    )

    cards_html = ""
    for i, desc in enumerate(contract_descriptions):
        color = _COLORS[i % len(_COLORS)]
        cards_html += f"""
  <div class="contract-card" style="border-left-color:{color}">
    <div class="contract-header" style="color:{color}">{desc['label']}</div>
    <p class="desc-text">{desc['en']}</p>
  </div>"""

    full_desc_html = f"""
<style>
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 0; }}
  .desc-container {{ max-width: 960px; margin: 0 auto; padding: 24px 32px 48px; }}
  .model-tag {{ display: inline-block; background: rgba(0,180,0,0.15); border: 1px solid rgba(0,220,0,0.3);
               color: #64dc64; font-size: 13px; padding: 6px 14px; border-radius: 20px; margin-bottom: 24px; }}
  .desc-title {{ font-size: 20px; font-weight: 700; color: #ffffff;
                 border-bottom: 2px solid #444; padding-bottom: 10px; margin: 8px 0 20px; }}
  .contract-card {{ border-left: 4px solid; background: #16213e;
                    border-radius: 6px; padding: 18px 22px; margin-bottom: 20px; }}
  .contract-header {{ font-size: 15px; font-weight: 700; margin-bottom: 10px; }}
  .desc-text {{ font-size: 14px; line-height: 1.75; color: #cccccc; margin: 0 0 4px; }}
</style>
<div class="desc-container">
  <div class="model-tag">{MODEL_NAME}</div>
  <div class="desc-title">Contract Descriptions</div>
  {cards_html}
</div>"""

    first_ticker = markets[0][0] if markets else "markets"
    cmp_path = CHART_DIR / f"{first_ticker}_comparison_ollama_en.html"
    chart_html = fig_cmp.to_html(include_plotlyjs="cdn", full_html=False)
    with open(cmp_path, "w", encoding="utf-8") as fh:
        fh.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Market Analysis · {MODEL_NAME}</title>
</head>
<body style="background:#1a1a2e; margin:0; padding:20px;">
  <div style="max-width:960px; margin:0 auto;">
    {chart_html}
    {full_desc_html}
  </div>
</body>
</html>""")
    print(f"  -> {cmp_path}")
    return cmp_path


def main() -> None:
    markets = resolve_markets(sys.argv[1:])

    print("=" * 60)
    print(f"Generating charts ({MODEL_NAME})")
    print(f"Markets: {[t for t, _, _ in markets]}")
    print("=" * 60)

    paths = [make_chart(ticker, title, color) for ticker, title, color in markets]

    print(f"\nRunning analysis ({len(markets)} contracts)...")
    contract_descriptions = []
    for ticker, title, _ in markets:
        desc = generate_description_ollama(ticker, title)
        contract_descriptions.append({"ticker": ticker, "label": title, **desc})

    cmp_path = None
    if len(markets) >= 2:
        print("\nGenerating comparison chart...")
        cmp_path = make_comparison_chart(markets, contract_descriptions)

    print(f"\nDone! {len(paths) + (1 if cmp_path else 0)} chart(s):")
    for p in paths:
        print(f"  {p.name}")
    if cmp_path:
        print(f"  {cmp_path.name}")


if __name__ == "__main__":
    main()
