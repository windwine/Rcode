"""
Turnaround Tuesday — 跨 ETF 版本

信号来源：SPY（每周首个交易日的 IBS / 涨跌幅等指标）
交易标的：SPY / IWM / QQQ / DIA（各自用自己的 adj_close 计算收益）

动态止盈（策略4）的离场条件也基于 SPY（收盘>SPY昨日最高价），
保持信号与 SPY 的绑定关系，仅换标的。
"""

import sys
import io
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import polars as pl
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from shared.data_loader import load_tiingo
from shared.perf import pnl_report_polars_plotnine

# ── 参数 ──
START      = "2002-01-01"
END        = "2026-03-13"
SIGNAL_TKR = "SPY"                         # 信号来源
TRADE_TKRS = ["SPY", "IWM", "QQQ", "DIA"]  # 交易标的
FEE        = 0.0

OUT_HTML = Path(__file__).parent / "report_crossetf.html"

STRAT_KEYS   = ["s1", "s2", "s3", "s4", "rev"]
STRAT_LABELS = {
    "s1":  "策略1 · 首日跌1%买入",
    "s2":  "策略2 · 首日IBS<0.2",
    "s3":  "策略3 · 持有至周末",
    "s4":  "策略4 · 动态止盈/周末",
    "rev": "反向 · 首日追涨",
}

TICKER_COLORS = {
    "SPY": "#1f77b4",
    "IWM": "#e8600a",
    "QQQ": "#2ca02c",
    "DIA": "#9467bd",
}

STRAT_COLORS = {
    "s1":  "#1f77b4",
    "s2":  "#ff7f0e",
    "s3":  "#2ca02c",
    "s4":  "#c0001a",
    "rev": "#9467bd",
}

MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ── 数据加载 ──

def load_spy_signals() -> pl.DataFrame:
    """加载 SPY 并计算信号所需的字段及首/末交易日标记。"""
    df = load_tiingo(SIGNAL_TKR, START, END)
    df = (
        df.sort("date")
        .with_columns([
            pl.when(pl.col("high") > pl.col("low"))
              .then((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low")))
              .otherwise(0.5)
              .alias("ibs"),
            pl.col("adj_close").shift(1).alias("prev_adj_close"),
            pl.col("high").shift(1).alias("prev_high"),
            pl.col("date").dt.iso_year().alias("iso_year"),
            pl.col("date").dt.week().alias("iso_week"),
        ])
        .drop_nulls(subset=["prev_adj_close"])
    )
    df = df.with_columns([
        pl.col("iso_year").shift(1).alias("_py"),
        pl.col("iso_week").shift(1).alias("_pw"),
        pl.col("iso_year").shift(-1).alias("_ny"),
        pl.col("iso_week").shift(-1).alias("_nw"),
    ]).with_columns([
        (
            pl.col("_py").is_null() |
            (pl.col("iso_year") != pl.col("_py")) |
            (pl.col("iso_week") != pl.col("_pw"))
        ).alias("is_first_trade_day"),
        (
            pl.col("_ny").is_null() |
            (pl.col("iso_year") != pl.col("_ny")) |
            (pl.col("iso_week") != pl.col("_nw"))
        ).alias("is_last_trade_day"),
    ]).drop(["_py","_pw","_ny","_nw"])
    return df


def load_target(ticker: str, dates_spy: np.ndarray) -> np.ndarray:
    """
    加载交易标的数据，按 SPY 日期对齐后返回 adj_close 数组。
    两个市场同步开市，date 应完全匹配；若缺失则用前向填充。
    """
    raw = load_tiingo(ticker, START, END).sort("date")
    spy_dates_pl = pl.Series("date", dates_spy)

    aligned = (
        pl.DataFrame({"date": spy_dates_pl})
        .join(raw.select(["date", "adj_close"]), on="date", how="left")
        .with_columns(pl.col("adj_close").forward_fill())
    )
    return aligned["adj_close"].to_numpy()


# ── 回测引擎（信号来自 SPY，收益来自 target） ──

def backtest_next_day(
    spy_adj: np.ndarray,       # SPY adj_close（信号用）
    tgt_adj: np.ndarray,       # 交易标的 adj_close（收益用）
    dates: np.ndarray,
    entry_mask: np.ndarray,
    fee: float = FEE,
) -> tuple[pl.DataFrame, list[float]]:
    """次日平仓（策略1/2/反向）。"""
    n = len(dates)
    daily_rets = np.zeros(n, dtype=float)
    trades: list[float] = []
    in_trade, entry_tgt = False, 0.0

    for i in range(1, n):
        if not in_trade:
            if entry_mask[i]:
                in_trade    = True
                entry_tgt   = tgt_adj[i]
                daily_rets[i] = -fee
        else:
            day_ret       = tgt_adj[i] / tgt_adj[i - 1] - 1.0
            daily_rets[i] = day_ret - fee
            trades.append(tgt_adj[i] / entry_tgt - 1.0 - 2.0 * fee)
            in_trade = False

    return pl.DataFrame({"Date": dates, "strategy_ret": daily_rets}), trades


def backtest_end_of_week(
    spy_adj: np.ndarray,
    spy_close: np.ndarray,    # SPY 收盘（动态止盈用）
    spy_high: np.ndarray,     # SPY 最高价（动态止盈用）
    tgt_adj: np.ndarray,
    dates: np.ndarray,
    entry_mask: np.ndarray,
    week_end_mask: np.ndarray,
    week_id: np.ndarray,
    dynamic_exit: bool = False,
    fee: float = FEE,
) -> tuple[pl.DataFrame, list[float]]:
    """持有至当周最后交易日（策略3/4），动态止盈条件基于 SPY。"""
    n = len(dates)
    daily_rets = np.zeros(n, dtype=float)
    trades: list[float] = []
    in_trade, entry_tgt, entry_wid = False, 0.0, -1

    for i in range(1, n):
        if not in_trade:
            if entry_mask[i]:
                in_trade      = True
                entry_tgt     = tgt_adj[i]
                entry_wid     = week_id[i]
                daily_rets[i] = -fee
        else:
            day_ret   = tgt_adj[i] / tgt_adj[i - 1] - 1.0
            # 止盈信号：SPY 收盘 > SPY 昨日最高价
            dyn_exit  = dynamic_exit and (spy_close[i] > spy_high[i - 1])
            eow_exit  = week_end_mask[i] and (week_id[i] == entry_wid)
            new_week  = week_id[i] != entry_wid

            if dyn_exit or eow_exit or new_week:
                daily_rets[i] = day_ret - fee
                trades.append(tgt_adj[i] / entry_tgt - 1.0 - 2.0 * fee)
                in_trade = False
            else:
                daily_rets[i] = day_ret

    return pl.DataFrame({"Date": dates, "strategy_ret": daily_rets}), trades


# ── 绩效摘要 ──

def build_summary_row(ticker, label, rpt, trades, rets_np) -> dict:
    m        = {r["Metric"]: r["Value"] for r in rpt["metrics"].to_dicts()}
    exposure = float(np.mean(rets_np != 0.0))
    tr       = np.array(trades, dtype=float)
    n_tr     = len(tr)
    win   = float(np.mean(tr > 0))         if n_tr > 0       else float("nan")
    avg_r = float(np.mean(tr))             if n_tr > 0       else float("nan")
    avg_w = float(np.mean(tr[tr > 0]))     if np.any(tr > 0) else float("nan")
    avg_l = float(np.mean(tr[tr <= 0]))    if np.any(tr <= 0)else float("nan")
    return dict(
        标的=ticker, 策略=label,
        交易次数=n_tr,
        年化收益=m.get("Ann.Return", float("nan")),
        年化波动=m.get("Ann.Vol",    float("nan")),
        Sharpe  =m.get("Sharpe",    float("nan")),
        MaxDD   =m.get("MaxDD",     float("nan")),
        Calmar  =m.get("Calmar",    float("nan")),
        暴露时间=exposure,
        胜率=win, 单笔均值=avg_r, 均盈=avg_w, 均亏=avg_l,
    )


# ── Plotly 图表 ──

def _base_layout(title_text, height=480):
    return dict(
        title=dict(text=title_text, font=dict(size=14, color="#111")),
        height=height,
        margin=dict(l=60, r=30, t=60, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#333"),
        hovermode="x unified",
    )


def fig_strat_cumret(skey: str, rpts_by_ticker: dict) -> go.Figure:
    """某策略下4个ETF的累计净值对比。"""
    fig = go.Figure()
    for tkr, rpt in rpts_by_ticker.items():
        df = rpt["df"]
        dates = pd.to_datetime(df["Date"].to_numpy())
        fig.add_trace(go.Scatter(
            x=dates, y=df["cum_ret"].to_numpy(),
            name=tkr,
            line=dict(color=TICKER_COLORS[tkr], width=1.8), mode="lines",
        ))
    layout = _base_layout(f"{STRAT_LABELS[skey]} — 各ETF累计净值（SPY信号）", height=420)
    layout.update(
        yaxis_title="累计净值（对数）", yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor="#eee", color="#555")
    fig.update_yaxes(showgrid=True, gridcolor="#eee", color="#555")
    return fig


def fig_all_strats_cumret(all_rpts: dict) -> go.Figure:
    """所有策略×所有ETF的累计净值（策略4为代表展示）。"""
    fig = go.Figure()
    # BH 参考线
    for tkr in TRADE_TKRS:
        rpt = all_rpts[tkr].get("bh")
        if rpt:
            df = rpt["df"]
            dates = pd.to_datetime(df["Date"].to_numpy())
            fig.add_trace(go.Scatter(
                x=dates, y=df["cum_ret"].to_numpy(),
                name=f"{tkr} Buy&Hold",
                line=dict(color=TICKER_COLORS[tkr], width=1, dash="dot"),
                mode="lines", opacity=0.5,
            ))
    # 策略4曲线（各ETF）
    for tkr in TRADE_TKRS:
        rpt = all_rpts[tkr].get("s4")
        if rpt:
            df = rpt["df"]
            dates = pd.to_datetime(df["Date"].to_numpy())
            fig.add_trace(go.Scatter(
                x=dates, y=df["cum_ret"].to_numpy(),
                name=f"{tkr} 策略4",
                line=dict(color=TICKER_COLORS[tkr], width=2),
                mode="lines",
            ))
    layout = _base_layout("策略4·动态止盈（SPY信号）× 各ETF  vs  Buy&Hold", height=480)
    layout.update(
        yaxis_title="累计净值（对数）", yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor="#eee", color="#555")
    fig.update_yaxes(showgrid=True, gridcolor="#eee", color="#555")
    return fig


def fig_wealth_dd_4(skey: str, rpts_by_ticker: dict) -> go.Figure:
    """同策略4个ETF的净值+回撤，2行×2列布局。"""
    tkrs = list(rpts_by_ticker.keys())
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=False,
        subplot_titles=[f"{t}" for t in tkrs],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for idx, tkr in enumerate(tkrs):
        r, c = positions[idx]
        df = rpts_by_ticker[tkr]["df"]
        dates = pd.to_datetime(df["Date"].to_numpy())
        color = TICKER_COLORS[tkr]
        fig.add_trace(go.Scatter(
            x=dates, y=df["wealth"].to_numpy(), name=tkr,
            line=dict(color=color, width=1.4),
            fill="tozeroy", fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4))+(0.08,)}",
            showlegend=False,
        ), row=r, col=c)
    for r, c in positions:
        fig.update_yaxes(type="log", showgrid=True, gridcolor="#eee", row=r, col=c)
        fig.update_xaxes(showgrid=True, gridcolor="#eee", row=r, col=c)
    fig.update_layout(
        title=dict(text=f"{STRAT_LABELS[skey]} — 净值曲线（SPY信号）", font=dict(size=13)),
        height=520, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=55, r=25, t=70, b=40), font=dict(color="#333"),
    )
    return fig


def fig_calendar_heatmap(cal_df, title) -> go.Figure:
    pdf  = cal_df.to_pandas().set_index("Year")
    cols = MONTH_ORDER + (["Annual"] if "Annual" in pdf.columns else [])
    pdf  = pdf[[c for c in cols if c in pdf.columns]].astype(float)
    vals = pdf.values[~np.isnan(pdf.values)]
    zmax = max(abs(vals).max(), 0.01) if len(vals) else 0.01
    text = pdf.map(lambda v: f"{v:.1%}" if isinstance(v, float) and not np.isnan(v) else "")
    fig  = go.Figure(go.Heatmap(
        z=pdf.values, x=pdf.columns.tolist(), y=pdf.index.astype(str).tolist(),
        colorscale=[[0.0,"rgb(198,40,40)"],[0.5,"rgb(255,255,255)"],[1.0,"rgb(46,125,50)"]],
        zmid=0, zmin=-zmax, zmax=zmax,
        text=text.values, texttemplate="%{text}", textfont=dict(size=8),
        showscale=False, hoverongaps=False,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color="#111")),
        height=max(260, 20 * len(pdf) + 80),
        margin=dict(l=55, r=15, t=45, b=30),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="#333"), yaxis=dict(autorange="reversed"),
    )
    return fig


# ── HTML 生成 ──

def build_html(summary_rows: list[dict], all_rpts: dict) -> str:
    def pct(v, d=2):
        return f"{v:.{d}%}" if isinstance(v, float) and not np.isnan(v) else "—"
    def flt(v, d=2):
        return f"{v:.{d}f}" if isinstance(v, float) and not np.isnan(v) else "—"
    def div(fig, fid):
        return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                           div_id=fid, config={"responsive": True})

    # 绩效汇总表：按策略分组
    table_html = ""
    for skey in STRAT_KEYS:
        slabel = STRAT_LABELS[skey]
        table_html += f"""
        <tr class="group-header"><td colspan="13">{slabel}</td></tr>"""
        rows_for_strat = [r for r in summary_rows if r["策略"] == slabel]
        for r in rows_for_strat:
            cagr_cls = "pos" if r["年化收益"] > 0 else "neg"
            avg_cls  = "pos" if r["单笔均值"] > 0 else "neg"
            table_html += f"""
        <tr>
          <td class="tkr-cell">{r['标的']}</td><td>{r['策略']}</td><td>{r['交易次数']}</td>
          <td class="{cagr_cls}">{pct(r['年化收益'])}</td>
          <td class="neg">{pct(r['MaxDD'])}</td>
          <td>{flt(r['Sharpe'])}</td><td>{flt(r['Calmar'])}</td>
          <td>{pct(r['年化波动'])}</td><td>{pct(r['暴露时间'],1)}</td>
          <td>{pct(r['胜率'],1)}</td>
          <td class="{avg_cls}">{pct(r['单笔均值'],3)}</td>
          <td class="pos">{pct(r['均盈'],3)}</td>
          <td class="neg">{pct(r['均亏'],3)}</td>
        </tr>"""

    # 总览图
    overview_div = div(fig_all_strats_cumret(all_rpts), "overview")

    # 各策略分节
    strat_sections = ""
    for skey in STRAT_KEYS:
        rpts_by_tkr = {tkr: all_rpts[tkr][skey] for tkr in TRADE_TKRS if skey in all_rpts[tkr]}
        # 累计净值
        cumret_div = div(fig_strat_cumret(skey, rpts_by_tkr), f"{skey}_cum")
        # 净值曲线 2×2
        wd_div = div(fig_wealth_dd_4(skey, rpts_by_tkr), f"{skey}_wd")
        # 日历热图：2×2 grid
        cal_divs = "".join(
            f'<div class="chart">{div(fig_calendar_heatmap(rpts_by_tkr[tkr]["calendar"], f"{tkr} 月度收益"), f"{skey}_{tkr}_cal")}</div>'
            for tkr in TRADE_TKRS if tkr in rpts_by_tkr
        )
        strat_sections += f"""
        <section>
          <h2>{STRAT_LABELS[skey]}</h2>
          <div class="chart">{cumret_div}</div>
          <div style="height:14px"></div>
          <div class="chart">{wd_div}</div>
          <div style="height:14px"></div>
          <div class="four-col">{cal_divs}</div>
        </section>"""

    plotly_js = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Turnaround Tuesday — 跨ETF（SPY信号）</title>
  {plotly_js}
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif;
            background: #f5f7fa; color: #222; margin: 0; padding: 0; font-size: 14px; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 28px 20px 60px; }}
    h1 {{ font-size: 1.6rem; font-weight: 700; margin: 0 0 4px; color: #111; }}
    .subtitle {{ color: #666; font-size: 0.85rem; margin-bottom: 24px; }}
    h2 {{ font-size: 1.05rem; font-weight: 600; margin: 0 0 14px; color: #1a1a1a;
          border-left: 3px solid #4a90d9; padding-left: 10px; }}
    section {{ background: #fff; border-radius: 6px; padding: 20px 22px; margin-bottom: 20px;
               box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .chart {{ min-height: 100px; }}
    .tbl-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; }}
    th, td {{ padding: 7px 11px; text-align: right; border-bottom: 1px solid #eee; white-space: nowrap; }}
    th {{ background: #f0f4f8; font-weight: 600; text-align: center; color: #333; }}
    td:first-child {{ text-align: left; font-weight: 600; }}
    td:nth-child(2) {{ text-align: left; font-weight: 400; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f8fafc; }}
    tr.group-header td {{ background: #eef3fa; color: #1a4a8a; font-weight: 700;
                          font-size: 0.85rem; padding: 6px 11px; border-top: 2px solid #c5d8fb; }}
    .pos {{ color: #2e7d32; font-weight: 600; }}
    .neg {{ color: #c62828; font-weight: 600; }}
    .tkr-cell {{ font-weight: 700; letter-spacing: .3px; }}
    .four-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    @media (max-width: 700px) {{ .four-col {{ grid-template-columns: 1fr; }} }}
    .badge {{ display:inline-block; background:#e8f0fe; color:#1a73e8; font-size:.75rem;
              padding:2px 8px; border-radius:4px; margin-left:8px; border:1px solid #c5d8fb;
              vertical-align:middle; font-weight:600; }}
    .legend-dots {{ display:flex; gap:20px; flex-wrap:wrap; margin-bottom:10px; font-size:.83rem; }}
    .dot {{ width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:4px; }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Turnaround Tuesday <span class="badge">跨ETF · SPY信号</span></h1>
  <p class="subtitle">
    信号来源：SPY（每周首个交易日指标）&nbsp;|&nbsp;
    交易标的：SPY / IWM（小盘）/ QQQ（纳斯达克）/ DIA（道琼斯）&nbsp;|&nbsp;
    {START[:7]} ~ {END[:7]}&nbsp;|&nbsp;不计手续费与滑点
  </p>

  <section>
    <h2>策略说明</h2>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;font-size:.83rem;line-height:1.8">
      <div style="background:#f8f9fb;border:1px solid #e4e8ee;border-radius:4px;padding:10px 12px">
        <b>策略1 · 首日跌1%买入</b><br>SPY 每周首日收盘 &lt; 前日收盘×0.99<br>→ 当日买，次日收盘卖
      </div>
      <div style="background:#f8f9fb;border:1px solid #e4e8ee;border-radius:4px;padding:10px 12px">
        <b>策略2 · 首日IBS&lt;0.2</b><br>SPY 每周首日 收盘&lt;开盘 且 IBS&lt;0.2<br>→ 当日买，次日收盘卖
      </div>
      <div style="background:#f8f9fb;border:1px solid #e4e8ee;border-radius:4px;padding:10px 12px">
        <b>策略3 · 持有至周末</b><br>SPY 首日 收盘&lt;前日收盘 且 IBS&lt;0.5<br>→ 当日买，当周最后交易日收盘卖
      </div>
      <div style="background:#f8f9fb;border:1px solid #e4e8ee;border-radius:4px;padding:10px 12px">
        <b>策略4 · 动态止盈/周末</b><br>同策略3入场<br>→ SPY收盘&gt;SPY昨日最高价则止盈，否则当周末平仓
      </div>
      <div style="background:#f8f9fb;border:1px solid #e4e8ee;border-radius:4px;padding:10px 12px">
        <b>反向 · 首日追涨</b><br>SPY 首日 收盘&gt;开盘 且 IBS&gt;0.8<br>→ 当日买，次日收盘卖
      </div>
    </div>
    <div class="legend-dots" style="margin-top:14px">
      {"".join(f'<span><span class="dot" style="background:{TICKER_COLORS[t]}"></span>{t}</span>' for t in TRADE_TKRS)}
    </div>
  </section>

  <section>
    <h2>绩效汇总（按策略分组）</h2>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>标的</th><th>策略</th><th>交易次数</th><th>年化收益</th><th>MaxDD</th>
          <th>Sharpe</th><th>Calmar</th><th>年化波动</th><th>暴露时间</th>
          <th>胜率</th><th>单笔均值</th><th>均盈</th><th>均亏</th>
        </tr></thead>
        <tbody>{table_html}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>总览：策略4 × 各ETF vs Buy&Hold</h2>
    <div class="chart">{overview_div}</div>
  </section>

  {strat_sections}
</div>
</body>
</html>"""


# ── 主流程 ──

if __name__ == "__main__":
    print("Loading SPY signal data...")
    spy_df = load_spy_signals()
    print(f"  rows={len(spy_df)}, date={spy_df['date'][0]}~{spy_df['date'][-1]}")

    # SPY 信号数组
    spy_dates   = spy_df["date"].to_numpy()
    is_first    = spy_df["is_first_trade_day"].to_numpy()
    is_last     = spy_df["is_last_trade_day"].to_numpy()
    week_id     = (spy_df["iso_year"].to_numpy() * 100 + spy_df["iso_week"].to_numpy()).astype(int)

    spy_adj     = spy_df["adj_close"].to_numpy()
    spy_prev    = spy_df["prev_adj_close"].to_numpy()
    spy_close   = spy_df["close"].to_numpy()
    spy_high    = spy_df["high"].to_numpy()
    spy_open    = spy_df["open"].to_numpy()
    ibs         = spy_df["ibs"].to_numpy()

    # 信号掩码（所有信号均来自 SPY）
    mask1    = is_first & (spy_adj  < spy_prev * 0.99)
    mask2    = is_first & (spy_close < spy_open) & (ibs < 0.2)
    mask3    = is_first & (spy_adj  < spy_prev) & (ibs < 0.5)
    mask4    = mask3.copy()
    mask_rev = is_first & (spy_close > spy_open) & (ibs > 0.8)

    print(f"\n  信号触发次数: s1={mask1.sum()}, s2={mask2.sum()}, "
          f"s3={mask3.sum()}, s4={mask4.sum()}, rev={mask_rev.sum()}")

    # 逐个交易标的回测
    all_rpts     : dict[str, dict] = {}
    summary_rows : list[dict]      = []

    for tkr in TRADE_TKRS:
        print(f"\nLoading {tkr}...")
        tgt_adj = load_target(tkr, spy_dates)

        print(f"  Running backtests for {tkr}...")
        r1,  t1  = backtest_next_day(spy_adj, tgt_adj, spy_dates, mask1)
        r2,  t2  = backtest_next_day(spy_adj, tgt_adj, spy_dates, mask2)
        r3,  t3  = backtest_end_of_week(spy_adj, spy_close, spy_high, tgt_adj, spy_dates,
                                         mask3, is_last, week_id)
        r4,  t4  = backtest_end_of_week(spy_adj, spy_close, spy_high, tgt_adj, spy_dates,
                                         mask4, is_last, week_id, dynamic_exit=True)
        r_rv, trr= backtest_next_day(spy_adj, tgt_adj, spy_dates, mask_rev)

        # BH for this ticker
        bh_ret = pl.DataFrame({
            "Date": spy_dates,
            "strategy_ret": np.where(
                np.arange(len(spy_dates)) > 0,
                np.concatenate([[0.0], tgt_adj[1:] / tgt_adj[:-1] - 1.0]),
                0.0,
            ),
        })

        strat_inputs = [
            ("s1",  r1,   t1,  STRAT_LABELS["s1"]),
            ("s2",  r2,   t2,  STRAT_LABELS["s2"]),
            ("s3",  r3,   t3,  STRAT_LABELS["s3"]),
            ("s4",  r4,   t4,  STRAT_LABELS["s4"]),
            ("rev", r_rv, trr, STRAT_LABELS["rev"]),
        ]

        rpts_this = {}
        for skey, ret_df, trades, label in strat_inputs:
            rpt = pnl_report_polars_plotnine(
                ret_df, date_col="Date", ret_col="strategy_ret", info=f"{tkr}·{label}", freq=252,
            )
            rpts_this[skey] = rpt
            summary_rows.append(build_summary_row(tkr, label, rpt, trades, ret_df["strategy_ret"].to_numpy()))
            m = {r["Metric"]: r["Value"] for r in rpt["metrics"].to_dicts()}
            print(f"    {skey}: 年化={m['Ann.Return']:.2%}  Sharpe={m['Sharpe']:.2f}  MaxDD={m['MaxDD']:.2%}")

        rpts_this["bh"] = pnl_report_polars_plotnine(
            bh_ret, date_col="Date", ret_col="strategy_ret", info=f"{tkr} BH", freq=252,
        )
        all_rpts[tkr] = rpts_this

    print("\nGenerating HTML...")
    html = build_html(summary_rows, all_rpts)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved: {OUT_HTML}")
