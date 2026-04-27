import polars as pl
import numpy as np
from plotnine import (
    ggplot, aes, geom_line, labs, theme_minimal,
    scale_y_log10
)

MONTH_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def pnl_report_polars_plotnine(
    strategy: pl.DataFrame,
    date_col: str = "Date",
    ret_col: str = "strategy_ret",
    info: str = "Strategy",
    freq: int = 52,         # weekly
    rf_annual: float = 0.0,
    force_all_months: bool = True,   # <- show Jan..Dec even if missing
):
    # --- clean + sort ---
    df = (
        strategy
        .select([date_col, ret_col])
        .drop_nulls()
        .sort(date_col)
        .with_columns(pl.col(date_col).cast(pl.Datetime))  # for group_by_dynamic
    )

    # --- wealth & drawdown ---
    df = (
        df.with_columns((1 + pl.col(ret_col)).cum_prod().alias("wealth"))
          .with_columns((pl.col("wealth") / pl.col("wealth").cum_max() - 1).alias("drawdown"))
          .with_columns((1 + pl.col(ret_col)).cum_prod().alias("cum_ret"))
    )

    # --- numpy stats ---
    r = df.select(ret_col).to_numpy().ravel()
    n = int(r.size)
    if n == 0:
        raise ValueError("No returns left after drop_nulls().")

    ann_ret = float(np.prod(1.0 + r) ** (freq / n) - 1.0)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(freq)) if n > 1 else np.nan
    rf_per_period = rf_annual / freq
    ann_sharpe = float(((np.mean(r) - rf_per_period) / np.std(r, ddof=1)) * np.sqrt(freq)) if n > 1 else np.nan

    max_dd = float(df.select(pl.col("drawdown").min()).item())
    calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else np.nan

    metrics = pl.DataFrame({
        "Metric": ["Ann.Return", "Ann.Vol", "Sharpe", "MaxDD", "Calmar", "#Obs"],
        "Value":  [ann_ret,      ann_vol,   ann_sharpe, max_dd, calmar,  float(n)],
    })

    # --- yearly returns from wealth ---
    yearly = (
        df.with_columns(pl.col(date_col).dt.year().alias("Year"))
          .group_by("Year")
          .agg([
              pl.col("wealth").first().alias("wealth_start"),
              pl.col("wealth").last().alias("wealth_end"),
          ])
          .with_columns((pl.col("wealth_end") / pl.col("wealth_start") - 1).alias("year_ret"))
          .select(["Year", "year_ret"])
          .sort("Year")
    )

    # --- last 20 weekly returns ---
    last20 = df.select([date_col, ret_col]).tail(20)

    # --- monthly returns from wealth (month-end wealth -> pct change) ---
    month_end = (
        df.group_by_dynamic(date_col, every="1mo", closed="right", label="right")
          .agg(pl.col("wealth").last().alias("wealth_m"))
          .sort(date_col)
          .with_columns((pl.col("wealth_m") / pl.col("wealth_m").shift(1) - 1).alias("month_ret"))
          .drop_nulls("month_ret")
          .with_columns([
              pl.col(date_col).dt.year().alias("Year"),
              pl.col(date_col).dt.month().alias("Month"),
          ])
          .select(["Year", "Month", "month_ret"])
    )

    # --- calendar table: Year x Month + Annual ---
    cal_table = (
        month_end
        .pivot(values="month_ret", index="Year", columns="Month", aggregate_function="first")
        .sort("Year")
    )

    # pivot month columns are strings like "1","2",... so rename using string keys
    month_cols = [c for c in cal_table.columns if c.isdigit()]
    rename_map = {c: MONTH_MAP[int(c)] for c in month_cols}
    cal_table = cal_table.rename(rename_map)

    # add Annual
    cal_table = cal_table.join(yearly.rename({"year_ret": "Annual"}), on="Year", how="left")

    # enforce month column order
    if force_all_months:
        missing = [m for m in MONTH_ORDER if m not in cal_table.columns]
        if missing:
            cal_table = cal_table.with_columns(
                [pl.lit(None).cast(pl.Float64).alias(m) for m in missing]
            )
        cal_table = cal_table.select(["Year"] + MONTH_ORDER + (["Annual"] if "Annual" in cal_table.columns else []))
    else:
        cols_order = ["Year"] + [m for m in MONTH_ORDER if m in cal_table.columns] + (["Annual"] if "Annual" in cal_table.columns else [])
        cal_table = cal_table.select(cols_order)

    # --- plotnine plots (convert only here) ---
    pdf = df.select([date_col, ret_col, "wealth", "drawdown", "cum_ret"]).to_pandas()

    p_wealth = (
        ggplot(pdf, aes(x=date_col, y="wealth"))
        + geom_line()
        + scale_y_log10()
        + labs(title=f"{info} — Wealth Index (log scale)", x="Date", y="Wealth")
        + theme_minimal()
    )

    p_dd = (
        ggplot(pdf, aes(x=date_col, y="drawdown"))
        + geom_line()
        + labs(title=f"{info} — Drawdown", x="Date", y="Drawdown")
        + theme_minimal()
    )

    p_ret = (
        ggplot(pdf, aes(x=date_col, y=ret_col))
        + geom_line()
        + labs(title=f"{info} — Weekly Returns", x="Date", y="Return")
        + theme_minimal()
    )

    return {
        "df": df,                 # Polars df with wealth/drawdown/cum_ret
        "metrics": metrics,       # Polars table
        "yearly": yearly,         # Polars table
        "last20": last20,         # Polars table
        "calendar": cal_table,    # Polars table (Jan..Dec order)
        "plot_wealth": p_wealth,  # plotnine object
        "plot_dd": p_dd,          # plotnine object
        "plot_ret": p_ret,        # plotnine object
    }




