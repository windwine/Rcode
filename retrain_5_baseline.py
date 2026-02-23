
#%%
import os
import pandas as pd
import numpy as np
import time
from datetime import date
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from lightgbm import LGBMRegressor
from sklearn.utils import resample
import catboost as cat
from catboost import Pool
# import pyfolio as pf
import polars as pl
from plotnine import (
    ggplot, aes, geom_bar, geom_line,
    labs, theme_minimal,scale_y_log10
)

import warnings
warnings.filterwarnings('ignore')

#%%
os.chdir(r'e:/laosongdata')
# plXs = pl.read_parquet('alldataFeatures.parquet')

#%%
for actdays in np.arange(1,2):
    market_index = 'SHSZ'
    input_filename=f"{market_index}_{actdays}_EODtestraw.parquet"
    print(input_filename)
    
    MLdata = pl.read_parquet(input_filename) # the data is w/o index, the data is already Z scored on classy and fillna with 0 for x
    
    front = ["ID", "Date", "y", "classy", "Janind", "d1"]

    MLdata = (
        MLdata
        .with_columns(pl.col("Date").cast(pl.Date))
        .select(
            *front,                    # <-- IMPORTANT: splat the list
            pl.all().exclude(front)
        )
    )
    print(MLdata.columns)
    ###########################################
    

    y_name = "classy"

    cols = MLdata.columns
    x_cols = cols[4:]   # R: 5:94  → Python 0-based

    pred_blocks = []

    for y in range(2016, 2026):
        time1 = pl.date(y, 1, 1)
        time2 = pl.date(y + 1, 1, 1)

        train = MLdata.filter(pl.col("Date") < time1)
        test  = MLdata.filter(
            (pl.col("Date") >= time1) &
            (pl.col("Date") <  time2)
        )

        # ---- NumPy bridge for CatBoost ----
        Xtrain = train.select(x_cols).to_numpy()
        Ytrain = train.select(y_name).to_numpy().ravel()

        model = cat.CatBoostRegressor(
            logging_level="Silent",
            iterations=5000,
            task_type="GPU"
        )

        model.fit(Pool(Xtrain, Ytrain))

        # ---- predict ----
        Xtest = test.select(x_cols).to_numpy()
        Ytest = test.select(y_name).to_numpy().ravel()

        pred_y = model.predict(Pool(Xtest, Ytest))

        test_with_pred = test.with_columns(
            pl.Series("pred_y", pred_y)
        )

        pred_blocks.append(test_with_pred)
        print(f"Finished year {y}")


    bigtest = pl.concat(pred_blocks)


    filename = "catboost_" + input_filename
    bigtest.write_parquet(filename)



#%%


topN = 100

thresh = 0.2

zz500 = bigtest.filter(pl.col("ID") == "SH000905").select(pl.col("Date"), pl.col("y").alias("y_zz500"))


bigtest2 = (
    bigtest
    # 1) inner join on Date, ID
    # .join(alldata, on=["Date", "ID"], how="inner")
    
    # 2) create derived distance variables
    .with_columns([
        ((pl.col("d1") - 1.1).abs() * 100).alias("d1_1"),
        ((pl.col("d1") - 1.05).abs() * 100).alias("d1_st"),
        ((pl.col("d1") - 1.2).abs() * 100).alias("d1_chy"),
    ])
    
    # 3) filter by threshold conditions
    .filter(
        (pl.col("d1_1")   >= thresh) &
        (pl.col("d1_st")  >= thresh * 1) &
        (pl.col("d1_chy") >= thresh)
    )
    .with_columns(y=pl.col("y") - 1)  # adjust y to be excess return (assuming original y is 1 + return
)


strategy = (
    bigtest2
    # 1) Cross-sectional rank by Date
    .with_columns(
        pl.col("pred_y")
          .rank(method="average", descending=True)
          .over("Date")
          .alias("cs_rank")
    )
    # 2) Keep top N per Date
    .filter(pl.col("cs_rank") <= topN)
    # 3) Equal-weight portfolio return = mean(y)
    .group_by("Date")
    .agg(
        pl.col("y").mean().alias("strategy_ret")
    )
    .sort("Date")
    .join(zz500, on="Date",how="inner")
    .with_columns(
        (pl.col("strategy_ret") - pl.col("y_zz500")+1).alias("strategy_ret2")
    )  
    # 4) Cumulative return
    .with_columns(
        (1 + pl.col("strategy_ret")-2e-3).cum_prod().alias("cum_ret"),
        (1 + pl.col("strategy_ret2")-2e-3).cum_prod().alias("cum_ret2")
    )
)



# -------------------------------------------------
# 5) Plotting (plotnine / ggplot style)
# -------------------------------------------------



# Cumulative long–short return
p2 = (
    ggplot(strategy.to_pandas(), aes(x="Date", y="cum_ret"))
    + geom_line()
    + labs(
        title="Cumulative Long–Short Strategy Returns",
        x="Date",
        y="Cumulative Return"
    )
    + scale_y_log10()  # Log scale for better visibility

)

p2.draw()
# %%
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

# ---- usage ----
# out = pnl_report_polars_plotnine(strategy, date_col="Date", ret_col="strategy_ret", info="MyStrategy", freq=52)
# print(out["metrics"])
# print(out["yearly"])
# print(out["calendar"])
# print(out["plot_wealth"]); print(out["plot_dd"]); print(out["plot_ret"])

# ---- usage ----
# out = pnl_report_polars_plotnine(strategy, date_col="Date", ret_col="strategy_ret", info="MyStrategy", freq=52)
# out["metrics"], out["yearly"], out["calendar"]
# print(out["plot_wealth"]); print(out["plot_dd"]); print(out["plot_ret"])


# ---- usage ----
out = pnl_report_polars_plotnine(strategy, date_col="Date", ret_col="strategy_ret", info="MyStrategy", freq=52)
print(out["metrics"])
print(out["yearly"]) 
print(out["calendar"])
out["plot_wealth"].draw()
print(out["plot_wealth"].draw()); print(out["plot_dd"]); print(out["plot_ret"])

# %%
