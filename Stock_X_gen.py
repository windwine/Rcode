#%%
import polars as pl
import pyarrow as pa
import pandas as pd
import gc

#%%
pathname = "E:/OMdata"

# ---- Load Parquet file ----
alldataEODs = (
    pl.read_parquet(f"{pathname}/adPrices3OHLC.parquet")
)
#%% 

# ---- Construct alldataXs ----

alldataXs = (
    alldataEODs
    .lazy()
    .with_columns(
    pl.col("Date").cast(pl.Date),
    pl.col("SecurityID").cast(pl.Int64),
    amount = pl.col("Volume")*pl.col("Price")
    )
    .rename({"Adj_prices2": "C",
             "SecurityID": "ID"})
    .select(["ID","C", "Date","amount"])
    .filter(pl.col("Date") >= pl.date(1019, 12, 1))
    .sort(["ID", "Date"])
    .with_columns([
        pl.col("Date").cum_count().over("ID").alias("age"),
    ])  
    .with_columns(
    (pl.col("C") / pl.col("C").shift(1) - 1).over("ID").alias("ret")
    )
    .collect()

)

print(alldataXs.head())

base = alldataXs.select("ID","Date","age","ret").sort(["ID", "Date"]) # need this one for the top K

#remove alldataEODs to free memory
del alldataEODs
gc.collect()

#%% get the index ret so we can do the regression
# get the index returns for regression
market_returns = (
    alldataXs
    .lazy()
    .filter(pl.col("ID") == 109820)  
    .select(["Date", "ret"])
    .rename({"ret": "market_ret"})
    .collect()
)

alldata = (
    alldataXs
    .filter(pl.col("ID") != 109820) 
    .join(market_returns, on="Date", how="left")  # Join market returns by Date
)
                  
del alldataXs
gc.collect()


#%%
# rolling ret stats, rolling z score for C and short term stats, cumret for the past few days,
# rolling correlation between amount and C, distance to rolling Max/Min C and amount.
# rolling alpha,beta
alldata = alldata.drop_nulls(subset=["market_ret"])  # Drop rows where market return is null
# =============================================================================
# CONFIGURATION: Specify which variables to calculate features for
# =============================================================================

# Variables to calculate rolling statistics for
rolling_vars = ["ret"]

# Variables to calculate rolling z-scores for
zscore_vars = ["C", "ret_mu2", "ret_mu5","ret_sd5"]

# Variables to calculate cumulative returns for
cumret_vars = ["C"]

# Variables (pairs) to calculate rolling correlations for  <-- YOU CONTROL THIS
# Use tuples of (x_col, y_col)
corr_pairs = [
    ("C", "amount")
    # ("C", "ret_mu2"),   # add/remove pairs as you like
]

# =============================================================================
# ROLLING STATISTICS SPECIFICATION
# =============================================================================

spec = {
    "mu":   (pl.Expr.rolling_mean,     [2, 5, 10, 20, 60, 120, 250]),
    "sd":   (pl.Expr.rolling_std,      [5, 10, 20, 60, 120, 250]),
    "skew": (pl.Expr.rolling_skew,     [5, 10, 20, 60, 120, 250]),
    "kurt": (pl.Expr.rolling_kurtosis, [5, 10, 20, 60, 120, 250]),
}

# Generate rolling statistics expressions
rolling_exprs = []
for var in rolling_vars:
    for name, (fn, wins) in spec.items():
        for w in wins:
            rolling_exprs.append(
                fn(pl.col(var), window_size=w)
                .over("ID")
                .alias(f"{var}_{name}{w}")
            )

# =============================================================================
# ROLLING Z-SCORE SPECIFICATION
# =============================================================================

zscore_windows = [5, 10, 20, 60, 120, 250]

# Generate z-score expressions
zscore_exprs = []
for var in zscore_vars:
    for w in zscore_windows:
        zscore_exprs.append(
            (
                (pl.col(var) - pl.col(var).rolling_mean(window_size=w))
                / pl.col(var).rolling_std(window_size=w)
            )
            .over("ID")
            .alias(f"{var}_zscore{w}")
        )

# =============================================================================
# CUMULATIVE RETURNS SPECIFICATION
# =============================================================================

cumret_windows = range(1, 21)  # n = 1..20

# Generate cumulative return expressions
cumret_exprs = []
for var in cumret_vars:
    for n in cumret_windows:
        cumret_exprs.append(
            (pl.col(var) / pl.col(var).shift(n) - 1)
            .over("ID")
            .alias(f"{var}_cumret_{n}")
        )

# =============================================================================
# ROLLING CORRELATION SPECIFICATION
# =============================================================================

# You can choose to reuse zscore_windows or define a separate list:
corr_windows = [5, 10, 20, 60, 120, 250]

corr_exprs = []
for x, y in corr_pairs:
    for w in corr_windows:
        corr_exprs.append(
            pl.rolling_corr(pl.col(x), pl.col(y), window_size=w)
            .over("ID")
            .alias(f"corr_{x}_{y}_{w}")
        )

# =============================================================================
# ROLLING REGRESSION (SIMPLE, FAST, POLARS-NATIVE)
#   For each (y, x) and each window w:
#     beta      = Cov(y,x) / Var(x)
#     intercept = mean(y) - beta * mean(x)
#     rsq       = corr(y,x)^2
# =============================================================================

# Pairs of variables for rolling regression
# Format: list of tuples (y_var, x_var) - y is dependent, x is independent
regression_pairs = [
    ("ret", "market_ret"),  # stock return vs market return
    # ("ret", "amount"),      # return vs volume
    # ("volume", "ret"),      # volume vs return
]

regression_windows = [20, 60, 250]

regression_exprs = []

for y_var, x_var in regression_pairs:
    y = pl.col(y_var)
    x = pl.col(x_var)

    for w in regression_windows:
        cov_xy = pl.rolling_cov(y, x, window_size=w)
        var_x  = x.rolling_var(window_size=w)

        beta = cov_xy / var_x
        intercept = y.rolling_mean(window_size=w) - beta * x.rolling_mean(window_size=w)

        # corr^2
        rsq = (pl.rolling_corr(y, x, window_size=w) ** 2)

        # (optional) alpha-like: mean(y - beta*x) over window
        # alpha = (y - beta * x).rolling_mean(window_size=w)

        regression_exprs.append(beta.over("ID").alias(f"beta_{y_var}_{x_var}_{w}"))
        regression_exprs.append(intercept.over("ID").alias(f"intercept_{y_var}_{x_var}_{w}"))
        regression_exprs.append(rsq.over("ID").alias(f"rsq_{y_var}_{x_var}_{w}"))
        # regression_exprs.append(alpha.over("ID").alias(f"alpha_{y_var}_{x_var}_{w}"))

        
        
# Variables to calculate running max/min ratios for
extreme_vars = ["C", "amount"]

# Windows
extreme_windows = [5, 10, 20, 60, 120, 250]

# =============================================================================
# RUNNING MAX / MIN OVER ID + RATIOS
# =============================================================================

extreme_exprs = []

for var in extreme_vars:
    for w in extreme_windows:

        # running max / min over ID
        running_max = (
            pl.col(var)
            .rolling_max(window_size=w)
            .over("ID")
        )

        running_min = (
            pl.col(var)
            .rolling_min(window_size=w)
            .over("ID")
        )

        # ratio to running max
        extreme_exprs.append(
            (pl.col(var) / running_max)
            .alias(f"{var}_to_max{w}")
        )

        # ratio to running min
        extreme_exprs.append(
            (pl.col(var) / running_min)
            .alias(f"{var}_to_min{w}")
        )






# =============================================================================
# BUILD THE LAZY FRAME WITH ALL FEATURES
# =============================================================================

lf2 = (
    alldata.lazy()
    .sort(["ID", "Date"])
    # .with_columns(
    #     [
    #         (pl.int_range(0, pl.len()).over("ID") + 1).alias("hist"),
    #         (pl.col("C") / pl.col("C").shift(1) - 1).over("ID").alias("ret"),
    #     ]
    # )
    .with_columns(rolling_exprs)   # Rolling statistics
    .with_columns(zscore_exprs)    # Rolling z-scores
    .with_columns(cumret_exprs)    # Cumulative returns
    .with_columns(corr_exprs)      # Rolling correlations
    .with_columns(regression_exprs)  # <-- rolling regressions
    .with_columns(extreme_exprs)
    # .with_columns(top_bot_k_mean_exprs)  # <- top/bottom K means kept separate
)

df2 = lf2.collect()

df2.write_parquet(f"{pathname}/StockFeatures.parquet")



# %% top 3 rets in certain lookback window

rolling_vars = ["ret"]
topk_windows = [20, 60, 120, 250]
k = 3

base = (
    alldata
    .sort(["ID", "Date"])   # make sure sorted
    .with_columns(pl.col("age").cast(pl.Int64))
)

feat_frames = []

for w in topk_windows:
    top_names = [f"{v}_top{k}mean{w}" for v in rolling_vars]
    bot_names = [f"{v}_bot{k}mean{w}" for v in rolling_vars]

    feats_w = (
        base
        .rolling(index_column="age", period=f"{w}i", group_by="ID")
        .agg([
            # carry Date so we can join on it
            pl.col("Date").last().alias("Date"),

            *[
                pl.col(v).drop_nulls().top_k(k).mean().alias(f"{v}_top{k}mean{w}")
                for v in rolling_vars
            ],
            *[
                pl.col(v).drop_nulls().bottom_k(k).mean().alias(f"{v}_bot{k}mean{w}")
                for v in rolling_vars
            ],
        ])
        .select(["ID", "Date"] + top_names + bot_names)
    )

    feat_frames.append(feats_w)

# join all windows into one features table
features = feat_frames[0]
for f in feat_frames[1:]:
    features = features.join(f, on=["ID", "Date"], how="left")

# join back to original data
df3 = df2.join(features, on=["ID", "Date"], how="inner")

df3.write_parquet(f"{pathname}/StockFeatures2.parquet")



# %%
