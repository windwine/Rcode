#%%
# =============================================================================
# pl_factorgen_vol.py  —  Group VOL: 成交额动量
#
# 输入 : alldataFeatures3.parquet  (读 ID, Date, amount)
# 输出 : features_vol.parquet      (ID, Date + 19 列)
#
# 新列:
#   amount_mu5/20/60/120/250       滚动均值
#   amount_sd5/20/60/120/250       滚动标准差
#   amount_zscore5/20/60/120/250   滚动 z-score
#   amount_cumret1/5/10/20         相对 n 日前的成交额倍数变化
# =============================================================================

import polars as pl
import gc

#%%
pathname = "d:/laosongdata"

df = pl.read_parquet(f"{pathname}/alldataFeatures3.parquet", columns=["ID", "Date", "amount"])
df = df.sort(["ID", "Date"])
print(f"Loaded: {df.shape}")

#%%
mu_windows     = [5, 20, 60, 120, 250]
sd_windows     = [5, 20, 60, 120, 250]
cumret_windows = [1, 5, 10, 20]

mu_exprs = [
    pl.col("amount").rolling_mean(window_size=w).over("ID").alias(f"amount_mu{w}")
    for w in mu_windows
]
sd_exprs = [
    pl.col("amount").rolling_std(window_size=w).over("ID").alias(f"amount_sd{w}")
    for w in sd_windows
]
zscore_exprs = [
    (
        (pl.col("amount") - pl.col("amount").rolling_mean(window_size=w))
        / pl.col("amount").rolling_std(window_size=w)
    ).over("ID").alias(f"amount_zscore{w}")
    for w in mu_windows
]
cumret_exprs = [
    (pl.col("amount") / pl.col("amount").shift(n) - 1).over("ID").alias(f"amount_cumret{n}")
    for n in cumret_windows
]

result = (
    df
    .with_columns(mu_exprs)
    .with_columns(sd_exprs)
    .with_columns(zscore_exprs)
    .with_columns(cumret_exprs)
    .select(["ID", "Date"] +
            [f"amount_mu{w}"      for w in mu_windows] +
            [f"amount_sd{w}"      for w in sd_windows] +
            [f"amount_zscore{w}"  for w in mu_windows] +
            [f"amount_cumret{n}"  for n in cumret_windows])
)

del df; gc.collect()

print(f"Output shape: {result.shape}")
result.write_parquet(f"{pathname}/features_vol.parquet")
print(f"Saved features_vol.parquet")

# %%
