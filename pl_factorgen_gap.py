#%%
# =============================================================================
# pl_factorgen_gap.py  —  Group GAP: 隔夜跳空
#
# 输入 : alldataFeatures3.parquet  (读 ID, Date, O, C)
# 输出 : features_gap.parquet      (ID, Date + 9 列)
#
# 新列:
#   gap               O/C.shift(1)-1, 隔夜跳空收益
#   gap_mu5/20/60     gap 滚动均值
#   gap_sd5/20        gap 滚动标准差
#   gap_zscore5/20/60 gap 滚动 z-score
# =============================================================================

import polars as pl
import gc

#%%
pathname = "d:/laosongdata"

df = pl.read_parquet(f"{pathname}/alldataFeatures3.parquet", columns=["ID", "Date", "O", "C"])
df = df.sort(["ID", "Date"])
print(f"Loaded: {df.shape}")

#%%
mu_windows     = [5, 20, 60]
sd_windows     = [5, 20]
zscore_windows = [5, 20, 60]

df = df.with_columns(
    (pl.col("O") / pl.col("C").shift(1) - 1).over("ID").alias("gap")
)

gap_mu = [
    pl.col("gap").rolling_mean(window_size=w).over("ID").alias(f"gap_mu{w}")
    for w in mu_windows
]
gap_sd = [
    pl.col("gap").rolling_std(window_size=w).over("ID").alias(f"gap_sd{w}")
    for w in sd_windows
]
gap_zscore = [
    (
        (pl.col("gap") - pl.col("gap").rolling_mean(window_size=w))
        / pl.col("gap").rolling_std(window_size=w)
    ).over("ID").alias(f"gap_zscore{w}")
    for w in zscore_windows
]

result = (
    df
    .with_columns(gap_mu + gap_sd + gap_zscore)
    .select(
        ["ID", "Date", "gap"] +
        [f"gap_mu{w}"     for w in mu_windows] +
        [f"gap_sd{w}"     for w in sd_windows] +
        [f"gap_zscore{w}" for w in zscore_windows]
    )
)

del df; gc.collect()

print(f"Output shape: {result.shape}")
result.write_parquet(f"{pathname}/features_gap.parquet")
print(f"Saved features_gap.parquet")

# %%
