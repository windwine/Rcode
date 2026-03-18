#%%
# =============================================================================
# pl_factorgen_ma.py  —  Group MA: 均线系统
#
# 输入 : alldataFeatures3.parquet  (读 ID, Date, C)
# 输出 : features_ma.parquet       (ID, Date + 7 列)
#
# 新列:
#   ma_cross_20_60   MA20 / MA60, >1 金叉区域, <1 死叉区域
#   ma_cross_5_20    MA5  / MA20, 短期趋势
#   C_above_ma20     C > MA20 (0/1)
#   C_above_ma60     C > MA60 (0/1)
#   ma_slope_20      MA20 当前 / MA20 5日前 - 1, 均线斜率
#   ma_slope_60      MA60 当前 / MA60 10日前 - 1
#   ma_ratio_60_250  MA60 / MA250, 中长期趋势
# =============================================================================

import polars as pl
import gc

#%%
pathname = "d:/laosongdata"

df = pl.read_parquet(f"{pathname}/alldataFeatures3.parquet", columns=["ID", "Date", "C"])
df = df.sort(["ID", "Date"])
print(f"Loaded: {df.shape}")

#%%
# 先算各均线
df = df.with_columns([
    pl.col("C").rolling_mean(window_size=5).over("ID").alias("_ma5"),
    pl.col("C").rolling_mean(window_size=20).over("ID").alias("_ma20"),
    pl.col("C").rolling_mean(window_size=60).over("ID").alias("_ma60"),
    pl.col("C").rolling_mean(window_size=250).over("ID").alias("_ma250"),
])

result = (
    df
    .with_columns([
        (pl.col("_ma20") / pl.col("_ma60")).alias("ma_cross_20_60"),
        (pl.col("_ma5")  / pl.col("_ma20")).alias("ma_cross_5_20"),
        (pl.col("C") > pl.col("_ma20")).cast(pl.Int8).alias("C_above_ma20"),
        (pl.col("C") > pl.col("_ma60")).cast(pl.Int8).alias("C_above_ma60"),
        (pl.col("_ma20") / pl.col("_ma20").shift(5)  - 1).over("ID").alias("ma_slope_20"),
        (pl.col("_ma60") / pl.col("_ma60").shift(10) - 1).over("ID").alias("ma_slope_60"),
        (pl.col("_ma60") / pl.col("_ma250")).alias("ma_ratio_60_250"),
    ])
    .select(["ID", "Date",
             "ma_cross_20_60", "ma_cross_5_20",
             "C_above_ma20", "C_above_ma60",
             "ma_slope_20", "ma_slope_60",
             "ma_ratio_60_250"])
)

del df; gc.collect()

print(f"Output shape: {result.shape}")
result.write_parquet(f"{pathname}/features_ma.parquet")
print(f"Saved features_ma.parquet")

# %%
