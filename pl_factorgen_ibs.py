#%%
# =============================================================================
# pl_factorgen_ibs.py  —  Group IBS: 日内价格结构
#
# 输入 : alldataFeatures3.parquet  (读 ID, Date, O, H, L, C)
# 输出 : features_ibs.parquet      (ID, Date + 13 列)
#
# 新列:
#   ibs               (C-L)/(H-L), Internal Bar Strength, 0~1
#   ibs_mu5/20/60     ibs 滚动均值
#   intraday_ret      C/O - 1, 日内收益
#   intraday_ret_mu5/20/60
#   intraday_ret_sd5/20/60
#   H_ratio           H/C, 当日最高相对收盘
#   L_ratio           L/C, 当日最低相对收盘
# =============================================================================

import polars as pl
import gc

#%%
pathname = "d:/laosongdata"

df = pl.read_parquet(f"{pathname}/alldataFeatures3.parquet", columns=["ID", "Date", "O", "H", "L", "C"])
df = df.sort(["ID", "Date"])
print(f"Loaded: {df.shape}")

#%%
roll_windows = [5, 20, 60]

# 基础日内指标
df = df.with_columns([
    # IBS: (C-L)/(H-L), H==L 时设为 0.5
    pl.when(pl.col("H") > pl.col("L"))
      .then((pl.col("C") - pl.col("L")) / (pl.col("H") - pl.col("L")))
      .otherwise(pl.lit(0.5))
      .alias("ibs"),
    # 日内收益
    (pl.col("C") / pl.col("O") - 1).alias("intraday_ret"),
    # 高低价比
    (pl.col("H") / pl.col("C")).alias("H_ratio"),
    (pl.col("L") / pl.col("C")).alias("L_ratio"),
])

# 滚动统计
ibs_roll = [
    pl.col("ibs").rolling_mean(window_size=w).over("ID").alias(f"ibs_mu{w}")
    for w in roll_windows
]
intra_mu = [
    pl.col("intraday_ret").rolling_mean(window_size=w).over("ID").alias(f"intraday_ret_mu{w}")
    for w in roll_windows
]
intra_sd = [
    pl.col("intraday_ret").rolling_std(window_size=w).over("ID").alias(f"intraday_ret_sd{w}")
    for w in roll_windows
]

result = (
    df
    .with_columns(ibs_roll + intra_mu + intra_sd)
    .select(
        ["ID", "Date",
         "ibs"] +
        [f"ibs_mu{w}"           for w in roll_windows] +
        ["intraday_ret"] +
        [f"intraday_ret_mu{w}"  for w in roll_windows] +
        [f"intraday_ret_sd{w}"  for w in roll_windows] +
        ["H_ratio", "L_ratio"]
    )
)

del df; gc.collect()

print(f"Output shape: {result.shape}")
result.write_parquet(f"{pathname}/features_ibs.parquet")
print(f"Saved features_ibs.parquet")

# %%
