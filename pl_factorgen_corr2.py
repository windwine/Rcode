#%%
# =============================================================================
# pl_factorgen_corr2.py  —  Group CORR2: 收益率-成交额相关
#
# 输入 : alldataFeatures3.parquet  (读 ID, Date, ret, amount)
# 输出 : features_corr2.parquet    (ID, Date + 6 列)
#
# 新列:
#   corr_ret_amount_5/10/20/60/120/250
#   涨时放量/跌时缩量强度，价格发现质量的代理变量
#   (现有 corr_C_amount 是价格水平 vs 成交额，本组是收益率 vs 成交额)
# =============================================================================

import polars as pl
import gc

#%%
pathname = "d:/laosongdata"

df = pl.read_parquet(f"{pathname}/alldataFeatures3.parquet", columns=["ID", "Date", "ret", "amount"])
df = df.sort(["ID", "Date"])
print(f"Loaded: {df.shape}")

#%%
corr_windows = [5, 10, 20, 60, 120, 250]

corr_exprs = [
    pl.rolling_corr(pl.col("ret"), pl.col("amount"), window_size=w)
    .over("ID")
    .alias(f"corr_ret_amount_{w}")
    for w in corr_windows
]

result = (
    df
    .with_columns(corr_exprs)
    .select(["ID", "Date"] + [f"corr_ret_amount_{w}" for w in corr_windows])
)

del df; gc.collect()

print(f"Output shape: {result.shape}")
result.write_parquet(f"{pathname}/features_corr2.parquet")
print(f"Saved features_corr2.parquet")

# %%
