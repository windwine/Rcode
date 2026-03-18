#%%
# =============================================================================
# pl_factorgen_beta2.py  —  Group BETA2: 非对称 Beta
#
# 输入 : alldataFeatures3.parquet  (读 ID, Date, age, ret, market_ret)
# 输出 : features_beta2.parquet    (ID, Date + 9 列)
#
# 新列 (windows = 20, 60, 250):
#   beta_up_{w}    仅上涨市场日 (market_ret>0) 的滚动 beta
#   beta_down_{w}  仅下跌市场日 (market_ret<0) 的滚动 beta
#   beta_asym_{w}  beta_up / beta_down, 非对称程度
#
# 用 .rolling(index_column="age") API 保证按上市天数对齐
# 在 .agg() 内用 .filter() 分别计算上行/下行子集的 beta
# =============================================================================

import polars as pl
import gc

#%%
pathname = "d:/laosongdata"

df = pl.read_parquet(
    f"{pathname}/alldataFeatures3.parquet",
    columns=["ID", "Date", "age", "ret", "market_ret"]
)
base = df.sort(["ID", "Date"]).with_columns(pl.col("age").cast(pl.Int64))
del df; gc.collect()
print(f"Loaded: {base.shape}")

#%%
windows = [20, 60, 250]
feat_frames = []

for w in windows:
    feats_w = (
        base
        .rolling(index_column="age", period=f"{w}i", group_by="ID")
        .agg([
            pl.col("Date").last(),

            # ── beta_up: 仅 market_ret > 0 的观测 ──
            # cov(ret, mkt) = E[ret*mkt] - E[ret]*E[mkt]
            # var(mkt)      = E[mkt^2]   - E[mkt]^2
            (
                (pl.col("ret").filter(pl.col("market_ret") > 0) *
                 pl.col("market_ret").filter(pl.col("market_ret") > 0)).mean()
                - pl.col("ret").filter(pl.col("market_ret") > 0).mean()
                * pl.col("market_ret").filter(pl.col("market_ret") > 0).mean()
            ).alias(f"_cov_up_{w}"),
            (
                (pl.col("market_ret").filter(pl.col("market_ret") > 0) ** 2).mean()
                - pl.col("market_ret").filter(pl.col("market_ret") > 0).mean() ** 2
            ).alias(f"_var_up_{w}"),

            # ── beta_down: 仅 market_ret < 0 的观测 ──
            (
                (pl.col("ret").filter(pl.col("market_ret") < 0) *
                 pl.col("market_ret").filter(pl.col("market_ret") < 0)).mean()
                - pl.col("ret").filter(pl.col("market_ret") < 0).mean()
                * pl.col("market_ret").filter(pl.col("market_ret") < 0).mean()
            ).alias(f"_cov_down_{w}"),
            (
                (pl.col("market_ret").filter(pl.col("market_ret") < 0) ** 2).mean()
                - pl.col("market_ret").filter(pl.col("market_ret") < 0).mean() ** 2
            ).alias(f"_var_down_{w}"),
        ])
        .with_columns([
            (pl.col(f"_cov_up_{w}")   / pl.col(f"_var_up_{w}")).alias(f"beta_up_{w}"),
            (pl.col(f"_cov_down_{w}") / pl.col(f"_var_down_{w}")).alias(f"beta_down_{w}"),
        ])
        .with_columns(
            (pl.col(f"beta_up_{w}") / pl.col(f"beta_down_{w}")).alias(f"beta_asym_{w}")
        )
        .select(["ID", "Date", f"beta_up_{w}", f"beta_down_{w}", f"beta_asym_{w}"])
    )
    feat_frames.append(feats_w)
    print(f"  window {w} done")

#%%
result = feat_frames[0]
for f in feat_frames[1:]:
    result = result.join(f, on=["ID", "Date"], how="inner")

print(f"Output shape: {result.shape}")
result.write_parquet(f"{pathname}/features_beta2.parquet")
print(f"Saved features_beta2.parquet")

# %%
