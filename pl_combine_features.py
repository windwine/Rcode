#%%
# =============================================================================
# pl_combine_features.py  —  增量因子合并器
#
# 用法:
#   1. 修改下方 active_groups 选择要合并的组
#   2. 修改 output_file 指定输出文件名
#   3. 运行整个脚本
#
# 每个 Group 脚本先独立运行生成对应 features_xxx.parquet，再用本脚本合并
# =============================================================================

import polars as pl
import gc

#%%
pathname = "d:/laosongdata"

# ── 在这里配置 ──────────────────────────────────────────────────────────────
active_groups = ["cal", "vol", "ibs", "gap", "beta2", "corr2", "ma"]
output_file   = "alldataFeatures3_all.parquet"
# ────────────────────────────────────────────────────────────────────────────

# Group 注册表
#   file : 增量 parquet 文件名
#   on   : join key (date-level 用 "Date", stock-level 用 ["ID","Date"])
REGISTRY = {
    "cal":   {"file": "features_cal.parquet",   "on": "Date"},
    "vol":   {"file": "features_vol.parquet",   "on": ["ID", "Date"]},
    "ibs":   {"file": "features_ibs.parquet",   "on": ["ID", "Date"]},
    "gap":   {"file": "features_gap.parquet",   "on": ["ID", "Date"]},
    "beta2": {"file": "features_beta2.parquet", "on": ["ID", "Date"]},
    "corr2": {"file": "features_corr2.parquet", "on": ["ID", "Date"]},
    "ma":    {"file": "features_ma.parquet",    "on": ["ID", "Date"]},
}

#%%
print(f"Base: alldataFeatures3.parquet")
base = pl.read_parquet(f"{pathname}/alldataFeatures3.parquet")
print(f"  shape: {base.shape}")

for g in active_groups:
    cfg = REGISTRY[g]
    feat = pl.read_parquet(f"{pathname}/{cfg['file']}")
    key  = cfg["on"]
    n_key = 1 if isinstance(key, str) else len(key)
    n_new = feat.width - n_key
    base = base.join(feat, on=key, how="left")
    print(f"  + {g:<6s}  {cfg['file']:<30s}  +{n_new} cols  → total {base.width}")
    del feat; gc.collect()

#%%
print(f"\nFinal shape: {base.shape}")
base.write_parquet(f"{pathname}/{output_file}")
print(f"Saved: {pathname}/{output_file}")

# %%
