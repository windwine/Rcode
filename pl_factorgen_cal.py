#%%
# =============================================================================
# pl_factorgen_cal.py  —  Group CAL: 日历效应
#
# 输入 : alldataFeatures3.parquet  (只读 Date 列)
# 输出 : features_cal.parquet      (Date + 7 列, date-level)
#
# 新列:
#   days_into_month   本月第几个交易日 (1,2,3,...)
#   days_to_month_end 距月末剩余交易日数 (含当天)
#   is_month_start    月首第1交易日     (0/1)
#   is_month_start2   月首第1-2交易日   (0/1)  ← S1/S2 hold-2
#   is_month_start4   月首第1-4交易日   (0/1)  ← S4 hold-4
#   is_month_end2     月末最后2交易日   (0/1)
#   is_month_end4     月末最后4交易日   (0/1)
# =============================================================================

import polars as pl

#%%
pathname = "d:/laosongdata"

unique_dates = (
    pl.read_parquet(f"{pathname}/alldataFeatures3.parquet", columns=["Date"])
    .unique()
    .sort("Date")
)

#%%
calendar_feats = (
    unique_dates
    .with_columns([
        pl.col("Date").dt.year().alias("_year"),
        pl.col("Date").dt.month().alias("_month"),
    ])
    .with_columns(
        pl.col("Date").cum_count().over(["_year", "_month"]).cast(pl.Int32).alias("days_into_month")
    )
    .with_columns(
        pl.col("days_into_month").max().over(["_year", "_month"]).alias("_total_in_month")
    )
    .with_columns(
        (pl.col("_total_in_month") - pl.col("days_into_month") + 1).alias("days_to_month_end")
    )
    .with_columns([
        (pl.col("days_into_month") == 1).cast(pl.Int8).alias("is_month_start"),
        (pl.col("days_into_month") <= 2).cast(pl.Int8).alias("is_month_start2"),
        (pl.col("days_into_month") <= 4).cast(pl.Int8).alias("is_month_start4"),
        (pl.col("days_to_month_end") <= 2).cast(pl.Int8).alias("is_month_end2"),
        (pl.col("days_to_month_end") <= 4).cast(pl.Int8).alias("is_month_end4"),
    ])
    .drop(["_year", "_month", "_total_in_month"])
)

#%%
calendar_feats.write_parquet(f"{pathname}/features_cal.parquet")
print(f"Saved features_cal.parquet: {calendar_feats.shape}")

# %%
