#%%
import polars as pl
import pyarrow as pa
import pandas as pd
import gc

#%% get the ATM put IV
pathname = "E:/OMdata"

# ---- Load Parquet file ----

alldataEODs = (
    pl.scan_parquet(f"{pathname}/2025to2025ETFIVallinfo.parquet")
    .filter(pl.col("Days") == 30, pl.col("Delta") == -50)
    .collect()
)

alldataXs = (
    alldataEODs
    .lazy()
    .with_columns(
    pl.col("Date").cast(pl.Date),
    pl.col("SecurityID").cast(pl.Int64),
    # amount = pl.col("Volume")*pl.col("Price")
    )
    .rename({"ImpliedVolatility": "IV",
             "SecurityID": "ID"})
    .select(["ID","IV", "Date"])
    .filter(pl.col("Date") >= pl.date(1019, 12, 1))
    .collect()

)

print(alldataXs.head())

# base = alldataXs.select("ID","Date","age","ret").sort(["ID", "Date"]) # need this one for the top K

#remove alldataEODs to free memory
del alldataEODs
gc.collect()

#%% the price part
# ---- Load Parquet file ----
alldataEODs = (
    pl.read_parquet(f"{pathname}/adPrices3OHLC.parquet")
)

# ---- Construct alldataXs ----

alldataPXs = (
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

print(alldataPXs.head())


#remove alldataEODs to free memory
del alldataEODs
gc.collect()
# %%

#inner join on ID and Date for alldataPXs and alldataXs
alldata = (
    alldataPXs
    .join(alldataXs, on=["ID", "Date"], how="inner")
)

# %% create target variable
# lookahead horizon (trading rows) for target y
lookahead_days = 5
rv_windows = [5, 20, 60]

alldata = (
    alldata
    .lazy()
    .sort(["ID", "Date"])
    .with_columns([
        (
            pl.col("IV").shift(-lookahead_days).over("ID") / pl.col("IV") - 1
        ).alias("y"),
        pl.col("Date").dt.weekday().alias("weekdays"),
        (
            pl.col("ret").pow(2)
            .reverse().over("ID")
            .rolling_mean(window_size=5)
            .reverse().over("ID")
            .shift(-1).over("ID")
            .sqrt()
            * (252 ** 0.5)
        ).alias("rv_fwd5"),
        (
            pl.col("ret").pow(2)
            .reverse().over("ID")
            .rolling_mean(window_size=20)
            .reverse().over("ID")
            .shift(-1).over("ID")
            .sqrt()
            * (252 ** 0.5)
        ).alias("rv_fwd20"),
        (
            pl.col("ret").pow(2)
            .reverse().over("ID")
            .rolling_mean(window_size=60)
            .reverse().over("ID")
            .shift(-1).over("ID")
            .sqrt()
            * (252 ** 0.5)
        ).alias("rv_fwd60"),
    ])
    .with_columns([
        (pl.col("IV") - pl.col("rv_fwd5")).alias("y_iv_minus_rv5"),
        (pl.col("IV") - pl.col("rv_fwd20")).alias("y_iv_minus_rv20"),
        (pl.col("IV") - pl.col("rv_fwd60")).alias("y_iv_minus_rv60"),
    ])
    .collect()
)

print(alldata.head())
# %%
