
#%%
import os
import pandas as pd
import numpy as np
import time
from datetime import date
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from lightgbm import LGBMRegressor
from sklearn.utils import resample
import catboost as cat
from catboost import Pool
# import pyfolio as pf
import polars as pl


import warnings
warnings.filterwarnings('ignore')

#%%
os.chdir(r'e:/laosongdata')
plXs = pl.read_parquet('alldataFeatures3.parquet')

#%%
for actdays in np.arange(1,2):
    market_index = 'SHSZ'
    input_filename=f"{market_index}_{actdays}_EODtestraw.parquet"
    print(input_filename)
    
    MLdata = pl.read_parquet(input_filename) # the data is w/o index, the data is already Z scored on classy and fillna with 0 for x
    MLdata = (
        MLdata
        .select(pl.col("ID"),pl.col('Date').cast(pl.Date), pl.col('y'), pl.col("classy"), pl.col("Janind"), pl.col("d1"))
        ) # the data is w/o index, the data is already Z scored on classy and fillna with 0 for x
    print(MLdata.columns)
    ###########################################
    MLdata = MLdata.join(plXs, on=['ID','Date'], how='inner')
    
    cols = MLdata.columns
    i0 = cols.index("O")
    i1 = cols.index("ret") + 1

    MLdata = MLdata.select(cols[:i0] + cols[i1:])


    y_name = "classy"

    cols = MLdata.columns
    x_cols = cols[4:]   # R: 5:94  → Python 0-based

    pred_blocks = []

    for y in range(2016, 2026):
        time1 = pl.date(y, 1, 1)
        time2 = pl.date(y + 1, 1, 1)

        train = MLdata.filter(pl.col("Date") < time1)
        test  = MLdata.filter(
            (pl.col("Date") >= time1) &
            (pl.col("Date") <  time2)
        )

        # ---- NumPy bridge for CatBoost ----
        Xtrain = train.select(x_cols).to_numpy()
        Ytrain = train.select(y_name).to_numpy().ravel()

        model = cat.CatBoostRegressor(
            logging_level="Silent",
            iterations=10000,
            task_type="GPU"
        )

        model.fit(Pool(Xtrain, Ytrain))

        # ---- predict ----
        Xtest = test.select(x_cols).to_numpy()
        Ytest = test.select(y_name).to_numpy().ravel()

        pred_y = model.predict(Pool(Xtest, Ytest))

        test_with_pred = test.with_columns(
            pl.Series("pred_y", pred_y)
        )

        pred_blocks.append(test_with_pred)
        print(f"Finished year {y}")


    bigtest = pl.concat(pred_blocks)


    filename = "catboost_" + input_filename
    bigtest.write_parquet(filename)






    
# %%
# ===============================
# Decile backtest + plots
# (Polars + plotnine)
# ===============================

from plotnine import (
    ggplot, aes, geom_bar, geom_line,
    labs, theme_minimal
)

ngroup = 10
long_decile = ngroup - 1
short_decile = 0

# -------------------------------------------------
# 1) Cross-sectional deciles (qcut replacement)
# -------------------------------------------------
bt = (
    bigtest
    .sort(["Date", "pred_y"], descending=[False, False])
    .with_columns(
        (
            pl.col("pred_y")
            .rank(method="average", descending=False)
            .over("Date")
            / pl.len().over("Date")
        ).alias("pct_rank")
    )
    .with_columns(
        (pl.col("pct_rank") * ngroup)
        .floor()
        .clip(0, ngroup - 1)
        .cast(pl.Int32)
        .alias("decile")
    )
    .with_columns(y=pl.col("y")-1)
)

# -------------------------------------------------
# 2) Daily decile returns
# -------------------------------------------------
daily_decile_ret = (
    bt
    .group_by(["Date", "decile"])
    .agg((pl.col("y")).mean().alias("ret"))
)


# -------------------------------------------------
# 3) Long–short strategy return (weight-based)
# -------------------------------------------------

# 1) Count names in each decile per Date
decile_counts = (
    bt
    .group_by(["Date", "decile"])
    .agg(pl.len().alias("n_decile"))
)

# 2) Join counts back to main table
bt_w = (
    bt
    .join(decile_counts, on=["Date", "decile"], how="left")
    .with_columns(
        pl.when(pl.col("decile") == long_decile)
          .then( 1.0 / pl.col("n_decile"))
          .when(pl.col("decile") == short_decile)
          .then(-1.0 / pl.col("n_decile"))
          .otherwise(0.0)
          .alias("weight")
    )
)

# 3) Daily portfolio return = sum_i weight_i * y_i
strategy = (
    bt_w
    .group_by("Date")
    .agg(
        (pl.col("weight") * pl.col("y")).sum().alias("strategy_ret")
    )
    .sort("Date")
    .with_columns(
        (1 + pl.col("strategy_ret")).cum_prod().alias("cum_ret")
    )
)


# -------------------------------------------------
# 4) Average return by decile
# -------------------------------------------------
decile_mean = (
    bt
    .group_by("decile")
    .agg(pl.col("y").mean().alias("avg_y"))
    .sort("decile")
)

# -------------------------------------------------
# 5) Plotting (plotnine / ggplot style)
# -------------------------------------------------

# Decile bar plot
p1 = (
    ggplot(decile_mean.to_pandas(), aes(x="factor(decile)", y="avg_y"))
    + geom_bar(stat="identity")
    + labs(
        title="Average Realized Return by Decile",
        x="Decile (0 = Short, 9 = Long)",
        y="Average y"
    )

)

# Cumulative long–short return
p2 = (
    ggplot(strategy.to_pandas(), aes(x="Date", y="cum_ret"))
    + geom_line()
    + labs(
        title="Cumulative Long–Short Strategy Returns",
        x="Date",
        y="Cumulative Return"
    )

)

p1.draw()
p2.draw()

# %%
topN = 100

thresh = 0.2

zz500 = bigtest.filter(pl.col("ID") == "SH000905").select(pl.col("Date"), pl.col("y").alias("y_zz500"))


bigtest2 = (
    bigtest
    # 1) inner join on Date, ID
    # .join(alldata, on=["Date", "ID"], how="inner")
    
    # 2) create derived distance variables
    .with_columns([
        ((pl.col("d1") - 1.1).abs() * 100).alias("d1_1"),
        ((pl.col("d1") - 1.05).abs() * 100).alias("d1_st"),
        ((pl.col("d1") - 1.2).abs() * 100).alias("d1_chy"),
    ])
    
    # 3) filter by threshold conditions
    .filter(
        (pl.col("d1_1")   >= thresh) &
        (pl.col("d1_st")  >= thresh * 1) &
        (pl.col("d1_chy") >= thresh)
    )
    .with_columns(y=pl.col("y") - 1)  # adjust y to be excess return (assuming original y is 1 + return
)


strategy = (
    bigtest2
    # 1) Cross-sectional rank by Date
    .with_columns(
        pl.col("pred_y")
          .rank(method="average", descending=True)
          .over("Date")
          .alias("cs_rank")
    )
    # 2) Keep top N per Date
    .filter(pl.col("cs_rank") <= topN)
    # 3) Equal-weight portfolio return = mean(y)
    .group_by("Date")
    .agg(
        pl.col("y").mean().alias("strategy_ret")
    )
    .sort("Date")
    .join(zz500, on="Date",how="inner")
    .with_columns(
        (pl.col("strategy_ret") - pl.col("y_zz500")+1).alias("strategy_ret2")
    )  
    # 4) Cumulative return
    .with_columns(
        (1 + pl.col("strategy_ret")).cum_prod().alias("cum_ret"),
        (1 + pl.col("strategy_ret2")).cum_prod().alias("cum_ret2")
    )
)



# -------------------------------------------------
# 5) Plotting (plotnine / ggplot style)
# -------------------------------------------------



# Cumulative long–short return
p2 = (
    ggplot(strategy.to_pandas(), aes(x="Date", y="cum_ret"))
    + geom_line()
    + labs(
        title="Cumulative Long only Strategy Returns",
        x="Date",
        y="Cumulative Return"
    )

)

p2
# %%
