import pandas as pd
import numpy as np
from pympler import asizeof

# ============================================================
# 1) Filter option data
# ============================================================

# Keep only rows after 2011-01-01
optiondata = optiondata.query("date >= '2011-01-01'").copy()

# ============================================================
# 2) Rename columns for consistency
# ============================================================

optiondata.columns = [
    "Date", "SecurityID", "Date2", "Symbol", "SymbolFlag", "Strike",
    "Expiration", "CallPut", "BestBid", "BestOffer", "LastTradeDate",
    "Volume", "OpenInterest", "SpecialSettlement", "ImpliedVolatility",
    "Delta", "Gamma", "Vega", "Theta", "OptionID", "AdjustmentFactor",
    "AMSettlement", "ContractSize", "ExpiryIndicator"
]

# ============================================================
# 3) Convert date columns and compute time to maturity
# ============================================================

optiondata["Date"] = pd.to_datetime(optiondata["Date"])
optiondata["Expiration"] = pd.to_datetime(optiondata["Expiration"])
optiondata["time2mat"] = (optiondata["Expiration"] - optiondata["Date"]).dt.days

# ============================================================
# 4) Handle Delta values (optional cleanup)
# ============================================================

# Example equivalent of:
# optiondata$Delta = ifelse(abs(optiondata$Delta) > 1, NA, tempoptions$Delta)
# Here we simply nullify |Delta| > 1
optiondata.loc[optiondata["Delta"].abs() > 1, "Delta"] = np.nan

# ============================================================
# 5) Drop unnecessary columns (optional)
# ============================================================

# Uncomment if you want to remove the same columns as in R
# drop_cols = [
#     "AdjustmentFactor", "Symbol", "SymbolFlag", "LastTradeDate",
#     "SpecialSettlement", "Volume", "OpenInterest", "Gamma", "Theta"
# ]
# optiondata = optiondata.drop(columns=drop_cols, errors="ignore")

# ============================================================
# 6) Inspect memory size
# ============================================================

deep_size = asizeof.asizeof(optiondata) / 1e9
print(f"Size of optiondata (using pympler): {deep_size:.3f} GB")
