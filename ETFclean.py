import pandas as pd
import numpy as np

# ============================================================
# 1) Filter and rename base price data
# ============================================================
os.chdir("e:/jiaqifiles")

# Filter prices from 2011-01-01 onward
prices = (
    sprices
    .query("date >= '2011-01-01'")
    .copy()
)

# Check difference with ivyDate (assuming ivyDate exists)
check = (
    prices
    .assign(Datediff=lambda df: pd.to_datetime(df["ivyDate"]) - pd.to_datetime(df["date"]))
    [["date", "Datediff"]]
)
print(check["Datediff"].astype("timedelta64[D]").describe())  # summary equivalent

# Rename columns
prices.columns = [
    "Date", "SecurityID", "Date2", "BidLow", "AskHigh", "ClosePrice",
    "Volume", "TotalReturn", "AdjustmentFactor", "OpenPrice",
    "SharesOutstanding", "AdjustmentFactor2"
]

# Drop unused columns and rename remaining ones
prices = (
    prices
    .drop(columns=["BidLow", "AskHigh", "OpenPrice"])
    .rename(columns={
        "ClosePrice": "Price",
        "AdjustmentFactor2": "Adj",
        "AdjustmentFactor": "splitAdj"
    })
)

# Inspect one SecurityID
check = prices.query("SecurityID == 121075")
print(check.head())

# ============================================================
# 2) Compute market cap and unique IDs
# ============================================================

C_prices = prices.assign(cap=lambda df: df["Price"] * df["SharesOutstanding"])
IDs = C_prices["SecurityID"].unique()

# Replace with fixed ID list if needed
IDs = [
    106445, 109820, 107899, 103823, 110015, 116959, 110008, 110009,
    110010, 110011, 110012, 110013, 110014, 127107, 122392, 116070,
    127724, 151482, 151483, 100479, 126776
]

# ============================================================
# 3) Extract SPX (benchmark security)
# ============================================================

SPX = prices.query("SecurityID == 109820").copy()

# Convert Date to datetime and ensure order
SPX["Date"] = pd.to_datetime(SPX["Date"])
SPX = SPX.sort_values("Date").reset_index(drop=True)

# Keep only Date and Price for simplicity (4th column originally)
benchmark = SPX[["Date", "Price"]].copy()
benchmark = benchmark.query("Date >= '1996-01-01'")
SPX = benchmark.rename(columns={"Price": "tempSPX"}).reset_index(drop=True)

# ============================================================
# 4) Process each SecurityID
# ============================================================

alldata = []
for i, sec_id in enumerate(IDs, 1):
    temp = C_prices.query("SecurityID == @sec_id").copy()
    temp = temp.sort_values("Date").reset_index(drop=True)
    
    if temp.empty:
        continue

    end_adj = temp["Adj"].iloc[-1]
    beg_price = temp["Price"].iloc[0]

    # Remove invalid prices
    temp = temp.query("Price > 0").copy()
    temp["TotalReturn"] = np.where(temp["TotalReturn"] <= -99, 0, temp["TotalReturn"])
    temp["Adj_prices"] = temp["Adj"] * temp["Price"] / end_adj  # back-adjusted

    if len(temp) < 5:
        continue

    beg_price = beg_price / (temp["TotalReturn"].iloc[0] + 1)
    temp["cumret"] = (1 + temp["TotalReturn"]).cumprod()
    temp["Adj_prices2"] = temp["cumret"] * beg_price  # forward-adjusted
    temp["ratio"] = temp["Adj_prices2"] / temp["Adj_prices"]

    temp["Date"] = pd.to_datetime(temp["Date"])
    start, end = temp["Date"].iloc[0], temp["Date"].iloc[-1]

    tempSPX = SPX.query("Date >= @start and Date <= @end").copy()
    temp2 = pd.merge(temp, tempSPX, on="Date", how="right")

    # Move Date to front for clarity
    cols = ["Date"] + [c for c in temp2.columns if c != "Date"]
    temp2 = temp2[cols]

    alldata.append(temp2)

    if i % 100 == 0:
        print(f"Processed {i} securities")

# ============================================================
# 5) Combine all data and finalize adjustments
# ============================================================

adPrices = pd.concat(alldata, ignore_index=True)

# Market cap and size rank per Date
adPrices["cap"] = adPrices["Price"] * adPrices["SharesOutstanding"]
adPrices["size_rank"] = adPrices.groupby("Date")["cap"].rank(ascending=False)

# Replace NA or infinite adjusted prices
badloc = adPrices["Adj_prices"].isna() | np.isinf(adPrices["Adj_prices"])
adPrices.loc[badloc, "Adj_prices"] = adPrices.loc[badloc, "Adj_prices2"]

# ============================================================
# 6) Export results
# ============================================================

# Save in multiple formats
adPrices.to_parquet("adPrices2ETFadj.parquet", index=False)
adPrices.to_csv("adPrices2ETFadj.csv", index=False)
print("✅ Exported adPrices2ETFadj.parquet and adPrices2ETFadj.csv")

