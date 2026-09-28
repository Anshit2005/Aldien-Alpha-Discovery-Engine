from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

UNIVERSE_PATH = Path(
    "data/processed/csi300_universe.parquet"
)

MARKET_DATA_PATH = Path(
    "data/raw/qlib_csi300/daily_pv.h5"
)

OUTPUT_PATH = Path(
    "data/processed/csi300_market_data.parquet"
)


# ============================================================
# 1. LOAD CSI300 UNIVERSE
# ============================================================

print("Loading CSI300 universe...")

universe = pd.read_parquet(UNIVERSE_PATH)

universe["datetime"] = pd.to_datetime(universe["datetime"])

print(f"Universe rows: {len(universe):,}")
print(f"Unique instruments: {universe['instrument'].nunique():,}")
print(f"Trading days: {universe['datetime'].nunique():,}")


# ============================================================
# 2. LOAD MARKET DATA
# ============================================================

print("\nLoading market data...")

market_data = pd.read_hdf(
    MARKET_DATA_PATH,
    key="/data"
)

print(f"Market-data rows: {len(market_data):,}")

# MultiIndex -> columns
market_data = market_data.reset_index()

market_data["datetime"] = pd.to_datetime(
    market_data["datetime"]
)

print(f"Market instruments: {market_data['instrument'].nunique():,}")
print(f"Market trading days: {market_data['datetime'].nunique():,}")


# ============================================================
# 3. FILTER MARKET DATA USING CSI300 UNIVERSE
# ============================================================

print("\nFiltering market data using CSI300 membership...")

csi300_market_data = market_data.merge(
    universe,
    on=["datetime", "instrument"],
    how="inner"
)

print(
    f"Filtered rows: "
    f"{len(csi300_market_data):,}"
)


# ============================================================
# 4. SELECT / ORDER COLUMNS
# ============================================================

columns = [
    "datetime",
    "instrument",
    "$open",
    "$high",
    "$low",
    "$close",
    "$volume",
    "$factor",
]

csi300_market_data = csi300_market_data[columns]

csi300_market_data = csi300_market_data.sort_values(
    ["datetime", "instrument"]
).reset_index(drop=True)


# ============================================================
# 5. VALIDATION
# ============================================================

print("\n========== CSI300 MARKET DATA VALIDATION ==========")

print(f"Shape: {csi300_market_data.shape}")

print(
    f"Unique instruments: "
    f"{csi300_market_data['instrument'].nunique():,}"
)

print(
    f"Unique dates: "
    f"{csi300_market_data['datetime'].nunique():,}"
)

print("\nDate range:")
print(csi300_market_data["datetime"].min())
print(csi300_market_data["datetime"].max())


# Constituents per day
daily_counts = (
    csi300_market_data
    .groupby("datetime")["instrument"]
    .nunique()
)

print("\nConstituents per day:")
print(daily_counts.describe())


# Duplicate check
duplicates = csi300_market_data.duplicated(
    subset=["datetime", "instrument"]
).sum()

print(f"\nDuplicate (datetime, instrument) pairs: {duplicates}")


# Missing values
print("\nMissing values:")
print(csi300_market_data.isna().sum())


# ============================================================
# 6. SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

csi300_market_data.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(f"\nSaved market data to: {OUTPUT_PATH}")