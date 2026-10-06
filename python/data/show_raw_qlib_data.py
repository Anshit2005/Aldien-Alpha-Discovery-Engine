import pandas as pd
from pathlib import Path

PATH = Path("data/raw/qlib_csi300/daily_pv.h5")

print("=" * 70)
print("RAW QLIB MARKET DATA")
print("=" * 70)

df = pd.read_hdf(PATH, key="/data")

print(f"\nFile: {PATH}")
print(f"Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print(f"\nUnique instruments: {df.index.get_level_values('instrument').nunique():,}")
print(f"Trading days: {df.index.get_level_values('datetime').nunique():,}")

print("\nDate range:")
print(f"Start: {df.index.get_level_values('datetime').min()}")
print(f"End:   {df.index.get_level_values('datetime').max()}")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset represents:")
print("Raw Qlib daily market data containing OHLCV + factor")
print("information for thousands of instruments.")