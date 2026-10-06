import pandas as pd
from pathlib import Path

PATH = Path("data/processed/csi300_market_data_clean.parquet")

print("=" * 70)
print("CLEAN CSI300 MARKET DATA")
print("=" * 70)

df = pd.read_parquet(PATH)

print(f"\nFile: {PATH}")
print(f"Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print(f"\nUnique instruments: {df['instrument'].nunique():,}")
print(f"Trading days: {df['datetime'].nunique():,}")

print("\nDate range:")
print(f"Start: {df['datetime'].min()}")
print(f"End:   {df['datetime'].max()}")

print("\nFirst 5 rows:")
print(df.head())

print("\nOHLCV missing values:")
print(df[["$open", "$high", "$low", "$close", "$volume"]].isna().sum())

print("\nDuplicate (datetime, instrument) pairs:")
print(df.duplicated(["datetime", "instrument"]).sum())

print("\nDataset represents:")
print("Historical CSI300 constituents with complete OHLCV observations.")
print("The raw Qlib market data has been filtered using the")
print("date-aware CSI300 universe and incomplete OHLCV rows removed.")