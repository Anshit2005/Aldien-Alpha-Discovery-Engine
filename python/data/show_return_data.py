import pandas as pd
from pathlib import Path

PATH = Path("data/processed/csi300_returns.parquet")

print("=" * 70)
print("CSI300 RETURN DATASET")
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

print("\nReturn statistics:")
print(df[["return_1d", "log_return_1d"]].describe())

print("\nReturn missingness:")
print(df[["return_1d", "log_return_1d"]].isna().sum())

print("\nCalendar gap statistics:")
print(df["calendar_gap_days"].describe())

print("\nDataset represents:")
print("Clean CSI300 market data with derived 1-day simple returns,")
print("log returns, and calendar-gap information.")