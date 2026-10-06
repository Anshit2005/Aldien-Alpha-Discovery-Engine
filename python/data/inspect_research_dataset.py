"""
Inspect the research dataset produced by build_research_dataset.py.
"""

from pathlib import Path
import pandas as pd


PATH = Path("data/processed/csi300_research_dataset.parquet")

if not PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {PATH}")

df = pd.read_parquet(PATH)

print("=" * 90)
print("CSI300 RESEARCH DATASET")
print("=" * 90)

print("\nINFO")
print("-" * 90)
print(f"Rows:               {len(df):,}")
print(f"Columns:            {len(df.columns)}")
print(f"Unique instruments: {df['instrument'].nunique():,}")
print(f"Unique dates:       {df['datetime'].nunique():,}")
print(f"Date range:         {df['datetime'].min()} -> {df['datetime'].max()}")

print("\nSPLIT COUNTS")
print("-" * 90)
print(df["split"].value_counts().sort_index().to_string())

print("\nTARGET AVAILABILITY")
print("-" * 90)

for h in [1, 5, 20, 30]:
    col = f"forward_return_{h}d"
    valid = df[col].notna().sum()
    total = len(df)

    print(
        f"{h:2d}d forward return: "
        f"{valid:,}/{total:,} valid ({valid / total:.2%})"
    )

print("\nTARGET STATISTICS")
print("-" * 90)

target_cols = [f"forward_return_{h}d" for h in [1, 5, 20, 30]]
print(df[target_cols].describe().T.to_string())

print("\nHEAD")
print("-" * 90)

display_cols = [
    "datetime",
    "instrument",
    "close",
    "split",
    "target_date_1d",
    "forward_return_1d",
    "target_date_20d",
    "forward_return_20d",
    "target_date_30d",
    "forward_return_30d",
]

print(df[display_cols].head(20).to_string(index=False))

print("\nCHECK FOR TARGET LEAKAGE")
print("-" * 90)

for h in [1, 5, 20, 30]:
    target_split = f"target_split_{h}d"

    leakage = (
        df["split"].isin(["train", "validation", "test", "holdout"])
        & df[target_split].notna()
        & (df["split"] != df[target_split])
    ).sum()

    print(f"{h:2d}d leakage rows: {leakage}")

print("\nDONE")
