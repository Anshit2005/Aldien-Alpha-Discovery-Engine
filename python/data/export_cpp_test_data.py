from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "csi300_research_dataset.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cpp_test_data.csv"
)


# --------------------------------------------------
# Load research dataset
# --------------------------------------------------

df = pd.read_parquet(INPUT_FILE)

df["datetime"] = pd.to_datetime(df["datetime"])


# --------------------------------------------------
# Select a small deterministic subset
# --------------------------------------------------

# First 20 instruments
instruments = (
    df["instrument"]
    .drop_duplicates()
    .sort_values()
    .head(20)
)

df = df[
    df["instrument"].isin(instruments)
].copy()


# First 500 trading dates
dates = (
    df["datetime"]
    .drop_duplicates()
    .sort_values()
    .head(500)
)

df = df[
    df["datetime"].isin(dates)
].copy()


# --------------------------------------------------
# Keep only columns required by C++ evaluator
# --------------------------------------------------

df = df[
    [
        "datetime",
        "instrument",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
].copy()


# --------------------------------------------------
# Sort exactly as C++ expects
#
# instrument → datetime
# --------------------------------------------------

df = df.sort_values(
    ["instrument", "datetime"]
).reset_index(drop=True)


# --------------------------------------------------
# Export
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("Exported:", OUTPUT_FILE)
print("Rows:", len(df))
print("Instruments:", df["instrument"].nunique())
print("Dates:", df["datetime"].nunique())
print("Date range:")
print(df["datetime"].min(), "→", df["datetime"].max())