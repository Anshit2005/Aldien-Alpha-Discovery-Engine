from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "data/processed/csi300_market_data_clean.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/csi300_returns.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

DATE_COL = "datetime"
INSTRUMENT_COL = "instrument"

CLOSE_COL = "$close"


# ============================================================
# 1. LOAD CLEAN MARKET DATA
# ============================================================

print("Loading clean CSI300 market data...")

df = pd.read_parquet(INPUT_PATH)

df[DATE_COL] = pd.to_datetime(df[DATE_COL])

print(f"Rows: {len(df):,}")
print(f"Unique instruments: {df[INSTRUMENT_COL].nunique():,}")
print(f"Trading days: {df[DATE_COL].nunique():,}")


# ============================================================
# 2. SORT
# ============================================================

df = df.sort_values(
    [INSTRUMENT_COL, DATE_COL]
).reset_index(drop=True)


# ============================================================
# 3. CHECK DUPLICATES
# ============================================================

duplicates = df.duplicated(
    subset=[DATE_COL, INSTRUMENT_COL]
).sum()

print("\n========== DUPLICATE CHECK ==========")
print(
    f"Duplicate (datetime, instrument) pairs: "
    f"{duplicates:,}"
)

if duplicates > 0:
    raise ValueError(
        "Duplicate datetime/instrument pairs detected."
    )


# ============================================================
# 4. PREVIOUS OBSERVATION
# ============================================================

grouped = df.groupby(
    INSTRUMENT_COL,
    sort=False
)

df["prev_close"] = grouped[CLOSE_COL].shift(1)
df["prev_datetime"] = grouped[DATE_COL].shift(1)


# ============================================================
# 5. GAP DETECTION
# ============================================================

# Number of calendar days between observations.
df["calendar_gap_days"] = (
    df[DATE_COL] - df["prev_datetime"]
).dt.days


# A return is valid only when the previous observation
# is from the immediately preceding trading day globally.
#
# We construct the global trading calendar first.

trading_dates = (
    df[DATE_COL]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

previous_trading_date = (
    trading_dates.shift(1)
)

previous_date_map = pd.Series(
    previous_trading_date.values,
    index=trading_dates.values
)

df["expected_prev_datetime"] = (
    df[DATE_COL].map(previous_date_map)
)


df["valid_return"] = (
    df["prev_datetime"]
    == df["expected_prev_datetime"]
)


# ============================================================
# 6. CALCULATE RETURNS
# ============================================================

df["return_1d"] = np.nan
df["log_return_1d"] = np.nan

valid = (
    df["valid_return"]
    & df["prev_close"].notna()
    & (df["prev_close"] > 0)
    & (df[CLOSE_COL] > 0)
)

df.loc[valid, "return_1d"] = (
    df.loc[valid, CLOSE_COL]
    / df.loc[valid, "prev_close"]
    - 1.0
)

df.loc[valid, "log_return_1d"] = (
    np.log(
        df.loc[valid, CLOSE_COL]
        / df.loc[valid, "prev_close"]
    )
)


# ============================================================
# 7. VALIDATION
# ============================================================

print("\n========== RETURN VALIDATION ==========")

total_rows = len(df)

valid_returns = df["return_1d"].notna().sum()

invalid_returns = total_rows - valid_returns

print(f"Total rows: {total_rows:,}")
print(f"Valid 1D returns: {valid_returns:,}")
print(f"No return available: {invalid_returns:,}")

print(
    "\nReturn statistics:"
)

print(
    df["return_1d"].describe()
)


# ============================================================
# 8. GAP STATISTICS
# ============================================================

print("\n========== GAP ANALYSIS ==========")

gap_rows = (
    df["prev_datetime"].notna()
    & ~df["valid_return"]
)

print(
    f"Rows following a data gap: "
    f"{gap_rows.sum():,}"
)

print(
    "\nCalendar gap distribution:"
)

print(
    df.loc[
        df["prev_datetime"].notna(),
        "calendar_gap_days"
    ].describe()
)


# ============================================================
# 9. RETURN SANITY CHECK
# ============================================================

print("\n========== RETURN SANITY ==========")

extreme_returns = (
    df["return_1d"].abs() > 1.0
).sum()

print(
    f"Returns with absolute value > 100%: "
    f"{extreme_returns:,}"
)

if extreme_returns > 0:

    print("\nSample extreme returns:")

    print(
        df.loc[
            df["return_1d"].abs() > 1.0,
            [
                DATE_COL,
                INSTRUMENT_COL,
                CLOSE_COL,
                "prev_close",
                "return_1d",
            ],
        ].head(20)
    )


# ============================================================
# 10. REMOVE TEMPORARY COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "prev_close",
        "prev_datetime",
        "calendar_gap_days",
        "expected_prev_datetime",
        "valid_return",
    ]
)


# ============================================================
# 11. SORT BACK TO DATE → INSTRUMENT
# ============================================================

df = df.sort_values(
    [DATE_COL, INSTRUMENT_COL]
).reset_index(drop=True)


# ============================================================
# 12. FINAL VALIDATION
# ============================================================

print("\n========== FINAL DATASET ==========")

print(f"Shape: {df.shape}")

print(
    f"Unique instruments: "
    f"{df[INSTRUMENT_COL].nunique():,}"
)

print(
    f"Unique dates: "
    f"{df[DATE_COL].nunique():,}"
)

print(
    f"Date range: "
    f"{df[DATE_COL].min()} "
    f"to "
    f"{df[DATE_COL].max()}"
)

print(
    "\nReturn missingness:"
)

print(
    df[
        ["return_1d", "log_return_1d"]
    ].isna().sum()
)


# ============================================================
# 13. SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved return dataset to: "
    f"{OUTPUT_PATH}"
)

print("\n========== RETURN BUILD COMPLETE ==========")