from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "data/processed/csi300_market_data.parquet"
)

UNIVERSE_PATH = Path(
    "data/processed/csi300_universe.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/csi300_market_data_clean.parquet"
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading CSI300 market data...")

market = pd.read_parquet(INPUT_PATH)

print(f"Input rows: {len(market):,}")
print(f"Input instruments: {market['instrument'].nunique():,}")
print(f"Input dates: {market['datetime'].nunique():,}")


print("\nLoading CSI300 universe...")

universe = pd.read_parquet(UNIVERSE_PATH)

print(f"Universe rows: {len(universe):,}")


# ============================================================
# 2. STANDARDIZE DATETIME
# ============================================================

market["datetime"] = pd.to_datetime(market["datetime"])
universe["datetime"] = pd.to_datetime(universe["datetime"])


# ============================================================
# 3. CHECK DUPLICATES
# ============================================================

print("\n========== DUPLICATE CHECK ==========")

duplicates = market.duplicated(
    subset=["datetime", "instrument"]
).sum()

print(f"Duplicate (datetime, instrument) pairs: {duplicates:,}")

if duplicates > 0:
    raise ValueError(
        "Duplicate datetime/instrument pairs found."
    )


# ============================================================
# 4. CHECK OHLCV MISSINGNESS
# ============================================================

OHLCV_COLUMNS = [
    "$open",
    "$high",
    "$low",
    "$close",
    "$volume",
]

print("\n========== MISSING VALUE CHECK ==========")

missing_mask = market[OHLCV_COLUMNS].isna().any(axis=1)

missing_rows = missing_mask.sum()
complete_rows = (~missing_mask).sum()

print(f"Complete OHLCV rows: {complete_rows:,}")
print(f"Missing OHLCV rows: {missing_rows:,}")
print(
    f"Missing percentage: "
    f"{missing_rows / len(market) * 100:.4f}%"
)


# ============================================================
# 5. REMOVE INCOMPLETE OHLCV ROWS
# ============================================================

print("\nRemoving rows with incomplete OHLCV...")

clean = market.loc[~missing_mask].copy()

print(f"Rows after cleaning: {len(clean):,}")


# ============================================================
# 6. VERIFY NO OHLCV MISSING VALUES REMAIN
# ============================================================

remaining_missing = clean[OHLCV_COLUMNS].isna().sum()

print("\n========== POST-CLEANING MISSINGNESS ==========")

print(remaining_missing)

if remaining_missing.sum() > 0:
    raise ValueError(
        "Missing OHLCV values remain after cleaning."
    )


# ============================================================
# 7. VERIFY UNIVERSE MEMBERSHIP
# ============================================================

print("\n========== UNIVERSE MEMBERSHIP CHECK ==========")

universe_pairs = set(
    zip(
        universe["datetime"],
        universe["instrument"]
    )
)

clean_pairs = list(
    zip(
        clean["datetime"],
        clean["instrument"]
    )
)

invalid_mask = [
    pair not in universe_pairs
    for pair in clean_pairs
]

invalid_count = sum(invalid_mask)

print(
    f"Rows outside CSI300 universe: "
    f"{invalid_count:,}"
)

if invalid_count > 0:
    raise ValueError(
        "Clean dataset contains rows outside CSI300 universe."
    )


# ============================================================
# 8. CHECK BASIC MARKET DATA SANITY
# ============================================================

print("\n========== MARKET DATA SANITY ==========")

invalid_ohlc = (
    (clean["$high"] < clean["$low"]) |
    (clean["$open"] <= 0) |
    (clean["$high"] <= 0) |
    (clean["$low"] <= 0) |
    (clean["$close"] <= 0) |
    (clean["$volume"] < 0)
)

invalid_ohlc_count = invalid_ohlc.sum()

print(
    f"Rows with invalid OHLCV values: "
    f"{invalid_ohlc_count:,}"
)

if invalid_ohlc_count > 0:
    print("\nSample invalid rows:")
    print(
        clean.loc[
            invalid_ohlc,
            [
                "datetime",
                "instrument",
                *OHLCV_COLUMNS,
            ]
        ].head(20)
    )

    raise ValueError(
        "Invalid OHLCV values detected."
    )


# ============================================================
# 9. SORT DATA
# ============================================================

clean = clean.sort_values(
    ["datetime", "instrument"]
).reset_index(drop=True)


# ============================================================
# 10. FINAL VALIDATION
# ============================================================

print("\n========== FINAL VALIDATION ==========")

print(f"Shape: {clean.shape}")

print(
    f"Unique instruments: "
    f"{clean['instrument'].nunique():,}"
)

print(
    f"Unique dates: "
    f"{clean['datetime'].nunique():,}"
)

print(
    f"Date range: "
    f"{clean['datetime'].min()} "
    f"to "
    f"{clean['datetime'].max()}"
)

print(
    "\nMissing values:"
)

print(
    clean[
        OHLCV_COLUMNS
    ].isna().sum()
)

print(
    "\nDuplicate pairs:",
    clean.duplicated(
        ["datetime", "instrument"]
    ).sum()
)


# ============================================================
# 11. SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

clean.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved clean dataset to: "
    f"{OUTPUT_PATH}"
)

print("\n========== CLEANING COMPLETE ==========")