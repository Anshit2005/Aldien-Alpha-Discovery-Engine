import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

MEMBERSHIP_PATH = Path(
    "data/raw/qlib_csi300/instruments/csi300.txt"
)

MARKET_DATA_PATH = Path(
    "data/raw/qlib_csi300/daily_pv.h5"
)

OUTPUT_PATH = Path(
    "data/processed/csi300_universe.parquet"
)


# ============================================================
# 1. Load CSI300 membership history
# ============================================================

print("Loading CSI300 membership history...")

membership = pd.read_csv(
    MEMBERSHIP_PATH,
    sep=r"\s+",
    header=None,
    names=["instrument", "start_date", "end_date"],
)

membership["start_date"] = pd.to_datetime(
    membership["start_date"]
)

membership["end_date"] = pd.to_datetime(
    membership["end_date"]
)

print(f"Membership records: {len(membership)}")
print(f"Unique instruments: {membership['instrument'].nunique()}")


# ============================================================
# 2. Load trading dates from market data
# ============================================================

print("\nLoading trading dates from market data...")

market_index = pd.read_hdf(
    MARKET_DATA_PATH,
    key="data"
)

trading_dates = (
    pd.DatetimeIndex(
        market_index.index.get_level_values("datetime")
    )
    .unique()
    .sort_values()
)

print(f"Trading days: {len(trading_dates)}")
print(f"Start: {trading_dates.min().date()}")
print(f"End: {trading_dates.max().date()}")


# ============================================================
# 3. Restrict membership data to our research period
# ============================================================

research_start = trading_dates.min()
research_end = trading_dates.max()

membership = membership[
    (membership["end_date"] >= research_start)
    & (membership["start_date"] <= research_end)
].copy()

print(
    f"\nMembership records overlapping research period: "
    f"{len(membership)}"
)


# ============================================================
# 4. Build date × instrument universe
# ============================================================

print("\nBuilding date-aware CSI300 universe...")

universe_parts = []

for date in trading_dates:

    active = membership[
        (membership["start_date"] <= date)
        & (membership["end_date"] >= date)
    ]

    instruments = active["instrument"].unique()

    daily = pd.DataFrame({
        "datetime": date,
        "instrument": instruments,
    })

    universe_parts.append(daily)


universe = pd.concat(
    universe_parts,
    ignore_index=True
)


# ============================================================
# 5. Sort and clean
# ============================================================

universe = universe.sort_values(
    ["datetime", "instrument"]
).reset_index(drop=True)


universe = universe.drop_duplicates(
    ["datetime", "instrument"]
)


# ============================================================
# 6. Validation
# ============================================================

daily_counts = (
    universe
    .groupby("datetime")["instrument"]
    .nunique()
)

print("\n========== UNIVERSE VALIDATION ==========")

print(
    f"Trading days: {len(daily_counts)}"
)

print(
    f"Universe rows: {len(universe):,}"
)

print(
    f"Minimum constituents/day: "
    f"{daily_counts.min()}"
)

print(
    f"Maximum constituents/day: "
    f"{daily_counts.max()}"
)

print(
    f"Mean constituents/day: "
    f"{daily_counts.mean():.2f}"
)

print(
    f"Median constituents/day: "
    f"{daily_counts.median():.2f}"
)


# ============================================================
# 7. Check for unexpected constituent counts
# ============================================================

invalid_dates = daily_counts[daily_counts != 300]

if len(invalid_dates) == 0:

    print("\n✓ Every trading day has exactly 300 constituents.")

else:

    print(
        f"\nWARNING: {len(invalid_dates)} "
        f"dates do not have exactly 300 constituents."
    )

    print("\nFirst 20 problematic dates:")
    print(invalid_dates.head(20))


# ============================================================
# 8. Save
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

universe.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved universe to: {OUTPUT_PATH}"
)