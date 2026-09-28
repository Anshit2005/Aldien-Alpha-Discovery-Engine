import pandas as pd

MARKET_DATA = "data/processed/csi300_market_data.parquet"

df = pd.read_parquet(MARKET_DATA)

missing = df[
    df[["$open", "$high", "$low", "$close", "$volume"]]
    .isna()
    .any(axis=1)
].copy()

print("========== MISSING OHLCV INVESTIGATION ==========")
print(f"Total missing rows: {len(missing):,}")
print(f"Unique instruments: {missing['instrument'].nunique()}")
print(f"Unique dates: {missing['datetime'].nunique()}")

print("\n========== BY YEAR ==========")
print(
    missing.groupby(missing["datetime"].dt.year)
    .size()
    .sort_index()
)

print("\n========== BY INSTRUMENT ==========")
print(
    missing.groupby("instrument")
    .size()
    .sort_values(ascending=False)
    .head(30)
)

print("\n========== BY DATE ==========")
print(
    missing.groupby("datetime")
    .size()
    .sort_values(ascending=False)
    .head(30)
)

print("\n========== FIRST OBSERVATION PER INSTRUMENT ==========")

for instrument in missing["instrument"].drop_duplicates().head(20):

    stock = df[df["instrument"] == instrument].sort_values("datetime")

    valid = stock[
        stock[["$open", "$high", "$low", "$close", "$volume"]]
        .notna()
        .all(axis=1)
    ]

    missing_dates = stock[
        stock[["$open", "$high", "$low", "$close", "$volume"]]
        .isna()
        .all(axis=1)
    ]

    print(f"\n{instrument}")
    print(
        f"  First valid: "
        f"{valid['datetime'].min() if len(valid) else 'NONE'}"
    )
    print(
        f"  Last valid: "
        f"{valid['datetime'].max() if len(valid) else 'NONE'}"
    )
    print(
        f"  Missing rows: {len(missing_dates)}"
    )

print("\n========== DONE ==========")