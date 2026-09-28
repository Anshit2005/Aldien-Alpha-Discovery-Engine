import pandas as pd

PATH = "data/processed/csi300_market_data.parquet"

df = pd.read_parquet(PATH)

ohlcv = ["$open", "$high", "$low", "$close", "$volume"]

missing = df[df[ohlcv].isna().all(axis=1)].copy()

print("========== MISSING PERIOD ANALYSIS ==========")

# -------------------------------------------------
# 1. Missing percentage by year
# -------------------------------------------------

total_by_year = df.groupby(df["datetime"].dt.year).size()
missing_by_year = missing.groupby(missing["datetime"].dt.year).size()

summary = pd.DataFrame({
    "total": total_by_year,
    "missing": missing_by_year
}).fillna(0)

summary["missing_pct"] = (
    summary["missing"] / summary["total"] * 100
)

print("\n========== YEARLY MISSINGNESS ==========")
print(summary.to_string())

# -------------------------------------------------
# 2. July-August 2015
# -------------------------------------------------

period = missing[
    (missing["datetime"] >= "2015-07-01") &
    (missing["datetime"] <= "2015-08-31")
]

print("\n========== JUL-AUG 2015 ==========")
print(f"Missing rows: {len(period)}")
print(f"Unique instruments: {period['instrument'].nunique()}")
print(f"Unique dates: {period['datetime'].nunique()}")

print("\nMissing by date:")
print(
    period.groupby("datetime")
    .size()
    .sort_values(ascending=False)
    .to_string()
)

# -------------------------------------------------
# 3. Longest missing streak per instrument
# -------------------------------------------------

print("\n========== LONGEST MISSING STREAKS ==========")

results = []

for instrument, stock in df.groupby("instrument"):

    stock = stock.sort_values("datetime")

    is_missing = stock[ohlcv].isna().all(axis=1)

    groups = (is_missing != is_missing.shift()).cumsum()

    streaks = (
        stock[is_missing]
        .assign(group=groups[is_missing])
        .groupby("group")
        .agg(
            start=("datetime", "min"),
            end=("datetime", "max"),
            days=("datetime", "count")
        )
    )

    if len(streaks):
        longest = streaks.loc[streaks["days"].idxmax()]

        results.append({
            "instrument": instrument,
            "start": longest["start"],
            "end": longest["end"],
            "days": longest["days"]
        })

streaks_df = pd.DataFrame(results)

print(
    streaks_df
    .sort_values("days", ascending=False)
    .head(30)
    .to_string(index=False)
)

print("\n========== DONE ==========")