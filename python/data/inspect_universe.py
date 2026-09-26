import pandas as pd

path = "data/raw/qlib_csi300/daily_pv.h5"

df = pd.read_hdf(path, key="data")

# Extract instruments from MultiIndex
instruments = df.index.get_level_values("instrument").unique()

print("\n========== UNIVERSE ==========")

print("Number of unique instruments:", len(instruments))

print("\nFirst 50 instruments:")
for instrument in instruments[:50]:
    print(instrument)

print("\nLast 50 instruments:")
for instrument in instruments[-50:]:
    print(instrument)

print("\n==============================")

counts = df.groupby(level="datetime").size()

print("\n========== DAILY COVERAGE ==========")

print("Minimum instruments:", counts.min())
print("Maximum instruments:", counts.max())
print("Mean instruments:", counts.mean())
print("Median instruments:", counts.median())

print("\nFirst 10 days:")
print(counts.head(10))

print("\nLast 10 days:")
print(counts.tail(10))


missing_close = df["$close"].isna()

missing_by_instrument = (
    missing_close
    .groupby(level="instrument")
    .sum()
    .sort_values(ascending=False)
)

print("\n========== MISSING CLOSE ==========")
print(missing_by_instrument.head(30))