import pandas as pd

MEMBERSHIP_PATH = "data/raw/qlib_csi300/instruments/csi300.txt"
MARKET_DATA_PATH = "data/raw/qlib_csi300/daily_pv.h5"

# -----------------------------
# Load CSI300 membership
# -----------------------------

membership = pd.read_csv(
    MEMBERSHIP_PATH,
    sep=r"\s+",
    header=None,
    names=["instrument", "start_date", "end_date"],
)

membership["start_date"] = pd.to_datetime(membership["start_date"])
membership["end_date"] = pd.to_datetime(membership["end_date"])

membership_instruments = set(membership["instrument"].unique())

print("========== CSI300 UNIVERSE ==========")
print("Unique CSI300 instruments:", len(membership_instruments))


# -----------------------------
# Load market data index only
# -----------------------------

print("\nLoading market-data index...")

df = pd.read_hdf(
    MARKET_DATA_PATH,
    key="data",
)

market_instruments = set(df.index.get_level_values("instrument").unique())

print("Unique market-data instruments:", len(market_instruments))


# -----------------------------
# Compare
# -----------------------------

intersection = membership_instruments & market_instruments
missing = membership_instruments - market_instruments

print("\n========== OVERLAP ==========")
print("CSI300 instruments:", len(membership_instruments))
print("Market instruments:", len(market_instruments))
print("Present in both:", len(intersection))
print("Missing from market data:", len(missing))

if missing:
    print("\nMissing instruments:")
    for instrument in sorted(missing):
        print(instrument)