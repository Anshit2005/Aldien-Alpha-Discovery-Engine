import pandas as pd

MEMBERSHIP_PATH = "data/raw/qlib_csi300/instruments/csi300.txt"
MARKET_DATA_PATH = "data/raw/qlib_csi300/daily_pv.h5"

missing_instruments = [
    "SH600002",
    "SH600205",
    "SH600296",
    "SH600472",
    "SH600627",
    "SH600786",
    "SHT00018",
    "SZ000406",
    "SZ000618",
    "SZ000763",
    "SZ000817",
    "SZ000866",
    "SZ000956",
]

# Load membership information
membership = pd.read_csv(
    MEMBERSHIP_PATH,
    sep=r"\s+",
    header=None,
    names=["instrument", "start_date", "end_date"],
)

membership["start_date"] = pd.to_datetime(membership["start_date"])
membership["end_date"] = pd.to_datetime(membership["end_date"])

# Restrict to our actual market-data period
data_start = pd.Timestamp("2008-12-29")
data_end = pd.Timestamp("2026-01-09")

membership = membership[
    (membership["end_date"] >= data_start)
    & (membership["start_date"] <= data_end)
].copy()

print("========== MISSING INSTRUMENT ANALYSIS ==========")

for instrument in missing_instruments:

    rows = membership[membership["instrument"] == instrument]

    if rows.empty:
        print(f"\n{instrument}: No membership during dataset period")
        continue

    start = max(rows["start_date"].min(), data_start)
    end = min(rows["end_date"].max(), data_end)

    print(f"\n{instrument}")
    print(f"Membership range in dataset: {start.date()} -> {end.date()}")
    print(f"Membership records: {len(rows)}")