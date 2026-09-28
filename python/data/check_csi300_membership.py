import pandas as pd

path = "data/raw/qlib_csi300/instruments/csi300.txt"

# Read CSI300 membership file
membership = pd.read_csv(
    path,
    sep=r"\s+",
    header=None,
    names=["instrument", "start_date", "end_date"],
)

# Convert dates
membership["start_date"] = pd.to_datetime(membership["start_date"])
membership["end_date"] = pd.to_datetime(membership["end_date"])

print("========== CSI300 MEMBERSHIP ==========")
print("Total membership records:", len(membership))
print("Unique instruments:", membership["instrument"].nunique())

print("\nDate range:")
print("Start:", membership["start_date"].min())
print("End:", membership["end_date"].max())

# Dates we want to inspect
test_dates = [
    "2008-12-29",
    "2010-01-04",
    "2015-01-05",
    "2020-01-02",
    "2023-01-03",
    "2025-01-02",
    "2026-01-09",
]

print("\n========== CONSTITUENTS BY DATE ==========")

for date in test_dates:
    date = pd.Timestamp(date)

    active = membership[
        (membership["start_date"] <= date)
        & (membership["end_date"] >= date)
    ]

    instruments = active["instrument"].unique()

    print(f"{date.date()} : {len(instruments)} instruments")