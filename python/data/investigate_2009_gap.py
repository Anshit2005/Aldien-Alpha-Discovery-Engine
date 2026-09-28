import pandas as pd

MEMBERSHIP_PATH = "data/raw/qlib_csi300/instruments/csi300.txt"

membership = pd.read_csv(
    MEMBERSHIP_PATH,
    sep=r"\s+",
    header=None,
    names=["instrument", "start_date", "end_date"],
)

membership["start_date"] = pd.to_datetime(membership["start_date"])
membership["end_date"] = pd.to_datetime(membership["end_date"])

for date in [
    "2009-12-28",
    "2009-12-29",
    "2009-12-30",
    "2009-12-31",
    "2010-01-04",
]:

    date = pd.Timestamp(date)

    active = membership[
        (membership["start_date"] <= date)
        & (membership["end_date"] >= date)
    ]

    stocks = set(active["instrument"])

    print(f"{date.date()}: {len(stocks)} stocks")

    if date != pd.Timestamp("2009-12-28"):
        previous_date = date - pd.Timedelta(days=1)

        previous = membership[
            (membership["start_date"] <= previous_date)
            & (membership["end_date"] >= previous_date)
        ]

        previous_stocks = set(previous["instrument"])

        print("  Added:", sorted(stocks - previous_stocks))
        print("  Removed:", sorted(previous_stocks - stocks))