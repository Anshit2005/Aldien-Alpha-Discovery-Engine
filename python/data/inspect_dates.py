import pandas as pd

path = "data/raw/qlib_csi300/daily_pv.h5"

df = pd.read_hdf(path, key="data")

dates = pd.to_datetime(
    df.index.get_level_values("datetime")
)

print("\n========== DATE INFORMATION ==========")

print("Start:", dates.min())
print("End:", dates.max())

print("Trading days:", dates.nunique())

print("\nRows per year:")
print(dates.to_series().groupby(dates.year).size())

print("\n======================================")