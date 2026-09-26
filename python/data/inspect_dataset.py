import pandas as pd

path = "data/raw/qlib_csi300/daily_pv.h5"

df = pd.read_hdf(path, key="data")

print(df)
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nIndex:")
print(df.index)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())