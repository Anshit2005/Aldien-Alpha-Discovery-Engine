from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "csi300_research_dataset.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cpp_fitness_data.csv"
)


def main():

    print(f"Loading: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    df["datetime"] = pd.to_datetime(df["datetime"])

    # ------------------------------------------------------------
    # Use the same 20 instruments as the previous C++ test
    # ------------------------------------------------------------

    instruments = (
        df["instrument"]
        .drop_duplicates()
        .sort_values()
        .head(20)
    )

    df = df[
        df["instrument"].isin(instruments)
    ].copy()

    # ------------------------------------------------------------
    # Use the same first 500 trading dates
    # ------------------------------------------------------------

    dates = (
        df["datetime"]
        .drop_duplicates()
        .sort_values()
        .head(500)
    )

    df = df[
        df["datetime"].isin(dates)
    ].copy()

    # ------------------------------------------------------------
    # Select columns required by C++ fitness engine
    # ------------------------------------------------------------

    columns = [
        "datetime",
        "instrument",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_20d",
        "forward_return_30d",
    ]

    df = df[columns].copy()

    # ------------------------------------------------------------
    # Sort exactly as required by the C++ time-series evaluator
    # ------------------------------------------------------------

    df = (
        df.sort_values(
            ["instrument", "datetime"]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Export complete.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(
        f"Instruments: "
        f"{df['instrument'].nunique()}"
    )
    print(
        f"Dates: "
        f"{df['datetime'].nunique()}"
    )
    print(
        f"Date range: "
        f"{df['datetime'].min()} → "
        f"{df['datetime'].max()}"
    )

    print()
    print("Missing forward returns:")

    for horizon in [1, 5, 20, 30]:

        column = f"forward_return_{horizon}d"

        missing = df[column].isna().sum()

        print(
            f"{column}: "
            f"{missing}"
        )


if __name__ == "__main__":
    main()