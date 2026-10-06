"""
Build a leakage-safe research dataset from the cleaned CSI300 daily dataset.

Input:
    data/processed/csi300_market_data_clean.parquet

Outputs:
    data/processed/csi300_research_dataset.parquet
    data/processed/csi300_research_summary.txt

Creates 1, 5, 20 and 30 trading-day forward returns and applies
train/validation/test/holdout splits without allowing a target to cross
a split boundary.
"""

from pathlib import Path
import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/csi300_market_data_clean.parquet")
OUTPUT_PATH = Path("data/processed/csi300_research_dataset.parquet")
SUMMARY_PATH = Path("data/processed/csi300_research_summary.txt")

SPLITS = {
    "train": ("2009-01-01", "2018-12-31"),
    "validation": ("2019-01-01", "2020-12-31"),
    "test": ("2021-01-01", "2024-12-31"),
    "holdout": ("2025-01-01", "2026-12-31"),
}

HORIZONS = [1, 5, 20, 30]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Qlib-style columns to clean names."""
    df = df.rename(
        columns={
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
            "$factor": "factor",
        }
    ).copy()

    required = {
        "datetime", "instrument", "open", "high", "low", "close", "volume"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["instrument"] = df["instrument"].astype(str)

    return df


def assign_split(date_series: pd.Series) -> pd.Series:
    result = pd.Series("outside", index=date_series.index, dtype="object")

    for split_name, (start, end) in SPLITS.items():
        mask = date_series.between(pd.Timestamp(start), pd.Timestamp(end))
        result.loc[mask] = split_name

    return result


def build_trading_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Map each unique market date to a global trading-session id."""
    dates = (
        pd.Series(df["datetime"].drop_duplicates().sort_values().unique())
        .reset_index(drop=True)
        .rename("datetime")
    )

    return pd.DataFrame(
        {
            "datetime": dates,
            "trading_day_id": np.arange(len(dates), dtype=np.int32),
        }
    )


def add_forward_returns(
    df: pd.DataFrame,
    calendar: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Create forward returns using exact global trading-day offsets.

    For date t and horizon h:
        target_date = h trading sessions after t
        forward_return = close(target_date) / close(t) - 1

    A target exists only when the same instrument has a valid close on the
    exact target trading date.
    """
    out = df.copy()

    out = out.merge(
        calendar,
        on="datetime",
        how="left",
        validate="many_to_one",
    )

    if out["trading_day_id"].isna().any():
        raise ValueError("Some rows could not be mapped to the trading calendar.")

    # Lookup indexed by (instrument, trading_day_id).
    close_lookup = (
        out[["instrument", "trading_day_id", "close"]]
        .drop_duplicates(["instrument", "trading_day_id"])
        .set_index(["instrument", "trading_day_id"])["close"]
    )

    # Global trading-day-id -> actual date.
    day_to_date = (
        calendar.set_index("trading_day_id")["datetime"]
    )

    for h in horizons:
        target_day_id = out["trading_day_id"] + h

        target_index = pd.MultiIndex.from_arrays(
            [out["instrument"].to_numpy(), target_day_id.to_numpy()],
            names=["instrument", "trading_day_id"],
        )

        future_close = close_lookup.reindex(target_index).to_numpy()

        out[f"future_close_{h}d"] = future_close
        out[f"target_date_{h}d"] = day_to_date.reindex(
            target_day_id
        ).to_numpy()

        ratio = out[f"future_close_{h}d"] / out["close"]

        out[f"forward_return_{h}d"] = ratio - 1.0
        out[f"log_forward_return_{h}d"] = np.log(ratio)

    return out


def enforce_split_target_boundary(
    df: pd.DataFrame,
    calendar: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Ensure a target stays inside the same split as its observation.

    Example:
      A 20-day target for a training observation in Dec-2018 cannot use
      a target close in Jan-2019 because Jan-2019 belongs to validation.
    """
    out = df.copy()

    date_split = (
        out[["datetime", "split"]]
        .drop_duplicates("datetime")
        .set_index("datetime")["split"]
    )

    day_to_date = calendar.set_index("trading_day_id")["datetime"]

    for h in horizons:
        future_day_id = out["trading_day_id"] + h
        future_date = day_to_date.reindex(future_day_id)

        out[f"target_date_{h}d"] = future_date.to_numpy()

        future_split = future_date.map(date_split).to_numpy()
        out[f"target_split_{h}d"] = future_split

        valid = (
            out["split"].isin(["train", "validation", "test", "holdout"])
            & (out["split"].to_numpy() == future_split)
            & out[f"future_close_{h}d"].notna().to_numpy()
        )

        out.loc[~valid, f"forward_return_{h}d"] = np.nan
        out.loc[~valid, f"log_forward_return_{h}d"] = np.nan

    return out


def validate(df: pd.DataFrame) -> list[str]:
    lines = [
        "=== RESEARCH DATASET VALIDATION ===",
        f"Rows: {len(df):,}",
        f"Unique instruments: {df['instrument'].nunique():,}",
        f"Unique dates: {df['datetime'].nunique():,}",
        f"Date range: {df['datetime'].min()} -> {df['datetime'].max()}",
    ]

    duplicates = df.duplicated(["datetime", "instrument"]).sum()
    lines.append(f"Duplicate (datetime, instrument) pairs: {duplicates}")

    lines.append("")
    lines.append("Split counts:")
    for split_name, count in df["split"].value_counts().sort_index().items():
        lines.append(f"  {split_name:12s}: {count:,}")

    lines.append("")
    lines.append("Forward-return availability:")

    for h in HORIZONS:
        col = f"forward_return_{h}d"
        valid = df[col].notna().sum()
        missing = len(df) - valid
        coverage = valid / len(df) if len(df) else 0.0

        lines.append(
            f"  {h:2d}d: valid={valid:,}, "
            f"missing={missing:,}, coverage={coverage:.2%}"
        )

    lines.append("")
    lines.append("Split leakage checks:")

    for h in HORIZONS:
        target_split_col = f"target_split_{h}d"
        target_return_col = f"forward_return_{h}d"

        leakage = (
            df["split"].isin(["train", "validation", "test", "holdout"])
            & df[target_return_col].notna()
            & df[target_split_col].notna()
            & (df["split"] != df[target_split_col])
        ).sum()
        
        lines.append(f"  {h:2d}d target crosses split boundary: {leakage}")

    lines.append("")
    lines.append("Forward-return statistics:")

    for h in HORIZONS:
        col = f"forward_return_{h}d"
        s = df[col].dropna()

        lines.append(
            f"  {h:2d}d: "
            f"mean={s.mean():.6f}, "
            f"std={s.std():.6f}, "
            f"min={s.min():.6f}, "
            f"median={s.median():.6f}, "
            f"max={s.max():.6f}"
        )

    return lines


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}\n"
            "Run the cleaned market-data pipeline first."
        )

    print(f"Loading: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    print(f"Input rows: {len(df):,}")

    df = standardize_columns(df)
    df = df.sort_values(["datetime", "instrument"]).reset_index(drop=True)

    if df.duplicated(["datetime", "instrument"]).any():
        raise ValueError("Duplicate (datetime, instrument) pairs found.")

    if df[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("Input still contains missing OHLCV values.")

    print("\n1. Building global trading calendar...")
    calendar = build_trading_calendar(df)
    print(f"Trading sessions: {len(calendar):,}")

    print("\n2. Creating forward returns...")
    df = add_forward_returns(df, calendar, HORIZONS)

    print("\n3. Assigning observation splits...")
    df["split"] = assign_split(df["datetime"])

    print("\n4. Preventing split-boundary target leakage...")
    df = enforce_split_target_boundary(df, calendar, HORIZONS)

    # Keep only the research period.
    df = df[df["split"] != "outside"].copy()

    # Put important columns first.
    first_cols = [
        "datetime",
        "instrument",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "factor",
        "trading_day_id",
        "split",
    ]

    target_cols = []
    for h in HORIZONS:
        target_cols += [
            f"future_close_{h}d",
            f"target_date_{h}d",
            f"target_split_{h}d",
            f"forward_return_{h}d",
            f"log_forward_return_{h}d",
        ]

    first_cols = [c for c in first_cols if c in df.columns]
    target_cols = [c for c in target_cols if c in df.columns]

    remaining = [
        c for c in df.columns
        if c not in first_cols and c not in target_cols
    ]

    df = df[first_cols + target_cols + remaining]
    df = df.sort_values(["datetime", "instrument"]).reset_index(drop=True)

    print("\n5. Validation")
    validation_lines = validate(df)
    print("\n".join(validation_lines))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        OUTPUT_PATH,
        engine="pyarrow",
        index=False,
    )

    SUMMARY_PATH.write_text(
        "\n".join(validation_lines) + "\n",
        encoding="utf-8",
    )

    print("\nSaved:")
    print(f"  {OUTPUT_PATH}")
    print(f"  {SUMMARY_PATH}")

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
