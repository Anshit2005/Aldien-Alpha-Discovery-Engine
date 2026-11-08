"""Save and load the train daily IC of every formula a search evaluated.

The statistical filter needs each formula's daily IC series, not only its
summary ICIR, and it must count every formula tried, not only the top-k. The
series are too large for the result JSON (a few hundred formulas x ~2,400
days), so they go into a compressed ``.npz`` file next to it:

    evolution_seed_17.json
    evolution_seed_17_daily_ic.npz   <- expressions, directions, dates, ic

``ic`` is a (formulas x dates) matrix; a NaN means the IC was undefined that
day (too few stocks, constant values, or a failed evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from python.search.direction import signal_direction


@dataclass
class DailyICTable:
    expressions: list[str]
    directions: np.ndarray  # int8, sign of train ICIR (0 = undefined)
    dates: pd.DatetimeIndex
    ic: np.ndarray  # float64, shape (len(expressions), len(dates))

    def __len__(self) -> int:
        return len(self.expressions)

    def series(self, expression: str) -> pd.Series:
        row = self.expressions.index(expression)
        return pd.Series(self.ic[row], index=self.dates, name=expression)


def sidecar_path(result_json: Path | str) -> Path:
    result_json = Path(result_json)
    return result_json.with_name(f"{result_json.stem}_daily_ic.npz")


def table_from_candidates(candidates: Iterable) -> DailyICTable:
    """Build the table from search candidates (``expression`` and ``fitness``)."""

    candidates = list(candidates)
    series = [candidate.fitness.daily_ic for candidate in candidates]
    dates = pd.DatetimeIndex([])
    for item in series:
        if item is not None:
            dates = dates.union(pd.DatetimeIndex(item.index))
    ic = np.full((len(candidates), len(dates)), np.nan)
    for row, item in enumerate(series):
        if item is not None and len(item):
            ic[row] = item.reindex(dates).to_numpy(dtype=float)
    return DailyICTable(
        expressions=[candidate.expression for candidate in candidates],
        directions=np.array(
            [signal_direction(candidate.fitness.icir) for candidate in candidates],
            dtype=np.int8,
        ),
        dates=dates,
        ic=ic,
    )


def save_daily_ic(path: Path | str, table: DailyICTable) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        expressions=np.array(table.expressions, dtype=str),
        directions=table.directions,
        dates=table.dates.to_numpy(dtype="datetime64[ns]"),
        ic=table.ic,
    )
    return path


def load_daily_ic(path: Path | str) -> DailyICTable:
    with np.load(Path(path), allow_pickle=False) as data:
        return DailyICTable(
            expressions=[str(item) for item in data["expressions"]],
            directions=data["directions"].astype(np.int8),
            dates=pd.DatetimeIndex(data["dates"]),
            ic=data["ic"].astype(float),
        )
