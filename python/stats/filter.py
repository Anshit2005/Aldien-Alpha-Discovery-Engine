"""Statistical filter: which searched formulas have a train IC that is
distinguishable from zero after accounting for how many formulas were tried?

Two steps:

1. Significance per formula. The daily IC is multiplied by the formula's
   train direction and tested against a mean of zero.

   * Overlap. With a 20-day horizon, the forward returns of consecutive days
     share 19 of their 20 days, so consecutive daily ICs are strongly
     correlated. A plain t-test treats them as independent and overstates
     significance roughly by sqrt(horizon). Three corrections are offered:

     - ``hansen_hodrick`` (default): keep every day and use the long-run
       variance with equal weights on the first ``horizon - 1`` lags, the
       exact length of the overlap (Hansen and Hodrick, 1980).
     - ``newey_west``: the same with Bartlett weights and ``2 * (horizon - 1)``
       lags; always positive but somewhat liberal at these sample sizes.
     - ``non_overlapping``: keep only every ``horizon``-th day and run a
       plain t-test; simplest, but discards most of the data.

     The two HAC methods use ``n / horizon - 1`` degrees of freedom, the
     number of independent horizons in the sample. On simulated pure noise
     with a 20-day overlap (2,000 days), the false-positive rate at 5% is
     about 5.4% for Hansen-Hodrick and 8% for Newey-West, against over 40%
     for a plain t-test (see tests/stats/test_filter.py).
   * Direction. The direction is the sign of the same train IC, so the
     oriented mean is positive by construction. Testing it one-sided would
     let the data pick the tail; the reported p-value is therefore the
     two-sided one, ``2 * P(T > t)``.

2. Benjamini-Hochberg false-discovery-rate control over *every* unique
   formula the search evaluated (``unique_train_evaluations``), not just the
   top-k, because the top-k were chosen from all of them.

Both steps use the train split only. Formulas that pass are then checked
once on the test split (see ``python.stats.test_split``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np
from scipy import stats

from .ic_store import DailyICTable


METHODS = ("hansen_hodrick", "newey_west", "non_overlapping")


@dataclass(frozen=True)
class FilterConfig:
    q: float = 0.05
    method: str = "hansen_hodrick"
    horizon: int = 20
    # Fewer usable days than this gives p = 1 (no evidence either way).
    min_days: int = 30

    def __post_init__(self):
        if not 0 < self.q < 1:
            raise ValueError("q must be between 0 and 1")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.min_days < 3:
            raise ValueError("min_days must be at least 3")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def long_run_variance(values: np.ndarray, lags: int, kernel: str = "bartlett") -> float:
    """
    Long-run variance of ``values``: the variance of its mean times ``n``.

    ``kernel="bartlett"`` is Newey-West (always non-negative);
    ``kernel="uniform"`` is Hansen-Hodrick (can be negative in small samples).
    """

    n = len(values)
    centred = values - values.mean()
    variance = float(centred @ centred) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1) if kernel == "bartlett" else 1.0
        variance += 2.0 * weight * float(centred[lag:] @ centred[:-lag]) / n
    return variance


def newey_west_variance(values: np.ndarray, lags: int) -> float:
    return long_run_variance(values, lags, "bartlett")


def ic_significance(
    daily_ic: Sequence[float],
    direction: int,
    horizon: int = 20,
    method: str = "hansen_hodrick",
    min_days: int = 30,
) -> dict[str, Any]:
    """
    Test whether the direction-adjusted daily IC has a non-zero mean.

    ``daily_ic`` must be in date order; NaN days are skipped.
    """

    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")
    values = np.asarray(daily_ic, dtype=float) * direction
    if method == "non_overlapping":
        values = values[::horizon]
    values = values[np.isfinite(values)]
    n = len(values)
    result = {"n_days": n, "mean_ic": float(values.mean()) if n else float("nan"),
              "t_stat": float("nan"), "p_value": 1.0}
    if direction == 0 or n < min_days or np.ptp(values) == 0:
        return result

    # Dropping NaN days joins the series across gaps; the gaps are a small
    # share of days, so the lag structure is barely affected.
    if method == "non_overlapping":
        variance, df = float(values.var(ddof=1)), n - 1
    else:
        df = max(n / horizon - 1, 2)
        variance = float("nan")
        if method == "hansen_hodrick":
            variance = long_run_variance(values, horizon - 1, "uniform")
        if not variance > 0:
            # Newey-West is the fallback when Hansen-Hodrick is not positive.
            variance = long_run_variance(values, 2 * (horizon - 1), "bartlett")
    if not math.isfinite(variance) or variance <= 0:
        return result

    t_stat = result["mean_ic"] / math.sqrt(variance / n)
    result["t_stat"] = float(t_stat)
    result["p_value"] = float(min(1.0, 2.0 * stats.t.sf(t_stat, df=df)))
    return result


def benjamini_hochberg(
    p_values: Sequence[float], q: float = 0.05, num_tests: int | None = None
) -> tuple[float | None, np.ndarray]:
    """
    Benjamini-Hochberg step-up procedure.

    ``num_tests`` is the number of hypotheses actually tested; it may exceed
    ``len(p_values)`` when some tests are not listed (they count as p = 1).
    Returns the largest passing p-value (None when nothing passes) and a
    boolean mask of the formulas that pass.
    """

    p = np.asarray(p_values, dtype=float)
    m = len(p) if num_tests is None else int(num_tests)
    if m < len(p):
        raise ValueError("num_tests cannot be smaller than the number of p-values")
    passed = np.zeros(len(p), dtype=bool)
    if m == 0 or len(p) == 0:
        return None, passed
    order = np.argsort(p, kind="stable")
    thresholds = q * np.arange(1, len(p) + 1) / m
    below = np.nonzero(p[order] <= thresholds)[0]
    if len(below) == 0:
        return None, passed
    cutoff = float(p[order][below[-1]])
    passed = p <= cutoff
    return cutoff, passed


def filter_table(
    table: DailyICTable, config: FilterConfig, num_tests: int | None = None
) -> dict[str, Any]:
    """
    Run both steps on every formula in ``table``.

    ``num_tests`` defaults to the number of formulas in the table, which is
    every unique train evaluation when the table came from a search.
    """

    num_tests = len(table) if num_tests is None else num_tests
    tests = [
        ic_significance(row, int(direction), config.horizon, config.method, config.min_days)
        for row, direction in zip(table.ic, table.directions)
    ]
    cutoff, passed = benjamini_hochberg([t["p_value"] for t in tests], config.q, num_tests)
    formulas = [
        {"expression": expression, "direction": int(direction), **test, "passed": bool(ok)}
        for expression, direction, test, ok in zip(table.expressions, table.directions, tests, passed)
    ]
    formulas.sort(key=lambda item: (item["p_value"], item["expression"]))
    return {
        "config": config.to_dict(),
        "num_tests": int(num_tests),
        "num_formulas_with_ic": sum(item["n_days"] >= config.min_days for item in formulas),
        "bh_cutoff": cutoff,
        "passed_count": int(passed.sum()),
        "passed": [item["expression"] for item in formulas if item["passed"]],
        "formulas": formulas,
    }
