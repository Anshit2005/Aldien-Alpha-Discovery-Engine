import numpy as np
import pandas as pd
import pytest
from scipy import stats

from python.stats.filter import (
    FilterConfig,
    benjamini_hochberg,
    filter_table,
    ic_significance,
    newey_west_variance,
)
from python.stats.ic_store import DailyICTable


def overlapping_noise(rng, n=2000, horizon=20):
    """Daily IC with no signal but the autocorrelation of a 20-day horizon."""
    shocks = rng.normal(0, 0.1, n + horizon - 1)
    return np.convolve(shocks, np.ones(horizon) / horizon, mode="valid")


def test_pure_noise_is_not_significant():

    rng = np.random.default_rng(0)

    result = ic_significance(rng.normal(0, 0.1, 2000), direction=1)

    assert result["n_days"] == 2000
    assert result["p_value"] > 0.05


def test_clear_signal_is_significant():

    rng = np.random.default_rng(1)

    result = ic_significance(0.03 + rng.normal(0, 0.1, 2000), direction=1)

    assert result["p_value"] < 1e-6
    assert result["t_stat"] > 5


def test_negative_signal_is_significant_with_its_train_direction():

    rng = np.random.default_rng(2)
    daily_ic = -0.03 + rng.normal(0, 0.1, 2000)

    result = ic_significance(daily_ic, direction=-1)

    assert result["mean_ic"] > 0
    assert result["p_value"] < 1e-6
    assert ic_significance(daily_ic, direction=0)["p_value"] == 1.0


@pytest.mark.parametrize("method, limit", [("hansen_hodrick", 0.09), ("newey_west", 0.12)])
def test_hac_methods_control_false_positives_from_overlap(method, limit):

    # A plain t-test treats overlapping days as independent and finds
    # "significance" in pure noise far more often than 5% of the time.
    rng = np.random.default_rng(3)
    plain = corrected = 0
    trials = 200
    for _ in range(trials):
        series = overlapping_noise(rng)
        direction = 1 if series.mean() > 0 else -1
        plain += stats.ttest_1samp(series, 0).pvalue < 0.05
        corrected += ic_significance(series, direction, 20, method)["p_value"] < 0.05

    assert plain / trials > 0.4
    assert corrected / trials < limit


def test_non_overlapping_keeps_every_horizon_th_day_and_matches_a_t_test():

    rng = np.random.default_rng(4)
    series = 0.02 + overlapping_noise(rng, n=2000)

    result = ic_significance(series, 1, horizon=20, method="non_overlapping")

    sample = series[::20]
    assert result["n_days"] == len(sample) == 100
    assert result["p_value"] == pytest.approx(stats.ttest_1samp(sample, 0).pvalue)


def test_nan_days_are_skipped_and_short_series_give_p_one():

    series = np.r_[np.full(10, np.nan), 0.05 + np.random.default_rng(5).normal(0, 0.1, 500)]

    assert ic_significance(series, 1)["n_days"] == 500
    assert ic_significance(series[:25], 1)["p_value"] == 1.0
    assert ic_significance(np.full(100, 0.02), 1)["p_value"] == 1.0  # zero variance


def test_newey_west_without_lags_is_the_plain_variance():

    values = np.random.default_rng(6).normal(size=300)

    assert newey_west_variance(values, 0) == pytest.approx(values.var())
    assert newey_west_variance(overlapping_noise(np.random.default_rng(7)), 19) > (
        overlapping_noise(np.random.default_rng(7)).var() * 5
    )


def test_benjamini_hochberg_hand_example():

    p = [0.01, 0.039, 0.03, 0.005, 0.5]

    # m = 5: thresholds 0.01, 0.02, 0.03, 0.04, 0.05 for the sorted p-values
    # 0.005, 0.01, 0.03, 0.039, 0.5 -> the 4th is the largest that passes.
    cutoff, passed = benjamini_hochberg(p, q=0.05)
    assert cutoff == 0.039
    assert passed.tolist() == [True, True, True, True, False]

    # Counting 10 tests (5 unlisted ones had p = 1): thresholds 0.005, 0.01,
    # 0.015, 0.02, 0.025 -> only the two smallest pass.
    cutoff, passed = benjamini_hochberg(p, q=0.05, num_tests=10)
    assert cutoff == 0.01
    assert passed.tolist() == [True, False, False, True, False]


def test_benjamini_hochberg_step_up_and_edge_cases():

    # 0.04 fails its own threshold (0.02) but passes because a larger
    # p-value further up the list passes (step-up).
    cutoff, passed = benjamini_hochberg([0.04, 0.045], q=0.05)
    assert cutoff == 0.045 and passed.all()

    assert benjamini_hochberg([0.2, 0.6], q=0.05)[0] is None
    assert benjamini_hochberg([], q=0.05)[1].size == 0
    with pytest.raises(ValueError):
        benjamini_hochberg([0.01, 0.02], num_tests=1)


def test_filter_table_keeps_the_real_signal_among_noise():

    rng = np.random.default_rng(8)
    dates = pd.bdate_range("2010-01-01", periods=1500)
    rows = [0.04 + overlapping_noise(rng, 1500)]
    rows += [overlapping_noise(rng, 1500) for _ in range(49)]
    ic = np.vstack(rows)
    table = DailyICTable(
        expressions=["SIGNAL"] + [f"NOISE{i}" for i in range(49)],
        directions=np.sign(np.nanmean(ic, axis=1)).astype(np.int8),
        dates=dates,
        ic=ic,
    )

    result = filter_table(table, FilterConfig(q=0.05, horizon=20), num_tests=50)

    # The signal passes first. At q = 0.05 one noise series also passes
    # (p = 0.0015, under its BH threshold 2 * 0.05 / 50): the false-discovery
    # rate is controlled on average, not made zero.
    assert result["passed"][0] == "SIGNAL"
    assert len(result["passed"]) <= 2
    assert result["num_tests"] == 50
    assert result["bh_cutoff"] == max(
        item["p_value"] for item in result["formulas"] if item["passed"]
    )
    strict = filter_table(table, FilterConfig(q=0.01, horizon=20), num_tests=50)
    assert strict["passed"] == ["SIGNAL"]


@pytest.mark.parametrize(
    "options", [{"q": 0}, {"q": 1}, {"method": "bootstrap"}, {"horizon": 0}, {"min_days": 2}]
)
def test_filter_config_is_validated(options):

    with pytest.raises(ValueError):
        FilterConfig(**options)
