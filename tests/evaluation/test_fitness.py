import numpy as np
import pandas as pd
import pytest

from python.dsl import parse, validate
from python.evaluation import (
    AlphaEvaluator,
    FitnessEvaluator,
    FitnessEvaluationError,
)


# =========================================================
# Test Data
# =========================================================

def make_research_data():

    dates = pd.date_range(
        "2025-01-01",
        periods=5,
        freq="D",
    )

    rows = []

    for i, date in enumerate(dates):

        # AAA
        rows.append({
            "datetime": date,
            "instrument": "AAA",

            "open": 10 + i,
            "high": 11 + i,
            "low": 9 + i,
            "close": 10 + i,
            "volume": 100 + i,

            "forward_return_1d":
                0.01 + 0.001 * i,

            "forward_return_5d":
                0.02 + 0.001 * i,

            "forward_return_20d":
                0.03 + 0.001 * i,

            "forward_return_30d":
                0.04 + 0.001 * i,
        })

        # BBB
        rows.append({
            "datetime": date,
            "instrument": "BBB",

            "open": 20 + i,
            "high": 21 + i,
            "low": 19 + i,
            "close": 20 + i,
            "volume": 200 + i,

            "forward_return_1d":
                -0.01 - 0.001 * i,

            "forward_return_5d":
                -0.02 - 0.001 * i,

            "forward_return_20d":
                -0.03 - 0.001 * i,

            "forward_return_30d":
                -0.04 - 0.001 * i,
        })

    return pd.DataFrame(rows)


def make_fitness(expression):

    ast = parse(expression)

    validate(ast)

    evaluator = FitnessEvaluator(
        make_research_data()
    )

    return evaluator.evaluate(
        ast,
        horizon=20,
    )


# =========================================================
# Basic Fitness Test
# =========================================================

def test_positive_alpha():

    result = make_fitness(
        "SUB(30,CLOSE)"
    )

    assert result.horizon == 20

    assert result.num_days == 5

    assert result.num_observations == 10

    assert result.coverage == 1.0

    assert result.mean_ic > 0

# =========================================================
# Horizon Tests
# =========================================================

@pytest.mark.parametrize(
    "horizon",
    [1, 5, 20, 30],
)
def test_supported_horizons(horizon):

    ast = parse("CLOSE")
    validate(ast)

    evaluator = FitnessEvaluator(
        make_research_data()
    )

    result = evaluator.evaluate(
        ast,
        horizon=horizon,
    )

    assert result.horizon == horizon


# =========================================================
# Invalid Horizon
# =========================================================

def test_invalid_horizon():

    ast = parse("CLOSE")
    validate(ast)

    evaluator = FitnessEvaluator(
        make_research_data()
    )

    with pytest.raises(
        FitnessEvaluationError
    ):

        evaluator.evaluate(
            ast,
            horizon=10,
        )


def test_split_aware_fitness_excludes_cross_split_targets():
    data = make_research_data()
    data["split"] = "train"
    data["target_split_20d"] = "train"
    # The last day's signal belongs to train but its target crosses into
    # validation, so it must not contribute to train fitness.
    data.loc[data["datetime"] == data["datetime"].max(), "target_split_20d"] = "validation"

    result = FitnessEvaluator(data).evaluate(
        parse("CLOSE"), horizon=20, split="train"
    )

    assert result.num_days == 4
    assert result.num_observations == 8
    assert result.coverage == 0.8


def test_split_aware_fitness_requires_split_metadata():
    evaluator = FitnessEvaluator(make_research_data())

    with pytest.raises(FitnessEvaluationError, match="requires columns"):
        evaluator.evaluate(parse("CLOSE"), horizon=20, split="train")


# =========================================================
# Result Structure
# =========================================================

def test_result_structure():

    result = make_fitness(
        "CLOSE"
    )

    assert isinstance(
        result.mean_ic,
        float,
    )

    assert isinstance(
        result.icir,
        float,
    )

    assert isinstance(
        result.ic_std,
        float,
    )

    assert isinstance(
        result.num_days,
        int,
    )

    assert isinstance(
        result.num_observations,
        int,
    )

    assert isinstance(
        result.coverage,
        float,
    )


# =========================================================
# Dictionary Conversion
# =========================================================

def test_result_to_dict():

    result = make_fitness(
        "CLOSE"
    )

    data = result.to_dict()

    assert data["horizon"] == 20

    assert "mean_ic" in data

    assert "ic_std" in data

    assert "icir" in data

    assert "num_days" in data

    assert "num_observations" in data

    assert "coverage" in data


# =========================================================
# Missing Future Returns
# =========================================================

def test_missing_future_returns():

    data = make_research_data()

    data.loc[
        0,
        "forward_return_20d"
    ] = np.nan

    evaluator = FitnessEvaluator(data)

    ast = parse("CLOSE")
    validate(ast)

    result = evaluator.evaluate(
        ast,
        horizon=20,
    )

    assert result.num_observations == 9

    assert result.coverage == 0.9


# =========================================================
# Constant Alpha
# =========================================================

def test_constant_alpha():

    ast = parse("5")
    validate(ast)

    evaluator = FitnessEvaluator(
        make_research_data()
    )

    result = evaluator.evaluate(
        ast,
        horizon=20,
    )

    # A constant alpha cannot have
    # a meaningful correlation.
    assert np.isnan(
        result.mean_ic
    )

    assert np.isnan(
        result.icir
    )


# =========================================================
# Daily IC Series
# =========================================================

def test_daily_ic_series_is_returned():

    result = make_fitness(
        "SUB(30,CLOSE)"
    )

    daily_ic = result.daily_ic

    assert isinstance(daily_ic, pd.Series)

    assert len(daily_ic) == 5

    assert daily_ic.notna().sum() == result.num_days

    # Two instruments ranked in the same order
    # as their returns on every date.
    assert np.allclose(daily_ic.to_numpy(), 1.0)

    assert np.isclose(
        daily_ic.mean(),
        result.mean_ic,
    )


def test_daily_ic_is_not_serialized():

    result = make_fitness(
        "CLOSE"
    )

    assert "daily_ic" not in result.to_dict()


def make_random_research_data(
    num_dates=30,
    num_instruments=12,
    seed=3,
):

    rng = np.random.default_rng(seed)

    rows = []

    for date in pd.date_range("2025-01-01", periods=num_dates, freq="D"):
        for k in range(num_instruments):
            close = float(rng.uniform(5, 50))
            rows.append({
                "datetime": date,
                "instrument": f"S{k:02d}",
                "open": close * float(rng.uniform(0.98, 1.02)),
                "high": close * 1.03,
                "low": close * 0.97,
                "close": close,
                "volume": float(rng.uniform(1e5, 1e6)),
                "forward_return_1d": float(rng.normal(0, 0.02)),
                "forward_return_5d": float(rng.normal(0, 0.04)),
                "forward_return_20d": float(rng.normal(0, 0.08)),
                "forward_return_30d": float(rng.normal(0, 0.1)),
            })

    data = pd.DataFrame(rows)

    # Gaps the IC must skip.
    data.loc[data.sample(frac=0.1, random_state=seed).index,
             "forward_return_20d"] = np.nan

    return data


def test_daily_ic_matches_pandas_correlation():

    data = make_random_research_data()

    evaluator = FitnessEvaluator(data)

    ast = parse("RANK(DIV(SUB(CLOSE,OPEN),VOLUME))")
    validate(ast)

    result = evaluator.evaluate(
        ast,
        horizon=20,
    )

    alpha = AlphaEvaluator(evaluator.data).evaluate(ast)
    alpha["ret"] = evaluator.data["forward_return_20d"].to_numpy()

    expected = (
        alpha
        .dropna()
        .groupby("datetime")
        .apply(
            lambda group: group["alpha"].corr(group["ret"]),
            include_groups=False,
        )
    )

    assert np.allclose(
        result.daily_ic.loc[expected.index].to_numpy(),
        expected.to_numpy(),
        atol=1e-12,
    )


def test_constant_alpha_with_rounding_is_undefined():

    # 0.1 * 3 is not exactly 0.3 in floating point, but the
    # alpha is still constant on every date.
    ast = parse("MUL(0.1,3)")
    validate(ast)

    evaluator = FitnessEvaluator(
        make_random_research_data()
    )

    result = evaluator.evaluate(
        ast,
        horizon=20,
    )

    assert result.daily_ic.isna().all()

    assert result.num_days == 0


def test_evaluator_is_reused_across_calls():

    evaluator = FitnessEvaluator(
        make_random_research_data()
    )

    first = parse("CLOSE")
    second = parse("RANK(VOLUME)")

    evaluator.evaluate(first, horizon=20)
    cached = evaluator._alpha_evaluator

    evaluator.evaluate(second, horizon=5)

    assert cached is not None
    assert evaluator._alpha_evaluator is cached
