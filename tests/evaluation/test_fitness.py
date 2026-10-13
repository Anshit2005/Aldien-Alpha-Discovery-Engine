import numpy as np
import pandas as pd
import pytest

from python.dsl import parse, validate
from python.evaluation import (
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