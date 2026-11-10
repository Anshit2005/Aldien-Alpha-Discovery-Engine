"""
The load-once C++ engine (alpha_engine_cpp.Engine via CppFitnessEvaluator)
gives the same alpha values, fitness and daily IC as the Python reference.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("python.cpp_engine", reason="C++ module alpha_engine_cpp is not built (see README)")

from python.cpp_engine import alpha_engine_cpp  # noqa: E402
from python.dsl.parser import parse  # noqa: E402
from python.evaluation.cpp_fitness import CppEvaluationError, CppFitnessEvaluator  # noqa: E402
from python.evaluation.evaluator import AlphaEvaluator  # noqa: E402
from python.evaluation.fitness import FitnessEvaluationError, FitnessEvaluator  # noqa: E402
from python.search.evolution import EvolutionConfig, EvolutionSearch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_SAMPLE = PROJECT_ROOT / "data" / "processed" / "cpp_fitness_data.csv"


def split_panel(days=90, stocks=12, seed=0):
    """Panel with train/validation/test splits, some missing rows and flat prices."""

    rng = np.random.default_rng(seed)
    rows = []
    for day, date in enumerate(pd.bdate_range("2016-01-01", periods=days)):
        split = "train" if day < 50 else ("validation" if day < 70 else "test")
        for stock in range(stocks):
            if stock == 3 and 20 <= day < 25:
                continue  # a trading halt
            close = 10.0 + stock + rng.normal(0, 0.5)
            if stock == 5:
                close = 8.0  # flat price: windows of identical values
            row = {
                "datetime": date, "instrument": f"S{stock:02d}", "open": close * 0.99,
                "high": close * 1.02, "low": close * 0.98, "close": close,
                "volume": float(rng.uniform(1e5, 1e6)) if stock != 5 else 5e5, "split": split,
            }
            if stock == 7 and day % 9 == 0:
                row["volume"] = np.nan
            for horizon in (1, 5, 20, 30):
                row[f"forward_return_{horizon}d"] = 0.002 * close + rng.normal(0, 0.01)
                # Targets near a boundary fall into the next split.
                ahead = day + horizon
                row[f"target_split_{horizon}d"] = (
                    "train" if ahead < 50 else ("validation" if ahead < 70 else "test")
                ) if ahead < days else None
            rows.append(row)
    # Shuffled: both evaluators must sort it the same way.
    return pd.DataFrame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)


def random_formulas(count, seed=11):
    generator = EvolutionSearch.__new__(EvolutionSearch)
    generator.rng = random.Random(seed)
    generator.config = EvolutionConfig()
    formulas = list(EvolutionSearch.SEEDS) + [
        "TS_STD(CLOSE,3)", "TS_MEAN(CLOSE,4)", "RANK(VOLUME)", "SIGN(SUB(OPEN,CLOSE))",
        "LOG(SUB(OPEN,CLOSE))", "DIV(CLOSE,SUB(CLOSE,CLOSE))", "TS_MIN(TS_STD(CLOSE,2),5)",
    ]
    while len(formulas) < count:
        formulas.append(str(generator._random_tree(5)))
    return formulas


@pytest.fixture(scope="module")
def evaluators():
    data = split_panel()
    return FitnessEvaluator(data), CppFitnessEvaluator(data)


# pandas computes rolling means and standard deviations with a running sum
# (adding the newest value, removing the oldest), the C++ engine from scratch
# for every window, so values differ by rounding (~1e-10 here). Counts of days
# and observations, and which days have an IC, must match exactly.
VALUE_RTOL, VALUE_ATOL = 1e-7, 1e-9


def assert_same_fitness(expected, actual):
    assert actual.horizon == expected.horizon
    assert actual.num_days == expected.num_days
    assert actual.num_observations == expected.num_observations
    assert actual.coverage == pytest.approx(expected.coverage, rel=1e-12, abs=0)
    for field in ("mean_ic", "ic_std", "icir"):
        a, b = getattr(expected, field), getattr(actual, field)
        assert np.isnan(a) == np.isnan(b), field
        if not np.isnan(a):
            assert b == pytest.approx(a, rel=VALUE_RTOL, abs=VALUE_ATOL), field
    assert actual.daily_ic.index.equals(expected.daily_ic.index)
    assert actual.daily_ic.index.name == "datetime" and actual.daily_ic.name == "ic"
    assert np.array_equal(np.isnan(actual.daily_ic.to_numpy()), np.isnan(expected.daily_ic.to_numpy()))
    np.testing.assert_allclose(
        actual.daily_ic.to_numpy(), expected.daily_ic.to_numpy(), rtol=VALUE_RTOL, atol=VALUE_ATOL
    )


@pytest.mark.parametrize("split", [None, "train", "validation", "test"])
@pytest.mark.parametrize("horizon", [1, 20])
def test_fitness_matches_python_for_generated_formulas(evaluators, split, horizon):

    python, cpp = evaluators
    for expression in random_formulas(40):
        ast = parse(expression)
        assert_same_fitness(python.evaluate(ast, horizon, split), cpp.evaluate(ast, horizon, split))


def test_alpha_values_match_python(evaluators):

    python, cpp = evaluators
    reference = AlphaEvaluator(python.data)
    for expression in random_formulas(25, seed=3):
        ast = parse(expression)
        expected = reference.evaluate(ast)["alpha"].to_numpy(dtype=float)
        actual = cpp.evaluate_alpha(ast)
        expected = np.where(np.isfinite(expected), expected, np.nan)
        assert np.array_equal(np.isnan(actual), np.isnan(expected)), expression
        np.testing.assert_allclose(actual, expected, rtol=VALUE_RTOL, atol=VALUE_ATOL, equal_nan=True)


def test_constant_windows_are_exact(evaluators):

    _, cpp = evaluators
    data = cpp.data
    flat = data["instrument"].eq("S05").to_numpy()
    std = cpp.evaluate_alpha(parse("TS_STD(CLOSE,3)"))
    mean = cpp.evaluate_alpha(parse("TS_MEAN(CLOSE,4)"))
    assert np.all(std[flat][2:] == 0.0)
    assert np.all(mean[flat][3:] == 8.0)


def test_engine_counts_days_and_instruments(evaluators):

    python, cpp = evaluators
    assert cpp.engine.size == len(python.data)
    assert cpp.engine.num_days == python.data["datetime"].nunique()
    assert cpp.engine.num_instruments == python.data["instrument"].nunique()


def test_errors_match_the_python_evaluator(evaluators):

    _, cpp = evaluators
    with pytest.raises(FitnessEvaluationError, match="Unsupported horizon"):
        cpp.evaluate(parse("CLOSE"), horizon=2)
    with pytest.raises(FitnessEvaluationError, match="Unsupported split"):
        cpp.evaluate(parse("CLOSE"), split="future")


def test_engine_rejects_unsorted_rows():

    ids = np.array([0, 0], dtype=np.int32)
    values = np.ones(2)
    empty = np.empty(0, dtype=np.int8)
    with pytest.raises(ValueError, match="sorted"):
        alpha_engine_cpp.Engine(
            ids, np.array([1, 0], dtype=np.int32), values, values, values, values, values,
            [values] * 4, empty, [],
        )
    with pytest.raises(ValueError, match="one array per horizon"):
        alpha_engine_cpp.Engine(
            ids, np.array([0, 1], dtype=np.int32), values, values, values, values, values,
            [values], empty, [],
        )


def test_data_without_splits_evaluates_the_whole_panel():

    data = split_panel().drop(columns=["split"] + [f"target_split_{h}d" for h in (1, 5, 20, 30)])
    python, cpp = FitnessEvaluator(data), CppFitnessEvaluator(data)
    for expression in random_formulas(15, seed=5):
        ast = parse(expression)
        assert_same_fitness(python.evaluate(ast, 20), cpp.evaluate(ast, 20))
    with pytest.raises(FitnessEvaluationError, match="requires columns"):
        cpp.evaluate(parse("CLOSE"), split="train")


def test_errors_inside_cpp_are_value_errors():

    assert issubclass(CppEvaluationError, ValueError)
    assert issubclass(CppEvaluationError, FitnessEvaluationError)


@pytest.mark.skipif(
    not REAL_SAMPLE.exists(),
    reason=f"Missing fixture {REAL_SAMPLE}; run python -m python.data.export_cpp_fitness_data",
)
def test_real_sample_matches_python():

    data = pd.read_csv(REAL_SAMPLE)
    python, cpp = FitnessEvaluator(data), CppFitnessEvaluator(data)
    for expression in random_formulas(30, seed=21):
        ast = parse(expression)
        for horizon in (5, 20):
            assert_same_fitness(python.evaluate(ast, horizon), cpp.evaluate(ast, horizon))
