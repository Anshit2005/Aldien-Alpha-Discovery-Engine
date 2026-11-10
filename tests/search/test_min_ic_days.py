import numpy as np
import pandas as pd
import pytest

from python.dsl.parser import parse
from python.evaluation.fitness import FitnessEvaluator, FitnessResult
from python.search.evolution import EvolutionConfig, EvolutionSearch, train_score
from python.search.random_search import RandomSearch, RandomSearchConfig
from tests.search.test_evolution import make_research_data


def _sparse_ic_data(train_days=8, varying_days=2, stocks=5, seed=0):
    """
    VOLUME differs between stocks on the first ``varying_days`` train days
    only and is the same for every stock afterwards, so its daily IC exists
    on ``varying_days`` days although it has a value on every row.
    CLOSE differs between stocks every day.
    """

    rng = np.random.default_rng(seed)
    rows = []
    for day, date in enumerate(pd.bdate_range("2020-01-01", periods=train_days + 2)):
        split = "train" if day < train_days else "validation"
        for stock in range(stocks):
            close = 10.0 + stock + rng.normal(0, 0.5)
            volume = 1000.0 + (stock * 10 + rng.normal(0, 5) if day < varying_days else 0.0)
            row = {
                "datetime": date, "instrument": f"S{stock}", "open": close, "high": close + 1,
                "low": close - 1, "close": close, "volume": volume, "split": split,
            }
            for horizon in (1, 5, 20, 30):
                row[f"forward_return_{horizon}d"] = 0.001 * stock + rng.normal(0, 0.002)
                row[f"target_split_{horizon}d"] = split
            rows.append(row)
    return pd.DataFrame(rows)


def _fitness(num_days, icir=-0.4):
    return FitnessResult(
        horizon=20, mean_ic=0.01, ic_std=0.1, icir=icir,
        num_days=num_days, num_observations=100, coverage=0.9,
    )


def test_ic_days_available_counts_days_with_two_usable_stocks():

    evaluator = FitnessEvaluator(make_research_data())

    assert evaluator.ic_days_available(20, "train") == 4
    assert evaluator.ic_days_available(20, "validation") == 2


def test_train_score_needs_enough_ic_days():

    assert train_score(_fitness(9), min_coverage=0.5, min_ic_days=10) == 0.0
    assert train_score(_fitness(10), min_coverage=0.5, min_ic_days=10) == pytest.approx(0.4)
    # The default keeps the old behaviour for callers that do not pass it.
    assert train_score(_fitness(1), min_coverage=0.5) == pytest.approx(0.4)


def test_formula_with_few_ic_days_scores_zero_in_both_searches():

    data = _sparse_ic_data()
    sparse, dense = parse("VOLUME"), parse("CLOSE")
    evolution = EvolutionSearch(data, EvolutionConfig(population_size=8, elite_count=2))
    random_search = RandomSearch(data, RandomSearchConfig(evaluation_budget=1))

    scored = evolution._score(sparse)
    assert scored.fitness.coverage == 1.0
    assert scored.fitness.num_days == 2 and np.isfinite(scored.fitness.icir)
    assert scored.score == 0.0
    assert evolution._score(dense).score > 0
    candidates = []
    random_search._evaluate(sparse, set(), candidates)
    assert candidates[0].score == 0.0

    # Without the rule the same formula competes on its two-day ICIR.
    lenient = EvolutionSearch(
        data, EvolutionConfig(population_size=8, elite_count=2, min_ic_day_share=0.0)
    )
    assert lenient._score(sparse).score == pytest.approx(abs(scored.fitness.icir))


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_min_ic_day_share_must_be_a_fraction(value):

    with pytest.raises(ValueError, match="min_ic_day_share"):
        EvolutionConfig(min_ic_day_share=value)
    with pytest.raises(ValueError, match="min_ic_day_share"):
        RandomSearchConfig(min_ic_day_share=value)
