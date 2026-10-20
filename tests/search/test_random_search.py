import pandas as pd

from python.search.evolution import EvolutionSearch
from python.search.random_search import RandomSearch, RandomSearchConfig


def make_research_data():
    rows = []
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    for day, date in enumerate(dates):
        split = "train" if day < 4 else "validation"
        for stock, base in (("A", 10.0), ("B", 20.0), ("C", 30.0)):
            close = base + day
            row = {
                "datetime": date,
                "instrument": stock,
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 100 + base + day,
                "split": split,
            }
            for horizon in (1, 5, 20, 30):
                row[f"forward_return_{horizon}d"] = (base - 20) * 0.001 + day * 0.0001
                row[f"target_split_{horizon}d"] = split
            rows.append(row)
    return pd.DataFrame(rows)


def test_random_search_is_reproducible_deduplicated_and_validates_top_k():
    config = RandomSearchConfig(
        evaluation_budget=8,
        seed=91,
        max_depth=4,
        validation_top_k=3,
    )
    first = RandomSearch(make_research_data(), config).run()
    second = RandomSearch(make_research_data(), config).run()

    first_expressions = [candidate.expression for candidate in first.best]
    second_expressions = [candidate.expression for candidate in second.best]
    assert first_expressions == second_expressions
    assert [candidate.score for candidate in first.best] == [
        candidate.score for candidate in second.best
    ]
    assert first.unique_evaluations == config.evaluation_budget
    assert len(first.validation) == config.validation_top_k
    assert all("fitness" in item for item in first.validation)
    assert len({candidate.expression for candidate in first.best}) == len(first.best)
    assert all(
        EvolutionSearch._depth(EvolutionSearch._validated(candidate.expression))
        <= config.max_depth
        for candidate in first.best
    )


def test_random_search_rejects_zero_evaluation_budget():
    try:
        RandomSearchConfig(evaluation_budget=0)
    except ValueError as error:
        assert "evaluation_budget" in str(error)
    else:
        raise AssertionError("Random search must reject an empty budget")
