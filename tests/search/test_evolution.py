import pandas as pd

from python.search.evolution import EvolutionConfig, EvolutionSearch


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


def test_evolution_search_is_reproducible_and_reports_validation_separately():
    config = EvolutionConfig(
        population_size=8,
        generations=2,
        elite_count=2,
        tournament_size=2,
        max_depth=5,
        seed=29,
        validation_top_k=3,
    )
    first = EvolutionSearch(make_research_data(), config).run()
    second = EvolutionSearch(make_research_data(), config).run()

    assert [item.expression for item in first.best] == [item.expression for item in second.best]
    assert [item.score for item in first.best] == [item.score for item in second.best]
    assert len(first.generations) == config.generations + 1
    assert len(first.validation) == 3
    assert first.unique_evaluations >= config.population_size
    assert all(item.fitness.num_days <= 4 for item in first.best)


def test_evolution_only_accepts_train_for_search():
    try:
        EvolutionConfig(search_split="validation")
    except ValueError as error:
        assert "train split" in str(error)
    else:
        raise AssertionError("Evolution search must be restricted to train")


def test_search_reports_progress_to_optional_callback():
    messages = []
    config = EvolutionConfig(
        population_size=4,
        generations=0,
        elite_count=1,
        max_depth=4,
        validation_top_k=1,
        progress_interval_seconds=0.001,
    )
    EvolutionSearch(
        make_research_data(), config, progress_callback=messages.append
    ).run()

    assert any("Starting evolutionary search" in message for message in messages)
    assert any("Generation 1/1 complete" in message for message in messages)
    assert any("validation does not affect selection" in message for message in messages)
    assert any("Progress:" in message for message in messages)
