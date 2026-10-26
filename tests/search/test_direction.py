import math

import pytest

from python.search.direction import directional_fitness, signal_direction
from python.search.evolution import EvolutionConfig, EvolutionSearch
from python.search.random_search import RandomSearch, RandomSearchConfig
from python.search.run_comparison import _metrics
from tests.search.test_evolution import make_research_data


@pytest.mark.parametrize(
    "icir, expected",
    [
        (0.3, 1),
        (-0.2, -1),
        (0.0, 0),
        (float("nan"), 0),
        (None, 0),
        ("not a number", 0),
    ],
)
def test_signal_direction(icir, expected):

    assert signal_direction(icir) == expected


def test_directional_fitness_flips_signed_statistics():

    fitness = {"mean_ic": -0.02, "icir": -0.4, "num_days": 10}

    assert directional_fitness(fitness, -1) == {"mean_ic": 0.02, "icir": 0.4}
    assert directional_fitness(fitness, 1) == {"mean_ic": -0.02, "icir": -0.4}


def test_directional_fitness_is_undefined_without_direction():

    oriented = directional_fitness({"mean_ic": 0.01, "icir": 0.2}, 0)

    assert math.isnan(oriented["mean_ic"])
    assert math.isnan(oriented["icir"])


def test_directional_fitness_handles_missing_values():

    oriented = directional_fitness({"mean_ic": None, "icir": float("nan")}, 1)

    assert math.isnan(oriented["mean_ic"])
    assert math.isnan(oriented["icir"])


def _check_payload(payload):

    by_expression = {item["expression"]: item for item in payload["best"]}

    for item in payload["best"]:
        assert item["direction"] == signal_direction(item["fitness"]["icir"])

    for result in payload["validation"]:
        candidate = by_expression[result["expression"]]
        assert result["direction"] == candidate["direction"]
        expected = directional_fitness(result["fitness"], candidate["direction"])
        for name in ("mean_ic", "icir"):
            a, b = result["directional"][name], expected[name]
            assert (math.isnan(a) and math.isnan(b)) or a == b


def test_evolution_output_records_direction():

    config = EvolutionConfig(
        population_size=8,
        generations=1,
        elite_count=2,
        tournament_size=2,
        max_depth=4,
        seed=5,
        validation_top_k=4,
    )

    _check_payload(EvolutionSearch(make_research_data(), config).run().to_dict())


def test_random_output_records_direction():

    config = RandomSearchConfig(
        evaluation_budget=10,
        seed=5,
        max_depth=4,
        validation_top_k=4,
    )

    _check_payload(RandomSearch(make_research_data(), config).run().to_dict())


def test_comparison_metrics_use_train_direction():

    # Two formulas: one held its train relationship on validation, the
    # other was negative on train and stayed negative on validation.
    # Both held, so both are positive once the train direction is applied,
    # while the raw signed mean cancels to zero.
    payload = {
        "best": [
            {"expression": "CLOSE", "direction": 1, "fitness": {"icir": 0.5}},
            {"expression": "OPEN", "direction": -1, "fitness": {"icir": -0.5}},
        ],
        "validation": [
            {"expression": "CLOSE", "fitness": {"icir": 0.2, "mean_ic": 0.01}},
            {"expression": "OPEN", "fitness": {"icir": -0.2, "mean_ic": -0.01}},
        ],
        "unique_train_evaluations": 2,
        "elapsed_seconds": 1.0,
    }

    metrics = _metrics(payload, top_k=2)

    assert metrics["mean_validation_icir_top_k"] == pytest.approx(0.0)
    assert metrics["mean_validation_directional_icir_top_k"] == pytest.approx(0.2)
    assert metrics["best_validation_directional_icir_top_k"] == pytest.approx(0.2)


def test_comparison_metrics_infer_direction_for_older_results():

    payload = {
        "best": [{"expression": "OPEN", "fitness": {"icir": -0.5}}],
        "validation": [{"expression": "OPEN", "fitness": {"icir": 0.3, "mean_ic": 0.01}}],
        "unique_train_evaluations": 1,
        "elapsed_seconds": 1.0,
    }

    metrics = _metrics(payload, top_k=1)

    # Negative on train, positive on validation: the relationship flipped.
    assert metrics["mean_validation_directional_icir_top_k"] == pytest.approx(-0.3)
