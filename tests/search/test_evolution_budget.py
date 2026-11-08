import pytest

from python.llm.client import MockClient
from python.llm.operator import LLMProposalOperator
from python.search.evolution import EvolutionConfig, EvolutionSearch
from tests.search.test_evolution import make_research_data


BASE = dict(
    population_size=8, generations=3, elite_count=2, tournament_size=2,
    max_depth=5, seed=29, validation_top_k=3,
)


def _run(evaluation_budget, llm=False, **overrides):
    config = EvolutionConfig(
        **{**BASE, **overrides},
        evaluation_budget=evaluation_budget,
        llm_rate=0.3 if llm else 0.0,
        llm_backend="mock" if llm else "none",
    )
    operator = LLMProposalOperator(MockClient()) if llm else None
    return EvolutionSearch(make_research_data(), config, llm_operator=operator).run()


@pytest.mark.parametrize("llm", [False, True])
def test_search_stops_at_exactly_the_budget(llm):

    # Without a budget this configuration evaluates 23 formulas.
    result = _run(evaluation_budget=15, llm=llm)

    assert result.unique_evaluations == 15
    assert len(result.evaluated) == 15
    assert result.to_dict()["config"]["evaluation_budget"] == 15


def test_search_runs_extra_generations_to_spend_the_budget():

    result = _run(evaluation_budget=40)

    assert result.unique_evaluations == 40
    # The configured 3 generations (4 populations) evaluate only 23 formulas.
    assert len(result.generations) > 4


def test_unreachable_budget_stops_at_the_generation_cap():

    # A 2-formula population cannot find 10,000 new formulas in 10 x 1 generations.
    result = _run(evaluation_budget=10_000, population_size=2, elite_count=1, generations=1)

    assert len(result.generations) == 11
    assert result.unique_evaluations < 10_000


def test_budget_must_be_positive():

    with pytest.raises(ValueError, match="evaluation_budget"):
        EvolutionConfig(evaluation_budget=0)
