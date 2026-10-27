import hashlib

import pytest

from python.dsl.json_io import dumps
from python.dsl.parser import parse
from python.llm.client import MockClient
from python.llm.operator import LLMProposalOperator
from python.search.evolution import EvolutionConfig, EvolutionSearch
from python.search.run_comparison import _metrics
from tests.search.test_evolution import make_research_data


BASE = dict(
    population_size=8, generations=3, elite_count=2, tournament_size=2,
    max_depth=5, seed=29, validation_top_k=3,
)

# Every formula evaluated by this configuration before the LLM operator
# existed. llm_rate=0 must reproduce the same random stream exactly.
BASELINE_EVALUATIONS = 23
BASELINE_DIGEST = "ce8e8f3b08fbb61b50f61d32bccc220a5892d92115d3a5c5ab3d889fee9074ff"
BASELINE_BEST = [
    "SUB(DIV(SUB(HIGH,LOW),CLOSE),VOLUME)",
    "DIV(SUB(HIGH,LOW),CLOSE)",
    "DIV(SUB(HIGH,LOW),ADD(HIGH,LOW))",
]


def _run(llm_rate=0.0, operator=None, **overrides):
    config = EvolutionConfig(
        **{**BASE, **overrides},
        llm_rate=llm_rate,
        llm_backend="mock" if llm_rate > 0 else "none",
    )
    search = EvolutionSearch(make_research_data(), config, llm_operator=operator)
    return search, search.run()


def test_llm_rate_zero_reproduces_the_baseline_exactly():

    search, result = _run(llm_rate=0.0)
    evaluated = list(search._fitness_cache)

    assert len(evaluated) == BASELINE_EVALUATIONS
    assert hashlib.sha256("\n".join(evaluated).encode()).hexdigest() == BASELINE_DIGEST
    assert [item.expression for item in result.best] == BASELINE_BEST
    assert "llm" not in result.to_dict()
    assert not any(item.from_llm for item in result.best)


def test_llm_rate_zero_never_calls_the_operator():

    client = MockClient()
    _run(llm_rate=0.0, operator=LLMProposalOperator(client))

    assert client.calls == 0


def test_mock_llm_children_enter_the_population_and_are_reported():

    client = MockClient()
    search, result = _run(llm_rate=0.5, operator=LLMProposalOperator(client))
    payload = result.to_dict()

    assert client.calls > 0
    llm = payload["llm"]
    assert llm["rate"] == 0.5
    assert llm["backend"] == "mock"
    assert llm["operator"]["calls"] == client.calls + llm["operator"]["cache_hits"]
    assert llm["children_added"] > 0
    assert llm["unique_llm_evaluations"] == len(search._llm_expressions) > 0
    assert all(
        search._fitness_cache[expression].from_llm for expression in search._llm_expressions
    )
    assert llm["top_k_from_llm"] == [item.expression for item in result.best if item.from_llm]
    assert all("llm_formulas_in_population" in item for item in payload["generations"][1:])
    assert all("from_llm" in item for item in payload["best"])


def test_llm_runs_are_reproducible():

    _, first = _run(llm_rate=0.5, operator=LLMProposalOperator(MockClient()))
    _, second = _run(llm_rate=0.5, operator=LLMProposalOperator(MockClient()))

    assert [item.expression for item in first.best] == [item.expression for item in second.best]
    assert first.llm == second.llm


def test_failed_proposals_fall_back_to_random_variation():

    client = MockClient(["not a formula"])
    operator = LLMProposalOperator(client)
    search, result = _run(llm_rate=1.0, operator=operator)

    assert operator.accepted == 0
    assert result.llm["children_added"] == 0
    assert not search._llm_expressions
    # The population is still filled by the ordinary operators.
    assert all(item["population_size"] > 1 for item in result.generations)


def test_proposals_already_evaluated_are_not_credited_to_the_llm():

    # Always proposes a seed formula, which generation 0 has already scored.
    answer = dumps(parse("RANK(CLOSE)"))
    operator = LLMProposalOperator(MockClient([answer]))
    search, result = _run(llm_rate=1.0, operator=operator)

    assert "RANK(CLOSE)" not in search._llm_expressions
    assert result.llm["children_already_evaluated"] >= 1
    assert not search._fitness_cache["RANK(CLOSE)"].from_llm


def test_llm_config_is_validated():

    with pytest.raises(ValueError, match="llm_rate must be"):
        EvolutionConfig(llm_rate=1.5, llm_backend="mock")
    with pytest.raises(ValueError, match="needs an llm_backend"):
        EvolutionConfig(llm_rate=0.3)
    with pytest.raises(ValueError, match="needs an llm_operator"):
        EvolutionSearch(make_research_data(), EvolutionConfig(llm_rate=0.3, llm_backend="mock"))


def test_comparison_metrics_count_llm_formulas():

    payload = {
        "best": [
            {"expression": "CLOSE", "direction": 1, "from_llm": True, "fitness": {"icir": 0.5}},
            {"expression": "OPEN", "direction": 1, "fitness": {"icir": 0.4}},
        ],
        "validation": [],
        "unique_train_evaluations": 2,
        "elapsed_seconds": 1.0,
    }

    assert _metrics(payload, top_k=2)["top_k_from_llm"] == 1
