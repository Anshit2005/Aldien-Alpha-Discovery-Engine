from python.dsl.json_io import loads
from python.dsl.parser import parse
from python.dsl.validator import validate
from python.evaluation.fitness import FitnessResult
from python.llm.prompt import (
    _EXAMPLE,
    parent_expression_from_prompt,
    system_prompt,
    user_prompt,
)
from python.llm.retrieval import FactorRetriever, formula_tokens, load_library
from python.search.evolution import EvolutionSearch


PARENT = "DIV(SUB(HIGH,LOW),CLOSE)"

# Pinned so that accidental prompt changes are noticed: a changed prompt
# also changes every cache key, so earlier cached answers stop matching.
EXPECTED_USER_PROMPT = """Parent formula: DIV(SUB(HIGH,LOW),CLOSE)
Parent as JSON: {"node_type":"OPERATOR","value":"DIV","children":[{"node_type":"OPERATOR","value":"SUB","children":[{"node_type":"VARIABLE","value":"HIGH"},{"node_type":"VARIABLE","value":"LOW"}]},{"node_type":"VARIABLE","value":"CLOSE"}]}
Parent train ICIR: -0.1146 (high values predict low returns, defined on 96% of rows)

Related known factors:
- TS_MEAN(DIV(SUB(HIGH,LOW),CLOSE),20): Average relative daily range over 20 days.
- DIV(SUB(CLOSE,LOW),ADD(SUB(HIGH,LOW),0.001)): Where the close sits inside the day's range.
- DIV(SUB(HIGH,CLOSE),ADD(SUB(HIGH,LOW),0.001)): Gap between the high and the close as a share of range; selling pressure.
- DIV(SUB(CLOSE,OPEN),ADD(SUB(HIGH,LOW),0.001)): Open-to-close move as a share of the day's range (Alpha101 #101).

Recently rejected proposals (avoid these mistakes):
- {"node_type": "OPERATOR", "value": "DELAY"} -> invalid_formula: Unknown operator: DELAY

Task: propose ONE new formula that is a meaningful variation of the parent, for example a different window, normalisation, ranking, or an added volume or volatility term, that could predict 20-day forward returns better. It must differ from the parent. Reply with the JSON tree only."""


def _fitness(icir=-0.1146, coverage=0.962):
    return FitnessResult(
        horizon=20, mean_ic=-0.01, ic_std=0.09, icir=icir,
        num_days=2411, num_observations=10, coverage=coverage,
    )


def test_user_prompt_is_pinned():

    parent = parse(PARENT)
    failures = [{
        "response": '{"node_type": "OPERATOR", "value": "DELAY"}',
        "error": "invalid_formula: Unknown operator: DELAY",
    }]

    prompt = user_prompt(parent, _fitness(), FactorRetriever().similar(parent, 4), failures)

    assert prompt == EXPECTED_USER_PROMPT


def test_user_prompt_without_context_or_fitness():

    prompt = user_prompt(parse("CLOSE"), None, [], [])

    assert "Parent train ICIR: undefined" in prompt
    assert "Related known factors" not in prompt
    assert "Recently rejected" not in prompt
    assert parent_expression_from_prompt(prompt) == "CLOSE"


def test_long_failed_responses_are_shortened_to_one_line():

    failure = {"response": "x\n" * 200, "error": "no_json: reply contains no JSON object"}

    prompt = user_prompt(parse("CLOSE"), None, [], [failure])
    line = next(line for line in prompt.splitlines() if "no_json" in line)

    assert len(line) < 220
    assert "..." in line


def test_system_prompt_lists_the_grammar_and_a_valid_example():

    prompt = system_prompt(max_depth=6)

    for name in ("CLOSE", "VOLUME", "RANK", "DIV", "TS_STD", "252", "6 levels", "JSON tree only"):
        assert name in prompt
    assert str(loads(_EXAMPLE)) == "DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20))"


def test_library_formulas_are_valid_unique_and_include_the_seeds():

    library = load_library()
    expressions = [factor.expression for factor in library]

    assert len(library) >= 30
    assert len(set(expressions)) == len(expressions)
    assert len({factor.name for factor in library}) == len(library)
    assert set(EvolutionSearch.SEEDS) <= set(expressions)
    for factor in library:
        validate(parse(factor.expression))
        assert factor.meaning


def test_retrieval_prefers_shared_operators_and_skips_the_parent():

    retriever = FactorRetriever()
    parent = parse("DIV(SUB(VOLUME,TS_MEAN(VOLUME,20)),TS_MEAN(VOLUME,20))")

    similar = retriever.similar(parent, k=3)

    assert len(similar) == 3
    assert all(factor.expression != str(parent) for factor in similar)
    assert all("VOLUME" in factor.expression for factor in similar)
    assert formula_tokens(parent) == {"DIV", "SUB", "TS_MEAN", "VOLUME"}
