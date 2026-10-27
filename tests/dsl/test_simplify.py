import random

import numpy as np
import pytest

from python.dsl.parser import parse
from python.dsl.simplify import is_non_negative, simplify
from python.dsl.validator import validate
from python.evaluation.evaluator import AlphaEvaluator
from python.search.evolution import EvolutionSearch
from tests.evaluation.test_fitness import make_random_research_data


def _simplified(expression):
    return str(simplify(parse(expression)))


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("ABS(HIGH)", "HIGH"),
        ("DIV(SUB(ABS(HIGH),LOW),CLOSE)", "DIV(SUB(HIGH,LOW),CLOSE)"),
        ("ABS(TS_MEAN(VOLUME,20))", "TS_MEAN(VOLUME,20)"),
        ("ABS(TS_STD(SUB(CLOSE,OPEN),5))", "TS_STD(SUB(CLOSE,OPEN),5)"),
        ("ABS(RANK(SUB(CLOSE,OPEN)))", "RANK(SUB(CLOSE,OPEN))"),
        ("ABS(ABS(SUB(CLOSE,OPEN)))", "ABS(SUB(CLOSE,OPEN))"),
        ("SIGN(SIGN(SUB(CLOSE,OPEN)))", "SIGN(SUB(CLOSE,OPEN))"),
        ("RANK(RANK(VOLUME))", "RANK(VOLUME)"),
        ("ADD(CLOSE,0)", "CLOSE"),
        ("ADD(0,CLOSE)", "CLOSE"),
        ("SUB(CLOSE,0)", "CLOSE"),
        ("MUL(1,VOLUME)", "VOLUME"),
        ("DIV(VOLUME,1)", "VOLUME"),
        ("ADD(VOLUME,CLOSE)", "ADD(CLOSE,VOLUME)"),
        ("MUL(TS_MEAN(CLOSE,5),OPEN)", "MUL(OPEN,TS_MEAN(CLOSE,5))"),
        ("ADD(CLOSE,MUL(2,0.5))", "ADD(1,CLOSE)"),
        ("MUL(CLOSE,SUB(2,1))", "CLOSE"),
        ("TS_MEAN(ABS(CLOSE),20)", "TS_MEAN(CLOSE,20)"),
    ],
)
def test_simplify_rules(expression, expected):

    assert _simplified(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "ABS(SUB(CLOSE,OPEN))",       # can be negative
        "ABS(LOG(CLOSE))",            # can be negative
        "ABS(TS_MIN(SUB(CLOSE,OPEN),5))",
        "SUB(CLOSE,OPEN)",            # not commutative
        "DIV(CLOSE,0)",               # constant division by zero is not folded
        "LOG(-1)",                    # undefined, not folded
        "RANK(2)",                    # rank of a constant depends on the day
        "SIGN(ABS(SUB(CLOSE,OPEN)))",
        "TS_MEAN(TS_MEAN(CLOSE,5),5)",  # windows are left alone
    ],
)
def test_simplify_leaves_non_equivalent_forms(expression):

    assert _simplified(expression) == str(parse(expression))


def test_non_negative_inference():

    assert is_non_negative(parse("DIV(VOLUME,TS_MEAN(VOLUME,20))"))
    assert not is_non_negative(parse("SUB(HIGH,LOW)"))
    assert not is_non_negative(parse("TS_MAX(SIGN(SUB(CLOSE,OPEN)),5)"))


def test_simplify_keeps_seed_formulas_valid():

    for expression in EvolutionSearch.SEEDS:
        simplified = simplify(parse(expression))
        validate(simplified)
        assert simplify(simplified) == simplified


def _random_trees(count, seed=11):
    generator = EvolutionSearch.__new__(EvolutionSearch)
    generator.rng = random.Random(seed)
    return [generator._random_tree(5) for _ in range(count)]


def test_simplify_is_idempotent_on_random_trees():

    for tree in _random_trees(300):
        once = simplify(tree)
        assert simplify(once) == once


def test_simplified_formulas_evaluate_identically():

    # Same values and the same NaN positions, so coverage and IC are unchanged.
    evaluator = AlphaEvaluator(make_random_research_data(num_dates=12, num_instruments=6))
    changed = 0
    for tree in _random_trees(200, seed=5):
        try:
            validate(tree)
        except Exception:
            continue
        simplified = simplify(tree)
        if simplified == tree:
            continue
        changed += 1
        original = evaluator.evaluate(tree)["alpha"].to_numpy(dtype=float)
        rewritten = evaluator.evaluate(simplified)["alpha"].to_numpy(dtype=float)
        np.testing.assert_array_equal(np.isnan(original), np.isnan(rewritten), err_msg=str(tree))
        np.testing.assert_allclose(original, rewritten, rtol=1e-12, equal_nan=True, err_msg=str(tree))

    assert changed > 20
