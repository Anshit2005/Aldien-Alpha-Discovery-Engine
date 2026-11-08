from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from python.evaluation.fitness import FitnessResult
from python.stats.ic_store import (
    load_daily_ic,
    save_daily_ic,
    sidecar_path,
    table_from_candidates,
)
from python.stats.test_split import OneLookLedger, check_on_test, run_id


def _fitness(icir, daily_ic=None):
    return FitnessResult(
        horizon=20, mean_ic=0.01 * np.sign(icir) if icir == icir else float("nan"),
        ic_std=0.1, icir=icir, num_days=3, num_observations=30, coverage=1.0,
        daily_ic=daily_ic,
    )


def _candidate(expression, icir, values=None, dates=None):
    series = pd.Series(values, index=pd.DatetimeIndex(dates)) if values is not None else None
    return SimpleNamespace(expression=expression, fitness=_fitness(icir, series))


def test_table_aligns_dates_and_marks_failed_evaluations(tmp_path):

    candidates = [
        _candidate("CLOSE", 0.3, [0.1, 0.2], ["2020-01-01", "2020-01-02"]),
        _candidate("OPEN", -0.2, [0.3], ["2020-01-03"]),
        _candidate("LOG(SUB(OPEN,CLOSE))", float("nan")),  # failed evaluation
    ]

    table = table_from_candidates(candidates)

    assert table.expressions == ["CLOSE", "OPEN", "LOG(SUB(OPEN,CLOSE))"]
    assert table.directions.tolist() == [1, -1, 0]
    assert len(table.dates) == 3
    np.testing.assert_array_equal(table.ic[0], [0.1, 0.2, np.nan])
    np.testing.assert_array_equal(table.ic[1], [np.nan, np.nan, 0.3])
    assert np.isnan(table.ic[2]).all()

    path = save_daily_ic(sidecar_path(tmp_path / "evolution_seed_17.json"), table)
    loaded = load_daily_ic(path)

    assert path.name == "evolution_seed_17_daily_ic.npz"
    assert loaded.expressions == table.expressions
    assert loaded.directions.tolist() == [1, -1, 0]
    assert loaded.dates.equals(table.dates)
    np.testing.assert_array_equal(loaded.ic, table.ic)
    assert loaded.series("OPEN").iloc[-1] == 0.3


class FakeEvaluator:
    def __init__(self, icir_by_expression):
        self.icir_by_expression = icir_by_expression
        self.calls = []

    def evaluate(self, ast, horizon, split):
        self.calls.append((str(ast), split))
        icir = self.icir_by_expression[str(ast)]
        return _fitness(icir)


FORMULAS = [{"expression": "CLOSE", "direction": 1}, {"expression": "OPEN", "direction": -1}]


def test_test_split_is_checked_once_and_reused(tmp_path):

    ledger_path = tmp_path / "ledger.jsonl"
    evaluator = FakeEvaluator({"CLOSE": 0.2, "OPEN": 0.1})

    first = check_on_test(evaluator, FORMULAS, 20, "run1", OneLookLedger(ledger_path))

    assert evaluator.calls == [("CLOSE", "test"), ("OPEN", "test")]
    assert first["evaluated"] == 2
    # CLOSE kept its positive train direction; OPEN was negative on train and
    # positive on test, so its relationship flipped.
    assert [result["held"] for result in first["results"]] == [True, False]
    assert first["held_count"] == 1
    assert first["results"][1]["directional"]["icir"] == pytest.approx(-0.1)

    # Same run again, from a new process: nothing is evaluated a second time.
    second = check_on_test(evaluator, FORMULAS, 20, "run1", OneLookLedger(ledger_path))

    assert len(evaluator.calls) == 2
    assert second["reused_from_ledger"] == 2
    assert second["results"] == first["results"]


def test_new_formulas_on_a_checked_run_need_permission(tmp_path):

    ledger = OneLookLedger(tmp_path / "ledger.jsonl")
    evaluator = FakeEvaluator({"CLOSE": 0.2, "OPEN": 0.1, "HIGH": 0.3})
    check_on_test(evaluator, FORMULAS[:1], 20, "run1", ledger)

    with pytest.raises(RuntimeError, match="second look"):
        check_on_test(evaluator, FORMULAS, 20, "run1", ledger)

    result = check_on_test(evaluator, FORMULAS, 20, "run1", ledger, allow_second_look=True)
    assert [item["second_look"] for item in result["results"]] == [False, True]

    # A different run is a first look.
    other = check_on_test(evaluator, [{"expression": "HIGH", "direction": 1}], 20, "run2", ledger)
    assert other["results"][0]["second_look"] is False


def test_run_id_is_stable_and_content_sensitive():

    payload = {"best": [{"expression": "CLOSE"}], "config": {"seed": 17}}

    assert run_id(payload) == run_id({"config": {"seed": 17}, "best": [{"expression": "CLOSE"}]})
    assert run_id(payload) != run_id({**payload, "config": {"seed": 29}})
    assert len(run_id(payload)) == 16
