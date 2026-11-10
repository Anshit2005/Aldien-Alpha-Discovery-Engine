import json

import pandas as pd
import pytest

from python.stats import holdout
from python.stats.test_split import OneLookLedger
from tests.stats.test_comparison_filter_and_resume import _run, make_split_dataset


def make_holdout_dataset(path, days=90, holdout_days=20):
    """The comparison test panel with its last days moved into a holdout split."""

    make_split_dataset(path, days=days)
    data = pd.read_parquet(path)
    last = sorted(data["datetime"].unique())[-holdout_days:]
    rows = data["datetime"].isin(last)
    data.loc[rows, "split"] = "holdout"
    for horizon in (1, 5, 20, 30):
        data.loc[rows, f"target_split_{horizon}d"] = "holdout"
    data.to_parquet(path)
    return path


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    dataset = make_holdout_dataset(tmp_path / "data.parquet")
    output_dir = tmp_path / "out"
    _run(monkeypatch, dataset, output_dir, tmp_path / "test_ledger.jsonl", "--seeds", "3", "4")
    return dataset, output_dir / "comparison_summary.json", tmp_path / "holdout_ledger.jsonl"


def _main(dataset, summary, ledger, *extra):
    return holdout.main([
        "--summary", str(summary), "--dataset", str(dataset), "--ledger", str(ledger), *extra,
    ])


def test_selection_uses_each_runs_best_formula_without_the_holdout(experiment, monkeypatch):

    dataset, summary, ledger = experiment
    monkeypatch.setattr(holdout, "make_fitness_evaluator", lambda *a: pytest.fail("holdout touched"))

    selection_file = _main(dataset, summary, ledger, "--dry-run")

    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    assert set(selection["methods"]) == {"evolutionary", "random"}
    chosen = selection["methods"]["evolutionary"]["formulas"]
    assert chosen and all(item["test_icir"] > 0 for item in chosen)
    assert {seed for item in chosen for seed in item["seeds"]} <= {3, 4}
    for item in chosen:  # each run's finalist is its best test ICIR
        for seed in item["seeds"]:
            runs = holdout.run_candidates(seed, summary.parent / f"evolution_seed_{seed}_filtered.json")
            assert item["test_icir"] == runs[0]["test_icir"]
    assert not ledger.exists()
    assert not (summary.parent / "holdout_results.json").exists()


def test_holdout_is_checked_once_and_then_reused(experiment, monkeypatch):

    dataset, summary, ledger = experiment

    table_file = _main(dataset, summary, ledger)

    table = table_file.read_text(encoding="utf-8")
    assert table.startswith("## Holdout check (")
    assert "| Evolution |" in table
    output = json.loads((summary.parent / "holdout_results.json").read_text(encoding="utf-8"))
    checked = output["check"]["evaluated"]
    assert checked == len(holdout.unique_formulas(json.loads(
        (summary.parent / "holdout_selection.json").read_text(encoding="utf-8"))))
    assert checked > 0 and output["period"]["days"] > 0
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == checked

    # Same selection again: nothing is evaluated a second time.
    monkeypatch.setattr(holdout, "make_fitness_evaluator", lambda *a: object())
    _main(dataset, summary, ledger)
    again = json.loads((summary.parent / "holdout_results.json").read_text(encoding="utf-8"))
    assert again["check"]["evaluated"] == 0
    assert again["check"]["reused_from_ledger"] == checked
    assert again["methods"] == output["methods"]


def test_a_different_selection_is_a_refused_second_look(experiment):

    dataset, summary, ledger = experiment
    OneLookLedger(ledger).record("an-earlier-selection", {"expression": "CLOSE"})

    with pytest.raises(SystemExit):
        _main(dataset, summary, ledger)

    _main(dataset, summary, ledger, "--allow-second-look")
    output = json.loads((summary.parent / "holdout_results.json").read_text(encoding="utf-8"))
    assert all(result["second_look"] for result in output["check"]["results"])
    assert "second look" in (summary.parent / "holdout.md").read_text(encoding="utf-8")


def test_old_or_unfiltered_comparisons_are_refused(experiment, tmp_path):

    dataset, summary, ledger = experiment
    payload = json.loads(summary.read_text(encoding="utf-8"))
    del payload["protocol"]["min_train_ic_day_share"]
    old = summary.with_name("old_summary.json")
    old.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(holdout.HoldoutError, match="IC-day scoring rule"):
        holdout.select_formulas(old)

    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["protocol"]["statistical_filter"] = None
    summary.with_name("unfiltered.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(holdout.HoldoutError, match="statistical filter"):
        holdout.select_formulas(summary.with_name("unfiltered.json"))


def test_identical_signals_count_once():

    def entry(expression, seed, icir=0.3):
        return {"expression": expression, "direction": 1, "seeds": [seed], "train_t_stat": 3.0,
                "train_p_value": 0.01, "test_icir": icir, "test_mean_ic": 0.02, "test_days": 900,
                "equivalents": []}

    merged = holdout.collapse([
        entry("DIV(CLOSE,2)", 3), entry("CLOSE", 4), entry("CLOSE", 5), entry("VOLUME", 3, icir=0.5),
    ])

    assert [item["expression"] for item in merged] == ["VOLUME", "CLOSE"]
    assert merged[1]["equivalents"] == ["DIV(CLOSE,2)"]
    assert merged[1]["seeds"] == [3, 4, 5]
