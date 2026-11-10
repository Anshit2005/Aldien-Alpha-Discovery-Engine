import json
import sys

import numpy as np
import pandas as pd
import pytest

from python.search import run_comparison, tables
from python.stats import run_filter
from python.stats.ic_store import load_daily_ic, sidecar_path


def make_split_dataset(path, days=70, stocks=10, seed=0):
    """Small panel with train/validation/test splits and a strong CLOSE signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for day, date in enumerate(pd.bdate_range("2015-01-01", periods=days)):
        split = "train" if day < 45 else ("validation" if day < 57 else "test")
        for stock in range(stocks):
            close = 10.0 + stock + rng.normal(0, 0.3)
            row = {
                "datetime": date, "instrument": f"S{stock}", "open": close * 0.99,
                "high": close * 1.02, "low": close * 0.98, "close": close,
                "volume": float(rng.uniform(1e5, 1e6)), "split": split,
            }
            for horizon in (1, 5, 20, 30):
                row[f"forward_return_{horizon}d"] = 0.002 * close + rng.normal(0, 0.001)
                row[f"target_split_{horizon}d"] = split
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path)
    return path


def _run(monkeypatch, dataset, output_dir, ledger, *extra):
    argv = [
        "run_comparison", "--dataset", str(dataset), "--output-dir", str(output_dir),
        "--seeds", "3", "--population-size", "6", "--generations", "1",
        "--elite-count", "2", "--top-k", "3", "--ledger", str(ledger), *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    run_comparison.main()
    return json.loads((output_dir / "comparison_summary.json").read_text(encoding="utf-8"))


def _forbid_searches(monkeypatch):
    def fail(self):
        raise AssertionError("a finished run must not be recomputed")
    monkeypatch.setattr(run_comparison.EvolutionSearch, "run", fail)
    monkeypatch.setattr(run_comparison.RandomSearch, "run", fail)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    dataset = make_split_dataset(tmp_path / "data.parquet")
    output_dir = tmp_path / "out"
    ledger = tmp_path / "ledger.jsonl"
    summary = _run(monkeypatch, dataset, output_dir, ledger)
    return dataset, output_dir, ledger, summary


def test_comparison_saves_daily_ic_and_filters_every_method(experiment):

    _, output_dir, ledger, summary = experiment
    run = summary["paired_runs"][0]

    assert summary["protocol"]["statistical_filter"]["method"] == "hansen_hodrick"
    for method, stem in (("evolutionary", "evolution_seed_3"), ("random", "random_seed_3")):
        payload = json.loads((output_dir / f"{stem}.json").read_text(encoding="utf-8"))
        table = load_daily_ic(sidecar_path(output_dir / f"{stem}.json"))
        filtered = json.loads((output_dir / f"{stem}_filtered.json").read_text(encoding="utf-8"))

        # Every unique formula tried has a daily IC series and is counted.
        assert len(table) == payload["unique_train_evaluations"]
        assert set(table.expressions) >= {item["expression"] for item in payload["best"]}
        assert filtered["counts"]["num_tests"] == payload["unique_train_evaluations"]
        assert len(filtered["formulas"]) == len(table)
        counts = run[method]["filter"]
        assert counts["passed_filter"] == filtered["counts"]["passed_filter"]
        assert counts["checked_on_test"] == counts["passed_filter"]

    # The CLOSE signal is strong, so something passes and is checked on test.
    assert run["evolutionary"]["filter"]["passed_filter"] > 0
    assert ledger.is_file()


def test_rerun_reuses_finished_seeds_and_the_test_ledger(experiment, monkeypatch):

    dataset, output_dir, ledger, first = experiment
    ledger_lines = ledger.read_text(encoding="utf-8").count("\n")
    _forbid_searches(monkeypatch)

    second = _run(monkeypatch, dataset, output_dir, ledger)

    assert second["paired_runs"] == first["paired_runs"]
    # No second look at the test split.
    assert ledger.read_text(encoding="utf-8").count("\n") == ledger_lines


def test_interrupted_seed_finishes_only_the_missing_method(experiment, monkeypatch):

    dataset, output_dir, ledger, first = experiment
    (output_dir / "random_seed_3.json").unlink()
    monkeypatch.setattr(
        run_comparison.EvolutionSearch, "run",
        lambda self: (_ for _ in ()).throw(AssertionError("evolution was finished")),
    )

    second = _run(monkeypatch, dataset, output_dir, ledger)

    assert (output_dir / "random_seed_3.json").is_file()
    assert second["paired_runs"][0]["evolutionary"] == first["paired_runs"][0]["evolutionary"]


def test_runs_saved_before_a_setting_existed_are_still_reused(experiment, monkeypatch):

    dataset, output_dir, ledger, first = experiment
    path = output_dir / "evolution_seed_3.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["config"]["evaluation_budget"], payload["config"]["llm_retrieval"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    _forbid_searches(monkeypatch)

    second = _run(monkeypatch, dataset, output_dir, ledger)

    assert second["paired_runs"][0]["evolutionary"]["unique_evaluations"] == (
        first["paired_runs"][0]["evolutionary"]["unique_evaluations"]
    )


def test_changed_settings_refuse_to_reuse_results(experiment, monkeypatch):

    dataset, output_dir, ledger, _ = experiment

    with pytest.raises(SystemExit, match="different settings"):
        _run(monkeypatch, dataset, output_dir, ledger, "--min-coverage", "0.9")


def test_run_filter_cli_matches_the_comparison(experiment, monkeypatch, tmp_path):

    dataset, output_dir, ledger, _ = experiment
    source = output_dir / "evolution_seed_3.json"
    destination = tmp_path / "standalone_filtered.json"
    monkeypatch.setattr(sys, "argv", [
        "run_filter", "--input", str(source), "--output", str(destination),
        "--dataset", str(dataset), "--ledger", str(ledger),
    ])

    run_filter.main()

    standalone = json.loads(destination.read_text(encoding="utf-8"))
    built_in = json.loads((output_dir / "evolution_seed_3_filtered.json").read_text(encoding="utf-8"))
    assert standalone["run_id"] == built_in["run_id"]
    assert standalone["filter"]["passed"] == built_in["filter"]["passed"]
    assert standalone["test"]["reused_from_ledger"] == standalone["counts"]["checked_on_test"]


def _llm_flags(tmp_path):
    return (
        "--llm-rate", "0.3", "--llm-backend", "mock",
        "--llm-cache", str(tmp_path / "llm_cache.jsonl"),
        "--llm-failures", str(tmp_path / "llm_failures.jsonl"),
    )


def test_llm_rate_adds_a_third_method_and_reuses_the_other_two(experiment, monkeypatch, tmp_path):

    dataset, output_dir, ledger, first = experiment
    real_run = run_comparison.EvolutionSearch.run
    runs = []

    def only_llm_runs(self):
        assert self.config.llm_rate > 0, "plain evolution was finished already"
        runs.append(self.config)
        return real_run(self)

    monkeypatch.setattr(run_comparison.EvolutionSearch, "run", only_llm_runs)
    monkeypatch.setattr(
        run_comparison.RandomSearch, "run",
        lambda self: (_ for _ in ()).throw(AssertionError("random was finished already")),
    )

    summary = _run(monkeypatch, dataset, output_dir, ledger, *_llm_flags(tmp_path))

    run = summary["paired_runs"][0]
    budget = run["evaluation_budget"]
    assert summary["protocol"]["methods"] == ["evolutionary", "random", "evolutionary_llm"]
    assert summary["protocol"]["evolution_config"].get("llm_rate") is None
    assert summary["protocol"]["evolution_llm_config"]["llm_rate"] == 0.3
    # The two finished methods are reused unchanged.
    for method in ("evolutionary", "random"):
        assert run[method] == first["paired_runs"][0][method]
    # The LLM method ran once, with the same seed and the same budget.
    assert len(runs) == 1 and runs[0].seed == 3 and runs[0].evaluation_budget == budget
    llm = run["evolutionary_llm"]
    assert llm["unique_evaluations"] == budget
    assert llm["llm"]["calls"] > 0
    assert llm["filter"]["num_tests"] == budget
    payload = json.loads((output_dir / "evolution_llm_seed_3.json").read_text(encoding="utf-8"))
    assert payload["config"]["llm_backend"] == "mock"
    assert len(load_daily_ic(sidecar_path(output_dir / "evolution_llm_seed_3.json"))) == budget

    # A second LLM run reuses everything.
    _forbid_searches(monkeypatch)
    again = _run(monkeypatch, dataset, output_dir, ledger, *_llm_flags(tmp_path))
    assert again["paired_runs"] == summary["paired_runs"]


def test_ablation_seeds_add_a_no_retrieval_method_and_tables(experiment, monkeypatch, tmp_path):

    dataset, output_dir, ledger, _ = experiment

    summary = _run(
        monkeypatch, dataset, output_dir, ledger, *_llm_flags(tmp_path), "--ablation-seeds", "3"
    )

    run = summary["paired_runs"][0]
    assert summary["protocol"]["methods"][-1] == "evolutionary_llm_no_retrieval"
    assert summary["protocol"]["ablation_seeds"] == [3]
    ablation = run["evolutionary_llm_no_retrieval"]
    assert ablation["unique_evaluations"] == run["evaluation_budget"]
    assert ablation["llm"]["retrieval"] is False
    assert run["evolutionary_llm"]["llm"]["retrieval"] is True
    payload = json.loads(
        (output_dir / "evolution_llm_no_retrieval_seed_3.json").read_text(encoding="utf-8")
    )
    assert payload["config"]["llm_retrieval"] is False

    # Tables 2 and 3 have one row per method.
    text = tables.build(summary)
    for label in ("Evolution |", "Random |", "Evolution + LLM |", "Evolution + LLM, no retrieval |"):
        assert text.count(label) == 2
    assert "Table 2" in text and "Table 3" in text


def test_ablation_seeds_need_the_llm(experiment, monkeypatch):

    dataset, output_dir, ledger, _ = experiment

    with pytest.raises(SystemExit):
        _run(monkeypatch, dataset, output_dir, ledger, "--ablation-seeds", "3")


def test_tables_for_a_two_method_summary(experiment):

    _, _, _, summary = experiment

    text = tables.build(summary)

    assert "| Evolution | 1 |" in text and "| Random | 1 |" in text
    assert "Evolution + LLM" not in text


def test_filter_rejects_a_sidecar_from_another_run(experiment):

    _, output_dir, _, _ = experiment
    payload = json.loads((output_dir / "evolution_seed_3.json").read_text(encoding="utf-8"))
    table = load_daily_ic(sidecar_path(output_dir / "random_seed_3.json"))
    payload["unique_train_evaluations"] = len(table) + 1

    with pytest.raises(ValueError, match="does not match"):
        run_filter.filter_search(payload, table, run_filter.FilterConfig())
