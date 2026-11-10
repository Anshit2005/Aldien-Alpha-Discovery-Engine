import argparse
import json
import sys

import pandas as pd
import pytest

from python.evaluation import engines
from python.evaluation.fitness import FitnessEvaluator
from python.search import run_comparison
from python.search.evolution import EvolutionConfig, EvolutionSearch
from python.search.random_search import RandomSearch, RandomSearchConfig
from tests.stats.test_comparison_filter_and_resume import _forbid_searches, _run, make_split_dataset

needs_cpp = pytest.mark.skipif(
    engines.cpp_engine_error() is not None, reason="C++ module alpha_engine_cpp is not built (see README)"
)


@pytest.mark.parametrize("config", [EvolutionConfig, RandomSearchConfig])
def test_unknown_engine_is_rejected(config):

    with pytest.raises(ValueError, match="engine"):
        config(engine="gpu")


def test_python_engine_is_the_default(tmp_path):

    data = pd.read_parquet(make_split_dataset(tmp_path / "data.parquet"))

    assert EvolutionConfig().engine == "python" and RandomSearchConfig().engine == "python"
    assert type(engines.make_fitness_evaluator(data)) is FitnessEvaluator
    with pytest.raises(ValueError, match="engine must be one of"):
        engines.make_fitness_evaluator(data, "gpu")


def test_missing_cpp_build_gives_a_clear_error(tmp_path, monkeypatch):

    data = pd.read_parquet(make_split_dataset(tmp_path / "data.parquet"))
    monkeypatch.setattr(engines, "cpp_engine_error", lambda: "the C++ engine is not built")

    with pytest.raises(RuntimeError, match="--engine cpp: the C\\+\\+ engine is not built"):
        engines.make_fitness_evaluator(data, "cpp")
    parser = argparse.ArgumentParser()
    engines.add_engine_argument(parser)
    with pytest.raises(SystemExit):
        engines.check_engine_argument(parser, parser.parse_args(["--engine", "cpp"]))


@needs_cpp
def test_both_engines_find_the_same_formulas(tmp_path):

    data = pd.read_parquet(make_split_dataset(tmp_path / "data.parquet", days=90, stocks=12))
    settings = dict(population_size=10, generations=3, elite_count=2, seed=5, validation_top_k=3)

    python_run = EvolutionSearch(data, EvolutionConfig(**settings)).run()
    cpp_run = EvolutionSearch(data, EvolutionConfig(**settings, engine="cpp")).run()

    from python.evaluation.cpp_fitness import CppFitnessEvaluator

    assert isinstance(EvolutionSearch(data, EvolutionConfig(engine="cpp")).evaluator, CppFitnessEvaluator)
    assert [c.expression for c in cpp_run.best] == [c.expression for c in python_run.best]
    assert [c.score for c in cpp_run.best] == pytest.approx([c.score for c in python_run.best], rel=1e-7)
    assert cpp_run.unique_evaluations == python_run.unique_evaluations

    random_python = RandomSearch(data, RandomSearchConfig(evaluation_budget=30, seed=5)).run()
    random_cpp = RandomSearch(data, RandomSearchConfig(evaluation_budget=30, seed=5, engine="cpp")).run()
    assert [c.expression for c in random_cpp.best] == [c.expression for c in random_python.best]


@needs_cpp
def test_comparison_on_cpp_reuses_python_runs(tmp_path, monkeypatch):

    dataset = make_split_dataset(tmp_path / "data.parquet")
    output_dir, ledger = tmp_path / "out", tmp_path / "ledger.jsonl"
    first = _run(monkeypatch, dataset, output_dir, ledger)
    _forbid_searches(monkeypatch)

    again = _run(monkeypatch, dataset, output_dir, ledger, "--engine", "cpp")

    assert again["protocol"]["engine"] == "cpp"
    assert again["paired_runs"][0]["evolutionary"] == first["paired_runs"][0]["evolutionary"]
    # The reused runs still say which engine produced them.
    payload = json.loads((output_dir / "evolution_seed_3.json").read_text(encoding="utf-8"))
    assert payload["config"]["engine"] == "python"


@needs_cpp
def test_comparison_runs_end_to_end_on_cpp(tmp_path, monkeypatch):

    dataset = make_split_dataset(tmp_path / "data.parquet")
    output_dir, ledger = tmp_path / "out", tmp_path / "ledger.jsonl"

    summary = _run(monkeypatch, dataset, output_dir, ledger, "--engine", "cpp")

    for name in ("evolution_seed_3.json", "random_seed_3.json"):
        payload = json.loads((output_dir / name).read_text(encoding="utf-8"))
        assert payload["config"]["engine"] == "cpp"
    run = summary["paired_runs"][0]
    assert run["evolutionary"]["filter"]["num_tests"] == run["evaluation_budget"]


def test_cli_rejects_cpp_when_not_built(tmp_path, monkeypatch):

    dataset = make_split_dataset(tmp_path / "data.parquet")
    monkeypatch.setattr(engines, "cpp_engine_error", lambda: "the C++ engine is not built")
    monkeypatch.setattr(sys, "argv", [
        "run_comparison", "--dataset", str(dataset), "--output-dir", str(tmp_path / "out"),
        "--engine", "cpp",
    ])

    with pytest.raises(SystemExit):
        run_comparison.main()
