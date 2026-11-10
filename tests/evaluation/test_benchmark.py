import json

import pandas as pd
import pytest

from python.evaluation import benchmark
from python.evaluation.engines import cpp_engine_error
from python.search.evolution import EvolutionSearch
from tests.stats.test_comparison_filter_and_resume import make_split_dataset

needs_cpp = pytest.mark.skipif(
    cpp_engine_error() is not None, reason="C++ module alpha_engine_cpp is not built (see README)"
)


def test_formula_set_is_fixed_and_mixed():

    formulas = benchmark.benchmark_formulas(20, seed=2026)

    assert formulas == benchmark.benchmark_formulas(20, seed=2026)
    assert len(formulas) == len(set(formulas)) == 20
    from_pool = [f for f in formulas if f in EvolutionSearch.SEEDS]
    assert len(from_pool) == 10
    assert from_pool != list(EvolutionSearch.SEEDS[:10])  # spread over the pool


@needs_cpp
def test_benchmark_writes_table_1(tmp_path, capsys):

    dataset = make_split_dataset(tmp_path / "data.parquet")

    table_file = benchmark.main([
        "--dataset", str(dataset), "--output-dir", str(tmp_path / "bench"),
        "--formulas", "4", "--repeats", "2", "--with-search", "--search-seed", "3",
    ])

    table = table_file.read_text(encoding="utf-8")
    assert table.startswith("## Table 1 — Fitness evaluation speed, Python vs C++")
    assert "| Fitness of one formula, median |" in table
    assert "Full evolutionary search, seed 3" in table
    assert "same top 10 formulas" in table or "top 10 formulas differ" in table
    assert table.count("| `") == 4  # one per-formula row each
    raw = json.loads((tmp_path / "bench" / "benchmark.json").read_text(encoding="utf-8"))
    assert raw["summary"]["all_agree"] is True
    assert raw["repeats"] == 2
    assert all(len(row["fitness_seconds"]["cpp"]) == 2 for row in raw["formulas"])
    assert raw["machine"]["logical_cpus"]
    assert "Table 1:" in capsys.readouterr().out


def test_benchmark_needs_the_cpp_engine(tmp_path, monkeypatch):

    monkeypatch.setattr(benchmark, "cpp_engine_error", lambda: "the C++ engine is not built")

    with pytest.raises(SystemExit):
        benchmark.main(["--dataset", str(tmp_path / "missing.parquet")])


def test_disagreeing_engines_are_reported(tmp_path):

    data = pd.read_parquet(make_split_dataset(tmp_path / "data.parquet"))
    rows = [{
        "formula": "CLOSE", "depth": 1, "nodes": 1, "window_sum": 0,
        "icir_python": 0.5, "icir_cpp": 0.4, "num_days": 10, "engines_agree": False,
        "fitness_median": {"python": 0.2, "cpp": 0.1}, "alpha_median": {"python": 0.1, "cpp": 0.05},
    }] * 2
    result = {"setup_seconds": {"python": 1.0, "cpp": 1.0}, "formulas": rows, "repeats": 1,
              "horizon": 20, "split": "train", "rows": len(data), "split_rows": len(data)}

    table = benchmark.table_markdown(result, benchmark.machine_info(), "data.parquet")

    assert "The engines disagreed" in table
    assert "| Fitness of one formula, median | 200.0 ms | 100.0 ms | 2.0× |" in table
