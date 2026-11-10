import json
import re

import pytest

from python import run_pipeline
from python.search import run_comparison
from tests.stats.test_comparison_filter_and_resume import make_split_dataset


def _argv(tmp_path, *extra):
    return [
        "--dataset", str(tmp_path / "data.parquet"), "--output-dir", str(tmp_path / "out"),
        "--seeds", "3", "--population-size", "6", "--generations", "1", "--elite-count", "2",
        "--top-k", "3", "--ledger", str(tmp_path / "ledger.jsonl"), *extra,
    ]


def _llm_argv(tmp_path):
    return _argv(
        tmp_path, "--llm-rate", "0.3", "--llm-backend", "mock",
        "--llm-cache", str(tmp_path / "llm_cache.jsonl"),
        "--llm-failures", str(tmp_path / "llm_failures.jsonl"),
        "--ablation-seeds", "3",
    )


@pytest.fixture
def four_methods(tmp_path):
    make_split_dataset(tmp_path / "data.parquet")
    files = run_pipeline.main(_llm_argv(tmp_path))
    return tmp_path, files


def test_one_command_writes_summary_report_and_tables(tmp_path, capsys):

    make_split_dataset(tmp_path / "data.parquet")

    files = run_pipeline.main(_argv(tmp_path))

    out = tmp_path / "out"
    assert files == {
        "summary": out / "comparison_summary.json",
        "report": out / "comparison_summary.html",
        "tables": out / "tables.md",
    }
    summary = json.loads(files["summary"].read_text(encoding="utf-8"))
    assert summary["protocol"]["methods"] == ["evolutionary", "random"]
    report = files["report"].read_text(encoding="utf-8")
    assert "Table 2 — Effect of the statistical filter" in report
    assert "Evolution + LLM" not in report  # no LLM section without --llm-rate
    assert "| Evolution | 1 |" in files["tables"].read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    assert "HTML report:" in printed and "Table 3" in printed


def test_report_only_rebuilds_without_searching(four_methods, monkeypatch):

    tmp_path, files = four_methods
    files["report"].unlink()
    files["tables"].unlink()
    monkeypatch.setattr(
        run_comparison, "run",
        lambda *args: (_ for _ in ()).throw(AssertionError("--report-only must not search")),
    )

    again = run_pipeline.main(_argv(tmp_path, "--report-only", "--title", "Final run"))

    assert again["report"].is_file() and again["tables"].is_file()
    assert "<title>Final run</title>" in again["report"].read_text(encoding="utf-8")


def test_report_only_needs_a_summary(tmp_path):

    with pytest.raises(SystemExit):
        run_pipeline.main(_argv(tmp_path, "--report-only"))


def test_report_covers_all_four_methods_filter_and_llm(four_methods):

    _, files = four_methods
    report = files["report"].read_text(encoding="utf-8")
    summary = json.loads(files["summary"].read_text(encoding="utf-8"))
    run = summary["paired_runs"][0]

    for label in ("Evolution", "Random", "Evolution + LLM", "Evolution + LLM, no retrieval"):
        assert f"<td>{label}</td>" in report  # a row in Tables 2 and 3
    # LLM section with the counts of both LLM methods.
    assert "<h2>LLM operator</h2>" in report
    assert "After node_type fix" in report
    calls = run["evolutionary_llm"]["llm"]["calls"]
    assert re.search(rf"<td>Evolution \+ LLM</td><td[^>]*>3</td><td[^>]*>mock</td><td[^>]*>yes</td><td[^>]*>{calls}</td>", report)
    assert re.search(r"<td>Evolution \+ LLM, no retrieval</td><td[^>]*>3</td><td[^>]*>mock</td><td[^>]*>no</td>", report)
    # Formulas carry their filter verdicts, and the badge says test was used.
    if run["evolutionary"]["filter"]["passed_filter"]:
        assert "passed filter" in report and "test ICIR" in report
    assert "TEST SPLIT CHECKED ONCE" in report
    # Charts get every method with its own colour.
    data = json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>', report, re.S).group(1))
    assert data["methods"] == summary["protocol"]["methods"]
    assert len(set(data["colors"].values())) == 4


def test_llm_formulas_are_tagged(four_methods):

    _, files = four_methods
    from python.search.report_comparison import _load_json, _prepare_view_data

    view = _prepare_view_data(files["summary"], _load_json(files["summary"]))
    details = view["paired_runs"][0]["candidate_details"]["evolutionary_llm"]
    report = files["report"].read_text(encoding="utf-8")

    assert all(item["passed_filter"] is not None for item in details)
    if any(item["from_llm"] for item in details):
        assert '<span class="tag llm">LLM</span>' in report
