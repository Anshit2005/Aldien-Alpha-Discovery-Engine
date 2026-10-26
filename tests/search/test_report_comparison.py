import json

import pytest

from python.search.report_comparison import generate_report


def test_comparison_report_renders_dashboard_and_embedded_data(tmp_path):
    source = tmp_path / "comparison_summary.json"
    output = tmp_path / "report.html"
    source.write_text(
        json.dumps({
            "dataset": {"path": "research.parquet", "date_min": "2020-01-01", "date_max": "2024-12-31", "sha256": "abc123", "rows": 100},
            "protocol": {"top_k": 2, "train_split": "train", "validation_split": "validation", "fitness_objective": "absolute_train_icir", "validation_used_for_selection": False, "evolution_seed_pool": "expanded", "evolution_config": {"horizon": 20, "max_depth": 8}},
            "paired_runs": [{
                "seed": 17,
                "evaluation_budget": 50,
                "evolutionary": {"best_train_abs_icir": 0.4, "mean_train_abs_icir_top_k": 0.3, "mean_validation_icir_top_k": 0.1, "mean_validation_abs_icir_top_k": 0.12, "train_to_validation_abs_icir_degradation": 0.18, "train_validation_direction_agreement_rate": 0.8, "unique_evaluations": 50, "elapsed_seconds": 12, "top_k_expression_diversity": {"mean_pairwise_operator_jaccard_distance": 0.4}},
                "random": {"best_train_abs_icir": 0.35, "mean_train_abs_icir_top_k": 0.25, "mean_validation_icir_top_k": 0.05, "mean_validation_abs_icir_top_k": 0.1, "train_to_validation_abs_icir_degradation": 0.15, "train_validation_direction_agreement_rate": 0.6, "unique_evaluations": 50, "elapsed_seconds": 10, "top_k_expression_diversity": {"mean_pairwise_operator_jaccard_distance": 0.3}},
            }],
        }),
        encoding="utf-8",
    )

    generate_report(source, output)
    report = output.read_text(encoding="utf-8")

    assert "Search experiment report" in report
    assert "Validation ICIR by seed" in report
    assert "mean_validation_icir_top_k" in report
    assert "research.parquet" in report
    assert "<script src=\"https://" not in report


def test_report_prefers_direction_adjusted_validation(tmp_path):
    source = tmp_path / "comparison_summary.json"
    output = tmp_path / "report.html"
    base = {"best_train_abs_icir": 0.4, "mean_train_abs_icir_top_k": 0.3, "unique_evaluations": 50, "elapsed_seconds": 12}
    source.write_text(
        json.dumps({
            "protocol": {"top_k": 2},
            "paired_runs": [{
                "seed": 17,
                "evaluation_budget": 50,
                # New summary: both metrics present.
                "evolutionary": {**base, "mean_validation_icir_top_k": -0.05, "mean_validation_directional_icir_top_k": 0.11},
                # Older summary: only the raw signed metric.
                "random": {**base, "mean_validation_icir_top_k": 0.07},
            }],
        }),
        encoding="utf-8",
    )

    from python.search.report_comparison import _prepare_view_data, _load_json

    view = _prepare_view_data(source, _load_json(source))
    run = view["paired_runs"][0]

    assert run["evolutionary"]["validation_score_top_k"] == 0.11
    assert run["evolutionary"]["validation_score_is_directional"] is True
    assert run["random"]["validation_score_top_k"] == 0.07
    assert run["random"]["validation_score_is_directional"] is False

    generate_report(source, output)
    assert "train direction" in output.read_text(encoding="utf-8")


def test_report_rejects_non_comparison_json(tmp_path):
    source = tmp_path / "wrong.json"
    source.write_text(json.dumps({"best": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="paired comparison JSON"):
        generate_report(source, tmp_path / "wrong.html")
