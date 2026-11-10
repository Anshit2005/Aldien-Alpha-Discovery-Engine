"""
Build the report tables from a comparison summary, as Markdown.

Table 2, filter effect: how many of the formulas each method evaluated pass
the statistical filter, and how many of those keep their direction on the
test split.
Table 3, search quality: train and validation quality of each method's top-k
formulas, averaged over seeds (mean ± standard deviation).

    python -m python.search.tables --input data/processed/experiment/comparison_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

LABELS = {
    "evolutionary": "Evolution",
    "random": "Random",
    "evolutionary_llm": "Evolution + LLM",
    "evolutionary_llm_no_retrieval": "Evolution + LLM, no retrieval",
}


def methods_of(summary: dict[str, Any]) -> list[str]:
    listed = summary.get("protocol", {}).get("methods")
    if listed:
        return list(listed)
    runs = summary.get("paired_runs", [])
    return [method for method in LABELS if any(method in run for run in runs)]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _mean_sd(values: list[float], places: int = 3) -> str:
    values = [value for value in values if _finite(value)]
    if not values:
        return "—"
    mean = sum(values) / len(values)
    if len(values) == 1:
        return f"{mean:.{places}f}"
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return f"{mean:.{places}f} ± {sd:.{places}f}"


def _percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "—"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


Table = tuple[list[str], list[list[str]]]

FILTER_TITLE = "Table 2 — Effect of the statistical filter"
QUALITY_TITLE = "Table 3 — Search quality (mean ± sd over seeds)"
FILTER_HEADER = [
    "Method", "Seeds", "Formulas tested", "Passed filter", "Pass rate", "Distinct signals",
    "Checked on test", "Held on test", "Held rate",
]


def _distinct(counts: list[dict[str, Any]]) -> str:
    """Summed distinct-signal count, or a dash for summaries made before it existed."""

    if all("distinct_passed" in item for item in counts):
        return str(sum(item["distinct_passed"] for item in counts))
    return "—"


def filter_rows(summary: dict[str, Any]) -> Table:
    """Table 2 as a header and rows of text; no rows when no run was filtered."""

    rows = []
    for method in methods_of(summary):
        runs = [run[method] for run in summary.get("paired_runs", []) if method in run]
        counts = [run["filter"] for run in runs if run.get("filter")]
        if not counts:
            continue
        tested = sum(item["num_tests"] for item in counts)
        passed = sum(item["passed_filter"] for item in counts)
        checked = sum(item["checked_on_test"] for item in counts)
        held = sum(item["held_on_test"] for item in counts)
        rows.append([
            LABELS.get(method, method), str(len(counts)), str(tested), str(passed),
            _percent(passed, tested), _distinct(counts), str(checked), str(held),
            _percent(held, checked),
        ])
    return FILTER_HEADER, rows


def filter_table(summary: dict[str, Any]) -> str:
    header, rows = filter_rows(summary)
    if not rows:
        return "_No filter results in this summary (run without --no-filter)._"
    return _table(header, rows)


def quality_rows(summary: dict[str, Any]) -> Table:
    """Table 3 as a header and rows of text; ``|`` in headers is escaped for Markdown."""

    columns: list[tuple[str, Callable[[dict[str, Any]], Any], int]] = [
        ("Best train \\|ICIR\\|", lambda m: m.get("best_train_abs_icir"), 3),
        ("Mean top-k train \\|ICIR\\|", lambda m: m.get("mean_train_abs_icir_top_k"), 3),
        ("Mean top-k validation ICIR (directional)",
         lambda m: m.get("mean_validation_directional_icir_top_k"), 3),
        ("Direction agreement", lambda m: m.get("train_validation_direction_agreement_rate"), 2),
        ("Top-k diversity",
         lambda m: m.get("top_k_expression_diversity", {}).get(
             "mean_pairwise_operator_jaccard_distance"), 3),
        ("Passed filter", lambda m: (m.get("filter") or {}).get("passed_filter"), 1),
        ("Distinct signals", lambda m: (m.get("filter") or {}).get("distinct_passed"), 1),
        ("Held on test", lambda m: (m.get("filter") or {}).get("held_on_test"), 1),
        ("Top-k from LLM", lambda m: m.get("top_k_from_llm") if m.get("llm") else None, 1),
        ("LLM acceptance", lambda m: (m.get("llm") or {}).get("acceptance_rate"), 2),
    ]
    rows = []
    for method in methods_of(summary):
        runs = [run[method] for run in summary.get("paired_runs", []) if method in run]
        if not runs:
            continue
        rows.append(
            [LABELS.get(method, method), str(len(runs))]
            + [_mean_sd([get(run) for run in runs], places) for _, get, places in columns]
        )
    return ["Method", "Seeds"] + [name for name, _, _ in columns], rows


def quality_table(summary: dict[str, Any]) -> str:
    return _table(*quality_rows(summary))


def notes(summary: dict[str, Any]) -> dict[str, str]:
    """The explanations printed above Table 2 ("filter") and Table 3 ("quality")."""

    protocol = summary.get("protocol", {})
    top_k = protocol.get("top_k", "k")
    filter_settings = protocol.get("statistical_filter") or {}
    budget = [run.get("evaluation_budget") for run in summary.get("paired_runs", [])]
    note = (
        f"Seeds: {', '.join(str(run['seed']) for run in summary.get('paired_runs', []))}. "
        f"Per-seed evaluation budget (unique formulas, same for every method): "
        f"{', '.join(str(value) for value in budget)}. Top-k = {top_k}."
    )
    coverage = protocol.get("min_train_coverage")
    ic_day_share = protocol.get("min_train_ic_day_share")
    if coverage is not None and ic_day_share is not None:
        note += (
            f" Train |ICIR| is the selection score; a formula scores 0 when it has a value "
            f"on fewer than {coverage:.0%} of train rows or a daily IC on fewer than "
            f"{ic_day_share:.0%} of the train days that can have one."
        )
    filter_note = (
        f"Filter: {filter_settings.get('method', '—')} t-test on the direction-adjusted "
        f"train daily IC, Benjamini–Hochberg at q = {filter_settings.get('q', '—')} over "
        "every unique formula a run evaluated; formulas that pass are checked once on "
        "the test split, and \"held\" means their train direction kept a positive ICIR there."
        if filter_settings else "Filter: not run."
    )
    threshold = filter_settings.get("signal_correlation")
    if threshold is not None:
        filter_note += (
            f" \"Distinct signals\" counts the passed formulas once per signal: in p-value "
            f"order, a formula joins an earlier one when their train daily IC series "
            f"correlate above {threshold:g}, so near-copies of the same formula (which "
            f"loosen the Benjamini–Hochberg threshold for each other) count once."
        )
    quality_note = (
        note + " Validation ICIR is oriented by each formula's train direction, so "
        "positive means the train relationship held. Diversity is the mean pairwise "
        "Jaccard distance between the operator sets of the top-k formulas."
    )
    return {"filter": filter_note, "quality": quality_note}


def build(summary: dict[str, Any]) -> str:
    text = notes(summary)
    return "\n\n".join([
        f"## {FILTER_TITLE}",
        text["filter"],
        filter_table(summary),
        f"## {QUALITY_TITLE}",
        text["quality"],
        quality_table(summary),
    ]) + "\n"


def write_tables(summary_file: Path, output: Path | None = None) -> tuple[Path, str]:
    """Write tables.md (or ``output``) for a comparison summary; return its path and text."""

    summary_file = Path(summary_file)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    output = output or summary_file.with_name("tables.md")
    text = build(summary)
    output.write_text(text, encoding="utf-8")
    return output, text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="comparison_summary.json")
    parser.add_argument("--output", type=Path, help="Markdown file (default: next to the input)")
    args = parser.parse_args()
    output, text = write_tables(args.input, args.output)
    print(text)
    print(f"Tables written to {output}")


if __name__ == "__main__":
    main()
