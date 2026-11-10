"""
Aldien in one command: search -> statistical filter -> report.

For every seed this runs evolution and random search with the same
evaluation budget (plus LLM-guided evolution with --llm-rate > 0, and the
no-retrieval ablation with --ablation-seeds), filters every evaluated formula,
checks the survivors once on the test split, and then writes:

    <output-dir>/comparison_summary.json   all numbers
    <output-dir>/comparison_summary.html   the report (Tables 2 and 3, charts, LLM counts, formulas)
    <output-dir>/tables.md                 Tables 2 and 3 as Markdown

Re-running the same command resumes: finished runs in <output-dir> are reused.

    python -m python.run_pipeline
    python -m python.run_pipeline --llm-rate 0.3 --llm-backend ollama --ablation-seeds 17 29
    python -m python.run_pipeline --report-only     # rebuild the report from an existing summary

Every option of python.search.run_comparison is accepted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from python.search import run_comparison
from python.search.report_comparison import generate_report
from python.search.tables import write_tables

DEFAULT_OUTPUT_DIR = Path("data/processed/experiment")


def build_parser() -> argparse.ArgumentParser:
    parser = run_comparison.build_parser(description=__doc__)
    parser.set_defaults(output_dir=DEFAULT_OUTPUT_DIR)
    report = parser.add_argument_group("report")
    report.add_argument(
        "--report-only", action="store_true",
        help="skip the searches and rebuild the report and tables from <output-dir>/comparison_summary.json",
    )
    report.add_argument("--title", help="title of the HTML report")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.report_only:
        summary = args.output_dir / "comparison_summary.json"
        if not summary.is_file():
            parser.error(f"--report-only needs an existing summary: {summary} not found")
    else:
        summary = run_comparison.run(args, parser)
    report = generate_report(summary, summary.with_suffix(".html"), title=args.title)
    tables, text = write_tables(summary)
    print(text)
    print(f"Summary: {summary}")
    print(f"HTML report: {report}")
    print(f"Tables: {tables}")
    return {"summary": summary, "report": report, "tables": tables}


if __name__ == "__main__":
    main()
