"""Apply the statistical filter to a saved search result.

    python -m python.stats.run_filter --input data/processed/search_comparison/evolution_seed_17.json

Reads the search JSON and its ``_daily_ic.npz`` sidecar, tests every unique
formula on its train daily IC, applies Benjamini-Hochberg over all of them,
checks the passing formulas once on the test split, and writes
``<name>_filtered.json`` next to the input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from python.evaluation.fitness import FitnessEvaluator

from .filter import METHODS, FilterConfig, filter_table
from .ic_store import DailyICTable, load_daily_ic, sidecar_path
from .test_split import DEFAULT_LEDGER, OneLookLedger, check_on_test, run_id


def search_method(payload: dict[str, Any]) -> str:
    if payload.get("method") == "random_search":
        return "random"
    return "evolutionary" if "generations" in payload else str(payload.get("method", "unknown"))


def filter_search(
    payload: dict[str, Any],
    table: DailyICTable,
    config: FilterConfig,
    evaluator: FitnessEvaluator | None = None,
    ledger: OneLookLedger | None = None,
    allow_second_look: bool = False,
) -> dict[str, Any]:
    """
    Filter one search result. With ``evaluator`` and ``ledger``, formulas that
    pass are also checked once on the test split.

    ``payload`` must be the JSON-safe search result (as saved to disk), so its
    run id matches the one computed from the file.
    """

    num_tests = int(payload.get("unique_train_evaluations", len(table)))
    if len(table) != num_tests:
        raise ValueError(
            f"Daily IC table has {len(table)} formulas but the search reports "
            f"{num_tests} unique train evaluations; the sidecar does not match this result."
        )
    run = run_id(payload)
    filtered = filter_table(table, config, num_tests)

    test = None
    if evaluator is not None and ledger is not None and filtered["passed_count"]:
        passing = [item for item in filtered["formulas"] if item["passed"]]
        test = check_on_test(
            evaluator, passing, config.horizon, run, ledger, allow_second_look
        )

    return {
        "run_id": run,
        "method": search_method(payload),
        "seed": payload.get("config", {}).get("seed"),
        "filter": {key: value for key, value in filtered.items() if key != "formulas"},
        "counts": filter_counts(filtered, test),
        "test": test,
        "formulas": filtered["formulas"],
    }


def filter_counts(filtered: dict[str, Any], test: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "num_tests": filtered["num_tests"],
        "passed_filter": filtered["passed_count"],
        "distinct_passed": filtered["distinct_passed"],
        "checked_on_test": len(test["results"]) if test else 0,
        "held_on_test": test["held_count"] if test else 0,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def filtered_path(result_json: Path) -> Path:
    return result_json.with_name(f"{result_json.stem}_filtered.json")


def write_filtered(path: Path, output: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="evolution or random search JSON")
    parser.add_argument("--daily-ic", type=Path, help="defaults to <input>_daily_ic.npz")
    parser.add_argument("--output", type=Path, help="defaults to <input>_filtered.json")
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/processed/csi300_research_dataset.parquet")
    )
    parser.add_argument("--q", type=float, default=0.05, help="false discovery rate (default 0.05)")
    parser.add_argument("--method", choices=METHODS, default="hansen_hodrick")
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument("--no-test", action="store_true", help="skip the test-split check")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--allow-second-look", action="store_true",
        help="evaluate formulas on test even if this run was already checked (marked in output)",
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    ic_path = args.daily_ic or sidecar_path(args.input)
    if not ic_path.is_file():
        parser.error(f"Daily IC sidecar not found: {ic_path} (re-run the search with this version)")
    config = FilterConfig(
        q=args.q,
        method=args.method,
        horizon=int(payload.get("config", {}).get("horizon", 20)),
        min_days=args.min_days,
    )

    evaluator = ledger = None
    if not args.no_test:
        if not args.dataset.is_file():
            parser.error(f"Dataset does not exist: {args.dataset}")
        evaluator = FitnessEvaluator(pd.read_parquet(args.dataset))
        ledger = OneLookLedger(args.ledger)

    output = filter_search(
        payload, load_daily_ic(ic_path), config, evaluator, ledger, args.allow_second_look
    )
    output["source"] = str(args.input)
    destination = args.output or filtered_path(args.input)
    write_filtered(destination, output)

    counts = output["counts"]
    cutoff = output["filter"]["bh_cutoff"]
    print(
        f"{output['method']} seed={output['seed']} run={output['run_id']}: "
        f"{counts['passed_filter']}/{counts['num_tests']} passed, "
        f"{counts['distinct_passed']} distinct signals "
        f"(BH q={config.q}, cutoff p={cutoff if cutoff is not None else 'none'}, {config.method}); "
        f"held on test: {counts['held_on_test']}/{counts['checked_on_test']}"
    )
    for item in output["formulas"][:10]:
        mark = "PASS" if item["passed"] else "    "
        print(f"  {mark} p={item['p_value']:.3g} t={item['t_stat']:.2f} {item['expression']}")
    print(f"Filtered results: {destination}")


if __name__ == "__main__":
    main()
