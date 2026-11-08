"""Run paired random/evolutionary searches with exactly matched evaluation budgets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from python.dsl.parser import parse
from python.llm.cli import add_llm_arguments, build_llm_operator, llm_config_fields
from python.search.direction import directional_fitness, signal_direction
from python.search.evolution import EvolutionConfig, EvolutionSearch
from python.search.random_search import RandomSearch, RandomSearchConfig, operator_set
from python.evaluation.fitness import FitnessEvaluator
from python.stats.filter import METHODS, FilterConfig
from python.stats.ic_store import (
    DailyICTable,
    load_daily_ic,
    save_daily_ic,
    sidecar_path,
    table_from_candidates,
)
from python.stats.run_filter import filter_search, filtered_path, write_filtered
from python.stats.test_split import DEFAULT_LEDGER, OneLookLedger


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _dataset_info(path: Path, data: pd.DataFrame) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "rows": len(data),
        "date_min": str(pd.to_datetime(data["datetime"]).min().date()),
        "date_max": str(pd.to_datetime(data["datetime"]).max().date()),
        "split_rows": {
            str(split): int(count) for split, count in data["split"].value_counts().items()
        },
    }


def _diversity(expressions: list[str]) -> dict[str, Any]:
    sets = [operator_set(parse(expression)) for expression in expressions]
    distances = []
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            union = left | right
            distances.append(1.0 - (len(left & right) / len(union) if union else 1.0))
    return {
        "unique_expression_count": len(set(expressions)),
        "unique_root_operators": sorted(
            {expression.split("(", 1)[0] if "(" in expression else "TERMINAL"
             for expression in expressions}
        ),
        "mean_pairwise_operator_jaccard_distance": (
            float(np.mean(distances)) if distances else 0.0
        ),
    }


def _metrics(payload: dict[str, Any], top_k: int) -> dict[str, Any]:
    candidates = payload["best"][:top_k]
    validation_by_expression = {
        result["expression"]: result["fitness"]
        for result in payload["validation"]
        if "fitness" in result
    }
    train_abs = [abs(item["fitness"]["icir"]) for item in candidates]
    train_signed = [item["fitness"]["icir"] for item in candidates]
    validation = [
        validation_by_expression[item["expression"]]
        for item in candidates
        if item["expression"] in validation_by_expression
    ]
    validation_signed = [item["icir"] for item in validation]
    validation_abs = [abs(value) for value in validation_signed]
    matched_pairs = [
        (candidate, validation_by_expression[candidate["expression"]])
        for candidate in candidates
        if candidate["expression"] in validation_by_expression
    ]
    # Validation ICIR oriented by each formula's train direction, so a
    # formula only scores positively if its train relationship held.
    validation_directional = [
        directional_fitness(
            validated,
            candidate.get("direction", signal_direction(candidate["fitness"]["icir"])),
        )["icir"]
        for candidate, validated in matched_pairs
    ]
    validation_directional = [
        value for value in validation_directional if math.isfinite(value)
    ]
    return {
        "best_train_abs_icir": max(train_abs, default=None),
        "mean_train_abs_icir_top_k": float(np.mean(train_abs)) if train_abs else None,
        "best_validation_directional_icir_top_k": max(validation_directional, default=None),
        "mean_validation_directional_icir_top_k": (
            float(np.mean(validation_directional)) if validation_directional else None
        ),
        # Raw signed values mix formulas of opposite direction; kept for
        # comparison with earlier reports.
        "best_validation_icir_top_k": max(validation_signed, default=None),
        "mean_validation_icir_top_k": (
            float(np.mean(validation_signed)) if validation_signed else None
        ),
        "mean_validation_abs_icir_top_k": (
            float(np.mean(validation_abs)) if validation_abs else None
        ),
        "train_to_validation_abs_icir_degradation": (
            float(np.mean(train_abs) - np.mean(validation_abs))
            if train_abs and validation_abs
            else None
        ),
        "train_validation_direction_agreement_rate": (
            float(np.mean([
                np.sign(candidate["fitness"]["icir"]) == np.sign(validated["icir"])
                for candidate, validated in matched_pairs
            ]))
            if matched_pairs
            else None
        ),
        "top_k_from_llm": sum(bool(item.get("from_llm")) for item in candidates),
        "unique_evaluations": payload["unique_train_evaluations"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "top_k_expression_diversity": _diversity(
            [item["expression"] for item in candidates]
        ),
    }


def _comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    # Reporting cadence does not affect results.
    return {key: value for key, value in config.items() if key != "progress_interval_seconds"}


def _load_saved_run(
    result_file: Path, config: Any
) -> tuple[dict[str, Any], DailyICTable] | None:
    """
    Return a finished run saved by an earlier, interrupted invocation, or None.

    A run counts as finished when both its JSON and its daily-IC sidecar
    exist; the sidecar is written first, so a JSON file implies a sidecar.
    """

    ic_file = sidecar_path(result_file)
    if not (result_file.is_file() and ic_file.is_file()):
        return None
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    expected = _json_safe(asdict(config))
    if _comparable_config(payload.get("config", {})) != _comparable_config(expected):
        raise SystemExit(
            f"{result_file} was produced with different settings; "
            "use a new --output-dir or delete it to re-run."
        )
    return payload, load_daily_ic(ic_file)


def _save_run(result_file: Path, result: Any, dataset: dict[str, Any]) -> tuple[dict[str, Any], DailyICTable]:
    """Write the daily-IC sidecar, then the JSON; return the JSON-safe payload."""

    payload = result.to_dict()
    payload["dataset"] = dataset
    payload = _json_safe(payload)
    table = table_from_candidates(result.evaluated)
    save_daily_ic(sidecar_path(result_file), table)
    result_file.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/processed/csi300_research_dataset.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/search_comparison"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41, 53, 67])
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--mutation-rate", type=float, default=0.7)
    parser.add_argument("--crossover-rate", type=float, default=0.3)
    parser.add_argument("--no-elitism", action="store_true")
    parser.add_argument("--horizon", type=int, choices=(1, 5, 20, 30), default=20)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--min-coverage", type=float, default=0.5,
        help="Formulas defined on fewer train rows than this fraction score 0.",
    )
    parser.add_argument("--progress-interval-seconds", type=float, default=60.0)
    statistics = parser.add_argument_group("statistical filter")
    statistics.add_argument("--no-filter", action="store_true", help="skip the statistical filter")
    statistics.add_argument("--fdr-q", type=float, default=0.05, help="Benjamini-Hochberg q (default 0.05)")
    statistics.add_argument("--ic-test-method", choices=METHODS, default="hansen_hodrick")
    statistics.add_argument(
        "--no-test-split", action="store_true", help="filter only; do not check passing formulas on test"
    )
    statistics.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    add_llm_arguments(parser)
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"Dataset does not exist: {args.dataset}")
    data = pd.read_parquet(args.dataset)
    dataset = _dataset_info(args.dataset, data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    llm_fields = {}
    filter_config = None if args.no_filter else FilterConfig(
        q=args.fdr_q, method=args.ic_test_method, horizon=args.horizon
    )
    ledger = None if args.no_filter or args.no_test_split else OneLookLedger(args.ledger)
    test_evaluator = None

    for seed in args.seeds:
        print(f"\n=== Paired experiment seed={seed} ===", flush=True)
        # A fresh operator per seed keeps its counters per run; the answer
        # cache file is shared, so repeated prompts are never re-sent.
        llm_operator = build_llm_operator(args, args.max_depth, args.horizon)
        llm_fields = llm_config_fields(args, llm_operator)
        evolution_config = EvolutionConfig(
            population_size=args.population_size,
            generations=args.generations,
            elite_count=min(args.elite_count, args.population_size - 1),
            elitism_enabled=not args.no_elitism,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            seed=seed,
            horizon=args.horizon,
            max_depth=args.max_depth,
            seed_pool="expanded",
            validation_top_k=args.top_k,
            progress_interval_seconds=args.progress_interval_seconds,
            min_coverage=args.min_coverage,
            **llm_fields,
        )
        evolution_file = args.output_dir / f"evolution_seed_{seed}.json"
        evolution_saved = _load_saved_run(evolution_file, evolution_config)
        if evolution_saved is not None:
            print(f"Reusing saved evolutionary run: {evolution_file}", flush=True)
            evolution_payload, evolution_table = evolution_saved
        else:
            evolution_result = EvolutionSearch(
                data,
                evolution_config,
                progress_callback=lambda message: print(message, flush=True),
                llm_operator=llm_operator,
            ).run()
            evolution_payload, evolution_table = _save_run(
                evolution_file, evolution_result, dataset
            )

        random_config = RandomSearchConfig(
            evaluation_budget=evolution_payload["unique_train_evaluations"],
            seed=seed,
            horizon=args.horizon,
            max_depth=args.max_depth,
            seed_pool="expanded",
            validation_top_k=args.top_k,
            progress_interval_seconds=args.progress_interval_seconds,
            min_coverage=args.min_coverage,
        )
        random_file = args.output_dir / f"random_seed_{seed}.json"
        random_saved = _load_saved_run(random_file, random_config)
        if random_saved is not None:
            print(f"Reusing saved random run: {random_file}", flush=True)
            random_payload, random_table = random_saved
        else:
            random_result = RandomSearch(
                data, random_config, progress_callback=lambda message: print(message, flush=True)
            ).run()
            random_payload, random_table = _save_run(random_file, random_result, dataset)

        run = {
            "seed": seed,
            "evaluation_budget": evolution_payload["unique_train_evaluations"],
            "evolutionary": _metrics(evolution_payload, args.top_k),
            "random": _metrics(random_payload, args.top_k),
            "result_files": {"evolutionary": str(evolution_file), "random": str(random_file)},
        }
        if filter_config is not None:
            if ledger is not None and test_evaluator is None:
                test_evaluator = FitnessEvaluator(data)
            for method, payload, table, source in (
                ("evolutionary", evolution_payload, evolution_table, evolution_file),
                ("random", random_payload, random_table, random_file),
            ):
                filtered = filter_search(payload, table, filter_config, test_evaluator, ledger)
                filtered["source"] = str(source)
                write_filtered(filtered_path(source), filtered)
                run[method]["filter"] = {
                    **filtered["counts"],
                    "bh_cutoff": filtered["filter"]["bh_cutoff"],
                    "run_id": filtered["run_id"],
                }
                counts = filtered["counts"]
                print(
                    f"Filter ({method}): {counts['passed_filter']}/{counts['num_tests']} passed, "
                    f"{counts['held_on_test']}/{counts['checked_on_test']} held on test.",
                    flush=True,
                )
        comparisons.append(run)
        _write_summary(args, dataset, comparisons, llm_fields, filter_config)

    print(f"Comparison summary: {args.output_dir / 'comparison_summary.json'}")


def _write_summary(args, dataset, comparisons, llm_fields, filter_config) -> None:
    """Rewritten after every seed, so an interrupted run keeps a summary."""

    summary = {
        "dataset": dataset,
        "protocol": {
            "train_split": "train",
            "validation_split": "validation",
            "fitness_objective": "absolute_train_icir",
            "validation_used_for_selection": False,
            "evolution_seed_pool": "expanded",
            "random_seed_pool": "expanded (same formulas; remaining budget filled by random ASTs)",
            "evaluation_budget": "random search matched to each evolutionary run's unique evaluations",
            "top_k": args.top_k,
            "min_train_coverage": args.min_coverage,
            "formulas_simplified_before_deduplication": True,
            "seeds": args.seeds,
            "statistical_filter": (
                {**filter_config.to_dict(), "split": "train",
                 "num_tests": "all unique train evaluations of the run",
                 "test_split_checked": not args.no_test_split}
                if filter_config is not None else None
            ),
            "evolution_config": {
                "population_size": args.population_size,
                "generations": args.generations,
                "elite_count": min(args.elite_count, args.population_size - 1),
                "elitism_enabled": not args.no_elitism,
                "mutation_rate": args.mutation_rate,
                "crossover_rate": args.crossover_rate,
                "max_depth": args.max_depth,
                "horizon": args.horizon,
                **llm_fields,
            },
        },
        "paired_runs": comparisons,
    }
    summary_file = args.output_dir / "comparison_summary.json"
    summary_file.write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
