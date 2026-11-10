"""
Run paired searches with exactly matched evaluation budgets, per seed:
plain evolution, random search, and (with --llm-rate > 0) LLM-guided evolution.

Plain evolution never uses the LLM, so a run with --llm-rate reuses the
evolution and random results of an earlier run without it.
"""

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
from python.evaluation.engines import (
    add_engine_argument,
    check_engine_argument,
    make_fitness_evaluator,
)
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
    # Reporting cadence does not affect results, and the two fitness engines
    # agree to rounding, so a run made on either engine can be reused.
    ignored = {"progress_interval_seconds", "engine"}
    return {key: value for key, value in config.items() if key not in ignored}


# Settings whose default changed the results when they were introduced: a run
# saved before them is treated as having been made with these values, so it is
# not reused under the new rule.
LEGACY_SETTINGS = {"min_ic_day_share": 0.0}


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
    # A setting added after the run was saved counts as its default value,
    # unless the run predates it and was made under different rules.
    defaults = {**_json_safe(asdict(type(config)())), **LEGACY_SETTINGS}
    saved = {
        **{key: value for key, value in defaults.items() if key not in payload.get("config", {})},
        **payload.get("config", {}),
    }
    if _comparable_config(saved) != _comparable_config(expected):
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


LLM_METHOD = "evolutionary_llm"
ABLATION_METHOD = "evolutionary_llm_no_retrieval"


def _evolution_settings(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    """EvolutionConfig fields shared by plain and LLM-guided evolution."""

    return {
        "population_size": args.population_size,
        "generations": args.generations,
        "elite_count": min(args.elite_count, args.population_size - 1),
        "elitism_enabled": not args.no_elitism,
        "mutation_rate": args.mutation_rate,
        "crossover_rate": args.crossover_rate,
        "seed": seed,
        "horizon": args.horizon,
        "max_depth": args.max_depth,
        "seed_pool": "expanded",
        "validation_top_k": args.top_k,
        "progress_interval_seconds": args.progress_interval_seconds,
        "engine": args.engine,
        "min_coverage": args.min_coverage,
        "min_ic_day_share": args.min_ic_day_share,
    }


def _run_or_reuse(
    method: str, result_file: Path, config: Any, dataset: dict[str, Any], search: Any
) -> tuple[dict[str, Any], DailyICTable, Path]:
    """Reuse a finished run with the same settings, or run ``search()`` and save it."""

    saved = _load_saved_run(result_file, config)
    if saved is not None:
        print(f"Reusing saved {method} run: {result_file}", flush=True)
        payload, table = saved
    else:
        payload, table = _save_run(result_file, search(), dataset)
    return payload, table, result_file


def _llm_counts(llm: dict[str, Any]) -> dict[str, Any]:
    """The LLM numbers worth a column in the summary."""

    operator = llm.get("operator", {})
    keys = (
        "calls", "cache_hits", "accepted", "repaired_node_types", "rejected",
        "acceptance_rate", "rejected_by_reason",
    )
    return {
        "model": llm.get("model"),
        "retrieval": llm.get("retrieval", True),
        **{key: operator[key] for key in keys if key in operator},
        "children_added": llm.get("children_added"),
        "unique_llm_evaluations": llm.get("unique_llm_evaluations"),
    }


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    """All run_comparison options; run_pipeline adds its own on top."""

    parser = argparse.ArgumentParser(
        description=description or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    parser.add_argument(
        "--min-ic-day-share", type=float, default=0.5,
        help="Formulas whose train daily IC exists on fewer than this share of the "
        "possible train days score 0 (default 0.5).",
    )
    parser.add_argument("--progress-interval-seconds", type=float, default=60.0)
    add_engine_argument(parser)
    statistics = parser.add_argument_group("statistical filter")
    statistics.add_argument("--no-filter", action="store_true", help="skip the statistical filter")
    statistics.add_argument("--fdr-q", type=float, default=0.05, help="Benjamini-Hochberg q (default 0.05)")
    statistics.add_argument("--ic-test-method", choices=METHODS, default="hansen_hodrick")
    statistics.add_argument(
        "--no-test-split", action="store_true", help="filter only; do not check passing formulas on test"
    )
    statistics.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    add_llm_arguments(parser)
    parser.add_argument(
        "--ablation-seeds", type=int, nargs="*", default=[],
        help="Seeds on which LLM-guided evolution also runs without retrieval (E3).",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    return run(parser.parse_args(argv), parser)


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    """Run every seed and method, filter them, and return the summary file."""

    if args.ablation_seeds and args.llm_rate <= 0:
        parser.error("--ablation-seeds needs --llm-rate > 0")
    if args.no_llm_retrieval:
        parser.error(
            "use --ablation-seeds to compare without retrieval; "
            "--no-llm-retrieval is for run_evolution"
        )

    check_engine_argument(parser, args)
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

    def progress(message: str) -> None:
        print(message, flush=True)

    for seed in args.seeds:
        print(f"\n=== Paired experiment seed={seed} ===", flush=True)
        # A fresh operator per seed keeps its counters per run; the answer
        # cache file is shared, so repeated prompts are never re-sent.
        llm_operator = build_llm_operator(args, args.max_depth, args.horizon)
        llm_fields = llm_config_fields(args, llm_operator)
        results: dict[str, tuple[dict[str, Any], DailyICTable, Path]] = {}

        # 1. Plain evolution (never uses the LLM). Its unique-evaluation count
        #    is the budget every other method gets for this seed.
        evolution_config = EvolutionConfig(
            **_evolution_settings(args, seed),
        )
        evolution_file = args.output_dir / f"evolution_seed_{seed}.json"
        results["evolutionary"] = _run_or_reuse(
            "evolutionary", evolution_file, evolution_config, dataset,
            lambda: EvolutionSearch(data, evolution_config, progress_callback=progress).run(),
        )
        budget = results["evolutionary"][0]["unique_train_evaluations"]

        # 2. Random search with the same budget.
        random_config = RandomSearchConfig(
            evaluation_budget=budget,
            seed=seed,
            horizon=args.horizon,
            max_depth=args.max_depth,
            seed_pool="expanded",
            validation_top_k=args.top_k,
            progress_interval_seconds=args.progress_interval_seconds,
            engine=args.engine,
            min_coverage=args.min_coverage,
            min_ic_day_share=args.min_ic_day_share,
        )
        random_file = args.output_dir / f"random_seed_{seed}.json"
        results["random"] = _run_or_reuse(
            "random", random_file, random_config, dataset,
            lambda: RandomSearch(data, random_config, progress_callback=progress).run(),
        )

        # 3. LLM-guided evolution, only when --llm-rate > 0: same settings and
        #    seed as plain evolution, stopped at exactly the same budget.
        if llm_operator is not None:
            llm_config = EvolutionConfig(
                **_evolution_settings(args, seed), **llm_fields, evaluation_budget=budget,
            )
            llm_file = args.output_dir / f"evolution_llm_seed_{seed}.json"
            results[LLM_METHOD] = _run_or_reuse(
                LLM_METHOD, llm_file, llm_config, dataset,
                lambda: EvolutionSearch(
                    data, llm_config, progress_callback=progress, llm_operator=llm_operator
                ).run(),
            )

            # 4. Ablation (E3) on the chosen seeds: the same LLM run with no
            #    related library factors in the prompt.
            if seed in args.ablation_seeds:
                ablation_operator = build_llm_operator(
                    args, args.max_depth, args.horizon, retrieval=False
                )
                ablation_config = EvolutionConfig(
                    **_evolution_settings(args, seed),
                    **llm_config_fields(args, ablation_operator),
                    evaluation_budget=budget,
                )
                ablation_file = args.output_dir / f"evolution_llm_no_retrieval_seed_{seed}.json"
                results[ABLATION_METHOD] = _run_or_reuse(
                    ABLATION_METHOD, ablation_file, ablation_config, dataset,
                    lambda: EvolutionSearch(
                        data, ablation_config, progress_callback=progress,
                        llm_operator=ablation_operator,
                    ).run(),
                )

        run = {"seed": seed, "evaluation_budget": budget}
        for method, (payload, _, _) in results.items():
            run[method] = _metrics(payload, args.top_k)
            if payload.get("llm") is not None:
                run[method]["llm"] = _llm_counts(payload["llm"])
        run["result_files"] = {method: str(source) for method, (_, _, source) in results.items()}
        if filter_config is not None:
            if ledger is not None and test_evaluator is None:
                test_evaluator = make_fitness_evaluator(data, args.engine)
            for method, (payload, table, source) in results.items():
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

    summary_file = args.output_dir / "comparison_summary.json"
    print(f"Comparison summary: {summary_file}")
    return summary_file


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
            "evaluation_budget": (
                "random search (and LLM-guided evolution, when run) matched to each "
                "evolutionary run's unique evaluations"
            ),
            "top_k": args.top_k,
            "engine": args.engine,
            "engine_note": (
                "fitness engine of this invocation; runs reused from an earlier one "
                "record theirs in their result file's config"
            ),
            "min_train_coverage": args.min_coverage,
            "min_train_ic_day_share": args.min_ic_day_share,
            "formulas_simplified_before_deduplication": True,
            "seeds": args.seeds,
            "methods": (
                ["evolutionary", "random"]
                + ([LLM_METHOD] if llm_fields else [])
                + ([ABLATION_METHOD] if llm_fields and args.ablation_seeds else [])
            ),
            "ablation_seeds": args.ablation_seeds if llm_fields else [],
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
            },
            # Same evolution settings, plus the LLM operator, stopped at the
            # plain evolutionary run's unique-evaluation count.
            "evolution_llm_config": (
                {**llm_fields, "evaluation_budget": "matched to the evolutionary run of the same seed"}
                if llm_fields else None
            ),
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
