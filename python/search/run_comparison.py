"""Run paired random/evolutionary searches with exactly matched evaluation budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from python.dsl.parser import parse
from python.search.evolution import EvolutionConfig, EvolutionSearch
from python.search.random_search import RandomSearch, RandomSearchConfig, operator_set


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
    return {
        "best_train_abs_icir": max(train_abs, default=None),
        "mean_train_abs_icir_top_k": float(np.mean(train_abs)) if train_abs else None,
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
        "unique_evaluations": payload["unique_train_evaluations"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "top_k_expression_diversity": _diversity(
            [item["expression"] for item in candidates]
        ),
    }


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
    parser.add_argument("--progress-interval-seconds", type=float, default=60.0)
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"Dataset does not exist: {args.dataset}")
    data = pd.read_parquet(args.dataset)
    dataset = _dataset_info(args.dataset, data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []

    for seed in args.seeds:
        print(f"\n=== Paired experiment seed={seed} ===", flush=True)
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
        )
        evolution_result = EvolutionSearch(
            data, evolution_config, progress_callback=lambda message: print(message, flush=True)
        ).run()
        evolution_payload = evolution_result.to_dict()
        evolution_payload["dataset"] = dataset

        random_config = RandomSearchConfig(
            evaluation_budget=evolution_result.unique_evaluations,
            seed=seed,
            horizon=args.horizon,
            max_depth=args.max_depth,
            seed_pool="expanded",
            validation_top_k=args.top_k,
            progress_interval_seconds=args.progress_interval_seconds,
        )
        random_result = RandomSearch(
            data, random_config, progress_callback=lambda message: print(message, flush=True)
        ).run()
        random_payload = random_result.to_dict()
        random_payload["dataset"] = dataset

        evolution_file = args.output_dir / f"evolution_seed_{seed}.json"
        random_file = args.output_dir / f"random_seed_{seed}.json"
        evolution_file.write_text(
            json.dumps(_json_safe(evolution_payload), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        random_file.write_text(
            json.dumps(_json_safe(random_payload), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        comparisons.append({
            "seed": seed,
            "evaluation_budget": evolution_result.unique_evaluations,
            "evolutionary": _metrics(evolution_payload, args.top_k),
            "random": _metrics(random_payload, args.top_k),
            "result_files": {"evolutionary": str(evolution_file), "random": str(random_file)},
        })

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
            "seeds": args.seeds,
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
        },
        "paired_runs": comparisons,
    }
    summary_file = args.output_dir / "comparison_summary.json"
    summary_file.write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Comparison summary: {summary_file}")


if __name__ == "__main__":
    main()
