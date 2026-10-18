"""Command line entry point for the non-LLM evolutionary baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from python.search.evolution import EvolutionConfig, EvolutionSearch


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/csi300_research_dataset.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("evolution_results.json"))
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.7)
    parser.add_argument("--crossover-rate", type=float, default=0.3)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--horizon", type=int, choices=(1, 5, 20, 30), default=20)
    parser.add_argument("--validation-top-k", type=int, default=10)
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=60.0,
        help="Emit a candidate-evaluation heartbeat at this interval (default: 60).",
    )
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"Dataset does not exist: {args.dataset}")
    data = pd.read_parquet(args.dataset)
    config = EvolutionConfig(
        population_size=args.population_size,
        generations=args.generations,
        elite_count=args.elite_count,
        tournament_size=args.tournament_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        max_depth=args.max_depth,
        seed=args.seed,
        horizon=args.horizon,
        validation_top_k=args.validation_top_k,
        progress_interval_seconds=args.progress_interval_seconds,
    )
    result = EvolutionSearch(
        data,
        config,
        progress_callback=lambda message: print(message, flush=True),
    ).run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with args.dataset.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    result_payload = result.to_dict()
    result_payload["dataset"] = {
        "path": str(args.dataset),
        "sha256": digest.hexdigest(),
        "rows": len(data),
        "date_min": str(pd.to_datetime(data["datetime"]).min().date()),
        "date_max": str(pd.to_datetime(data["datetime"]).max().date()),
        "split_rows": {
            str(split): int(count)
            for split, count in data["split"].value_counts().items()
        },
    }
    args.output.write_text(
        json.dumps(_json_safe(result_payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Unique train evaluations: {result.unique_evaluations}")
    print(f"Cache hits: {result.cache_hits}")
    print(f"Validation evaluations: {result.validation_evaluations}")
    print(f"Results: {args.output}")
    for index, candidate in enumerate(result.best[:10], start=1):
        print(
            f"{index:2d}. score={candidate.score:.6g} "
            f"ICIR={candidate.fitness.icir:.6g} {candidate.expression}"
        )


if __name__ == "__main__":
    main()
