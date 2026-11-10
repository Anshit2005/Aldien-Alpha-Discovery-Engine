"""
Table 1: fitness evaluation speed of the Python reference and the C++ engine.

Times the same 20 formulas on both engines (train split, 20-day horizon by
default): ten spread over the search's starting pool plus ten from the
search's own random generator (fixed seed), so simple and deep formulas are
both represented. For each engine it measures

  - setup: building the evaluator from the loaded panel, once;
  - fitness: one full fitness call per formula, as the search makes it
    (alpha, daily IC, ICIR);
  - alpha only: evaluating the formula without the IC.

Each formula is run once untimed (warm-up) and then --repeats times, the two
engines alternating so that background load affects both alike; the median
of the repeats is reported. Both engines must agree on every result.

With --with-search it also times one full evolutionary search (seed 17 by
default, the experiment's settings) on each engine.

Close other programs first (LLM servers, browsers, Docker): the numbers are
only comparable on a quiet machine.

    python -m python.evaluation.benchmark
    python -m python.evaluation.benchmark --with-search

Writes <output-dir>/table1.md and <output-dir>/benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from python.dsl.ast import ASTNode, BinaryOp, TimeSeriesOp, UnaryOp
from python.dsl.parser import parse
from python.evaluation.engines import cpp_engine_error, make_fitness_evaluator
from python.search.evolution import EvolutionConfig, EvolutionSearch

ENGINES = ("python", "cpp")
LABELS = {"python": "Python (pandas/NumPy)", "cpp": "C++ engine"}


# ------------------------------------------------------------------ formulas

def benchmark_formulas(count: int = 20, seed: int = 2026) -> list[str]:
    """Half spread evenly over the search's starting pool, half from its random generator."""

    pool = EvolutionSearch.SEEDS
    picks = min(count // 2, len(pool))
    from_pool = [pool[round(i * len(pool) / picks)] for i in range(picks)] if picks else []
    generator = EvolutionSearch.__new__(EvolutionSearch)
    generator.rng = random.Random(seed)
    generator.config = EvolutionConfig()
    formulas = list(from_pool)
    while len(formulas) < count:
        expression = str(generator._random_tree(generator.config.max_depth))
        if expression not in formulas:
            formulas.append(expression)
    return formulas


def _nodes(node: ASTNode) -> int:
    if isinstance(node, UnaryOp):
        return 1 + _nodes(node.operand)
    if isinstance(node, BinaryOp):
        return 1 + _nodes(node.left) + _nodes(node.right)
    if isinstance(node, TimeSeriesOp):
        return 1 + _nodes(node.operand)
    return 1


def _windows(node: ASTNode) -> int:
    """Sum of the time-series windows (a rough measure of how much history a formula uses)."""

    if isinstance(node, UnaryOp):
        return _windows(node.operand)
    if isinstance(node, BinaryOp):
        return _windows(node.left) + _windows(node.right)
    if isinstance(node, TimeSeriesOp):
        return int(node.window) + _windows(node.operand)
    return 0


# ------------------------------------------------------------------- machine

def machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "logical_cpus": os.cpu_count(),
        "cpu": platform.processor() or None,
        "memory_gb": None,
    }
    try:
        if sys.platform == "win32":
            import ctypes
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            info["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong)
                    for name in ("total", "available", "total_page", "available_page",
                                 "total_virtual", "available_virtual", "available_extended")
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            info["memory_gb"] = round(status.total / 2**30, 1)
        elif Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    info["memory_gb"] = round(int(line.split()[1]) / 2**20, 1)
                    break
    except Exception:  # machine details are informative only
        pass
    return info


# ------------------------------------------------------------------- timing

def _timed(function: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    value = function()
    return time.perf_counter() - start, value


def _close(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= 1e-6 * max(1.0, abs(a))


def run_benchmark(
    data: pd.DataFrame,
    formulas: list[str],
    repeats: int = 5,
    horizon: int = 20,
    split: str = "train",
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    evaluators, setup = {}, {}
    for engine in ENGINES:
        setup[engine], evaluators[engine] = _timed(lambda e=engine: make_fitness_evaluator(data, e))
    # The Python evaluator builds its alpha evaluator on first use; count that
    # as setup too, so that both engines' setup covers all one-off work.
    first_use, _ = _timed(lambda: evaluators["python"]._evaluate_alpha(parse("CLOSE")))
    setup["python"] += first_use
    progress(f"Setup: Python {setup['python']:.2f} s, C++ {setup['cpp']:.2f} s")

    alpha_only = {
        "python": lambda ast: evaluators["python"]._evaluate_alpha(ast),
        "cpp": lambda ast: evaluators["cpp"].evaluate_alpha(ast),
    }
    rows = []
    for index, expression in enumerate(formulas, start=1):
        ast = parse(expression)
        fitness_times = {engine: [] for engine in ENGINES}
        alpha_times = {engine: [] for engine in ENGINES}
        results = {}
        for engine in ENGINES:  # warm-up, untimed
            results[engine] = evaluators[engine].evaluate(ast, horizon, split)
            alpha_only[engine](ast)
        for _ in range(repeats):
            for engine in ENGINES:
                seconds, _ = _timed(lambda e=engine: evaluators[e].evaluate(ast, horizon, split))
                fitness_times[engine].append(seconds)
                seconds, _ = _timed(lambda e=engine: alpha_only[e](ast))
                alpha_times[engine].append(seconds)
        python_result, cpp_result = results["python"], results["cpp"]
        agree = (
            python_result.num_days == cpp_result.num_days
            and python_result.num_observations == cpp_result.num_observations
            and _close(python_result.icir, cpp_result.icir)
        )
        row = {
            "formula": expression,
            "depth": EvolutionSearch._depth(ast),
            "nodes": _nodes(ast),
            "window_sum": _windows(ast),
            "icir_python": python_result.icir,
            "icir_cpp": cpp_result.icir,
            "num_days": python_result.num_days,
            "engines_agree": agree,
            "fitness_seconds": fitness_times,
            "alpha_seconds": alpha_times,
            "fitness_median": {e: statistics.median(fitness_times[e]) for e in ENGINES},
            "alpha_median": {e: statistics.median(alpha_times[e]) for e in ENGINES},
        }
        rows.append(row)
        median = row["fitness_median"]
        progress(
            f"{index:2d}/{len(formulas)} Python {1000 * median['python']:8.1f} ms  "
            f"C++ {1000 * median['cpp']:7.1f} ms  x{median['python'] / median['cpp']:5.1f}  {expression}"
        )
    return {
        "setup_seconds": setup,
        "formulas": rows,
        "repeats": repeats,
        "horizon": horizon,
        "split": split,
        "rows": len(data),
        "split_rows": int((data["split"] == split).sum()) if "split" in data.columns else len(data),
    }


def run_search_benchmark(
    data: pd.DataFrame, seed: int = 17, progress: Callable[[str], None] = print, **settings: Any
) -> dict[str, Any]:
    """One full evolutionary search per engine, with the experiment's settings."""

    out = {}
    for engine in ENGINES:
        config = EvolutionConfig(seed=seed, engine=engine, progress_interval_seconds=3600, **settings)
        seconds, result = _timed(lambda c=config: EvolutionSearch(data, c).run())
        out[engine] = {
            "seconds": seconds,
            "unique_evaluations": result.unique_evaluations,
            "top": [candidate.expression for candidate in result.best[:10]],
        }
        progress(f"Search, seed {seed}, {engine}: {seconds:.1f} s, {result.unique_evaluations} formulas")
    out["seed"] = seed
    out["same_top_10"] = out["python"]["top"] == out["cpp"]["top"]
    return out


# ------------------------------------------------------------------ report

def _ms(seconds: float) -> str:
    return f"{1000 * seconds:,.1f} ms"


def _s(seconds: float) -> str:
    return f"{seconds:,.2f} s"


def _x(python: float, cpp: float) -> str:
    return f"{python / cpp:,.1f}×" if cpp > 0 else "—"


def summarize(benchmark: dict[str, Any]) -> dict[str, Any]:
    rows = benchmark["formulas"]
    summary: dict[str, Any] = {}
    for kind in ("fitness", "alpha"):
        medians = {e: [row[f"{kind}_median"][e] for row in rows] for e in ENGINES}
        summary[kind] = {
            e: {
                "median": statistics.median(medians[e]),
                "mean": statistics.fmean(medians[e]),
                "total": sum(medians[e]),
                "min": min(medians[e]),
                "max": max(medians[e]),
            }
            for e in ENGINES
        }
        speedups = [p / c for p, c in zip(medians["python"], medians["cpp"]) if c > 0]
        summary[kind]["speedup_range"] = (min(speedups), max(speedups)) if speedups else None
    differences = [
        abs(row["icir_python"] - row["icir_cpp"])
        for row in rows
        if not (math.isnan(row["icir_python"]) or math.isnan(row["icir_cpp"]))
    ]
    summary["max_icir_difference"] = max(differences) if differences else None
    summary["all_agree"] = all(row["engines_agree"] for row in rows)
    return summary


def table_markdown(benchmark: dict[str, Any], machine: dict[str, Any], dataset: str) -> str:
    summary = summarize(benchmark)
    setup = benchmark["setup_seconds"]
    fitness, alpha = summary["fitness"], summary["alpha"]
    n = len(benchmark["formulas"])
    from_pool = sum(1 for row in benchmark["formulas"] if row["formula"] in EvolutionSearch.SEEDS)
    low, high = fitness["speedup_range"]
    lines = [
        "## Table 1 — Fitness evaluation speed, Python vs C++",
        "",
        f"{n} formulas ({from_pool} from the search's starting pool, {n - from_pool} from its "
        "random generator), "
        f"{benchmark['split']} split ({benchmark['split_rows']:,} of {benchmark['rows']:,} rows), "
        f"{benchmark['horizon']}-day horizon. Each time is the median of {benchmark['repeats']} runs "
        "after one warm-up; per-formula medians are then summarised across formulas. "
        "\"Fitness\" is the full call the search makes (alpha, daily IC, ICIR); \"alpha only\" "
        f"leaves out the IC. Machine: {machine.get('cpu') or 'unknown CPU'}, "
        f"{machine.get('logical_cpus')} logical CPUs"
        + (f", {machine['memory_gb']} GB RAM" if machine.get("memory_gb") else "")
        + f", {machine.get('platform')}, Python {machine.get('python')}. Both engines run on one core.",
        "",
        f"| Measure | {LABELS['python']} | {LABELS['cpp']} | Speed-up |",
        "|---|---|---|---|",
        f"| Setup: load the panel into the evaluator (once) | {_s(setup['python'])} | {_s(setup['cpp'])} | — |",
        f"| Fitness of one formula, median | {_ms(fitness['python']['median'])} | {_ms(fitness['cpp']['median'])} | {_x(fitness['python']['median'], fitness['cpp']['median'])} |",
        f"| Fitness of one formula, mean | {_ms(fitness['python']['mean'])} | {_ms(fitness['cpp']['mean'])} | {_x(fitness['python']['mean'], fitness['cpp']['mean'])} |",
        f"| Fitness of one formula, fastest – slowest | {_ms(fitness['python']['min'])} – {_ms(fitness['python']['max'])} | {_ms(fitness['cpp']['min'])} – {_ms(fitness['cpp']['max'])} | {low:,.1f}× – {high:,.1f}× |",
        f"| All {n} formulas | {_s(fitness['python']['total'])} | {_s(fitness['cpp']['total'])} | {_x(fitness['python']['total'], fitness['cpp']['total'])} |",
        f"| Formulas per second | {1 / fitness['python']['mean']:,.1f} | {1 / fitness['cpp']['mean']:,.1f} | — |",
        f"| Alpha only, median | {_ms(alpha['python']['median'])} | {_ms(alpha['cpp']['median'])} | {_x(alpha['python']['median'], alpha['cpp']['median'])} |",
    ]
    search = benchmark.get("search")
    if search:
        lines.append(
            f"| Full evolutionary search, seed {search['seed']} ({search['python']['unique_evaluations']} formulas) "
            f"| {_s(search['python']['seconds'])} | {_s(search['cpp']['seconds'])} "
            f"| {_x(search['python']['seconds'], search['cpp']['seconds'])} |"
        )
    agreement = (
        f"Both engines gave the same IC days and observation counts for every formula; the largest "
        f"ICIR difference was {summary['max_icir_difference']:.1e}."
        if summary["all_agree"] else
        "**The engines disagreed on at least one formula; see benchmark.json.**"
    )
    if search:
        agreement += (
            " The two searches found the same top 10 formulas."
            if search["same_top_10"] else
            " The two searches' top 10 formulas differ: formulas with equal scores (such as X and "
            "DIV(X,2)) are ordered by rounding-level differences, which can send the search down a "
            "different path; the timing comparison is unaffected."
        )
    lines += ["", agreement, "", f"Dataset: `{dataset}`.", "", "### Per formula", "",
              "| # | Formula | Depth | Nodes | Window sum | Python | C++ | Speed-up |",
              "|---|---|---|---|---|---|---|---|"]
    for index, row in enumerate(benchmark["formulas"], start=1):
        median = row["fitness_median"]
        lines.append(
            f"| {index} | `{row['formula']}` | {row['depth']} | {row['nodes']} | {row['window_sum']} "
            f"| {_ms(median['python'])} | {_ms(median['cpp'])} | {_x(median['python'], median['cpp'])} |"
        )
    return "\n".join(lines) + "\n"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/csi300_research_dataset.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/benchmark"))
    parser.add_argument("--formulas", type=int, default=20, help="number of formulas (default 20)")
    parser.add_argument("--repeats", type=int, default=5, help="timed runs per formula and engine (default 5)")
    parser.add_argument("--formula-seed", type=int, default=2026, help="seed of the generated formulas")
    parser.add_argument("--split", choices=("train", "validation", "test", "holdout"), default="train")
    parser.add_argument("--horizon", type=int, choices=(1, 5, 20, 30), default=20)
    parser.add_argument("--with-search", action="store_true", help="also time one full evolutionary search per engine")
    parser.add_argument("--search-seed", type=int, default=17)
    args = parser.parse_args(argv)
    problem = cpp_engine_error()
    if problem is not None:
        parser.error(problem)
    if not args.dataset.is_file():
        parser.error(f"Dataset does not exist: {args.dataset}")
    if args.formulas < 2 or args.repeats < 1:
        parser.error("--formulas must be at least 2 and --repeats at least 1")

    data = pd.read_parquet(args.dataset)
    formulas = benchmark_formulas(args.formulas, args.formula_seed)
    machine = machine_info()
    print(f"Benchmarking {len(formulas)} formulas x {args.repeats} repeats on {machine.get('cpu')}", flush=True)
    benchmark = run_benchmark(
        data, formulas, args.repeats, args.horizon, args.split,
        progress=lambda message: print(message, flush=True),
    )
    if args.with_search:
        benchmark["search"] = run_search_benchmark(
            data, args.search_seed, progress=lambda message: print(message, flush=True)
        )
    benchmark["summary"] = summarize(benchmark)
    benchmark["machine"] = machine
    benchmark["dataset"] = str(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(_json_safe(benchmark), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    table = table_markdown(benchmark, machine, str(args.dataset))
    table_file = args.output_dir / "table1.md"
    table_file.write_text(table, encoding="utf-8")
    print()
    print(table)
    print(f"Table 1: {table_file}")
    print(f"Raw timings: {args.output_dir / 'benchmark.json'}")
    if not benchmark["summary"]["all_agree"]:
        raise SystemExit("The engines disagreed on at least one formula; see benchmark.json")
    return table_file


if __name__ == "__main__":
    main()
