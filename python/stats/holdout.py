"""One final look at the holdout split (2025 onwards) for the chosen formulas.

    python -m python.stats.holdout --summary data/processed/experiment/comparison_summary.json

Run this once, after every search, filter and test-split check is final. It

1. picks the formulas to check from the saved results, using only train and
   test information: from every search run (method and seed), the formula
   that passed the statistical filter, held its train direction on the test
   split, and has the highest test ICIR in that direction (``--per-seed``
   picks more than one); formulas whose test IC is identical (such as ``X``
   and ``DIV(X,2)``) count once, shown by the shortest expression;
2. writes that selection to ``holdout_selection.json`` before anything is
   evaluated (``--dry-run`` stops here, so the selection can be reviewed);
3. evaluates each selected formula on the holdout split once, with the same
   Hansen-Hodrick t-test as the filter, and records every result in a ledger
   (``data/processed/holdout_ledger.jsonl``);
4. writes ``holdout_results.json`` and ``holdout.md`` next to the summary.

Running it again with the same selection reuses the recorded results instead
of evaluating again. A different selection (other settings, or results that
changed) is refused while the ledger holds an earlier one, because a second
look at the holdout would no longer be a clean test; ``--allow-second-look``
overrides this and marks every result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from python.dsl.parser import parse
from python.evaluation.engines import add_engine_argument, check_engine_argument, make_fitness_evaluator
from python.search.direction import directional_fitness
from python.search.tables import LABELS

from .filter import ic_significance
from .run_filter import filtered_path
from .test_split import OneLookLedger, run_id


DEFAULT_LEDGER = Path("data/processed/holdout_ledger.jsonl")
DEFAULT_DATASET = Path("data/processed/csi300_research_dataset.parquet")
SPLIT = "holdout"
SELECTION_RULE = (
    "From each search run (method and seed): of the formulas that passed the statistical filter "
    "(train split, Hansen-Hodrick, Benjamini-Hochberg) and held their train direction on the "
    "test split, the {per_seed} with the highest test ICIR in the train direction; formulas "
    "with identical test IC count once (shortest expression shown). The holdout split is not "
    "used to choose them."
)


class HoldoutError(RuntimeError):
    """The saved results cannot be used for the holdout check."""


# ---------------------------------------------------------------------------
# Selection (train and test information only)
# ---------------------------------------------------------------------------

def _resolve(path_text: str, summary_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_file():
        return path
    beside = summary_dir / path.name
    if beside.is_file():
        return beside
    raise HoldoutError(f"Result file not found: {path_text}")


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def run_candidates(seed: int, path: Path) -> list[dict[str, Any]]:
    """Formulas of one search run that passed the filter and held on test."""

    filtered = json.loads(path.read_text(encoding="utf-8"))
    if filtered["counts"]["passed_filter"] and not filtered.get("test"):
        raise HoldoutError(
            f"{path.name}: formulas passed the filter but were never checked on the test "
            "split. Re-run the comparison without --no-test-split first."
        )
    train = {item["expression"]: item for item in filtered["formulas"]}
    entries = []
    for result in (filtered.get("test") or {}).get("results", []):
        test_icir = _finite(result["directional"]["icir"])
        if not result["held"] or result.get("second_look") or test_icir is None:
            continue
        expression = result["expression"]
        entries.append({
            "expression": expression,
            "direction": int(result["direction"]),
            "seeds": [seed],
            "train_t_stat": _finite(train.get(expression, {}).get("t_stat")),
            "train_p_value": _finite(train.get(expression, {}).get("p_value")),
            "test_icir": test_icir,
            "test_mean_ic": _finite(result["directional"]["mean_ic"]),
            "test_days": int(result["fitness"]["num_days"]),
            "equivalents": [],
        })
    return collapse(entries)


def collapse(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge entries with the same expression (joining their seeds), then those
    with identical test IC (the same signal, e.g. ``X`` and ``DIV(X,2)``),
    keeping the shortest expression. Sorted by test ICIR, best first.
    """

    by_expression: dict[str, dict[str, Any]] = {}
    for entry in entries:
        kept = by_expression.setdefault(entry["expression"], {**entry, "seeds": [], "equivalents": []})
        kept["seeds"] = sorted(set(kept["seeds"]) | set(entry["seeds"]))
        kept["equivalents"] = sorted(set(kept["equivalents"]) | set(entry["equivalents"]))
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for entry in by_expression.values():
        key = (
            round(entry["test_icir"], 10),
            None if entry["test_mean_ic"] is None else round(entry["test_mean_ic"], 10),
            entry["test_days"],
        )
        groups.setdefault(key, []).append(entry)
    merged = []
    for group in groups.values():
        group.sort(key=lambda item: (len(item["expression"]), item["expression"]))
        keep = dict(group[0])
        others = {e for item in group for e in [item["expression"], *item["equivalents"]]}
        keep["equivalents"] = sorted(others - {keep["expression"]})
        keep["seeds"] = sorted({seed for item in group for seed in item["seeds"]})
        merged.append(keep)
    merged.sort(key=lambda item: (-item["test_icir"], item["expression"]))
    return merged


def method_selection(
    filtered_files: Sequence[tuple[int, Path]], per_seed: int = 1
) -> dict[str, Any]:
    """
    The best ``per_seed`` formulas of each run of one method, by test ICIR.

    Taking finalists per run, rather than the best of the pooled list, keeps
    one lucky run (whose top formulas are often near-copies of each other)
    from filling the whole selection.
    """

    per_run = [run_candidates(seed, path) for seed, path in filtered_files]
    eligible = collapse([entry for entries in per_run for entry in entries])
    chosen = collapse([entry for entries in per_run for entry in entries[:per_seed]])
    return {"eligible": len(eligible), "formulas": chosen}


def select_formulas(summary_file: Path, per_seed: int = 1) -> dict[str, Any]:
    """The holdout selection for a finished comparison, from train and test results only."""

    if per_seed < 1:
        raise ValueError("per_seed must be at least 1")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    protocol = summary.get("protocol", {})
    if "min_train_ic_day_share" not in protocol:
        raise HoldoutError(
            f"{summary_file} was made before the IC-day scoring rule; re-run the comparison "
            "with this version first, so the holdout is spent on the final results."
        )
    statistical_filter = protocol.get("statistical_filter")
    if not statistical_filter or not statistical_filter.get("test_split_checked"):
        raise HoldoutError(
            "The comparison ran without the statistical filter and test-split check; "
            "re-run it with both before the holdout check."
        )

    files: dict[str, list[tuple[int, Path]]] = {}
    for run in summary["paired_runs"]:
        for method, source in run.get("result_files", {}).items():
            path = filtered_path(_resolve(source, summary_file.parent))
            if not path.is_file():
                raise HoldoutError(f"Filtered results not found: {path}")
            files.setdefault(method, []).append((int(run["seed"]), path))

    methods = {}
    sources = {}
    for method in [m for m in LABELS if m in files] + sorted(set(files) - set(LABELS)):
        methods[method] = method_selection(files[method], per_seed)
        sources[method] = [{"seed": seed, "file": str(path)} for seed, path in files[method]]

    horizon = int(statistical_filter.get("horizon", 20))
    core = {
        "horizon": horizon,
        "per_seed": per_seed,
        "methods": {
            method: [[item["expression"], item["direction"]] for item in chosen["formulas"]]
            for method, chosen in methods.items()
        },
    }
    return {
        "selection_id": run_id(core),
        "summary": str(summary_file),
        "dataset": summary.get("dataset", {}),
        "horizon": horizon,
        "per_seed": per_seed,
        "rule": SELECTION_RULE.format(per_seed=per_seed),
        "methods": methods,
        "sources": sources,
        "selected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def unique_formulas(selection: dict[str, Any]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for chosen in selection["methods"].values():
        for item in chosen["formulas"]:
            seen.setdefault(item["expression"], {"expression": item["expression"], "direction": item["direction"]})
    return list(seen.values())


# ---------------------------------------------------------------------------
# The one look
# ---------------------------------------------------------------------------

def _dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def holdout_period(data: pd.DataFrame, horizon: int) -> dict[str, Any]:
    """Dates whose row and forward-return target both fall in the holdout split."""

    target = f"target_split_{horizon}d"
    rows = data["split"].eq(SPLIT)
    if target in data:
        rows &= data[target].eq(SPLIT)
    dates = pd.to_datetime(data.loc[rows, "datetime"])
    if dates.empty:
        return {"first_date": None, "last_date": None, "days": 0}
    return {"first_date": str(dates.min().date()), "last_date": str(dates.max().date()),
            "days": int(dates.nunique())}


def check_on_holdout(
    evaluator: Any,
    selection: dict[str, Any],
    ledger: OneLookLedger,
    allow_second_look: bool = False,
) -> dict[str, Any]:
    """Evaluate every selected formula on the holdout once; reuse recorded results."""

    selection_id = selection["selection_id"]
    earlier = [run for run in ledger.runs() if run != selection_id]
    if earlier and not allow_second_look:
        raise HoldoutError(
            f"The holdout was already checked for a different selection ({', '.join(earlier)}). "
            "A second look would not be a clean test; pass --allow-second-look to do it anyway "
            "(every result is then marked as a second look)."
        )
    previous = ledger.recorded(selection_id)
    horizon = selection["horizon"]
    results, reused = [], 0
    for item in unique_formulas(selection):
        expression = item["expression"]
        if expression in previous:
            results.append({k: v for k, v in previous[expression].items() if k != "run_id"})
            reused += 1
            continue
        fitness = evaluator.evaluate(parse(expression), horizon=horizon, split=SPLIT)
        significance = ic_significance(fitness.daily_ic.to_numpy(), item["direction"], horizon)
        directional = directional_fitness(fitness.to_dict(), item["direction"])
        oriented_icir = directional["icir"]
        result = {
            "expression": expression,
            "direction": item["direction"],
            "fitness": {key: _finite(value) if isinstance(value, float) else value
                        for key, value in fitness.to_dict().items()},
            "directional": {key: _finite(value) for key, value in directional.items()},
            "t_stat": _finite(significance["t_stat"]),
            "p_value": _finite(significance["p_value"]),
            "held": bool(oriented_icir == oriented_icir and oriented_icir > 0),
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "second_look": bool(earlier),
        }
        ledger.record(selection_id, result)
        results.append(result)
    return {"selection_id": selection_id, "split": SPLIT, "evaluated": len(results) - reused,
            "reused_from_ledger": reused, "results": results}


def method_rows(selection: dict[str, Any], check: dict[str, Any]) -> list[dict[str, Any]]:
    by_expression = {result["expression"]: result for result in check["results"]}
    rows = []
    for method, chosen in selection["methods"].items():
        formulas = []
        for item in chosen["formulas"]:
            result = by_expression[item["expression"]]
            formulas.append({**item, "holdout": result})
        holdout_icirs = [f["holdout"]["directional"]["icir"] for f in formulas
                         if f["holdout"]["directional"]["icir"] is not None]
        test_icirs = [f["test_icir"] for f in formulas]
        rows.append({
            "method": method,
            "eligible": chosen["eligible"],
            "checked": len(formulas),
            "held": sum(f["holdout"]["held"] for f in formulas),
            "significant": sum((f["holdout"]["p_value"] or 1.0) < 0.05 and f["holdout"]["held"]
                               for f in formulas),
            "mean_test_icir": sum(test_icirs) / len(test_icirs) if test_icirs else None,
            "mean_holdout_icir": sum(holdout_icirs) / len(holdout_icirs) if holdout_icirs else None,
            "formulas": formulas,
        })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _p(value: float | None) -> str:
    return "—" if value is None else (f"{value:.3f}" if value >= 0.001 else "<0.001")


def holdout_markdown(output: dict[str, Any]) -> str:
    period = output["period"]
    horizon = output["selection"]["horizon"]
    lines = [
        f"## Holdout check ({period['first_date']} to {period['last_date']})",
        "",
        "| Method | Eligible | Checked | Held direction | Significant (p < 0.05) "
        "| Mean test ICIR | Mean holdout ICIR |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in output["methods"]:
        lines.append(
            f"| {LABELS.get(row['method'], row['method'])} | {row['eligible']} | {row['checked']} "
            f"| {row['held']}/{row['checked']} | {row['significant']}/{row['checked']} "
            f"| {_num(row['mean_test_icir'])} | {_num(row['mean_holdout_icir'])} |"
        )
    second = any(result.get("second_look") for result in output["check"]["results"])
    lines += [
        "",
        f"ICIR and IC are in each formula's train direction, so a positive holdout ICIR means "
        f"the train relationship held. \"Eligible\" counts the formulas that passed the filter "
        f"and held on test (identical signals once); \"Checked\" are "
        + ("each run's best formula" if output["selection"]["per_seed"] == 1
           else f"the best {output['selection']['per_seed']} formulas of each run")
        + f" by test ICIR. The holdout has {period['days']} days with a "
        f"{horizon}-day forward return inside it, about {period['days'] // horizon} "
        f"non-overlapping periods, so single-formula t-tests have little power; the direction "
        f"count and the mean ICIR carry more weight. p-values use the Hansen-Hodrick correction "
        f"and are not adjusted for the number of formulas checked.",
        "",
        f"Selection `{output['selection']['selection_id']}`, evaluated once "
        f"({output['check']['evaluated']} evaluated now, "
        f"{output['check']['reused_from_ledger']} reused from the ledger)."
        + (" **This includes a second look at the holdout.**" if second else ""),
        "",
        "### Per formula",
        "",
        "| Method | Formula | Seeds | Test ICIR | Holdout ICIR | Holdout mean IC | t | p | Held |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in output["methods"]:
        for formula in row["formulas"]:
            result = formula["holdout"]
            name = f"`{formula['expression']}`"
            if formula["equivalents"]:
                name += f" (+{len(formula['equivalents'])} equivalent)"
            lines.append(
                f"| {LABELS.get(row['method'], row['method'])} | {name} "
                f"| {', '.join(str(s) for s in formula['seeds'])} | {_num(formula['test_icir'])} "
                f"| {_num(result['directional']['icir'])} | {_num(result['directional']['mean_ic'], 4)} "
                f"| {_num(result['t_stat'], 2)} | {_p(result['p_value'])} "
                f"| {'yes' if result['held'] else 'no'} |"
            )
    return "\n".join(lines) + "\n"


def selection_text(selection: dict[str, Any]) -> str:
    lines = [f"Holdout selection {selection['selection_id']} ({selection['rule']})"]
    for method, chosen in selection["methods"].items():
        lines.append(f"  {LABELS.get(method, method)}: {len(chosen['formulas'])} of {chosen['eligible']} eligible")
        for item in chosen["formulas"]:
            lines.append(f"    test ICIR {item['test_icir']:.3f}  {item['expression']}")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path,
                        default=Path("data/processed/experiment/comparison_summary.json"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--per-seed", type=int, default=1,
                        help="formulas checked per search run (default 1, the run's best)")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, help="defaults to the summary's folder")
    parser.add_argument("--dry-run", action="store_true",
                        help="write and print the selection only; the holdout is not touched")
    parser.add_argument("--allow-second-look", action="store_true",
                        help="check a different selection although the holdout was already used")
    add_engine_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.summary.is_file():
        parser.error(f"Comparison summary not found: {args.summary}")
    output_dir = args.output_dir or args.summary.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        selection = select_formulas(args.summary, args.per_seed)
    except HoldoutError as error:
        parser.exit(2, f"holdout check: {error}\n")

    selection_file = output_dir / "holdout_selection.json"
    _write_json(selection_file, selection)
    print(selection_text(selection))
    if args.dry_run:
        print(f"Selection written to {selection_file}; the holdout was not touched.")
        return selection_file
    if not unique_formulas(selection):
        parser.exit(2, "holdout check: no formula passed the filter and held on test; nothing to check.\n")

    check_engine_argument(parser, args)
    if not args.dataset.is_file():
        parser.error(f"Dataset does not exist: {args.dataset}")
    expected = selection["dataset"].get("sha256")
    if expected and _dataset_sha256(args.dataset) != expected:
        parser.exit(2, f"holdout check: {args.dataset} is not the dataset the comparison used.\n")
    data = pd.read_parquet(args.dataset)
    period = holdout_period(data, selection["horizon"])
    if not period["days"]:
        parser.exit(2, "holdout check: the dataset has no holdout days for this horizon.\n")

    evaluator = make_fitness_evaluator(data, args.engine)
    try:
        check = check_on_holdout(evaluator, selection, OneLookLedger(args.ledger), args.allow_second_look)
    except HoldoutError as error:
        parser.exit(2, f"holdout check: {error}\n")

    output = {
        "selection": {key: selection[key] for key in ("selection_id", "rule", "per_seed", "horizon", "summary")},
        "period": period,
        "engine": args.engine,
        "methods": method_rows(selection, check),
        "check": check,
    }
    _write_json(output_dir / "holdout_results.json", output)
    table_file = output_dir / "holdout.md"
    table_file.write_text(holdout_markdown(output), encoding="utf-8")
    print()
    print(holdout_markdown(output))
    print(f"Holdout results: {output_dir / 'holdout_results.json'} and {table_file}")
    return table_file


if __name__ == "__main__":
    main()
