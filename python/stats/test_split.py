"""One look at the test split for formulas that passed the filter.

The test split is touched exactly once per search run. Every look is
recorded in a ledger (JSON lines) keyed by the run id, a hash of the search
result. Running the filter again on the same run reuses the recorded test
results instead of evaluating again, so a changed setting can never turn into
a second, quietly better look. Evaluating formulas that were not looked at
before needs ``allow_second_look=True`` and is marked in the output.

Test results are reported only; nothing in the search or filter reads them.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from python.dsl.parser import parse
from python.search.direction import directional_fitness


DEFAULT_LEDGER = Path("data/processed/test_split_ledger.jsonl")


def run_id(payload: dict[str, Any]) -> str:
    """Stable id of a search result: hash of its canonical JSON."""

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class OneLookLedger:
    def __init__(self, path: Path | str | None = DEFAULT_LEDGER):
        self.path = Path(path) if path is not None else None
        self._results: dict[str, dict[str, dict[str, Any]]] = {}
        if self.path is not None and self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self._results.setdefault(record["run_id"], {})[record["expression"]] = record

    def recorded(self, run: str) -> dict[str, dict[str, Any]]:
        return dict(self._results.get(run, {}))

    def record(self, run: str, result: dict[str, Any]) -> None:
        record = {"run_id": run, **result}
        self._results.setdefault(run, {})[result["expression"]] = record
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, allow_nan=False) + "\n")


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and value != value:
        return None
    return value


def check_on_test(
    evaluator: Any,
    formulas: Sequence[dict[str, Any]],
    horizon: int,
    run: str,
    ledger: OneLookLedger,
    allow_second_look: bool = False,
) -> dict[str, Any]:
    """
    Evaluate each ``{"expression", "direction"}`` once on the test split.

    Results already in the ledger for this run are reused. Returns the
    results, how many were reused, and how many held their train direction.
    """

    previous = ledger.recorded(run)
    new = [item for item in formulas if item["expression"] not in previous]
    if new and previous and not allow_second_look:
        raise RuntimeError(
            f"Run {run} was already checked on the test split for other formulas; "
            f"{len(new)} new formula(s) would be a second look. "
            "Re-run with the original settings, or pass allow_second_look."
        )

    results = []
    reused = 0
    for item in formulas:
        expression = item["expression"]
        if expression in previous:
            results.append({k: v for k, v in previous[expression].items() if k != "run_id"})
            reused += 1
            continue
        fitness = evaluator.evaluate(parse(expression), horizon=horizon, split="test")
        raw = {key: _finite_or_none(value) for key, value in fitness.to_dict().items()}
        directional = directional_fitness(fitness.to_dict(), item["direction"])
        oriented_icir = directional["icir"]
        result = {
            "expression": expression,
            "direction": item["direction"],
            "fitness": raw,
            "directional": {key: _finite_or_none(value) for key, value in directional.items()},
            # The train relationship held if the oriented test IC is positive.
            "held": bool(oriented_icir == oriented_icir and oriented_icir > 0),
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "second_look": bool(previous),
        }
        ledger.record(run, result)
        results.append(result)

    return {
        "run_id": run,
        "split": "test",
        "evaluated": len(results) - reused,
        "reused_from_ledger": reused,
        "held_count": sum(result["held"] for result in results),
        "results": results,
    }
