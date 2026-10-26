"""Signal direction for candidates ranked by absolute train ICIR.

Search ranks formulas by |train ICIR|, so a formula whose values move
opposite to future returns scores as well as one that moves with them.
Its *direction* is the sign of its train ICIR: +1 means "high value
predicts high return", -1 means "high value predicts low return", and the
formula is used as ``direction * formula``.

Every evaluation after train (validation, the statistical filter, test)
must multiply by the direction fixed on train. Re-reading the sign from
validation or test would let those splits choose the answer.
"""

from __future__ import annotations

import math
from typing import Any


def signal_direction(train_icir: Any) -> int:
    """Return +1, -1, or 0 (undefined) from a train ICIR."""

    try:
        value = float(train_icir)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(value) or value == 0.0:
        return 0
    return 1 if value > 0 else -1


def directional_fitness(fitness: dict[str, Any], direction: int) -> dict[str, Any]:
    """
    Apply a train direction to a fitness dict's signed statistics.

    Returns ``mean_ic`` and ``icir`` multiplied by ``direction``. Values
    are NaN when the direction is undefined or the statistic is missing.
    """

    def oriented(name: str) -> float:
        value = fitness.get(name)
        if direction == 0 or value is None:
            return float("nan")
        value = float(value)
        return direction * value if math.isfinite(value) else float("nan")

    return {
        "mean_ic": oriented("mean_ic"),
        "icir": oriented("icir"),
    }
