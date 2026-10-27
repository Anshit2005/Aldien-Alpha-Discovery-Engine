"""Prompts for the LLM proposal operator.

Both prompts are built deterministically from their inputs, so the same
parent and context always produce the same text and therefore the same
cache key.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from python.dsl.ast import ASTNode
from python.dsl.json_io import dumps
from python.dsl.validator import (
    ALLOWED_BINARY_OPERATORS,
    ALLOWED_TIME_SERIES_OPERATORS,
    ALLOWED_UNARY_OPERATORS,
    ALLOWED_VARIABLES,
    MAX_DEPTH,
    MAX_WINDOW,
)

from .retrieval import LibraryFactor


PARENT_PREFIX = "Parent formula: "

_EXAMPLE = (
    '{"node_type": "OPERATOR", "value": "DIV", "children": [\n'
    '  {"node_type": "OPERATOR", "value": "SUB", "children": [\n'
    '    {"node_type": "VARIABLE", "value": "CLOSE"},\n'
    '    {"node_type": "OPERATOR", "value": "TS_MEAN", "children": [\n'
    '      {"node_type": "VARIABLE", "value": "CLOSE"},\n'
    '      {"node_type": "CONSTANT", "value": "20"}\n'
    '    ]}\n'
    '  ]},\n'
    '  {"node_type": "OPERATOR", "value": "TS_STD", "children": [\n'
    '    {"node_type": "VARIABLE", "value": "CLOSE"},\n'
    '    {"node_type": "CONSTANT", "value": "20"}\n'
    '  ]}\n'
    ']}'
)


def system_prompt(max_depth: int = MAX_DEPTH) -> str:
    variables = ", ".join(sorted(ALLOWED_VARIABLES))
    unary = ", ".join(sorted(ALLOWED_UNARY_OPERATORS))
    binary = ", ".join(sorted(ALLOWED_BINARY_OPERATORS))
    time_series = ", ".join(sorted(ALLOWED_TIME_SERIES_OPERATORS))
    return f"""You design formulaic alpha factors for daily stock data.
A formula is evaluated for every stock on every day, and stocks are compared with each other on the same day.

Grammar:
- Variables: {variables} (daily prices and traded volume).
- Constants: any finite number.
- Unary operators, 1 input: {unary}. RANK is the cross-sectional percentile rank on that day. LOG is undefined for values <= 0.
- Binary operators, 2 inputs: {binary}. Division by zero gives an undefined value.
- Time-series operators, 2 inputs: {time_series}. The first input is a formula, the second is an integer window from 1 to {MAX_WINDOW} days, written as a CONSTANT. They look only at past and current days of the same stock.
- Maximum depth: {max_depth} levels, counting variables and constants as one level.

Answer format: one JSON tree, where every node has "node_type" (VARIABLE, CONSTANT or OPERATOR) and "value", and OPERATOR nodes have "children" in order. Constants are written as strings.
Example, DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20)):
{_EXAMPLE}

Reply with the JSON tree only: no explanation and no markdown."""


def user_prompt(
    parent: ASTNode,
    parent_fitness: Any | None,
    references: Sequence[LibraryFactor],
    recent_failures: Sequence[dict[str, str]],
    horizon: int = 20,
) -> str:
    lines = [
        f"{PARENT_PREFIX}{parent}",
        f"Parent as JSON: {dumps(parent)}",
        _fitness_line(parent_fitness),
    ]
    if references:
        lines += ["", "Related known factors:"]
        lines += [f"- {factor.expression}: {factor.meaning}" for factor in references]
    if recent_failures:
        lines += ["", "Recently rejected proposals (avoid these mistakes):"]
        for failure in recent_failures:
            response = " ".join(failure["response"].split())
            if len(response) > 160:
                response = response[:157] + "..."
            lines.append(f"- {response} -> {failure['error']}")
    lines += [
        "",
        "Task: propose ONE new formula that is a meaningful variation of the parent, "
        "for example a different window, normalisation, ranking, or an added volume or "
        f"volatility term, that could predict {horizon}-day forward returns better. "
        "It must differ from the parent. Reply with the JSON tree only.",
    ]
    return "\n".join(lines)


def _fitness_line(fitness: Any | None) -> str:
    icir = getattr(fitness, "icir", float("nan")) if fitness is not None else float("nan")
    if not math.isfinite(icir):
        return "Parent train ICIR: undefined"
    direction = (
        "high values predict high returns" if icir > 0 else "high values predict low returns"
    )
    coverage = getattr(fitness, "coverage", float("nan"))
    coverage_text = f", defined on {coverage:.0%} of rows" if math.isfinite(coverage) else ""
    return f"Parent train ICIR: {icir:+.4f} ({direction}{coverage_text})"


def parent_expression_from_prompt(user: str) -> str:
    """Recover the parent formula text from a user prompt (used by the mock)."""

    for line in user.splitlines():
        if line.startswith(PARENT_PREFIX):
            return line[len(PARENT_PREFIX):].strip()
    raise ValueError("Prompt has no parent formula line")
