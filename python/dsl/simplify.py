"""Rule-based simplification of alpha DSL expressions.

Search generates many formulas that compute the same values under different
spellings, for example ``ABS(HIGH)`` and ``HIGH``, or ``ADD(CLOSE,OPEN)`` and
``ADD(OPEN,CLOSE)``. Each spelling would otherwise be evaluated and ranked
separately. ``simplify`` rewrites a tree bottom-up into one canonical form so
that duplicates share a cache entry and a top-k slot.

Every rule keeps the evaluated values identical, including NaN positions,
so coverage and IC are unchanged:

* ``ABS(x)`` -> ``x`` when ``x`` can never be negative (prices, volume,
  ranks, rolling statistics of such values, another ``ABS``).
* ``SIGN(SIGN(x))`` -> ``SIGN(x)``, ``RANK(RANK(x))`` -> ``RANK(x)``.
* ``ADD(x,0)``, ``SUB(x,0)``, ``MUL(x,1)``, ``DIV(x,1)`` -> ``x``.
* Operators applied only to constants are folded into one constant.
* ``ADD`` and ``MUL`` order their operands by text, so ``ADD(a,b)`` and
  ``ADD(b,a)`` become the same expression.

Time-series windows are never changed.
"""

from __future__ import annotations

import math

from .ast import ASTNode, BinaryOp, Number, TimeSeriesOp, UnaryOp, Variable


# Every market variable in the DSL is a price or a traded volume.
NON_NEGATIVE_VARIABLES = frozenset({"OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"})

COMMUTATIVE_OPERATORS = frozenset({"ADD", "MUL"})

IDEMPOTENT_UNARY_OPERATORS = frozenset({"SIGN", "RANK"})

# Rolling statistics that cannot be negative when their input is not.
_SIGN_PRESERVING_TIME_SERIES = frozenset({"TS_MEAN", "TS_MIN", "TS_MAX"})


def simplify(node: ASTNode) -> ASTNode:
    """Return a canonical tree that evaluates to the same values as ``node``."""

    if isinstance(node, UnaryOp):
        return _simplify_unary(node.operator, simplify(node.operand))
    if isinstance(node, BinaryOp):
        return _simplify_binary(node.operator, simplify(node.left), simplify(node.right))
    if isinstance(node, TimeSeriesOp):
        return TimeSeriesOp(node.operator, simplify(node.operand), node.window)
    return node


def is_non_negative(node: ASTNode) -> bool:
    """True when ``node`` evaluates to values >= 0 (or NaN) on any data."""

    if isinstance(node, Variable):
        return node.name in NON_NEGATIVE_VARIABLES
    if isinstance(node, Number):
        return node.value >= 0
    if isinstance(node, UnaryOp):
        # Percentile ranks lie in (0, 1].
        return node.operator in {"ABS", "RANK"}
    if isinstance(node, TimeSeriesOp):
        if node.operator == "TS_STD":
            return True
        return node.operator in _SIGN_PRESERVING_TIME_SERIES and is_non_negative(node.operand)
    if isinstance(node, BinaryOp):
        if node.operator in {"ADD", "MUL", "DIV"}:
            return is_non_negative(node.left) and is_non_negative(node.right)
    return False


def _simplify_unary(operator: str, operand: ASTNode) -> ASTNode:
    if isinstance(operand, Number):
        folded = _fold_unary(operator, operand.value)
        if folded is not None:
            return folded
    if operator == "ABS" and is_non_negative(operand):
        return operand
    if (
        operator in IDEMPOTENT_UNARY_OPERATORS
        and isinstance(operand, UnaryOp)
        and operand.operator == operator
    ):
        return operand
    return UnaryOp(operator, operand)


def _simplify_binary(operator: str, left: ASTNode, right: ASTNode) -> ASTNode:
    if isinstance(left, Number) and isinstance(right, Number):
        folded = _fold_binary(operator, left.value, right.value)
        if folded is not None:
            return folded

    if operator in COMMUTATIVE_OPERATORS and str(right) < str(left):
        left, right = right, left

    if operator in {"ADD", "SUB"} and _is_constant(right, 0.0):
        return left
    if operator == "ADD" and _is_constant(left, 0.0):
        return right
    if operator in {"MUL", "DIV"} and _is_constant(right, 1.0):
        return left
    if operator == "MUL" and _is_constant(left, 1.0):
        return right
    return BinaryOp(operator, left, right)


def _is_constant(node: ASTNode, value: float) -> bool:
    return isinstance(node, Number) and node.value == value


def _fold_unary(operator: str, value: float) -> Number | None:
    if operator == "ABS":
        return _finite_number(abs(value))
    if operator == "SIGN":
        return Number(float((value > 0) - (value < 0)))
    if operator == "LOG" and value > 0:
        return _finite_number(math.log(value))
    # RANK of a constant depends on the number of stocks each day.
    return None


def _fold_binary(operator: str, left: float, right: float) -> Number | None:
    if operator == "ADD":
        return _finite_number(left + right)
    if operator == "SUB":
        return _finite_number(left - right)
    if operator == "MUL":
        return _finite_number(left * right)
    if operator == "DIV" and right != 0:
        return _finite_number(left / right)
    return None


def _finite_number(value: float) -> Number | None:
    return Number(float(value)) if math.isfinite(value) else None
