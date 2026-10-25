"""JSON serialization for the alpha DSL AST.

The JSON shape is a generic tree that an LLM can emit directly:

    {"node_type": "OPERATOR", "value": "ADD", "children": [
        {"node_type": "OPERATOR", "value": "TS_MEAN", "children": [
            {"node_type": "VARIABLE", "value": "CLOSE"},
            {"node_type": "CONSTANT", "value": "10"}
        ]},
        {"node_type": "VARIABLE", "value": "VOLUME"}
    ]}

``from_json`` turns that shape into the typed AST in ``python.dsl.ast``.
Arity, constant parsing and the integer window of time-series operators are
checked here because the typed nodes cannot represent them any other way; the
remaining research constraints (depth, window range, finite constants, known
variables) stay in ``python.dsl.validator.validate``, which callers should
still run on the result.
"""

from __future__ import annotations

import json
from typing import Any

from .ast import ASTNode, BinaryOp, Number, TimeSeriesOp, UnaryOp, Variable
from .validator import (
    ALLOWED_BINARY_OPERATORS,
    ALLOWED_TIME_SERIES_OPERATORS,
    ALLOWED_UNARY_OPERATORS,
    ValidationError,
)


NODE_VARIABLE = "VARIABLE"
NODE_CONSTANT = "CONSTANT"
NODE_OPERATOR = "OPERATOR"


# ============================================================
# AST -> JSON
# ============================================================

def to_json(node: ASTNode) -> dict[str, Any]:
    """Convert an AST into the JSON-compatible dict shape."""

    if isinstance(node, Variable):
        return {"node_type": NODE_VARIABLE, "value": node.name}

    if isinstance(node, Number):
        return {"node_type": NODE_CONSTANT, "value": node.to_string()}

    if isinstance(node, UnaryOp):
        return {
            "node_type": NODE_OPERATOR,
            "value": node.operator,
            "children": [to_json(node.operand)],
        }

    if isinstance(node, BinaryOp):
        return {
            "node_type": NODE_OPERATOR,
            "value": node.operator,
            "children": [to_json(node.left), to_json(node.right)],
        }

    if isinstance(node, TimeSeriesOp):
        return {
            "node_type": NODE_OPERATOR,
            "value": node.operator,
            "children": [
                to_json(node.operand),
                {"node_type": NODE_CONSTANT, "value": str(node.window)},
            ],
        }

    raise TypeError(f"Unsupported AST node: {type(node).__name__}")


def dumps(node: ASTNode) -> str:
    """Serialize an AST to a compact JSON string."""
    return json.dumps(to_json(node), separators=(",", ":"))


# ============================================================
# JSON -> AST
# ============================================================

def from_json(obj: Any) -> ASTNode:
    """
    Build a typed AST from the JSON dict shape.

    Raises:
        ValidationError: if the structure cannot be mapped onto a valid node
            (unknown node_type or operator, wrong arity, non-numeric constant,
            non-integer time-series window).
    """

    if not isinstance(obj, dict):
        raise ValidationError(
            f"AST node must be a JSON object, got {type(obj).__name__}"
        )

    node_type = obj.get("node_type")
    value = obj.get("value")
    children = obj.get("children", [])

    if not isinstance(children, list):
        raise ValidationError("'children' must be a list")

    # ---------------------------------------------------------
    # Leaves
    # ---------------------------------------------------------

    if node_type == NODE_VARIABLE:
        if children:
            raise ValidationError("Variable nodes cannot have children")
        if not isinstance(value, str):
            raise ValidationError(f"Variable name must be a string: {value!r}")
        return Variable(value)

    if node_type == NODE_CONSTANT:
        if children:
            raise ValidationError("Constant nodes cannot have children")
        return Number(_parse_constant(value))

    if node_type != NODE_OPERATOR:
        raise ValidationError(f"Unknown node_type: {node_type!r}")

    # ---------------------------------------------------------
    # Operators
    # ---------------------------------------------------------

    if not isinstance(value, str):
        raise ValidationError(f"Operator name must be a string: {value!r}")

    if value in ALLOWED_UNARY_OPERATORS:
        _check_arity(value, children, 1)
        return UnaryOp(value, from_json(children[0]))

    if value in ALLOWED_BINARY_OPERATORS:
        _check_arity(value, children, 2)
        return BinaryOp(value, from_json(children[0]), from_json(children[1]))

    if value in ALLOWED_TIME_SERIES_OPERATORS:
        _check_arity(value, children, 2)
        return TimeSeriesOp(
            value,
            from_json(children[0]),
            _parse_window(value, children[1]),
        )

    raise ValidationError(f"Unknown operator: {value!r}")


def loads(text: str) -> ASTNode:
    """Parse a JSON string into a typed AST."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValidationError(f"Invalid JSON: {error}") from error
    return from_json(obj)


# ============================================================
# Helpers
# ============================================================

def _check_arity(operator: str, children: list, expected: int) -> None:
    if len(children) != expected:
        raise ValidationError(
            f"{operator} expects {expected} children, got {len(children)}"
        )


def _parse_constant(value: Any) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"Invalid numeric constant: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    raise ValidationError(f"Invalid numeric constant: {value!r}")


def _parse_window(operator: str, child: Any) -> int:
    if not isinstance(child, dict) or child.get("node_type") != NODE_CONSTANT:
        raise ValidationError(
            f"{operator} window must be a CONSTANT node"
        )
    if child.get("children"):
        raise ValidationError("Constant nodes cannot have children")
    number = _parse_constant(child.get("value"))
    if not number.is_integer():
        raise ValidationError(
            f"{operator} window must be an integer: {child.get('value')!r}"
        )
    return int(number)
