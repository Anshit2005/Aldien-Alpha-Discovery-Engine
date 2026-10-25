import json

import pytest

from python.dsl import (
    BinaryOp,
    Number,
    TimeSeriesOp,
    UnaryOp,
    Variable,
    ValidationError,
    dumps,
    from_json,
    loads,
    parse,
    to_json,
    validate,
)
from python.search.evolution import EvolutionSearch


# =========================================================
# Round trip
# =========================================================

@pytest.mark.parametrize(
    "expression",
    [
        "CLOSE",
        "SUB(CLOSE,OPEN)",
        "RANK(CLOSE)",
        "TS_MEAN(CLOSE,20)",
        "RANK(DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20)))",
        "MUL(CLOSE,-1)",
        "ADD(CLOSE,1e-05)",
        "ADD(CLOSE,0.5)",
        *EvolutionSearch.SEEDS,
    ],
)
def test_json_roundtrip_preserves_expression(expression):

    ast = parse(expression)
    validate(ast)

    rebuilt = from_json(to_json(ast))
    validate(rebuilt)

    assert rebuilt == ast
    assert rebuilt.to_string() == ast.to_string()


def test_dumps_loads_roundtrip_is_valid_json():

    ast = parse("ADD(TS_MEAN(CLOSE,10),VOLUME)")
    text = dumps(ast)

    assert json.loads(text) == to_json(ast)
    assert loads(text) == ast


def test_to_json_shape_matches_wire_format():

    ast = parse("ADD(TS_MEAN(CLOSE,10),VOLUME)")

    assert to_json(ast) == {
        "node_type": "OPERATOR",
        "value": "ADD",
        "children": [
            {
                "node_type": "OPERATOR",
                "value": "TS_MEAN",
                "children": [
                    {"node_type": "VARIABLE", "value": "CLOSE"},
                    {"node_type": "CONSTANT", "value": "10"},
                ],
            },
            {"node_type": "VARIABLE", "value": "VOLUME"},
        ],
    }


# =========================================================
# Accepted input variants
# =========================================================

def test_from_json_accepts_numeric_constants():

    node = from_json({"node_type": "CONSTANT", "value": 2})
    assert node == Number(2.0)

    node = from_json({"node_type": "CONSTANT", "value": 0.25})
    assert node == Number(0.25)


def test_from_json_accepts_numeric_window():

    node = from_json({
        "node_type": "OPERATOR",
        "value": "TS_MAX",
        "children": [
            {"node_type": "VARIABLE", "value": "HIGH"},
            {"node_type": "CONSTANT", "value": 20},
        ],
    })

    assert node == TimeSeriesOp("TS_MAX", Variable("HIGH"), 20)


def test_from_json_builds_typed_nodes():

    node = from_json({
        "node_type": "OPERATOR",
        "value": "RANK",
        "children": [
            {
                "node_type": "OPERATOR",
                "value": "DIV",
                "children": [
                    {"node_type": "VARIABLE", "value": "CLOSE"},
                    {"node_type": "CONSTANT", "value": "2"},
                ],
            }
        ],
    })

    assert node == UnaryOp(
        "RANK",
        BinaryOp("DIV", Variable("CLOSE"), Number(2.0)),
    )


# =========================================================
# Rejections
# =========================================================

@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {"node_type": "OPERATOR", "value": "ADD",
             "children": [{"node_type": "VARIABLE", "value": "CLOSE"}]},
            "ADD expects 2 children",
        ),
        (
            {"node_type": "OPERATOR", "value": "RANK",
             "children": [
                 {"node_type": "VARIABLE", "value": "CLOSE"},
                 {"node_type": "VARIABLE", "value": "OPEN"},
             ]},
            "RANK expects 1 children",
        ),
        (
            {"node_type": "OPERATOR", "value": "TS_MEAN",
             "children": [
                 {"node_type": "VARIABLE", "value": "CLOSE"},
                 {"node_type": "CONSTANT", "value": "abc"},
             ]},
            "Invalid numeric constant",
        ),
        (
            {"node_type": "OPERATOR", "value": "TS_MEAN",
             "children": [
                 {"node_type": "VARIABLE", "value": "CLOSE"},
                 {"node_type": "CONSTANT", "value": "2.5"},
             ]},
            "window must be an integer",
        ),
        (
            {"node_type": "OPERATOR", "value": "TS_MEAN",
             "children": [
                 {"node_type": "VARIABLE", "value": "CLOSE"},
                 {"node_type": "VARIABLE", "value": "VOLUME"},
             ]},
            "window must be a CONSTANT node",
        ),
        (
            {"node_type": "OPERATOR", "value": "MEAN",
             "children": [
                 {"node_type": "VARIABLE", "value": "CLOSE"},
                 {"node_type": "CONSTANT", "value": "10"},
             ]},
            "Unknown operator",
        ),
        (
            {"node_type": "LEAF", "value": "CLOSE"},
            "Unknown node_type",
        ),
        (
            {"node_type": "CONSTANT", "value": "nan-ish"},
            "Invalid numeric constant",
        ),
        (
            {"node_type": "CONSTANT", "value": True},
            "Invalid numeric constant",
        ),
        (
            {"node_type": "VARIABLE", "value": "CLOSE",
             "children": [{"node_type": "VARIABLE", "value": "OPEN"}]},
            "cannot have children",
        ),
        (
            {"node_type": "OPERATOR", "value": "ADD", "children": "CLOSE"},
            "'children' must be a list",
        ),
        (
            ["not", "an", "object"],
            "must be a JSON object",
        ),
    ],
)
def test_from_json_rejects_malformed_trees(payload, message):

    with pytest.raises(ValidationError, match=message):
        from_json(payload)


def test_loads_rejects_invalid_json_text():

    with pytest.raises(ValidationError, match="Invalid JSON"):
        loads("{not json")


def test_validator_still_applies_after_from_json():

    # Structure is fine, but the window exceeds the research limit.
    node = from_json({
        "node_type": "OPERATOR",
        "value": "TS_MEAN",
        "children": [
            {"node_type": "VARIABLE", "value": "CLOSE"},
            {"node_type": "CONSTANT", "value": "500"},
        ],
    })

    with pytest.raises(ValidationError, match="Window exceeds maximum"):
        validate(node)

    # Unknown variable names are left to the validator as well.
    node = from_json({"node_type": "VARIABLE", "value": "VWAP"})

    with pytest.raises(ValidationError, match="Unknown variable"):
        validate(node)
