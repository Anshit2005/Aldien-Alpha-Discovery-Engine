import pytest

from python.dsl import (
    parse,
    Variable,
    Number,
    UnaryOp,
    BinaryOp,
    TimeSeriesOp,
)


def test_parse_variable():
    ast = parse("CLOSE")

    assert isinstance(ast, Variable)
    assert ast.name == "CLOSE"


def test_parse_number():
    ast = parse("20")

    assert isinstance(ast, Number)
    assert ast.value == 20


def test_parse_binary_operator():
    ast = parse("SUB(CLOSE,OPEN)")

    assert isinstance(ast, BinaryOp)
    assert ast.operator == "SUB"

    assert isinstance(ast.left, Variable)
    assert ast.left.name == "CLOSE"

    assert isinstance(ast.right, Variable)
    assert ast.right.name == "OPEN"


def test_parse_unary_operator():
    ast = parse("RANK(CLOSE)")

    assert isinstance(ast, UnaryOp)
    assert ast.operator == "RANK"

    assert isinstance(ast.operand, Variable)
    assert ast.operand.name == "CLOSE"


def test_parse_time_series_operator():
    ast = parse("TS_MEAN(CLOSE,20)")

    assert isinstance(ast, TimeSeriesOp)

    assert ast.operator == "TS_MEAN"
    assert ast.window == 20

    assert isinstance(ast.operand, Variable)
    assert ast.operand.name == "CLOSE"


def test_parse_nested_expression():

    expression = (
        "RANK("
        "DIV("
        "SUB(CLOSE,TS_MEAN(CLOSE,20)),"
        "TS_STD(CLOSE,20)"
        ")"
        ")"
    )

    ast = parse(expression)

    assert isinstance(ast, UnaryOp)
    assert ast.operator == "RANK"

    assert isinstance(ast.operand, BinaryOp)
    assert ast.operand.operator == "DIV"


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "HELLO",
        "ADD(CLOSE)",
        "ADD(CLOSE,)",
        "ADD(,CLOSE)",
        "TS_MEAN(CLOSE)",
        "TS_MEAN(CLOSE,20.5)",
        "TS_MEAN(CLOSE,0)",
        "CLOSE(20)",
    ],
)
def test_invalid_expressions(expression):

    with pytest.raises(ValueError):
        parse(expression)