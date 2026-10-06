import pytest

from python.dsl import (
    parse,
    validate,
    ValidationError,
)

from python.dsl.ast import (
    Variable,
    Number,
    UnaryOp,
    BinaryOp,
    TimeSeriesOp,
)


def test_valid_variable():

    ast = parse("CLOSE")

    validate(ast)


def test_valid_binary_expression():

    ast = parse("SUB(CLOSE,OPEN)")

    validate(ast)


def test_valid_nested_expression():

    ast = parse(
        "RANK("
        "DIV("
        "SUB(CLOSE,TS_MEAN(CLOSE,20)),"
        "TS_STD(CLOSE,20)"
        ")"
        ")"
    )

    validate(ast)


def test_valid_window():

    ast = parse("TS_MEAN(CLOSE,252)")

    validate(ast)


def test_window_too_large():

    ast = TimeSeriesOp(
        operator="TS_MEAN",
        operand=Variable("CLOSE"),
        window=253,
    )

    with pytest.raises(ValidationError):
        validate(ast)


def test_zero_window():

    ast = TimeSeriesOp(
        operator="TS_MEAN",
        operand=Variable("CLOSE"),
        window=0,
    )

    with pytest.raises(ValidationError):
        validate(ast)


def test_unknown_variable():

    ast = Variable("PRICE")

    with pytest.raises(ValidationError):
        validate(ast)


def test_unknown_unary_operator():

    ast = UnaryOp(
        operator="SQRT",
        operand=Variable("CLOSE"),
    )

    with pytest.raises(ValidationError):
        validate(ast)


def test_unknown_binary_operator():

    ast = BinaryOp(
        operator="POW",
        left=Variable("CLOSE"),
        right=Number(2),
    )

    with pytest.raises(ValidationError):
        validate(ast)


def test_invalid_number():

    ast = Number(float("nan"))

    with pytest.raises(ValidationError):
        validate(ast)


def test_infinite_number():

    ast = Number(float("inf"))

    with pytest.raises(ValidationError):
        validate(ast)


def test_depth_limit():

    ast = Variable("CLOSE")

    for _ in range(8):
        ast = UnaryOp(
            operator="ABS",
            operand=ast,
        )

    with pytest.raises(ValidationError):
        validate(ast)