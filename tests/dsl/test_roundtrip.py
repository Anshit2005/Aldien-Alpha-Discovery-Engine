import pytest

from python.dsl import parse, validate


@pytest.mark.parametrize(
    "expression",
    [
        "CLOSE",
        "OPEN",
        "SUB(CLOSE,OPEN)",
        "MUL(CLOSE,VOLUME)",
        "RANK(CLOSE)",
        "ABS(SUB(CLOSE,OPEN))",
        "TS_MEAN(CLOSE,20)",
        "TS_STD(VOLUME,60)",
        "RANK(DIV(SUB(CLOSE,OPEN),OPEN))",
        "RANK(DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20)))",
    ],
)
def test_roundtrip(expression):

    ast = parse(expression)

    validate(ast)

    serialized = ast.to_string()

    ast2 = parse(serialized)

    validate(ast2)

    assert ast2.to_string() == serialized