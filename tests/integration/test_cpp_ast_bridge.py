import numpy as np
import pandas as pd

from python.dsl.parser import parse
from python.evaluation.cpp_ast_bridge import (
    python_ast_to_cpp,
    CppASTConversionError,
)
from python.evaluation.evaluator import AlphaEvaluator


def make_data():

    rows = []

    dates = pd.date_range(
        "2020-01-01",
        periods=5,
        freq="D"
    )

    for instrument, base in [
        ("A", 10.0),
        ("B", 20.0),
        ("C", 30.0),
    ]:

        for i, date in enumerate(dates):

            close = base + i

            rows.append(
                {
                    "datetime": date,
                    "instrument": instrument,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0,
                }
            )

    return pd.DataFrame(rows)


def test_variable_conversion():

    python_ast = parse("CLOSE")

    cpp_ast = python_ast_to_cpp(
        python_ast
    )

    assert cpp_ast is not None


def test_number_conversion():

    python_ast = parse("5.0")

    cpp_ast = python_ast_to_cpp(
        python_ast
    )

    assert cpp_ast is not None


def test_binary_conversion():

    python_ast = parse(
        "SUB(CLOSE,OPEN)"
    )

    cpp_ast = python_ast_to_cpp(
        python_ast
    )

    assert cpp_ast is not None


def test_nested_conversion():

    expression = (
        "RANK("
        "DIV("
        "SUB(CLOSE,TS_MEAN(CLOSE,2)),"
        "TS_STD(CLOSE,2)"
        ")"
        ")"
    )

    python_ast = parse(expression)

    cpp_ast = python_ast_to_cpp(
        python_ast
    )

    assert cpp_ast is not None


def test_cpp_and_python_evaluation_match():

    data = make_data()

    expression = (
        "RANK("
        "SUB(CLOSE,TS_MEAN(CLOSE,2))"
        ")"
    )

    python_ast = parse(expression)

    # Python reference implementation
    python_result = AlphaEvaluator(
        data
    ).evaluate(python_ast)

    # C++ implementation
    from python.evaluation.cpp_engine import (
        CppAlphaEngine,
    )

    cpp_engine = CppAlphaEngine(data)

    cpp_result = cpp_engine.evaluate_expression(
        expression
    )

    expected = (
        python_result
        .sort_values(
            ["instrument", "datetime"]
        )["alpha"]
        .to_numpy()
    )

    np.testing.assert_allclose(
        cpp_result,
        expected,
        equal_nan=True
    )