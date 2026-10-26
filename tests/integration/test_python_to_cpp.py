import pytest

pytest.importorskip(
    "python.cpp_engine",
    reason="C++ module alpha_engine_cpp is not built (see README)",
)

import numpy as np
import pandas as pd

from python.cpp_engine import alpha_engine_cpp
from python.evaluation.cpp_engine import CppAlphaEngine


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


def test_cpp_module_import():
    assert alpha_engine_cpp is not None


def test_cpp_variable_evaluation():

    data = make_data()

    engine = CppAlphaEngine(data)

    ast = alpha_engine_cpp.ASTNode.variable(
        "CLOSE"
    )

    result = engine.evaluate(ast)

    expected = data["close"].to_numpy()

    np.testing.assert_allclose(
        result,
        expected,
        equal_nan=True
    )


def test_cpp_arithmetic():

    data = make_data()

    engine = CppAlphaEngine(data)

    close = alpha_engine_cpp.ASTNode.variable(
        "CLOSE"
    )

    number = alpha_engine_cpp.ASTNode.number(
        2.0
    )

    ast = alpha_engine_cpp.ASTNode.binary(
        "MUL",
        close,
        number
    )

    result = engine.evaluate(ast)

    expected = (
        data["close"].to_numpy() * 2.0
    )

    np.testing.assert_allclose(
        result,
        expected,
        equal_nan=True
    )