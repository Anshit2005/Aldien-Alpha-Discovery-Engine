import pytest

pytest.importorskip(
    "python.cpp_engine",
    reason="C++ module alpha_engine_cpp is not built (see README)",
)

import numpy as np
import pandas as pd

from python.dsl import parse
from python.evaluation.cpp_engine import CppAlphaEngine
from python.evaluation.evaluator import AlphaEvaluator


def build_test_data():
    return pd.DataFrame({
        "datetime": pd.to_datetime([
            "2020-01-01",
            "2020-01-02",

            "2020-01-01",
            "2020-01-02",

            "2020-01-01",
            "2020-01-02",
        ]),

        "instrument": [
            "A",
            "A",

            "B",
            "B",

            "C",
            "C",
        ],

        "open": [
            100,
            102,

            110,
            112,

            105,
            108,
        ],

        "high": [
            110,
            112,

            125,
            122,

            115,
            118,
        ],

        "low": [
            95,
            98,

            108,
            110,

            100,
            105,
        ],

        "close": [
            105,
            108,

            120,
            118,

            110,
            115,
        ],

        "volume": [
            1000,
            1100,

            2000,
            2100,

            1500,
            1600,
        ],
    })


def test_cpp_matches_python():

    data = build_test_data()

    # -----------------------------------------
    # Python evaluator
    # -----------------------------------------

    expression = (
        "RANK("
        "SUB("
        "CLOSE,"
        "TS_MEAN(CLOSE,2)"
        ")"
        ")"
    )

    ast = parse(expression)

    python_evaluator = AlphaEvaluator(data)

    python_result = python_evaluator.evaluate(ast)

    python_values = python_result["alpha"].to_numpy()


    # -----------------------------------------
    # C++ evaluator via the Python bridge. This
    # compares the same input rows and expression
    # instead of the standalone fitness demo data.
    # -----------------------------------------

    cpp_values = np.asarray(CppAlphaEngine(data).evaluate(ast))


    # -----------------------------------------
    # Compare
    # -----------------------------------------

    assert len(python_values) == len(cpp_values)

    np.testing.assert_allclose(
        python_values,
        cpp_values,
        rtol=1e-5,
        atol=1e-6,
        equal_nan=True,
    )
