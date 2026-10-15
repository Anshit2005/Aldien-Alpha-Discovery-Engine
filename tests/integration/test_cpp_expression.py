import numpy as np
import pandas as pd

from python.evaluation.cpp_engine import (
    CppAlphaEngine,
)
from python.evaluation.evaluator import (
    AlphaEvaluator,
)
from python.dsl.parser import parse


def load_test_data():

    path = (
        "data/processed/cpp_fitness_data.csv"
    )

    return pd.read_csv(
        path,
        parse_dates=["datetime"]
    )


def test_full_expression_matches_python():

    data = load_test_data()

    expression = (
        "RANK("
        "DIV("
        "SUB(CLOSE,TS_MEAN(CLOSE,20)),"
        "TS_STD(CLOSE,20)"
        ")"
        ")"
    )

    # --------------------------------------------------------
    # Python reference
    # --------------------------------------------------------

    python_ast = parse(expression)

    python_result = AlphaEvaluator(
        data
    ).evaluate(
        python_ast
    )

    python_result = (
        python_result
        .sort_values(
            ["instrument", "datetime"]
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # C++ engine
    # --------------------------------------------------------

    cpp_engine = CppAlphaEngine(
        data
    )

    cpp_result = (
        cpp_engine
        .evaluate_expression(expression)
    )


    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    expected = (
        python_result["alpha"]
        .to_numpy()
    )

    assert len(cpp_result) == len(expected)

    np.testing.assert_allclose(
        cpp_result,
        expected,
        equal_nan=True,
        rtol=1e-10,
        atol=1e-10
    )