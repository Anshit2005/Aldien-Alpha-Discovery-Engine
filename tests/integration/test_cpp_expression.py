import pytest

pytest.importorskip(
    "python.cpp_engine",
    reason="C++ module alpha_engine_cpp is not built (see README)",
)

from pathlib import Path

import numpy as np
import pandas as pd

from python.evaluation.cpp_engine import (
    CppAlphaEngine,
)
from python.evaluation.evaluator import (
    AlphaEvaluator,
)
from python.dsl.parser import parse


TEST_DATA = Path(
    "data/processed/cpp_fitness_data.csv"
)


def load_test_data():

    return pd.read_csv(
        TEST_DATA,
        parse_dates=["datetime"]
    )


@pytest.mark.skipif(
    not TEST_DATA.exists(),
    reason=(
        f"Missing fixture {TEST_DATA}; run "
        "python -m python.data.export_cpp_fitness_data"
    ),
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