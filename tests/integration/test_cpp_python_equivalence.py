import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from python.dsl import parse
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
    # C++ evaluator
    # -----------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    executable = (
        project_root
        / "engine"
        / "build"
        / "alpha_engine"
    )

    process = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        check=True,
    )

    cpp_lines = process.stdout.strip().splitlines()

    cpp_values = []

    for line in cpp_lines:

        parts = line.split(",")

        value = parts[2]

        if value.lower() == "nan":
            cpp_values.append(np.nan)
        else:
            cpp_values.append(float(value))

    cpp_values = np.array(cpp_values)


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