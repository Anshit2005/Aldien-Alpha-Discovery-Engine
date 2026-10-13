import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from python.dsl.parser import parse
from python.evaluation.evaluator import AlphaEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cpp_test_data.csv"
)

CPP_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cpp_alpha_output.csv"
)

CPP_EXECUTABLE = (
    PROJECT_ROOT
    / "engine"
    / "build"
    / "alpha_engine"
)


EXPRESSION = (
    "RANK("
    "DIV("
    "SUB(CLOSE,TS_MEAN(CLOSE,20)),"
    "TS_STD(CLOSE,20)"
    ")"
    ")"
)


def test_real_cpp_python_equivalence():

    # ------------------------------------------------------------
    # 1. Check required files
    # ------------------------------------------------------------

    assert INPUT_FILE.exists(), (
        f"Input dataset not found: {INPUT_FILE}"
    )

    assert CPP_EXECUTABLE.exists(), (
        f"C++ executable not found: {CPP_EXECUTABLE}"
    )

    # ------------------------------------------------------------
    # 2. Run C++ engine
    # ------------------------------------------------------------

    result = subprocess.run(
        [str(CPP_EXECUTABLE)],
        cwd=PROJECT_ROOT / "engine" / "build",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "C++ engine failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    assert CPP_OUTPUT_FILE.exists(), (
        f"C++ output not found: {CPP_OUTPUT_FILE}"
    )

    # ------------------------------------------------------------
    # 3. Load market data
    # ------------------------------------------------------------

    data = pd.read_csv(INPUT_FILE)

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    data = data.sort_values(
        ["instrument", "datetime"]
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # 4. Evaluate using Python reference implementation
    # ------------------------------------------------------------

    ast = parse(EXPRESSION)

    evaluator = AlphaEvaluator(data)

    python_output = evaluator.evaluate(ast)

    python_output = python_output[
        ["datetime", "instrument", "alpha"]
    ].copy()

    # ------------------------------------------------------------
    # 5. Load C++ output
    # ------------------------------------------------------------

    cpp_output = pd.read_csv(
        CPP_OUTPUT_FILE
    )

    cpp_output["datetime"] = pd.to_datetime(
        cpp_output["datetime"]
    )

    cpp_output = cpp_output[
        ["datetime", "instrument", "alpha"]
    ].copy()

    # ------------------------------------------------------------
    # 6. Check row counts
    # ------------------------------------------------------------

    assert len(python_output) == len(cpp_output), (
        f"Row count mismatch: "
        f"Python={len(python_output)}, "
        f"C++={len(cpp_output)}"
    )

    # ------------------------------------------------------------
    # 7. Match by (datetime, instrument)
    # ------------------------------------------------------------

    merged = python_output.merge(
        cpp_output,
        on=["datetime", "instrument"],
        how="outer",
        suffixes=("_python", "_cpp"),
        indicator=True,
    )

    # Every row must exist in both outputs
    assert (merged["_merge"] == "both").all(), (
        "Python and C++ have different "
        "(datetime, instrument) keys."
    )

    # ------------------------------------------------------------
    # 8. Compare numerical values
    # ------------------------------------------------------------

    python_alpha = (
        merged["alpha_python"].to_numpy()
    )

    cpp_alpha = (
        merged["alpha_cpp"].to_numpy()
    )

    # NaN must occur at exactly the same locations
    python_nan = np.isnan(python_alpha)
    cpp_nan = np.isnan(cpp_alpha)

    assert np.array_equal(
        python_nan,
        cpp_nan,
    ), (
        "Python and C++ disagree on NaN locations."
    )

    # Compare finite values
    valid = ~python_nan

    np.testing.assert_allclose(
        python_alpha[valid],
        cpp_alpha[valid],
        rtol=1e-10,
        atol=1e-12,
    )

    # ------------------------------------------------------------
    # 9. Print useful information
    # ------------------------------------------------------------

    print()
    print("Expression:")
    print(EXPRESSION)

    print(
        f"Rows compared: {len(merged)}"
    )

    print(
        f"Valid values compared: {valid.sum()}"
    )

    print(
        f"NaN values: {python_nan.sum()}"
    )

    print(
        "Python and C++ outputs match."
    )