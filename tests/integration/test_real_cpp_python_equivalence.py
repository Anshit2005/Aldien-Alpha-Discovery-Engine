"""
C++ and Python give the same alpha values on a sample of the real data.

Uses the compiled module (engine/build/alpha_engine_cpp) through
CppAlphaEngine. The standalone executable no longer writes alpha values;
it prints a fitness summary instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from python.dsl.parser import parse
from python.evaluation.evaluator import AlphaEvaluator

pytest.importorskip(
    "python.cpp_engine",
    reason="C++ module alpha_engine_cpp is not built (see README)",
)
from python.evaluation import cpp_engine  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cpp_test_data.csv"

EXPRESSIONS = [
    "RANK(DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20)))",
    "DIV(ADD(MUL(LOG(OPEN),TS_MAX(VOLUME,20)),VOLUME),TS_MEAN(2,60))",
    "SIGN(SUB(TS_MIN(LOW,10),TS_MAX(HIGH,5)))",
    "ABS(DIV(VOLUME,TS_MEAN(VOLUME,60)))",
]


@pytest.fixture(scope="module")
def sample():
    if not INPUT_FILE.exists():
        pytest.skip(
            f"Missing fixture {INPUT_FILE}; run python -m python.data.export_cpp_test_data"
        )
    data = pd.read_csv(INPUT_FILE)
    data["datetime"] = pd.to_datetime(data["datetime"])
    return data.sort_values(["instrument", "datetime"]).reset_index(drop=True)


@pytest.mark.parametrize("expression", EXPRESSIONS)
def test_real_cpp_python_equivalence(sample, expression):

    ast = parse(expression)
    python_alpha = AlphaEvaluator(sample).evaluate(ast)["alpha"].to_numpy(dtype=float)
    cpp_alpha = np.asarray(cpp_engine.CppAlphaEngine(sample).evaluate(ast), dtype=float)

    assert len(python_alpha) == len(cpp_alpha) == len(sample)
    python_nan, cpp_nan = np.isnan(python_alpha), np.isnan(cpp_alpha)
    assert np.array_equal(python_nan, cpp_nan), "Python and C++ disagree on NaN locations."
    np.testing.assert_allclose(
        python_alpha[~python_nan], cpp_alpha[~python_nan], rtol=1e-10, atol=1e-12
    )
