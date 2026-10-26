import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from python.dsl.parser import parse
from python.evaluation.fitness import FitnessEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "cpp_fitness_data.csv"
CPP_EXECUTABLE = PROJECT_ROOT / "engine" / "build" / "alpha_engine"

# Windows builds add an .exe suffix.
if not CPP_EXECUTABLE.exists() and CPP_EXECUTABLE.with_suffix(".exe").exists():
    CPP_EXECUTABLE = CPP_EXECUTABLE.with_suffix(".exe")
CPP_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cpp_fitness_output.txt"


EXPRESSION = "RANK(DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20)))"


def run_cpp_engine():
    """Run the C++ engine and capture its fitness output."""

    result = subprocess.run(
        [str(CPP_EXECUTABLE)],
        cwd=CPP_EXECUTABLE.parent,
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout

    print("\n===== C++ OUTPUT =====")
    print(output)

    return output


def parse_cpp_fitness(output):
    """Extract fitness metrics from C++ stdout."""

    metrics = {}

    for line in output.splitlines():

        if line.startswith("Mean IC:"):
            metrics["mean_ic"] = float(line.split(":", 1)[1].strip())

        elif line.startswith("IC Std:"):
            metrics["ic_std"] = float(line.split(":", 1)[1].strip())

        elif line.startswith("ICIR:"):
            metrics["icir"] = float(line.split(":", 1)[1].strip())

        elif line.startswith("Number of IC days:"):
            metrics["num_days"] = int(line.split(":", 1)[1].strip())

        elif line.startswith("Valid observations:"):
            metrics["num_observations"] = int(
                line.split(":", 1)[1].strip()
            )

        elif line.startswith("Coverage:"):
            metrics["coverage"] = float(
                line.split(":", 1)[1].strip()
            )

    required = {
        "mean_ic",
        "ic_std",
        "icir",
        "num_days",
        "num_observations",
        "coverage",
    }

    missing = required - metrics.keys()

    if missing:
        raise AssertionError(
            f"Could not parse C++ fitness metrics: {missing}"
        )

    return metrics


@pytest.mark.skipif(
    not CPP_EXECUTABLE.exists(),
    reason=f"C++ executable not built: {CPP_EXECUTABLE}",
)
@pytest.mark.skipif(
    not CSV_PATH.exists(),
    reason=(
        f"Missing fixture {CSV_PATH}; run "
        "python -m python.data.export_cpp_fitness_data"
    ),
)
def test_cpp_python_fitness_equivalence():

    # ---------------------------------------------------------
    # 1. Load the exact same dataset used by C++
    # ---------------------------------------------------------

    data = pd.read_csv(CSV_PATH)

    data["datetime"] = pd.to_datetime(data["datetime"])

    print("\n===== DATASET =====")
    print(f"Rows: {len(data)}")
    print(f"Instruments: {data['instrument'].nunique()}")
    print(f"Dates: {data['datetime'].nunique()}")

    # ---------------------------------------------------------
    # 2. Python Fitness
    # ---------------------------------------------------------

    ast = parse(EXPRESSION)

    python_fitness = FitnessEvaluator(data).evaluate(
        ast,
        horizon=20,
    )

    print("\n===== PYTHON FITNESS =====")
    print(f"Mean IC: {python_fitness.mean_ic}")
    print(f"IC Std: {python_fitness.ic_std}")
    print(f"ICIR: {python_fitness.icir}")
    print(f"Number of IC days: {python_fitness.num_days}")
    print(f"Valid observations: {python_fitness.num_observations}")
    print(f"Coverage: {python_fitness.coverage}")

    # ---------------------------------------------------------
    # 3. C++ Fitness
    # ---------------------------------------------------------

    cpp_output = run_cpp_engine()

    cpp_fitness = parse_cpp_fitness(cpp_output)

    # ---------------------------------------------------------
    # 4. Compare
    # ---------------------------------------------------------

    print("\n===== COMPARISON =====")

    print(
        f"Mean IC:          "
        f"{python_fitness.mean_ic:.12f} vs "
        f"{cpp_fitness['mean_ic']:.12f}"
    )

    print(
        f"IC Std:           "
        f"{python_fitness.ic_std:.12f} vs "
        f"{cpp_fitness['ic_std']:.12f}"
    )

    print(
        f"ICIR:             "
        f"{python_fitness.icir:.12f} vs "
        f"{cpp_fitness['icir']:.12f}"
    )

    print(
        f"Number of IC days:"
        f" {python_fitness.num_days} vs "
        f"{cpp_fitness['num_days']}"
    )

    print(
        f"Valid observations:"
        f" {python_fitness.num_observations} vs "
        f"{cpp_fitness['num_observations']}"
    )

    print(
        f"Coverage:         "
        f"{python_fitness.coverage:.12f} vs "
        f"{cpp_fitness['coverage']:.12f}"
    )

    # ---------------------------------------------------------
    # 5. Exact integer checks
    # ---------------------------------------------------------

    assert (
        python_fitness.num_days
        == cpp_fitness["num_days"]
    )

    assert (
        python_fitness.num_observations
        == cpp_fitness["num_observations"]
    )

    # ---------------------------------------------------------
    # 6. Floating-point checks
    # ---------------------------------------------------------

    np.testing.assert_allclose(
        python_fitness.mean_ic,
        cpp_fitness["mean_ic"],
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        python_fitness.ic_std,
        cpp_fitness["ic_std"],
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        python_fitness.icir,
        cpp_fitness["icir"],
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        python_fitness.coverage,
        cpp_fitness["coverage"],
        rtol=1e-10,
        atol=1e-12,
    )

    print("\n======================================")
    print("Python and C++ fitness outputs match.")
    print("======================================")