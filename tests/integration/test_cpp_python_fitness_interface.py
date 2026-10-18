import numpy as np
import pandas as pd

from python.evaluation.cpp_engine import CppAlphaEngine


def test_cpp_python_fitness_interface():

    data = pd.read_csv(
        "data/processed/cpp_fitness_data.csv"
    )

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    expression = (
        "RANK("
        "DIV("
        "SUB(CLOSE,TS_MEAN(CLOSE,20)),"
        "TS_STD(CLOSE,20)"
        ")"
        ")"
    )

    engine = CppAlphaEngine(data)

    result = engine.evaluate_fitness_expression(
        expression,
        horizon=20
    )

    print("\n===== PYTHON → C++ FITNESS INTERFACE =====")
    print(f"Mean IC: {result.mean_ic}")
    print(f"IC Std: {result.ic_std}")
    print(f"ICIR: {result.icir}")
    print(f"Number of IC days: {result.num_days}")
    print(f"Valid observations: {result.num_observations}")
    print(f"Coverage: {result.coverage}")

    assert np.isfinite(result.mean_ic)
    assert np.isfinite(result.ic_std)
    assert np.isfinite(result.icir)

    assert result.num_days == 481
    assert result.num_observations == 6696

    np.testing.assert_allclose(
        result.mean_ic,
        -0.0520248771269,
        rtol=1e-8,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        result.ic_std,
        0.327282677141,
        rtol=1e-8,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        result.icir,
        -0.158960069569,
        rtol=1e-8,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        result.coverage,
        0.916757940854,
        rtol=1e-8,
        atol=1e-10,
    )


def test_cpp_python_fitness_interface_is_split_aware():
    rows = []
    for day in range(5):
        for instrument, offset in [("A", 0), ("B", 10), ("C", 20)]:
            rows.append({
                "datetime": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day),
                "instrument": instrument,
                "open": 1.0 + offset + day,
                "high": 2.0 + offset + day,
                "low": 0.0 + offset + day,
                "close": 1.0 + offset + day,
                "volume": 100.0,
                "forward_return_20d": float(offset) + day,
                "split": "train",
                "target_split_20d": "train" if day < 4 else "validation",
            })

    result = CppAlphaEngine(pd.DataFrame(rows)).evaluate_fitness_expression(
        "CLOSE", horizon=20, split="train"
    )

    assert result.num_days == 4
    assert result.num_observations == 12
    assert result.coverage == 0.8
