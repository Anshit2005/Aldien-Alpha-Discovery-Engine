import numpy as np
import pandas as pd

from python.dsl import parse, validate
from python.evaluation import AlphaEvaluator


# =========================================================
# Test Data
# =========================================================

def make_test_data():

    dates = pd.date_range(
        "2025-01-01",
        periods=5,
        freq="D",
    )

    rows = []

    for i, date in enumerate(dates):

        rows.append({
            "datetime": date,
            "instrument": "AAA",
            "open": 10 + i,
            "high": 11 + i,
            "low": 9 + i,
            "close": 10 + i,
            "volume": 100 + i,
        })

        rows.append({
            "datetime": date,
            "instrument": "BBB",
            "open": 20 + i,
            "high": 21 + i,
            "low": 19 + i,
            "close": 20 + i,
            "volume": 200 + i,
        })

    return pd.DataFrame(rows)


# =========================================================
# Helper
# =========================================================

def evaluate(expression):

    ast = parse(expression)
    validate(ast)

    evaluator = AlphaEvaluator(
        make_test_data()
    )

    return evaluator.evaluate(ast)


def get_instrument_values(result, instrument):

    return (
        result[
            result["instrument"] == instrument
        ]
        .sort_values("datetime")
        .reset_index(drop=True)
    )


# =========================================================
# Basic Operators
# =========================================================

def test_variable():

    result = evaluate("CLOSE")

    aaa = get_instrument_values(result, "AAA")
    bbb = get_instrument_values(result, "BBB")

    assert aaa["alpha"].iloc[0] == 10
    assert aaa["alpha"].iloc[1] == 11

    assert bbb["alpha"].iloc[0] == 20
    assert bbb["alpha"].iloc[1] == 21


def test_number():

    result = evaluate("5")

    assert (result["alpha"] == 5).all()


def test_add():

    result = evaluate(
        "ADD(CLOSE,OPEN)"
    )

    aaa = get_instrument_values(result, "AAA")

    assert aaa["alpha"].iloc[0] == 20


def test_sub():

    result = evaluate(
        "SUB(CLOSE,OPEN)"
    )

    assert (result["alpha"] == 0).all()


def test_mul():

    result = evaluate(
        "MUL(CLOSE,2)"
    )

    aaa = get_instrument_values(result, "AAA")

    assert aaa["alpha"].iloc[0] == 20


def test_div():

    result = evaluate(
        "DIV(CLOSE,2)"
    )

    aaa = get_instrument_values(result, "AAA")

    assert aaa["alpha"].iloc[0] == 5


# =========================================================
# Unary Operators
# =========================================================

def test_log():

    result = evaluate(
        "LOG(CLOSE)"
    )

    aaa = get_instrument_values(result, "AAA")

    assert np.isclose(
        aaa["alpha"].iloc[0],
        np.log(10),
    )


def test_abs():

    result = evaluate(
        "ABS(SUB(OPEN,CLOSE))"
    )

    assert (result["alpha"] == 0).all()


def test_sign():

    result = evaluate(
        "SIGN(SUB(CLOSE,OPEN))"
    )

    assert (result["alpha"] == 0).all()


# =========================================================
# Time-Series Operators
# =========================================================

def test_ts_mean():

    result = evaluate(
        "TS_MEAN(CLOSE,3)"
    )

    aaa = get_instrument_values(result, "AAA")

    # AAA close:
    #
    # 10
    # 11
    # 12
    # 13
    # 14
    #
    # 3-period rolling mean:
    #
    # NaN
    # NaN
    # 11
    # 12
    # 13

    assert np.isnan(
        aaa["alpha"].iloc[0]
    )

    assert np.isnan(
        aaa["alpha"].iloc[1]
    )

    assert np.isclose(
        aaa["alpha"].iloc[2],
        11,
    )

    assert np.isclose(
        aaa["alpha"].iloc[3],
        12,
    )

    assert np.isclose(
        aaa["alpha"].iloc[4],
        13,
    )


# =========================================================
# Cross-Sectional Rank
# =========================================================

def test_cross_sectional_rank():

    result = evaluate(
        "RANK(CLOSE)"
    )

    # On every date:
    #
    # AAA CLOSE < BBB CLOSE
    #
    # Therefore:
    #
    # rank(AAA) < rank(BBB)

    for date, group in result.groupby(
        "datetime"
    ):

        aaa_rank = group.loc[
            group["instrument"] == "AAA",
            "alpha",
        ].iloc[0]

        bbb_rank = group.loc[
            group["instrument"] == "BBB",
            "alpha",
        ].iloc[0]

        assert aaa_rank < bbb_rank


# =========================================================
# Nested Expression
# =========================================================

def test_nested_expression():

    result = evaluate(
        "DIV("
        "SUB(CLOSE,TS_MEAN(CLOSE,3)),"
        "TS_STD(CLOSE,3)"
        ")"
    )

    aaa = get_instrument_values(result, "AAA")

    # First two observations are invalid
    # because the rolling window is incomplete.

    assert np.isnan(
        aaa["alpha"].iloc[0]
    )

    assert np.isnan(
        aaa["alpha"].iloc[1]
    )

    assert np.isfinite(
        aaa["alpha"].iloc[2:]
    ).all()


# =========================================================
# Edge Cases
# =========================================================

def test_division_by_zero():

    result = evaluate(
        "DIV("
        "CLOSE,"
        "SUB(CLOSE,CLOSE)"
        ")"
    )

    assert result["alpha"].isna().all()


def test_invalid_log():

    data = make_test_data()

    data.loc[0, "close"] = -1

    evaluator = AlphaEvaluator(data)

    ast = parse("LOG(CLOSE)")
    validate(ast)

    result = evaluator.evaluate(ast)

    assert np.isnan(
        result["alpha"].iloc[0]
    )


# =========================================================
# Output Structure
# =========================================================

def test_output_structure():

    result = evaluate("CLOSE")

    assert list(result.columns) == [
        "datetime",
        "instrument",
        "alpha",
    ]

    assert len(result) == 10

    assert result["datetime"].notna().all()

    assert result["instrument"].notna().all()

    assert result["alpha"].notna().all()


def test_output_identity():

    result = evaluate("CLOSE")

    aaa = (
        result[
            result["instrument"] == "AAA"
        ]
        .sort_values("datetime")
    )

    assert aaa["alpha"].tolist() == [
        10,
        11,
        12,
        13,
        14,
    ]