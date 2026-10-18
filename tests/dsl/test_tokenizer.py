from python.dsl.tokenizer import tokenize


def test_simple_expression():
    tokens = tokenize("ADD(CLOSE,OPEN)")

    values = [token.value for token in tokens]

    assert values == [
        "ADD",
        "(",
        "CLOSE",
        ",",
        "OPEN",
        ")",
    ]


def test_whitespace_is_ignored():
    tokens = tokenize("  ADD( CLOSE , OPEN ) ")

    values = [token.value for token in tokens]

    assert values == [
        "ADD",
        "(",
        "CLOSE",
        ",",
        "OPEN",
        ")",
    ]


def test_number_token():
    tokens = tokenize("TS_MEAN(CLOSE,20)")

    values = [token.value for token in tokens]

    assert "20" in values


def test_decimal_number():
    tokens = tokenize("MUL(CLOSE,0.5)")

    values = [token.value for token in tokens]

    assert "0.5" in values


def test_signed_and_scientific_numbers():
    values = [token.value for token in tokenize("ADD(CLOSE,-1.25e-3)")]

    assert "-1.25e-3" in values


def test_invalid_character():
    import pytest

    with pytest.raises(ValueError):
        tokenize("ADD(CLOSE,@)")
