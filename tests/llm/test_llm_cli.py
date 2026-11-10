import argparse

from python.dsl.parser import parse
from python.llm.cli import add_llm_arguments, build_llm_operator, llm_config_fields


def _args(tmp_path, *extra):
    parser = argparse.ArgumentParser()
    add_llm_arguments(parser)
    return parser.parse_args([
        "--llm-rate", "0.3", "--llm-backend", "mock",
        "--llm-cache", str(tmp_path / "cache.jsonl"),
        "--llm-failures", str(tmp_path / "failures.jsonl"),
        *extra,
    ])


def _prompt_seen(operator):
    prompts = []

    def answer(system, user):
        prompts.append(user)
        return "not json"

    operator.client._responses = answer
    operator.propose(parse("DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20))"))
    return prompts[0]


def test_retrieval_puts_related_factors_in_the_prompt(tmp_path):

    operator = build_llm_operator(_args(tmp_path), max_depth=8, horizon=20)

    assert "Related known factors:" in _prompt_seen(operator)
    assert llm_config_fields(_args(tmp_path), operator)["llm_retrieval"] is True


def test_no_retrieval_leaves_them_out(tmp_path):

    for operator in (
        build_llm_operator(_args(tmp_path, "--no-llm-retrieval"), max_depth=8, horizon=20),
        build_llm_operator(_args(tmp_path), max_depth=8, horizon=20, retrieval=False),
    ):
        assert "Related known factors:" not in _prompt_seen(operator)
        assert llm_config_fields(_args(tmp_path), operator)["llm_retrieval"] is False
