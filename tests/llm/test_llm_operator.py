import json

import pytest

from python.dsl.json_io import dumps
from python.dsl.parser import parse
from python.llm.cache import ResponseCache, request_key
from python.llm.client import LLMError, MockClient, make_client
from python.llm.operator import LLMProposalOperator, extract_json


PARENT = parse("DIV(SUB(HIGH,LOW),CLOSE)")
GOOD = dumps(parse("DIV(SUB(HIGH,LOW),TS_MEAN(CLOSE,20))"))


def _operator(responses, tmp_path=None, **options):
    failures = tmp_path / "failures.jsonl" if tmp_path is not None else None
    return LLMProposalOperator(MockClient(responses), failures_path=failures, **options)


def test_valid_answer_is_accepted_and_simplified():

    answer = dumps(parse("DIV(SUB(ABS(HIGH),LOW),TS_MEAN(CLOSE,20))"))
    operator = _operator([answer])

    proposal = operator.propose(PARENT)

    assert str(proposal) == "DIV(SUB(HIGH,LOW),TS_MEAN(CLOSE,20))"
    assert operator.stats()["accepted"] == 1
    assert operator.stats()["acceptance_rate"] == 1.0


@pytest.mark.parametrize(
    "wrapped",
    [
        "```json\n{answer}\n```",
        "Here is a formula:\n{answer}\nIt normalises the range.",
    ],
)
def test_markdown_fences_and_prose_are_stripped(wrapped):

    operator = _operator([wrapped.format(answer=GOOD)])

    assert str(operator.propose(PARENT)) == "DIV(SUB(HIGH,LOW),TS_MEAN(CLOSE,20))"


def _deep(depth):
    node = "CLOSE"
    for _ in range(depth - 1):
        node = f"ABS(SUB({node},OPEN))"
    return node


@pytest.mark.parametrize(
    "response, reason",
    [
        ("I cannot help with that.", "no_json"),
        ('{"node_type": "OPERATOR", "value": "ADD", "children": [', "no_json"),
        ('{"node_type": "OPERATOR", "value": "ADD" "children": []}', "invalid_json"),
        (
            json.dumps({"node_type": "OPERATOR", "value": "ADD",
                        "children": [{"node_type": "VARIABLE", "value": "CLOSE"}]}),
            "invalid_formula",
        ),
        (
            json.dumps({"node_type": "OPERATOR", "value": "DELAY",
                        "children": [{"node_type": "VARIABLE", "value": "CLOSE"},
                                     {"node_type": "CONSTANT", "value": "5"}]}),
            "invalid_formula",
        ),
        (
            json.dumps({"node_type": "OPERATOR", "value": "TS_MEAN",
                        "children": [{"node_type": "VARIABLE", "value": "CLOSE"},
                                     {"node_type": "CONSTANT", "value": "500"}]}),
            "invalid_formula",
        ),
        (json.dumps({"node_type": "VARIABLE", "value": "VWAP"}), "invalid_formula"),
        (dumps(parse("DIV(SUB(HIGH,LOW),ABS(CLOSE))")), "same_as_parent"),
    ],
)
def test_bad_answers_are_rejected_logged_and_return_none(response, reason, tmp_path):

    operator = _operator([response], tmp_path)

    assert operator.propose(PARENT) is None
    assert operator.stats()["rejected_by_reason"] == {reason: 1}
    record = json.loads((tmp_path / "failures.jsonl").read_text(encoding="utf-8"))
    assert record["response"] == response
    assert record["error"].startswith(reason)
    assert len(record["prompt_hash"]) == 64


def test_too_deep_answer_is_rejected(tmp_path):

    operator = _operator([dumps(parse(_deep(5)))], tmp_path, max_depth=4)

    assert operator.propose(PARENT) is None
    assert operator.stats()["rejected_by_reason"] == {"too_deep": 1}


def test_client_errors_are_rejected_and_not_cached(tmp_path):

    cache = ResponseCache(tmp_path / "cache.jsonl")
    operator = _operator([LLMError("connection refused"), GOOD], tmp_path, cache=cache)

    assert operator.propose(PARENT) is None
    assert operator.stats()["rejected_by_reason"] == {"client_error": 1}
    assert len(cache) == 0
    # The next call reaches the model again and succeeds.
    assert operator.propose(PARENT) is not None


def test_failures_are_shown_in_the_next_prompt():

    prompts = []

    def respond(system, user):
        prompts.append(user)
        return "no formula here" if len(prompts) == 1 else GOOD

    operator = _operator(respond)
    operator.propose(PARENT)
    operator.propose(PARENT)

    assert "Recently rejected" not in prompts[0]
    assert "no_json" in prompts[1]


def test_cache_answers_repeated_prompts_and_survives_restart(tmp_path):

    path = tmp_path / "cache.jsonl"
    client = MockClient([GOOD])
    first = LLMProposalOperator(client, cache=ResponseCache(path))

    a = first.propose(PARENT)
    b = first.propose(PARENT)

    assert client.calls == 1
    assert first.stats()["cache_hits"] == 1
    assert a == b

    # A new run with the same prompts makes no request at all.
    offline = MockClient([LLMError("must not be called")])
    second = LLMProposalOperator(offline, cache=ResponseCache(path))
    assert second.propose(PARENT) == a
    assert offline.calls == 0


def test_cache_ignores_a_truncated_last_line(tmp_path):

    path = tmp_path / "cache.jsonl"
    key = request_key("mock", "mock", "s", "u")
    path.write_text(json.dumps({"key": key, "response": "r"}) + "\n{\"key\": \"trunc", encoding="utf-8")

    assert ResponseCache(path).get(key) == "r"


def test_cache_key_depends_on_every_part():

    keys = {
        request_key("mock", "mock", "s", "u"),
        request_key("groq", "mock", "s", "u"),
        request_key("mock", "other", "s", "u"),
        request_key("mock", "mock", "s2", "u"),
        request_key("mock", "mock", "s", "u2"),
    }
    assert len(keys) == 5


def test_extract_json_takes_the_outermost_object():

    assert extract_json('text {"a": {"b": 1}} more') == '{"a": {"b": 1}}'


def test_default_mock_varies_the_parent_offline():

    operator = LLMProposalOperator(MockClient())

    proposal = operator.propose(PARENT)

    assert proposal is not None
    assert str(PARENT) in str(proposal)


def test_make_client_backends(monkeypatch, tmp_path):

    assert make_client("mock").backend == "mock"
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        make_client("groq")
    with pytest.raises(ValueError, match="base-url"):
        make_client("openai-compatible")
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        make_client("gpt")
    ollama = make_client("ollama")
    assert (ollama.backend, ollama.model) == ("ollama", "qwen2.5:7b")
