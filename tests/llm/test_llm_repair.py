import json

import pytest

from python.dsl.parser import parse
from python.llm.client import MockClient
from python.llm.operator import LLMProposalOperator, repair_node_types
from python.llm.prompt import system_prompt


PARENT = parse("DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20))")

# The two answers qwen2.5:7b gave in the D7 smoke run that were rejected
# before the repair: the operator name was written as the node type.
NO_VALUE = (
    '{"node_type": "OPERATOR", "value": "RANK", "children": [{"node_type": "OPERATOR", '
    '"value": "DIV", "children": [{"node_type": "SUB", "children": [{"node_type": '
    '"VARIABLE", "value": "CLOSE"}, {"node_type": "OPERATOR", "value": "TS_MEAN", '
    '"children": [{"node_type": "VARIABLE", "value": "CLOSE"}, {"node_type": "CONSTANT", '
    '"value": "20"}]}]}, {"node_type": "OPERATOR", "value": "TS_STD", "children": '
    '[{"node_type": "VARIABLE", "value": "CLOSE"}, {"node_type": "CONSTANT", "value": '
    '"20"}]}]}]}'
)
SAME_VALUE = (
    '{"node_type": "OPERATOR", "value": "DIV", "children": [{"node_type": "SUB", "value": '
    '"SUB", "children": [{"node_type": "VARIABLE", "value": "CLOSE"}, {"node_type": '
    '"OPERATOR", "value": "TS_MEAN", "children": [{"node_type": "VARIABLE", "value": '
    '"CLOSE"}, {"node_type": "CONSTANT", "value": "50"}]}]}, {"node_type": "OPERATOR", '
    '"value": "TS_STD", "children": [{"node_type": "VARIABLE", "value": "CLOSE"}, '
    '{"node_type": "CONSTANT", "value": "50"}]}]}'
)


@pytest.mark.parametrize(
    "answer, expected",
    [
        (NO_VALUE, "RANK(DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20)))"),
        (SAME_VALUE, "DIV(SUB(CLOSE,TS_MEAN(CLOSE,50)),TS_STD(CLOSE,50))"),
    ],
)
def test_real_node_type_slips_are_repaired_and_accepted(answer, expected):

    operator = LLMProposalOperator(MockClient([answer]))

    proposal = operator.propose(PARENT)

    assert str(proposal) == expected
    stats = operator.stats()
    assert stats["accepted"] == 1 and stats["repaired_node_types"] == 1
    assert stats["rejected"] == 0


def test_repair_counts_every_fixed_node_and_leaves_correct_trees_alone():

    tree = {"node_type": "ADD", "children": [
        {"node_type": "ABS", "value": "ABS", "children": [{"node_type": "VARIABLE", "value": "OPEN"}]},
        {"node_type": "OPERATOR", "value": "LOG", "children": [{"node_type": "VARIABLE", "value": "CLOSE"}]},
    ]}

    fixed, count = repair_node_types(tree)

    assert count == 2
    assert fixed["node_type"] == "OPERATOR" and fixed["value"] == "ADD"
    assert fixed["children"][0] == {
        "node_type": "OPERATOR", "value": "ABS",
        "children": [{"node_type": "VARIABLE", "value": "OPEN"}],
    }
    assert tree["node_type"] == "ADD"  # the input is not modified
    assert repair_node_types(fixed) == (fixed, 0)


@pytest.mark.parametrize(
    "node",
    [
        # Two different operator names: the intent is unclear.
        {"node_type": "SUB", "value": "DIV", "children": []},
        # Not an operator of the grammar.
        {"node_type": "DELAY", "children": []},
        {"node_type": "variable", "value": "CLOSE"},
    ],
)
def test_ambiguous_or_unknown_node_types_are_not_repaired(node, tmp_path):

    assert repair_node_types(node) == (node, 0)
    operator = LLMProposalOperator(
        MockClient([json.dumps(node)]), failures_path=tmp_path / "failures.jsonl"
    )
    assert operator.propose(PARENT) is None
    assert operator.stats()["rejected_by_reason"] == {"invalid_formula": 1}
    assert operator.stats()["repaired_node_types"] == 0


def test_system_prompt_states_the_node_type_rule():

    prompt = system_prompt()

    assert '"node_type" is only ever VARIABLE, CONSTANT or OPERATOR' in prompt
    assert 'never {"node_type": "SUB", ...}' in prompt
