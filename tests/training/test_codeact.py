import pytest

from recursive_agent_harness.actions import ActionType
from recursive_agent_training.codeact.environment import CodeActEnvironment
from recursive_agent_training.codeact.parser import parse_codeact
from recursive_agent_training.codeact.sandbox import validate_codeact_ast


def test_codeact_parser_and_validator():
    action = parse_codeact("think\n```python\nx = 1 + 1\n```")
    assert action.thought == "think"
    validate_codeact_ast(action.code)


def test_codeact_parser_accepts_plain_and_thinking_prefixed_python():
    plain = parse_codeact('search_web("paper title")')
    thinking = parse_codeact(
        '<think>Need a concise search.</think>\nsearch_web("paper title")'
    )
    generic_fence = parse_codeact('```\nfinish("Exact title")\n```')
    assert plain.code == 'search_web("paper title")'
    assert thinking.thought == "Need a concise search."
    assert thinking.code == 'search_web("paper title")'
    assert generic_fence.code == 'finish("Exact title")'


def test_codeact_blocks_import():
    with pytest.raises(ValueError):
        validate_codeact_ast("import os")


def test_codeact_environment_persists_locals_and_emits_action():
    environment = CodeActEnvironment()
    action = environment.execute('query = "paper title"\nsearch_web(query)')
    assert action.type == ActionType.TOOL_CALL
    assert action.arguments["query"] == "paper title"
    second = environment.execute('finish(query)')
    assert second.type == ActionType.FINISH
    assert second.answer == "paper title"


def test_codeact_parallel_subagents_map_to_parallel_action():
    environment = CodeActEnvironment()
    action = environment.execute(
        'asyncio.gather(launch_subagent("find source"), launch_subagent("verify title"))'
    )
    assert action.type == ActionType.LAUNCH_SUBAGENTS_PARALLEL
    assert [item.goal for item in action.subtasks] == ["find source", "verify title"]


def test_codeact_blocks_arbitrary_attribute_access():
    with pytest.raises(ValueError, match="attribute"):
        validate_codeact_ast("x.__class__")


def test_codeact_blocks_loops():
    with pytest.raises(ValueError, match="While"):
        validate_codeact_ast("while True:\n    pass")


def test_codeact_requires_exactly_one_action_call():
    environment = CodeActEnvironment()

    with pytest.raises(ValueError, match="exactly one action"):
        environment.execute('think("first")\nfinish("second")')

    with pytest.raises(ValueError, match="exactly one action"):
        environment.execute("x = 1 + 1")
