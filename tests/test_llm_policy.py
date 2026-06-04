import pytest

from recursive_agent_harness.actions import ActionType
from recursive_agent_harness.policy import ChatClient, ChatCompletionResult, LLMPolicy
from recursive_agent_harness.state import AgentState
from tests.helpers import make_config


class FakeChatClient(ChatClient):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages, model, temperature, max_tokens):
        self.calls.append(messages)
        return ChatCompletionResult(content=self.responses.pop(0), raw_response={"ok": True})


def make_state() -> AgentState:
    return AgentState(
        node_id="node_0001",
        parent_id=None,
        task="Summarize task",
        depth=0,
        max_depth=2,
        remaining_steps=3,
        remaining_children=2,
        parent_context={},
        expected_output="answer",
        constraints={},
        trajectory=[],
        child_summaries=[],
        available_tools=[],
        global_tree_summary={"total_nodes": 1},
    )


@pytest.mark.asyncio
async def test_llm_policy_parses_valid_json_action():
    client = FakeChatClient(['{"type":"FINISH","answer":"done"}'])
    policy = LLMPolicy(client=client, config=make_config())

    action = await policy.act(make_state())

    assert action.type == ActionType.FINISH
    assert action.answer == "done"
    assert action.metadata["raw_response"] == '{"type":"FINISH","answer":"done"}'


@pytest.mark.asyncio
async def test_llm_policy_repairs_invalid_json_once():
    client = FakeChatClient(["not json", '{"type":"FINISH","answer":"repaired"}'])
    policy = LLMPolicy(client=client, config=make_config())

    action = await policy.act(make_state())

    assert action.type == ActionType.FINISH
    assert action.answer == "repaired"
    assert action.metadata["repaired"] is True
    assert "parse_error" in action.metadata
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_llm_policy_falls_back_after_repair_failure():
    client = FakeChatClient(["not json", "still not json"])
    policy = LLMPolicy(client=client, config=make_config())

    action = await policy.act(make_state())

    assert action.type == ActionType.FINISH
    assert action.metadata["fallback"] is True
    assert "could not be parsed" in action.limitations[0]
