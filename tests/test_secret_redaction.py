import json

import pytest

from recursive_agent_harness.actions import ActionType, AgentAction
from recursive_agent_harness.policy import Policy
from recursive_agent_harness.runner import Runner
from recursive_agent_harness.state import AgentState
from tests.helpers import make_config


class LeakyPolicy(Policy):
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    async def act(self, state: AgentState) -> AgentAction:
        return AgentAction(
            type=ActionType.FINISH,
            answer=f"Configured endpoint is {self.base_url}; configured key is {self.api_key}.",
            evidence=[f"evidence mentions {self.api_key}"],
        )


@pytest.mark.asyncio
async def test_runner_redacts_config_secrets_from_result_and_trace(tmp_path):
    base_url = "https://sensitive-api.example/v1"
    api_key = "sk-sensitive-test-key"
    trace_path = tmp_path / "trace.json"
    config = make_config(
        base_url=base_url,
        api_key=api_key,
        trace_output_path=str(trace_path),
        mermaid_output_path="",
    )
    runner = Runner(policy=LeakyPolicy(base_url=base_url, api_key=api_key), config=config)

    result, tree = await runner.arun("Return the configured endpoint and key.")
    trace_text = trace_path.read_text(encoding="utf-8")
    tree_text = json.dumps(tree.to_dict(), ensure_ascii=False)
    result_text = result.model_dump_json()

    assert base_url not in result_text
    assert api_key not in result_text
    assert base_url not in trace_text
    assert api_key not in trace_text
    assert base_url not in tree_text
    assert api_key not in tree_text
    assert "[REDACTED_BASE_URL]" in result.answer
    assert "[REDACTED_API_KEY]" in result.answer
