import pytest

from recursive_agent_harness.policy import Policy
from recursive_agent_harness.runner import Runner
from recursive_agent_harness.state import AgentState, NodeStatus
from tests.helpers import make_config


class InvalidActionPolicy(Policy):
    async def act(self, state: AgentState):  # type: ignore[override]
        return {"type": "NOT_A_REAL_ACTION"}


@pytest.mark.asyncio
async def test_invalid_action_falls_back_to_structured_finish(tmp_path):
    config = make_config(
        invalid_action_limit=1,
        max_steps_per_node=3,
        trace_output_path=str(tmp_path / "trace.json"),
        mermaid_output_path=str(tmp_path / "trace.mmd"),
    )
    runner = Runner(policy=InvalidActionPolicy(), config=config)

    result, tree = await runner.arun("Task that receives invalid action")

    assert result.status == NodeStatus.COMPLETED
    assert "best effort" in (result.answer or "").lower()
    observations = [
        step.observation
        for node in tree.nodes.values()
        for step in node.trajectory
        if step.kind == "observation"
    ]
    assert any(obs and obs.get("type") == "invalid_action" for obs in observations)
