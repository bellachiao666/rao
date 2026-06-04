import pytest

from recursive_agent_harness.actions import ActionType, AgentAction, SubtaskSpec
from recursive_agent_harness.policy import Policy
from recursive_agent_harness.runner import Runner
from recursive_agent_harness.state import AgentState
from tests.helpers import make_config


class TrackingPolicy(Policy):
    def __init__(self):
        self.seen_depths: list[int] = []
        self.self_ids: list[int] = []

    async def act(self, state: AgentState) -> AgentAction:
        self.seen_depths.append(state.depth)
        self.self_ids.append(id(self))
        if state.depth == 0 and not state.child_summaries:
            return AgentAction(
                type=ActionType.LAUNCH_SUBAGENT,
                subtask=SubtaskSpec(goal="child task", expected_output="child answer"),
            )
        return AgentAction(type=ActionType.FINISH, answer=f"done at depth {state.depth}")


@pytest.mark.asyncio
async def test_root_and_child_use_same_policy_object(tmp_path):
    policy = TrackingPolicy()
    runner = Runner(policy=policy, config=make_config(trace_output_path=str(tmp_path / "trace.json")))

    await runner.arun("Root task")

    assert 0 in policy.seen_depths
    assert 1 in policy.seen_depths
    assert len(set(policy.self_ids)) == 1
