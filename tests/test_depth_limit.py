import pytest

from recursive_agent_harness.actions import ActionType, AgentAction, SubtaskSpec
from recursive_agent_harness.policy import Policy
from recursive_agent_harness.runner import Runner
from recursive_agent_harness.state import AgentState, NodeStatus
from tests.helpers import make_config


class ScriptedPolicy(Policy):
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = 0

    async def act(self, state: AgentState) -> AgentAction:
        if self.calls < len(self.actions):
            action = self.actions[self.calls]
        else:
            action = AgentAction(type=ActionType.FINISH, answer="script complete")
        self.calls += 1
        return action


@pytest.mark.asyncio
async def test_depth_limit_returns_structured_child_result(tmp_path):
    policy = ScriptedPolicy(
        [
            AgentAction(
                type=ActionType.LAUNCH_SUBAGENT,
                subtask=SubtaskSpec(goal="Too deep", expected_output="anything"),
            ),
            AgentAction(type=ActionType.FINISH, answer="finished after depth limit"),
        ]
    )
    config = make_config(max_depth=0, trace_output_path=str(tmp_path / "trace.json"))
    runner = Runner(policy=policy, config=config)

    result, tree = await runner.arun("Root task")

    assert result.status == NodeStatus.COMPLETED
    assert result.children[0].status == NodeStatus.DEPTH_LIMITED
    assert result.children[0].task == "Too deep"
    assert tree.summary()["total_nodes"] == 1
