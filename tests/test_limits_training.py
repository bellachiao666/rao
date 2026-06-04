import pytest

from recursive_agent_harness.actions import ActionType, AgentAction, SubtaskSpec
from recursive_agent_harness.policy import Policy
from recursive_agent_harness.runner import Runner
from recursive_agent_harness.state import AgentState, NodeStatus
from tests.helpers import make_config


class ManyChildrenPolicy(Policy):
    async def act(self, state: AgentState) -> AgentAction:
        if state.depth == 0 and not state.child_summaries:
            return AgentAction(
                type=ActionType.LAUNCH_SUBAGENTS_PARALLEL,
                subtasks=[
                    SubtaskSpec(goal="child one", expected_output="answer"),
                    SubtaskSpec(goal="child two", expected_output="answer"),
                ],
            )
        return AgentAction(type=ActionType.FINISH, answer=f"done: {state.task}")


@pytest.mark.asyncio
async def test_parallel_launch_respects_max_children_per_node(tmp_path):
    config = make_config(
        max_children_per_node=1,
        max_parallel_children=4,
        trace_output_path=str(tmp_path / "trace.json"),
    )
    runner = Runner(policy=ManyChildrenPolicy(), config=config)

    result, tree = await runner.arun("Root task")

    assert result.status == NodeStatus.COMPLETED
    assert len(result.children) == 2
    assert result.children[0].status == NodeStatus.COMPLETED
    assert result.children[1].status == NodeStatus.CHILD_LIMITED
    assert tree.summary()["total_nodes"] == 2


@pytest.mark.asyncio
async def test_total_node_limit_returns_node_limited_child_result(tmp_path):
    config = make_config(
        total_node_limit=1,
        trace_output_path=str(tmp_path / "trace.json"),
    )
    runner = Runner(policy=ManyChildrenPolicy(), config=config)

    result, tree = await runner.arun("Root task")

    assert result.children[0].status == NodeStatus.NODE_LIMITED
    assert tree.summary()["total_nodes"] == 1


@pytest.mark.asyncio
async def test_trace_nodes_keep_training_extension_placeholders(tmp_path):
    config = make_config(trace_output_path=str(tmp_path / "trace.json"))
    runner = Runner(policy=ManyChildrenPolicy(), config=config)

    _, tree = await runner.arun("Root task")

    root_metadata = tree.nodes["node_0001"].metadata
    assert "success_signal" in root_metadata
    assert "reward" in root_metadata
    assert root_metadata["success_signal"] is None
    assert root_metadata["reward"] is None
