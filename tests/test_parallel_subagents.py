import asyncio
import time

import pytest

from recursive_agent_harness.actions import ActionType, AgentAction, SubtaskSpec
from recursive_agent_harness.policy import Policy
from recursive_agent_harness.runner import Runner
from recursive_agent_harness.state import AgentState, NodeStatus
from tests.helpers import make_config


class ParallelDemoPolicy(Policy):
    async def act(self, state: AgentState) -> AgentAction:
        if state.depth == 0 and not state.child_summaries:
            return AgentAction(
                type=ActionType.LAUNCH_SUBAGENTS_PARALLEL,
                subtasks=[
                    SubtaskSpec(goal="leaf one", expected_output="answer"),
                    SubtaskSpec(goal="leaf two", expected_output="answer"),
                ],
            )
        if state.depth == 0:
            return AgentAction(type=ActionType.FINISH, answer="combined")
        await asyncio.sleep(0.1)
        return AgentAction(type=ActionType.FINISH, answer=f"answer for {state.task}")


@pytest.mark.asyncio
async def test_parallel_subagents_run_concurrently(tmp_path):
    config = make_config(trace_output_path=str(tmp_path / "trace.json"))
    runner = Runner(policy=ParallelDemoPolicy(), config=config)

    started = time.perf_counter()
    result, tree = await runner.arun("Root task")
    elapsed = time.perf_counter() - started

    assert result.status == NodeStatus.COMPLETED
    assert elapsed < 0.18
    assert tree.summary()["total_nodes"] == 3
    assert tree.nodes["node_0001"].children_ids == ["node_0002", "node_0003"]
