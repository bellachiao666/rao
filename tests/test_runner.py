import asyncio
import json

import pytest

from recursive_agent_harness.actions import ActionType, AgentAction
from recursive_agent_harness.policy import Policy
from recursive_agent_harness.runner import Runner
from recursive_agent_harness.state import AgentState, NodeStatus
from tests.helpers import make_config


class SlowPolicy(Policy):
    async def act(self, state: AgentState) -> AgentAction:
        await asyncio.sleep(1)
        return AgentAction(type=ActionType.FINISH, answer="too late")


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
async def test_runner_timeout_preserves_partial_trace(tmp_path):
    config = make_config(timeout_seconds=0.05, trace_output_path=str(tmp_path / "trace.json"))
    runner = Runner(policy=SlowPolicy(), config=config)

    result, tree = await runner.arun("Slow root")

    assert result.status == NodeStatus.TIMEOUT
    assert tree.nodes["node_0001"].status == NodeStatus.TIMEOUT
    assert (tmp_path / "trace.json").exists()


@pytest.mark.asyncio
async def test_batch_runner_exports_manifest(tmp_path):
    config = make_config(trace_output_path=str(tmp_path / "trace.json"))
    runner = Runner(
        policy=ScriptedPolicy([AgentAction(type=ActionType.FINISH, answer="done")]),
        config=config,
    )

    manifest = await runner.arun_batch(["task one", "task two"], output_dir=tmp_path)

    assert manifest["total_runs"] == 2
    assert len(manifest["runs"]) == 2
    assert (tmp_path / "batch_manifest.json").exists()
    saved = json.loads((tmp_path / "batch_manifest.json").read_text(encoding="utf-8"))
    assert saved["total_runs"] == 2
