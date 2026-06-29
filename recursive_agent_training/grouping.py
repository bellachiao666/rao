"""Grouped recursive rollout collection."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Callable

from recursive_agent_harness.config import HarnessConfig
from recursive_agent_harness.policy import Policy
from recursive_agent_harness.runner import Runner
from recursive_agent_harness.tools import ToolRegistry
from recursive_agent_training.rollout import tree_to_rollout_record
from recursive_agent_training.schemas import GroupStatus, RolloutGroupBundle, RolloutGroupRecord, RootTaskRecord
from recursive_agent_training.snapshots import PolicySnapshot


PolicyFactory = Callable[[int, int], Policy]


def make_group_id(task_id: str, policy_version: str, sequence: int) -> str:
    digest = hashlib.sha256(f"{task_id}:{policy_version}:{sequence}".encode("utf-8")).hexdigest()[:16]
    return f"group_{digest}"


async def collect_rollout_group(
    task: RootTaskRecord,
    snapshot: PolicySnapshot,
    group_size: int,
    policy_factory: PolicyFactory,
    harness_config: HarnessConfig,
    tools: ToolRegistry | None = None,
    group_sequence: int = 0,
    output_dir: str | Path | None = None,
) -> RolloutGroupBundle:
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    group_id = make_group_id(task.task_id, snapshot.version, group_sequence)

    async def collect_one(index: int):
        seed = group_sequence * 100_000 + index
        rollout_id = f"{group_id}_rollout_{index:04d}"
        config_updates = {}
        if output_dir:
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            config_updates = {
                "trace_output_path": str(output / f"{rollout_id}.json"),
                "mermaid_output_path": str(output / f"{rollout_id}.mmd"),
            }
        config = harness_config.model_copy(update=config_updates)
        policy = policy_factory(index, seed)
        runner = Runner(policy=policy, config=config, tools=tools)
        _, tree = await runner.arun(task.prompt)
        return tree_to_rollout_record(tree, task.task_id, group_id, rollout_id, index, seed, snapshot)

    rollouts = await asyncio.gather(*(collect_one(index) for index in range(group_size)))
    complete = all(tree.tree_status.value == "complete" for tree in rollouts)
    group = RolloutGroupRecord(
        group_id=group_id,
        task_id=task.task_id,
        expected_group_size=group_size,
        completed_group_size=sum(tree.tree_status.value == "complete" for tree in rollouts),
        behavior_policy_version=snapshot.version,
        behavior_checkpoint_path=snapshot.checkpoint_path,
        behavior_checkpoint_hash=snapshot.checkpoint_hash,
        sampling_config_hash=hashlib.sha256(
            f"{harness_config.temperature}:{harness_config.max_tokens}:{group_size}".encode("utf-8")
        ).hexdigest(),
        rollout_ids=[tree.rollout_id for tree in rollouts],
        status=GroupStatus.COMPLETE if complete else GroupStatus.INCOMPLETE,
    )
    return RolloutGroupBundle(group=group, task=task, rollouts=rollouts)
