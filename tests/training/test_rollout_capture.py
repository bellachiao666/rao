import pytest

from recursive_agent_harness.state import NodeStatus
from recursive_agent_harness.tree import ExecutionTree
from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.grouping import collect_rollout_group
from recursive_agent_training.mini import make_harness_config, make_scripted_policy, synthetic_task
from recursive_agent_training.rollout import tree_to_rollout_record
from recursive_agent_training.schemas import TreeStatus
from recursive_agent_training.snapshots import PolicySnapshotProvider
from recursive_agent_training.validation import validate_group


@pytest.mark.asyncio
async def test_grouped_recursive_rollout_captures_tokens_and_shared_snapshot(tmp_path):
    config = TrainingConfig.from_yaml("configs/train/mini_rao.yaml")
    snapshot = PolicySnapshotProvider(model_name="scripted").current()
    bundle = await collect_rollout_group(
        synthetic_task(),
        snapshot,
        2,
        lambda index, seed: make_scripted_policy(snapshot, "alpha" if index == 0 else "beta"),
        make_harness_config(config, str(tmp_path)),
        output_dir=tmp_path,
    )
    assert validate_group(bundle).valid
    assert len(bundle.rollouts) == 2
    assert all(len(tree.nodes) == 2 for tree in bundle.rollouts)
    assert all(node.tokens.generated_ids for tree in bundle.rollouts for node in tree.nodes.values())
    assert all(len(tree.nodes[tree.root_node_id].turns) == 2 for tree in bundle.rollouts)
    assert all(
        len(node.turns) == 1
        for tree in bundle.rollouts
        for node_id, node in tree.nodes.items()
        if node_id != tree.root_node_id
    )
    assert all(
        "_training_trace" not in tree.nodes[tree.root_node_id].turns[1].model_input.rendered_prompt
        for tree in bundle.rollouts
    )
    assert {tree.behavior_policy_version for tree in bundle.rollouts} == {snapshot.version}
    assert {tree.behavior_checkpoint_path for tree in bundle.rollouts} == {snapshot.checkpoint_path}


@pytest.mark.parametrize("status", [NodeStatus.TIMEOUT, NodeStatus.FAILED])
def test_failed_execution_tree_is_not_complete_for_training(status):
    tree = ExecutionTree(run_id="failed")
    tree.register_node("node_0001", None, 0, "task")
    tree.update_node_status("node_0001", status, error="failed")
    snapshot = PolicySnapshotProvider(model_name="scripted").current()

    record = tree_to_rollout_record(
        tree,
        task_id="task",
        group_id="group",
        rollout_id="rollout",
        rollout_index=0,
        seed=0,
        snapshot=snapshot,
    )

    assert record.tree_status == TreeStatus.INCOMPLETE
