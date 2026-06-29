import pytest

from recursive_agent_training.compilation import compile_optimizer_batch, verify_and_score_group
from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.grouping import collect_rollout_group
from recursive_agent_training.mini import make_harness_config, make_scripted_policy, synthetic_task
from recursive_agent_training.rewards import compute_node_reward
from recursive_agent_training.schemas import (
    ModelInputRecord,
    RolloutGroupBundle,
    RolloutGroupRecord,
    RolloutTreeRecord,
    RootTaskRecord,
    SuccessSignalResult,
    TokenRecord,
    TrainingNodeRecord,
    TreeStatus,
    GroupStatus,
    VerifierStatus,
)
from recursive_agent_training.snapshots import PolicySnapshotProvider
from recursive_agent_training.verifiers.exact import ExactMatchVerifier


@pytest.mark.asyncio
async def test_compiler_assigns_reward_advantage_and_depth_weight(tmp_path):
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
    await verify_and_score_group(bundle, ExactMatchVerifier(), 0)
    batch = compile_optimizer_batch([bundle], config, 0, snapshot.version)
    assert batch.manifest.tree_count == 2
    assert batch.manifest.trainable_node_count == 4
    assert batch.manifest.trainable_turn_count == 6
    assert len(batch.samples) == 6
    assert batch.manifest.depth_counts == {0: 2, 1: 2}
    root_samples = [sample for sample in batch.samples if sample.depth == 0]
    assert {sample.advantage for sample in root_samples} == {-1.0, 1.0}


def _scored_root(
    rollout_id,
    value,
    *,
    is_fallback=False,
    terminal_reason="finish",
    terminal_status="completed",
):
    return TrainingNodeRecord(
        task_id="t",
        group_id="g",
        rollout_id=rollout_id,
        node_id="root",
        parent_id=None,
        depth=0,
        node_task="task",
        terminal_status=terminal_status,
        terminal_reason=terminal_reason,
        final_answer="answer",
        is_fallback=is_fallback,
        model_input=ModelInputRecord(rendered_prompt="prompt"),
        tokens=TokenRecord(
            input_ids=[1],
            generated_ids=[2],
            action_mask=[1],
            behavior_logprobs=[-1.0],
        ),
        evaluation=SuccessSignalResult(
            value=value,
            provider="exact",
            status=VerifierStatus.COMPLETE,
        ),
        credit={"reward": compute_node_reward(value, [], 0)},
        behavior_policy_version="policy_000000",
    )


def _two_rollout_bundle():
    success = _scored_root("r_success", 1.0)
    step_limited = _scored_root(
        "r_step_limit",
        0.0,
        is_fallback=True,
        terminal_reason="step_limit",
        terminal_status="step_limited",
    )
    trees = [
        RolloutTreeRecord(
            rollout_id=node.rollout_id,
            group_id="g",
            rollout_index=index,
            seed=index,
            root_node_id="root",
            behavior_policy_version="policy_000000",
            nodes={"root": node},
            tree_status=TreeStatus.COMPLETE,
        ).finalize()
        for index, node in enumerate([success, step_limited])
    ]
    return RolloutGroupBundle(
        group=RolloutGroupRecord(
            group_id="g",
            task_id="t",
            expected_group_size=2,
            completed_group_size=2,
            behavior_policy_version="policy_000000",
            rollout_ids=["r_success", "r_step_limit"],
            status=GroupStatus.COMPLETE,
        ),
        task=RootTaskRecord(
            task_id="t",
            domain="test",
            split="train",
            prompt="task",
            ground_truth="answer",
        ),
        rollouts=trees,
    )


def test_compiler_uses_train_failed_trajectories_for_step_limit_failures():
    config = TrainingConfig.from_yaml("configs/train/mini_rao.yaml")
    config.reward.train_failed_trajectories = True

    batch = compile_optimizer_batch(
        [_two_rollout_bundle()],
        config,
        0,
        "policy_000000",
    )

    by_rollout = {sample.rollout_id: sample for sample in batch.samples}
    assert by_rollout["r_success"].advantage == pytest.approx(1.0)
    assert by_rollout["r_step_limit"].advantage == pytest.approx(-1.0)


def test_compiler_excludes_step_limit_failures_when_failed_training_disabled():
    config = TrainingConfig.from_yaml("configs/train/mini_rao.yaml")
    config.reward.train_failed_trajectories = False

    batch = compile_optimizer_batch(
        [_two_rollout_bundle()],
        config,
        0,
        "policy_000000",
    )

    assert {sample.rollout_id for sample in batch.samples} == {"r_success"}
