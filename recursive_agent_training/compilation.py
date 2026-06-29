"""Verification, credit assignment, and optimizer-batch compilation."""

from __future__ import annotations

from collections import Counter

from recursive_agent_training.advantages import compute_group_advantages
from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.rewards import compute_node_reward
from recursive_agent_training.schemas import (
    CompiledBatch,
    OptimizerBatchManifest,
    OptimizerSample,
    PolicyTurnRecord,
    RolloutGroupBundle,
    VerifierStatus,
)
from recursive_agent_training.validation import assess_node_trainability, validate_group
from recursive_agent_training.verifiers.base import SuccessSignalProvider
from recursive_agent_training.weighting import compute_depth_weights


async def verify_and_score_group(
    bundle: RolloutGroupBundle,
    verifier: SuccessSignalProvider,
    delegation_lambda: float,
) -> RolloutGroupBundle:
    for tree in bundle.rollouts:
        for node in tree.nodes.values():
            node.evaluation = await verifier.evaluate(node, tree, bundle.task)
        for node in tree.nodes.values():
            if node.evaluation is None or node.evaluation.status != VerifierStatus.COMPLETE or node.evaluation.value is None:
                continue
            child_values = [
                tree.nodes[child_id].evaluation.value
                for child_id in node.children_ids
                if child_id in tree.nodes
                and tree.nodes[child_id].evaluation is not None
                and tree.nodes[child_id].evaluation.value is not None
            ]
            node.credit.reward = compute_node_reward(node.evaluation.value, child_values, delegation_lambda)
        tree.finalize()
    return bundle


def compile_optimizer_batch(
    bundles: list[RolloutGroupBundle],
    config: TrainingConfig,
    optimizer_step: int,
    policy_version_before: str,
) -> CompiledBatch:
    if not bundles:
        raise ValueError("at least one rollout group is required")
    candidate_nodes = []
    for bundle in bundles:
        group_validation = validate_group(bundle)
        if not group_validation.valid:
            raise ValueError("invalid rollout group: " + "; ".join(group_validation.reasons))
        root_rewards: dict[str, float] = {}
        node_rewards: dict[str, dict[str, float]] = {}
        for tree in bundle.rollouts:
            root = tree.nodes[tree.root_node_id]
            if root.credit.reward is None:
                raise ValueError(f"root reward missing for {tree.rollout_id}")
            root_rewards[tree.rollout_id] = root.credit.reward.reward
            node_rewards[tree.rollout_id] = {}
            for node in tree.nodes.values():
                if node.credit.reward is None:
                    raise ValueError(f"node reward missing for {tree.rollout_id}/{node.node_id}")
                node_rewards[tree.rollout_id][node.node_id] = node.credit.reward.reward
        advantages = compute_group_advantages(root_rewards, node_rewards)
        for tree in bundle.rollouts:
            result = advantages[tree.rollout_id]
            for node in tree.nodes.values():
                node.credit.loo_baseline = result.baseline
                node.credit.advantage = result.node_advantages[node.node_id]
                eligibility = assess_node_trainability(
                    node,
                    config.rollout.max_staleness_batches,
                    require_optimizer_credit=False,
                    allow_step_limit_failure=config.reward.train_failed_trajectories,
                )
                node.is_trainable = eligibility.valid
                node.trainability_reasons = eligibility.reasons
                if eligibility.valid:
                    candidate_nodes.append(node)
    depth_result = compute_depth_weights(node.depth for node in candidate_nodes)
    samples: list[OptimizerSample] = []
    for node in candidate_nodes:
        node.credit.depth_count = depth_result.depth_counts[node.depth]
        node.credit.depth_alpha = depth_result.alpha
        node.credit.depth_weight = (
            depth_result.depth_weights[node.depth]
            if config.weighting.type == "depth_inverse_frequency"
            else 1.0
        )
        final_eligibility = assess_node_trainability(
            node,
            config.rollout.max_staleness_batches,
            allow_step_limit_failure=config.reward.train_failed_trajectories,
        )
        if not final_eligibility.valid:
            raise ValueError(f"node became invalid after credit assignment: {final_eligibility.reasons}")
        turns = node.turns or [
            PolicyTurnRecord(
                turn_index=0,
                model_input=node.model_input,
                tokens=node.tokens,
            )
        ]
        for turn in turns:
            samples.append(
                OptimizerSample(
                    task_id=node.task_id,
                    group_id=node.group_id,
                    rollout_id=node.rollout_id,
                    node_id=node.node_id,
                    turn_index=turn.turn_index,
                    depth=node.depth,
                    input_ids=turn.tokens.input_ids,
                    generated_ids=turn.tokens.generated_ids,
                    action_mask=turn.tokens.action_mask,
                    behavior_logprobs=turn.tokens.behavior_logprobs,
                    advantage=node.credit.advantage or 0.0,
                    depth_weight=node.credit.depth_weight,
                    behavior_policy_version=node.behavior_policy_version,
                    staleness=node.staleness,
                    sampling_temperature=turn.sampling_temperature,
                )
            )
    counts = Counter(node.depth for node in candidate_nodes)
    manifest = OptimizerBatchManifest(
        optimizer_step=optimizer_step,
        policy_version_before=policy_version_before,
        root_task_count=len(bundles),
        group_size=bundles[0].group.expected_group_size,
        tree_count=sum(len(bundle.rollouts) for bundle in bundles),
        trainable_node_count=len(candidate_nodes),
        trainable_turn_count=len(samples),
        depth_counts=dict(sorted(counts.items())),
        depth_weights={
            depth: depth_result.depth_weights[depth] if config.weighting.type == "depth_inverse_frequency" else 1.0
            for depth in counts
        },
        max_staleness_batches=config.rollout.max_staleness_batches,
        config_hash=config.stable_hash(),
        source_rollout_ids=[tree.rollout_id for bundle in bundles for tree in bundle.rollouts],
    )
    return CompiledBatch(manifest=manifest, samples=samples)
