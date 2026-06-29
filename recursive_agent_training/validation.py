"""Graph integrity and training eligibility checks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from recursive_agent_training.schemas import GroupStatus, RolloutGroupBundle, RolloutTreeRecord, TrainingNodeRecord, TreeStatus, VerifierStatus


@dataclass
class ValidationResult:
    valid: bool
    reasons: list[str] = field(default_factory=list)


def validate_tree(tree: RolloutTreeRecord, verify_checksum: bool = True) -> ValidationResult:
    reasons: list[str] = []
    if tree.root_node_id not in tree.nodes:
        reasons.append("root node is missing")
    if tree.tree_status != TreeStatus.COMPLETE:
        reasons.append("tree is not complete")
    versions = {node.behavior_policy_version for node in tree.nodes.values()}
    if versions and versions != {tree.behavior_policy_version}:
        reasons.append("nodes use mixed behavior policy versions")
    for node in tree.nodes.values():
        if node.parent_id is None:
            if node.node_id != tree.root_node_id:
                reasons.append(f"non-root node {node.node_id} has no parent")
        else:
            parent = tree.nodes.get(node.parent_id)
            if parent is None:
                reasons.append(f"parent missing for {node.node_id}")
            else:
                if node.node_id not in parent.children_ids:
                    reasons.append(f"parent edge missing for {node.node_id}")
                if node.depth != parent.depth + 1:
                    reasons.append(f"invalid depth for {node.node_id}")
        for child_id in node.children_ids:
            child = tree.nodes.get(child_id)
            if child is None or child.parent_id != node.node_id:
                reasons.append(f"invalid child edge {node.node_id}->{child_id}")
    if verify_checksum and tree.checksum and tree.content_checksum() != tree.checksum:
        reasons.append("tree checksum mismatch")
    return ValidationResult(not reasons, reasons)


def validate_group(bundle: RolloutGroupBundle) -> ValidationResult:
    reasons: list[str] = []
    group = bundle.group
    if group.status != GroupStatus.COMPLETE:
        reasons.append("group is not complete")
    if len(bundle.rollouts) != group.expected_group_size:
        reasons.append("rollout count does not match expected group size")
    if group.expected_group_size < 2:
        reasons.append("group size must be at least 2")
    indices = [tree.rollout_index for tree in bundle.rollouts]
    if len(indices) != len(set(indices)):
        reasons.append("duplicate rollout index")
    versions = {tree.behavior_policy_version for tree in bundle.rollouts}
    if versions != {group.behavior_policy_version}:
        reasons.append("group contains mixed behavior policy versions")
    checkpoint_paths = {tree.behavior_checkpoint_path for tree in bundle.rollouts}
    if checkpoint_paths != {group.behavior_checkpoint_path}:
        reasons.append("group contains mixed behavior checkpoint paths")
    checkpoint_hashes = {tree.behavior_checkpoint_hash for tree in bundle.rollouts}
    if checkpoint_hashes != {group.behavior_checkpoint_hash}:
        reasons.append("group contains mixed behavior checkpoint hashes")
    for tree in bundle.rollouts:
        if tree.group_id != group.group_id:
            reasons.append(f"tree {tree.rollout_id} belongs to another group")
        tree_result = validate_tree(tree)
        reasons.extend(f"{tree.rollout_id}: {reason}" for reason in tree_result.reasons)
    return ValidationResult(not reasons, reasons)


def assess_node_trainability(
    node: TrainingNodeRecord,
    max_staleness: int,
    require_optimizer_credit: bool = True,
    allow_step_limit_failure: bool = False,
) -> ValidationResult:
    reasons: list[str] = []
    trainable_step_limit = (
        allow_step_limit_failure
        and node.is_fallback
        and node.terminal_status == "step_limited"
        and node.terminal_reason == "step_limit"
    )
    allowed_terminal_statuses = {"completed"}
    if trainable_step_limit:
        allowed_terminal_statuses.add("step_limited")
    if node.terminal_status not in allowed_terminal_statuses:
        reasons.append(f"terminal status {node.terminal_status}")
    if node.is_fallback and not trainable_step_limit:
        reasons.append("fallback trajectory")
    if node.terminal_reason in {"fallback_finish", "invalid_action_limit"}:
        reasons.append(f"terminal reason {node.terminal_reason}")
    turn_tokens = [turn.tokens for turn in node.turns] or [node.tokens]
    for turn_index, tokens in enumerate(turn_tokens):
        prefix = f"turn {turn_index}: " if len(turn_tokens) > 1 else ""
        if not tokens.input_ids:
            reasons.append(prefix + "no prompt tokens")
        if not tokens.generated_ids:
            reasons.append(prefix + "no generated tokens")
        if len(tokens.generated_ids) != len(tokens.action_mask):
            reasons.append(prefix + "action mask is incomplete")
        if len(tokens.generated_ids) != len(tokens.behavior_logprobs):
            reasons.append(prefix + "behavior logprobs are incomplete")
        if tokens.truncated:
            reasons.append(prefix + "trajectory was truncated")
    if node.evaluation is None or node.evaluation.status != VerifierStatus.COMPLETE or node.evaluation.value is None:
        reasons.append("success signal is incomplete")
    if node.credit.reward is None:
        reasons.append("reward is missing")
    if node.staleness > max_staleness:
        reasons.append("trajectory is stale")
    if require_optimizer_credit and (
        node.credit.loo_baseline is None or node.credit.advantage is None or node.credit.depth_weight is None
    ):
        reasons.append("optimizer credit is incomplete")
    values = [node.credit.loo_baseline, node.credit.advantage, node.credit.depth_weight]
    if any(value is not None and not math.isfinite(value) for value in values):
        reasons.append("credit values must be finite")
    return ValidationResult(not reasons, reasons)
