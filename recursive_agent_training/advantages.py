"""RAO Equation 3 root-group leave-one-out advantages."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutAdvantage:
    rollout_id: str
    baseline: float
    node_advantages: dict[str, float]


def compute_group_advantages(
    root_rewards: dict[str, float],
    node_rewards_by_rollout: dict[str, dict[str, float]],
) -> dict[str, RolloutAdvantage]:
    if len(root_rewards) < 2:
        raise ValueError("at least two rollouts are required for leave-one-out advantages")
    if set(root_rewards) != set(node_rewards_by_rollout):
        raise ValueError("root reward and node reward rollout ids must match")
    if any(not math.isfinite(value) for value in root_rewards.values()):
        raise ValueError("root rewards must be finite")
    total = sum(root_rewards.values())
    denominator = len(root_rewards) - 1
    results: dict[str, RolloutAdvantage] = {}
    for rollout_id, root_reward in root_rewards.items():
        baseline = (total - root_reward) / denominator
        rewards = node_rewards_by_rollout[rollout_id]
        if any(not math.isfinite(value) for value in rewards.values()):
            raise ValueError("node rewards must be finite")
        results[rollout_id] = RolloutAdvantage(
            rollout_id=rollout_id,
            baseline=baseline,
            node_advantages={node_id: reward - baseline for node_id, reward in rewards.items()},
        )
    return results
