"""RAO Equation 1 local reward."""

from __future__ import annotations

import math

from recursive_agent_training.schemas import RewardBreakdown


def compute_node_reward(
    own_success: float,
    child_successes: list[float],
    delegation_lambda: float,
) -> RewardBreakdown:
    values = [own_success, delegation_lambda, *child_successes]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("reward inputs must be finite")
    if not 0 <= own_success <= 1 or any(not 0 <= value <= 1 for value in child_successes):
        raise ValueError("success signals must be in [0, 1]")
    if delegation_lambda < 0:
        raise ValueError("delegation_lambda must be non-negative")
    child_mean = sum(child_successes) / len(child_successes) if child_successes else 0.0
    bonus = delegation_lambda * child_mean
    return RewardBreakdown(
        own_success=own_success,
        child_count=len(child_successes),
        child_success_mean=child_mean,
        delegation_lambda=delegation_lambda,
        delegation_bonus=bonus,
        reward=own_success + bonus,
    )
