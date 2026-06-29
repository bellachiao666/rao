"""Training components for Recursive Agent Optimization."""

from recursive_agent_training.advantages import compute_group_advantages
from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.rewards import compute_node_reward
from recursive_agent_training.weighting import compute_depth_weights

__all__ = [
    "TrainingConfig",
    "compute_depth_weights",
    "compute_group_advantages",
    "compute_node_reward",
]
