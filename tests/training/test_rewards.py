import pytest

from recursive_agent_training.rewards import compute_node_reward


def test_leaf_reward_has_no_delegation_bonus():
    result = compute_node_reward(0.5, [], 0.4)
    assert result.reward == 0.5
    assert result.delegation_bonus == 0


def test_parent_reward_uses_direct_child_mean():
    result = compute_node_reward(1.0, [0.0, 1.0], 0.4)
    assert result.child_success_mean == 0.5
    assert result.reward == pytest.approx(1.2)


def test_invalid_success_is_rejected():
    with pytest.raises(ValueError):
        compute_node_reward(2.0, [], 0)
