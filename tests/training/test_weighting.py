import pytest

from recursive_agent_training.weighting import compute_depth_weights


def test_depth_weights_preserve_total_and_balance_depths():
    result = compute_depth_weights([0, 0, 1, 1, 1, 1, *([2] * 8)])
    assert result.alpha == pytest.approx(14 / 3)
    totals = {
        depth: result.depth_counts[depth] * result.depth_weights[depth]
        for depth in result.depth_counts
    }
    assert len(set(round(value, 8) for value in totals.values())) == 1
    assert sum(totals.values()) == pytest.approx(14)
