import pytest

from recursive_agent_training.advantages import compute_group_advantages


def test_leave_one_out_root_baseline_applies_to_all_tree_nodes():
    root = {"r0": 1.0, "r1": 0.0, "r2": 0.5, "r3": 1.0}
    nodes = {
        "r0": {"root": 1.0, "child": 0.5},
        "r1": {"root": 0.0},
        "r2": {"root": 0.5},
        "r3": {"root": 1.0},
    }
    result = compute_group_advantages(root, nodes)
    assert result["r0"].baseline == pytest.approx(0.5)
    assert result["r0"].node_advantages["child"] == pytest.approx(0.0)
    assert result["r1"].baseline == pytest.approx(2.5 / 3)


def test_group_size_one_is_rejected():
    with pytest.raises(ValueError):
        compute_group_advantages({"r": 1.0}, {"r": {"root": 1.0}})
