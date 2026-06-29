from recursive_agent_training.evaluation import (
    bootstrap_mean_interval,
    summarize_grouped_evaluation,
)
import pytest


def test_bootstrap_interval_is_deterministic():
    first = bootstrap_mean_interval([0.0, 1.0, 1.0], seed=7, samples=100)
    second = bootstrap_mean_interval([0.0, 1.0, 1.0], seed=7, samples=100)
    assert first == second


def test_grouped_summary_calculates_success_and_pass_at_8():
    results = [
        {
            "task_id": "a",
            "rollouts": [
                {"success": False, "nodes": 1, "max_depth": 0},
                {"success": True, "nodes": 3, "max_depth": 1},
            ],
        },
        {
            "task_id": "b",
            "rollouts": [
                {"success": False, "nodes": 1, "max_depth": 0},
                {"success": False, "nodes": 1, "max_depth": 0},
            ],
        },
    ]
    summary = summarize_grouped_evaluation(
        results,
        bootstrap_seed=42,
        bootstrap_samples=100,
    )
    assert summary.task_count == 2
    assert summary.rollout_count == 4
    assert summary.success_rate == 0.25
    assert summary.pass_at_8 is None
    assert summary.pass_at_k == 0.5
    assert summary.pass_k == 2


def test_grouped_summary_rejects_missing_judge_signal():
    with pytest.raises(ValueError, match="complete judge signal"):
        summarize_grouped_evaluation(
            [{"task_id": "a", "rollouts": [{"success": None}]}],
            bootstrap_seed=42,
            bootstrap_samples=10,
        )
