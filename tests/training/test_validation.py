import pytest

from recursive_agent_training.schemas import (
    CreditRecord,
    ModelInputRecord,
    RewardBreakdown,
    SuccessSignalResult,
    TokenRecord,
    TrainingNodeRecord,
    VerifierStatus,
)
from recursive_agent_training.validation import assess_node_trainability


def make_node(
    is_fallback=False,
    terminal_reason=None,
    terminal_status="completed",
):
    return TrainingNodeRecord(
        task_id="t",
        group_id="g",
        rollout_id="r",
        node_id="n",
        parent_id=None,
        depth=0,
        node_task="task",
        terminal_status=terminal_status,
        terminal_reason=terminal_reason
        or ("fallback_finish" if is_fallback else "finish"),
        is_fallback=is_fallback,
        model_input=ModelInputRecord(rendered_prompt="p"),
        tokens=TokenRecord(
            input_ids=[1],
            generated_ids=[2],
            action_mask=[1],
            behavior_logprobs=[-1],
        ),
        evaluation=SuccessSignalResult(
            value=0,
            provider="exact",
            status=VerifierStatus.COMPLETE,
        ),
        credit=CreditRecord(
            reward=RewardBreakdown(
                own_success=0,
                child_count=0,
                child_success_mean=0,
                delegation_lambda=0,
                delegation_bonus=0,
                reward=0,
            ),
            loo_baseline=0.5,
            advantage=-0.5,
            depth_count=1,
            depth_alpha=1,
            depth_weight=1,
        ),
    )


def test_real_failure_is_trainable_but_fallback_is_not():
    assert assess_node_trainability(make_node(), 3).valid
    assert not assess_node_trainability(make_node(is_fallback=True), 3).valid


def test_step_limit_failure_is_trainable_only_when_enabled():
    node = make_node(
        is_fallback=True,
        terminal_reason="step_limit",
        terminal_status="step_limited",
    )

    assert not assess_node_trainability(node, 3).valid
    assert assess_node_trainability(
        node,
        3,
        allow_step_limit_failure=True,
    ).valid


def test_invalid_action_limit_remains_untrainable_when_failed_training_enabled():
    node = make_node(is_fallback=True, terminal_reason="invalid_action_limit")

    result = assess_node_trainability(
        node,
        3,
        allow_step_limit_failure=True,
    )

    assert not result.valid
    assert "terminal reason invalid_action_limit" in result.reasons


@pytest.mark.parametrize(
    "terminal_status",
    ["timeout", "failed", "cancelled", "depth_limited", "node_limited", "child_limited"],
)
def test_failed_terminal_statuses_are_not_trainable(terminal_status):
    result = assess_node_trainability(
        make_node(terminal_status=terminal_status),
        3,
        allow_step_limit_failure=True,
    )

    assert not result.valid
    assert f"terminal status {terminal_status}" in result.reasons
