import pytest

from recursive_agent_harness.actions import ActionType, AgentAction, SubtaskSpec
from recursive_agent_harness.agent import fallback_aggregate
from recursive_agent_harness.policy import Policy
from recursive_agent_harness.runner import Runner
from recursive_agent_harness.state import AgentState, ChildSummary, NodeStatus
from tests.helpers import make_config


AUDIT_TASK = (
    "Review a backend release for correctness risks. Cover authentication, "
    "migration safety, caching behavior, and observability gaps."
)


class GenericRootAfterChildrenPolicy(Policy):
    async def act(self, state: AgentState) -> AgentAction:
        if state.depth == 0 and not state.child_summaries:
            return AgentAction(
                type=ActionType.LAUNCH_SUBAGENTS_PARALLEL,
                subtasks=[
                    SubtaskSpec(goal="Audit authentication middleware", expected_output="answer"),
                    SubtaskSpec(goal="Audit database migration safety", expected_output="answer"),
                    SubtaskSpec(goal="Audit cache invalidation behavior", expected_output="answer"),
                    SubtaskSpec(goal="Audit observability coverage", expected_output="answer"),
                ],
            )
        if state.depth == 0:
            assert state.allowed_actions == ["AGGREGATE", "FINISH"]
            assert len(state.child_summaries) == 4
            assert state.child_summaries[0].task == "Audit authentication middleware"
            assert state.child_summaries[0].answer
            return AgentAction(type=ActionType.FINISH, answer="Best effort answer for task: generic placeholder")
        if "authentication" in state.task.lower():
            return AgentAction(
                type=ActionType.FINISH,
                answer="The middleware rejects expired tokens before the request reaches handlers.",
                evidence=["auth.py validates exp before accepting the request"],
            )
        if "migration" in state.task.lower():
            return AgentAction(
                type=ActionType.FINISH,
                answer="The migration creates the replacement index before dropping the legacy index.",
                limitations=["Did not run against production data"],
            )
        if "cache" in state.task.lower():
            return AgentAction(type=ActionType.FINISH, answer="Cache invalidation covers writes to account and billing records.")
        if "observability" in state.task.lower():
            return AgentAction(type=ActionType.FINISH, answer="The release emits metrics for login failures and migration duration.")
        return AgentAction(type=ActionType.FINISH, answer="leaf answer")


def test_fallback_aggregate_not_generic_when_children_exist():
    answer = fallback_aggregate(
        [
            ChildSummary(node_id="node_0002", task="Audit authentication middleware", status=NodeStatus.COMPLETED, answer="Expired tokens are rejected."),
            ChildSummary(node_id="node_0003", task="Audit database migration safety", status=NodeStatus.COMPLETED, answer="The replacement index is created first."),
            ChildSummary(node_id="node_0004", task="Audit cache invalidation behavior", status=NodeStatus.COMPLETED, answer="Account writes invalidate account cache keys."),
            ChildSummary(node_id="node_0005", task="Audit observability coverage", status=NodeStatus.COMPLETED, answer="Login failures emit a counter metric."),
        ],
        AUDIT_TASK,
    )

    assert "Best effort answer for task" not in answer
    assert "Synthesis" in answer
    assert AUDIT_TASK in answer
    assert "Final answer from child results:" in answer
    assert "1. Audit authentication middleware: Expired tokens are rejected." in answer
    assert "2. Audit database migration safety: The replacement index is created first." in answer
    assert "3. Audit cache invalidation behavior: Account writes invalidate account cache keys." in answer
    assert "4. Audit observability coverage: Login failures emit a counter metric." in answer
    assert "Limitations:" in answer


def test_fallback_aggregate_does_not_reuse_first_verbose_child_for_all_slots():
    verbose_auth_answer = (
        "Authentication review covers token expiry, cache interactions, migration ordering, "
        "and observability notes, but it is still only the authentication child answer."
    )
    answer = fallback_aggregate(
        [
            ChildSummary(
                node_id="node_0002",
                task="Audit authentication middleware",
                status=NodeStatus.COMPLETED,
                answer=verbose_auth_answer,
            ),
            ChildSummary(
                node_id="node_0003",
                task="Audit database migration safety",
                status=NodeStatus.COMPLETED,
                answer="The migration creates the replacement index before dropping the old one.",
            ),
            ChildSummary(
                node_id="node_0004",
                task="Audit cache invalidation behavior",
                status=NodeStatus.COMPLETED,
                answer="Billing writes invalidate account and billing cache keys.",
            ),
            ChildSummary(
                node_id="node_0005",
                task="Audit observability coverage",
                status=NodeStatus.COMPLETED,
                answer="Migration duration and login failures are emitted as metrics.",
            ),
        ],
        AUDIT_TASK,
    )

    assert answer.count(verbose_auth_answer) == 1
    assert f"1. Audit authentication middleware: {verbose_auth_answer}" in answer
    assert "2. Audit database migration safety: The migration creates the replacement index before dropping the old one." in answer
    assert "3. Audit cache invalidation behavior: Billing writes invalidate account and billing cache keys." in answer
    assert "4. Audit observability coverage: Migration duration and login failures are emitted as metrics." in answer


def test_fallback_aggregate_is_domain_agnostic_for_non_travel_task():
    answer = fallback_aggregate(
        [
            ChildSummary(
                node_id="node_0002",
                task="Audit authentication middleware",
                status=NodeStatus.COMPLETED,
                answer="The middleware accepts expired tokens when clock skew is negative.",
                evidence=["auth.py validates exp after skew adjustment"],
            ),
            ChildSummary(
                node_id="node_0003",
                task="Audit database migration safety",
                status=NodeStatus.COMPLETED,
                answer="The migration drops the legacy index before creating the replacement index.",
                limitations=["Did not run against production data"],
            ),
        ],
        "Review the backend release for correctness risks.",
    )

    assert "Synthesis" in answer
    assert "Final answer from child results:" in answer
    assert "Review the backend release for correctness risks." in answer
    assert "Audit authentication middleware" in answer
    assert "The middleware accepts expired tokens" in answer
    assert "Audit database migration safety" in answer
    assert "drops the legacy index" in answer
    assert "Did not run against production data" in answer


@pytest.mark.asyncio
async def test_root_final_answer_uses_child_results(tmp_path):
    config = make_config(trace_output_path=str(tmp_path / "trace.json"), max_steps_per_node=4)
    runner = Runner(policy=GenericRootAfterChildrenPolicy(), config=config)

    result, _ = await runner.arun(AUDIT_TASK)

    assert result.status == NodeStatus.COMPLETED
    assert "Best effort answer for task" not in (result.answer or "")
    assert "Final answer from child results:" in (result.answer or "")
    assert "Audit authentication middleware" in (result.answer or "")
    assert "The middleware rejects expired tokens before the request reaches handlers." in (result.answer or "")
    assert "Audit database migration safety" in (result.answer or "")
    assert "The migration creates the replacement index before dropping the legacy index." in (result.answer or "")
    assert "auth.py validates exp before accepting the request" in (result.answer or "")
    assert "Did not run against production data" in (result.answer or "")


@pytest.mark.asyncio
async def test_completed_root_has_non_empty_final_answer(tmp_path):
    config = make_config(trace_output_path=str(tmp_path / "trace.json"), max_steps_per_node=4)
    runner = Runner(policy=GenericRootAfterChildrenPolicy(), config=config)

    result, _ = await runner.arun(AUDIT_TASK)

    assert result.status == NodeStatus.COMPLETED
    assert result.answer
    assert "Best effort answer for task" not in result.answer
    assert "Synthesis" in result.answer
    assert AUDIT_TASK in result.answer
