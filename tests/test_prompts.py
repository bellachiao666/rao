from recursive_agent_harness.state import AgentState, ChildSummary, NodeStatus
from recursive_agent_harness.prompts import (
    ACTION_SCHEMA_PROMPT,
    ROOT_TASK_JUDGE_PROMPT,
    SUBTASK_JUDGE_PROMPT,
    SYSTEM_PROMPT,
    render_state_prompt,
)


def test_state_prompt_disables_delegation_at_depth_limit():
    state = AgentState(
        node_id="node_0001",
        parent_id=None,
        task="Root task",
        depth=2,
        max_depth=2,
        remaining_steps=3,
        remaining_children=1,
        parent_context={},
        expected_output="answer",
        constraints={},
        trajectory=[],
        child_summaries=[],
        available_tools=[],
        global_tree_summary={"total_nodes": 1},
    )

    rendered = render_state_prompt(state)

    assert "can_launch_subagents: false" in rendered
    assert "Return exactly one strict JSON action" in rendered


def test_state_prompt_puts_direct_title_result_rule_near_the_action_request():
    state = AgentState(
        node_id="node_0001",
        parent_id=None,
        task="Identify the exact paper title.",
        depth=0,
        max_depth=2,
        remaining_steps=3,
        remaining_children=1,
        trajectory=[],
        child_summaries=[],
        available_tools=[{"name": "search_web"}],
        global_tree_summary={"total_nodes": 1},
    )

    rendered = render_state_prompt(state)

    assert "DIRECT SEARCH RESULT RULE" in rendered
    assert '"type":"FINISH"' in rendered
    assert rendered.index("DIRECT SEARCH RESULT RULE") < rendered.index(
        "Return exactly one strict JSON action"
    )


def test_system_prompt_and_action_schema_include_recursive_constraints():
    assert "recursive agent execution tree" in SYSTEM_PROMPT
    assert "Do not use a fixed decomposition template" in SYSTEM_PROMPT
    assert "Do not output markdown" in SYSTEM_PROMPT
    assert "Start broad, then refine" in SYSTEM_PROMPT
    assert "Cross-check important claims across multiple sources" in SYSTEM_PROMPT
    assert "Tell subagents exactly what to return" in SYSTEM_PROMPT
    assert "Subagents can run in parallel" in SYSTEM_PROMPT
    assert "LAUNCH_SUBAGENTS_PARALLEL" in ACTION_SCHEMA_PROMPT


def test_judge_prompts_follow_deepdive_success_schema():
    assert "judge the performance of a deep research agent on a task" in ROOT_TASK_JUDGE_PROMPT
    assert "ground truth answer" in ROOT_TASK_JUDGE_PROMPT
    assert '"success": true or false' in ROOT_TASK_JUDGE_PROMPT
    assert "judge the performance of a deep research agent on a sub-task" in SUBTASK_JUDGE_PROMPT
    assert "degenerate delegation behavior" in SUBTASK_JUDGE_PROMPT
    assert "concrete, non-overlapping" in SUBTASK_JUDGE_PROMPT


def test_state_prompt_requires_aggregation_when_child_summaries_exist():
    state = AgentState(
        node_id="node_0001",
        parent_id=None,
        task="Review a backend release",
        depth=0,
        max_depth=2,
        remaining_steps=3,
        remaining_children=2,
        parent_context={},
        expected_output="answer",
        constraints={},
        trajectory=[],
        child_summaries=[
            ChildSummary(
                node_id="node_0002",
                task="Audit authentication middleware",
                status=NodeStatus.COMPLETED,
                answer="Expired tokens are rejected.",
                evidence=["auth.py validates exp before accepting the request"],
                limitations=["Did not run against production data"],
            )
        ],
        available_tools=[],
        global_tree_summary={"total_nodes": 2},
        allowed_actions=["AGGREGATE", "FINISH"],
    )

    rendered = render_state_prompt(state)

    assert '"AGGREGATE", "FINISH"' in rendered
    assert "Audit authentication middleware" in rendered
    assert "Expired tokens are rejected." in rendered
    assert "auth.py validates exp before accepting the request" in rendered
    assert "Did not run against production data" in rendered
    assert "Use the child_summaries to synthesize a final answer. Do not answer generically." in rendered
