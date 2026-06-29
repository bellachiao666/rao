"""Prompt templates and state rendering for LLMPolicy."""

from __future__ import annotations

import json

from recursive_agent_harness.actions import AgentAction
from recursive_agent_harness.state import AgentState


SYSTEM_PROMPT = """You are one node in a recursive agent execution tree.

You are not the whole system. You are responsible only for the task assigned to this node.
You may solve your task directly, call an available tool, launch one subagent, launch multiple independent subagents in parallel, aggregate child results, or finish.

RESEARCH STRATEGY:
- Start broad, then refine based on what you learn.
- Cross-check important claims across multiple sources when possible.
- Use available tools deliberately; do not call tools just to create activity.
- When snippets or short observations are insufficient, use tools that can inspect fuller source content.
- If a webpage observation has truncated=true, treat it as incomplete and disclose the limitation or find a more precise source.
- For questions asking for a precise title or name, copy the exact title or name from a relevant search result.
- If a search result directly answers the task, finish with that exact result instead of continuing to think.
- Keep intermediate notes concise and use structured observations to organize findings.

DELEGATION STRATEGY:
- Break the task into a small number of meaningful subquestions when decomposition is useful.
- Use subagents for coherent subproblems such as source discovery, fact verification, or answering one component of a multi-hop task.
- Tell subagents exactly what to return, including format and evidence expectations when useful.
- Subagents can run in parallel when their tasks are independent.
- Subagents can themselves delegate recursively, subject to depth and budget limits.
- Do not repeatedly forward nearly the same whole goal to child agents.

All subagents use the same policy as you. A subagent receives a fresh context with a narrower delegated task.
The recursive execution tree must be generated dynamically from the current task and state. Do not use a fixed decomposition template.

Do not delegate just to use recursion. Delegate only when a subtask is meaningfully narrower, clearer, and easier to verify than the current task.
If depth is greater than or equal to max_depth, you must not launch subagents.

Subtasks must be narrower than the parent task, explicit about the requested output, supplied with only necessary context, independently useful to the parent, and verifiable or easy to evaluate.
Use parallel subagents only when the subtasks are independent.
Use sequential subagents when later subtasks depend on earlier results.

The parent node is responsible for aggregating child outputs and making the final judgment for its own assigned task.
If a child fails, continue with available information and disclose limitations when finishing.

You must output exactly one action as strict JSON.
Do not output markdown.
Do not output natural-language explanation outside JSON.
Do not output multiple actions.
Do not include comments in JSON.

Choose the smallest useful next action."""


ACTION_SCHEMA_PROMPT = """Return one JSON object matching the AgentAction schema.

Allowed action types:
- THINK
- TOOL_CALL
- DIRECT_ANSWER
- LAUNCH_SUBAGENT
- LAUNCH_SUBAGENTS_PARALLEL
- AGGREGATE
- FINISH

Example FINISH:
{"type":"FINISH","reason":"ready","answer":"final answer","evidence":[],"limitations":[]}

Example LAUNCH_SUBAGENT:
{"type":"LAUNCH_SUBAGENT","reason":"narrower verification needed","subtask":{"goal":"narrow delegated task","context":{},"expected_output":"specific requested output","constraints":{}}}

Example LAUNCH_SUBAGENTS_PARALLEL:
{"type":"LAUNCH_SUBAGENTS_PARALLEL","reason":"independent subtasks","subtasks":[{"goal":"narrow delegated task","context":{},"expected_output":"specific requested output","constraints":{}}]}

Full JSON schema:
{schema}

If depth == 0 and the task contains multiple independent requirements, you should not directly FINISH in the first step.
You should first either:
1. launch parallel subagents for independent subtasks, or
2. explicitly explain why delegation is unnecessary.

For planning/research tasks with multiple constraints, prefer LAUNCH_SUBAGENTS_PARALLEL over DIRECT_ANSWER.
"""


REPAIR_PROMPT = """The previous response could not be parsed as a valid AgentAction JSON object.

Parse error:
{parse_error}

Previous response:
{raw_response}

Return exactly one corrected JSON object matching the AgentAction schema.
Do not add markdown.
Do not explain the correction."""


ROOT_TASK_JUDGE_PROMPT = """We need to judge the performance of a deep research agent on a task.
The task requires searching for information across various sources and synthesizing information together to answer a question.
The agent may use subagents to solve parts of the task. Do not penalize the model for relying on subagents, unless the subtasks delegated to the subagents are not meaningful or useful for the task.

You will be given the ground truth answer to the task and the agent's answer to the task.
When comparing the agent's answer to the ground truth answer, minor formatting differences are acceptable as long as the core information is equivalent.

Please provide a reason and success flag (boolean value) in the following JSON format:
{
  "reason": "Brief reasoning for success flag here.",
  "success": true or false
}

Return strict JSON only. Do not output markdown."""


SUBTASK_JUDGE_PROMPT = """We need to judge the performance of a deep research agent on a sub-task.
The sub-task is part of a larger task that requires searching for information across various sources and synthesizing information together to answer a question.
The agent may use subagents to solve parts of the sub-task. Do not penalize the model for relying on subagents when the delegated subtasks are meaningful and useful.

Mark the sub-task as successful if the agent's final answer is correct and, if the agent delegates, its delegation strategy is useful and efficient.
Reward delegation only when it is genuinely useful: the subtasks should be concrete, non-overlapping, and should help the agent search, verify, or synthesize information more effectively.
Mark the sub-task as unsuccessful for degenerate delegation behavior even if the final answer is correct.
Degenerate delegation behavior includes repeatedly forwarding nearly the same or whole goal to subagents without meaningful decomposition, search, or reading work, or wasteful failed launches caused by trying to delegate past the depth limit.

It is acceptable for the agent to do all the work itself, part of the work itself, or to use subagents heavily if those subagents do distinct useful work.
Do not give credit just because the wording of child tasks changes slightly from the agent's own goal; judge whether the decomposition is actually useful and efficient.

Please provide a reason and success flag (boolean value) in the following JSON format:
{
  "reason": "Brief reasoning for success flag here.",
  "success": true or false
}

Return strict JSON only. Do not output markdown."""


def render_action_schema_prompt() -> str:
    return ACTION_SCHEMA_PROMPT.replace("{schema}", json.dumps(AgentAction.json_schema(), ensure_ascii=False))


def render_state_prompt(state: AgentState) -> str:
    can_launch = state.depth < state.max_depth and state.remaining_children > 0
    return "\n".join(
        [
            "Current node state:",
            "",
            f"node_id: {state.node_id}",
            f"parent_id: {state.parent_id}",
            f"task: {state.task}",
            f"depth: {state.depth}",
            f"max_depth: {state.max_depth}",
            f"remaining_steps: {state.remaining_steps}",
            f"remaining_children: {state.remaining_children}",
            f"can_launch_subagents: {str(can_launch).lower()}",
            "",
            "Expected output:",
            state.expected_output or "",
            "",
            "Constraints:",
            json.dumps(state.constraints, ensure_ascii=False),
            "",
            "Parent context:",
            json.dumps(state.parent_context, ensure_ascii=False),
            "",
            "Available tools:",
            render_available_tools(state),
            "",
            "Allowed actions:",
            json.dumps(state.allowed_actions, ensure_ascii=False),
            "",
            "Recent trajectory for this node:",
            json.dumps([step.model_dump(mode="json") for step in state.trajectory], ensure_ascii=False),
            "",
            "Completed child summaries:",
            render_child_summaries(state),
            "",
            "Global tree summary:",
            json.dumps(state.global_tree_summary, ensure_ascii=False),
            "",
            _aggregation_instruction(state),
            "",
            _direct_search_result_instruction(state),
            "",
            "Return exactly one strict JSON action. Do not output markdown or explanatory text.",
        ]
    )


def render_child_summaries(state: AgentState) -> str:
    return json.dumps([summary.model_dump(mode="json") for summary in state.child_summaries], ensure_ascii=False)


def render_available_tools(state: AgentState) -> str:
    return json.dumps(state.available_tools, ensure_ascii=False)


def render_repair_prompt(raw_response: str, parse_error: str) -> str:
    return REPAIR_PROMPT.format(raw_response=raw_response, parse_error=parse_error)


def _aggregation_instruction(state: AgentState) -> str:
    if state.child_summaries:
        return 'Use the child_summaries to synthesize a final answer. Do not answer generically.'
    return "Use the current task, trajectory, tools, and constraints to choose the next action."


def _direct_search_result_instruction(state: AgentState) -> str:
    normalized_task = state.task.casefold()
    if "title" not in normalized_task and "name" not in normalized_task:
        return ""
    return (
        "DIRECT SEARCH RESULT RULE: If a recent search_web result title directly "
        "answers this exact-title or exact-name task, choose FINISH now. Copy the "
        "matching result title exactly. Do not delegate, search again, or summarize "
        "the task. Use this shape: "
        '{"type":"FINISH","reason":"direct search result answers the task",'
        '"answer":"Exact matching result title","evidence":["result URL"],'
        '"limitations":[]}'
    )
