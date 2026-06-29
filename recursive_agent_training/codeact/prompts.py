"""Prompt contract for the restricted CodeAct action profile."""

import json

from recursive_agent_harness.state import AgentState


CODEACT_SYSTEM_PROMPT = """You are one node in a recursive research agent.
Think briefly, then return exactly one fenced Python block and no text after it.
The Python REPL persists for this node. Do not import modules or access files.
Call exactly one environment action per block:
- search_web(query, max_results=5)
- view_webpage_content(url)
- launch_subagent(goal, context={}, expected_output=None, constraints={})
- asyncio.gather(launch_subagent(...), launch_subagent(...))
- think(text)
- direct_answer(answer, evidence=[], limitations=[])
- finish(answer, evidence=[], limitations=[])
Use _last_observation to inspect the most recent tool or child observation.
If a webpage observation has truncated=true, treat it as incomplete and disclose the limitation or find a more precise source.
For precise-title or precise-name questions, copy the exact value from a relevant search result.
If a search result directly answers the task, call finish with that exact value.
Do not invent tool results. Use finish only when the assigned task is answered."""


def render_codeact_state(state: AgentState) -> str:
    lines = [
            f"task = {state.task}",
            f"depth = {state.depth}",
            f"max_depth = {state.max_depth}",
            f"remaining_steps = {state.remaining_steps}",
            f"remaining_children = {state.remaining_children}",
            "allowed_actions = " + json.dumps(state.allowed_actions, ensure_ascii=False),
            "available_tools = " + json.dumps(state.available_tools, ensure_ascii=False),
            "recent_trajectory = "
            + json.dumps(
                [item.model_dump(mode="json") for item in state.trajectory],
                ensure_ascii=False,
            ),
            "child_summaries = "
            + json.dumps(
                [item.model_dump(mode="json") for item in state.child_summaries],
                ensure_ascii=False,
            ),
    ]
    if "title" in state.task.casefold() or "name" in state.task.casefold():
        lines.extend(
            [
                (
                    "For an exact-title/name task, search with a concise query. "
                    "If _last_observation contains the exact result title, call "
                    "finish immediately and copy it exactly."
                ),
                'Search example: ```python\nsearch_web("concise distinctive query")\n```',
                (
                    "Finish example: ```python\n"
                    'finish("Exact result title", evidence=["result URL"])\n```'
                ),
            ]
        )
    lines.append(
        "Return exactly one Python action block. Plain Python is also accepted."
    )
    return "\n".join(lines)


def render_codeact_repair(raw: str, error: str) -> str:
    return "\n".join(
        [
            "The previous CodeAct output was invalid.",
            f"Error: {error}",
            "Previous output:",
            raw,
            "Return one corrected Python block that calls exactly one allowed action.",
        ]
    )
