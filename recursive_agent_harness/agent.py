"""Recursive agent node execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from recursive_agent_harness.actions import ActionType, AgentAction, SubtaskSpec
from recursive_agent_harness.config import HarnessConfig
from recursive_agent_harness.errors import NodeLimitError
from recursive_agent_harness.policy import Policy
from recursive_agent_harness.state import AgentState, ChildSummary, NodeResult, NodeStatus, TrajectoryStep
from recursive_agent_harness.tools import ToolObservation, ToolRegistry
from recursive_agent_harness.tree import ExecutionTree


@dataclass
class GlobalBudget:
    total_node_limit: int
    allocated_nodes: int = 0

    def allocate_node_id_or_raise(self) -> str:
        if self.allocated_nodes >= self.total_node_limit:
            raise NodeLimitError("total_node_limit has been reached")
        self.allocated_nodes += 1
        return f"node_{self.allocated_nodes:04d}"

    @property
    def remaining_nodes(self) -> int:
        return max(self.total_node_limit - self.allocated_nodes, 0)


@dataclass
class RecursiveAgent:
    node_id: str
    parent_id: str | None
    task: str
    depth: int
    max_depth: int
    policy: Policy
    tools: ToolRegistry
    tree: ExecutionTree
    budget: GlobalBudget
    config: HarnessConfig
    parent_context: dict[str, Any] = field(default_factory=dict)
    expected_output: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    children: list[NodeResult] = field(default_factory=list)
    final_answer: str | None = None
    draft_answer: str | None = None
    evidence: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    invalid_action_count: int = 0
    tool_call_count: int = 0
    step_count: int = 0
    terminal_reason: str | None = None
    is_fallback: bool = False

    async def run(self) -> NodeResult:
        self.status = NodeStatus.RUNNING
        self.tree.register_node(
            self.node_id,
            self.parent_id,
            self.depth,
            self.task,
            metadata={
                "expected_output": self.expected_output,
                "constraints": self.constraints,
                "success_signal": None,
                "reward": None,
            },
        )

        done = False
        while not done and self.step_count < self.config.max_steps_per_node:
            state = self.build_state()
            try:
                raw_action = await self.policy.act(state)
                action = raw_action if isinstance(raw_action, AgentAction) else AgentAction.model_validate(raw_action)
                if action.type.value not in state.allowed_actions:
                    raise ValueError(f"Action {action.type.value} is not allowed in the current state.")
            except Exception as exc:
                self.invalid_action_count += 1
                self.append_observation({"type": "invalid_action", "error": str(exc)})
                if self.invalid_action_count >= self.config.invalid_action_limit:
                    self.final_answer = self.best_effort_answer()
                    self.terminal_reason = "invalid_action_limit"
                    self.is_fallback = True
                    done = True
                self.step_count += 1
                continue

            self.append_action(action)

            if action.type == ActionType.THINK:
                self.append_observation({"type": "thought", "content": action.thought})
            elif action.type == ActionType.TOOL_CALL:
                observation = await self.execute_tool_call(action)
                self.append_observation(observation.model_dump(mode="json"))
            elif action.type == ActionType.LAUNCH_SUBAGENT:
                result = await self.launch_subagent(action.subtask)  # type: ignore[arg-type]
                self.children.append(result)
                if self.depth == 0 and self.children:
                    self.draft_answer = self.aggregate_child_summaries()
                self.append_observation({"type": "child_result", "result": result.model_dump(mode="json")})
            elif action.type == ActionType.LAUNCH_SUBAGENTS_PARALLEL:
                results = await self.launch_subagents_parallel(action.subtasks or [])
                self.children.extend(results)
                if self.depth == 0 and self.children:
                    self.draft_answer = self.aggregate_child_summaries()
                self.append_observation({"type": "child_results", "results": [result.model_dump(mode="json") for result in results]})
            elif action.type == ActionType.DIRECT_ANSWER:
                self.draft_answer = action.answer
                self.evidence = action.evidence
                self.limitations = action.limitations
                self.append_observation(
                    {
                        "type": "draft_answer",
                        "answer": action.answer,
                        "evidence": action.evidence,
                        "limitations": action.limitations,
                    }
                )
            elif action.type == ActionType.AGGREGATE:
                aggregate_answer = self.aggregate_child_summaries()
                if aggregate_answer:
                    self.draft_answer = aggregate_answer
                self.append_observation(
                    {
                        "type": "aggregate",
                        "instructions": action.instructions,
                        "answer": aggregate_answer,
                        "child_summaries": [summary.model_dump(mode="json") for summary in self.child_summaries()],
                    }
                )
            elif action.type == ActionType.FINISH:
                if self.children and _is_generic_placeholder(action.answer):
                    self.final_answer = self.aggregate_child_summaries()
                else:
                    self.final_answer = action.answer
                    self.evidence = action.evidence
                    self.limitations = action.limitations
                self.is_fallback = bool(action.metadata.get("fallback"))
                self.terminal_reason = "fallback_finish" if self.is_fallback else "finish"
                done = True

            self.step_count += 1

        if not done:
            self.final_answer = self.best_effort_answer()
            self.status = NodeStatus.STEP_LIMITED
            self.terminal_reason = "step_limit"
            self.is_fallback = True
        else:
            self.status = NodeStatus.COMPLETED
            self.terminal_reason = self.terminal_reason or "finish"

        return self.finish(self.final_answer or "")

    def build_state(self) -> AgentState:
        child_summaries = self.child_summaries()
        return AgentState(
            node_id=self.node_id,
            parent_id=self.parent_id,
            task=self.task,
            depth=self.depth,
            max_depth=self.max_depth,
            remaining_steps=max(self.config.max_steps_per_node - self.step_count, 0),
            remaining_children=max(self.config.max_children_per_node - len(self.children), 0),
            parent_context=self.parent_context,
            expected_output=self.expected_output,
            constraints=self.constraints,
            trajectory=self.trajectory[-10:],
            child_summaries=child_summaries,
            available_tools=self.tools.specs(),
            global_tree_summary={**self.tree.summary(), "remaining_nodes": self.budget.remaining_nodes},
            allowed_actions=self.allowed_actions(child_summaries),
        )

    def allowed_actions(self, child_summaries: list[ChildSummary]) -> list[str]:
        if self.draft_answer:
            return ["FINISH"]
        if self.depth == 0 and child_summaries:
            return ["AGGREGATE", "FINISH"]
        return [
            "THINK",
            "TOOL_CALL",
            "DIRECT_ANSWER",
            "LAUNCH_SUBAGENT",
            "LAUNCH_SUBAGENTS_PARALLEL",
            "AGGREGATE",
            "FINISH",
        ]

    async def launch_subagent(self, subtask: SubtaskSpec) -> NodeResult:
        if self.depth >= self.config.max_depth:
            return NodeResult.failed(
                task=subtask.goal,
                error="Cannot launch child because max_depth has been reached.",
                status=NodeStatus.DEPTH_LIMITED,
            )
        if len(self.children) >= self.config.max_children_per_node:
            return NodeResult.failed(
                task=subtask.goal,
                error="Cannot launch child because max_children_per_node has been reached.",
                status=NodeStatus.CHILD_LIMITED,
            )
        try:
            child_node_id = self.budget.allocate_node_id_or_raise()
        except NodeLimitError as exc:
            return NodeResult.failed(task=subtask.goal, error=str(exc), status=NodeStatus.NODE_LIMITED)

        self.tree.attach_child(self.node_id, child_node_id)
        child = RecursiveAgent(
            node_id=child_node_id,
            parent_id=self.node_id,
            task=subtask.goal,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            policy=self.policy,
            tools=self.tools,
            tree=self.tree,
            budget=self.budget,
            config=self.config,
            parent_context=subtask.context,
            expected_output=subtask.expected_output,
            constraints=subtask.constraints,
        )
        try:
            return await child.run()
        except Exception as exc:  # pragma: no cover - defensive boundary
            self.tree.update_node_status(child_node_id, NodeStatus.FAILED, error=str(exc))
            return NodeResult.failed(node_id=child_node_id, task=subtask.goal, error=str(exc))

    async def launch_subagents_parallel(self, subtasks: list[SubtaskSpec]) -> list[NodeResult]:
        remaining_children = max(self.config.max_children_per_node - len(self.children), 0)
        allowed_count = min(self.config.max_parallel_children, remaining_children)
        allowed = subtasks[:allowed_count]
        skipped = subtasks[allowed_count:]
        tasks = [asyncio.create_task(self.launch_subagent(subtask)) for subtask in allowed]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        results: list[NodeResult] = []
        for raw in raw_results:
            if isinstance(raw, Exception):
                results.append(NodeResult.failed(error=str(raw)))
            else:
                results.append(raw)
        for subtask in skipped:
            results.append(
                NodeResult.failed(
                    task=subtask.goal,
                    error="Skipped because max_parallel_children was reached.",
                    status=NodeStatus.CHILD_LIMITED,
                )
            )
        return results

    async def execute_tool_call(self, action: AgentAction) -> ToolObservation:
        if self.tool_call_count >= self.config.max_tool_calls_per_node:
            return ToolObservation(tool_name=action.tool_name or "", ok=False, error="max_tool_calls_per_node has been reached.")
        self.tool_call_count += 1
        return await self.tools.call(action.tool_name or "", action.arguments)

    def append_action(self, action: AgentAction) -> None:
        state_action = action.model_dump(mode="json")
        metadata = dict(state_action.get("metadata") or {})
        for key in (
            "_training_trace",
            "raw_response",
            "original_raw_response",
            "parse_error",
        ):
            metadata.pop(key, None)
        state_action["metadata"] = metadata
        step = TrajectoryStep(kind="action", action=self.config.redact_data(state_action))
        self.trajectory.append(step)
        self.tree.append_action(self.node_id, action)

    def append_observation(self, observation: dict[str, Any]) -> None:
        redacted_observation = self.config.redact_data(observation)
        step = TrajectoryStep(kind="observation", observation=redacted_observation)
        self.trajectory.append(step)
        self.tree.append_observation(self.node_id, redacted_observation)

    def child_summaries(self) -> list[ChildSummary]:
        return [
            ChildSummary(
                node_id=result.node_id,
                task=result.task,
                status=result.status,
                answer=self.config.redact_text(result.answer),
                error=self.config.redact_text(result.error),
                evidence=self.config.redact_data(result.evidence),
                limitations=self.config.redact_data(result.limitations),
                metadata=self.config.redact_data(result.metadata),
            )
            for result in self.children
        ]

    def best_effort_answer(self) -> str:
        if self.children:
            return self.aggregate_child_summaries()
        if self.draft_answer:
            return self.draft_answer
        return f"Best effort answer for task: {self.task}"

    def aggregate_child_summaries(self) -> str:
        summaries = self.child_summaries()
        if not summaries:
            return self.draft_answer or ""
        return fallback_aggregate(summaries, self.task)

    def finish(self, answer: str) -> NodeResult:
        redacted_answer = self.config.redact_text(answer) or ""
        self.final_answer = redacted_answer
        self.tree.update_node_metadata(
            self.node_id,
            {
                "terminal_reason": self.terminal_reason,
                "is_fallback": self.is_fallback,
                "is_trainable": not self.is_fallback,
            },
        )
        self.tree.update_node_status(self.node_id, self.status, final_answer=redacted_answer)
        return NodeResult(
            node_id=self.node_id,
            task=self.config.redact_text(self.task) or "",
            status=self.status,
            answer=redacted_answer,
            children=self.child_summaries(),
            trajectory=self.trajectory,
            evidence=self.config.redact_data(self.evidence),
            limitations=self.config.redact_data(self.limitations if self.status == NodeStatus.COMPLETED else [*self.limitations, self.status.value]),
            success_signal=None,
            reward=None,
            metadata=self.config.redact_data(
                {
                    "depth": self.depth,
                    "terminal_reason": self.terminal_reason,
                    "is_fallback": self.is_fallback,
                    "is_trainable": not self.is_fallback,
                }
            ),
        )


def fallback_aggregate(child_summaries: list[ChildSummary], task: str) -> str:
    successful = [summary for summary in child_summaries if summary.answer]
    failed = [summary for summary in child_summaries if not summary.answer or summary.error]
    lines = [
        "Synthesis",
        "",
        f"Task: {task}",
        "",
        "Final answer from child results:",
    ]

    if successful:
        for index, summary in enumerate(successful, start=1):
            lines.append(f"{index}. {summary.task}: {summary.answer}")
    else:
        lines.append("No child returned a successful answer.")

    evidence_lines = _collect_evidence_lines(successful)
    if evidence_lines:
        lines.extend(["", "Evidence:", *evidence_lines])

    limitation_lines = _collect_limitation_lines(child_summaries)
    if failed:
        for summary in failed:
            if summary.error:
                limitation_lines.append(f"{summary.task}: {summary.error}")
            elif not summary.answer:
                limitation_lines.append(f"{summary.task}: no answer returned")

    lines.append("")
    lines.append("Limitations:")
    if limitation_lines:
        lines.extend(f"- {limitation}" for limitation in limitation_lines)
    else:
        lines.append("- This deterministic fallback is based only on completed child answers; verify time-sensitive or high-stakes facts independently.")

    return "\n".join(lines)


def _collect_evidence_lines(child_summaries: list[ChildSummary]) -> list[str]:
    evidence_lines: list[str] = []
    for summary in child_summaries:
        for evidence in summary.evidence:
            evidence_lines.append(f"- {summary.task}: {evidence}")
    return evidence_lines


def _collect_limitation_lines(child_summaries: list[ChildSummary]) -> list[str]:
    limitation_lines: list[str] = []
    for summary in child_summaries:
        for limitation in summary.limitations:
            limitation_lines.append(f"{summary.task}: {limitation}")
    return limitation_lines


def _is_generic_placeholder(answer: str | None) -> bool:
    if not answer:
        return True
    lowered = answer.lower()
    return "best effort answer for task" in lowered or "generic placeholder" in lowered
