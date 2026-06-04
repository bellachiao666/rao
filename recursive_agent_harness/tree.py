"""Execution tree and structured trace logging."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from recursive_agent_harness.actions import AgentAction
from recursive_agent_harness.state import NodeStatus, TrajectoryStep, utc_now_iso


TERMINAL_STATUSES = {
    NodeStatus.COMPLETED,
    NodeStatus.FAILED,
    NodeStatus.CANCELLED,
    NodeStatus.TIMEOUT,
    NodeStatus.STEP_LIMITED,
    NodeStatus.DEPTH_LIMITED,
    NodeStatus.NODE_LIMITED,
    NodeStatus.CHILD_LIMITED,
}


class TraceNode(BaseModel):
    node_id: str
    parent_id: str | None
    depth: int
    task: str
    status: NodeStatus = NodeStatus.RUNNING
    children_ids: list[str] = Field(default_factory=list)
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    final_answer: str | None = None
    error: str | None = None
    start_time: str = Field(default_factory=utc_now_iso)
    end_time: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionTree:
    """In-memory execution tree with structured export helpers."""

    def __init__(
        self,
        run_id: str = "run",
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        redactions: dict[str, str] | None = None,
    ):
        self.run_id = run_id
        self._redactions = {secret: replacement for secret, replacement in (redactions or {}).items() if secret}
        self.config = self._redact_data(config or {})
        self.metadata = {"schema_version": "0.1.0", "created_at": utc_now_iso()}
        if metadata:
            self.metadata.update(self._redact_data(metadata))
        self.root_node_id: str | None = None
        self.nodes: dict[str, TraceNode] = {}

    def register_node(
        self,
        node_id: str,
        parent_id: str | None,
        depth: int,
        task: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.status = NodeStatus.RUNNING
            return
        if parent_id is None and self.root_node_id is None:
            self.root_node_id = node_id
        self.nodes[node_id] = TraceNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            task=self._redact_text(task),
            metadata=self._redact_data(metadata or {}),
        )

    def update_node_status(
        self,
        node_id: str,
        status: NodeStatus | str,
        error: str | None = None,
        final_answer: str | None = None,
    ) -> None:
        node = self.nodes[node_id]
        node.status = NodeStatus(status)
        if error is not None:
            node.error = self._redact_text(error)
        if final_answer is not None:
            node.final_answer = self._redact_text(final_answer)
        if node.status in TERMINAL_STATUSES:
            node.end_time = utc_now_iso()

    def append_action(
        self,
        node_id: str,
        action: AgentAction,
        raw_response: str | None = None,
        parse_error: str | None = None,
    ) -> None:
        raw_response = raw_response if raw_response is not None else action.metadata.get("raw_response")
        parse_error = parse_error if parse_error is not None else action.metadata.get("parse_error")
        self.nodes[node_id].trajectory.append(
            TrajectoryStep(
                kind="action",
                action=self._redact_data(action.model_dump(mode="json")),
                raw_response=self._redact_text(raw_response),
                parse_error=self._redact_text(parse_error),
            )
        )

    def append_observation(self, node_id: str, observation: dict[str, Any]) -> None:
        self.nodes[node_id].trajectory.append(TrajectoryStep(kind="observation", observation=self._redact_data(observation)))

    def attach_child(self, parent_id: str, child_id: str) -> None:
        parent = self.nodes.get(parent_id)
        if parent is None:
            return
        if child_id not in parent.children_ids:
            parent.children_ids.append(child_id)

    def mark_unfinished(self, status: NodeStatus, error: str) -> None:
        for node in self.nodes.values():
            if node.status not in TERMINAL_STATUSES:
                self.update_node_status(node.node_id, status, error=error)

    def to_dict(self) -> dict[str, Any]:
        return self._redact_data(
            {
                "run_id": self.run_id,
                "root_node_id": self.root_node_id,
                "config": self.config,
                "nodes": {node_id: node.model_dump(mode="json") for node_id, node in self.nodes.items()},
                "metadata": self.metadata,
            }
        )

    def export_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def pretty_print(self) -> str:
        if not self.root_node_id:
            return ""

        lines: list[str] = []

        def visit(node_id: str, indent: int) -> None:
            node = self.nodes[node_id]
            task = _shorten(node.task)
            lines.append(f'{_indent(indent)}{node.node_id} depth={node.depth} status={node.status.value} task="{task}"')
            for child_id in node.children_ids:
                if child_id in self.nodes:
                    visit(child_id, indent + 1)

        visit(self.root_node_id, 0)
        return "\n".join(lines)

    def export_mermaid(self, path: str | Path | None = None) -> str:
        lines = ["graph TD"]
        for node in self.nodes.values():
            label = f"{node.node_id} d={node.depth} {node.status.value}<br/>{_escape_mermaid(_shorten(node.task))}"
            lines.append(f'  {node.node_id}["{label}"]')
        for node in self.nodes.values():
            for child_id in node.children_ids:
                lines.append(f"  {node.node_id} --> {child_id}")
        graph = "\n".join(lines)
        if path is not None:
            Path(path).write_text(graph, encoding="utf-8")
        return graph

    def summary(self) -> dict[str, Any]:
        depth_counts = Counter(node.depth for node in self.nodes.values())
        status_counts = Counter(node.status.value for node in self.nodes.values())
        return {
            "run_id": self.run_id,
            "root_node_id": self.root_node_id,
            "total_nodes": len(self.nodes),
            "depth_histogram": dict(sorted(depth_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
        }

    def _redact_text(self, text: str | None) -> str | None:
        if text is None:
            return None
        redacted = text
        for secret, replacement in sorted(self._redactions.items(), key=lambda item: len(item[0]), reverse=True):
            redacted = redacted.replace(secret, replacement)
        return redacted

    def _redact_data(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact_text(value)
        if isinstance(value, list):
            return [self._redact_data(item) for item in value]
        if isinstance(value, tuple):
            return [self._redact_data(item) for item in value]
        if isinstance(value, dict):
            return {key: self._redact_data(item) for key, item in value.items()}
        return value


def _indent(level: int) -> str:
    return "  " * level


def _shorten(text: str, limit: int = 80) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _escape_mermaid(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")
