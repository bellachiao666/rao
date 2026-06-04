"""Serializable state and result models for recursive agent nodes."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    STEP_LIMITED = "step_limited"
    DEPTH_LIMITED = "depth_limited"
    NODE_LIMITED = "node_limited"
    CHILD_LIMITED = "child_limited"


class TrajectoryStep(BaseModel):
    kind: str
    content: str | None = None
    action: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    raw_response: str | None = None
    parse_error: str | None = None
    timestamp: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChildSummary(BaseModel):
    node_id: str
    task: str
    status: NodeStatus
    answer: str | None = None
    error: str | None = None
    evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeResult(BaseModel):
    node_id: str = ""
    task: str = ""
    status: NodeStatus
    answer: str | None = None
    error: str | None = None
    children: list[ChildSummary] = Field(default_factory=list)
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    success_signal: float | None = None
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def failed(cls, node_id: str = "", task: str = "", error: str = "", status: NodeStatus = NodeStatus.FAILED) -> "NodeResult":
        return cls(node_id=node_id, task=task, status=status, error=error, limitations=[error] if error else [])

    @classmethod
    def timeout(cls, node_id: str = "", task: str = "", error: str = "Run timed out.") -> "NodeResult":
        return cls.failed(node_id=node_id, task=task, error=error, status=NodeStatus.TIMEOUT)


class AgentState(BaseModel):
    node_id: str
    parent_id: str | None
    task: str
    depth: int
    max_depth: int
    remaining_steps: int
    remaining_children: int
    parent_context: dict[str, Any] = Field(default_factory=dict)
    expected_output: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    child_summaries: list[ChildSummary] = Field(default_factory=list)
    available_tools: list[dict[str, Any]] = Field(default_factory=list)
    global_tree_summary: dict[str, Any] = Field(default_factory=dict)
    allowed_actions: list[str] = Field(
        default_factory=lambda: [
            "THINK",
            "TOOL_CALL",
            "DIRECT_ANSWER",
            "LAUNCH_SUBAGENT",
            "LAUNCH_SUBAGENTS_PARALLEL",
            "AGGREGATE",
            "FINISH",
        ]
    )
