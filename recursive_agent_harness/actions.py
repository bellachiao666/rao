"""Structured action schemas produced by policies."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    THINK = "THINK"
    TOOL_CALL = "TOOL_CALL"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    LAUNCH_SUBAGENT = "LAUNCH_SUBAGENT"
    LAUNCH_SUBAGENTS_PARALLEL = "LAUNCH_SUBAGENTS_PARALLEL"
    AGGREGATE = "AGGREGATE"
    FINISH = "FINISH"


class SubtaskSpec(BaseModel):
    goal: str
    context: dict[str, Any] = Field(default_factory=dict)
    expected_output: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)


class ToolCallPayload(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    type: ActionType
    reason: str | None = None
    thought: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    answer: str | None = None
    evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    subtask: SubtaskSpec | None = None
    subtasks: list[SubtaskSpec] | None = None
    instructions: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_payload_for_type(self) -> "AgentAction":
        if self.type == ActionType.FINISH and not self.answer:
            raise ValueError("FINISH requires answer")
        if self.type == ActionType.DIRECT_ANSWER and not self.answer:
            raise ValueError("DIRECT_ANSWER requires answer")
        if self.type == ActionType.LAUNCH_SUBAGENT and self.subtask is None:
            raise ValueError("LAUNCH_SUBAGENT requires subtask")
        if self.type == ActionType.LAUNCH_SUBAGENTS_PARALLEL and not self.subtasks:
            raise ValueError("LAUNCH_SUBAGENTS_PARALLEL requires non-empty subtasks")
        if self.type == ActionType.TOOL_CALL and not self.tool_name:
            raise ValueError("TOOL_CALL requires tool_name")
        if self.type == ActionType.THINK and not self.thought:
            raise ValueError("THINK requires thought")
        if self.type == ActionType.AGGREGATE and not self.instructions:
            raise ValueError("AGGREGATE requires instructions")
        return self

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema()
