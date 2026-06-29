"""Versioned data contracts for recursive rollouts and optimizer batches."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class GroupStatus(str, Enum):
    PENDING = "pending"
    COLLECTING = "collecting"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TreeStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    REJECTED = "rejected"


class VerifierStatus(str, Enum):
    PENDING = "pending"
    COMPLETE = "complete"
    ERROR = "error"


class RootTaskRecord(SchemaModel):
    schema_version: str = SCHEMA_VERSION
    task_id: str
    domain: str
    split: str
    prompt: str
    ground_truth: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    dataset_hash: str = ""


class ModelInputRecord(SchemaModel):
    messages: list[dict[str, str]] = Field(default_factory=list)
    rendered_prompt: str = ""
    template_version: str = ""
    tokenizer_name: str = ""
    tokenizer_revision: str = ""


class TokenRecord(SchemaModel):
    input_ids: list[int] = Field(default_factory=list)
    generated_ids: list[int] = Field(default_factory=list)
    action_mask: list[int] = Field(default_factory=list)
    behavior_logprobs: list[float] = Field(default_factory=list)
    generated_text: str = ""
    truncated: bool = False

    @model_validator(mode="after")
    def _aligned(self) -> "TokenRecord":
        size = len(self.generated_ids)
        if self.action_mask and len(self.action_mask) != size:
            raise ValueError("action_mask length must match generated_ids")
        if self.behavior_logprobs and len(self.behavior_logprobs) != size:
            raise ValueError("behavior_logprobs length must match generated_ids")
        return self


class PolicyTurnRecord(SchemaModel):
    turn_index: int
    model_input: ModelInputRecord
    tokens: TokenRecord
    sampling_temperature: float = 1.0
    sampling_seed: int | None = None


class SuccessSignalResult(SchemaModel):
    value: float | None = None
    provider: str
    provider_version: str = ""
    reason: str = ""
    raw_output: str | None = None
    evaluated_at: str = Field(default_factory=utc_now_iso)
    error: str | None = None
    is_proxy: bool = False
    status: VerifierStatus = VerifierStatus.PENDING
    attempt_count: int = 0

    @model_validator(mode="after")
    def _range(self) -> "SuccessSignalResult":
        if self.value is not None and not 0 <= self.value <= 1:
            raise ValueError("success signal must be in [0, 1]")
        return self


class RewardBreakdown(SchemaModel):
    own_success: float
    child_count: int
    child_success_mean: float
    delegation_lambda: float
    delegation_bonus: float
    reward: float
    formula_version: str = "rao_eq1_v1"


class CreditRecord(SchemaModel):
    reward: RewardBreakdown | None = None
    loo_baseline: float | None = None
    advantage: float | None = None
    depth_count: int | None = None
    depth_alpha: float | None = None
    depth_weight: float | None = None


class TrainingNodeRecord(SchemaModel):
    schema_version: str = SCHEMA_VERSION
    task_id: str
    group_id: str
    rollout_id: str
    node_id: str
    parent_id: str | None
    depth: int
    node_task: str
    children_ids: list[str] = Field(default_factory=list)
    terminal_status: str
    terminal_reason: str
    final_answer: str = ""
    is_fallback: bool = False
    is_trainable: bool = False
    trainability_reasons: list[str] = Field(default_factory=list)
    model_input: ModelInputRecord = Field(default_factory=ModelInputRecord)
    tokens: TokenRecord = Field(default_factory=TokenRecord)
    turns: list[PolicyTurnRecord] = Field(default_factory=list)
    evaluation: SuccessSignalResult | None = None
    credit: CreditRecord = Field(default_factory=CreditRecord)
    behavior_policy_version: str = ""
    staleness: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RolloutTreeRecord(SchemaModel):
    schema_version: str = SCHEMA_VERSION
    rollout_id: str
    group_id: str
    rollout_index: int
    seed: int
    root_node_id: str
    behavior_policy_version: str
    behavior_checkpoint_path: str = ""
    behavior_checkpoint_hash: str = ""
    prompt_template_version: str = ""
    tool_schema_version: str = ""
    environment_version: str = ""
    nodes: dict[str, TrainingNodeRecord] = Field(default_factory=dict)
    tree_status: TreeStatus = TreeStatus.INCOMPLETE
    checksum: str = ""
    created_at: str = Field(default_factory=utc_now_iso)

    def content_checksum(self) -> str:
        data = self.model_dump(mode="json", exclude={"checksum"})
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def finalize(self) -> "RolloutTreeRecord":
        self.checksum = self.content_checksum()
        return self


class RolloutGroupRecord(SchemaModel):
    schema_version: str = SCHEMA_VERSION
    group_id: str
    task_id: str
    expected_group_size: int
    completed_group_size: int = 0
    behavior_policy_version: str
    behavior_checkpoint_path: str = ""
    behavior_checkpoint_hash: str = ""
    sampling_config_hash: str = ""
    rollout_ids: list[str] = Field(default_factory=list)
    status: GroupStatus = GroupStatus.PENDING
    created_at: str = Field(default_factory=utc_now_iso)


class RolloutGroupBundle(SchemaModel):
    group: RolloutGroupRecord
    task: RootTaskRecord
    rollouts: list[RolloutTreeRecord] = Field(default_factory=list)


class OptimizerSample(SchemaModel):
    task_id: str
    group_id: str
    rollout_id: str
    node_id: str
    turn_index: int = 0
    depth: int
    input_ids: list[int]
    generated_ids: list[int]
    action_mask: list[int]
    behavior_logprobs: list[float]
    advantage: float
    depth_weight: float
    behavior_policy_version: str
    staleness: int = 0
    sampling_temperature: float = 1.0


class OptimizerBatchManifest(SchemaModel):
    schema_version: str = SCHEMA_VERSION
    optimizer_step: int
    policy_version_before: str
    policy_version_after: str = ""
    root_task_count: int
    group_size: int
    tree_count: int
    trainable_node_count: int
    trainable_turn_count: int = 0
    depth_counts: dict[int, int] = Field(default_factory=dict)
    depth_weights: dict[int, float] = Field(default_factory=dict)
    max_staleness_batches: int
    objective: str = "rao_cispo"
    config_hash: str
    source_rollout_ids: list[str] = Field(default_factory=list)


class CompiledBatch(SchemaModel):
    manifest: OptimizerBatchManifest
    samples: list[OptimizerSample]


def write_json(path: str | Path, value: SchemaModel | dict[str, Any]) -> None:
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_group_bundle(path: str | Path) -> RolloutGroupBundle:
    return RolloutGroupBundle.model_validate_json(Path(path).read_text(encoding="utf-8"))
