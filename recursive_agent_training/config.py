"""Validated configuration for RAO training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ExperimentConfig(StrictModel):
    name: str = "mini_rao"
    seed: int = 42
    faithful_profile: bool = False
    action_profile: Literal["existing_harness", "codeact"] = "existing_harness"


class DatasetConfig(StrictModel):
    train_path: str = ""
    eval_path: str = ""
    split_manifest_path: str = ""
    eval_count: int = 50


class ModelConfig(StrictModel):
    name: str = "tiny"
    path: str = ""
    source: Literal["local", "modelscope"] = "local"
    model_id: str = ""
    revision: str = ""
    tokenizer_revision: str = ""
    train_context_tokens: int = 4096
    eval_context_tokens: int = 4096
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16


class RolloutConfig(StrictModel):
    root_batch_size: int = 2
    group_size: int = 2
    max_depth: int = 2
    max_steps_per_node: int = 8
    temperature: float = 1.0
    max_tokens: int = 256
    timeout_seconds: float = 300.0
    max_staleness_batches: int = 3
    max_concurrent_groups: int = 2


class RewardConfig(StrictModel):
    delegation_lambda: float = Field(default=0.0, alias="lambda")
    root_provider: str = "exact"
    subtask_provider: str = "exact"
    train_failed_trajectories: bool = True


class AdvantageConfig(StrictModel):
    type: Literal["root_group_leave_one_out"] = "root_group_leave_one_out"


class WeightingConfig(StrictModel):
    type: Literal["depth_inverse_frequency", "uniform"] = "depth_inverse_frequency"


class ObjectiveConfig(StrictModel):
    type: Literal["cispo"] = "cispo"
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    kl_coefficient: float = 0.0


class OptimizerConfig(StrictModel):
    backend: Literal["torch", "areal"] = "torch"
    learning_rate: float = 3e-6
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1


class CheckpointConfig(StrictModel):
    output_dir: str = "output/train/mini_rao"
    save_every_steps: int = 1


class EvaluationConfig(StrictModel):
    heldout_task_count: int = 50
    every_steps: int = 5


class SearchServiceConfig(StrictModel):
    provider: Literal["duckduckgo", "tavily", "disabled"] = "duckduckgo"
    endpoint: str = ""
    api_key_env: str = "TAVILY_API_KEY"
    cache_dir: str = "output/cache/deepdive_search"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    requests_per_second: float = 1.0
    max_concurrency: int = 4
    max_results: int = 5
    max_content_bytes: int = 1_000_000


class JudgeServiceConfig(StrictModel):
    provider: Literal["local_transformers", "openai_compatible", "exact"] = "exact"
    model_id: str = "Qwen/Qwen3-0.6B"
    model_path: str = ""
    revision: str = "master"
    device: str = "cuda:7"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    max_new_tokens: int = 256
    max_retries: int = 2
    base_url_env: str = "RAO_JUDGE_BASE_URL"
    api_key_env: str = "RAO_JUDGE_API_KEY"
    model_env: str = "RAO_JUDGE_MODEL"
    provider_version: str = ""


class ServicesConfig(StrictModel):
    search: SearchServiceConfig = Field(default_factory=SearchServiceConfig)
    judge: JudgeServiceConfig = Field(default_factory=JudgeServiceConfig)


class AsyncConfig(StrictModel):
    enabled: bool = False
    rollout_queue_size: int = 8
    verifier_queue_size: int = 8
    train_queue_size: int = 4


class SyncFlywheelConfig(StrictModel):
    root_tasks_per_round: int = 4
    target_rounds: int = 2
    max_collection_attempts: int = 1
    ratio_min: float = 0.95
    ratio_max: float = 1.05
    nonzero_advantage_epsilon: float = 1e-8


class TrainingConfig(StrictModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    advantage: AdvantageConfig = Field(default_factory=AdvantageConfig)
    weighting: WeightingConfig = Field(default_factory=WeightingConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    async_config: AsyncConfig = Field(default_factory=AsyncConfig, alias="async")
    sync_flywheel: SyncFlywheelConfig = Field(default_factory=SyncFlywheelConfig)
    parameter_sources: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_training(self) -> "TrainingConfig":
        if self.rollout.group_size < 2:
            raise ValueError("group_size must be at least 2 for leave-one-out advantages")
        if (
            self.rollout.root_batch_size <= 0
            or self.rollout.max_steps_per_node <= 0
            or self.rollout.timeout_seconds <= 0
        ):
            raise ValueError("batch and step limits must be positive")
        if self.rollout.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.reward.delegation_lambda < 0:
            raise ValueError("reward lambda must be non-negative")
        if self.objective.epsilon_low < 0 or self.objective.epsilon_high < 0:
            raise ValueError("CISPO clipping bounds must be non-negative")
        if (
            self.sync_flywheel.root_tasks_per_round <= 0
            or self.sync_flywheel.target_rounds <= 0
            or self.sync_flywheel.max_collection_attempts <= 0
        ):
            raise ValueError("sync flywheel task and round counts must be positive")
        if not 0 < self.sync_flywheel.ratio_min <= self.sync_flywheel.ratio_max:
            raise ValueError("sync flywheel ratio bounds are invalid")
        if self.sync_flywheel.nonzero_advantage_epsilon < 0:
            raise ValueError("sync flywheel advantage epsilon must be non-negative")
        if self.model.source == "modelscope" and not self.model.model_id:
            raise ValueError("model.model_id is required for ModelScope models")
        if self.model.train_context_tokens <= self.rollout.max_tokens:
            raise ValueError("train context must exceed rollout max_tokens")
        search = self.services.search
        if search.timeout_seconds <= 0 or search.max_retries < 0:
            raise ValueError("search timeout and retry settings are invalid")
        if search.requests_per_second <= 0 or search.max_concurrency <= 0:
            raise ValueError("search rate and concurrency settings must be positive")
        if search.max_results <= 0 or search.max_content_bytes <= 0:
            raise ValueError("search result and content limits must be positive")
        judge = self.services.judge
        if judge.max_new_tokens <= 0 or judge.max_retries < 0:
            raise ValueError("judge generation and retry settings are invalid")
        if judge.provider == "local_transformers" and not judge.model_path:
            raise ValueError("services.judge.model_path is required for local_transformers")
        if self.experiment.faithful_profile and "deepdive" in self.experiment.name.lower():
            expected = {
                "group_size": (self.rollout.group_size, 8),
                "root_batch_size": (self.rollout.root_batch_size, 16),
                "max_depth": (self.rollout.max_depth, 4),
                "max_steps_per_node": (self.rollout.max_steps_per_node, 25),
                "max_staleness_batches": (self.rollout.max_staleness_batches, 3),
                "train_context_tokens": (self.model.train_context_tokens, 40960),
                "eval_context_tokens": (self.model.eval_context_tokens, 262144),
            }
            mismatches = [f"{key}={actual}, expected {target}" for key, (actual, target) in expected.items() if actual != target]
            if self.reward.delegation_lambda != 0:
                mismatches.append("reward.lambda must be 0")
            if self.optimizer.learning_rate != 3e-6:
                mismatches.append("learning_rate must be 3e-6")
            if mismatches:
                raise ValueError("faithful DeepDive config mismatch: " + "; ".join(mismatches))
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle) or {})

    def stable_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json", by_alias=True), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def stable_hash_ignoring_runtime_retry_fields(self) -> str:
        payload = self.model_dump(mode="json", by_alias=True)
        payload["sync_flywheel"].pop("max_collection_attempts", None)
        payload["rollout"].pop("max_concurrent_groups", None)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def matches_hash_except_runtime_retry_fields(self, candidate_hash: str) -> bool:
        if candidate_hash == self.stable_hash():
            return True
        current_attempts = self.sync_flywheel.max_collection_attempts
        current_concurrency = self.rollout.max_concurrent_groups
        max_attempts = max(256, current_attempts * 2)
        max_concurrency = max(16, current_concurrency * 2)
        current_fingerprint = self.stable_hash_ignoring_runtime_retry_fields()
        for attempts in range(1, max_attempts + 1):
            for concurrency in range(1, max_concurrency + 1):
                if attempts == current_attempts and concurrency == current_concurrency:
                    continue
                clone = self.model_copy(deep=True)
                clone.sync_flywheel.max_collection_attempts = attempts
                clone.rollout.max_concurrent_groups = concurrency
                if (
                    clone.stable_hash() == candidate_hash
                    and clone.stable_hash_ignoring_runtime_retry_fields()
                    == current_fingerprint
                ):
                    return True
        return False

    def to_redacted_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)
