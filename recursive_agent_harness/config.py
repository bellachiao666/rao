"""Configuration model for recursive agent execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


DEFAULT_CONFIG_PATH = Path("config.yaml")


class HarnessConfig(BaseModel):
    """Runtime and model configuration for the harness."""

    model_config = ConfigDict(protected_namespaces=())

    max_depth: int
    max_steps_per_node: int
    max_children_per_node: int
    total_node_limit: int
    timeout_seconds: float
    invalid_action_limit: int
    max_tool_calls_per_node: int
    max_parallel_children: int

    model_name: str
    base_url: str
    api_key: str | None
    temperature: float
    max_tokens: int

    trace_output_path: str
    mermaid_output_path: str
    log_level: str

    @field_validator(
        "max_steps_per_node",
        "max_children_per_node",
        "total_node_limit",
        "timeout_seconds",
        "invalid_action_limit",
        "max_tool_calls_per_node",
        "max_parallel_children",
        "max_tokens",
    )
    @classmethod
    def _positive(cls, value: int | float) -> int | float:
        if value <= 0:
            raise ValueError("limit values must be positive")
        return value

    @field_validator("max_depth")
    @classmethod
    def _non_negative_depth(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_depth must be non-negative")
        return value

    @field_validator("temperature")
    @classmethod
    def _valid_temperature(cls, value: float) -> float:
        if value < 0:
            raise ValueError("temperature must be non-negative")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "HarnessConfig":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("YAML support requires PyYAML") from exc

        with Path(path).open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls(**data)

    def redaction_map(self) -> dict[str, str]:
        redactions: dict[str, str] = {}
        if self.base_url:
            redactions[self.base_url] = "[REDACTED_BASE_URL]"
        if self.api_key:
            redactions[self.api_key] = "[REDACTED_API_KEY]"
        return redactions

    def redact_text(self, text: str | None) -> str | None:
        if text is None:
            return None
        redacted = text
        for secret, replacement in sorted(self.redaction_map().items(), key=lambda item: len(item[0]), reverse=True):
            redacted = redacted.replace(secret, replacement)
        return redacted

    def redact_data(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, list):
            return [self.redact_data(item) for item in value]
        if isinstance(value, tuple):
            return [self.redact_data(item) for item in value]
        if isinstance(value, dict):
            return {key: self.redact_data(item) for key, item in value.items()}
        return value

    def redacted_model_dump(self) -> dict[str, Any]:
        data = self.redact_data(self.model_dump(mode="json"))
        data["base_url"] = "[REDACTED_BASE_URL]" if self.base_url else ""
        data["api_key"] = "[REDACTED_API_KEY]" if self.api_key else None
        return data
