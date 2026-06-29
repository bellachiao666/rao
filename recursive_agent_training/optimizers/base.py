"""Policy optimizer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from recursive_agent_training.schemas import CompiledBatch


class PolicyOptimizer(ABC):
    @abstractmethod
    def current_version(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def step(self, batch: CompiledBatch) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: str) -> str:
        raise NotImplementedError
