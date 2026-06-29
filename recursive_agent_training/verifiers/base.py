"""Verifier interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from recursive_agent_training.schemas import RootTaskRecord, RolloutTreeRecord, SuccessSignalResult, TrainingNodeRecord


class SuccessSignalProvider(ABC):
    @abstractmethod
    async def evaluate(
        self,
        node: TrainingNodeRecord,
        tree: RolloutTreeRecord,
        root_task: RootTaskRecord,
    ) -> SuccessSignalResult:
        raise NotImplementedError
