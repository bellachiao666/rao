"""Exact and normalized answer verification."""

from __future__ import annotations

import re

from recursive_agent_training.schemas import RootTaskRecord, RolloutTreeRecord, SuccessSignalResult, TrainingNodeRecord, VerifierStatus
from recursive_agent_training.verifiers.base import SuccessSignalProvider


def normalize_answer(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


class ExactMatchVerifier(SuccessSignalProvider):
    def __init__(self, answer_by_node_task: dict[str, str] | None = None):
        self.answer_by_node_task = answer_by_node_task or {}

    async def evaluate(
        self,
        node: TrainingNodeRecord,
        tree: RolloutTreeRecord,
        root_task: RootTaskRecord,
    ) -> SuccessSignalResult:
        expected = self.answer_by_node_task.get(node.node_task, root_task.ground_truth or "")
        success = (
            not node.is_fallback
            and bool(expected)
            and normalize_answer(node.final_answer) == normalize_answer(expected)
        )
        return SuccessSignalResult(
            value=1.0 if success else 0.0,
            provider="normalized_exact_match",
            provider_version="1.0.0",
            reason=(
                "fallback trajectories cannot succeed"
                if node.is_fallback
                else "normalized answers match"
                if success
                else "normalized answers differ"
            ),
            raw_output=None,
            status=VerifierStatus.COMPLETE,
            attempt_count=1,
        )
