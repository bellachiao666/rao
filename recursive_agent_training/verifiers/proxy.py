"""Root-success proxy used for sparse-reward ablations."""

from __future__ import annotations

from recursive_agent_training.schemas import RootTaskRecord, RolloutTreeRecord, SuccessSignalResult, TrainingNodeRecord, VerifierStatus
from recursive_agent_training.verifiers.base import SuccessSignalProvider


class RootProxyVerifier(SuccessSignalProvider):
    def __init__(self, root_values: dict[str, float]):
        self.root_values = root_values

    async def evaluate(
        self,
        node: TrainingNodeRecord,
        tree: RolloutTreeRecord,
        root_task: RootTaskRecord,
    ) -> SuccessSignalResult:
        value = self.root_values[tree.rollout_id]
        if node.is_fallback:
            value = 0.0
        return SuccessSignalResult(
            value=value,
            provider="root_success_proxy",
            provider_version="1.0.0",
            reason=(
                "fallback trajectories cannot receive proxy reward"
                if node.is_fallback
                else "sparse reward ablation proxy"
            ),
            is_proxy=True,
            status=VerifierStatus.COMPLETE,
            attempt_count=1,
        )


class LazyRootProxyVerifier(SuccessSignalProvider):
    """Evaluate each rollout root once, then propagate that signal to every node."""

    def __init__(self, root_verifier: SuccessSignalProvider):
        self.root_verifier = root_verifier
        self.root_results: dict[str, SuccessSignalResult] = {}

    async def evaluate(
        self,
        node: TrainingNodeRecord,
        tree: RolloutTreeRecord,
        root_task: RootTaskRecord,
    ) -> SuccessSignalResult:
        if tree.rollout_id not in self.root_results:
            root = tree.nodes[tree.root_node_id]
            self.root_results[tree.rollout_id] = await self.root_verifier.evaluate(
                root,
                tree,
                root_task,
            )
        root_result = self.root_results[tree.rollout_id]
        if root_result.status != VerifierStatus.COMPLETE or root_result.value is None:
            return SuccessSignalResult(
                value=None,
                provider="root_success_proxy",
                provider_version=root_result.provider_version,
                reason="root success signal is unavailable",
                error=root_result.error,
                is_proxy=True,
                status=root_result.status,
                attempt_count=root_result.attempt_count,
            )
        if node.is_fallback:
            return SuccessSignalResult(
                value=0.0,
                provider="root_success_proxy",
                provider_version=root_result.provider_version,
                reason="fallback trajectories cannot receive proxy reward",
                raw_output=root_result.raw_output,
                is_proxy=True,
                status=VerifierStatus.COMPLETE,
                attempt_count=root_result.attempt_count,
            )
        return SuccessSignalResult(
            value=root_result.value,
            provider="root_success_proxy",
            provider_version=root_result.provider_version,
            reason="sparse reward ablation proxy from root success",
            raw_output=root_result.raw_output,
            is_proxy=True,
            status=VerifierStatus.COMPLETE,
            attempt_count=root_result.attempt_count,
        )

    def close(self) -> None:
        close = getattr(self.root_verifier, "close", None)
        if callable(close):
            close()

    def metrics(self) -> dict:
        metrics = getattr(self.root_verifier, "metrics", None)
        return metrics() if callable(metrics) else {}
