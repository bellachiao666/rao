"""Optional AReaL backend adapter boundary."""

from __future__ import annotations

from typing import Any

from recursive_agent_training.optimizers.base import PolicyOptimizer
from recursive_agent_training.schemas import CompiledBatch


class AReaLOptimizerAdapter(PolicyOptimizer):
    """Thin adapter that requires AReaL to be installed by the training cluster."""

    def __init__(self, backend: Any):
        self.backend = backend

    @classmethod
    def from_environment(cls, **kwargs: Any) -> "AReaLOptimizerAdapter":
        try:
            import areal  # type: ignore
        except ImportError as exc:
            raise RuntimeError("AReaL is not installed; install the cluster backend before using this adapter") from exc
        backend_factory = getattr(areal, "create_policy_optimizer", None)
        if backend_factory is None:
            raise RuntimeError("installed AReaL version does not expose create_policy_optimizer")
        return cls(backend_factory(**kwargs))

    def current_version(self) -> str:
        return str(self.backend.current_version())

    def step(self, batch: CompiledBatch) -> dict[str, Any]:
        return dict(self.backend.step(batch.model_dump(mode="python")))

    def save_checkpoint(self, path: str) -> str:
        self.backend.save_checkpoint(path)
        return path
