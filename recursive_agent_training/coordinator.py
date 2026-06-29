"""Async queue coordinator with policy-staleness enforcement."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from recursive_agent_training.schemas import CompiledBatch, RolloutGroupBundle


@dataclass
class CoordinatorStats:
    accepted_groups: int = 0
    rejected_stale_groups: int = 0
    optimized_batches: int = 0


class AsyncTrainingCoordinator:
    def __init__(
        self,
        max_staleness_batches: int,
        current_policy_version: Callable[[], int],
        compile_fn: Callable[[list[RolloutGroupBundle]], Awaitable[CompiledBatch]],
        optimize_fn: Callable[[CompiledBatch], Awaitable[dict]],
        queue_size: int = 8,
    ):
        self.max_staleness_batches = max_staleness_batches
        self.current_policy_version = current_policy_version
        self.compile_fn = compile_fn
        self.optimize_fn = optimize_fn
        self.rollout_queue: asyncio.Queue[RolloutGroupBundle | None] = asyncio.Queue(maxsize=queue_size)
        self.stats = CoordinatorStats()

    @staticmethod
    def parse_version(version: str) -> int:
        return int(version.rsplit("_", 1)[-1])

    def group_staleness(self, bundle: RolloutGroupBundle) -> int:
        behavior = self.parse_version(bundle.group.behavior_policy_version)
        return self.current_policy_version() - behavior

    async def submit(self, bundle: RolloutGroupBundle) -> None:
        await self.rollout_queue.put(bundle)

    async def close(self) -> None:
        await self.rollout_queue.put(None)

    async def run(self) -> CoordinatorStats:
        while True:
            bundle = await self.rollout_queue.get()
            if bundle is None:
                break
            staleness = self.group_staleness(bundle)
            if staleness > self.max_staleness_batches:
                self.stats.rejected_stale_groups += 1
                continue
            for tree in bundle.rollouts:
                for node in tree.nodes.values():
                    node.staleness = staleness
            self.stats.accepted_groups += 1
            batch = await self.compile_fn([bundle])
            await self.optimize_fn(batch)
            self.stats.optimized_batches += 1
        return self.stats
