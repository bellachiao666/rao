import pytest

from recursive_agent_training.coordinator import AsyncTrainingCoordinator
from recursive_agent_training.schemas import GroupStatus, RolloutGroupBundle, RolloutGroupRecord, RootTaskRecord


def test_policy_version_parser():
    assert AsyncTrainingCoordinator.parse_version("policy_000003") == 3


@pytest.mark.asyncio
async def test_coordinator_rejects_group_over_staleness_limit():
    optimized = []

    async def compile_fn(bundles):
        raise AssertionError("stale groups must not compile")

    async def optimize_fn(batch):
        optimized.append(batch)

    coordinator = AsyncTrainingCoordinator(3, lambda: 4, compile_fn, optimize_fn)
    bundle = RolloutGroupBundle(
        group=RolloutGroupRecord(
            group_id="g",
            task_id="t",
            expected_group_size=2,
            behavior_policy_version="policy_000000",
            status=GroupStatus.COMPLETE,
        ),
        task=RootTaskRecord(task_id="t", domain="x", split="train", prompt="q"),
        rollouts=[],
    )
    await coordinator.submit(bundle)
    await coordinator.close()
    stats = await coordinator.run()
    assert stats.rejected_stale_groups == 1
    assert not optimized
