"""Collect a deterministic recursive rollout group for inspection."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.grouping import collect_rollout_group
from recursive_agent_training.mini import make_harness_config, make_scripted_policy, synthetic_task
from recursive_agent_training.schemas import write_json
from recursive_agent_training.snapshots import PolicySnapshotProvider


async def run(args) -> None:
    config = TrainingConfig.from_yaml(args.config)
    snapshot = PolicySnapshotProvider(model_name=config.model.name).current()
    bundle = await collect_rollout_group(
        synthetic_task(),
        snapshot,
        config.rollout.group_size,
        lambda index, seed: make_scripted_policy(snapshot, "alpha" if index % 2 == 0 else "beta"),
        make_harness_config(config, args.output),
        output_dir=Path(args.output) / "traces",
    )
    write_json(Path(args.output) / "rollout_group.json", bundle)
    print(f"Wrote {args.output}/rollout_group.json with {len(bundle.rollouts)} trees")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/mini_rao.yaml")
    parser.add_argument("--output", default="output/train/collected")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
