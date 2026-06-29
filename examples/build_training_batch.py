"""Verify a rollout group and compile an optimizer batch."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_training.compilation import compile_optimizer_batch, verify_and_score_group
from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.schemas import read_group_bundle, write_json
from recursive_agent_training.verifiers.exact import ExactMatchVerifier


async def run(args) -> None:
    config = TrainingConfig.from_yaml(args.config)
    bundle = read_group_bundle(args.rollouts)
    await verify_and_score_group(bundle, ExactMatchVerifier(), config.reward.delegation_lambda)
    batch = compile_optimizer_batch([bundle], config, args.step, bundle.group.behavior_policy_version)
    write_json(args.output, batch)
    print(f"Wrote {args.output} with {len(batch.samples)} node samples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/mini_rao.yaml")
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", default="output/train/compiled_batch.json")
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
