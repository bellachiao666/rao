"""Run or resume synchronous round-based RAO training."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.flywheel import SyncFlywheelRunner
from recursive_agent_training.hf_flywheel import HuggingFaceRuntimeFactory
from recursive_agent_training.synthetic import make_stochastic_exact_tasks
from recursive_agent_training.verifiers.exact import ExactMatchVerifier


async def run(args: argparse.Namespace) -> None:
    config = TrainingConfig.from_yaml(args.config)
    config.model.path = str(Path(args.model).resolve())
    config.model.name = Path(args.model).name
    config.checkpoint.output_dir = str(Path(args.output).resolve())
    rounds = args.rounds or config.sync_flywheel.target_rounds
    task_count = config.sync_flywheel.root_tasks_per_round
    tasks = make_stochastic_exact_tasks(task_count)
    runtime_factory = HuggingFaceRuntimeFactory(
        config=config,
        base_model_path=config.model.path,
        device=args.device or "cuda:0",
        devices=tuple(
            item.strip()
            for item in (args.devices or "").split(",")
            if item.strip()
        ),
    )
    runner = SyncFlywheelRunner(
        config=config,
        tasks=tasks,
        output_dir=config.checkpoint.output_dir,
        model_name=config.model.name,
        initial_checkpoint_path=config.model.path,
        runtime_factory=runtime_factory,
        verifier_factory=ExactMatchVerifier,
    )
    completed = await runner.run(rounds, resume=args.resume)
    print(
        json.dumps(
            {
                "output": config.checkpoint.output_dir,
                "target_rounds": rounds,
                "completed_this_invocation": [item.round_index for item in completed],
                "latest": str(Path(config.checkpoint.output_dir) / "latest.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/sync_qwen_0_6b.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--devices",
        help="Comma-separated rollout devices; the first device also performs training.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
