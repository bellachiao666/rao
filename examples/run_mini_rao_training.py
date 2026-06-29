"""Run a deterministic recursive rollout-to-CISPO training step."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_training.compilation import compile_optimizer_batch, verify_and_score_group
from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.grouping import collect_rollout_group
from recursive_agent_training.metrics import MetricsLogger
from recursive_agent_training.mini import build_toy_model, make_harness_config, make_scripted_policy, synthetic_task
from recursive_agent_training.optimizers.torch_optimizer import TorchCISPOOptimizer, TorchOptimizerSettings
from recursive_agent_training.schemas import write_json
from recursive_agent_training.snapshots import PolicySnapshotProvider
from recursive_agent_training.verifiers.exact import ExactMatchVerifier


async def run(config: TrainingConfig, steps: int) -> None:
    output = Path(config.checkpoint.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_provider = PolicySnapshotProvider(model_name=config.model.name)
    model = build_toy_model()
    optimizer = TorchCISPOOptimizer(
        model,
        TorchOptimizerSettings(
            learning_rate=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            beta1=config.optimizer.beta1,
            beta2=config.optimizer.beta2,
            grad_clip_norm=config.optimizer.grad_clip_norm,
            epsilon_low=config.objective.epsilon_low,
            epsilon_high=config.objective.epsilon_high,
        ),
    )
    logger = MetricsLogger(output / "metrics.jsonl")
    task = synthetic_task()
    harness_config = make_harness_config(config, str(output))
    for step in range(steps):
        snapshot = snapshot_provider.current()
        bundle = await collect_rollout_group(
            task,
            snapshot,
            config.rollout.group_size,
            lambda index, seed: make_scripted_policy(snapshot, "alpha" if index % 2 == 0 else "beta"),
            harness_config,
            group_sequence=step,
            output_dir=output / "rollouts",
        )
        await verify_and_score_group(bundle, ExactMatchVerifier(), config.reward.delegation_lambda)
        batch = compile_optimizer_batch([bundle], config, step, optimizer.current_version())
        metrics = optimizer.step(batch)
        checkpoint = output / f"checkpoint_{step + 1:04d}.pt"
        optimizer.save_checkpoint(str(checkpoint))
        write_json(output / f"group_{step:04d}.json", bundle)
        write_json(output / f"batch_{step:04d}.json", batch)
        logger.log({"step": step, **metrics})
        snapshot_provider.publish(str(checkpoint))
        print(json.dumps({"step": step, **metrics}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/mini_rao.yaml")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(run(TrainingConfig.from_yaml(args.config), args.steps))


if __name__ == "__main__":
    main()
