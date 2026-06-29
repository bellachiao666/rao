"""Run fixed-split DeepDive rollouts and write auditable evaluation artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_training.compilation import verify_and_score_group
from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.datasets import load_deepdive_csv
from recursive_agent_training.domains.deepdive import build_deepdive_tools
from recursive_agent_training.evaluation import summarize_grouped_evaluation
from recursive_agent_training.flywheel import sha256_path
from recursive_agent_training.grouping import collect_rollout_group
from recursive_agent_training.hf_flywheel import HuggingFacePolicyPool
from recursive_agent_training.mini import make_harness_config
from recursive_agent_training.modelscope import inspect_modelscope_snapshot
from recursive_agent_training.provenance import collect_source_manifest
from recursive_agent_training.schemas import VerifierStatus
from recursive_agent_training.snapshots import PolicySnapshot
from recursive_agent_training.split import read_split_manifest, select_split_records, split_hash
from recursive_agent_training.verifiers.llm import build_judge_verifier


async def run(args: argparse.Namespace) -> None:
    config = TrainingConfig.from_yaml(args.config)
    if args.model:
        config.model.path = str(Path(args.model).expanduser().resolve())
    if args.judge_model:
        config.services.judge.model_path = str(Path(args.judge_model).expanduser().resolve())
    if args.max_depth is not None:
        config.rollout.max_depth = args.max_depth
    config.rollout.group_size = args.rollouts
    records = load_deepdive_csv(config.dataset.eval_path or config.dataset.train_path)
    split = read_split_manifest(config.dataset.split_manifest_path)
    tasks = select_split_records(records, split, "eval")[: args.tasks]
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    model_manifest = inspect_modelscope_snapshot(
        config.model.path,
        model_id=config.model.model_id,
        revision=config.model.revision,
    )
    judge_model_manifest = None
    if config.services.judge.provider == "local_transformers":
        if Path(config.services.judge.model_path).resolve() == Path(config.model.path).resolve():
            judge_model_manifest = model_manifest
        else:
            judge_model_manifest = inspect_modelscope_snapshot(
                config.services.judge.model_path,
                model_id=config.services.judge.model_id,
                revision=config.services.judge.revision,
            )
    adapter_path = str(Path(args.adapter).expanduser().resolve()) if args.adapter else ""
    snapshot = PolicySnapshot(
        version=args.policy_version,
        model_name=config.model.name,
        tokenizer_revision=model_manifest.resolved_revision,
        template_version="hf_chat_template_v1",
        checkpoint_path=adapter_path or config.model.path,
        checkpoint_hash=sha256_path(adapter_path) if adapter_path else model_manifest.snapshot_hash,
    )
    devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
    pool = HuggingFacePolicyPool(
        config=config,
        base_model_path=config.model.path,
        snapshot=snapshot,
        devices=devices,
        adapter_path=adapter_path,
    )
    tools = build_deepdive_tools(settings=config.services.search)
    verifier = build_judge_verifier(config.services.judge)
    per_task = []
    judge_metrics = {}
    try:
        for task_index, task in enumerate(tasks):
            started = time.monotonic()
            bundle = await collect_rollout_group(
                task,
                snapshot,
                args.rollouts,
                pool.policy_factory,
                make_harness_config(config, str(output / "traces" / f"task_{task_index:04d}")),
                tools=tools,
                group_sequence=task_index,
                output_dir=output / "traces" / f"task_{task_index:04d}",
            )
            await verify_and_score_group(bundle, verifier, config.reward.delegation_lambda)
            task_seconds = time.monotonic() - started
            rollouts = []
            for tree in bundle.rollouts:
                root = tree.nodes[tree.root_node_id]
                tool_errors = sum(
                    int(node.metadata.get("tool_error_count", 0))
                    for node in tree.nodes.values()
                )
                rollouts.append(
                    {
                        "rollout_id": tree.rollout_id,
                        "success": (
                            bool(root.evaluation.value)
                            if root.evaluation
                            and root.evaluation.status == VerifierStatus.COMPLETE
                            and root.evaluation.value is not None
                            else None
                        ),
                        "answer": root.final_answer,
                        "nodes": len(tree.nodes),
                        "max_depth": max(node.depth for node in tree.nodes.values()),
                        "steps": sum(len(node.turns) for node in tree.nodes.values()),
                        "tokens": sum(
                            len(turn.tokens.generated_ids)
                            for node in tree.nodes.values()
                            for turn in node.turns
                        ),
                        "delegations": sum(bool(node.children_ids) for node in tree.nodes.values()),
                        "latency_seconds": task_seconds / len(bundle.rollouts),
                        "judge_error": (
                            root.evaluation.error
                            if root.evaluation and root.evaluation.status != VerifierStatus.COMPLETE
                            else None
                        ),
                        "tool_errors": tool_errors,
                    }
                )
            per_task.append(
                {
                    "task_id": task.task_id,
                    "prompt": task.prompt,
                    "ground_truth": task.ground_truth,
                    "rollouts": rollouts,
                }
            )
    finally:
        read_judge_metrics = getattr(verifier, "metrics", None)
        if callable(read_judge_metrics):
            judge_metrics = read_judge_metrics()
        close_verifier = getattr(verifier, "close", None)
        if callable(close_verifier):
            close_verifier()
        pool_metrics = pool.metrics()
        pool.close()

    summary = summarize_grouped_evaluation(
        per_task,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    source_manifest = collect_source_manifest(PROJECT_ROOT)
    manifest = {
        "config_hash": config.stable_hash(),
        "dataset_hash": split.dataset_hash,
        "split_hash": split_hash(split),
        "task_ids": [task.task_id for task in tasks],
        "task_count": len(tasks),
        "rollouts_per_task": args.rollouts,
        "experiment_seed": config.experiment.seed,
        "model_id": config.model.model_id,
        "model_revision": model_manifest.resolved_revision,
        "model_snapshot_hash": model_manifest.snapshot_hash,
        "modelscope_version": model_manifest.modelscope_version,
        "source_hash": source_manifest["source_hash"],
        "adapter_path": adapter_path,
        "adapter_hash": sha256_path(adapter_path) if adapter_path else "",
        "policy_version": args.policy_version,
        "action_profile": config.experiment.action_profile,
        "max_depth": config.rollout.max_depth,
        "sampling_temperature": config.rollout.temperature,
        "max_new_tokens": config.rollout.max_tokens,
        "eval_context_tokens": config.model.eval_context_tokens,
        "judge_model_id": config.services.judge.model_id,
        "judge_revision": config.services.judge.revision,
        "judge_snapshot_hash": (
            judge_model_manifest.snapshot_hash if judge_model_manifest else ""
        ),
        "judge_provider_version": config.services.judge.provider_version,
        "search_provider": config.services.search.provider,
        "search_endpoint": config.services.search.endpoint,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "runtime": pool_metrics,
        "tool_service": tools.service_metrics(),
        "judge_service": judge_metrics,
    }
    (output / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output / "per_task_results.jsonl").open("w", encoding="utf-8") as handle:
        for item in per_task:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output / "summary.json").write_text(
        json.dumps(summary.__dict__, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary.__dict__, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/deepdive_qwen06b_s1.yaml")
    parser.add_argument("--model")
    parser.add_argument("--judge-model")
    parser.add_argument("--adapter")
    parser.add_argument("--policy-version", default="policy_000000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", default="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6")
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.rollouts < 2:
        parser.error("--rollouts must be at least 2")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
