"""Run or resume hardware-adapted DeepDive RAO training."""

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
from recursive_agent_training.datasets import load_deepdive_csv
from recursive_agent_training.domains.deepdive import build_deepdive_tools
from recursive_agent_training.flywheel import SyncFlywheelRunner
from recursive_agent_training.hf_flywheel import HuggingFaceRuntimeFactory
from recursive_agent_training.modelscope import inspect_modelscope_snapshot, write_modelscope_manifest
from recursive_agent_training.provenance import (
    collect_environment_manifest,
    collect_source_manifest,
)
from recursive_agent_training.schemas import write_json
from recursive_agent_training.split import (
    build_split_manifest,
    read_split_manifest,
    select_split_records,
    split_hash,
    write_split_manifest,
)
from recursive_agent_training.verifiers.exact import ExactMatchVerifier
from recursive_agent_training.verifiers.llm import build_judge_verifier
from recursive_agent_training.verifiers.proxy import LazyRootProxyVerifier


def select_requested_train_records(records: list, requested_ids: list[str] | None) -> list:
    if not requested_ids:
        return records
    by_stable_id = {record.task_id: record for record in records}
    by_source_id = {
        str(record.metadata.get("source_row_id")): record
        for record in records
        if record.metadata.get("source_row_id") is not None
    }
    selected = []
    missing = []
    seen: set[str] = set()
    for requested_id in requested_ids:
        record = by_stable_id.get(requested_id) or by_source_id.get(requested_id)
        if record is None:
            missing.append(requested_id)
            continue
        if record.task_id not in seen:
            selected.append(record)
            seen.add(record.task_id)
    if missing:
        raise ValueError(
            "requested train task ids are absent from the fixed train split: "
            + ", ".join(missing)
        )
    return selected


def preflight_config_matches_except_runtime_retry_fields(
    existing: dict,
    current: dict,
) -> bool:
    existing_copy = json.loads(json.dumps(existing))
    current_copy = json.loads(json.dumps(current))
    for payload in (existing_copy, current_copy):
        sync = payload.get("sync_flywheel")
        if isinstance(sync, dict):
            sync.pop("max_collection_attempts", None)
        rollout = payload.get("rollout")
        if isinstance(rollout, dict):
            rollout.pop("max_concurrent_groups", None)
    return existing_copy == current_copy


def _prepare(args: argparse.Namespace) -> tuple[TrainingConfig, list, dict]:
    config = TrainingConfig.from_yaml(args.config)
    if args.model:
        config.model.path = str(Path(args.model).expanduser().resolve())
    if args.output:
        config.checkpoint.output_dir = str(Path(args.output).expanduser().resolve())
    if args.judge_model:
        config.services.judge.model_path = str(Path(args.judge_model).expanduser().resolve())
    records = load_deepdive_csv(config.dataset.train_path)
    split_path = Path(config.dataset.split_manifest_path)
    if not split_path.exists():
        manifest = build_split_manifest(records, config.dataset.eval_count, config.experiment.seed)
        write_split_manifest(split_path, manifest)
    manifest = read_split_manifest(split_path)
    train_records = select_split_records(records, manifest, "train")
    train_records = select_requested_train_records(
        train_records,
        args.train_task_id,
    )
    model_manifest = inspect_modelscope_snapshot(
        config.model.path,
        model_id=config.model.model_id,
        revision=config.model.revision or "master",
    )
    report = {
        "config_hash": config.stable_hash(),
        "dataset_hash": records[0].dataset_hash,
        "split_hash": split_hash(manifest),
        "train_task_count": len(train_records),
        "train_task_filter_active": bool(args.train_task_id),
        "selected_train_task_ids": (
            [record.task_id for record in train_records]
            if args.train_task_id
            else []
        ),
        "selected_train_source_row_ids": (
            [record.metadata.get("source_row_id") for record in train_records]
            if args.train_task_id
            else []
        ),
        "eval_task_count": len(manifest.eval_task_ids),
        "group_size": config.rollout.group_size,
        "root_batch_size": config.sync_flywheel.root_tasks_per_round,
        "trees_per_optimizer_batch": (
            config.rollout.group_size * config.sync_flywheel.root_tasks_per_round
        ),
        "model_id": config.model.model_id,
        "model_revision": model_manifest.resolved_revision,
        "model_snapshot_hash": model_manifest.snapshot_hash,
        "model_path": model_manifest.local_path,
        "judge_provider": config.services.judge.provider,
        "judge_model_id": config.services.judge.model_id,
        "search_provider": config.services.search.provider,
        "backend": config.optimizer.backend,
    }
    output = Path(config.checkpoint.output_dir)
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    preflight_path = manifests / "preflight.json"
    training_config_path = manifests / "training_config.json"
    if preflight_path.exists():
        existing = json.loads(preflight_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "dataset_hash",
            "split_hash",
            "selected_train_task_ids",
        )
        changed = [
            key for key in immutable_keys
            if existing.get(key) != report.get(key)
        ]
        if existing.get("config_hash") != report.get("config_hash"):
            if training_config_path.exists():
                existing_config = json.loads(
                    training_config_path.read_text(encoding="utf-8")
                )
                if not preflight_config_matches_except_runtime_retry_fields(
                    existing_config,
                    config.to_redacted_dict(),
                ):
                    changed.append("config_hash")
            else:
                changed.append("config_hash")
        if changed:
            raise ValueError(
                "resume preflight differs for immutable fields: "
                + ", ".join(changed)
            )
    write_modelscope_manifest(manifests / "policy_model.json", model_manifest)
    if config.services.judge.provider == "local_transformers":
        judge_manifest = inspect_modelscope_snapshot(
            config.services.judge.model_path,
            model_id=config.services.judge.model_id,
            revision=config.services.judge.revision,
        )
        write_modelscope_manifest(manifests / "judge_model.json", judge_manifest)
    write_json(preflight_path, report)
    write_json(training_config_path, config.to_redacted_dict())
    write_json(manifests / "environment.json", collect_environment_manifest())
    write_json(manifests / "source.json", collect_source_manifest(PROJECT_ROOT))
    return config, train_records, report


def _verifier_factory(config: TrainingConfig):
    if config.services.judge.provider == "exact":
        base_factory = ExactMatchVerifier
    else:
        base_factory = lambda: build_judge_verifier(config.services.judge)
    if config.reward.subtask_provider == "root_proxy":
        return lambda: LazyRootProxyVerifier(base_factory())
    return base_factory


async def run(args: argparse.Namespace) -> None:
    config, train_records, report = _prepare(args)
    if args.preflight:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if config.optimizer.backend != "torch":
        raise ValueError("hardware-adapted DeepDive runner currently requires optimizer.backend=torch")
    devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
    if not devices:
        raise ValueError("at least one rollout device is required")
    runtime_factory = HuggingFaceRuntimeFactory(
        config=config,
        base_model_path=config.model.path,
        devices=devices,
        trainer_device=args.trainer_device,
    )
    runner = SyncFlywheelRunner(
        config=config,
        tasks=train_records,
        output_dir=config.checkpoint.output_dir,
        model_name=config.model.name,
        initial_checkpoint_path=config.model.path,
        runtime_factory=runtime_factory,
        verifier_factory=_verifier_factory(config),
        tools_factory=lambda: build_deepdive_tools(settings=config.services.search),
    )
    rounds = args.rounds or config.sync_flywheel.target_rounds
    completed = await runner.run(rounds, resume=args.resume)
    print(
        json.dumps(
            {
                **report,
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
    parser.add_argument("--config", default="configs/train/deepdive_qwen06b_s0.yaml")
    parser.add_argument("--model")
    parser.add_argument("--judge-model")
    parser.add_argument("--output")
    parser.add_argument("--devices", default="cuda:1,cuda:2")
    parser.add_argument("--trainer-device", default="cuda:0")
    parser.add_argument("--rounds", type=int)
    parser.add_argument(
        "--train-task-id",
        action="append",
        help=(
            "Restrict training to a task in the fixed train split. "
            "May be repeated; accepts a stable task id or source row id."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
