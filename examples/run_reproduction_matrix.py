"""Prepare and execute the hardware-adapted comparison and ablation matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
from datetime import datetime, timezone

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_training.config import TrainingConfig


SINGLE_AGENT_DELEGATION_LAMBDA = 0.0
RECURSIVE_DELEGATION_LAMBDA = 0.2
HARDWARE_ADAPTED_LEARNING_RATE = 3e-5
HARDWARE_ADAPTED_MAX_STALENESS = 3
HARDWARE_ADAPTED_RATIO_MIN = 0.95
HARDWARE_ADAPTED_RATIO_MAX = 1.05
DEFAULT_MIN_FILTERED_MAIN_TASKS = 128


TRAINING_CONDITIONS = {
    "single_trained": {
        "max_depth": 0,
        "max_steps_per_node": 8,
        "weighting": "depth_inverse_frequency",
        "sparse": False,
        "delegation_lambda": SINGLE_AGENT_DELEGATION_LAMBDA,
    },
    "dense_weighted": {
        "max_depth": 2,
        "max_steps_per_node": 8,
        "weighting": "depth_inverse_frequency",
        "sparse": False,
        "delegation_lambda": RECURSIVE_DELEGATION_LAMBDA,
    },
    "dense_unweighted": {
        "max_depth": 2,
        "max_steps_per_node": 8,
        "weighting": "uniform",
        "sparse": False,
        "delegation_lambda": RECURSIVE_DELEGATION_LAMBDA,
    },
    "sparse_weighted": {
        "max_depth": 2,
        "max_steps_per_node": 8,
        "weighting": "depth_inverse_frequency",
        "sparse": True,
        "delegation_lambda": RECURSIVE_DELEGATION_LAMBDA,
        "max_collection_attempts": 32,
        "max_concurrent_groups": 4,
    },
    "sparse_unweighted": {
        "max_depth": 2,
        "max_steps_per_node": 8,
        "weighting": "uniform",
        "sparse": True,
        "delegation_lambda": RECURSIVE_DELEGATION_LAMBDA,
        "max_collection_attempts": 32,
        "max_concurrent_groups": 4,
    },
}


EVALUATION_CONDITIONS = {
    "M1_base_single": {"training": None, "config": "single_trained"},
    "M2_trained_single": {"training": "single_trained", "config": "single_trained"},
    "M3_base_recursive": {"training": None, "config": "dense_weighted"},
    "M4_rao_recursive": {"training": "dense_weighted", "config": "dense_weighted"},
    "A1_dense_weighted": {"training": "dense_weighted", "config": "dense_weighted"},
    "A2_dense_unweighted": {"training": "dense_unweighted", "config": "dense_unweighted"},
    "A3_sparse_weighted": {"training": "sparse_weighted", "config": "sparse_weighted"},
    "A4_sparse_unweighted": {"training": "sparse_unweighted", "config": "sparse_unweighted"},
}


def _write_condition_configs(
    base: TrainingConfig,
    output: Path,
    action_profile: str,
) -> dict[str, Path]:
    config_dir = output / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    model_slug = re.sub(r"[^a-z0-9]+", "_", base.model.name.casefold()).strip("_")
    for name, overrides in TRAINING_CONDITIONS.items():
        payload = base.model_dump(mode="json", by_alias=True)
        payload["experiment"]["name"] = f"{model_slug}_{name}"
        payload["experiment"]["action_profile"] = action_profile
        payload["rollout"]["max_depth"] = overrides["max_depth"]
        payload["rollout"]["max_steps_per_node"] = overrides["max_steps_per_node"]
        payload["rollout"]["max_staleness_batches"] = HARDWARE_ADAPTED_MAX_STALENESS
        payload["weighting"]["type"] = overrides["weighting"]
        payload["reward"]["lambda"] = overrides["delegation_lambda"]
        payload["reward"]["root_provider"] = "llm_judge"
        payload["reward"]["subtask_provider"] = (
            "root_proxy" if overrides["sparse"] else "llm_judge"
        )
        payload["optimizer"]["learning_rate"] = HARDWARE_ADAPTED_LEARNING_RATE
        payload["sync_flywheel"]["ratio_min"] = HARDWARE_ADAPTED_RATIO_MIN
        payload["sync_flywheel"]["ratio_max"] = HARDWARE_ADAPTED_RATIO_MAX
        if "max_collection_attempts" in overrides:
            payload["sync_flywheel"]["max_collection_attempts"] = overrides[
                "max_collection_attempts"
            ]
        if "max_concurrent_groups" in overrides:
            payload["rollout"]["max_concurrent_groups"] = overrides[
                "max_concurrent_groups"
            ]
        payload.setdefault("parameter_sources", {})
        payload["parameter_sources"]["reward.lambda"] = (
            "single_agent_no_delegation_bonus"
            if overrides["max_depth"] == 0
            else "review_adjusted_delegation_bonus"
        )
        payload["parameter_sources"]["rollout.max_staleness_batches"] = (
            "paper_staleness_reuse"
        )
        payload["parameter_sources"]["optimizer.learning_rate"] = (
            "review_adjusted_8x4090_conservative_lora"
        )
        payload["parameter_sources"]["sync_flywheel.ratio_bounds"] = (
            "paper_clipped_range"
        )
        payload["checkpoint"]["output_dir"] = str(output / "training" / name)
        config = TrainingConfig.model_validate(payload)
        path = config_dir / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(
                config.model_dump(mode="json", by_alias=True),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        result[name] = path
    return result


def _latest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "latest.json"
    if not path.exists():
        raise FileNotFoundError(f"training publication is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _evaluation_complete(output: Path, *, tasks: int, rollouts: int) -> bool:
    required = (
        output / "evaluation_manifest.json",
        output / "per_task_results.jsonl",
        output / "summary.json",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (output / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(summary.get("task_count") or 0) == tasks
        and int(summary.get("rollout_count") or 0) == tasks * rollouts
        and int(manifest.get("rollouts_per_task") or 0) == rollouts
    )


def _archive_incomplete_evaluation(output: Path) -> Path | None:
    if not output.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archived = output.with_name(f"{output.name}.incomplete_{timestamp}")
    suffix = 1
    while archived.exists():
        archived = output.with_name(f"{output.name}.incomplete_{timestamp}_{suffix}")
        suffix += 1
    shutil.move(str(output), str(archived))
    return archived


def _train(
    configs: dict[str, Path],
    output: Path,
    *,
    rounds: int,
    devices: str,
    trainer_device: str,
    train_task_ids: list[str],
) -> None:
    for name, config in configs.items():
        run_dir = output / "training" / name
        command = [
            sys.executable,
            "examples/run_deepdive_rao_training.py",
            "--config",
            str(config),
            "--devices",
            devices,
            "--trainer-device",
            trainer_device,
            "--rounds",
            str(rounds),
        ]
        for task_id in train_task_ids:
            command.extend(["--train-task-id", task_id])
        if (run_dir / "run_manifest.json").exists():
            command.append("--resume")
        _run(command)


def _evaluate(
    configs: dict[str, Path],
    output: Path,
    *,
    tasks: int,
    rollouts: int,
    devices: str,
) -> None:
    evaluated: dict[tuple[str | None, str], Path] = {}
    for condition, settings in EVALUATION_CONDITIONS.items():
        training = settings["training"]
        config_name = settings["config"]
        key = (training, config_name)
        condition_output = output / "evaluations" / condition
        if _evaluation_complete(condition_output, tasks=tasks, rollouts=rollouts):
            print(f"= evaluation {condition} already complete; skipping", flush=True)
            evaluated[key] = condition_output
            continue
        if key in evaluated:
            source = evaluated[key]
            archived = _archive_incomplete_evaluation(condition_output)
            if archived is not None:
                print(f"= archived incomplete evaluation {archived}", flush=True)
            condition_output.mkdir(parents=True, exist_ok=True)
            for filename in (
                "evaluation_manifest.json",
                "per_task_results.jsonl",
                "summary.json",
            ):
                (condition_output / filename).write_bytes((source / filename).read_bytes())
            reuse = {"reused_from": source.name}
            (condition_output / "reuse.json").write_text(
                json.dumps(reuse, indent=2),
                encoding="utf-8",
            )
            continue
        archived = _archive_incomplete_evaluation(condition_output)
        if archived is not None:
            print(f"= archived incomplete evaluation {archived}", flush=True)
        condition_output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "examples/run_deepdive_evaluation.py",
            "--config",
            str(configs[config_name]),
            "--output",
            str(condition_output),
            "--devices",
            devices,
            "--tasks",
            str(tasks),
            "--rollouts",
            str(rollouts),
        ]
        if training:
            publication = _latest(output / "training" / training)
            command.extend(
                [
                    "--adapter",
                    publication["adapter_path"],
                    "--policy-version",
                    publication["policy_version"],
                ]
            )
        _run(command)
        evaluated[key] = condition_output


def _join_results(output: Path, base: TrainingConfig) -> None:
    rows = []
    controls: dict[str, Any] | None = None
    comparable_fields = (
        "dataset_hash",
        "split_hash",
        "task_ids",
        "task_count",
        "rollouts_per_task",
        "experiment_seed",
        "model_id",
        "model_revision",
        "model_snapshot_hash",
        "modelscope_version",
        "source_hash",
        "action_profile",
        "sampling_temperature",
        "max_new_tokens",
        "eval_context_tokens",
        "judge_model_id",
        "judge_revision",
        "judge_snapshot_hash",
        "judge_provider_version",
        "search_provider",
        "search_endpoint",
        "bootstrap_seed",
        "bootstrap_samples",
    )
    for condition in EVALUATION_CONDITIONS:
        directory = output / "evaluations" / condition
        manifest = json.loads((directory / "evaluation_manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        current = {field: manifest.get(field) for field in comparable_fields}
        if controls is None:
            controls = current
        elif current != controls:
            raise ValueError(f"evaluation controls differ for {condition}")
        rows.append({"condition": condition, **summary})
    report_dir = output / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "joined_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (report_dir / "joined_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "# Hardware-Adapted RAO Result Matrix",
        "",
        "| Condition | Success | Pass@K | K | Success 95% CI | Pass@K 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {condition} | {success_rate:.4f} | {pass_at_k:.4f} | {pass_k} | "
            "[{success_ci_low:.4f}, {success_ci_high:.4f}] | "
            "[{pass_at_k_ci_low:.4f}, {pass_at_k_ci_high:.4f}] |".format(**row)
        )
    (report_dir / "result_matrix.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    matrix_manifest = json.loads(
        (output / "matrix_manifest.json").read_text(encoding="utf-8")
    )
    differences = {
        "claim": "hardware-adapted method reproduction",
        "paper_model": "Qwen-3-4B-Instruct-2507",
        "reproduction_model": f"ModelScope {base.model.model_id}",
        "paper_root_batch_group": "16 x 8",
        "reproduction_root_batch_group": (
            f"{base.sync_flywheel.root_tasks_per_round} x "
            f"{base.rollout.group_size}"
        ),
        "paper_depth_steps": "depth 4, 25 steps per node",
        "reproduction_depth_steps": (
            f"depth {base.rollout.max_depth}, "
            f"{base.rollout.max_steps_per_node} steps per node"
        ),
        "paper_context": "40K train, 256K eval",
        "reproduction_context": (
            f"{base.model.train_context_tokens} train, "
            f"{base.model.eval_context_tokens} eval"
        ),
        "training_task_ids": matrix_manifest.get("train_task_ids", []),
        "controls": controls,
    }
    (report_dir / "implementation_differences.json").write_text(
        json.dumps(differences, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    difference_lines = [
        "# Implementation Differences",
        "",
        f"- Claim: {differences['claim']}.",
        f"- Model: {differences['reproduction_model']} instead of {differences['paper_model']}.",
        "- Root batch/group: "
        f"{differences['reproduction_root_batch_group']} instead of "
        f"{differences['paper_root_batch_group']}.",
        f"- Recursion: {differences['reproduction_depth_steps']} instead of "
        f"{differences['paper_depth_steps']}.",
        f"- Context: {differences['reproduction_context']} instead of "
        f"{differences['paper_context']}.",
        "- Training curriculum: "
        + (
            ", ".join(differences["training_task_ids"])
            if differences["training_task_ids"]
            else "the complete fixed training split"
        )
        + ".",
        "- Action runtime: restricted persistent CodeAct state mapped to harness actions, "
        "not a general asynchronous Python interpreter.",
    ]
    (report_dir / "implementation_differences.md").write_text(
        "\n".join(difference_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/train/deepdive_qwen06b_s1.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--action-profile", choices=["existing_harness", "codeact"], default="codeact")
    parser.add_argument("--rounds", type=int, default=75)
    parser.add_argument("--eval-tasks", type=int, default=50)
    parser.add_argument("--eval-rollouts", type=int, default=8)
    parser.add_argument(
        "--min-train-task-filter-count",
        type=int,
        default=DEFAULT_MIN_FILTERED_MAIN_TASKS,
        help=(
            "Minimum number of explicitly filtered training tasks allowed for a "
            "75-step main run. Ignored when no --train-task-id filter is used."
        ),
    )
    parser.add_argument(
        "--allow-task-filtered-main-run",
        action="store_true",
        help=(
            "Allow a 75-step main run to train on fewer than "
            "--min-train-task-filter-count tasks. Intended only for deliberate "
            "curriculum or debugging runs."
        ),
    )
    parser.add_argument("--training-devices", default="cuda:1,cuda:2,cuda:3,cuda:4")
    parser.add_argument("--trainer-device", default="cuda:0")
    parser.add_argument("--evaluation-devices", default="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6")
    parser.add_argument(
        "--train-task-id",
        action="append",
        help=(
            "Restrict every trained matrix condition to the same task in the "
            "fixed train split. May be repeated."
        ),
    )
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--execute-evaluation", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = TrainingConfig.from_yaml(args.base_config)
    if (
        args.execute_training
        and args.rounds >= 75
        and args.train_task_id
        and len(set(args.train_task_id)) < args.min_train_task_filter_count
        and not args.allow_task_filtered_main_run
    ):
        raise SystemExit(
            "refusing 75-step main run with only "
            f"{len(set(args.train_task_id))} filtered training task(s); "
            "omit --train-task-id for the complete train split, provide at least "
            f"{args.min_train_task_filter_count} tasks, or pass "
            "--allow-task-filtered-main-run for an intentional curriculum run"
        )
    configs = _write_condition_configs(base, output, args.action_profile)
    matrix_manifest = {
        "base_config": str(Path(args.base_config).resolve()),
        "model_id": base.model.model_id,
        "model_revision": base.model.revision,
        "action_profile": args.action_profile,
        "rounds": args.rounds,
        "eval_tasks": args.eval_tasks,
        "eval_rollouts": args.eval_rollouts,
        "train_task_ids": args.train_task_id or [],
        "training_conditions": TRAINING_CONDITIONS,
        "evaluation_conditions": EVALUATION_CONDITIONS,
        "configs": {name: str(path) for name, path in configs.items()},
    }
    (output / "matrix_manifest.json").write_text(
        json.dumps(matrix_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.execute_training:
        _train(
            configs,
            output,
            rounds=args.rounds,
            devices=args.training_devices,
            trainer_device=args.trainer_device,
            train_task_ids=args.train_task_id or [],
        )
    if args.execute_evaluation:
        _evaluate(
            configs,
            output,
            tasks=args.eval_tasks,
            rollouts=args.eval_rollouts,
            devices=args.evaluation_devices,
        )
        _join_results(output, base)
        report_command = [
            sys.executable,
            "examples/build_reproduction_report.py",
            "--matrix-root",
            str(output),
        ]
        if args.rounds >= 75:
            report_command.append("--require-75")
        _run(report_command)
    if not args.execute_training and not args.execute_evaluation:
        print(json.dumps(matrix_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
