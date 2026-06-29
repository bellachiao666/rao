"""Run resumable DeepDive training with fixed-interval held-out evaluations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_training.config import TrainingConfig


def milestones(rounds: int, interval: int) -> list[int]:
    if rounds <= 0 or interval <= 0:
        raise ValueError("rounds and interval must be positive")
    values = list(range(interval, rounds + 1, interval))
    if not values or values[-1] != rounds:
        values.append(rounds)
    return values


def _run(command: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def completed_rounds(training_output: Path) -> int:
    manifest_path = training_output / "run_manifest.json"
    if not manifest_path.exists():
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return int(manifest.get("completed_rounds", 0))


def publication_for_target(training_output: Path, target: int) -> tuple[Path, str]:
    manifest_path = (
        training_output
        / "rounds"
        / f"round_{target - 1:04d}"
        / "round_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_version = f"policy_{target:06d}"
    if manifest.get("status") != "published":
        raise ValueError(f"round {target - 1} is not published")
    if manifest.get("policy_version_after") != expected_version:
        raise ValueError(
            f"round {target - 1} published {manifest.get('policy_version_after')}, "
            f"expected {expected_version}"
        )
    adapter = Path(str(manifest.get("adapter_path", "")))
    if not adapter.exists():
        raise FileNotFoundError(adapter)
    return adapter, expected_version


def evaluation_complete(output: Path) -> bool:
    required = (
        output / "evaluation_manifest.json",
        output / "per_task_results.jsonl",
        output / "summary.json",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def archive_incomplete_evaluation(output: Path) -> Path | None:
    if not output.exists() or evaluation_complete(output):
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archived = output.with_name(f"{output.name}.incomplete_{timestamp}")
    suffix = 1
    while archived.exists():
        archived = output.with_name(
            f"{output.name}.incomplete_{timestamp}_{suffix}"
        )
        suffix += 1
    shutil.move(str(output), str(archived))
    return archived


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/deepdive_qwen06b_s1.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-output")
    parser.add_argument("--rounds", type=int, default=75)
    parser.add_argument("--interval", type=int)
    parser.add_argument("--action-profile", choices=["existing_harness", "codeact"])
    parser.add_argument("--training-devices", default="cuda:1,cuda:2,cuda:3,cuda:4")
    parser.add_argument("--trainer-device", default="cuda:0")
    parser.add_argument("--evaluation-devices", default="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6")
    parser.add_argument("--periodic-tasks", type=int, default=10)
    parser.add_argument("--periodic-rollouts", type=int, default=4)
    parser.add_argument("--final-tasks", type=int, default=50)
    parser.add_argument("--final-rollouts", type=int, default=8)
    parser.add_argument(
        "--train-task-id",
        action="append",
        help=(
            "Restrict training to a task in the fixed train split. "
            "May be repeated; accepts a stable task id or source row id."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    training_output = (
        Path(args.training_output).expanduser().resolve()
        if args.training_output
        else output / "training"
    )
    config = TrainingConfig.from_yaml(args.config)
    config_path = Path(args.config)
    if args.action_profile:
        payload = config.model_dump(mode="json", by_alias=True)
        payload["experiment"]["action_profile"] = args.action_profile
        payload["checkpoint"]["output_dir"] = str(training_output)
        config = TrainingConfig.model_validate(payload)
        config_path = output / "frozen_training_config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                config.model_dump(mode="json", by_alias=True),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    interval = args.interval or config.evaluation.every_steps
    schedule = milestones(args.rounds, interval)
    initial_completed_rounds = completed_rounds(training_output)
    current_completed_rounds = initial_completed_rounds
    archived_evaluations: list[str] = []
    for target in schedule:
        if args.dry_run or target > current_completed_rounds:
            train_command = [
                sys.executable,
                "examples/run_deepdive_rao_training.py",
                "--config",
                str(config_path),
                "--output",
                str(training_output),
                "--devices",
                args.training_devices,
                "--trainer-device",
                args.trainer_device,
                "--rounds",
                str(target),
            ]
            for task_id in args.train_task_id or []:
                train_command.extend(["--train-task-id", task_id])
            if current_completed_rounds > 0 or (
                training_output / "run_manifest.json"
            ).exists():
                train_command.append("--resume")
            _run(train_command, dry_run=args.dry_run)
            if not args.dry_run:
                current_completed_rounds = completed_rounds(training_output)
                if current_completed_rounds < target:
                    raise RuntimeError(
                        f"training stopped at {current_completed_rounds}, "
                        f"before milestone {target}"
                    )
        else:
            print(
                f"= training milestone {target} already published; skipping",
                flush=True,
            )

        if args.dry_run:
            adapter = training_output / "rounds" / f"round_{target - 1:04d}" / "adapter"
            policy_version = f"policy_{target:06d}"
        else:
            adapter, policy_version = publication_for_target(
                training_output,
                target,
            )
        final = target == args.rounds
        evaluation_output = output / "evaluations" / f"step_{target:04d}"
        if not args.dry_run and evaluation_complete(evaluation_output):
            print(
                f"= evaluation milestone {target} already complete; skipping",
                flush=True,
            )
            continue
        if not args.dry_run:
            archived = archive_incomplete_evaluation(evaluation_output)
            if archived is not None:
                archived_evaluations.append(str(archived))
        evaluation_command = [
            sys.executable,
            "examples/run_deepdive_evaluation.py",
            "--config",
            str(config_path),
            "--output",
            str(evaluation_output),
            "--devices",
            args.evaluation_devices,
            "--adapter",
            str(adapter),
            "--policy-version",
            policy_version,
            "--tasks",
            str(args.final_tasks if final else args.periodic_tasks),
            "--rollouts",
            str(args.final_rollouts if final else args.periodic_rollouts),
        ]
        _run(evaluation_command, dry_run=args.dry_run)

    manifest = {
        "config": str(config_path.resolve()),
        "action_profile": config.experiment.action_profile,
        "training_output": str(training_output),
        "rounds": args.rounds,
        "interval": interval,
        "milestones": schedule,
        "periodic_tasks": args.periodic_tasks,
        "periodic_rollouts": args.periodic_rollouts,
        "final_tasks": args.final_tasks,
        "final_rollouts": args.final_rollouts,
        "train_task_ids": args.train_task_id or [],
        "initial_completed_rounds": initial_completed_rounds,
        "archived_incomplete_evaluations": archived_evaluations,
        "dry_run": args.dry_run,
    }
    (output / "periodic_schedule.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
