"""Run hardware-admission candidates and freeze the selected S1 configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.model_admission import (
    freeze_selected_codeact_s0_config,
    freeze_selected_s1_config,
    inspect_candidate_run,
    model_slug,
    select_first_accepted,
)


DEFAULT_CANDIDATES = (
    "configs/train/deepdive_qwen06b_s0.yaml",
    "configs/train/deepdive_qwen17b_s0.yaml",
    "configs/train/deepdive_qwen3_4b_s0.yaml",
    "configs/train/deepdive_qwen25_3b_s0.yaml",
)


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-config", action="append")
    parser.add_argument(
        "--s1-template",
        default="configs/train/deepdive_qwen06b_s1.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", default="cuda:1,cuda:2,cuda:3,cuda:4")
    parser.add_argument("--trainer-device", default="cuda:0")
    parser.add_argument(
        "--train-task-id",
        action="append",
        help=(
            "Restrict S0 admission to a task in the fixed train split. "
            "May be repeated."
        ),
    )
    parser.add_argument("--max-gpu-memory-mib", type=float, default=21 * 1024)
    parser.add_argument(
        "--action-profile",
        choices=["existing_harness", "codeact"],
        default="codeact",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluate-all", action="store_true")
    parser.add_argument("--skip-canary", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate_paths = args.candidate_config or list(DEFAULT_CANDIDATES)
    results: list[dict] = []
    canary_complete = False

    for raw_config in candidate_paths:
        config_path = Path(raw_config).expanduser().resolve()
        config = TrainingConfig.from_yaml(config_path)
        slug = model_slug(config)
        run_dir = output / "candidates" / slug
        failure: str | None = None
        if args.execute:
            try:
                if not canary_complete and not args.skip_canary:
                    _run(
                        [
                            sys.executable,
                            "examples/run_service_canary.py",
                            "--config",
                            str(config_path),
                            "--output",
                            str(output / "service_canary.json"),
                        ]
                    )
                    canary_complete = True
                command = [
                    sys.executable,
                    "examples/run_deepdive_rao_training.py",
                    "--config",
                    str(config_path),
                    "--output",
                    str(run_dir),
                    "--devices",
                    args.devices,
                    "--trainer-device",
                    args.trainer_device,
                    "--rounds",
                    "1",
                ]
                for task_id in args.train_task_id or []:
                    command.extend(["--train-task-id", task_id])
                if (run_dir / "run_manifest.json").exists():
                    command.append("--resume")
                _run(command)
            except subprocess.CalledProcessError as exc:
                failure = f"candidate command failed with exit code {exc.returncode}"

        result = inspect_candidate_run(
            config_path,
            run_dir,
            max_gpu_memory_mib=args.max_gpu_memory_mib,
        )
        if failure:
            result["accepted"] = False
            result["failures"].insert(0, failure)
        results.append(result)
        if result["accepted"] and not args.evaluate_all:
            break

    selected = select_first_accepted(results)
    manifest = {
        "candidate_order": [str(Path(item).expanduser().resolve()) for item in candidate_paths],
        "max_gpu_memory_mib": args.max_gpu_memory_mib,
        "results": results,
        "selected_model_id": selected["model_id"] if selected else None,
        "selected_candidate_config": selected["config"] if selected else None,
        "selected_run_dir": selected["run_dir"] if selected else None,
        "admission_train_task_ids": args.train_task_id or [],
    }
    if selected:
        selected_codeact_s0 = freeze_selected_codeact_s0_config(
            selected["config"],
            output / "selected_codeact_s0.yaml",
            checkpoint_output=output / "selected_codeact_s0_training",
        )
        selected_config = freeze_selected_s1_config(
            selected["config"],
            args.s1_template,
            output / "selected_s1.yaml",
            checkpoint_output=output / "selected_training",
            action_profile=args.action_profile,
        )
        manifest["selected_s1_config"] = str(output / "selected_s1.yaml")
        manifest["selected_s1_config_hash"] = selected_config.stable_hash()
        manifest["selected_codeact_s0_config"] = str(
            output / "selected_codeact_s0.yaml"
        )
        manifest["selected_codeact_s0_config_hash"] = (
            selected_codeact_s0.stable_hash()
        )

    (output / "model_admission.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.execute and not selected:
        raise SystemExit("no model candidate passed admission")


if __name__ == "__main__":
    main()
