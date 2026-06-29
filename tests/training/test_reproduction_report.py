import json
from pathlib import Path
import subprocess
import sys

from examples.build_reproduction_report import (
    EXPECTED_EVALUATION_CONDITIONS,
    _evaluation_evidence,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_report_builder_collects_training_and_resume_evidence(tmp_path):
    run = tmp_path / "training" / "dense_weighted"
    scored = run / "rounds" / "round_0000" / "scored_groups" / "group_0000.json"
    batch = run / "rounds" / "round_0000" / "optimizer_batch" / "batch.json"
    _write(
        scored,
        {
            "task": {"ground_truth": "correct"},
            "rollouts": [
                {
                    "rollout_id": "rollout",
                    "root_node_id": "root",
                    "nodes": {
                        "root": {
                            "node_id": "root",
                            "depth": 0,
                            "node_task": "answer",
                            "final_answer": "correct",
                            "is_fallback": False,
                            "children_ids": [],
                            "evaluation": {"value": 1.0, "reason": "correct"},
                            "credit": {
                                "reward": {
                                    "own_success": 1.0,
                                    "child_success_mean": 0.0,
                                    "reward": 1.0,
                                }
                            },
                        }
                    },
                }
            ],
        },
    )
    _write(
        batch,
        {
            "samples": [
                {
                    "rollout_id": "rollout",
                    "node_id": "root",
                    "advantage": 1.0,
                }
            ]
        },
    )
    _write(
        run / "run_manifest.json",
        {
            "run_id": "run",
            "completed_rounds": 1,
            "latest_policy_version": "policy_000001",
        },
    )
    _write(
        run / "rounds" / "round_0000" / "round_manifest.json",
        {
            "round_index": 0,
            "policy_version_before": "policy_000000",
            "policy_version_after": "policy_000001",
            "status": "published",
            "scored_group_paths": [str(scored)],
            "optimizer_batch_path": str(batch),
            "metrics": {
                "root_success_rate": 0.5,
                "loss": 0.1,
                "ratio_mean": 1.0,
                "ratio_min": 0.9,
                "ratio_max": 1.1,
                "grad_norm": 0.2,
                "action_tokens": 10,
                "rollout_seconds": 2,
                "training_seconds": 1,
                "gpu_max_memory_mib": 1000,
                "tool_service": {"terminal_failures": 0},
                "judge_service": {"terminal_failures": 0},
            },
        },
    )
    _write(
        tmp_path / "report" / "joined_results.json",
        [
            {
                "condition": "M4_rao_recursive",
                "success_rate": 0.5,
                "pass_at_k": 0.75,
                "pass_k": 8,
                "success_ci_low": 0.25,
                "success_ci_high": 0.75,
                "pass_at_k_ci_low": 0.5,
                "pass_at_k_ci_high": 1.0,
            }
        ],
    )
    _write(
        tmp_path / "report" / "implementation_differences.json",
        {
            "paper_model": "4B",
            "reproduction_model": "0.6B",
            "paper_root_batch_group": "16 x 8",
            "reproduction_root_batch_group": "4 x 4",
            "paper_depth_steps": "4 / 25",
            "reproduction_depth_steps": "2 / 8",
            "paper_context": "40K / 256K",
            "reproduction_context": "8K / 16K",
        },
    )
    _write_jsonl(
        tmp_path / "evaluations/M2_trained_single/per_task_results.jsonl",
        [
            {"task_id": "a", "rollouts": [{"success": False}, {"success": False}]},
            {"task_id": "b", "rollouts": [{"success": False}, {"success": True}]},
        ],
    )
    _write_jsonl(
        tmp_path / "evaluations/M4_rao_recursive/per_task_results.jsonl",
        [
            {"task_id": "a", "rollouts": [{"success": True}, {"success": False}]},
            {"task_id": "b", "rollouts": [{"success": True}, {"success": True}]},
        ],
    )
    _write(
        tmp_path / "evaluations/M4_rao_recursive/evaluation_manifest.json",
        {"bootstrap_seed": 42, "bootstrap_samples": 100},
    )
    _write(
        tmp_path
        / "periodic/dense_weighted/evaluations/step_0010/summary.json",
        {
            "task_count": 10,
            "rollout_count": 40,
            "success_rate": 0.1,
            "success_ci_low": 0.0,
            "success_ci_high": 0.2,
            "pass_at_k": 0.2,
            "pass_k": 4,
            "pass_at_k_ci_low": 0.0,
            "pass_at_k_ci_high": 0.4,
            "judge_error_count": 0,
            "tool_error_count": 2,
        },
    )
    _write(
        tmp_path
        / "periodic/dense_weighted/evaluations/step_0010/evaluation_manifest.json",
        {
            "policy_version": "policy_000010",
            "adapter_hash": "abc",
            "judge_service": {"terminal_failures": 0},
            "tool_service": {"terminal_failures": 0},
        },
    )
    subprocess.run(
        [
            sys.executable,
            "examples/build_reproduction_report.py",
            "--matrix-root",
            str(tmp_path),
        ],
        check=True,
    )
    audit = json.loads(
        (tmp_path / "report" / "resume_audit.json").read_text(encoding="utf-8")
    )
    assert audit["dense_weighted"]["continuous"] is True
    comparison = json.loads(
        (tmp_path / "report" / "method_comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["available"] is True
    assert comparison["success_rate_delta_m4_minus_m2"] == 0.5
    periodic = json.loads(
        (tmp_path / "report" / "periodic_evaluations.json").read_text(
            encoding="utf-8"
        )
    )
    assert periodic[0]["step"] == 10
    assert periodic[0]["success_rate"] == 0.1
    assert "Periodic Held-Out Trend" in (
        tmp_path / "report" / "final_report.md"
    ).read_text(encoding="utf-8")
    assert (tmp_path / "report" / "final_report.md").exists()


def _write_complete_evaluation(root: Path, condition: str) -> dict:
    summary = {
        "condition": condition,
        "task_count": 1,
        "rollout_count": 2,
        "success_rate": 0.5,
        "pass_at_k": 1.0,
        "pass_k": 2,
        "success_ci_low": 0.0,
        "success_ci_high": 1.0,
        "pass_at_k_ci_low": 1.0,
        "pass_at_k_ci_high": 1.0,
        "judge_error_count": 0,
        "tool_error_count": 0,
    }
    evaluation_dir = root / "evaluations" / condition
    _write(evaluation_dir / "summary.json", summary)
    _write(
        evaluation_dir / "evaluation_manifest.json",
        {
            "rollouts_per_task": 2,
            "judge_service": {"terminal_failures": 0},
            "tool_service": {"terminal_failures": 0},
            "adapter_hash": "adapter",
            "policy_version": "policy_000001",
        },
    )
    _write_jsonl(
        evaluation_dir / "per_task_results.jsonl",
        [
            {
                "task_id": "task",
                "rollouts": [{"success": True}, {"success": False}],
            }
        ],
    )
    return summary


def test_evaluation_evidence_reports_missing_matrix_condition(tmp_path):
    joined_rows = []
    missing = "A4_sparse_unweighted"
    for condition in sorted(EXPECTED_EVALUATION_CONDITIONS - {missing}):
        joined_rows.append(_write_complete_evaluation(tmp_path, condition))
    _write(tmp_path / "report" / "joined_results.json", joined_rows)

    rows, problems = _evaluation_evidence(tmp_path)

    assert {row["condition"] for row in rows} == EXPECTED_EVALUATION_CONDITIONS - {
        missing
    }
    assert any(missing in problem and "missing" in problem for problem in problems)
    assert any("joined_results.json does not contain exactly" in problem for problem in problems)
