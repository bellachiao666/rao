"""Build the evidence package and final report from completed matrix runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_training.admission import audit_training_round


EXPECTED_TRAINING_CONDITIONS = {
    "single_trained",
    "dense_weighted",
    "dense_unweighted",
    "sparse_weighted",
    "sparse_unweighted",
}


EXPECTED_EVALUATION_CONDITIONS = {
    "M1_base_single",
    "M2_trained_single",
    "M3_base_recursive",
    "M4_rao_recursive",
    "A1_dense_weighted",
    "A2_dense_unweighted",
    "A3_sparse_weighted",
    "A4_sparse_unweighted",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_evidence(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    training_root = root / "training"
    for run_dir in sorted(path for path in training_root.iterdir() if path.is_dir()):
        run_manifest = _load(run_dir / "run_manifest.json")
        rounds = []
        problems = []
        previous_after = "policy_000000"
        for round_dir in sorted((run_dir / "rounds").glob("round_[0-9][0-9][0-9][0-9]")):
            manifest = _load(round_dir / "round_manifest.json")
            rounds.append(manifest)
            expected_index = len(rounds) - 1
            if manifest["round_index"] != expected_index:
                problems.append(
                    f"round index {manifest['round_index']} does not match {expected_index}"
                )
            if manifest["policy_version_before"] != previous_after:
                problems.append(
                    f"round {expected_index} predecessor is {manifest['policy_version_before']}, "
                    f"expected {previous_after}"
                )
            if manifest["status"] != "published":
                problems.append(f"round {expected_index} status is {manifest['status']}")
            admission = audit_training_round(round_dir)
            if not admission["accepted"]:
                problems.append(
                    f"round {expected_index} admission failed: "
                    + "; ".join(admission["failures"])
                )
            previous_after = manifest["policy_version_after"]
            metrics = manifest.get("metrics", {})
            rows.append(
                {
                    "condition": run_dir.name,
                    "round": manifest["round_index"],
                    "policy_before": manifest["policy_version_before"],
                    "policy_after": manifest["policy_version_after"],
                    "status": manifest["status"],
                    "root_success_rate": metrics.get("root_success_rate"),
                    "loss": metrics.get("loss"),
                    "ratio_mean": metrics.get("ratio_mean"),
                    "ratio_min": metrics.get("ratio_min"),
                    "ratio_max": metrics.get("ratio_max"),
                    "grad_norm": metrics.get("grad_norm"),
                    "action_tokens": metrics.get("action_tokens"),
                    "rollout_seconds": metrics.get("rollout_seconds"),
                    "training_seconds": metrics.get("training_seconds"),
                    "gpu_max_memory_mib": metrics.get("gpu_max_memory_mib"),
                    "admission_accepted": admission["accepted"],
                    "trusted_positive_reward_count": admission[
                        "trusted_positive_reward_count"
                    ],
                    "tool_failures": metrics.get("tool_service", {}).get("failures", 0),
                    "tool_terminal_failures": metrics.get("tool_service", {}).get(
                        "terminal_failures", 0
                    ),
                    "tool_retries": metrics.get("tool_service", {}).get("retries", 0),
                    "judge_failures": metrics.get("judge_service", {}).get("failures", 0),
                    "judge_terminal_failures": metrics.get("judge_service", {}).get(
                        "terminal_failures", 0
                    ),
                    "judge_retries": metrics.get("judge_service", {}).get("retries", 0),
                    "judge_estimated_cost": metrics.get("judge_service", {}).get(
                        "estimated_cost", 0
                    ),
                }
            )
        if run_manifest.get("completed_rounds") != len(rounds):
            problems.append(
                f"run manifest completed_rounds={run_manifest.get('completed_rounds')} "
                f"but found {len(rounds)} published round directories"
            )
        audits[run_dir.name] = {
            "run_id": run_manifest.get("run_id"),
            "completed_rounds": run_manifest.get("completed_rounds"),
            "latest_policy_version": run_manifest.get("latest_policy_version"),
            "round_directory_count": len(rounds),
            "continuous": not problems,
            "problems": problems,
        }
    return rows, audits


def _write_training_metrics(report_dir: Path, rows: list[dict[str, Any]]) -> None:
    (report_dir / "training_metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if rows:
        with (report_dir / "training_metrics.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _periodic_evidence(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    periodic_root = root / "periodic"
    if not periodic_root.exists():
        return rows
    for condition_dir in sorted(
        path for path in periodic_root.iterdir() if path.is_dir()
    ):
        evaluation_root = condition_dir / "evaluations"
        if not evaluation_root.exists():
            continue
        for evaluation_dir in sorted(evaluation_root.glob("step_[0-9][0-9][0-9][0-9]")):
            summary_path = evaluation_dir / "summary.json"
            manifest_path = evaluation_dir / "evaluation_manifest.json"
            if not summary_path.exists() or not manifest_path.exists():
                continue
            summary = _load(summary_path)
            manifest = _load(manifest_path)
            rows.append(
                {
                    "condition": condition_dir.name,
                    "step": int(evaluation_dir.name.rsplit("_", 1)[-1]),
                    "task_count": summary.get("task_count"),
                    "rollout_count": summary.get("rollout_count"),
                    "success_rate": summary.get("success_rate"),
                    "success_ci_low": summary.get("success_ci_low"),
                    "success_ci_high": summary.get("success_ci_high"),
                    "pass_at_k": summary.get("pass_at_k"),
                    "pass_k": summary.get("pass_k"),
                    "pass_at_k_ci_low": summary.get("pass_at_k_ci_low"),
                    "pass_at_k_ci_high": summary.get("pass_at_k_ci_high"),
                    "judge_error_count": summary.get("judge_error_count"),
                    "tool_error_count": summary.get("tool_error_count"),
                    "judge_terminal_failures": manifest.get(
                        "judge_service", {}
                    ).get("terminal_failures", 0),
                    "tool_terminal_failures": manifest.get(
                        "tool_service", {}
                    ).get("terminal_failures", 0),
                    "policy_version": manifest.get("policy_version"),
                    "adapter_hash": manifest.get("adapter_hash"),
                }
            )
    return rows


def _write_periodic_evidence(
    report_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    (report_dir / "periodic_evaluations.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not rows:
        return
    with (report_dir / "periodic_evaluations.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Periodic Held-Out Evaluations",
        "",
        "| Condition | Step | Tasks | Rollouts | Success | Pass@K | K | Judge errors | Tool errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {step} | {task_count} | {rollout_count} | "
            "{success_rate:.4f} | {pass_at_k:.4f} | {pass_k} | "
            "{judge_error_count} | {tool_error_count} |".format(**row)
        )
    (report_dir / "periodic_evaluations.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _evaluation_evidence(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for condition in sorted(EXPECTED_EVALUATION_CONDITIONS):
        evaluation_dir = root / "evaluations" / condition
        summary_path = evaluation_dir / "summary.json"
        manifest_path = evaluation_dir / "evaluation_manifest.json"
        per_task_path = evaluation_dir / "per_task_results.jsonl"
        missing = [
            path.name
            for path in (summary_path, manifest_path, per_task_path)
            if not path.is_file() or path.stat().st_size <= 0
        ]
        if missing:
            problems.append(f"{condition} missing {', '.join(missing)}")
            continue
        try:
            summary = _load(summary_path)
            manifest = _load(manifest_path)
            per_task = _read_jsonl(per_task_path)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{condition} has unreadable evaluation evidence: {exc}")
            continue
        task_count = int(summary.get("task_count") or 0)
        rollout_count = int(summary.get("rollout_count") or 0)
        rollouts_per_task = int(manifest.get("rollouts_per_task") or 0)
        if len(per_task) != task_count:
            problems.append(
                f"{condition} per-task row count {len(per_task)} != task_count {task_count}"
            )
        for item in per_task:
            rollouts = item.get("rollouts", [])
            if rollouts_per_task and len(rollouts) != rollouts_per_task:
                problems.append(
                    f"{condition} task {item.get('task_id')} has {len(rollouts)} "
                    f"rollouts, expected {rollouts_per_task}"
                )
            if any(rollout.get("success") is None for rollout in rollouts):
                problems.append(
                    f"{condition} task {item.get('task_id')} has incomplete judge signal"
                )
        if rollouts_per_task and rollout_count != task_count * rollouts_per_task:
            problems.append(
                f"{condition} rollout_count {rollout_count} != "
                f"{task_count} * {rollouts_per_task}"
            )
        rows.append(
            {
                "condition": condition,
                "task_count": task_count,
                "rollout_count": rollout_count,
                "rollouts_per_task": rollouts_per_task,
                "success_rate": summary.get("success_rate"),
                "pass_at_k": summary.get("pass_at_k"),
                "pass_k": summary.get("pass_k"),
                "judge_error_count": int(summary.get("judge_error_count") or 0),
                "tool_error_count": int(summary.get("tool_error_count") or 0),
                "judge_terminal_failures": int(
                    manifest.get("judge_service", {}).get("terminal_failures") or 0
                ),
                "tool_terminal_failures": int(
                    manifest.get("tool_service", {}).get("terminal_failures") or 0
                ),
                "adapter_hash": manifest.get("adapter_hash"),
                "policy_version": manifest.get("policy_version"),
            }
        )
    joined_path = root / "report" / "joined_results.json"
    if joined_path.exists():
        try:
            joined_rows = _load(joined_path)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"joined_results.json is unreadable: {exc}")
        else:
            joined_by_condition = {str(row.get("condition")): row for row in joined_rows}
            expected = {row["condition"] for row in rows}
            if set(joined_by_condition) != EXPECTED_EVALUATION_CONDITIONS:
                problems.append("joined_results.json does not contain exactly the expected matrix")
            for row in rows:
                joined = joined_by_condition.get(row["condition"])
                if not joined:
                    continue
                for field in ("task_count", "rollout_count", "success_rate", "pass_at_k", "pass_k"):
                    if joined.get(field) != row.get(field):
                        problems.append(
                            f"{row['condition']} joined_results {field} differs from "
                            "evaluation summary"
                        )
                expected.discard(row["condition"])
    else:
        problems.append("report/joined_results.json is missing")
    return rows, problems


def _aggregate_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, dict[str, float]] = {}
    for row in rows:
        values = by_condition.setdefault(
            row["condition"],
            {
                "tool_failures": 0,
                "tool_terminal_failures": 0,
                "tool_retries": 0,
                "judge_failures": 0,
                "judge_terminal_failures": 0,
                "judge_retries": 0,
                "judge_estimated_cost": 0.0,
                "rollout_seconds": 0.0,
                "training_seconds": 0.0,
            },
        )
        for key in values:
            values[key] += float(row.get(key) or 0)
    return by_condition


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _task_metrics(items: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    result = {}
    for item in items:
        successes = [rollout.get("success") for rollout in item.get("rollouts", [])]
        if not successes or any(value is None for value in successes):
            raise ValueError(f"incomplete evaluation for task {item.get('task_id')}")
        numeric = [1.0 if value else 0.0 for value in successes]
        result[str(item["task_id"])] = (
            sum(numeric) / len(numeric),
            1.0 if any(numeric) else 0.0,
        )
    return result


def _paired_interval(
    differences: list[float],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    generator = random.Random(seed)
    count = len(differences)
    estimates = sorted(
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return estimates[int(0.025 * samples)], estimates[min(int(0.975 * samples), samples - 1)]


def _method_comparison(root: Path) -> dict[str, Any]:
    baseline_path = root / "evaluations/M2_trained_single/per_task_results.jsonl"
    rao_path = root / "evaluations/M4_rao_recursive/per_task_results.jsonl"
    if not baseline_path.exists() or not rao_path.exists():
        return {"available": False, "reason": "M2 or M4 per-task results are missing"}
    baseline = _task_metrics(_read_jsonl(baseline_path))
    rao = _task_metrics(_read_jsonl(rao_path))
    if set(baseline) != set(rao):
        raise ValueError("M2 and M4 task IDs differ")
    manifest = _load(root / "evaluations/M4_rao_recursive/evaluation_manifest.json")
    seed = int(manifest.get("bootstrap_seed", 42))
    samples = int(manifest.get("bootstrap_samples", 2000))
    task_ids = sorted(baseline)
    success_differences = [rao[item][0] - baseline[item][0] for item in task_ids]
    pass_differences = [rao[item][1] - baseline[item][1] for item in task_ids]
    success_low, success_high = _paired_interval(
        success_differences,
        seed=seed + 100,
        samples=samples,
    )
    pass_low, pass_high = _paired_interval(
        pass_differences,
        seed=seed + 101,
        samples=samples,
    )
    success_delta = sum(success_differences) / len(success_differences)
    pass_delta = sum(pass_differences) / len(pass_differences)
    return {
        "available": True,
        "task_count": len(task_ids),
        "success_rate_delta_m4_minus_m2": success_delta,
        "success_rate_delta_ci_low": success_low,
        "success_rate_delta_ci_high": success_high,
        "pass_at_k_delta_m4_minus_m2": pass_delta,
        "pass_at_k_delta_ci_low": pass_low,
        "pass_at_k_delta_ci_high": pass_high,
        "point_estimate_improves": success_delta > 0 and pass_delta >= 0,
        "success_interval_excludes_zero": success_low > 0 or success_high < 0,
        "pass_interval_excludes_zero": pass_low > 0 or pass_high < 0,
        "bootstrap_seed": seed,
        "bootstrap_samples": samples,
    }


def _final_report(
    root: Path,
    audits: dict[str, Any],
    failures: dict[str, Any],
    method_comparison: dict[str, Any],
    periodic_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> str:
    result_rows = _load(root / "report/joined_results.json")
    differences = _load(root / "report/implementation_differences.json")
    lines = [
        "# Hardware-Adapted RAO Reproduction Report",
        "",
        "## Claim",
        "",
        "This run evaluates RAO method behavior under the hardware-adapted "
        f"{differences['reproduction_model']} configuration. It is not a Qwen 4B "
        "paper-number reproduction.",
        "",
        "## Training Completion",
        "",
        "| Condition | Steps | Continuous | Latest policy |",
        "|---|---:|---:|---|",
    ]
    for condition, audit in sorted(audits.items()):
        lines.append(
            f"| {condition} | {audit['completed_rounds']} | "
            f"{'yes' if audit['continuous'] else 'no'} | "
            f"{audit['latest_policy_version']} |"
        )
    if periodic_rows:
        lines.extend(
            [
                "",
                "## Periodic Held-Out Trend",
                "",
                "| Condition | Step | Success | Pass@K | K | Success 95% CI | Pass@K 95% CI |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in periodic_rows:
            lines.append(
                "| {condition} | {step} | {success_rate:.4f} | "
                "{pass_at_k:.4f} | {pass_k} | "
                "[{success_ci_low:.4f}, {success_ci_high:.4f}] | "
                "[{pass_at_k_ci_low:.4f}, {pass_at_k_ci_high:.4f}] |".format(
                    **row
                )
            )
    lines.extend(
        [
            "",
            "## Evaluation",
            "",
            "| Condition | Success | Pass@K | K | Success 95% CI | Pass@K 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result_rows:
        lines.append(
            "| {condition} | {success_rate:.4f} | {pass_at_k:.4f} | {pass_k} | "
            "[{success_ci_low:.4f}, {success_ci_high:.4f}] | "
            "[{pass_at_k_ci_low:.4f}, {pass_at_k_ci_high:.4f}] |".format(**row)
        )
    if evaluation_rows:
        lines.extend(
            [
                "",
                "## Evaluation Completeness",
                "",
                "| Condition | Tasks | Rollouts | Rollouts/task | Judge terminal failures | Tool terminal failures |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in evaluation_rows:
            lines.append(
                "| {condition} | {task_count} | {rollout_count} | {rollouts_per_task} | "
                "{judge_terminal_failures} | {tool_terminal_failures} |".format(**row)
            )
    lines.extend(["", "## M4 Versus M2", ""])
    if method_comparison.get("available"):
        lines.extend(
            [
                "- Success-rate delta (M4 - M2): "
                f"{method_comparison['success_rate_delta_m4_minus_m2']:.4f}, 95% paired "
                f"bootstrap CI [{method_comparison['success_rate_delta_ci_low']:.4f}, "
                f"{method_comparison['success_rate_delta_ci_high']:.4f}].",
                "- Pass@K delta (M4 - M2): "
                f"{method_comparison['pass_at_k_delta_m4_minus_m2']:.4f}, 95% paired "
                f"bootstrap CI [{method_comparison['pass_at_k_delta_ci_low']:.4f}, "
                f"{method_comparison['pass_at_k_delta_ci_high']:.4f}].",
                "- Point-estimate method trend: "
                + (
                    "positive."
                    if method_comparison["point_estimate_improves"]
                    else "not positive."
                ),
            ]
        )
    else:
        lines.append(f"- Unavailable: {method_comparison.get('reason', 'unknown reason')}.")
    lines.extend(["", "## Service And Runtime Totals", ""])
    for condition, values in sorted(failures.items()):
        lines.append(
            f"- `{condition}`: tool failures={int(values['tool_failures'])}, "
            f"terminal tool failures={int(values['tool_terminal_failures'])}, "
            f"judge failures={int(values['judge_failures'])}, "
            f"terminal judge failures={int(values['judge_terminal_failures'])}, "
            f"rollout hours={values['rollout_seconds'] / 3600:.2f}, "
            f"training hours={values['training_seconds'] / 3600:.2f}."
        )
    lines.extend(
        [
            "",
            "## Implementation Differences",
            "",
            f"- Model: {differences['reproduction_model']} instead of {differences['paper_model']}.",
            f"- Batch/group: {differences['reproduction_root_batch_group']} instead of "
            f"{differences['paper_root_batch_group']}.",
            f"- Recursion: {differences['reproduction_depth_steps']} instead of "
            f"{differences['paper_depth_steps']}.",
            f"- Context: {differences['reproduction_context']} instead of "
            f"{differences['paper_context']}.",
            "- Training curriculum: "
            + (
                ", ".join(differences.get("training_task_ids", []))
                if differences.get("training_task_ids")
                else "the complete fixed training split"
            )
            + ".",
            "",
            "## Interpretation Rule",
            "",
            "The method trend is supported only if M4 improves over M2 under the fixed "
            "evaluation protocol and the uncertainty intervals are reported. Isolated "
            "headline-number matching is not treated as reproduction evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--require-75", action="store_true")
    args = parser.parse_args()
    root = Path(args.matrix_root).expanduser().resolve()
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows, audits = _training_evidence(root)
    _write_training_metrics(report_dir, rows)
    periodic_rows = _periodic_evidence(root)
    _write_periodic_evidence(report_dir, periodic_rows)
    evaluation_rows, evaluation_problems = _evaluation_evidence(root)
    (report_dir / "evaluation_evidence.json").write_text(
        json.dumps(evaluation_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / "resume_audit.json").write_text(
        json.dumps(audits, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    failures = _aggregate_failures(rows)
    (report_dir / "cost_and_failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    method_comparison = _method_comparison(root)
    (report_dir / "method_comparison.json").write_text(
        json.dumps(method_comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = _final_report(
        root,
        audits,
        failures,
        method_comparison,
        periodic_rows,
        evaluation_rows,
    )
    (report_dir / "final_report.md").write_text(report, encoding="utf-8")
    if args.require_75:
        missing = sorted(EXPECTED_TRAINING_CONDITIONS - set(audits))
        incomplete = [
            condition
            for condition, audit in audits.items()
            if audit["completed_rounds"] < 75 or not audit["continuous"]
        ]
        incomplete.extend(f"{condition} (missing)" for condition in missing)
        terminal_failures = [
            condition
            for condition, values in failures.items()
            if values["judge_terminal_failures"] > 0
        ]
        incomplete.extend(
            f"{condition} (terminal judge failures)" for condition in terminal_failures
        )
        final_periodic = [
            row
            for row in periodic_rows
            if row["condition"] == "dense_weighted"
            and row["step"] == 75
            and int(row.get("task_count") or 0) == 50
            and int(row.get("rollout_count") or 0) == 400
            and int(row.get("judge_terminal_failures") or 0) == 0
            and int(row.get("tool_terminal_failures") or 0) == 0
        ]
        if not final_periodic:
            incomplete.append("dense_weighted (missing complete step-75 periodic evaluation)")
        incomplete.extend(evaluation_problems)
        observed_evaluations = {row["condition"] for row in evaluation_rows}
        missing_evaluations = sorted(
            EXPECTED_EVALUATION_CONDITIONS - observed_evaluations
        )
        incomplete.extend(
            f"{condition} (missing final evaluation)" for condition in missing_evaluations
        )
        for row in evaluation_rows:
            if int(row.get("task_count") or 0) != 50:
                incomplete.append(f"{row['condition']} (expected 50 evaluation tasks)")
            if int(row.get("rollout_count") or 0) != 400:
                incomplete.append(f"{row['condition']} (expected 400 evaluation rollouts)")
            if int(row.get("rollouts_per_task") or 0) != 8:
                incomplete.append(f"{row['condition']} (expected 8 rollouts per task)")
            if int(row.get("judge_terminal_failures") or 0) > 0:
                incomplete.append(f"{row['condition']} (terminal judge failures)")
            if int(row.get("tool_terminal_failures") or 0) > 0:
                incomplete.append(f"{row['condition']} (terminal tool failures)")
        if incomplete:
            raise SystemExit("incomplete 75-step runs: " + ", ".join(incomplete))
    print(
        json.dumps(
            {
                "report": str(report_dir / "final_report.md"),
                "training_conditions": len(audits),
                "training_rows": len(rows),
                "periodic_evaluations": len(periodic_rows),
                "evaluation_conditions": len(evaluation_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
