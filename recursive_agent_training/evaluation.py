"""Evaluation aggregation and uncertainty for recursive-agent experiments."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any


@dataclass(frozen=True)
class EvaluationSummary:
    task_count: int
    success_rate: float
    average_nodes: float
    average_max_depth: float
    rollout_count: int = 0
    pass_at_8: float | None = None
    pass_at_k: float | None = None
    pass_k: int = 0
    pass_at_k_ci_low: float | None = None
    pass_at_k_ci_high: float | None = None
    success_ci_low: float | None = None
    success_ci_high: float | None = None
    pass_at_8_ci_low: float | None = None
    pass_at_8_ci_high: float | None = None
    judge_error_count: int = 0
    tool_error_count: int = 0
    average_tokens: float = 0.0
    average_latency_seconds: float = 0.0


def summarize_evaluation(results: list[dict]) -> EvaluationSummary:
    if not results:
        raise ValueError("evaluation results are empty")
    return EvaluationSummary(
        task_count=len(results),
        success_rate=sum(bool(item["success"]) for item in results) / len(results),
        average_nodes=sum(int(item.get("nodes", 1)) for item in results) / len(results),
        average_max_depth=sum(int(item.get("max_depth", 0)) for item in results) / len(results),
    )


def bootstrap_mean_interval(
    values: list[float],
    *,
    seed: int,
    samples: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap values are empty")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    generator = random.Random(seed)
    count = len(values)
    estimates = sorted(
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    low_index = min(int(tail * samples), samples - 1)
    high_index = min(int((1.0 - tail) * samples), samples - 1)
    return estimates[low_index], estimates[high_index]


def summarize_grouped_evaluation(
    per_task_results: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int = 2000,
) -> EvaluationSummary:
    if not per_task_results:
        raise ValueError("evaluation results are empty")
    task_success_rates: list[float] = []
    task_pass: list[float] = []
    all_rollouts: list[dict[str, Any]] = []
    pass_k = min(8, min(len(task.get("rollouts", [])) for task in per_task_results))
    if pass_k <= 0:
        raise ValueError("evaluation tasks must contain rollouts")
    for task in per_task_results:
        rollouts = list(task.get("rollouts", []))
        if not rollouts:
            raise ValueError(f"task {task.get('task_id')} has no rollouts")
        missing = [item for item in rollouts if item.get("success") is None]
        if missing:
            raise ValueError(
                f"task {task.get('task_id')} has {len(missing)} rollout(s) without "
                "a complete judge signal"
            )
        valid = [item for item in rollouts if item.get("success") is not None]
        successes = [1.0 if item["success"] else 0.0 for item in valid]
        task_success_rates.append(sum(successes) / len(successes))
        task_pass.append(1.0 if any(successes[:pass_k]) else 0.0)
        all_rollouts.extend(rollouts)
    success_low, success_high = bootstrap_mean_interval(
        task_success_rates,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    pass_low, pass_high = bootstrap_mean_interval(
        task_pass,
        seed=bootstrap_seed + 1,
        samples=bootstrap_samples,
    )
    rollout_count = len(all_rollouts)
    return EvaluationSummary(
        task_count=len(per_task_results),
        rollout_count=rollout_count,
        success_rate=sum(task_success_rates) / len(task_success_rates),
        pass_at_8=sum(task_pass) / len(task_pass) if pass_k == 8 else None,
        pass_at_k=sum(task_pass) / len(task_pass),
        pass_k=pass_k,
        pass_at_k_ci_low=pass_low,
        pass_at_k_ci_high=pass_high,
        success_ci_low=success_low,
        success_ci_high=success_high,
        pass_at_8_ci_low=pass_low if pass_k == 8 else None,
        pass_at_8_ci_high=pass_high if pass_k == 8 else None,
        average_nodes=sum(int(item.get("nodes", 1)) for item in all_rollouts) / rollout_count,
        average_max_depth=sum(int(item.get("max_depth", 0)) for item in all_rollouts) / rollout_count,
        average_tokens=sum(int(item.get("tokens", 0)) for item in all_rollouts) / rollout_count,
        average_latency_seconds=sum(float(item.get("latency_seconds", 0)) for item in all_rollouts) / rollout_count,
        judge_error_count=sum(bool(item.get("judge_error")) for item in all_rollouts),
        tool_error_count=sum(int(item.get("tool_errors", 0)) for item in all_rollouts),
    )
