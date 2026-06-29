"""Audit a published training round before it is admitted to longer runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recursive_agent_training.verifiers.llm import (
    root_answer_passes_lexical_guard,
    subtask_answer_passes_minimum_guard,
)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_training_round(
    round_dir: str | Path,
    *,
    advantage_epsilon: float = 1e-8,
    require_published: bool = True,
) -> dict[str, Any]:
    directory = Path(round_dir).expanduser().resolve()
    manifest = _load(directory / "round_manifest.json")
    failures: list[str] = []
    allowed_statuses = {"published"} if require_published else {
        "frozen",
        "training",
        "published",
    }
    if manifest.get("status") not in allowed_statuses:
        failures.append(f"round status is {manifest.get('status')!r}, expected 'published'")

    trusted_positive_evaluations: set[tuple[str, str]] = set()
    trusted_reward_nodes: set[tuple[str, str]] = set()
    positive_nodes: list[dict[str, Any]] = []
    for raw_path in manifest.get("scored_group_paths", []):
        group = _load(raw_path)
        task = group["task"]
        reference = task.get("ground_truth") or ""
        for rollout in group.get("rollouts", []):
            root_node_id = rollout.get("root_node_id")
            nodes = rollout.get("nodes", {})
            root_node = nodes.get(root_node_id, {})
            root_evaluation = root_node.get("evaluation") or {}
            root_signal_trusted = (
                float(root_evaluation.get("value") or 0.0) > 0
                and not root_node.get("is_fallback")
                and root_answer_passes_lexical_guard(
                    reference,
                    root_node.get("final_answer") or "",
                )
            )
            for node in nodes.values():
                evaluation = node.get("evaluation") or {}
                if float(evaluation.get("value") or 0.0) <= 0:
                    continue
                key = (rollout["rollout_id"], node["node_id"])
                trusted = True
                reason = ""
                if node.get("is_fallback"):
                    trusted = False
                    reason = "fallback node received positive reward"
                elif evaluation.get("is_proxy"):
                    trusted = root_signal_trusted
                    reason = (
                        "proxy reward lacks a trusted positive root success signal"
                    )
                elif node["node_id"] == root_node_id:
                    trusted = root_answer_passes_lexical_guard(
                        reference,
                        node.get("final_answer") or "",
                    )
                    reason = "root answer failed lexical guard"
                else:
                    trusted = subtask_answer_passes_minimum_guard(
                        node.get("final_answer") or "",
                        node.get("node_task") or "",
                        reference,
                    )
                    reason = (
                        "subtask answer failed substantive or root-reference guard"
                    )
                positive_nodes.append(
                    {
                        "rollout_id": key[0],
                        "node_id": key[1],
                        "depth": node.get("depth"),
                        "node_task": node.get("node_task"),
                        "answer": node.get("final_answer"),
                        "judge_reason": evaluation.get("reason"),
                        "is_proxy": bool(evaluation.get("is_proxy")),
                        "trusted": trusted,
                    }
                )
                if trusted:
                    trusted_positive_evaluations.add(key)
                else:
                    failures.append(f"{key[0]}/{key[1]}: {reason}")
            for node in nodes.values():
                key = (rollout["rollout_id"], node["node_id"])
                reward = node.get("credit", {}).get("reward") or {}
                reward_value = float(
                    reward.get("reward")
                    if reward.get("reward") is not None
                    else (node.get("evaluation") or {}).get("value") or 0.0
                )
                if reward_value <= 0:
                    continue
                own_success = float(
                    reward.get("own_success")
                    if reward.get("own_success") is not None
                    else (node.get("evaluation") or {}).get("value") or 0.0
                )
                child_success = float(reward.get("child_success_mean") or 0.0)
                own_trusted = own_success <= 0 or key in trusted_positive_evaluations
                child_keys = {
                    (rollout["rollout_id"], child_id)
                    for child_id in node.get("children_ids", [])
                }
                child_trusted = (
                    child_success <= 0
                    or bool(child_keys & trusted_positive_evaluations)
                )
                if own_trusted and child_trusted:
                    trusted_reward_nodes.add(key)
                else:
                    failures.append(
                        f"{key[0]}/{key[1]}: positive reward lacks trusted success source"
                    )

    if not positive_nodes:
        failures.append("round contains no positive reward")

    batch_path = manifest.get("optimizer_batch_path")
    positive_advantage_nodes: set[tuple[str, str]] = set()
    nonzero_advantage_count = 0
    if not batch_path:
        failures.append("optimizer batch path is missing")
    else:
        batch = _load(batch_path)
        for sample in batch.get("samples", []):
            advantage = float(sample.get("advantage") or 0.0)
            if abs(advantage) <= advantage_epsilon:
                continue
            nonzero_advantage_count += 1
            if advantage > advantage_epsilon:
                positive_advantage_nodes.add(
                    (sample["rollout_id"], sample["node_id"])
                )
        if not nonzero_advantage_count:
            failures.append("optimizer batch contains no non-zero advantage")
        untrusted = positive_advantage_nodes - trusted_reward_nodes
        for rollout_id, node_id in sorted(untrusted):
            failures.append(
                f"{rollout_id}/{node_id}: positive advantage lacks trusted positive reward"
            )
        if not positive_advantage_nodes:
            failures.append("optimizer batch contains no positive advantage")

    judge_metrics = manifest.get("metrics", {}).get("judge_service", {})
    if float(judge_metrics.get("terminal_failures") or 0.0) > 0:
        failures.append("judge service recorded terminal failures")

    return {
        "accepted": not failures,
        "round_dir": str(directory),
        "policy_version_after": manifest.get("policy_version_after"),
        "positive_reward_count": len(positive_nodes),
        "trusted_positive_reward_count": len(trusted_positive_evaluations),
        "trusted_reward_node_count": len(trusted_reward_nodes),
        "nonzero_advantage_sample_count": nonzero_advantage_count,
        "positive_advantage_node_count": len(positive_advantage_nodes),
        "positive_nodes": positive_nodes,
        "failures": failures,
    }


def latest_round_dir(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser().resolve()
    latest = _load(root / "latest.json")
    return root / "rounds" / f"round_{int(latest['round_index']):04d}"
