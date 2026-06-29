"""Run deterministic Search and local ModelScope Judge canaries."""

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
from recursive_agent_training.domains.deepdive import DeepDiveWebServices
from recursive_agent_training.schemas import (
    RootTaskRecord,
    RolloutTreeRecord,
    TrainingNodeRecord,
    TreeStatus,
    VerifierStatus,
)
from recursive_agent_training.verifiers.llm import build_judge_verifier


async def run(config_path: str) -> dict:
    config = TrainingConfig.from_yaml(config_path)
    search = DeepDiveWebServices(config.services.search)
    search_result = await search.search(
        {
            "query": "Recursive Agent Optimization arXiv 2605.06639",
            "max_results": 5,
        }
    )
    serialized_results = json.dumps(search_result.get("results", []), ensure_ascii=False)
    search_ok = bool(search_result.get("results")) and (
        "2605.06639" in serialized_results
        or "Recursive Agent Optimization" in serialized_results
    )

    task = RootTaskRecord(
        task_id="judge-canary",
        domain="canary",
        split="eval",
        prompt="What is the capital of France?",
        ground_truth="Paris",
    )
    node = TrainingNodeRecord(
        task_id=task.task_id,
        group_id="canary",
        rollout_id="canary",
        node_id="root",
        parent_id=None,
        depth=0,
        node_task=task.prompt,
        terminal_status="completed",
        terminal_reason="finish",
        final_answer="The capital of France is Paris.",
    )
    tree = RolloutTreeRecord(
        rollout_id="canary",
        group_id="canary",
        rollout_index=0,
        seed=42,
        root_node_id="root",
        behavior_policy_version="policy_000000",
        nodes={"root": node},
        tree_status=TreeStatus.COMPLETE,
    )
    judge = build_judge_verifier(config.services.judge)
    try:
        judge_result = await judge.evaluate(node, tree, task)
        child = TrainingNodeRecord(
            task_id=task.task_id,
            group_id="canary",
            rollout_id="canary",
            node_id="child",
            parent_id="root",
            depth=1,
            node_task="Directly search for the title of the paper.",
            terminal_status="completed",
            terminal_reason="finish",
            final_answer="Google Scholar",
        )
        node.children_ids = ["child"]
        tree.nodes["child"] = child
        subtask_guard_result = await judge.evaluate(child, tree, task)
        parrot_child = TrainingNodeRecord(
            task_id=task.task_id,
            group_id="canary",
            rollout_id="canary",
            node_id="parrot-child",
            parent_id="root",
            depth=1,
            node_task="Determine the surname of the political figure mentioned in the context.",
            terminal_status="completed",
            terminal_reason="finish",
            final_answer="surname of the political figure mentioned in the context",
        )
        node.children_ids.append("parrot-child")
        tree.nodes["parrot-child"] = parrot_child
        subtask_parrot_guard_result = await judge.evaluate(
            parrot_child,
            tree,
            task,
        )
        vague_child = TrainingNodeRecord(
            task_id=task.task_id,
            group_id="canary",
            rollout_id="canary",
            node_id="vague-child",
            parent_id="root",
            depth=1,
            node_task="narrow delegated task",
            terminal_status="completed",
            terminal_reason="finish",
            final_answer="the current leader is identified and the task is complete",
        )
        node.children_ids.append("vague-child")
        tree.nodes["vague-child"] = vague_child
        subtask_vague_guard_result = await judge.evaluate(
            vague_child,
            tree,
            task,
        )
        no_result_child = TrainingNodeRecord(
            task_id=task.task_id,
            group_id="canary",
            rollout_id="canary",
            node_id="no-result-child",
            parent_id="root",
            depth=1,
            node_task="Examine search results and identify a matching paper title.",
            terminal_status="completed",
            terminal_reason="finish",
            final_answer=(
                "None of the search results provided a clear matching title. "
                "The maximum depth was reached before a specific match was identified."
            ),
        )
        node.children_ids.append("no-result-child")
        tree.nodes["no-result-child"] = no_result_child
        subtask_no_result_guard_result = await judge.evaluate(
            no_result_child,
            tree,
            task,
        )
        no_result_paraphrase_child = TrainingNodeRecord(
            task_id=task.task_id,
            group_id="canary",
            rollout_id="canary",
            node_id="no-result-paraphrase-child",
            parent_id="root",
            depth=1,
            node_task="Identify initial relevant academic journals.",
            terminal_status="completed",
            terminal_reason="finish",
            final_answer=(
                "No initial academic journals were identified within the allowed depth "
                "limit due to the search results. Further research may be required."
            ),
        )
        node.children_ids.append("no-result-paraphrase-child")
        tree.nodes["no-result-paraphrase-child"] = no_result_paraphrase_child
        subtask_no_result_paraphrase_guard_result = await judge.evaluate(
            no_result_paraphrase_child,
            tree,
            task,
        )
        irrelevant_title_child = TrainingNodeRecord(
            task_id=task.task_id,
            group_id="canary",
            rollout_id="canary",
            node_id="irrelevant-title-child",
            parent_id="root",
            depth=1,
            node_task="Examine search results to identify a potential matching title.",
            terminal_status="completed",
            terminal_reason="finish",
            final_answer=(
                "A New Method of Surface Acoustic Wave Pressure Sensing Based on "
                "Acoustic Attenuation Mechanism"
            ),
        )
        node.children_ids.append("irrelevant-title-child")
        tree.nodes["irrelevant-title-child"] = irrelevant_title_child
        subtask_irrelevant_title_guard_result = await judge.evaluate(
            irrelevant_title_child,
            tree,
            task,
        )
        failed_children_synthesis = TrainingNodeRecord(
            task_id=task.task_id,
            group_id="canary",
            rollout_id="canary",
            node_id="failed-children-synthesis",
            parent_id="root",
            depth=1,
            node_task=(
                "Find relevant scientific publications related to neurological "
                "conditions in Nouvelle-Aquitaine."
            ),
            terminal_status="completed",
            terminal_reason="finish",
            final_answer=(
                "Synthesis\n\nTask: find relevant scientific publications related "
                "to neurological conditions in Nouvelle-Aquitaine\n\nFinal answer "
                "from child results:\nNo child returned a successful answer.\n\n"
                "Limitations:\n- narrow delegated task: Cannot launch child because "
                "max_depth has been reached."
            ),
        )
        node.children_ids.append("failed-children-synthesis")
        tree.nodes["failed-children-synthesis"] = failed_children_synthesis
        subtask_failed_children_guard_result = await judge.evaluate(
            failed_children_synthesis,
            tree,
            task,
        )
        matched_answer_child = TrainingNodeRecord(
            task_id=task.task_id,
            group_id="canary",
            rollout_id="canary",
            node_id="matched-answer-child",
            parent_id="root",
            depth=1,
            node_task="Find the exact answer to the parent question.",
            terminal_status="completed",
            terminal_reason="finish",
            final_answer="The exact answer is Paris.",
        )
        node.children_ids.append("matched-answer-child")
        tree.nodes["matched-answer-child"] = matched_answer_child
        subtask_matched_answer_result = await judge.evaluate(
            matched_answer_child,
            tree,
            task,
        )
        judge_metrics = judge.metrics() if callable(getattr(judge, "metrics", None)) else {}
    finally:
        close = getattr(judge, "close", None)
        if callable(close):
            close()
    judge_ok = (
        judge_result.status == VerifierStatus.COMPLETE
        and judge_result.value == 1.0
    )
    subtask_guard_ok = (
        subtask_guard_result.status == VerifierStatus.COMPLETE
        and subtask_guard_result.value == 0.0
    )
    subtask_parrot_guard_ok = (
        subtask_parrot_guard_result.status == VerifierStatus.COMPLETE
        and subtask_parrot_guard_result.value == 0.0
    )
    subtask_vague_guard_ok = (
        subtask_vague_guard_result.status == VerifierStatus.COMPLETE
        and subtask_vague_guard_result.value == 0.0
    )
    subtask_no_result_guard_ok = (
        subtask_no_result_guard_result.status == VerifierStatus.COMPLETE
        and subtask_no_result_guard_result.value == 0.0
    )
    subtask_no_result_paraphrase_guard_ok = (
        subtask_no_result_paraphrase_guard_result.status
        == VerifierStatus.COMPLETE
        and subtask_no_result_paraphrase_guard_result.value == 0.0
    )
    subtask_irrelevant_title_guard_ok = (
        subtask_irrelevant_title_guard_result.status
        == VerifierStatus.COMPLETE
        and subtask_irrelevant_title_guard_result.value == 0.0
    )
    subtask_failed_children_guard_ok = (
        subtask_failed_children_guard_result.status
        == VerifierStatus.COMPLETE
        and subtask_failed_children_guard_result.value == 0.0
    )
    subtask_matched_answer_ok = (
        subtask_matched_answer_result.status == VerifierStatus.COMPLETE
        and subtask_matched_answer_result.value == 1.0
    )
    result = {
        "search_ok": search_ok,
        "search_result_count": len(search_result.get("results", [])),
        "search_metrics": search.metrics.snapshot(),
        "judge_ok": judge_ok,
        "judge_result": judge_result.model_dump(mode="json"),
        "subtask_guard_ok": subtask_guard_ok,
        "subtask_guard_result": subtask_guard_result.model_dump(mode="json"),
        "subtask_parrot_guard_ok": subtask_parrot_guard_ok,
        "subtask_parrot_guard_result": subtask_parrot_guard_result.model_dump(
            mode="json"
        ),
        "subtask_vague_guard_ok": subtask_vague_guard_ok,
        "subtask_vague_guard_result": subtask_vague_guard_result.model_dump(
            mode="json"
        ),
        "subtask_no_result_guard_ok": subtask_no_result_guard_ok,
        "subtask_no_result_guard_result": subtask_no_result_guard_result.model_dump(
            mode="json"
        ),
        "subtask_no_result_paraphrase_guard_ok": (
            subtask_no_result_paraphrase_guard_ok
        ),
        "subtask_no_result_paraphrase_guard_result": (
            subtask_no_result_paraphrase_guard_result.model_dump(mode="json")
        ),
        "subtask_irrelevant_title_guard_ok": subtask_irrelevant_title_guard_ok,
        "subtask_irrelevant_title_guard_result": (
            subtask_irrelevant_title_guard_result.model_dump(mode="json")
        ),
        "subtask_failed_children_guard_ok": subtask_failed_children_guard_ok,
        "subtask_failed_children_guard_result": (
            subtask_failed_children_guard_result.model_dump(mode="json")
        ),
        "subtask_matched_answer_ok": subtask_matched_answer_ok,
        "subtask_matched_answer_result": (
            subtask_matched_answer_result.model_dump(mode="json")
        ),
        "judge_metrics": judge_metrics,
    }
    if (
        not search_ok
        or not judge_ok
        or not subtask_guard_ok
        or not subtask_parrot_guard_ok
        or not subtask_vague_guard_ok
        or not subtask_no_result_guard_ok
        or not subtask_no_result_paraphrase_guard_ok
        or not subtask_irrelevant_title_guard_ok
        or not subtask_failed_children_guard_ok
        or not subtask_matched_answer_ok
    ):
        raise RuntimeError("service canary failed: " + json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/deepdive_qwen06b_s0.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = asyncio.run(run(args.config))
    if args.output:
        path = Path(args.output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
