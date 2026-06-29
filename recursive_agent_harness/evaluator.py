"""LLM-based evaluation for recursive rollout results."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from recursive_agent_harness.config import HarnessConfig
from recursive_agent_harness.policy import ChatClient, OpenAICompatibleChatClient
from recursive_agent_harness.prompts import ROOT_TASK_JUDGE_PROMPT, SUBTASK_JUDGE_PROMPT
from recursive_agent_harness.tree import ExecutionTree


class EvaluationResult(BaseModel):
    reason: str
    success: bool
    scope: Literal["root", "subtask"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMJudge:
    """Judge root tasks and sub-tasks with an OpenAI-compatible LLM."""

    def __init__(self, client: ChatClient | None = None, config: HarnessConfig | None = None):
        self.config = config or HarnessConfig.from_yaml()
        self.client = client or OpenAICompatibleChatClient(self.config.base_url, self.config.api_key)

    async def evaluate_root(
        self,
        task: str,
        ground_truth: str,
        agent_answer: str,
        tree_summary: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        content = "\n".join(
            [
                "Task:",
                task,
                "",
                "Ground truth answer:",
                ground_truth,
                "",
                "Agent answer:",
                agent_answer,
                "",
                "Execution tree summary:",
                json.dumps(tree_summary or {}, ensure_ascii=False),
            ]
        )
        return await self._judge(ROOT_TASK_JUDGE_PROMPT, content, "root")

    async def evaluate_subtask(
        self,
        parent_task: str,
        subtask: str,
        agent_answer: str,
        ground_truth: str | None = None,
        child_tasks: list[str] | None = None,
    ) -> EvaluationResult:
        content = "\n".join(
            [
                "Parent task:",
                parent_task,
                "",
                "Sub-task:",
                subtask,
                "",
                "Ground truth answer:",
                ground_truth or "",
                "",
                "Agent answer:",
                agent_answer,
                "",
                "Delegated child tasks:",
                json.dumps(child_tasks or [], ensure_ascii=False),
            ]
        )
        return await self._judge(SUBTASK_JUDGE_PROMPT, content, "subtask")

    async def evaluate_tree(self, tree: ExecutionTree, ground_truth_by_node: dict[str, str]) -> dict[str, EvaluationResult]:
        results: dict[str, EvaluationResult] = {}
        for node_id, ground_truth in ground_truth_by_node.items():
            node = tree.nodes[node_id]
            if node.parent_id is None:
                result = await self.evaluate_root(
                    task=node.task,
                    ground_truth=ground_truth,
                    agent_answer=node.final_answer or "",
                    tree_summary=tree.summary(),
                )
            else:
                parent = tree.nodes.get(node.parent_id)
                result = await self.evaluate_subtask(
                    parent_task=parent.task if parent else "",
                    subtask=node.task,
                    ground_truth=ground_truth,
                    agent_answer=node.final_answer or "",
                    child_tasks=[tree.nodes[child_id].task for child_id in node.children_ids if child_id in tree.nodes],
                )
            results[node_id] = result
            node.metadata["llm_judge"] = result.model_dump(mode="json")
            node.metadata["success_signal"] = 1.0 if result.success else 0.0
        return results

    async def _judge(self, prompt: str, content: str, scope: Literal["root", "subtask"]) -> EvaluationResult:
        completion = await self.client.complete(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            model=self.config.model_name,
            temperature=0,
            max_tokens=self.config.max_tokens,
        )
        parsed = json.loads(completion.content)
        if not isinstance(parsed, dict):
            raise ValueError("judge response must be a JSON object")
        success = parsed.get("success")
        if not isinstance(success, bool):
            raise ValueError("judge JSON field 'success' must be boolean")
        reason = parsed.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("judge JSON field 'reason' must be a non-empty string")
        return EvaluationResult(
            reason=reason.strip(),
            success=success,
            scope=scope,
            metadata={"raw_response": completion.content},
        )
