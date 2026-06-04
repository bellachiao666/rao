"""Root runner for recursive agent execution."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from recursive_agent_harness.agent import GlobalBudget, RecursiveAgent
from recursive_agent_harness.config import HarnessConfig
from recursive_agent_harness.policy import Policy
from recursive_agent_harness.state import NodeResult, NodeStatus, utc_now_iso
from recursive_agent_harness.tools import ToolRegistry
from recursive_agent_harness.tree import ExecutionTree


class Runner:
    def __init__(self, policy: Policy, config: HarnessConfig | None = None, tools: ToolRegistry | None = None):
        self.policy = policy
        self.config = config or HarnessConfig.from_yaml()
        self.tools = tools or ToolRegistry.with_default_tools()

    async def arun(self, task: str) -> tuple[NodeResult, ExecutionTree]:
        run_id = f"run_{utc_now_iso().replace(':', '').replace('.', '_')}"
        tree = ExecutionTree(run_id=run_id, config=self.config.model_dump(mode="json"))
        budget = GlobalBudget(total_node_limit=self.config.total_node_limit)
        root_node_id = budget.allocate_node_id_or_raise()
        root = RecursiveAgent(
            node_id=root_node_id,
            parent_id=None,
            task=task,
            depth=0,
            max_depth=self.config.max_depth,
            policy=self.policy,
            tools=self.tools,
            tree=tree,
            budget=budget,
            config=self.config,
            parent_context={},
            expected_output="Final answer for the original user task.",
            constraints={},
        )
        try:
            result = await asyncio.wait_for(root.run(), timeout=self.config.timeout_seconds)
        except asyncio.TimeoutError:
            tree.mark_unfinished(NodeStatus.TIMEOUT, "Run timed out.")
            result = NodeResult.timeout(node_id=root_node_id, task=task)
        except asyncio.CancelledError:
            tree.mark_unfinished(NodeStatus.CANCELLED, "Run cancelled.")
            raise
        self._export_trace(tree)
        return result, tree

    def run(self, task: str) -> tuple[NodeResult, ExecutionTree]:
        return asyncio.run(self.arun(task))

    async def arun_batch(self, tasks: list[str], output_dir: str | Path) -> dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        runs: list[dict[str, Any]] = []
        for index, task in enumerate(tasks, start=1):
            stem = f"trace_{index:04d}"
            config = self.config.model_copy(
                update={
                    "trace_output_path": str(output / f"{stem}.json"),
                    "mermaid_output_path": str(output / f"{stem}.mmd"),
                }
            )
            runner = Runner(policy=self.policy, config=config, tools=self.tools)
            result, tree = await runner.arun(task)
            summary = tree.summary()
            runs.append(
                {
                    "task": task,
                    "run_id": tree.run_id,
                    "trace_path": config.trace_output_path,
                    "mermaid_path": config.mermaid_output_path,
                    "status": result.status.value,
                    "summary": summary,
                }
            )
        manifest = {"total_runs": len(runs), "runs": runs}
        (output / "batch_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest

    def _export_trace(self, tree: ExecutionTree) -> None:
        tree.export_json(self.config.trace_output_path)
        if self.config.mermaid_output_path:
            tree.export_mermaid(self.config.mermaid_output_path)
