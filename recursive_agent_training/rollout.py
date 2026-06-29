"""Conversion from inference execution trees to training rollout records."""

from __future__ import annotations

from recursive_agent_harness.state import NodeStatus
from recursive_agent_harness.tree import ExecutionTree
from recursive_agent_training.schemas import (
    ModelInputRecord,
    PolicyTurnRecord,
    RolloutTreeRecord,
    TokenRecord,
    TrainingNodeRecord,
    TreeStatus,
)
from recursive_agent_training.snapshots import TRAINING_TRACE_KEY, PolicySnapshot


TRAINING_COMPLETE_STATUSES = {
    NodeStatus.COMPLETED,
    NodeStatus.STEP_LIMITED,
}


def tree_to_rollout_record(
    tree: ExecutionTree,
    task_id: str,
    group_id: str,
    rollout_id: str,
    rollout_index: int,
    seed: int,
    snapshot: PolicySnapshot,
) -> RolloutTreeRecord:
    nodes: dict[str, TrainingNodeRecord] = {}
    complete = True
    for node_id, trace_node in tree.nodes.items():
        traces: list[dict] = []
        for step in trace_node.trajectory:
            if step.action:
                metadata = step.action.get("metadata") or {}
                training_trace = metadata.get(TRAINING_TRACE_KEY)
                if training_trace:
                    traces.append(training_trace)
        first = traces[0] if traces else {}
        tool_call_count = sum(
            bool(step.action and step.action.get("type") == "TOOL_CALL")
            for step in trace_node.trajectory
        )
        tool_error_count = sum(
            bool(
                step.observation
                and step.observation.get("tool_name")
                and step.observation.get("ok") is False
            )
            for step in trace_node.trajectory
        )
        turns = [
            PolicyTurnRecord(
                turn_index=index,
                model_input=ModelInputRecord(
                    messages=trace.get("messages", []),
                    rendered_prompt=trace.get("rendered_prompt", ""),
                    template_version=trace.get("template_version", snapshot.template_version),
                    tokenizer_name=trace.get("model_name", snapshot.model_name),
                    tokenizer_revision=trace.get("tokenizer_revision", snapshot.tokenizer_revision),
                ),
                tokens=TokenRecord(
                    input_ids=trace.get("input_ids", []),
                    generated_ids=trace.get("generated_ids", []),
                    action_mask=trace.get("action_mask", []),
                    behavior_logprobs=trace.get("behavior_logprobs", []),
                    generated_text=trace.get("generated_text", ""),
                    truncated=bool(trace.get("action_truncated", False)),
                ),
                sampling_temperature=float(trace.get("sampling_temperature", 1.0)),
                sampling_seed=trace.get("sampling_seed"),
            )
            for index, trace in enumerate(traces)
        ]
        terminal_reason = str(trace_node.metadata.get("terminal_reason") or trace_node.status.value)
        is_fallback = bool(trace_node.metadata.get("is_fallback"))
        if trace_node.status not in TRAINING_COMPLETE_STATUSES:
            complete = False
        nodes[node_id] = TrainingNodeRecord(
            task_id=task_id,
            group_id=group_id,
            rollout_id=rollout_id,
            node_id=node_id,
            parent_id=trace_node.parent_id,
            depth=trace_node.depth,
            node_task=trace_node.task,
            children_ids=list(trace_node.children_ids),
            terminal_status=trace_node.status.value,
            terminal_reason=terminal_reason,
            final_answer=trace_node.final_answer or "",
            is_fallback=is_fallback,
            is_trainable=bool(traces) and not is_fallback,
            model_input=ModelInputRecord(
                messages=first.get("messages", []),
                rendered_prompt=first.get("rendered_prompt", ""),
                template_version=first.get("template_version", snapshot.template_version),
                tokenizer_name=first.get("model_name", snapshot.model_name),
                tokenizer_revision=first.get("tokenizer_revision", snapshot.tokenizer_revision),
            ),
            tokens=TokenRecord(
                input_ids=first.get("input_ids", []),
                generated_ids=first.get("generated_ids", []),
                action_mask=first.get("action_mask", []),
                behavior_logprobs=first.get("behavior_logprobs", []),
                generated_text=first.get("generated_text", ""),
                truncated=bool(first.get("action_truncated", False)),
            ),
            turns=turns,
            behavior_policy_version=snapshot.version,
            metadata={
                "trace_metadata": trace_node.metadata,
                "prompt_truncated": any(bool(trace.get("prompt_truncated")) for trace in traces),
                "tool_call_count": tool_call_count,
                "tool_error_count": tool_error_count,
                "start_time": trace_node.start_time,
                "end_time": trace_node.end_time,
            },
        )
    record = RolloutTreeRecord(
        rollout_id=rollout_id,
        group_id=group_id,
        rollout_index=rollout_index,
        seed=seed,
        root_node_id=tree.root_node_id or "",
        behavior_policy_version=snapshot.version,
        behavior_checkpoint_path=snapshot.checkpoint_path,
        behavior_checkpoint_hash=snapshot.checkpoint_hash,
        prompt_template_version=snapshot.template_version,
        tool_schema_version="harness_tools_v1",
        environment_version="recursive_agent_harness_v1",
        nodes=nodes,
        tree_status=TreeStatus.COMPLETE if complete else TreeStatus.INCOMPLETE,
    )
    return record.finalize()
