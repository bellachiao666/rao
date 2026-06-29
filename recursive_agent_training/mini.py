"""Deterministic recursive task used for end-to-end RAO tests."""

from __future__ import annotations

from types import SimpleNamespace

from recursive_agent_harness.actions import ActionType, AgentAction, SubtaskSpec
from recursive_agent_harness.config import HarnessConfig
from recursive_agent_harness.state import AgentState
from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.schemas import RootTaskRecord
from recursive_agent_training.snapshots import PolicySnapshot, ScriptedTrainingPolicy


def make_harness_config(config: TrainingConfig, output_dir: str) -> HarnessConfig:
    return HarnessConfig(
        max_depth=config.rollout.max_depth,
        max_steps_per_node=config.rollout.max_steps_per_node,
        max_children_per_node=2,
        total_node_limit=16,
        timeout_seconds=config.rollout.timeout_seconds,
        invalid_action_limit=2,
        max_tool_calls_per_node=2,
        max_parallel_children=2,
        model_name=config.model.name,
        base_url="",
        api_key=None,
        temperature=config.rollout.temperature,
        max_tokens=config.rollout.max_tokens,
        trace_output_path=f"{output_dir}/trace.json",
        mermaid_output_path="",
        log_level="INFO",
    )


def synthetic_task(answer: str = "alpha") -> RootTaskRecord:
    return RootTaskRecord(
        task_id="synthetic:alpha",
        domain="synthetic",
        split="train",
        prompt="Return the token alpha. Delegate one verification subtask first.",
        ground_truth=answer,
        dataset_hash="synthetic",
    )


def make_scripted_policy(snapshot: PolicySnapshot, answer: str) -> ScriptedTrainingPolicy:
    def action(state: AgentState) -> AgentAction:
        if state.depth == 0 and not state.child_summaries:
            return AgentAction(
                type=ActionType.LAUNCH_SUBAGENT,
                reason="verify the token",
                subtask=SubtaskSpec(goal="Return the token alpha.", expected_output="one token"),
            )
        return AgentAction(type=ActionType.FINISH, reason="done", answer=answer)

    return ScriptedTrainingPolicy(snapshot, action)


def build_toy_model(vocab_size: int = 512, hidden_size: int = 32):
    import torch

    class ToyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
            self.output = torch.nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(logits=self.output(self.embedding(input_ids)))

    return ToyCausalLM()
