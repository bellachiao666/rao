"""Policy snapshots and deterministic training-trace policies."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from recursive_agent_harness.actions import AgentAction
from recursive_agent_harness.policy import Policy
from recursive_agent_harness.state import AgentState


TRAINING_TRACE_KEY = "_training_trace"


@dataclass(frozen=True)
class PolicySnapshot:
    version: str
    model_name: str
    tokenizer_revision: str = ""
    template_version: str = "existing_harness_v1"
    checkpoint_path: str = ""
    checkpoint_hash: str = ""


class PolicySnapshotProvider:
    def __init__(
        self,
        initial_version: int = 0,
        model_name: str = "scripted",
        checkpoint_path: str = "",
        checkpoint_hash: str = "",
    ):
        self._version = initial_version
        self.model_name = model_name
        self._checkpoint_path = checkpoint_path
        self._checkpoint_hash = checkpoint_hash

    def current(self) -> PolicySnapshot:
        return PolicySnapshot(
            version=f"policy_{self._version:06d}",
            model_name=self.model_name,
            checkpoint_path=self._checkpoint_path,
            checkpoint_hash=self._checkpoint_hash,
        )

    def publish(self, checkpoint_path: str = "", checkpoint_hash: str = "") -> PolicySnapshot:
        self._version += 1
        self._checkpoint_path = checkpoint_path
        self._checkpoint_hash = checkpoint_hash
        return self.current()


class TrainingPolicy(Policy, ABC):
    snapshot: PolicySnapshot
    supports_training_trace = True

    @abstractmethod
    async def act(self, state: AgentState) -> AgentAction:
        raise NotImplementedError


def deterministic_token_ids(text: str) -> list[int]:
    return [byte + 1 for byte in text.encode("utf-8")]


class ScriptedTrainingPolicy(TrainingPolicy):
    """Policy used for deterministic recursive rollout and optimizer tests."""

    def __init__(self, snapshot: PolicySnapshot, action_fn: Callable[[AgentState], AgentAction]):
        self.snapshot = snapshot
        self.action_fn = action_fn

    async def act(self, state: AgentState) -> AgentAction:
        action = self.action_fn(state)
        messages = [
            {"role": "system", "content": "You are a recursive agent node."},
            {"role": "user", "content": state.model_dump_json(exclude={"trajectory"})},
        ]
        prompt = "\n".join(message["content"] for message in messages)
        action_payload = action.model_dump(mode="json", exclude={"metadata"})
        generated_text = json.dumps(action_payload, sort_keys=True, ensure_ascii=False)
        generated_ids = deterministic_token_ids(generated_text)
        trace = {
            "messages": messages,
            "rendered_prompt": prompt,
            "input_ids": deterministic_token_ids(prompt),
            "generated_ids": generated_ids,
            "generated_text": generated_text,
            "action_mask": [1] * len(generated_ids),
            "behavior_logprobs": [-0.1] * len(generated_ids),
            "policy_version": self.snapshot.version,
            "model_name": self.snapshot.model_name,
            "tokenizer_revision": self.snapshot.tokenizer_revision,
            "template_version": self.snapshot.template_version,
            "sampling_hash": hashlib.sha256(generated_text.encode("utf-8")).hexdigest(),
        }
        action.metadata[TRAINING_TRACE_KEY] = trace
        return action
