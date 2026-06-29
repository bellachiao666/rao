"""Hugging Face CodeAct policy with persistent per-node REPL state."""

from __future__ import annotations

from recursive_agent_harness.actions import AgentAction
from recursive_agent_harness.state import AgentState
from recursive_agent_training.codeact.environment import CodeActEnvironment
from recursive_agent_training.codeact.parser import CodeActAction, parse_codeact
from recursive_agent_training.codeact.prompts import (
    CODEACT_SYSTEM_PROMPT,
    render_codeact_repair,
    render_codeact_state,
)
from recursive_agent_training.hf_policy import HuggingFaceTrainingPolicy
from recursive_agent_training.hf_policy import adapt_action_to_state
from recursive_agent_training.snapshots import TRAINING_TRACE_KEY


class CodeActPolicyAdapter:
    def __init__(self, policy_version: str):
        self.policy_version = policy_version

    def parse(self, content: str) -> CodeActAction:
        return parse_codeact(content)


class HuggingFaceCodeActTrainingPolicy(HuggingFaceTrainingPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.environments: dict[str, CodeActEnvironment] = {}

    async def act(self, state: AgentState) -> AgentAction:
        environment = self.environments.setdefault(state.node_id, CodeActEnvironment())
        if state.trajectory:
            latest = state.trajectory[-1]
            environment.update_observation(latest.observation)
        messages = [
            {"role": "system", "content": CODEACT_SYSTEM_PROMPT},
            {"role": "user", "content": render_codeact_state(state)},
        ]
        generated = await self._generate_async(messages)
        try:
            parsed = parse_codeact(generated["generated_text"])
            action = environment.execute(parsed.code)
        except Exception as first_error:
            repaired = await self._generate_async(
                [
                    {"role": "system", "content": CODEACT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": render_codeact_repair(
                            generated["generated_text"],
                            str(first_error),
                        ),
                    },
                ]
            )
            try:
                parsed = parse_codeact(repaired["generated_text"])
                action = environment.execute(parsed.code)
            except Exception as repair_error:
                raise ValueError(
                    "CodeAct parse or execution failed after repair: "
                    f"{repair_error}; original={generated['generated_text'][:500]!r}; "
                    f"repair={repaired['generated_text'][:500]!r}"
                ) from repair_error
            repaired["original_raw_response"] = generated["generated_text"]
            repaired["parse_error"] = str(first_error)
            repaired["repaired"] = True
            generated = repaired
        action = adapt_action_to_state(action, state)
        action.metadata.update(
            {
                "raw_response": generated["generated_text"],
                "codeact": True,
                TRAINING_TRACE_KEY: generated,
            }
        )
        if generated.get("repaired"):
            action.metadata.update(
                {
                    "original_raw_response": generated["original_raw_response"],
                    "parse_error": generated["parse_error"],
                    "repaired": True,
                }
            )
        return action
