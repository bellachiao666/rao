"""Policy abstractions and OpenAI-compatible API policy."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from recursive_agent_harness.actions import ActionType, AgentAction
from recursive_agent_harness.config import HarnessConfig
from recursive_agent_harness.prompts import SYSTEM_PROMPT, render_action_schema_prompt, render_repair_prompt, render_state_prompt
from recursive_agent_harness.state import AgentState


class Policy(ABC):
    @abstractmethod
    async def act(self, state: AgentState) -> AgentAction:
        """Return the next action for the current node."""


@dataclass
class ChatCompletionResult:
    content: str
    raw_response: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ChatClient(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict[str, str]], model: str, temperature: float, max_tokens: int) -> ChatCompletionResult:
        """Return a chat completion result."""


class OpenAICompatibleChatClient(ChatClient):
    """Minimal OpenAI-compatible Chat Completions client.

    This intentionally avoids depending on a provider SDK so the harness remains
    provider-agnostic. Tests use fake clients and never call the network.
    """

    def __init__(self, base_url: str, api_key: str | None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def complete(self, messages: list[dict[str, str]], model: str, temperature: float, max_tokens: int) -> ChatCompletionResult:
        import asyncio

        return await asyncio.to_thread(self._complete_sync, messages, model, temperature, max_tokens)

    def _complete_sync(self, messages: list[dict[str, str]], model: str, temperature: float, max_tokens: int) -> ChatCompletionResult:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"chat completion request failed: {exc}") from exc
        content = raw["choices"][0]["message"]["content"]
        return ChatCompletionResult(content=content, raw_response=raw)


class LLMPolicy(Policy):
    def __init__(self, client: ChatClient | None = None, config: HarnessConfig | None = None):
        self.config = config or HarnessConfig.from_yaml()
        self.client = client or OpenAICompatibleChatClient(self.config.base_url, self.config.api_key)

    async def act(self, state: AgentState) -> AgentAction:
        messages = self._messages_for_state(state)
        first = await self.client.complete(
            messages=messages,
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        try:
            action = parse_action_json(first.content)
            action.metadata.update({"raw_response": first.content})
            return action
        except Exception as exc:
            parse_error = str(exc)

        repair_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_action_schema_prompt()},
            {"role": "user", "content": render_repair_prompt(first.content, parse_error)},
        ]
        try:
            repaired = await self.client.complete(
                messages=repair_messages,
                model=self.config.model_name,
                temperature=0,
                max_tokens=self.config.max_tokens,
            )
            action = parse_action_json(repaired.content)
            action.metadata.update(
                {
                    "raw_response": repaired.content,
                    "original_raw_response": first.content,
                    "parse_error": parse_error,
                    "repaired": True,
                }
            )
            return action
        except Exception as repair_exc:
            return _fallback_action(state.task, first.content, f"{parse_error}; repair failed: {repair_exc}")

    def _messages_for_state(self, state: AgentState) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_action_schema_prompt()},
            {"role": "user", "content": render_state_prompt(state)},
        ]


def parse_action_json(content: str) -> AgentAction:
    return AgentAction.model_validate(json.loads(content))


def _fallback_action(task: str, raw_response: str | None, parse_error: str | None) -> AgentAction:
    return AgentAction(
        type=ActionType.FINISH,
        reason="Policy output could not be parsed; using fallback finish.",
        answer=f"Best effort answer for task: {task}",
        limitations=["The model response could not be parsed as a valid AgentAction."],
        metadata={"raw_response": raw_response, "parse_error": parse_error, "fallback": True},
    )
