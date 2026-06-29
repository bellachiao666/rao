"""On-policy Hugging Face generation with exact token-level training traces."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
import hashlib
import json
import threading
from typing import Any

from recursive_agent_harness.actions import ActionType, AgentAction
from recursive_agent_harness.prompts import (
    SYSTEM_PROMPT,
    render_action_schema_prompt,
    render_repair_prompt,
    render_state_prompt,
)
from recursive_agent_harness.state import AgentState
from recursive_agent_training.snapshots import TRAINING_TRACE_KEY, PolicySnapshot, TrainingPolicy


def extract_json_object(text: str) -> str:
    """Extract the first complete JSON object from optional model chatter."""

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return text[index : index + end]
    raise ValueError("model output does not contain a complete JSON object")


def parse_generated_action(text: str) -> AgentAction:
    payload = json.loads(extract_json_object(text))
    normalizations: list[str] = []
    action_type = str(payload.get("type") or "").upper()
    if action_type in {"SEARCH_WEB", "WEB_SEARCH"}:
        payload["type"] = "TOOL_CALL"
        payload["tool_name"] = "search_web"
        payload["arguments"] = {
            **dict(payload.get("arguments") or {}),
            **({"query": payload["query"]} if payload.get("query") else {}),
        }
        action_type = "TOOL_CALL"
        normalizations.append("search_alias_to_tool_call")
    elif action_type in {"READ_WEBPAGE", "VIEW_WEBPAGE_CONTENT"}:
        payload["type"] = "TOOL_CALL"
        payload["tool_name"] = "view_webpage_content"
        payload["arguments"] = {
            **dict(payload.get("arguments") or {}),
            **({"url": payload["url"]} if payload.get("url") else {}),
        }
        action_type = "TOOL_CALL"
        normalizations.append("webpage_alias_to_tool_call")
    if payload.get("tool_name") and action_type != "TOOL_CALL":
        payload["type"] = "TOOL_CALL"
        action_type = "TOOL_CALL"
        normalizations.append("tool_payload_forced_tool_call")
    if action_type == "TOOL_CALL" and not payload.get("tool_name"):
        reason = str(payload.get("reason") or "").strip()
        if reason in {"search_web", "view_webpage_content"}:
            payload["tool_name"] = reason
            normalizations.append("reason_promoted_to_tool_name")
        elif isinstance(payload.get("subtask"), dict) and payload["subtask"].get("goal"):
            payload["tool_name"] = "search_web"
            payload["arguments"] = {
                **dict(payload.get("arguments") or {}),
                "query": str(payload["subtask"]["goal"]),
            }
            normalizations.append("subtask_goal_promoted_to_search")
    if action_type == "THINK" and not payload.get("thought") and payload.get("reason"):
        payload["thought"] = payload["reason"]
    if action_type == "AGGREGATE" and not payload.get("instructions") and payload.get("reason"):
        payload["instructions"] = payload["reason"]
    action = AgentAction.model_validate(payload)
    if normalizations:
        action.metadata["parser_normalizations"] = normalizations
    return action


def adapt_action_to_state(action: AgentAction, state: AgentState) -> AgentAction:
    if (
        action.type == ActionType.TOOL_CALL
        and action.tool_name == "search_web"
        and action.arguments.get("query")
    ):
        query = str(action.arguments["query"]).strip()
        task = state.task.strip()
        if (
            task
            and len(query.split()) < 5
            and task.casefold() not in query.casefold()
        ):
            action.arguments["query"] = f"{query}\nTask context: {task[-2000:]}"
            action.metadata["search_query_augmented_with_task"] = True
    return action


def validate_input_token_ids(input_ids: Any, model: Any, *, source: str) -> None:
    """Fail on invalid token ids before CUDA turns the error asynchronous."""

    if input_ids.numel() == 0:
        raise ValueError(f"{source} input_ids must not be empty")
    embeddings = model.get_input_embeddings()
    vocab_size = int(embeddings.num_embeddings)
    minimum = int(input_ids.min().item())
    maximum = int(input_ids.max().item())
    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(
            f"{source} token ids out of range: min={minimum}, max={maximum}, "
            f"embedding_vocab_size={vocab_size}"
        )


class HuggingFaceTrainingPolicy(TrainingPolicy):
    """Generate recursive actions from a local causal LM and retain behavior logprobs."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        snapshot: PolicySnapshot,
        *,
        temperature: float,
        max_new_tokens: int,
        seed: int,
        max_input_tokens: int | None = None,
        threaded_generation: bool = False,
        generation_lock: threading.Lock | None = None,
    ):
        if temperature <= 0:
            raise ValueError("on-policy training requires a positive sampling temperature")
        self.model = model
        self.tokenizer = tokenizer
        self.snapshot = snapshot
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.seed = seed
        self.threaded_generation = threaded_generation
        self.generation_lock = generation_lock
        self._generation_index = 0

    async def act(self, state: AgentState) -> AgentAction:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_action_schema_prompt()},
            {"role": "user", "content": render_state_prompt(state)},
        ]
        generated = await self._generate_async(messages)
        try:
            action = parse_generated_action(generated["generated_text"])
        except Exception as first_error:
            repair_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_action_schema_prompt()},
                {
                    "role": "user",
                    "content": render_repair_prompt(generated["generated_text"], str(first_error)),
                },
            ]
            repaired = await self._generate_async(repair_messages)
            try:
                action = parse_generated_action(repaired["generated_text"])
            except Exception as repair_error:
                raise ValueError(f"model action parse failed after repair: {repair_error}") from repair_error
            repaired["original_raw_response"] = generated["generated_text"]
            repaired["parse_error"] = str(first_error)
            repaired["repaired"] = True
            generated = repaired

        action = adapt_action_to_state(action, state)
        action.metadata.update(
            {
                "raw_response": generated["generated_text"],
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

    async def _generate_async(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if not self.threaded_generation:
            return self._generate(messages)
        return await asyncio.to_thread(self._generate_with_lock, messages)

    def _generate_with_lock(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if self.generation_lock is None:
            return self._generate(messages)
        with self.generation_lock:
            return self._generate(messages)

    def _generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        import torch

        generation_seed = self.seed + self._generation_index
        self._generation_index += 1

        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self.tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        prompt_truncated = False
        if self.max_input_tokens is not None:
            if self.max_input_tokens <= 0:
                raise ValueError("max_input_tokens must be positive")
            raw_ids = encoded["input_ids"][0]
            if raw_ids.shape[0] > self.max_input_tokens:
                prompt_truncated = True
                prefix = min(512, max(self.max_input_tokens // 4, 1))
                tail = self.max_input_tokens - prefix
                shortened = torch.cat((raw_ids[:prefix], raw_ids[-tail:])) if tail else raw_ids[:prefix]
                encoded["input_ids"] = shortened.unsqueeze(0)
                encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        device = next(self.model.parameters()).device
        validate_input_token_ids(
            encoded["input_ids"],
            self.model,
            source="policy",
        )
        device_context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with device_context:
            if device.type == "cuda":
                torch.cuda.manual_seed(generation_seed)
            else:
                torch.manual_seed(generation_seed)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            input_ids = encoded["input_ids"][0]
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id

            self.model.eval()
            with torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=1.0,
                    top_k=0,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True,
                )
            generated_ids_tensor = output.sequences[0, input_ids.shape[0] :]
            generated_ids = generated_ids_tensor.tolist()
            if not generated_ids:
                raise ValueError("model generation returned no tokens")
            if len(output.scores) != len(generated_ids):
                raise ValueError("generation scores do not align with generated tokens")
            behavior_logprobs = [
                float(torch.log_softmax(score[0].float(), dim=-1)[token_id].cpu())
                for score, token_id in zip(output.scores, generated_ids, strict=True)
            ]
            input_id_values = input_ids.tolist()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        eos_token_id = self.tokenizer.eos_token_id
        action_truncated = (
            len(generated_ids) >= self.max_new_tokens
            and (eos_token_id is None or generated_ids[-1] != eos_token_id)
        )
        sampling_payload = {
            "temperature": self.temperature,
            "top_p": 1.0,
            "top_k": 0,
            "max_new_tokens": self.max_new_tokens,
            "seed": generation_seed,
        }
        return {
            "messages": messages,
            "rendered_prompt": rendered_prompt,
            "input_ids": input_id_values,
            "generated_ids": generated_ids,
            "generated_text": generated_text,
            "action_mask": [1] * len(generated_ids),
            "behavior_logprobs": behavior_logprobs,
            "prompt_truncated": prompt_truncated,
            "action_truncated": action_truncated,
            "policy_version": self.snapshot.version,
            "model_name": self.snapshot.model_name,
            "tokenizer_revision": self.snapshot.tokenizer_revision,
            "template_version": self.snapshot.template_version,
            "sampling_temperature": self.temperature,
            "sampling_seed": generation_seed,
            "sampling_metadata": sampling_payload,
            "sampling_hash": hashlib.sha256(
                json.dumps(sampling_payload, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
