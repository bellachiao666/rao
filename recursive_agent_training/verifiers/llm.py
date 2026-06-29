"""LLM judge adapter with strict success-signal provenance."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
import json
import re
import threading
import time

from recursive_agent_harness.evaluator import LLMJudge
from recursive_agent_training.config import JudgeServiceConfig
from recursive_agent_training.hf_policy import extract_json_object, validate_input_token_ids
from recursive_agent_training.schemas import RootTaskRecord, RolloutTreeRecord, SuccessSignalResult, TrainingNodeRecord, VerifierStatus
from recursive_agent_training.verifiers.base import SuccessSignalProvider
from recursive_agent_training.verifiers.exact import normalize_answer


LOCAL_JUDGE_SYSTEM_PROMPT = """You are a strict research-answer judge.
Compare the agent answer with the reference answer for the given task.
Return exactly one JSON object and no other text:
{"reason":"brief factual reason","success":true}
Set success true only when the agent answer identifies the same answer as the reference.
Ignore harmless capitalization, punctuation, and explanatory wording.
Do not reward an answer that merely discusses how to search."""

LOCAL_SUBTASK_JUDGE_SYSTEM_PROMPT = """You are a strict research subtask judge.
Return exactly one JSON object and no other text:
{"reason":"brief factual reason","success":true}
Set success true only when the answer provides concrete information that materially resolves
the delegated subtask. Reject restatements of the subtask, generic search advice, resource
names, placeholders, claims without a substantive result, and reports that no result was found
because of search quality, depth limits, tool limits, time limits, or a need for more research."""


def root_answer_passes_lexical_guard(reference: str, answer: str) -> bool:
    normalized_reference = normalize_answer(reference)
    normalized_answer = normalize_answer(answer)
    if not normalized_reference:
        return False
    if normalized_reference in normalized_answer:
        return True
    reference_tokens = set(re.findall(r"\w+", reference.casefold()))
    answer_tokens = set(re.findall(r"\w+", answer.casefold()))
    coverage = (
        len(reference_tokens & answer_tokens) / len(reference_tokens)
        if reference_tokens
        else 0.0
    )
    return coverage >= 0.9


def _subtask_requires_root_reference(subtask: str) -> bool:
    normalized = " ".join(re.findall(r"\w+", subtask.casefold()))
    answer_target_patterns = (
        r"\b(?:exact|precise|matching|candidate|potential)\s+"
        r"(?:paper\s+)?(?:title|name|answer)\b",
        r"\b(?:identify|find|determine|locate|provide|return|verify|search for)\b"
        r".{0,80}\b(?:paper\s+)?title\b",
        r"\b(?:identify|find|determine|locate|provide|return|verify)\b"
        r".{0,80}\bsurname\b",
        r"\b(?:answer|solve)\b.{0,40}\b(?:original|parent|root)\s+"
        r"(?:task|question)\b",
    )
    return any(
        re.search(pattern, normalized)
        for pattern in answer_target_patterns
    )


def subtask_answer_passes_minimum_guard(
    answer: str,
    subtask: str = "",
    root_reference: str = "",
) -> bool:
    tokens = re.findall(r"\w+", answer.casefold())
    if len(tokens) < 4 or len(answer.strip()) < 20:
        return False
    normalized = " ".join(tokens)
    no_result_patterns = (
        r"\b(?:could not|couldn't|did not|didn't|unable to|failed to)\s+"
        r"(?:find|identify|determine|locate|verify|confirm)\b",
        r"\b(?:cannot|can't|unable to|failed to)\s+"
        r"(?:launch|start|create|spawn|delegate)\b",
        r"\bno\s+(?:clear\s+|specific\s+|relevant\s+)?"
        r"(?:answer|match|matching title|result|results|information|evidence)\b",
        r"\bnone of (?:the )?(?:search )?results\b",
        r"\bno (?:child|subagent|agent) (?:returned|provided|produced)\b"
        r".{0,40}\b(?:successful|useful|valid|concrete)?\s*answer\b",
        r"\bno successful answer\b",
        r"\ball (?:child|subagent) (?:tasks?|subtasks?)\b"
        r".{0,30}\b(?:failed|unsuccessful)\b",
        r"\bmaximum depth (?:was )?reached\b",
        r"\bmax[_ ]depth (?:has |was )?(?:been )?(?:reached|exceeded)\b",
        r"\bdepth limit (?:was )?reached\b",
        r"\bnot enough (?:information|evidence)\b",
        r"\b(?:further|additional|more|deeper)\s+"
        r"(?:research|search|investigation).{0,40}\b(?:required|needed)\b",
    )
    if any(re.search(pattern, normalized) for pattern in no_result_patterns):
        return False
    process_limit_markers = (
        "search result",
        "search results",
        "depth limit",
        "maximum depth",
        "allowed depth",
        "tool limit",
        "time limit",
        "token limit",
        "within the allowed",
    )
    negative_markers = (
        " no ",
        " none ",
        " not ",
        " unable ",
        " failed ",
        " without ",
        " before ",
        " required ",
        " needed ",
    )
    padded = f" {normalized} "
    if (
        any(marker in normalized for marker in process_limit_markers)
        and any(marker in padded for marker in negative_markers)
    ):
        return False
    generic = {
        "google scholar",
        "search engine",
        "search the web",
        "i do not know",
        "unable to find",
        "no answer found",
    }
    if normalized in generic:
        return False
    subtask_tokens = re.findall(r"\w+", subtask.casefold())
    if subtask and (len(subtask_tokens) < 4 or len(subtask.strip()) < 20):
        return False
    normalized_subtask = " ".join(subtask_tokens)
    if normalized_subtask and normalized in normalized_subtask:
        return False
    generic_tokens = {
        "a",
        "an",
        "and",
        "answer",
        "complete",
        "completed",
        "current",
        "delegated",
        "done",
        "found",
        "identified",
        "information",
        "is",
        "it",
        "leader",
        "narrow",
        "of",
        "result",
        "resolved",
        "task",
        "the",
        "this",
        "was",
    }
    specific_tokens = set(tokens) - set(subtask_tokens) - generic_tokens
    if not specific_tokens:
        return False
    if root_reference:
        if not _subtask_requires_root_reference(subtask):
            return False
        if not root_answer_passes_lexical_guard(root_reference, answer):
            return False
    return True


def _fallback_result(provider: str, provider_version: str) -> SuccessSignalResult:
    return SuccessSignalResult(
        value=0.0,
        provider=provider,
        provider_version=provider_version,
        reason="fallback trajectories cannot succeed",
        status=VerifierStatus.COMPLETE,
        attempt_count=0,
    )


class LLMJudgeVerifier(SuccessSignalProvider):
    def __init__(self, judge: LLMJudge, provider_version: str):
        self.judge = judge
        self.provider_version = provider_version

    async def evaluate(
        self,
        node: TrainingNodeRecord,
        tree: RolloutTreeRecord,
        root_task: RootTaskRecord,
    ) -> SuccessSignalResult:
        if node.is_fallback:
            return _fallback_result("llm_judge", self.provider_version)
        if node.node_id == tree.root_node_id:
            passes_guard = root_answer_passes_lexical_guard(
                root_task.ground_truth or "",
                node.final_answer,
            )
            guard_reason = "lexical guard: the reference answer is absent from the agent answer"
        else:
            passes_guard = subtask_answer_passes_minimum_guard(
                node.final_answer,
                node.node_task,
                root_task.ground_truth or "",
            )
            guard_reason = "subtask guard: answer is non-substantive or inconsistent with the root reference"
        if not passes_guard:
            return SuccessSignalResult(
                value=0.0,
                provider="llm_judge",
                provider_version=self.provider_version,
                reason=guard_reason,
                status=VerifierStatus.COMPLETE,
                attempt_count=0,
            )
        try:
            if node.node_id == tree.root_node_id:
                result = await self.judge.evaluate_root(
                    root_task.prompt,
                    root_task.ground_truth or "",
                    node.final_answer,
                    {"total_nodes": len(tree.nodes)},
                )
            else:
                parent = tree.nodes.get(node.parent_id or "")
                result = await self.judge.evaluate_subtask(
                    parent_task=parent.node_task if parent else root_task.prompt,
                    subtask=node.node_task,
                    agent_answer=node.final_answer,
                    child_tasks=[tree.nodes[item].node_task for item in node.children_ids if item in tree.nodes],
                )
            return SuccessSignalResult(
                value=1.0 if result.success else 0.0,
                provider="llm_judge",
                provider_version=self.provider_version,
                reason=result.reason,
                raw_output=result.metadata.get("raw_response"),
                status=VerifierStatus.COMPLETE,
                attempt_count=1,
            )
        except Exception as exc:
            return SuccessSignalResult(
                value=None,
                provider="llm_judge",
                provider_version=self.provider_version,
                reason="judge failed",
                error=str(exc),
                status=VerifierStatus.ERROR,
                attempt_count=1,
            )


class LocalTransformersJudgeVerifier(SuccessSignalProvider):
    """Deterministic local judge backed by a pinned ModelScope model snapshot."""

    def __init__(self, settings: JudgeServiceConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not settings.model_path:
            raise ValueError("local judge requires model_path")
        self.settings = settings
        self.provider_version = settings.provider_version or (
            f"{settings.model_id}@{settings.revision}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_path,
            local_files_only=True,
        )
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[settings.dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_path,
            local_files_only=True,
            torch_dtype=dtype,
            device_map={"": settings.device},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self._lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "terminal_failures": 0,
            "retries": 0,
            "lexical_guard_rejections": 0,
            "latency_seconds": 0.0,
            "estimated_cost": 0.0,
        }

    async def evaluate(
        self,
        node: TrainingNodeRecord,
        tree: RolloutTreeRecord,
        root_task: RootTaskRecord,
    ) -> SuccessSignalResult:
        if node.is_fallback:
            return _fallback_result("local_modelscope_judge", self.provider_version)
        if node.node_id == tree.root_node_id:
            if not root_answer_passes_lexical_guard(
                root_task.ground_truth or "",
                node.final_answer,
            ):
                self._add_metrics(lexical_guard_rejections=1)
                return SuccessSignalResult(
                    value=0.0,
                    provider="local_modelscope_judge",
                    provider_version=self.provider_version,
                    reason="lexical guard: the reference answer is absent from the agent answer",
                    status=VerifierStatus.COMPLETE,
                    attempt_count=0,
                )
            content = "\n".join(
                [
                    "Task:",
                    root_task.prompt,
                    "",
                    "Reference answer:",
                    root_task.ground_truth or "",
                    "",
                    "Agent answer:",
                    node.final_answer,
                ]
            )
            system_prompt = LOCAL_JUDGE_SYSTEM_PROMPT
        else:
            if not subtask_answer_passes_minimum_guard(
                node.final_answer,
                node.node_task,
                root_task.ground_truth or "",
            ):
                self._add_metrics(lexical_guard_rejections=1)
                return SuccessSignalResult(
                    value=0.0,
                    provider="local_modelscope_judge",
                    provider_version=self.provider_version,
                    reason=(
                        "subtask guard: answer is non-substantive or inconsistent "
                        "with the root reference"
                    ),
                    status=VerifierStatus.COMPLETE,
                    attempt_count=0,
                )
            content = "\n".join(
                [
                    "Parent task:",
                    tree.nodes[node.parent_id].node_task if node.parent_id in tree.nodes else root_task.prompt,
                    "",
                    "Delegated subtask:",
                    node.node_task,
                    "",
                    "Agent answer:",
                    node.final_answer,
                    "",
                    "Judge whether this is a concrete, useful answer to the delegated subtask.",
                ]
            )
            system_prompt = LOCAL_SUBTASK_JUDGE_SYSTEM_PROMPT
        return await asyncio.to_thread(self._evaluate_sync, content, system_prompt)

    def _evaluate_sync(
        self,
        content: str,
        system_prompt: str = LOCAL_JUDGE_SYSTEM_PROMPT,
    ) -> SuccessSignalResult:
        outputs: list[str] = []
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            started = time.monotonic()
            self._add_metrics(requests=1)
            try:
                prompt = content
                if outputs:
                    prompt += (
                        "\n\nYour previous response was invalid JSON. Return only the required JSON object.\n"
                        + outputs[-1]
                    )
                raw = self._generate(prompt, system_prompt=system_prompt)
                outputs.append(raw)
                parsed = json.loads(extract_json_object(raw))
                if not isinstance(parsed.get("success"), bool):
                    raise ValueError("judge JSON field 'success' must be boolean")
                reason = parsed.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("judge JSON field 'reason' must be a non-empty string")
                self._add_metrics(
                    successes=1,
                    latency_seconds=time.monotonic() - started,
                )
                return SuccessSignalResult(
                    value=1.0 if parsed["success"] else 0.0,
                    provider="local_modelscope_judge",
                    provider_version=self.provider_version,
                    reason=reason.strip(),
                    raw_output=raw,
                    status=VerifierStatus.COMPLETE,
                    attempt_count=attempt + 1,
                )
            except Exception as exc:
                last_error = exc
                self._add_metrics(
                    failures=1,
                    latency_seconds=time.monotonic() - started,
                )
                if attempt < self.settings.max_retries:
                    self._add_metrics(retries=1)
        self._add_metrics(terminal_failures=1)
        return SuccessSignalResult(
            value=None,
            provider="local_modelscope_judge",
            provider_version=self.provider_version,
            reason="judge failed",
            raw_output=outputs[-1] if outputs else None,
            error=str(last_error),
            status=VerifierStatus.ERROR,
            attempt_count=self.settings.max_retries + 1,
        )

    def _generate(
        self,
        content: str,
        *,
        system_prompt: str = LOCAL_JUDGE_SYSTEM_PROMPT,
    ) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        device = next(self.model.parameters()).device
        validate_input_token_ids(
            encoded["input_ids"],
            self.model,
            source="judge",
        )
        device_context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with self._lock, device_context, torch.inference_mode():
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.settings.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True,
            )
            new_ids = generated[0, encoded["input_ids"].shape[1] :].tolist()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def close(self) -> None:
        import gc
        import torch

        if self.model is not None and torch.cuda.is_available():
            device = next(self.model.parameters()).device
            if device.type == "cuda":
                try:
                    torch.cuda.synchronize(device)
                except RuntimeError:
                    pass
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass

    def _add_metrics(self, **values: float) -> None:
        with self._metrics_lock:
            for key, value in values.items():
                self._metrics[key] += value

    def metrics(self) -> dict[str, float]:
        with self._metrics_lock:
            return dict(self._metrics)


def build_judge_verifier(settings: JudgeServiceConfig) -> SuccessSignalProvider:
    if settings.provider == "local_transformers":
        return LocalTransformersJudgeVerifier(settings)
    raise ValueError(f"unsupported judge provider for DeepDive: {settings.provider}")
