"""Persistent restricted CodeAct environment that emits harness actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recursive_agent_harness.actions import ActionType, AgentAction, SubtaskSpec
from recursive_agent_training.codeact.sandbox import validate_codeact_ast


@dataclass(frozen=True)
class DeferredSubtask:
    spec: SubtaskSpec


class _AsyncioProxy:
    def __init__(self, environment: "CodeActEnvironment"):
        self.environment = environment

    def gather(self, *items: DeferredSubtask) -> list[DeferredSubtask]:
        return self.environment.gather(*items)


class CodeActEnvironment:
    def __init__(self):
        self.locals: dict[str, Any] = {}
        self.pending_action: AgentAction | None = None
        self.action_call_count = 0
        self._deferred_action_ids: set[int] = set()
        self.asyncio = _AsyncioProxy(self)

    def update_observation(self, observation: Any) -> None:
        self.locals["_last_observation"] = observation

    def execute(self, code: str) -> AgentAction:
        if len(code) > 8_000:
            raise ValueError("CodeAct block exceeds the 8,000 character limit")
        tree = validate_codeact_ast(code)
        self.pending_action = None
        self.action_call_count = 0
        self._deferred_action_ids = set()
        safe_builtins = {
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        globals_dict = {
            "__builtins__": safe_builtins,
            "asyncio": self.asyncio,
            "launch_subagent": self.launch_subagent,
            "launch_subagents_parallel": self.gather,
            "search_web": self.search_web,
            "view_webpage_content": self.view_webpage_content,
            "finish": self.finish,
            "direct_answer": self.direct_answer,
            "think": self.think,
        }
        exec(compile(tree, "<codeact>", "exec"), globals_dict, self.locals)
        if len(self.locals) > 128:
            raise ValueError("CodeAct REPL local-variable limit exceeded")
        for name, value in self.locals.items():
            if len(repr(value)) > 100_000:
                raise ValueError(f"CodeAct local value is too large: {name}")
        if self.pending_action is None or self.action_call_count != 1:
            raise ValueError("CodeAct block must call exactly one action function")
        return self.pending_action

    def _set_action(self, action: AgentAction) -> None:
        self.action_call_count += 1
        self.pending_action = action

    def launch_subagent(
        self,
        goal: str,
        *,
        context: dict[str, Any] | None = None,
        expected_output: str | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> DeferredSubtask:
        deferred = DeferredSubtask(
            SubtaskSpec(
                goal=str(goal),
                context=context or {},
                expected_output=expected_output,
                constraints=constraints or {},
            )
        )
        self._set_action(AgentAction(
            type=ActionType.LAUNCH_SUBAGENT,
            reason="CodeAct launch_subagent",
            subtask=deferred.spec,
        ))
        self._deferred_action_ids.add(id(deferred))
        return deferred

    def gather(self, *items: DeferredSubtask) -> list[DeferredSubtask]:
        if not items or not all(isinstance(item, DeferredSubtask) for item in items):
            raise ValueError("gather requires launch_subagent results")
        item_ids = [id(item) for item in items]
        if len(item_ids) != len(set(item_ids)) or any(
            item_id not in self._deferred_action_ids for item_id in item_ids
        ):
            raise ValueError("gather requires unique launch_subagent results from the current block")
        self.action_call_count -= len(items)
        self._deferred_action_ids.difference_update(item_ids)
        self._set_action(AgentAction(
            type=ActionType.LAUNCH_SUBAGENTS_PARALLEL,
            reason="CodeAct parallel delegation",
            subtasks=[item.spec for item in items],
        ))
        return list(items)

    def search_web(self, query: str, max_results: int = 5) -> None:
        self._set_action(AgentAction(
            type=ActionType.TOOL_CALL,
            reason="CodeAct search_web",
            tool_name="search_web",
            arguments={"query": str(query), "max_results": int(max_results)},
        ))

    def view_webpage_content(self, url: str) -> None:
        self._set_action(AgentAction(
            type=ActionType.TOOL_CALL,
            reason="CodeAct view_webpage_content",
            tool_name="view_webpage_content",
            arguments={"url": str(url)},
        ))

    def think(self, thought: str) -> None:
        self._set_action(AgentAction(
            type=ActionType.THINK,
            reason="CodeAct thought",
            thought=str(thought),
        ))

    def direct_answer(
        self,
        answer: str,
        *,
        evidence: list[str] | None = None,
        limitations: list[str] | None = None,
    ) -> None:
        self._set_action(AgentAction(
            type=ActionType.DIRECT_ANSWER,
            reason="CodeAct draft",
            answer=str(answer),
            evidence=evidence or [],
            limitations=limitations or [],
        ))

    def finish(
        self,
        answer: str,
        *,
        evidence: list[str] | None = None,
        limitations: list[str] | None = None,
    ) -> None:
        self._set_action(AgentAction(
            type=ActionType.FINISH,
            reason="CodeAct finish",
            answer=str(answer),
            evidence=evidence or [],
            limitations=limitations or [],
        ))
