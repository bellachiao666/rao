"""Tool registry and built-in local tools."""

from __future__ import annotations

import ast
import inspect
import operator
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


ToolCallable = Callable[[dict[str, Any]], Any | Awaitable[Any]]


class ToolSpec(BaseModel):
    name: str
    description: str
    args_schema: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(BaseModel):
    tool_name: str
    ok: bool
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolCallable] = {}
        self._specs: dict[str, ToolSpec] = {}

    def register_tool(
        self,
        name: str,
        func: ToolCallable,
        description: str,
        args_schema: dict[str, Any] | None = None,
    ) -> None:
        self._tools[name] = func
        self._specs[name] = ToolSpec(name=name, description=description, args_schema=args_schema or {})

    async def call(self, tool_name: str, arguments: dict[str, Any] | None = None) -> ToolObservation:
        arguments = arguments or {}
        if tool_name not in self._tools:
            return ToolObservation(tool_name=tool_name, ok=False, error=f"Tool '{tool_name}' is not registered.")

        try:
            result = self._tools[tool_name](arguments)
            if inspect.isawaitable(result):
                result = await result
            return ToolObservation(tool_name=tool_name, ok=True, result=result)
        except Exception as exc:  # pragma: no cover - exact exception varies by tool
            return ToolObservation(tool_name=tool_name, ok=False, error=str(exc))

    def specs(self) -> list[dict[str, Any]]:
        return [spec.model_dump(mode="json") for spec in self._specs.values()]

    @classmethod
    def with_default_tools(cls) -> "ToolRegistry":
        registry = cls()
        registry.register_tool(
            "calculator",
            calculator,
            "Evaluate a simple arithmetic expression.",
            {"expression": "string"},
        )
        registry.register_tool(
            "read_text",
            read_text,
            "Read a UTF-8 text file from disk.",
            {"path": "string"},
        )
        registry.register_tool(
            "python_exec",
            python_exec_disabled,
            "Disabled placeholder for a future Python execution sandbox.",
            {"code": "string"},
        )
        return registry


def calculator(arguments: dict[str, Any]) -> int | float:
    expression = str(arguments.get("expression", ""))
    tree = ast.parse(expression, mode="eval")
    return _eval_arithmetic(tree.body)


def read_text(arguments: dict[str, Any]) -> str:
    path = Path(str(arguments.get("path", "")))
    return path.read_text(encoding="utf-8")


def python_exec_disabled(arguments: dict[str, Any]) -> None:
    raise RuntimeError("python_exec is a disabled placeholder; configure a sandbox before enabling it.")


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_arithmetic(node.left), _eval_arithmetic(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_arithmetic(node.operand))
    raise ValueError("calculator only supports arithmetic expressions")
