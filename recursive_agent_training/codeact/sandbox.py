"""Conservative validation boundary for future CodeAct execution."""

from __future__ import annotations

import ast


_BLOCKED = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.AsyncWith,
    ast.Global,
    ast.Nonlocal,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.Try,
    ast.Raise,
    ast.Delete,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def validate_codeact_ast(code: str) -> ast.Module:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED):
            raise ValueError(f"blocked CodeAct syntax: {type(node).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
            "eval",
            "exec",
            "compile",
            "open",
            "__import__",
        }:
            raise ValueError(f"blocked CodeAct call: {node.func.id}")
        if isinstance(node, ast.Attribute):
            allowed = (
                isinstance(node.value, ast.Name)
                and node.value.id == "asyncio"
                and node.attr == "gather"
            )
            if not allowed:
                raise ValueError("attribute access is blocked in CodeAct")
    return tree
