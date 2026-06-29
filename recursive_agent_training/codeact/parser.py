"""Minimal parser for thought-plus-Python CodeAct outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CodeActAction:
    thought: str
    code: str


def parse_codeact(text: str) -> CodeActAction:
    matches = list(
        re.finditer(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    )
    if len(matches) > 1:
        raise ValueError("CodeAct output must contain at most one code block")
    if matches:
        match = matches[0]
        code = match.group(1).strip()
        thought = text[: match.start()].strip()
        if text[match.end() :].strip():
            raise ValueError("CodeAct output must not contain text after the Python block")
    else:
        stripped = text.strip()
        thinking = re.match(
            r"<think>(.*?)</think>\s*(.*)",
            stripped,
            flags=re.DOTALL,
        )
        if thinking:
            thought = thinking.group(1).strip()
            code = thinking.group(2).strip()
        else:
            thought = ""
            code = stripped
        if code.startswith("python\n"):
            code = code.removeprefix("python\n").strip()
    if not code:
        raise ValueError("CodeAct code block is empty")
    return CodeActAction(thought=thought, code=code)
