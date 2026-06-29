"""Oolong-Real adapter contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OolongProfile:
    max_depth: int = 2
    max_steps_per_node: int = 15
    delegation_lambda: float = 0.4
    lora_rank: int = 32
    learning_rate: float = 3e-5
