"""TextCraft-Synth adapter contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextCraftProfile:
    train_max_depth: int = 6
    eval_max_depth: int = 12
    max_steps_per_node: int = 25
    delegation_lambda: float = 0.0
