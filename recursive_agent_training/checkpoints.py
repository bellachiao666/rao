"""Atomic checkpoint helpers."""

from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path
from typing import Any


def atomic_torch_save(path: str | Path, state: dict[str, Any]) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, target)


def load_torch_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
    *,
    required_keys: Collection[str] = (),
) -> dict[str, Any]:
    """Load tensor-only state from a checkpoint without arbitrary unpickling."""

    import torch

    state = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a state mapping")
    missing = sorted(set(required_keys) - set(state))
    if missing:
        raise ValueError("checkpoint is missing required keys: " + ", ".join(missing))
    return state
