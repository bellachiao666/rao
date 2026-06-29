"""Synthetic exact-match tasks for end-to-end flywheel validation."""

from __future__ import annotations

from recursive_agent_training.schemas import RootTaskRecord


_TOKEN_PAIRS = [
    ("alpha", "beta"),
    ("red", "blue"),
    ("left", "right"),
    ("sun", "moon"),
    ("up", "down"),
    ("hot", "cold"),
    ("yes", "no"),
    ("one", "two"),
]


def make_stochastic_exact_tasks(count: int = 4) -> list[RootTaskRecord]:
    if count <= 0 or count > len(_TOKEN_PAIRS):
        raise ValueError(f"synthetic task count must be in [1, {len(_TOKEN_PAIRS)}]")
    tasks: list[RootTaskRecord] = []
    for index, (expected, alternative) in enumerate(_TOKEN_PAIRS[:count]):
        tasks.append(
            RootTaskRecord(
                task_id=f"synthetic:stochastic-token:{index:02d}",
                domain="synthetic",
                split="train",
                prompt=(
                    f"Immediately finish by choosing exactly one token: {expected} or {alternative}. "
                    "Use the model's sampling randomness rather than a deterministic rule. "
                    "Return a FINISH JSON action whose answer field contains only the chosen token."
                ),
                ground_truth=expected,
                dataset_hash="synthetic-stochastic-exact-v1",
                metadata={"alternatives": [expected, alternative]},
            )
        )
    return tasks
