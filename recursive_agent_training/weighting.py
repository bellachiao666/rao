"""RAO Equation 4 depth inverse-frequency weighting."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DepthWeightResult:
    depth_counts: dict[int, int]
    depth_weights: dict[int, float]
    alpha: float


def compute_depth_weights(depths: Iterable[int]) -> DepthWeightResult:
    counts = Counter(depths)
    if not counts:
        raise ValueError("at least one trainable trajectory is required")
    if any(depth < 0 or count <= 0 for depth, count in counts.items()):
        raise ValueError("depths must be non-negative")
    total = sum(counts.values())
    alpha = total / len(counts)
    weights = {depth: alpha / count for depth, count in sorted(counts.items())}
    if not math.isclose(sum(counts[d] * weights[d] for d in counts), total):
        raise AssertionError("depth weights must preserve total batch weight")
    return DepthWeightResult(dict(sorted(counts.items())), weights, alpha)
