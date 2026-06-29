"""Clipped Importance Sampling Policy Optimization objective."""

from __future__ import annotations

from typing import Any


def cispo_loss(
    current_logprobs: Any,
    behavior_logprobs: Any,
    action_mask: Any,
    advantages: Any,
    depth_weights: Any,
    epsilon_low: float,
    epsilon_high: float,
) -> tuple[Any, dict[str, float]]:
    import torch

    if epsilon_low < 0 or epsilon_high < 0:
        raise ValueError("CISPO clipping bounds must be non-negative")
    if current_logprobs.shape != behavior_logprobs.shape or current_logprobs.shape != action_mask.shape:
        raise ValueError("logprob and action-mask shapes must match")
    if current_logprobs.ndim != 2:
        raise ValueError("CISPO expects [batch, generated_tokens] tensors")
    if advantages.ndim != 1 or depth_weights.ndim != 1:
        raise ValueError("advantages and depth weights must be one-dimensional")
    if advantages.shape[0] != current_logprobs.shape[0] or depth_weights.shape[0] != current_logprobs.shape[0]:
        raise ValueError("batch dimensions must match")
    log_ratio = current_logprobs - behavior_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, min=1.0 - epsilon_low, max=1.0 + epsilon_high).detach()
    scale = advantages[:, None] * depth_weights[:, None]
    token_objective = clipped_ratio * scale * current_logprobs
    mask = action_mask.to(dtype=current_logprobs.dtype)
    denominator = mask.sum()
    if denominator.item() <= 0:
        raise ValueError("CISPO batch has no trainable action tokens")
    loss = -(token_objective * mask).sum() / denominator
    active = mask.bool()
    metrics = {
        "loss": float(loss.detach()),
        "ratio_mean": float(ratio[active].mean().detach()),
        "ratio_min": float(ratio[active].min().detach()),
        "ratio_max": float(ratio[active].max().detach()),
        "clipped_fraction": float((ratio[active] != clipped_ratio[active]).float().mean().detach()),
        "action_tokens": float(denominator.detach()),
    }
    return loss, metrics
