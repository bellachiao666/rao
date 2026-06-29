"""Single-process PyTorch CISPO optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recursive_agent_training.objectives.cispo import cispo_loss
from recursive_agent_training.optimizers.base import PolicyOptimizer
from recursive_agent_training.schemas import CompiledBatch, OptimizerSample


@dataclass
class TorchOptimizerSettings:
    learning_rate: float = 3e-6
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2


class TorchCISPOOptimizer(PolicyOptimizer):
    def __init__(self, model: Any, settings: TorchOptimizerSettings, version: int = 0):
        import torch

        self.model = model
        self.settings = settings
        self.version = version
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            betas=(settings.beta1, settings.beta2),
        )

    def current_version(self) -> str:
        return f"policy_{self.version:06d}"

    def _device(self):
        return next(self.model.parameters()).device

    def _current_logprobs(self, samples: list[OptimizerSample]):
        import torch

        device = self._device()
        pad_id = 0
        sequences = [sample.input_ids + sample.generated_ids for sample in samples]
        if any(len(sample.input_ids) == 0 or len(sample.generated_ids) == 0 for sample in samples):
            raise ValueError("samples require non-empty prompt and generated tokens")
        max_sequence = max(len(sequence) for sequence in sequences)
        input_tensor = torch.full((len(samples), max_sequence), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(input_tensor)
        for row, sequence in enumerate(sequences):
            input_tensor[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            attention[row, : len(sequence)] = 1
        outputs = self.model(input_ids=input_tensor, attention_mask=attention)
        logits = outputs.logits
        max_generated = max(len(sample.generated_ids) for sample in samples)
        current = torch.zeros((len(samples), max_generated), dtype=logits.dtype, device=device)
        behavior = torch.zeros_like(current)
        mask = torch.zeros_like(current)
        for row, sample in enumerate(samples):
            prompt_length = len(sample.input_ids)
            generated_length = len(sample.generated_ids)
            positions = torch.arange(prompt_length - 1, prompt_length + generated_length - 1, device=device)
            token_ids = torch.tensor(sample.generated_ids, dtype=torch.long, device=device)
            if sample.sampling_temperature <= 0:
                raise ValueError("sampling temperature must be positive")
            token_logits = logits[row, positions] / sample.sampling_temperature
            current[row, :generated_length] = torch.log_softmax(token_logits, dim=-1).gather(
                -1, token_ids[:, None]
            ).squeeze(-1)
            behavior[row, :generated_length] = torch.tensor(
                sample.behavior_logprobs, dtype=logits.dtype, device=device
            )
            mask[row, :generated_length] = torch.tensor(sample.action_mask, dtype=logits.dtype, device=device)
        return current, behavior, mask

    def step(self, batch: CompiledBatch) -> dict[str, Any]:
        import torch

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_tokens = sum(sum(sample.action_mask) for sample in batch.samples)
        if total_tokens <= 0:
            raise ValueError("CISPO batch has no trainable action tokens")
        loss_value = 0.0
        ratio_sum = 0.0
        ratio_min = float("inf")
        ratio_max = float("-inf")
        clipped_tokens = 0.0
        for sample in batch.samples:
            current, behavior, mask = self._current_logprobs([sample])
            device = current.device
            advantages = torch.tensor([sample.advantage], dtype=current.dtype, device=device)
            depth_weights = torch.tensor([sample.depth_weight], dtype=current.dtype, device=device)
            loss, sample_metrics = cispo_loss(
                current,
                behavior,
                mask,
                advantages,
                depth_weights,
                self.settings.epsilon_low,
                self.settings.epsilon_high,
            )
            sample_tokens = sample_metrics["action_tokens"]
            scale = sample_tokens / total_tokens
            (loss * scale).backward()
            loss_value += sample_metrics["loss"] * scale
            ratio_sum += sample_metrics["ratio_mean"] * sample_tokens
            ratio_min = min(ratio_min, sample_metrics["ratio_min"])
            ratio_max = max(ratio_max, sample_metrics["ratio_max"])
            clipped_tokens += sample_metrics["clipped_fraction"] * sample_tokens
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.settings.grad_clip_norm)
        self.optimizer.step()
        self.version += 1
        batch.manifest.policy_version_after = self.current_version()
        metrics = {
            "loss": loss_value,
            "ratio_mean": ratio_sum / total_tokens,
            "ratio_min": ratio_min,
            "ratio_max": ratio_max,
            "clipped_fraction": clipped_tokens / total_tokens,
            "action_tokens": float(total_tokens),
            "grad_norm": float(grad_norm),
            "policy_version": self.current_version(),
            "sample_count": len(batch.samples),
        }
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "version": self.version,
            "settings": self.settings.__dict__,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.version = int(state["version"])

    def optimizer_state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "version": self.version,
            "settings": self.settings.__dict__,
        }

    def load_optimizer_state_dict(self, state: dict[str, Any], expected_version: int | None = None) -> None:
        version = int(state["version"])
        if expected_version is not None and version != expected_version:
            raise ValueError(f"optimizer checkpoint version {version} does not match expected {expected_version}")
        self.optimizer.load_state_dict(state["optimizer"])
        self.version = version

    def save_checkpoint(self, path: str) -> str:
        from recursive_agent_training.checkpoints import atomic_torch_save

        atomic_torch_save(path, self.state_dict())
        return path

    def save_optimizer_checkpoint(self, path: str) -> str:
        from recursive_agent_training.checkpoints import atomic_torch_save

        atomic_torch_save(path, self.optimizer_state_dict())
        return path
