"""Hugging Face and PEFT runtime for the synchronous flywheel."""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recursive_agent_training.checkpoints import load_torch_checkpoint
from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.flywheel import LatestPublication, RoundRuntime, parse_policy_version
from recursive_agent_training.hf_policy import HuggingFaceTrainingPolicy
from recursive_agent_training.optimizers.torch_optimizer import TorchCISPOOptimizer, TorchOptimizerSettings
from recursive_agent_training.snapshots import PolicySnapshot


def prepare_trainable_model(model: Any) -> Any:
    """Configure a causal LM for memory-bounded LoRA training."""

    model.config.use_cache = False
    enable_input_grads = getattr(model, "enable_input_require_grads", None)
    if callable(enable_input_grads):
        enable_input_grads()
    enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
    if callable(enable_checkpointing):
        try:
            enable_checkpointing(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            enable_checkpointing()
    return model


@dataclass
class HuggingFaceRuntimeFactory:
    config: TrainingConfig
    base_model_path: str
    device: str = "cuda:0"
    devices: tuple[str, ...] = ()
    trainer_device: str | None = None

    def __call__(
        self,
        snapshot: PolicySnapshot,
        latest: LatestPublication | None,
    ) -> RoundRuntime:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_path = str(Path(self.base_model_path).resolve())
        rollout_devices = self.devices or (self.device,)
        if len(set(rollout_devices)) != len(rollout_devices):
            raise ValueError("rollout devices must be unique")
        if self.config.rollout.group_size > len(rollout_devices):
            raise ValueError(
                "group_size cannot exceed rollout device count for parallel generation"
            )
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.config.model.dtype]
        def load_model(device: str, *, trainable: bool):
            model = AutoModelForCausalLM.from_pretrained(
                base_path,
                local_files_only=True,
                torch_dtype=dtype,
                device_map={"": device},
                low_cpu_mem_usage=True,
            )
            if latest is None:
                torch.manual_seed(self.config.experiment.seed)
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=self.config.model.lora_rank,
                        lora_alpha=self.config.model.lora_alpha,
                        lora_dropout=0.0,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        task_type="CAUSAL_LM",
                    ),
                )
            else:
                model = PeftModel.from_pretrained(
                    model,
                    latest.adapter_path,
                    is_trainable=trainable,
                    local_files_only=True,
                )
            if not trainable:
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                model.config.use_cache = True
            else:
                prepare_trainable_model(model)
            return model

        training_device = self.trainer_device or rollout_devices[0]
        trainer_is_rollout = training_device == rollout_devices[0]
        models = [
            load_model(device, trainable=trainer_is_rollout and index == 0)
            for index, device in enumerate(rollout_devices)
        ]
        model = (
            models[0]
            if trainer_is_rollout
            else load_model(training_device, trainable=True)
        )
        generation_locks = [threading.Lock() for _ in models]
        if torch.cuda.is_available():
            for device in set((*rollout_devices, training_device)):
                parsed_device = torch.device(device)
                if parsed_device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(parsed_device)
        version = parse_policy_version(snapshot.version)
        optimizer = TorchCISPOOptimizer(
            model,
            TorchOptimizerSettings(
                learning_rate=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
                beta1=self.config.optimizer.beta1,
                beta2=self.config.optimizer.beta2,
                grad_clip_norm=self.config.optimizer.grad_clip_norm,
                epsilon_low=self.config.objective.epsilon_low,
                epsilon_high=self.config.objective.epsilon_high,
            ),
            version=version,
        )
        if latest is not None:
            optimizer_state = load_torch_checkpoint(
                latest.optimizer_path,
                map_location="cpu",
                required_keys={"optimizer", "version", "settings"},
            )
            optimizer.load_optimizer_state_dict(optimizer_state, expected_version=version)

        tokenizer_revision = str(tokenizer.init_kwargs.get("_commit_hash") or "")
        runtime_snapshot = PolicySnapshot(
            version=snapshot.version,
            model_name=snapshot.model_name,
            tokenizer_revision=tokenizer_revision,
            template_version=snapshot.template_version,
            checkpoint_path=snapshot.checkpoint_path,
            checkpoint_hash=snapshot.checkpoint_hash,
        )

        def policy_factory(index: int, seed: int):
            policy_class = HuggingFaceTrainingPolicy
            if self.config.experiment.action_profile == "codeact":
                from recursive_agent_training.codeact.policy import (
                    HuggingFaceCodeActTrainingPolicy,
                )

                policy_class = HuggingFaceCodeActTrainingPolicy
            slot = index % len(models)
            return policy_class(
                models[slot],
                tokenizer,
                runtime_snapshot,
                temperature=self.config.rollout.temperature,
                max_new_tokens=self.config.rollout.max_tokens,
                max_input_tokens=(
                    self.config.model.train_context_tokens
                    - self.config.rollout.max_tokens
                ),
                seed=self.config.experiment.seed + seed,
                threaded_generation=len(models) > 1,
                generation_lock=generation_locks[slot],
            )

        def save_adapter(path: Path) -> None:
            model.save_pretrained(path, safe_serialization=True)
            tokenizer.save_pretrained(path)

        def save_optimizer(path: Path) -> None:
            optimizer.save_optimizer_checkpoint(str(path))

        def runtime_metrics() -> dict[str, Any]:
            if not torch.cuda.is_available():
                return {
                    "gpu_max_memory_mib": 0.0,
                    "rollout_devices": list(rollout_devices),
                    "gpu_max_memory_mib_by_device": {},
                }
            memory_by_device = {}
            for device in dict.fromkeys((*rollout_devices, training_device)):
                parsed_device = torch.device(device)
                memory_by_device[device] = (
                    round(
                        torch.cuda.max_memory_allocated(parsed_device) / 1024 / 1024,
                        1,
                    )
                    if parsed_device.type == "cuda"
                    else 0.0
                )
            return {
                "gpu_max_memory_mib": max(memory_by_device.values()),
                "rollout_devices": list(rollout_devices),
                "trainer_device": training_device,
                "gpu_max_memory_mib_by_device": memory_by_device,
            }

        def close() -> None:
            nonlocal model, models
            if torch.cuda.is_available():
                for device in dict.fromkeys((*rollout_devices, training_device)):
                    parsed_device = torch.device(device)
                    if parsed_device.type == "cuda":
                        try:
                            torch.cuda.synchronize(parsed_device)
                        except RuntimeError:
                            pass
            optimizer.model = None
            optimizer.optimizer = None
            models.clear()
            model = None
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass

        return RoundRuntime(
            policy_factory=policy_factory,
            optimizer=optimizer,
            save_adapter=save_adapter,
            save_optimizer=save_optimizer,
            runtime_metrics=runtime_metrics,
            close=close,
        )


@dataclass
class HuggingFacePolicyPool:
    config: TrainingConfig
    base_model_path: str
    snapshot: PolicySnapshot
    devices: tuple[str, ...]
    adapter_path: str = ""

    def __post_init__(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not self.devices:
            raise ValueError("evaluation policy pool requires at least one device")
        base_path = str(Path(self.base_model_path).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.config.model.dtype]
        self.models = []
        for device in self.devices:
            model = AutoModelForCausalLM.from_pretrained(
                base_path,
                local_files_only=True,
                torch_dtype=dtype,
                device_map={"": device},
                low_cpu_mem_usage=True,
            )
            if self.adapter_path:
                model = PeftModel.from_pretrained(
                    model,
                    self.adapter_path,
                    is_trainable=False,
                    local_files_only=True,
                )
            model.eval()
            model.config.use_cache = True
            self.models.append(model)
        self._locks = [threading.Lock() for _ in self.models]
        if torch.cuda.is_available():
            for device in self.devices:
                parsed = torch.device(device)
                if parsed.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(parsed)

    def policy_factory(self, index: int, seed: int) -> HuggingFaceTrainingPolicy:
        policy_class = HuggingFaceTrainingPolicy
        if self.config.experiment.action_profile == "codeact":
            from recursive_agent_training.codeact.policy import (
                HuggingFaceCodeActTrainingPolicy,
            )

            policy_class = HuggingFaceCodeActTrainingPolicy
        slot = index % len(self.models)
        return policy_class(
            self.models[slot],
            self.tokenizer,
            self.snapshot,
            temperature=self.config.rollout.temperature,
            max_new_tokens=self.config.rollout.max_tokens,
            max_input_tokens=(
                self.config.model.eval_context_tokens
                - self.config.rollout.max_tokens
            ),
            seed=self.config.experiment.seed + seed,
            threaded_generation=True,
            generation_lock=self._locks[slot],
        )

    def metrics(self) -> dict[str, Any]:
        import torch

        memory = {}
        if torch.cuda.is_available():
            for device in self.devices:
                parsed = torch.device(device)
                memory[device] = (
                    round(torch.cuda.max_memory_allocated(parsed) / 1024 / 1024, 1)
                    if parsed.type == "cuda"
                    else 0.0
                )
        return {
            "rollout_devices": list(self.devices),
            "gpu_max_memory_mib_by_device": memory,
        }

    def close(self) -> None:
        import torch

        self.models.clear()
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
