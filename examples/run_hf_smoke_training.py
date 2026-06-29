"""On-policy Qwen/Hugging Face recursive rollout and one LoRA CISPO step."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_training.compilation import compile_optimizer_batch, verify_and_score_group
from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.grouping import collect_rollout_group
from recursive_agent_training.hf_policy import HuggingFaceTrainingPolicy
from recursive_agent_training.mini import make_harness_config
from recursive_agent_training.optimizers.torch_optimizer import TorchCISPOOptimizer, TorchOptimizerSettings
from recursive_agent_training.schemas import RootTaskRecord, write_json
from recursive_agent_training.snapshots import PolicySnapshot, PolicySnapshotProvider
from recursive_agent_training.verifiers.exact import ExactMatchVerifier


DEFAULT_PROMPT = (
    "Delegate exactly one subagent to choose exactly one token from alpha or beta using its own "
    "judgment. After the child returns, finish with exactly that chosen token and no other text."
)


async def run(args) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig.from_yaml(args.config)
    config.model.path = args.model
    config.model.name = Path(args.model).name
    config.checkpoint.output_dir = str(output)
    config.rollout.group_size = args.group_size
    config.rollout.temperature = args.temperature
    config.rollout.max_depth = 1
    config.rollout.max_steps_per_node = min(config.rollout.max_steps_per_node, 4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    current = PolicySnapshotProvider(model_name=config.model.name).current()
    snapshot = PolicySnapshot(
        version=current.version,
        model_name=current.model_name,
        tokenizer_revision=str(tokenizer.init_kwargs.get("_commit_hash") or ""),
        template_version="hf_chat_template_v1",
        checkpoint_path=args.model,
    )
    task = RootTaskRecord(
        task_id="synthetic:qwen-on-policy",
        domain="synthetic",
        split="train",
        prompt=args.prompt,
        ground_truth=args.ground_truth,
        dataset_hash="synthetic-on-policy-v1",
    )
    harness_config = make_harness_config(config, str(output))
    bundle = await collect_rollout_group(
        task,
        snapshot,
        config.rollout.group_size,
        lambda index, seed: HuggingFaceTrainingPolicy(
            base_model,
            tokenizer,
            snapshot,
            temperature=config.rollout.temperature,
            max_new_tokens=config.rollout.max_tokens,
            seed=args.seed + seed,
        ),
        harness_config,
        output_dir=output / "rollouts",
    )
    await verify_and_score_group(bundle, ExactMatchVerifier(), config.reward.delegation_lambda)
    batch = compile_optimizer_batch([bundle], config, 0, snapshot.version)
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    optimizer = TorchCISPOOptimizer(
        model,
        TorchOptimizerSettings(
            learning_rate=max(config.optimizer.learning_rate, 1e-4),
            grad_clip_norm=config.optimizer.grad_clip_norm,
            epsilon_low=config.objective.epsilon_low,
            epsilon_high=config.objective.epsilon_high,
        ),
    )
    metrics = optimizer.step(batch)
    adapter_path = output / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    write_json(output / "rollout_group.json", bundle)
    write_json(output / "optimizer_batch.json", batch)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rollout_summaries = []
    for tree in bundle.rollouts:
        root = tree.nodes[tree.root_node_id]
        rollout_summaries.append(
            {
                "rollout_id": tree.rollout_id,
                "root_answer": root.final_answer,
                "root_reward": root.credit.reward.reward if root.credit.reward else None,
                "node_count": len(tree.nodes),
                "turn_count": sum(len(node.turns) for node in tree.nodes.values()),
                "model_generated_actions": [
                    turn.tokens.generated_text
                    for node in tree.nodes.values()
                    for turn in node.turns
                ],
            }
        )
    result = {
        **metrics,
        "on_policy_generation": True,
        "model_path": args.model,
        "adapter_path": str(adapter_path),
        "gpu_max_memory_mib": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
        "tree_count": len(bundle.rollouts),
        "node_count": sum(len(tree.nodes) for tree in bundle.rollouts),
        "turn_count": len(batch.samples),
        "rollouts": rollout_summaries,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/mini_rao.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--ground-truth", default="alpha")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
