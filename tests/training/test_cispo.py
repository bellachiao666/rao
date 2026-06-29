import pytest

torch = pytest.importorskip("torch")

from recursive_agent_training.objectives.cispo import cispo_loss
from recursive_agent_training.optimizers.torch_optimizer import TorchCISPOOptimizer, TorchOptimizerSettings
from recursive_agent_training.schemas import CompiledBatch, OptimizerBatchManifest, OptimizerSample


def test_cispo_masks_tokens_and_clips_ratio():
    current = torch.tensor([[0.0, 0.0]], requires_grad=True)
    behavior = torch.tensor([[-2.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    loss, metrics = cispo_loss(
        current,
        behavior,
        mask,
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        0.2,
        0.2,
    )
    loss.backward()
    assert metrics["clipped_fraction"] == 1.0
    assert current.grad[0, 0] != 0
    assert current.grad[0, 1] == 0


def test_cispo_rejects_empty_action_mask():
    with pytest.raises(ValueError, match="no trainable"):
        cispo_loss(
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            torch.ones(1),
            torch.ones(1),
            0.2,
            0.2,
        )


def test_same_policy_behavior_logprobs_produce_unit_importance_ratio():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.output = torch.nn.Linear(8, 16)

        def forward(self, input_ids, attention_mask=None):
            return type("Output", (), {"logits": self.output(self.embedding(input_ids))})()

    model = TinyModel()
    prompt = [1, 2]
    generated = [3, 4]
    temperature = 1.2
    sequence = torch.tensor([prompt + generated])
    with torch.no_grad():
        logits = model(sequence).logits[0]
        positions = torch.tensor([len(prompt) - 1, len(prompt)])
        token_logits = logits[positions] / temperature
        behavior = (
            torch.log_softmax(token_logits, dim=-1)
            .gather(-1, torch.tensor(generated)[:, None])
            .squeeze(-1)
            .tolist()
        )
    sample = OptimizerSample(
        task_id="t",
        group_id="g",
        rollout_id="r",
        node_id="n",
        depth=0,
        input_ids=prompt,
        generated_ids=generated,
        action_mask=[1, 1],
        behavior_logprobs=behavior,
        advantage=1.0,
        depth_weight=1.0,
        behavior_policy_version="policy_000000",
        sampling_temperature=temperature,
    )
    batch = CompiledBatch(
        manifest=OptimizerBatchManifest(
            optimizer_step=0,
            policy_version_before="policy_000000",
            root_task_count=1,
            group_size=2,
            tree_count=2,
            trainable_node_count=1,
            trainable_turn_count=1,
            max_staleness_batches=0,
            config_hash="test",
        ),
        samples=[sample],
    )
    optimizer = TorchCISPOOptimizer(model, TorchOptimizerSettings(learning_rate=1e-3))
    metrics = optimizer.step(batch)
    assert metrics["ratio_mean"] == pytest.approx(1.0, abs=1e-6)
