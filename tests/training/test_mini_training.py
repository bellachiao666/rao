import pytest

torch = pytest.importorskip("torch")

from recursive_agent_training.compilation import compile_optimizer_batch, verify_and_score_group
from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.grouping import collect_rollout_group
from recursive_agent_training.mini import build_toy_model, make_harness_config, make_scripted_policy, synthetic_task
from recursive_agent_training.optimizers.torch_optimizer import TorchCISPOOptimizer, TorchOptimizerSettings
from recursive_agent_training.snapshots import PolicySnapshotProvider
from recursive_agent_training.verifiers.exact import ExactMatchVerifier


@pytest.mark.asyncio
async def test_end_to_end_recursive_rollout_to_parameter_update(tmp_path):
    config = TrainingConfig.from_yaml("configs/train/mini_rao.yaml")
    snapshot = PolicySnapshotProvider(model_name="toy").current()
    bundle = await collect_rollout_group(
        synthetic_task(),
        snapshot,
        2,
        lambda index, seed: make_scripted_policy(snapshot, "alpha" if index == 0 else "beta"),
        make_harness_config(config, str(tmp_path)),
        output_dir=tmp_path,
    )
    await verify_and_score_group(bundle, ExactMatchVerifier(), 0)
    batch = compile_optimizer_batch([bundle], config, 0, snapshot.version)
    model = build_toy_model()
    optimizer = TorchCISPOOptimizer(model, TorchOptimizerSettings(learning_rate=1e-3))
    before = next(model.parameters()).detach().clone()
    metrics = optimizer.step(batch)
    after = next(model.parameters()).detach()
    assert metrics["sample_count"] == 6
    assert not torch.equal(before, after)
