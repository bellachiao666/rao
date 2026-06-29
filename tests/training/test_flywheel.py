import asyncio
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from recursive_agent_harness.actions import ActionType, AgentAction
from recursive_agent_training.checkpoints import atomic_torch_save, load_torch_checkpoint
from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.flywheel import (
    LatestPublication,
    RoundRuntime,
    RoundStatus,
    SyncFlywheelRunner,
)
from recursive_agent_training.optimizers.base import PolicyOptimizer
from recursive_agent_training.schemas import CompiledBatch
from recursive_agent_training.snapshots import ScriptedTrainingPolicy
from recursive_agent_training.synthetic import make_stochastic_exact_tasks
from recursive_agent_training.verifiers.exact import ExactMatchVerifier


class ToyParameterOptimizer(PolicyOptimizer):
    def __init__(self, value, version, ratio=1.0):
        self.value = value
        self.version = version
        self.ratio = ratio

    def current_version(self):
        return f"policy_{self.version:06d}"

    def step(self, batch: CompiledBatch):
        self.value.add_(1.0)
        self.version += 1
        batch.manifest.policy_version_after = self.current_version()
        return {
            "loss": 0.25,
            "ratio_mean": self.ratio,
            "ratio_min": self.ratio,
            "ratio_max": self.ratio,
            "clipped_fraction": 0.0,
            "action_tokens": 1.0,
            "grad_norm": 1.0,
            "policy_version": self.current_version(),
            "sample_count": len(batch.samples),
        }

    def save_checkpoint(self, path):
        atomic_torch_save(path, {"value": self.value.clone(), "version": self.version})
        return path


class ToyRuntimeFactory:
    def __init__(
        self,
        ratio=1.0,
        zero_first_task=False,
        fallback_first_task=False,
    ):
        self.ratio = ratio
        self.zero_first_task = zero_first_task
        self.fallback_first_task = fallback_first_task
        self.loaded_values = []
        self.policy_indices = []
        self.close_count = 0

    def __call__(self, snapshot, latest: LatestPublication | None):
        if latest is None:
            value = torch.tensor(0.0)
        else:
            value = torch.load(
                Path(latest.adapter_path) / "model.pt",
                map_location="cpu",
                weights_only=True,
            )["value"]
        self.loaded_values.append(float(value))
        version = int(snapshot.version.rsplit("_", 1)[-1])
        optimizer = ToyParameterOptimizer(value, version, ratio=self.ratio)

        def policy_factory(index, seed):
            self.policy_indices.append(index)

            def action(state):
                alternatives = {
                    "alpha or beta": ("alpha", "beta"),
                    "red or blue": ("red", "blue"),
                }
                expected, wrong = next(
                    pair for marker, pair in alternatives.items() if marker in state.task
                )
                answer = (
                    wrong
                    if self.zero_first_task and "alpha or beta" in state.task
                    else expected
                    if index % 2 == 0
                    else wrong
                )
                return AgentAction(
                    type=ActionType.FINISH,
                    reason=f"loaded_parameter_{float(value):.1f}",
                    answer=answer,
                    metadata={
                        "fallback": (
                            self.fallback_first_task
                            and "alpha or beta" in state.task
                        )
                    },
                )

            return ScriptedTrainingPolicy(snapshot, action)

        def save_adapter(path):
            torch.save({"value": value.clone()}, path / "model.pt")

        def save_optimizer(path):
            atomic_torch_save(path, {"value": value.clone(), "version": optimizer.version})

        def close():
            self.close_count += 1

        return RoundRuntime(
            policy_factory=policy_factory,
            optimizer=optimizer,
            save_adapter=save_adapter,
            save_optimizer=save_optimizer,
            close=close,
        )


def make_config():
    config = TrainingConfig.from_yaml("configs/train/sync_qwen_0_6b.yaml")
    config.rollout.group_size = 2
    config.rollout.max_depth = 0
    config.rollout.max_steps_per_node = 1
    config.sync_flywheel.root_tasks_per_round = 2
    return config


def make_runner(tmp_path, runtime_factory):
    model_path = tmp_path / "base_model"
    model_path.mkdir(exist_ok=True)
    return SyncFlywheelRunner(
        config=make_config(),
        tasks=make_stochastic_exact_tasks(2),
        output_dir=tmp_path / "run",
        model_name="toy",
        initial_checkpoint_path=str(model_path),
        runtime_factory=runtime_factory,
        verifier_factory=ExactMatchVerifier,
    )


@pytest.mark.asyncio
async def test_two_round_resume_reloads_published_parameter(tmp_path):
    first_factory = ToyRuntimeFactory()
    first_runner = make_runner(tmp_path, first_factory)
    first = await first_runner.run(1)
    assert first[0].status == RoundStatus.PUBLISHED
    assert first[0].policy_version_after == "policy_000001"
    assert first_factory.loaded_values == [0.0]

    incomplete = tmp_path / "run" / "rounds" / "round_0001"
    incomplete.mkdir()
    (incomplete / "partial.txt").write_text("interrupted", encoding="utf-8")

    resumed_factory = ToyRuntimeFactory()
    resumed_runner = make_runner(tmp_path, resumed_factory)
    second = await resumed_runner.run(2, resume=True)
    assert second[0].status == RoundStatus.PUBLISHED
    assert second[0].policy_version_before == "policy_000001"
    assert second[0].policy_version_after == "policy_000002"
    assert resumed_factory.loaded_values == [1.0]
    assert list((tmp_path / "run" / "rounds").glob("round_0001.incomplete_*"))

    raw = json.loads(
        (
            tmp_path
            / "run"
            / "rounds"
            / "round_0001"
            / "raw_rollouts"
            / "group_0000.json"
        ).read_text(encoding="utf-8")
    )
    generated = [
        turn["tokens"]["generated_text"]
        for tree in raw["rollouts"]
        for node in tree["nodes"].values()
        for turn in node["turns"]
    ]
    assert any("loaded_parameter_1.0" in text for text in generated)
    assert raw["group"]["behavior_policy_version"] == "policy_000001"
    assert raw["group"]["behavior_checkpoint_hash"] == first[0].adapter_hash
    assert raw["group"]["behavior_checkpoint_path"] == first[0].adapter_path

    latest = LatestPublication.model_validate_json(
        (tmp_path / "run" / "latest.json").read_text(encoding="utf-8")
    )
    assert latest.policy_version == "policy_000002"
    assert latest.round_index == 1


@pytest.mark.asyncio
async def test_failed_round_does_not_publish_latest(tmp_path):
    runner = make_runner(tmp_path, ToyRuntimeFactory(ratio=0.5))
    with pytest.raises(ValueError, match="importance ratio"):
        await runner.run(1)
    assert not (tmp_path / "run" / "latest.json").exists()
    manifest = json.loads(
        (
            tmp_path / "run" / "rounds" / "round_0000" / "round_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"


@pytest.mark.asyncio
async def test_zero_advantage_batch_retries_with_next_task(tmp_path):
    config = make_config()
    config.sync_flywheel.root_tasks_per_round = 1
    config.sync_flywheel.max_collection_attempts = 2
    model_path = tmp_path / "base_model"
    model_path.mkdir()
    factory = ToyRuntimeFactory(zero_first_task=True)
    runner = SyncFlywheelRunner(
        config=config,
        tasks=make_stochastic_exact_tasks(2),
        output_dir=tmp_path / "run",
        model_name="toy",
        initial_checkpoint_path=str(model_path),
        runtime_factory=factory,
        verifier_factory=ExactMatchVerifier,
    )
    completed = await runner.run(1)
    assert completed[0].status == RoundStatus.PUBLISHED
    assert completed[0].task_ids == ["synthetic:stochastic-token:01"]
    assert completed[0].metrics["collection_attempt_count"] == 2
    assert completed[0].metrics["zero_advantage_retry_count"] == 1
    discarded = completed[0].metrics["discarded_attempt_paths"]
    assert len(discarded) == 1
    assert Path(discarded[0]).exists()
    assert factory.loaded_values == [0.0]
    assert factory.close_count == 1


@pytest.mark.asyncio
async def test_missing_reward_batch_retries_with_next_task(tmp_path, monkeypatch):
    import recursive_agent_training.flywheel as flywheel_module

    config = make_config()
    config.sync_flywheel.root_tasks_per_round = 1
    config.sync_flywheel.max_collection_attempts = 2
    model_path = tmp_path / "base_model"
    model_path.mkdir()
    factory = ToyRuntimeFactory()
    original_compile = flywheel_module.compile_optimizer_batch
    calls = 0

    def fail_first_compile(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("root reward missing for synthetic_rollout")
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(
        flywheel_module,
        "compile_optimizer_batch",
        fail_first_compile,
    )
    runner = SyncFlywheelRunner(
        config=config,
        tasks=make_stochastic_exact_tasks(2),
        output_dir=tmp_path / "run",
        model_name="toy",
        initial_checkpoint_path=str(model_path),
        runtime_factory=factory,
        verifier_factory=ExactMatchVerifier,
    )

    completed = await runner.run(1)

    assert completed[0].status == RoundStatus.PUBLISHED
    assert completed[0].task_ids == ["synthetic:stochastic-token:01"]
    assert completed[0].metrics["collection_attempt_count"] == 2
    assert completed[0].metrics["zero_advantage_retry_count"] == 1
    assert len(completed[0].metrics["discarded_attempt_paths"]) == 1
    assert Path(completed[0].metrics["discarded_attempt_paths"][0]).exists()
    assert calls == 2


@pytest.mark.asyncio
async def test_all_fallback_batch_retries_with_next_task(tmp_path):
    config = make_config()
    config.sync_flywheel.root_tasks_per_round = 1
    config.sync_flywheel.max_collection_attempts = 2
    model_path = tmp_path / "base_model"
    model_path.mkdir()
    factory = ToyRuntimeFactory(fallback_first_task=True)
    runner = SyncFlywheelRunner(
        config=config,
        tasks=make_stochastic_exact_tasks(2),
        output_dir=tmp_path / "run",
        model_name="toy",
        initial_checkpoint_path=str(model_path),
        runtime_factory=factory,
        verifier_factory=ExactMatchVerifier,
    )
    completed = await runner.run(1)
    assert completed[0].status == RoundStatus.PUBLISHED
    assert completed[0].task_ids == ["synthetic:stochastic-token:01"]
    assert completed[0].metrics["collection_attempt_count"] == 2
    assert len(completed[0].metrics["discarded_attempt_paths"]) == 1
    assert factory.loaded_values == [0.0]
    assert factory.close_count == 1


@pytest.mark.asyncio
async def test_round_collection_respects_group_concurrency_limit(
    tmp_path,
    monkeypatch,
):
    import recursive_agent_training.flywheel as flywheel_module

    original = flywheel_module.collect_rollout_group
    active = 0
    peak = 0

    async def tracked_collect(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            return await original(*args, **kwargs)
        finally:
            active -= 1

    monkeypatch.setattr(flywheel_module, "collect_rollout_group", tracked_collect)
    config = make_config()
    config.rollout.max_concurrent_groups = 2
    model_path = tmp_path / "base_model"
    model_path.mkdir()
    factory = ToyRuntimeFactory()
    runner = SyncFlywheelRunner(
        config=config,
        tasks=make_stochastic_exact_tasks(2),
        output_dir=tmp_path / "run",
        model_name="toy",
        initial_checkpoint_path=str(model_path),
        runtime_factory=factory,
        verifier_factory=ExactMatchVerifier,
    )

    completed = await runner.run(1)

    assert completed[0].status == RoundStatus.PUBLISHED
    assert peak == 2
    assert sorted(factory.policy_indices) == [0, 1, 2, 3]
    assert completed[0].raw_group_paths == [
        str(
            tmp_path
            / "run"
            / "rounds"
            / "round_0000"
            / "raw_rollouts"
            / f"group_{index:04d}.json"
        )
        for index in range(2)
    ]


@pytest.mark.asyncio
async def test_runtime_closes_when_verifier_initialization_fails(tmp_path):
    config = make_config()
    model_path = tmp_path / "base_model"
    model_path.mkdir()
    factory = ToyRuntimeFactory()

    def fail_verifier():
        raise RuntimeError("judge init failed")

    runner = SyncFlywheelRunner(
        config=config,
        tasks=make_stochastic_exact_tasks(2),
        output_dir=tmp_path / "run",
        model_name="toy",
        initial_checkpoint_path=str(model_path),
        runtime_factory=factory,
        verifier_factory=fail_verifier,
    )
    with pytest.raises(RuntimeError, match="judge init failed"):
        await runner.run(1)
    assert factory.close_count == 1


@pytest.mark.asyncio
async def test_resume_rejects_corrupted_adapter(tmp_path):
    runner = make_runner(tmp_path, ToyRuntimeFactory())
    await runner.run(1)
    adapter = tmp_path / "run" / "rounds" / "round_0000" / "adapter" / "model.pt"
    adapter.write_bytes(adapter.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="adapter checksum"):
        await make_runner(tmp_path, ToyRuntimeFactory()).run(2, resume=True)


@pytest.mark.asyncio
async def test_resume_recovers_publication_when_latest_write_was_lost(tmp_path):
    runner = make_runner(tmp_path, ToyRuntimeFactory())
    await runner.run(1)
    (tmp_path / "run" / "latest.json").unlink()

    completed = await make_runner(tmp_path, ToyRuntimeFactory()).run(1, resume=True)
    assert completed == []
    latest = LatestPublication.model_validate_json(
        (tmp_path / "run" / "latest.json").read_text(encoding="utf-8")
    )
    run_manifest = json.loads(
        (tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert latest.policy_version == "policy_000001"
    assert run_manifest["status"] == "completed"
    assert run_manifest["completed_rounds"] == 1


def test_optimizer_checkpoint_version_must_match(tmp_path):
    state_path = tmp_path / "optimizer.pt"
    atomic_torch_save(state_path, {"version": 3, "optimizer": {}})
    state = load_torch_checkpoint(state_path)
    from recursive_agent_training.mini import build_toy_model
    from recursive_agent_training.optimizers.torch_optimizer import (
        TorchCISPOOptimizer,
        TorchOptimizerSettings,
    )

    optimizer = TorchCISPOOptimizer(build_toy_model(), TorchOptimizerSettings(), version=2)
    with pytest.raises(ValueError, match="does not match"):
        optimizer.load_optimizer_state_dict(state, expected_version=2)


def test_metric_delta_preserves_labels_and_subtracts_counters():
    result = SyncFlywheelRunner._metric_delta(
        {"requests": 5, "latency_seconds": 2.5, "provider": "local"},
        {"requests": 2, "latency_seconds": 1.0, "provider": "local"},
    )
    assert result == {
        "requests": 3,
        "latency_seconds": 1.5,
        "provider": "local",
    }
