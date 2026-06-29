import pytest

from recursive_agent_training.config import TrainingConfig


def test_training_config_loads_and_hashes_stably():
    config = TrainingConfig.from_yaml("configs/train/mini_rao.yaml")
    assert config.rollout.group_size == 2
    assert config.stable_hash() == config.stable_hash()


def test_group_size_one_is_rejected():
    with pytest.raises(ValueError, match="group_size"):
        TrainingConfig.model_validate({"rollout": {"group_size": 1}})


def test_non_positive_rollout_timeout_is_rejected():
    with pytest.raises(ValueError, match="batch and step limits"):
        TrainingConfig.model_validate({"rollout": {"timeout_seconds": 0}})


def test_faithful_deepdive_config_matches_paper_values():
    config = TrainingConfig.from_yaml("configs/train/deepdive_existing_harness.yaml")
    assert config.rollout.root_batch_size == 16
    assert config.rollout.group_size == 8
    assert config.rollout.max_depth == 4
    assert config.reward.delegation_lambda == 0
    assert config.model.train_context_tokens == 40960
    assert config.model.eval_context_tokens == 262144


def test_hardware_adapted_modelscope_config_loads():
    config = TrainingConfig.from_yaml("configs/train/deepdive_qwen06b_s1.yaml")
    assert config.model.source == "modelscope"
    assert config.model.model_id == "Qwen/Qwen3-0.6B"
    assert config.rollout.root_batch_size == 4
    assert config.rollout.group_size == 4
    assert config.rollout.max_staleness_batches == 3
    assert config.rollout.timeout_seconds == 900
    assert config.reward.delegation_lambda == 0.2
    assert config.optimizer.learning_rate == 3e-5
    assert config.sync_flywheel.ratio_min == 0.95
    assert config.sync_flywheel.ratio_max == 1.05
    assert config.model.train_context_tokens == 8192
    assert config.services.search.provider == "duckduckgo"
    assert config.services.judge.provider == "local_transformers"


def test_qwen3_4b_oom_probe_config_loads():
    config = TrainingConfig.from_yaml("configs/train/deepdive_qwen3_4b_s0.yaml")
    assert config.model.source == "modelscope"
    assert config.model.model_id == "Qwen/Qwen3-4B"
    assert config.rollout.root_batch_size == 1
    assert config.rollout.group_size == 4
    assert config.reward.delegation_lambda == 0.2
