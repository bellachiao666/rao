from examples.run_deepdive_rao_training import (
    preflight_config_matches_except_runtime_retry_fields,
)
from recursive_agent_training.config import TrainingConfig


def test_preflight_allows_collection_attempt_change_only():
    existing = TrainingConfig.from_yaml("configs/train/mini_rao.yaml").to_redacted_dict()
    current = TrainingConfig.from_yaml("configs/train/mini_rao.yaml").to_redacted_dict()
    current["sync_flywheel"]["max_collection_attempts"] += 5

    assert preflight_config_matches_except_runtime_retry_fields(existing, current)


def test_preflight_allows_concurrency_change_only():
    existing = TrainingConfig.from_yaml("configs/train/mini_rao.yaml").to_redacted_dict()
    current = TrainingConfig.from_yaml("configs/train/mini_rao.yaml").to_redacted_dict()
    current["rollout"]["max_concurrent_groups"] += 1

    assert preflight_config_matches_except_runtime_retry_fields(existing, current)


def test_preflight_rejects_training_semantic_changes():
    existing = TrainingConfig.from_yaml("configs/train/mini_rao.yaml").to_redacted_dict()
    current = TrainingConfig.from_yaml("configs/train/mini_rao.yaml").to_redacted_dict()
    current["optimizer"]["learning_rate"] *= 10

    assert not preflight_config_matches_except_runtime_retry_fields(existing, current)


def test_config_hash_matches_only_collection_attempt_changes():
    existing = TrainingConfig.from_yaml("configs/train/mini_rao.yaml")
    current = TrainingConfig.from_yaml("configs/train/mini_rao.yaml")
    current.sync_flywheel.max_collection_attempts += 5
    current.rollout.max_concurrent_groups += 1

    assert current.matches_hash_except_runtime_retry_fields(existing.stable_hash())

    changed_learning_rate = TrainingConfig.from_yaml("configs/train/mini_rao.yaml")
    changed_learning_rate.optimizer.learning_rate *= 10
    assert not current.matches_hash_except_runtime_retry_fields(
        changed_learning_rate.stable_hash()
    )
