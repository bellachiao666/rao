from recursive_agent_training.config import TrainingConfig
from recursive_agent_training.model_admission import (
    freeze_selected_codeact_s0_config,
    freeze_selected_s1_config,
    select_first_accepted,
)
from examples.run_model_admission import DEFAULT_CANDIDATES
from examples.run_deepdive_rao_training import select_requested_train_records
from recursive_agent_training.schemas import RootTaskRecord


def test_default_candidates_scale_up_in_admission_order():
    assert DEFAULT_CANDIDATES == (
        "configs/train/deepdive_qwen06b_s0.yaml",
        "configs/train/deepdive_qwen17b_s0.yaml",
        "configs/train/deepdive_qwen3_4b_s0.yaml",
        "configs/train/deepdive_qwen25_3b_s0.yaml",
    )


def test_select_first_accepted_respects_candidate_order():
    selected = select_first_accepted(
        [
            {"model_id": "small", "accepted": False},
            {"model_id": "medium", "accepted": True},
            {"model_id": "large", "accepted": True},
        ]
    )
    assert selected["model_id"] == "medium"


def test_freeze_selected_s1_uses_candidate_identity_and_template_scale(tmp_path):
    output = tmp_path / "selected.yaml"
    selected = freeze_selected_s1_config(
        "configs/train/deepdive_qwen17b_s0.yaml",
        "configs/train/deepdive_qwen06b_s1.yaml",
        output,
        checkpoint_output=tmp_path / "training",
    )
    reloaded = TrainingConfig.from_yaml(output)
    assert selected.model.model_id == "Qwen/Qwen3-1.7B"
    assert reloaded.model.path.endswith("Qwen3-1___7B")
    assert reloaded.model.train_context_tokens == 8192
    assert reloaded.model.eval_context_tokens == 16384
    assert reloaded.rollout.root_batch_size == 4
    assert reloaded.reward.delegation_lambda == 0.2
    assert reloaded.rollout.max_staleness_batches == 3
    assert reloaded.optimizer.learning_rate == 3e-5
    assert reloaded.sync_flywheel.ratio_min == 0.95
    assert reloaded.sync_flywheel.ratio_max == 1.05
    assert reloaded.experiment.action_profile == "codeact"
    assert reloaded.parameter_sources["model"] == "modelscope_admission_selected"


def test_freeze_selected_codeact_s0_preserves_scale_and_changes_runtime(tmp_path):
    output = tmp_path / "selected_codeact_s0.yaml"
    selected = freeze_selected_codeact_s0_config(
        "configs/train/deepdive_qwen25_3b_s0.yaml",
        output,
        checkpoint_output=tmp_path / "codeact_training",
    )
    reloaded = TrainingConfig.from_yaml(output)
    assert selected.model.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert reloaded.rollout.root_batch_size == 1
    assert reloaded.rollout.group_size == 4
    assert reloaded.rollout.max_depth == 1
    assert reloaded.reward.delegation_lambda == 0.2
    assert reloaded.experiment.action_profile == "codeact"
    assert reloaded.parameter_sources["model"] == "modelscope_admission_selected"
    assert reloaded.parameter_sources["rollout"] == "hardware_adapted_codeact_s0"


def test_select_requested_train_records_accepts_source_or_stable_id():
    records = [
        RootTaskRecord(
            task_id="deepdive:hash:100",
            domain="deepdive",
            split="train",
            prompt="question",
            ground_truth="answer",
            metadata={"source_row_id": "100"},
        ),
        RootTaskRecord(
            task_id="deepdive:hash:101",
            domain="deepdive",
            split="train",
            prompt="other question",
            ground_truth="other answer",
            metadata={"source_row_id": "101"},
        ),
    ]
    assert select_requested_train_records(records, ["100"]) == [records[0]]
    assert select_requested_train_records(
        records,
        ["deepdive:hash:101"],
    ) == [records[1]]


def test_select_requested_train_records_rejects_eval_or_unknown_id():
    record = RootTaskRecord(
        task_id="deepdive:hash:100",
        domain="deepdive",
        split="train",
        prompt="question",
        ground_truth="answer",
        metadata={"source_row_id": "100"},
    )
    import pytest

    with pytest.raises(ValueError, match="absent from the fixed train split"):
        select_requested_train_records([record], ["999"])
