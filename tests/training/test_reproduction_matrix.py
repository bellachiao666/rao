import subprocess
import sys
import json

from recursive_agent_training.config import TrainingConfig


def test_reproduction_matrix_prepares_all_condition_configs(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "examples/run_reproduction_matrix.py",
            "--output",
            str(tmp_path),
            "--rounds",
            "10",
            "--eval-tasks",
            "10",
            "--eval-rollouts",
            "4",
            "--train-task-id",
            "7",
            "--train-task-id",
            "1022",
        ],
        check=True,
    )
    expected = {
        "single_trained",
        "dense_weighted",
        "dense_unweighted",
        "sparse_weighted",
        "sparse_unweighted",
    }
    paths = {path.stem: path for path in (tmp_path / "configs").glob("*.yaml")}
    assert set(paths) == expected
    sparse = TrainingConfig.from_yaml(paths["sparse_weighted"])
    sparse_unweighted = TrainingConfig.from_yaml(paths["sparse_unweighted"])
    single = TrainingConfig.from_yaml(paths["single_trained"])
    assert sparse.reward.subtask_provider == "root_proxy"
    assert sparse.sync_flywheel.max_collection_attempts == 32
    assert sparse.rollout.max_concurrent_groups == 4
    assert sparse_unweighted.sync_flywheel.max_collection_attempts == 32
    assert sparse_unweighted.rollout.max_concurrent_groups == 4
    assert single.rollout.max_depth == 0
    assert single.rollout.max_steps_per_node == 8
    assert single.reward.delegation_lambda == 0.0
    assert sparse.reward.delegation_lambda == 0.2
    assert sparse.rollout.max_staleness_batches == 3
    assert sparse.optimizer.learning_rate == 3e-5
    assert sparse.sync_flywheel.ratio_min == 0.95
    assert sparse.sync_flywheel.ratio_max == 1.05
    manifest = json.loads(
        (tmp_path / "matrix_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_id"] == "Qwen/Qwen3-0.6B"
    assert manifest["train_task_ids"] == ["7", "1022"]
    assert single.experiment.name.startswith("qwen3_0_6b_")


def test_reproduction_matrix_rejects_tiny_filtered_main_run(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/run_reproduction_matrix.py",
            "--output",
            str(tmp_path),
            "--execute-training",
            "--rounds",
            "75",
            "--train-task-id",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "refusing 75-step main run" in completed.stderr
