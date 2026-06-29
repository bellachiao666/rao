import subprocess
import sys
import json

from examples.run_periodic_deepdive_training import (
    archive_incomplete_evaluation,
    completed_rounds,
    evaluation_complete,
    publication_for_target,
)
from recursive_agent_training.provenance import collect_source_manifest


def test_periodic_schedule_dry_run(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "examples/run_periodic_deepdive_training.py",
            "--output",
            str(tmp_path),
            "--rounds",
            "25",
            "--interval",
            "10",
            "--action-profile",
            "codeact",
            "--train-task-id",
            "7",
            "--train-task-id",
            "1022",
            "--dry-run",
        ],
        check=True,
    )
    payload = (tmp_path / "periodic_schedule.json").read_text(encoding="utf-8")
    assert '"milestones": [' in payload
    assert "25" in payload
    assert '"action_profile": "codeact"' in payload
    assert '"train_task_ids": [' in payload
    assert '"1022"' in payload


def test_source_manifest_is_stable_and_nonempty():
    first = collect_source_manifest(".")
    second = collect_source_manifest(".")
    assert first["source_hash"] == second["source_hash"]
    assert first["file_count"] > 10


def test_resume_helpers_use_requested_milestone_publication(tmp_path):
    training = tmp_path / "training"
    round_dir = training / "rounds" / "round_0029"
    adapter = round_dir / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "weights.bin").write_bytes(b"weights")
    (training / "run_manifest.json").write_text(
        json.dumps({"completed_rounds": 30}),
        encoding="utf-8",
    )
    (round_dir / "round_manifest.json").write_text(
        json.dumps(
            {
                "status": "published",
                "policy_version_after": "policy_000030",
                "adapter_path": str(adapter),
            }
        ),
        encoding="utf-8",
    )

    assert completed_rounds(training) == 30
    assert publication_for_target(training, 30) == (
        adapter,
        "policy_000030",
    )


def test_incomplete_evaluation_is_archived(tmp_path):
    output = tmp_path / "step_0030"
    output.mkdir()
    (output / "partial.json").write_text("{}", encoding="utf-8")

    archived = archive_incomplete_evaluation(output)

    assert archived is not None
    assert archived.exists()
    assert not output.exists()
    assert not evaluation_complete(archived)
