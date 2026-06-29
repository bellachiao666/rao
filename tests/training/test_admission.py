import json

from recursive_agent_training.admission import audit_training_round


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _round(tmp_path, answer, subtask, ground_truth="Roversi"):
    directory = tmp_path / "round_0000"
    scored = directory / "scored_groups" / "group_0000.json"
    batch = directory / "optimizer_batch" / "batch.json"
    _write(
        scored,
        {
            "task": {"ground_truth": ground_truth},
            "rollouts": [
                {
                    "rollout_id": "rollout",
                    "root_node_id": "root",
                    "nodes": {
                        "root": {
                            "node_id": "root",
                            "depth": 0,
                            "node_task": "Find the surname.",
                            "final_answer": "unknown",
                            "is_fallback": False,
                            "evaluation": {"value": 0.0, "reason": "wrong"},
                        },
                        "child": {
                            "node_id": "child",
                            "depth": 1,
                            "node_task": subtask,
                            "final_answer": answer,
                            "is_fallback": False,
                            "evaluation": {"value": 1.0, "reason": "judge positive"},
                        },
                    },
                }
            ],
        },
    )
    _write(
        batch,
        {
            "samples": [
                {
                    "rollout_id": "rollout",
                    "node_id": "child",
                    "advantage": 1.0,
                }
            ]
        },
    )
    _write(
        directory / "round_manifest.json",
        {
            "status": "published",
            "policy_version_after": "policy_000001",
            "scored_group_paths": [str(scored)],
            "optimizer_batch_path": str(batch),
            "metrics": {"judge_service": {"terminal_failures": 0}},
        },
    )
    return directory


def test_admission_rejects_vague_judge_positive(tmp_path):
    directory = _round(
        tmp_path,
        "the current leader is identified and the task is complete",
        "narrow delegated task",
    )
    report = audit_training_round(directory)
    assert report["accepted"] is False
    assert report["trusted_positive_reward_count"] == 0


def test_admission_accepts_specific_subtask_result(tmp_path):
    directory = _round(
        tmp_path,
        "The leader's surname is Roversi.",
        "Determine the current leader's surname from the evidence.",
    )
    report = audit_training_round(directory)
    assert report["accepted"] is True
    assert report["trusted_positive_reward_count"] == 1


def test_admission_rejects_no_result_judge_positive(tmp_path):
    directory = _round(
        tmp_path,
        "None of the search results provided a clear matching title. "
        "The maximum depth was reached before a specific match was identified.",
        "Examine search results and identify a matching paper title.",
    )
    report = audit_training_round(directory)
    assert report["accepted"] is False
    assert report["trusted_positive_reward_count"] == 0


def test_admission_rejects_depth_limited_no_result_paraphrase(tmp_path):
    directory = _round(
        tmp_path,
        "No initial academic journals were identified within the allowed depth "
        "limit due to the search results. Further research may be required.",
        "Identify initial relevant academic journals.",
    )
    report = audit_training_round(directory)
    assert report["accepted"] is False
    assert report["trusted_positive_reward_count"] == 0


def test_admission_rejects_no_successful_child_synthesis(tmp_path):
    directory = _round(
        tmp_path,
        "Synthesis\n\nTask: find relevant scientific publications related to "
        "neurological conditions in Nouvelle-Aquitaine\n\nFinal answer from "
        "child results:\nNo child returned a successful answer.\n\nLimitations:\n"
        "- narrow delegated task: Cannot launch child because max_depth has "
        "been reached.",
        "Find relevant scientific publications related to neurological "
        "conditions in Nouvelle-Aquitaine.",
        "Cognitive Evaluation by Tasks in a Virtual Reality Environment in "
        "Multiple Sclerosis",
    )
    report = audit_training_round(directory)
    assert report["accepted"] is False
    assert report["trusted_positive_reward_count"] == 0


def test_admission_rejects_depth_error_as_auxiliary_success(tmp_path):
    directory = _round(
        tmp_path,
        "Cannot launch child because max_depth has been reached.",
        "Alternative pedagogical models collective.",
        "Flipped Learning as a New Educational Paradigm",
    )
    report = audit_training_round(directory)
    assert report["accepted"] is False
    assert report["trusted_positive_reward_count"] == 0


def test_admission_rejects_irrelevant_concrete_title(tmp_path):
    directory = _round(
        tmp_path,
        "A New Method of Surface Acoustic Wave Pressure Sensing Based on "
        "Acoustic Attenuation Mechanism",
        "Examine search results to identify a potential matching title.",
        "Grains of Saws: Associating Quasi-Particles to Surface Acoustic Waves",
    )
    report = audit_training_round(directory)
    assert report["accepted"] is False
    assert report["trusted_positive_reward_count"] == 0


def test_admission_accepts_root_reward_from_trusted_child(tmp_path):
    directory = _round(
        tmp_path,
        "The leader's surname is Roversi.",
        "Determine the current leader's surname from the evidence.",
    )
    scored = directory / "scored_groups" / "group_0000.json"
    payload = json.loads(scored.read_text(encoding="utf-8"))
    root = payload["rollouts"][0]["nodes"]["root"]
    root["children_ids"] = ["child"]
    root["credit"] = {
        "reward": {
            "own_success": 0.0,
            "child_success_mean": 1.0,
            "reward": 1.0,
        }
    }
    scored.write_text(json.dumps(payload), encoding="utf-8")
    batch = directory / "optimizer_batch" / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "rollout_id": "rollout",
                        "node_id": "root",
                        "advantage": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = audit_training_round(directory)
    assert report["accepted"] is True
    assert report["trusted_reward_node_count"] == 2


def test_admission_accepts_sparse_proxy_from_trusted_root(tmp_path):
    directory = _round(
        tmp_path,
        "irrelevant child answer",
        "Investigate an auxiliary clue.",
        "Roversi",
    )
    scored = directory / "scored_groups" / "group_0000.json"
    payload = json.loads(scored.read_text(encoding="utf-8"))
    nodes = payload["rollouts"][0]["nodes"]
    nodes["root"]["final_answer"] = "Roversi"
    nodes["root"]["evaluation"] = {
        "value": 1.0,
        "reason": "sparse reward ablation proxy from root success",
        "is_proxy": True,
    }
    nodes["child"]["evaluation"] = {
        "value": 1.0,
        "reason": "sparse reward ablation proxy from root success",
        "is_proxy": True,
    }
    scored.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_training_round(directory)

    assert report["accepted"] is True
    assert report["trusted_positive_reward_count"] == 2
    assert all(node["is_proxy"] for node in report["positive_nodes"])


def test_admission_rejects_sparse_proxy_from_untrusted_root(tmp_path):
    directory = _round(
        tmp_path,
        "irrelevant child answer",
        "Investigate an auxiliary clue.",
        "Roversi",
    )
    scored = directory / "scored_groups" / "group_0000.json"
    payload = json.loads(scored.read_text(encoding="utf-8"))
    nodes = payload["rollouts"][0]["nodes"]
    nodes["root"]["final_answer"] = "unknown"
    nodes["root"]["evaluation"] = {
        "value": 1.0,
        "reason": "sparse reward ablation proxy from root success",
        "is_proxy": True,
    }
    nodes["child"]["evaluation"] = {
        "value": 1.0,
        "reason": "sparse reward ablation proxy from root success",
        "is_proxy": True,
    }
    scored.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_training_round(directory)

    assert report["accepted"] is False
    assert report["trusted_positive_reward_count"] == 0
    assert any(
        "proxy reward lacks a trusted positive root success signal" in failure
        for failure in report["failures"]
    )
