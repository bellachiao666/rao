import json

from recursive_agent_training.modelscope import inspect_modelscope_snapshot


def test_modelscope_snapshot_manifest_is_stable(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / ".mv").write_text(
        "Revision:0ce186d90797c4c5cfefb0d3695517b37d02dd3a,CreatedAt:1",
        encoding="utf-8",
    )
    first = inspect_modelscope_snapshot(
        tmp_path,
        model_id="Qwen/Qwen3-0.6B",
        revision="master",
    )
    second = inspect_modelscope_snapshot(
        tmp_path,
        model_id="Qwen/Qwen3-0.6B",
        revision="master",
    )
    assert first.resolved_revision == "0ce186d90797c4c5cfefb0d3695517b37d02dd3a"
    assert first.snapshot_hash == second.snapshot_hash
    assert ".msc" not in first.file_hashes
