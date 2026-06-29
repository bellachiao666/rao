from pathlib import Path

import pytest

from recursive_agent_training.datasets import load_deepdive_csv
from recursive_agent_training.split import (
    build_split_manifest,
    read_split_manifest,
    select_split_records,
    write_split_manifest,
)


def test_deepdive_loader_and_split(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "id,question,answer,conversations\n"
        '1,"q1","a1",[]\n'
        '2,"q2","a2",[]\n'
        '3,"q3","a3",[]\n',
        encoding="utf-8",
    )
    records = load_deepdive_csv(path)
    manifest = build_split_manifest(records, eval_count=1, seed=7)
    assert len(manifest.train_task_ids) == 2
    assert len(manifest.eval_task_ids) == 1
    assert records[0].ground_truth not in records[0].prompt


def test_duplicate_question_is_rejected(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "id,question,answer,conversations\n"
        '1,"same","a1",[]\n'
        '2,"same","a2",[]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_deepdive_csv(path)


def test_split_manifest_round_trip_and_selection(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "id,question,answer,conversations\n"
        '1,"question one","answer one",[]\n'
        '2,"question two","answer two",[]\n'
        '3,"question three","answer three",[]\n',
        encoding="utf-8",
    )
    records = load_deepdive_csv(path)
    manifest = build_split_manifest(records, eval_count=1, seed=42)
    manifest_path = tmp_path / "split.json"
    write_split_manifest(manifest_path, manifest)
    loaded = read_split_manifest(manifest_path)
    train = select_split_records(records, loaded, "train")
    evaluation = select_split_records(records, loaded, "eval")
    assert len(train) == 2
    assert len(evaluation) == 1
    assert {item.split for item in evaluation} == {"eval"}
