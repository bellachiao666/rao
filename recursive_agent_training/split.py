"""Deterministic dataset split manifests."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from recursive_agent_training.schemas import RootTaskRecord


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0.0"
    dataset_hash: str
    seed: int
    train_task_ids: list[str]
    eval_task_ids: list[str]
    eval_count: int


def build_split_manifest(records: list[RootTaskRecord], eval_count: int, seed: int) -> SplitManifest:
    if not records:
        raise ValueError("dataset is empty")
    if eval_count <= 0 or eval_count >= len(records):
        raise ValueError("eval_count must be positive and smaller than the dataset")
    dataset_hashes = {record.dataset_hash for record in records}
    if len(dataset_hashes) != 1:
        raise ValueError("all records must originate from the same dataset")
    ids = sorted(record.task_id for record in records)
    random.Random(seed).shuffle(ids)
    eval_ids = sorted(ids[:eval_count])
    train_ids = sorted(ids[eval_count:])
    return SplitManifest(
        dataset_hash=next(iter(dataset_hashes)),
        seed=seed,
        train_task_ids=train_ids,
        eval_task_ids=eval_ids,
        eval_count=eval_count,
    )


def write_split_manifest(path: str | Path, manifest: SplitManifest) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")


def read_split_manifest(path: str | Path) -> SplitManifest:
    return SplitManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def select_split_records(
    records: list[RootTaskRecord],
    manifest: SplitManifest,
    split: str,
) -> list[RootTaskRecord]:
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    ids = manifest.train_task_ids if split == "train" else manifest.eval_task_ids
    by_id = {record.task_id: record for record in records}
    missing = [task_id for task_id in ids if task_id not in by_id]
    if missing:
        raise ValueError(f"split references {len(missing)} missing task ids")
    if records and manifest.dataset_hash != records[0].dataset_hash:
        raise ValueError("split dataset hash does not match loaded records")
    return [by_id[task_id].model_copy(update={"split": split}) for task_id in ids]


def split_hash(manifest: SplitManifest) -> str:
    payload = json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
