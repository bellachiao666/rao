"""Dataset loading for RAO root tasks."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from recursive_agent_training.schemas import RootTaskRecord


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_deepdive_csv(path: str | Path, split: str = "train", domain: str = "deepdive") -> list[RootTaskRecord]:
    source = Path(path)
    dataset_hash = file_sha256(source)
    records: list[RootTaskRecord] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "question", "answer", "conversations"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"DeepDive CSV must contain columns: {sorted(required)}")
        for row in reader:
            source_id = str(row["id"]).strip()
            question = str(row["question"]).strip()
            if source_id in seen_ids:
                raise ValueError(f"duplicate DeepDive id: {source_id}")
            if question in seen_questions:
                raise ValueError(f"duplicate DeepDive question: {question[:80]}")
            seen_ids.add(source_id)
            seen_questions.add(question)
            records.append(
                RootTaskRecord(
                    task_id=f"{domain}:{dataset_hash[:12]}:{source_id}",
                    domain=domain,
                    split=split,
                    prompt=question,
                    ground_truth=str(row["answer"]).strip(),
                    metadata={"source_row_id": source_id, "conversations": row.get("conversations", "")},
                    dataset_hash=dataset_hash,
                )
            )
    return records
