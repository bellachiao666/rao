"""JSONL metrics logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recursive_agent_training.schemas import utc_now_iso


class MetricsLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any]) -> None:
        record = {"timestamp": utc_now_iso(), **metrics}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
