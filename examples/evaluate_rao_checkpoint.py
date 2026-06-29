"""Aggregate per-task evaluation JSON records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_training.evaluation import summarize_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    args = parser.parse_args()
    records = json.loads(Path(args.results).read_text(encoding="utf-8"))
    print(json.dumps(summarize_evaluation(records).__dict__, indent=2))


if __name__ == "__main__":
    main()
