"""Create a deterministic DeepDive train/eval split manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_training.datasets import load_deepdive_csv
from recursive_agent_training.split import build_split_manifest, split_hash, write_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/deepdive_qa_rl.csv")
    parser.add_argument("--output", default="output/train/deepdive_split.json")
    parser.add_argument("--eval-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = load_deepdive_csv(args.input)
    manifest = build_split_manifest(records, args.eval_count, args.seed)
    write_split_manifest(args.output, manifest)
    print(f"Wrote {args.output}")
    print(f"train={len(manifest.train_task_ids)} eval={len(manifest.eval_task_ids)} hash={split_hash(manifest)}")


if __name__ == "__main__":
    main()
