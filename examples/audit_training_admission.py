"""Audit the latest published round before resume or longer training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_training.admission import audit_training_round, latest_round_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--round-dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    round_dir = (
        Path(args.round_dir).expanduser().resolve()
        if args.round_dir
        else latest_round_dir(args.run_dir)
    )
    report = audit_training_round(round_dir)
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
