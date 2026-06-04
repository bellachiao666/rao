"""Run the recursive agent harness with OpenAI-compatible API policy."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recursive_agent_harness.config import HarnessConfig
from recursive_agent_harness.policy import LLMPolicy
from recursive_agent_harness.runner import Runner


KYOTO_TASK = (
    "Plan a 3-day Kyoto trip in early April for a family. "
    "We want cherry blossoms, one quiet temple, one kid-friendly activity, "
    "and dinner near Gion. Avoid overly crowded spots."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recursive agent harness with YAML OpenAI API config.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        raise SystemExit(2)

    config = HarnessConfig.from_yaml(config_path)
    if not config.api_key or config.api_key == "YOUR_OPENAI_API_KEY":
        print("api_key must be set in config.yaml", file=sys.stderr)
        raise SystemExit(2)
    runner = Runner(policy=LLMPolicy(config=config), config=config)
    result, tree = runner.run(KYOTO_TASK)

    print("Final answer:")
    print(result.answer)
    print()
    print("Execution tree:")
    print(tree.pretty_print())
    print()
    print(f"Trace saved to {config.trace_output_path}")
    print(f"Mermaid graph saved to {config.mermaid_output_path}")


if __name__ == "__main__":
    main()
