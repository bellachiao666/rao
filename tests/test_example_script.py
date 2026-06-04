import subprocess
import sys
from pathlib import Path


def test_example_script_requires_yaml_api_key(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "examples" / "run_recursive_agent.py"
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-should-not-be-used")
    (tmp_path / "config.yaml").write_text(
        "\n".join(
            [
                "max_depth: 3",
                "max_steps_per_node: 12",
                "max_children_per_node: 6",
                "total_node_limit: 32",
                "timeout_seconds: 120",
                "invalid_action_limit: 2",
                "max_tool_calls_per_node: 6",
                "max_parallel_children: 4",
                "model_name: gpt-test-model",
                "base_url: https://example.test/v1",
                "api_key: YOUR_OPENAI_API_KEY",
                "temperature: 0.1",
                "max_tokens: 333",
                "trace_output_path: trace.json",
                "mermaid_output_path: trace.mmd",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "api_key must be set in config.yaml" in completed.stderr
    assert not (tmp_path / "trace.json").exists()
    assert not (tmp_path / "trace.mmd").exists()
