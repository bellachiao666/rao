from recursive_agent_harness.config import HarnessConfig
from recursive_agent_harness.tools import ToolRegistry


TEST_CONFIG = {
    "max_depth": 3,
    "max_steps_per_node": 12,
    "max_children_per_node": 6,
    "total_node_limit": 32,
    "timeout_seconds": 120,
    "invalid_action_limit": 2,
    "max_tool_calls_per_node": 6,
    "max_parallel_children": 4,
    "model_name": "test-model",
    "base_url": "https://example.test/v1",
    "api_key": "test-api-key",
    "temperature": 0.2,
    "max_tokens": 1200,
    "trace_output_path": "trace.json",
    "mermaid_output_path": "",
    "log_level": "INFO",
}


def make_config(**overrides) -> HarnessConfig:
    return HarnessConfig(**{**TEST_CONFIG, **overrides})


def mock_search(arguments: dict) -> list[dict[str, str]]:
    query = str(arguments.get("query", ""))
    return [
        {
            "title": f"Mock result for {query}",
            "snippet": f"Deterministic search fixture for {query}.",
            "url": "mock://search/result",
        }
    ]


def make_tool_registry_with_mock_search() -> ToolRegistry:
    registry = ToolRegistry.with_default_tools()
    registry.register_tool(
        "mock_search",
        mock_search,
        "Deterministic local search fixture for tests.",
        {"query": "string"},
    )
    return registry
