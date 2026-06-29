import pytest
from pydantic import ValidationError

from recursive_agent_harness.actions import ActionType, AgentAction, SubtaskSpec
from recursive_agent_harness.config import HarnessConfig
from recursive_agent_harness.errors import DepthLimitError, HarnessError
from recursive_agent_harness.policy import Policy
from recursive_agent_harness.runner import Runner
from recursive_agent_harness.state import AgentState, ChildSummary, NodeResult, NodeStatus
from tests.helpers import make_config


class FinishPolicy(Policy):
    async def act(self, state: AgentState) -> AgentAction:
        return AgentAction(type=ActionType.FINISH, answer="done")


def test_config_values_must_come_from_yaml(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-should-not-be-used")

    with pytest.raises(ValidationError):
        HarnessConfig()


def test_runner_defaults_to_yaml_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "\n".join(
            [
                "max_depth: 1",
                "max_steps_per_node: 2",
                "max_children_per_node: 3",
                "total_node_limit: 4",
                "timeout_seconds: 5",
                "invalid_action_limit: 1",
                "max_tool_calls_per_node: 2",
                "max_parallel_children: 2",
                "model_name: yaml-model",
                "base_url: https://example.test/v1",
                "api_key: yaml-api-key",
                "temperature: 0.1",
                "max_tokens: 333",
                "trace_output_path: trace.json",
                "mermaid_output_path: trace.mmd",
                "log_level: DEBUG",
            ]
        ),
        encoding="utf-8",
    )

    runner = Runner(policy=FinishPolicy())

    assert runner.config.model_name == "yaml-model"
    assert runner.config.max_depth == 1
    assert runner.config.trace_output_path == "trace.json"


def test_config_loads_openai_api_settings_from_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-should-not-be-used")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
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
                "api_key: yaml-api-key",
                "temperature: 0.1",
                "max_tokens: 333",
                "trace_output_path: trace.json",
                "mermaid_output_path: trace.mmd",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )

    config = HarnessConfig.from_yaml(config_path)

    assert config.model_name == "gpt-test-model"
    assert config.base_url == "https://example.test/v1"
    assert config.api_key == "yaml-api-key"
    assert config.temperature == 0.1
    assert config.max_tokens == 333


def test_config_rejects_non_positive_limits():
    with pytest.raises(ValidationError):
        make_config(max_depth=-1)

    with pytest.raises(ValidationError):
        make_config(total_node_limit=0)


def test_errors_share_common_base_type():
    assert issubclass(DepthLimitError, HarnessError)


def test_finish_action_requires_answer():
    action = AgentAction(type=ActionType.FINISH, answer="done")

    assert action.type == ActionType.FINISH
    assert action.answer == "done"

    with pytest.raises(ValidationError):
        AgentAction(type=ActionType.FINISH)


def test_non_tool_action_rejects_tool_payload():
    with pytest.raises(ValidationError, match="tool_name"):
        AgentAction(
            type=ActionType.THINK,
            thought="search next",
            tool_name="search_web",
        )


def test_launch_subagent_requires_subtask():
    subtask = SubtaskSpec(goal="Check crowd risk", expected_output="short assessment")
    action = AgentAction(type=ActionType.LAUNCH_SUBAGENT, subtask=subtask)

    assert action.subtask.goal == "Check crowd risk"

    with pytest.raises(ValidationError):
        AgentAction(type=ActionType.LAUNCH_SUBAGENT)


def test_parallel_launch_requires_non_empty_subtasks():
    with pytest.raises(ValidationError):
        AgentAction(type=ActionType.LAUNCH_SUBAGENTS_PARALLEL, subtasks=[])


def test_agent_state_and_node_result_are_json_serializable():
    state = AgentState(
        node_id="node_0001",
        parent_id=None,
        task="Plan a trip",
        depth=0,
        max_depth=3,
        remaining_steps=5,
        remaining_children=2,
        parent_context={},
        expected_output="final answer",
        constraints={"avoid_crowds": True},
        trajectory=[],
        child_summaries=[
            ChildSummary(
                node_id="node_0002",
                task="Find temple",
                status=NodeStatus.COMPLETED,
                answer="Shoren-in",
            )
        ],
        available_tools=[{"name": "mock_search", "description": "Search fixture"}],
        global_tree_summary={"total_nodes": 2},
    )
    result = NodeResult(
        node_id="node_0001",
        task="Plan a trip",
        status=NodeStatus.COMPLETED,
        answer="A concise itinerary",
        success_signal=None,
        reward=None,
    )

    assert state.model_dump()["child_summaries"][0]["answer"] == "Shoren-in"
    assert result.model_dump()["success_signal"] is None
    assert result.model_dump()["reward"] is None
