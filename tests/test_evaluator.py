import pytest

from recursive_agent_harness.evaluator import EvaluationResult, LLMJudge
from recursive_agent_harness.policy import ChatClient, ChatCompletionResult
from recursive_agent_harness.state import NodeStatus
from recursive_agent_harness.tree import ExecutionTree
from tests.helpers import make_config


class FakeJudgeClient(ChatClient):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages, model, temperature, max_tokens):
        self.calls.append(messages)
        return ChatCompletionResult(content=self.responses.pop(0), raw_response={"ok": True})


@pytest.mark.asyncio
async def test_llm_judge_evaluates_root_task_with_ground_truth():
    client = FakeJudgeClient(['{"reason":"Equivalent answer.","success":true}'])
    judge = LLMJudge(client=client, config=make_config())

    result = await judge.evaluate_root(
        task="Who wrote Hamlet?",
        ground_truth="William Shakespeare",
        agent_answer="Hamlet was written by William Shakespeare.",
    )

    assert result.reason == "Equivalent answer."
    assert result.success is True
    assert result.scope == "root"
    assert result.metadata["raw_response"] == '{"reason":"Equivalent answer.","success":true}'
    assert "ground truth answer" in client.calls[0][0]["content"]
    assert "Who wrote Hamlet?" in client.calls[0][1]["content"]


@pytest.mark.asyncio
async def test_llm_judge_evaluates_subtask_with_delegation_guidance():
    client = FakeJudgeClient(['{"reason":"Useful decomposition.","success":true}'])
    judge = LLMJudge(client=client, config=make_config())

    result = await judge.evaluate_subtask(
        parent_task="Plan a Kyoto trip",
        subtask="Find quiet temple",
        agent_answer="Shoren-in early morning.",
    )

    assert result.success is True
    assert result.scope == "subtask"
    assert "degenerate delegation behavior" in client.calls[0][0]["content"]
    assert "Find quiet temple" in client.calls[0][1]["content"]


@pytest.mark.asyncio
async def test_llm_judge_annotates_execution_tree_success_signal():
    tree = ExecutionTree(run_id="run_test")
    tree.register_node("node_0001", None, 0, "Root task")
    tree.register_node("node_0002", "node_0001", 1, "Sub task")
    tree.attach_child("node_0001", "node_0002")
    tree.update_node_status("node_0001", NodeStatus.COMPLETED, final_answer="root answer")
    tree.update_node_status("node_0002", NodeStatus.COMPLETED, final_answer="child answer")
    client = FakeJudgeClient(
        [
            '{"reason":"Root correct.","success":true}',
            '{"reason":"Child incomplete.","success":false}',
        ]
    )
    judge = LLMJudge(client=client, config=make_config())

    results = await judge.evaluate_tree(
        tree,
        ground_truth_by_node={
            "node_0001": "root answer",
            "node_0002": "better child answer",
        },
    )

    assert results["node_0001"].success is True
    assert results["node_0002"].success is False
    assert tree.nodes["node_0001"].metadata["llm_judge"]["success"] is True
    assert tree.nodes["node_0001"].metadata["success_signal"] == 1.0
    assert tree.nodes["node_0002"].metadata["llm_judge"]["success"] is False
    assert tree.nodes["node_0002"].metadata["success_signal"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        '{"reason":"Wrong type.","success":"false"}',
        '{"reason":"Missing success."}',
        '{"reason":123,"success":false}',
        '{"reason":"","success":false}',
        '[]',
    ],
)
async def test_llm_judge_rejects_invalid_result_types(response):
    judge = LLMJudge(client=FakeJudgeClient([response]), config=make_config())

    with pytest.raises(ValueError, match="judge (JSON field|response)"):
        await judge.evaluate_root("task", "truth", "answer")
