import json

from recursive_agent_harness.actions import ActionType, AgentAction
from recursive_agent_harness.state import NodeStatus
from recursive_agent_harness.tree import ExecutionTree


def test_execution_tree_records_nodes_edges_and_trajectory(tmp_path):
    tree = ExecutionTree(run_id="run_test")
    tree.register_node("node_0001", None, 0, "Root task")
    tree.register_node("node_0002", "node_0001", 1, "Child task")
    tree.attach_child("node_0001", "node_0002")

    action = AgentAction(type=ActionType.THINK, thought="Need a child result")
    tree.append_action("node_0001", action)
    tree.append_observation("node_0001", {"type": "thought", "content": "Need a child result"})
    tree.update_node_status("node_0002", NodeStatus.COMPLETED, final_answer="child answer")
    tree.update_node_status("node_0001", NodeStatus.COMPLETED, final_answer="root answer")

    output_path = tmp_path / "trace.json"
    tree.export_json(output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run_test"
    assert data["root_node_id"] == "node_0001"
    assert data["nodes"]["node_0001"]["children_ids"] == ["node_0002"]
    assert data["nodes"]["node_0001"]["trajectory"][0]["action"]["type"] == "THINK"


def test_execution_tree_pretty_print_mermaid_and_summary():
    tree = ExecutionTree(run_id="run_test")
    tree.register_node("node_0001", None, 0, "Root task")
    tree.register_node("node_0002", "node_0001", 1, "Child task")
    tree.attach_child("node_0001", "node_0002")
    tree.update_node_status("node_0001", NodeStatus.COMPLETED)
    tree.update_node_status("node_0002", NodeStatus.FAILED, error="bad child")

    pretty = tree.pretty_print()
    mermaid = tree.export_mermaid()
    summary = tree.summary()

    assert 'node_0001 depth=0 status=completed task="Root task"' in pretty
    assert '  node_0002 depth=1 status=failed task="Child task"' in pretty
    assert "graph TD" in mermaid
    assert "node_0001 --> node_0002" in mermaid
    assert summary["total_nodes"] == 2
    assert summary["depth_histogram"] == {0: 1, 1: 1}
    assert summary["status_counts"]["failed"] == 1


def test_mark_unfinished_sets_running_nodes_to_timeout():
    tree = ExecutionTree(run_id="run_test")
    tree.register_node("node_0001", None, 0, "Root task")
    tree.register_node("node_0002", "node_0001", 1, "Child task")

    tree.mark_unfinished(NodeStatus.TIMEOUT, "Run timed out")

    assert tree.nodes["node_0001"].status == NodeStatus.TIMEOUT
    assert tree.nodes["node_0002"].status == NodeStatus.TIMEOUT
    assert tree.nodes["node_0001"].error == "Run timed out"
