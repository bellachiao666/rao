import pytest

from recursive_agent_harness.tools import ToolRegistry


@pytest.mark.asyncio
async def test_default_tools_support_calculator_mock_search_and_read_text(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello harness", encoding="utf-8")
    registry = ToolRegistry.with_default_tools()

    calc = await registry.call("calculator", {"expression": "2 + 3 * 4"})
    search = await registry.call("mock_search", {"query": "Kyoto blossoms"})
    text = await registry.call("read_text", {"path": str(sample)})

    assert calc.ok is True
    assert calc.result == 14
    assert search.ok is True
    assert "Kyoto blossoms" in search.result[0]["title"]
    assert text.ok is True
    assert text.result == "hello harness"


@pytest.mark.asyncio
async def test_unknown_tool_returns_structured_error():
    registry = ToolRegistry()

    observation = await registry.call("missing_tool", {})

    assert observation.ok is False
    assert observation.tool_name == "missing_tool"
    assert "not registered" in observation.error
