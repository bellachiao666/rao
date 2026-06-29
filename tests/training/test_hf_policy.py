import asyncio
from types import SimpleNamespace
import threading
import time

import pytest

from recursive_agent_training.hf_policy import (
    HuggingFaceTrainingPolicy,
    adapt_action_to_state,
    extract_json_object,
    parse_generated_action,
    validate_input_token_ids,
)


def test_extract_json_object_skips_model_thinking_text():
    text = '<think>brief reasoning</think>\n{"type":"FINISH","answer":"alpha"}'
    assert extract_json_object(text) == '{"type":"FINISH","answer":"alpha"}'


def test_extract_json_object_rejects_non_json_output():
    with pytest.raises(ValueError, match="complete JSON object"):
        extract_json_object("alpha")


def test_parse_generated_action_fills_semantic_fields_from_reason():
    thought = parse_generated_action('{"type":"THINK","reason":"search next"}')
    aggregate = parse_generated_action('{"type":"AGGREGATE","reason":"combine children"}')
    assert thought.thought == "search next"
    assert aggregate.instructions == "combine children"


def test_parse_generated_action_normalizes_small_model_search_aliases():
    alias = parse_generated_action('{"type":"SEARCH_WEB","query":"paper title"}')
    mixed = parse_generated_action(
        '{"type":"THINK","reason":"search","tool_name":"search_web",'
        '"arguments":{"query":"paper title"}}'
    )
    assert alias.type.value == "TOOL_CALL"
    assert alias.tool_name == "search_web"
    assert alias.arguments["query"] == "paper title"
    assert mixed.type.value == "TOOL_CALL"
    assert mixed.metadata["parser_normalizations"] == [
        "tool_payload_forced_tool_call"
    ]


def test_search_action_is_augmented_with_task_context():
    action = parse_generated_action('{"type":"SEARCH_WEB","query":"surface waves"}')
    adapted = adapt_action_to_state(
        action,
        SimpleNamespace(task="Find the precise paper title involving grains."),
    )
    assert "Task context:" in adapted.arguments["query"]
    assert "grains" in adapted.arguments["query"]
    assert adapted.metadata["search_query_augmented_with_task"] is True


def test_specific_search_query_is_not_polluted_with_full_task_context():
    query = "2002 International Journal Control discrete-time H2 controller"
    action = parse_generated_action(
        '{"type":"SEARCH_WEB","query":"'
        + query
        + '"}'
    )
    adapted = adapt_action_to_state(
        action,
        SimpleNamespace(
            task=(
                "Identify the precise title using a long set of biographical "
                "and publication clues."
            )
        ),
    )
    assert adapted.arguments["query"] == query
    assert "search_query_augmented_with_task" not in adapted.metadata


def test_validate_input_token_ids_rejects_embedding_overflow():
    torch = pytest.importorskip("torch")
    model = SimpleNamespace(
        get_input_embeddings=lambda: SimpleNamespace(num_embeddings=8)
    )
    validate_input_token_ids(torch.tensor([[0, 7]]), model, source="test")
    with pytest.raises(ValueError, match="max=8"):
        validate_input_token_ids(torch.tensor([[1, 8]]), model, source="test")


@pytest.mark.asyncio
async def test_threaded_generation_runs_off_event_loop_and_respects_lock():
    policy = object.__new__(HuggingFaceTrainingPolicy)
    policy.threaded_generation = True
    policy.generation_lock = threading.Lock()
    main_thread = threading.get_ident()
    active = 0
    max_active = 0

    def generate(messages):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        time.sleep(0.02)
        active -= 1
        return {"thread": threading.get_ident(), "messages": messages}

    policy._generate = generate
    first, second = await asyncio.gather(
        policy._generate_async([{"role": "user", "content": "a"}]),
        policy._generate_async([{"role": "user", "content": "b"}]),
    )
    assert first["thread"] != main_thread
    assert second["thread"] != main_thread
    assert max_active == 1
