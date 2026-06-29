import pytest

from recursive_agent_training.schemas import TokenRecord


def test_token_record_requires_aligned_arrays():
    with pytest.raises(ValueError, match="action_mask"):
        TokenRecord(generated_ids=[1, 2], action_mask=[1], behavior_logprobs=[-1, -1])


def test_token_record_round_trip():
    record = TokenRecord(
        input_ids=[1],
        generated_ids=[2],
        action_mask=[1],
        behavior_logprobs=[-0.5],
        generated_text="x",
    )
    assert TokenRecord.model_validate_json(record.model_dump_json()) == record
