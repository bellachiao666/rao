import recursive_agent_harness
import recursive_agent_harness.policy as policy_module
from recursive_agent_harness.policy import LLMPolicy, OpenAICompatibleChatClient


def test_rule_based_policy_is_not_part_of_public_api():
    assert not hasattr(policy_module, "RuleBasedPolicy")
    assert not hasattr(policy_module, "ScriptedPolicy")
    assert not hasattr(policy_module, "InvalidActionPolicy")
    assert "RuleBasedPolicy" not in recursive_agent_harness.__all__


def test_llm_policy_defaults_to_openai_compatible_client():
    policy = LLMPolicy(config=None)

    assert isinstance(policy.client, OpenAICompatibleChatClient)
