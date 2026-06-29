import pytest

from recursive_agent_training.schemas import RootTaskRecord, RolloutTreeRecord, TrainingNodeRecord, TreeStatus
from recursive_agent_training.verifiers.exact import ExactMatchVerifier
from recursive_agent_training.verifiers.llm import (
    root_answer_passes_lexical_guard,
    subtask_answer_passes_minimum_guard,
)
from recursive_agent_training.verifiers.proxy import LazyRootProxyVerifier


@pytest.mark.asyncio
async def test_exact_verifier_normalizes_answer():
    task = RootTaskRecord(
        task_id="t",
        domain="test",
        split="train",
        prompt="q",
        ground_truth="RackTest",
    )
    node = TrainingNodeRecord(
        task_id="t",
        group_id="g",
        rollout_id="r",
        node_id="root",
        parent_id=None,
        depth=0,
        node_task="q",
        terminal_status="completed",
        terminal_reason="finish",
        final_answer="rack test",
    )
    tree = RolloutTreeRecord(
        rollout_id="r",
        group_id="g",
        rollout_index=0,
        seed=0,
        root_node_id="root",
        behavior_policy_version="policy_000000",
        nodes={"root": node},
        tree_status=TreeStatus.COMPLETE,
    )
    result = await ExactMatchVerifier().evaluate(node, tree, task)
    assert result.value == 1


@pytest.mark.asyncio
async def test_exact_verifier_rejects_fallback_even_if_answer_matches():
    task = RootTaskRecord(
        task_id="t",
        domain="test",
        split="train",
        prompt="q",
        ground_truth="correct",
    )
    node = TrainingNodeRecord(
        task_id="t",
        group_id="g",
        rollout_id="r",
        node_id="root",
        parent_id=None,
        depth=0,
        node_task="q",
        terminal_status="completed",
        terminal_reason="step_limit",
        final_answer="correct",
        is_fallback=True,
    )
    tree = RolloutTreeRecord(
        rollout_id="r",
        group_id="g",
        rollout_index=0,
        seed=0,
        root_node_id="root",
        behavior_policy_version="policy_000000",
        nodes={"root": node},
        tree_status=TreeStatus.COMPLETE,
    )
    result = await ExactMatchVerifier().evaluate(node, tree, task)
    assert result.value == 0


def test_root_lexical_guard_rejects_prompt_clue_overlap():
    reference = "Grains of Saws: Associating Quasi-Particles to Surface Acoustic Waves"
    prompt_copy = (
        "The work associates particle-like characteristics or grains with surface "
        "acoustic waves. What is the precise title describing this association?"
    )
    assert root_answer_passes_lexical_guard(reference, prompt_copy) is False
    assert root_answer_passes_lexical_guard(
        reference,
        f"The title is {reference}.",
    ) is True


def test_subtask_guard_rejects_resource_names_and_accepts_substantive_answers():
    assert subtask_answer_passes_minimum_guard("Google Scholar") is False
    assert (
        subtask_answer_passes_minimum_guard(
            "surname of the political figure mentioned in the context",
            "Determine the surname of the political figure mentioned in the context",
        )
        is False
    )
    assert (
        subtask_answer_passes_minimum_guard(
            "No initial academic journals were identified within the allowed depth "
            "limit due to the search results. Further research may be required.",
            "Identify initial relevant academic journals.",
        )
        is False
    )
    assert (
        subtask_answer_passes_minimum_guard(
            "the current leader is identified and the task is complete",
            "narrow delegated task",
        )
        is False
    )
    assert (
        subtask_answer_passes_minimum_guard(
            "None of the search results provided a clear matching title. "
            "The maximum depth was reached before a specific match was identified.",
            "Examine search results and identify a matching paper title.",
        )
        is False
    )
    assert (
        subtask_answer_passes_minimum_guard(
            "Synthesis\n\nTask: find relevant scientific publications related "
            "to neurological conditions in Nouvelle-Aquitaine\n\nFinal answer "
            "from child results:\nNo child returned a successful answer.\n\n"
            "Limitations:\n- narrow delegated task: Cannot launch child because "
            "max_depth has been reached.",
            "Find relevant scientific publications related to neurological "
            "conditions in Nouvelle-Aquitaine.",
        )
        is False
    )
    assert subtask_answer_passes_minimum_guard(
        "The paper title is Grains of SAWs: Associating quasi-particles to surface acoustic waves.",
        "Find the exact title from the publication clues.",
        "Grains of Saws: Associating Quasi-Particles to Surface Acoustic Waves",
    ) is True
    assert (
        subtask_answer_passes_minimum_guard(
            "A New Method of Surface Acoustic Wave Pressure Sensing Based on "
            "Acoustic Attenuation Mechanism",
            "Examine search results to identify a potential matching title.",
            "Grains of Saws: Associating Quasi-Particles to Surface Acoustic Waves",
        )
        is False
    )
    assert (
        subtask_answer_passes_minimum_guard(
            "International Journal of Engineering Science",
            "Identify the journal in which the target paper appeared.",
            "Grains of Saws: Associating Quasi-Particles to Surface Acoustic Waves",
        )
        is False
    )
    assert (
        subtask_answer_passes_minimum_guard(
            "International Journal of Engineering Science",
            "Identify the journal in which the target paper appeared.",
        )
        is True
    )
    assert (
        subtask_answer_passes_minimum_guard(
            "Cannot launch child because max_depth has been reached.",
            "Alternative pedagogical models collective.",
        )
        is False
    )


@pytest.mark.asyncio
async def test_lazy_root_proxy_propagates_root_signal():
    task = RootTaskRecord(
        task_id="t",
        domain="test",
        split="train",
        prompt="q",
        ground_truth="correct",
    )
    root = TrainingNodeRecord(
        task_id="t",
        group_id="g",
        rollout_id="r",
        node_id="root",
        parent_id=None,
        depth=0,
        node_task="q",
        children_ids=["child"],
        terminal_status="completed",
        terminal_reason="finish",
        final_answer="correct",
    )
    child = TrainingNodeRecord(
        task_id="t",
        group_id="g",
        rollout_id="r",
        node_id="child",
        parent_id="root",
        depth=1,
        node_task="subtask",
        terminal_status="completed",
        terminal_reason="finish",
        final_answer="wrong",
    )
    tree = RolloutTreeRecord(
        rollout_id="r",
        group_id="g",
        rollout_index=0,
        seed=0,
        root_node_id="root",
        behavior_policy_version="policy_000000",
        nodes={"root": root, "child": child},
        tree_status=TreeStatus.COMPLETE,
    )
    verifier = LazyRootProxyVerifier(ExactMatchVerifier())
    root_result = await verifier.evaluate(root, tree, task)
    child_result = await verifier.evaluate(child, tree, task)
    assert root_result.value == 1
    assert child_result.value == 1
    assert child_result.is_proxy is True


@pytest.mark.asyncio
async def test_lazy_root_proxy_does_not_reward_fallback_nodes():
    task = RootTaskRecord(
        task_id="t",
        domain="test",
        split="train",
        prompt="q",
        ground_truth="correct",
    )
    root = TrainingNodeRecord(
        task_id="t",
        group_id="g",
        rollout_id="r",
        node_id="root",
        parent_id=None,
        depth=0,
        node_task="q",
        children_ids=["child"],
        terminal_status="completed",
        terminal_reason="finish",
        final_answer="correct",
    )
    child = TrainingNodeRecord(
        task_id="t",
        group_id="g",
        rollout_id="r",
        node_id="child",
        parent_id="root",
        depth=1,
        node_task="subtask",
        terminal_status="step_limited",
        terminal_reason="step_limit",
        final_answer="correct",
        is_fallback=True,
    )
    tree = RolloutTreeRecord(
        rollout_id="r",
        group_id="g",
        rollout_index=0,
        seed=0,
        root_node_id="root",
        behavior_policy_version="policy_000000",
        nodes={"root": root, "child": child},
        tree_status=TreeStatus.COMPLETE,
    )
    verifier = LazyRootProxyVerifier(ExactMatchVerifier())

    root_result = await verifier.evaluate(root, tree, task)
    child_result = await verifier.evaluate(child, tree, task)

    assert root_result.value == 1
    assert child_result.value == 0
    assert child_result.is_proxy is True
    assert "fallback" in child_result.reason
