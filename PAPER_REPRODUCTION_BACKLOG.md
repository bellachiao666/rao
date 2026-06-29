# RAO Paper Reproduction Backlog

This is the shortest implementation path from the current synthetic flywheel to a
defensible DeepDive result reproduction.

## P0: Required Before Any Paper Claim

### 1. Make DeepDive Training Executable

Implement:

- A task-source abstraction used by the flywheel, with synthetic and DeepDive providers.
- Loading train IDs from the split manifest rather than cycling over the full CSV.
- Construction of DeepDive tools and root/subtask judges from a redacted runtime config.
- A real `collect -> verify -> compile -> optimize -> publish` path in
  `examples/run_deepdive_rao_training.py`.
- A one-task and one-faithful-batch integration test.

Acceptance:

```text
1 task, G=2, depth=1, 1 optimizer step
16 tasks, G=8, depth=4, 1 optimizer step
all 128 trees present and auditable
```

### 2. Connect the Paper's CodeAct Runtime

Implement:

- A model policy that emits thought plus Python and records exact sampled tokens/logprobs.
- A persistent per-node REPL state.
- Awaitable `launch_subagent`, `search_web`, and webpage-read functions.
- Sequential and `asyncio.gather` child execution with structured return values.
- Time, depth, process, network, output-size, and allowed-call limits.
- Training trace conversion without losing the exact action-token span.

The existing parser and AST validator are only prototypes. The DeepDive faithful config
must use `action_profile: codeact`; otherwise the run must be labeled an implementation
variant.

### 3. Harden Search and Judge Services

Implement:

- Tavily-compatible search request/response normalization.
- Retry with bounded exponential backoff and explicit terminal error states.
- Per-provider concurrency and QPS limits.
- URL/content caching and content-size policy.
- Judge prompt and model version pinning.
- Strict judge schema parsing, retry, and missing-signal quarantine.
- Cost, latency, cache hit, timeout, and failure metrics.
- Credential separation between policy, judge, and search.

No judge failure may silently become reward zero.

### 4. Implement the Real Distributed Backend

Expand `AReaLOptimizerAdapter` or provide an equivalent backend that supports:

- Distributed model initialization for Qwen-3-4B-Instruct-2507.
- Rollout worker lifecycle and token-batch transport.
- Behavior-policy version publication and staleness enforcement.
- Current-policy logprob computation over captured action tokens.
- Distributed optimizer step, gradient clipping, and finite-value checks.
- Atomic checkpoint save, load, resume, and snapshot publication.
- Worker failure recovery without accepting partial groups.

The current eight-GPU Hugging Face runtime is useful for rollout throughput testing, but
GPU 0 performs the LoRA update. It is not this gate.

### 5. Enforce Long-Context Semantics

Implement and test:

- 40,960-token training and 262,144-token evaluation contexts.
- A documented prompt/observation truncation policy.
- Preservation of system prompt, root task, and recent observations.
- Rejection of samples whose generated action-token span would be truncated.
- Post-truncation behavior/current logprob alignment checks.
- Memory and token-budget preflight estimates.

### 6. Build the Evaluation Runner

Add a real held-out evaluator, not only an aggregator:

- Load exactly the split manifest's 50 evaluation tasks.
- Run a fixed number of rollouts per task and calculate success and pass@8.
- Record nodes, depth, steps, tokens, delegation, latency, concurrency, tool errors,
  judge errors, and cost.
- Calculate bootstrap confidence intervals with a fixed seed.
- Write `evaluation_manifest.json`, `per_task_results.jsonl`, and `summary.json`.
- Refuse comparisons when split, judge, tools, model revision, or sampling config differ.

### 7. Orchestrate the Paper Comparison

Implement one matrix command for:

1. Base model + single agent.
2. RL-trained model + single agent.
3. Base model + recursive agent.
4. RAO-trained model + recursive agent.

Then run the four dense/sparse and weighted/unweighted ablations with all other variables
fixed. The command must generate a joined result table and implementation-difference
manifest.

### 8. Produce the 75-Step Evidence Package

Required outputs:

- Run/config/dataset/split/model/tool/judge manifests.
- Checkpoints and resume audit.
- Training success and pass@8 curves.
- Loss, ratio, gradient norm, staleness, token, memory, and throughput curves.
- Search/judge failure and cost summaries.
- Four-way comparison and ablation table.
- Confidence intervals and per-task records.
- A clear list of deviations from the paper.

## P1: Strongly Recommended

- Run at least three training seeds or report that compute permits only one.
- Repeat judge scoring on a sample to measure judge variance.
- Add deterministic replay of captured batches for objective debugging.
- Add canary tasks to detect search-provider or judge drift.
- Validate checkpoint evaluation in a fresh process and environment.
- Add interruption tests for rollout worker, judge, search, and optimizer failures.

## Recommended Build Order

1. DeepDive synchronous LoRA, one task.
2. DeepDive synchronous LoRA, faithful 128-tree batch.
3. CodeAct runtime and exact token accounting.
4. Held-out evaluator and four-way matrix.
5. Distributed AReaL backend and async staleness.
6. Five-step resume run.
7. Ten-step short training with periodic evaluation.
8. Full 75-step run and ablations.

At steps 1-2, describe results as **DeepDive pipeline validation**, not paper reproduction.
