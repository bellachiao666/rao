# Recursive Agent Harness

This project implements a RAO-style recursive agent inference harness. It reproduces the inference-time execution scaffold described in Recursive Agent Optimization: one root agent can dynamically launch sub-agents, sub-agents use the same policy wrapper, and the rollout is recorded as a recursive execution tree.

The core `recursive_agent_harness` package implements inference. The optional
`recursive_agent_training` package adds RAO rollout grouping, credit assignment,
CISPO optimization, and parameter updates. Reward-model training is not included.

## Repository Layout

```text
configs/                    Training profiles
docs/                       Architecture history and reference material
examples/                   Runnable inference and training entry points
recursive_agent_harness/    Recursive inference runtime
recursive_agent_training/   Rollout, credit assignment, and optimization
tests/                      Inference and training tests
```

Runtime traces, checkpoints, caches, local credentials, and downloaded datasets
are intentionally excluded from version control.

## RAO Training

The repository now includes an opt-in `recursive_agent_training` package that
implements the training data contracts and RAO optimization path without adding
heavy ML dependencies to the inference-only install.

Implemented training components:

- grouped recursive rollouts with a shared policy snapshot;
- versioned task/group/tree/node/batch schemas;
- Equation 1 local node reward;
- Equation 3 root-group leave-one-out advantages;
- Equation 4 depth inverse-frequency weighting;
- CISPO-style token objective;
- PyTorch/LoRA optimizer and atomic checkpoints;
- asynchronous staleness gate;
- DeepDive dataset, verifier, and web-tool adapters;
- deterministic end-to-end mini training.

Install optional dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-train.txt
```

Run the offline recursive training smoke test:

```bash
python3 examples/run_mini_rao_training.py \
  --config configs/train/mini_rao.yaml \
  --steps 2
```

Run a local Hugging Face checkpoint where the model itself samples every
recursive action, captures exact behavior logprobs, and performs one LoRA CISPO update:

```bash
python3 examples/run_hf_smoke_training.py \
  --model /path/to/Qwen3-0.6B \
  --output output/train/qwen_smoke
```

Run a synchronous round-based flywheel. A round freezes all rollout data before
training, publishes the LoRA adapter and optimizer state atomically, and requires
the next round to reload that publication:

```bash
python3 examples/run_sync_flywheel.py \
  --model /path/to/Qwen3-0.6B \
  --output output/train/sync_qwen \
  --rounds 1

python3 examples/run_sync_flywheel.py \
  --model /path/to/Qwen3-0.6B \
  --output output/train/sync_qwen \
  --rounds 2 \
  --resume
```

Published rounds are recorded under `rounds/round_NNNN`, while `latest.json`
is updated only after the adapter, optimizer state, checksums, and round
manifest have all been written successfully.

For eight-GPU rollout collection, use one model replica per GPU. Rollouts run
concurrently across the listed devices; after the round is frozen, the first
device performs the LoRA update:

```bash
python3 examples/run_sync_flywheel.py \
  --config configs/train/sync_qwen_0_6b_8gpu.yaml \
  --model /path/to/Qwen3-0.6B \
  --output output/train/sync_qwen_8gpu \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
  --rounds 1
```

The paper-faithful DeepDive/AReaL experiment additionally requires the original
distributed backend and larger hardware. The hardware-adapted path uses a pinned
ModelScope Qwen3-0.6B snapshot, DuckDuckGo web search, a separate local Qwen
judge, LoRA, and the fixed 50-task split.

```bash
python3 examples/run_deepdive_rao_training.py \
  --config configs/train/deepdive_qwen06b_s0.yaml \
  --preflight

python3 examples/run_deepdive_rao_training.py \
  --config configs/train/deepdive_qwen06b_s0.yaml \
  --devices cuda:1,cuda:2 \
  --trainer-device cuda:0 \
  --rounds 1
```

The S1 profile uses four root tasks, four rollouts per task, depth two, eight
steps per node, an 8K training context, and a 16K evaluation context:

```bash
python3 examples/run_deepdive_rao_training.py \
  --config configs/train/deepdive_qwen06b_s1.yaml \
  --devices cuda:1,cuda:2,cuda:3,cuda:4 \
  --trainer-device cuda:0 \
  --rounds 10
```

Prepare or execute the four-way comparison and four ablations:

```bash
python3 examples/run_reproduction_matrix.py \
  --base-config configs/train/deepdive_qwen06b_s1.yaml \
  --output /path/to/reproduction-matrix \
  --rounds 75

python3 examples/run_reproduction_matrix.py \
  --base-config configs/train/deepdive_qwen06b_s1.yaml \
  --output /path/to/reproduction-matrix \
  --rounds 75 \
  --execute-training \
  --execute-evaluation
```

All model manifests record the ModelScope model ID, requested revision, local
snapshot hash, tokenizer revision, and per-file hashes. The final report must be
labeled a hardware-adapted method reproduction, not a Qwen 4B paper-number
reproduction.

## What It Implements

- Shared policy across root and sub-agents.
- Dynamic recursive execution tree generated by policy actions at runtime.
- `async launch_subagent(...)` and parallel child execution with `asyncio.gather`.
- Bounded execution with depth, step, child, total node, timeout, tool-call, and parallel-child limits.
- Structured JSON actions instead of free-form control text.
- Full trace logging for nodes, actions, observations, child results, status, timing, and errors.
- JSON trace export, pretty text tree, and Mermaid graph export.
- Training extension placeholders such as node-level `success_signal`, `reward`, rollout metadata, and batch manifests.

## Install

Use Python 3.10+; Python 3.11 or newer is recommended.

```bash
python3 -m pip install -r requirements.txt
```

Install the optional training stack separately:

```bash
python3 -m pip install -r requirements-train.txt
```

The equivalent package-extra installation is `python3 -m pip install ".[train]"`.

Install test dependencies separately when running the suite:

```bash
python3 -m pip install -r tests/requirements.txt
```

The core runtime depends on Pydantic. Tests use pytest and pytest-asyncio.

## Run The API Example

The example uses `LLMPolicy` with the OpenAI-compatible API client. API settings are loaded from `config.yaml`; the runtime does not read OpenAI API settings from environment variables.

Edit `config.yaml`:

```yaml
model_name: "gpt-4.1-mini"
base_url: "https://api.openai.com/v1"
api_key: "YOUR_OPENAI_API_KEY"
```

Then run:

```bash
python3 examples/run_recursive_agent.py
```

You can also pass a different YAML file:

```bash
python3 examples/run_recursive_agent.py --config path/to/config.yaml
```

Without a real `api_key` in YAML, the script exits with a configuration error rather than falling back to a rule-based policy.

```bash
python3 examples/run_recursive_agent.py
```

It runs this task:

```text
Plan a 3-day Kyoto trip in early April for a family. We want cherry blossoms, one quiet temple, one kid-friendly activity, and dinner near Gion. Avoid overly crowded spots.
```

Expected outputs:

- final answer printed to stdout;
- pretty execution tree printed to stdout;
- `trace.json`;
- `trace.mmd`.

## Use LLMPolicy

`LLMPolicy` uses an OpenAI-compatible Chat Completions client. Configure:

- `model_name`;
- `base_url`;
- `api_key`;
- `temperature`;
- `max_tokens`.

The harness remains provider-agnostic: `RecursiveAgent` depends only on the `Policy` interface, not on a specific model provider.

`LLMPolicy` behavior:

- renders `AgentState` into a system prompt and state prompt;
- requests exactly one strict JSON `AgentAction`;
- parses the result with Pydantic;
- attempts one JSON repair call after parse failure;
- falls back to a structured `FINISH` action if repair fails;
- records raw response and parse errors in action metadata for trace logging.

## Trace Files

`trace.json` stores the complete recursive rollout tree:

- `run_id`;
- `root_node_id`;
- redacted config snapshot;
- node map;
- node trajectory;
- final answer;
- error;
- timing metadata;
- future training placeholders.

`trace.mmd` stores a Mermaid graph:

```mermaid
graph TD
  node_0001["node_0001 d=0 completed<br/>Root task"]
  node_0001 --> node_0002
```

## LLM Evaluation

The harness includes an `LLMJudge` for evaluating root tasks and sub-tasks with OpenAI-compatible chat completions. The judge prompts follow the DeepDive root-task and sub-task judge structure:

- root evaluation compares the agent answer with a ground-truth answer;
- sub-task evaluation also checks whether delegation was useful, concrete, non-overlapping, and efficient;
- results are returned as strict JSON with `reason` and `success`;
- `evaluate_tree(...)` can write `llm_judge` and `success_signal` back into node metadata.

Example:

```python
from recursive_agent_harness.evaluator import LLMJudge

judge = LLMJudge(config=config)
results = await judge.evaluate_tree(
    tree,
    ground_truth_by_node={
        "node_0001": "ground truth for root",
        "node_0002": "ground truth for a sub-task",
    },
)
```

## Tests

Run the full suite:

```bash
python3 -m pytest -q
```

The tests do not require network access or a real LLM. They cover:

- action/config/state schemas;
- execution tree export;
- tool registry behavior;
- depth limit;
- total node limit;
- max children and parallel children;
- timeout;
- invalid action fallback;
- shared policy identity;
- LLMPolicy parse, repair, and fallback;
- batch rollout manifest.

## Design Principles

- Harness executes, policy decides.
- Recursion is dynamic, not hard-coded.
- Root and sub-agents share the same policy wrapper.
- Each node is a full agent instance.
- Each edge is a delegated sub-task.
- Recursive execution is bounded.
- Actions are structured JSON.
- Trace everything.
- Failure is data, not crash.
- API-backed policy by default; deterministic policies are limited to tests.
- Training hooks are reserved but not implemented.
