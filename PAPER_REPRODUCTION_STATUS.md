# RAO Paper Reproduction Status

Last audited: 2026-06-10

Target: [Recursive Agent Optimization, arXiv:2605.06639v1](https://arxiv.org/abs/2605.06639), submitted 2026-05-07.

## Verdict

**Current claim level: engineering flywheel validated, paper result reproduction blocked.**

The repository can run a real multi-round synthetic rollout/training/checkpoint loop. It
cannot yet run the paper's DeepDive experiment end to end, and it cannot produce the
paper's four-way comparison or 75-step training evidence.

Use these labels consistently:

| Level | Meaning | Current |
|---|---|---|
| L0 | Unit-tested RAO equations and schemas | READY |
| L1 | Synthetic model-in-the-loop data flywheel | READY |
| L2 | Real DeepDive short training run | BLOCKED |
| L3 | Faithful DeepDive 75-step reproduction | BLOCKED |
| L4 | Paper effect reproduced with baselines and uncertainty | BLOCKED |

Run the local audit:

```bash
python3 check_reproduction_readiness.py
python3 check_reproduction_readiness.py --json
```

## Gate Dashboard

| Gate | State | Current evidence | What is missing |
|---|---|---|---|
| RAO reward, LOO advantage, depth weighting, CISPO | READY | `recursive_agent_training/` and unit tests | Large-run numerical audit |
| Atomic synchronous flywheel and resume | READY | `flywheel.py`, Qwen 0.6B synthetic runs | DeepDive task source is not wired |
| DeepDive dataset and deterministic split code | PARTIAL | Dataset and loader exist | Checked-in/generated split manifest required |
| Paper action space | BLOCKED | CodeAct parser/sandbox are isolated prototypes | Python REPL policy loop with async `launch_subagent` |
| DeepDive training entry point | BLOCKED | Entry point ends in unconditional `RuntimeError` | Real collect/verify/compile/train loop |
| Search and webpage tools | PARTIAL | Basic adapters exist | Provider contract, retries, rate limits, caching, provenance |
| LLM node judge | PARTIAL | Basic adapter exists | Stable prompt/version, retries, concurrency, malformed-output handling |
| AReaL distributed backend | BLOCKED | Thin optimizer factory wrapper only | Workers, transport, versioning, checkpoint load/publish, distributed step |
| Faithful 40K train / 256K eval context | BLOCKED | Values exist only in YAML | Truncation policy and long-context runtime verification |
| Held-out evaluation | BLOCKED | Aggregator only | Rollout runner, pass@8, confidence intervals, per-task records |
| Four principal comparisons | BLOCKED | Some configs exist | Reproducible experiment matrix runner |
| Dense/sparse and weighted/unweighted ablations | BLOCKED | Configs and proxy components exist | Executable orchestration and common evaluation |
| 75-step DeepDive evidence | BLOCKED | No run artifacts | Training curves, checkpoints, cost/failure metrics |

## Largest Fidelity Gaps

1. The paper uses a thought-plus-Python REPL action space. Current runnable training uses
   structured harness actions; `recursive_agent_training/codeact/` is not connected to rollout.
2. `examples/run_deepdive_rao_training.py` is a preflight command, not a trainer.
3. Current eight-GPU support runs one rollout replica per GPU and trains LoRA on the first
   GPU. It is not the paper's distributed full DeepDive optimization path.
4. `evaluation.py` summarizes already-created records but does not generate held-out
   rollouts or calculate pass@8 and confidence intervals.
5. Search and judge calls have no production controls required for a long experiment.

## Paper-Effect Acceptance Bar

Do not label a run as a paper reproduction until all of the following are true:

- Qwen-3-4B-Instruct-2507 initialization and tokenizer revision are recorded.
- DeepDive dataset hash and a fixed 50-task held-out split are recorded.
- Training uses root batch size 16, group size 8, depth 4, 25 steps per node, learning
  rate `3e-6`, 40K training context, and staleness no greater than 3 batches.
- At least 75 optimizer steps complete with resumable checkpoints.
- Every training sample can be traced to behavior-policy tokens, logprobs, node reward,
  root-group baseline, advantage, and depth weight.
- Evaluation runs base/single, trained/single, base/recursive, and trained/recursive on the
  same held-out tasks with fixed judge/tool versions.
- The report contains success rate, pass@8, confidence intervals, per-task outputs,
  tool/judge failure rates, compute, latency, and implementation differences.
- The recursive trained condition improves over the trained single-agent condition with
  uncertainty reported. Matching an isolated headline number is not sufficient.

See `PAPER_REPRODUCTION_BACKLOG.md` for the ordered implementation work and
`reproduction_gates.json` for the machine-readable gate list.
