# RAO Training 实施计划

## 0. 文档定位

本文档由 `training-design.md` 派生，用于指导当前仓库逐步实现 Recursive Agent Optimization（RAO）的训练部分。

实施目标：

- 复用现有 `recursive_agent_harness` 作为递归 rollout environment；
- 将 execution tree 中每个 node 转换为独立训练单元；
- 实现 Eq.1 local reward；
- 实现 Eq.3 root-group leave-one-out advantage；
- 实现 Eq.4 depth inverse-frequency weighting；
- 实现 Eq.5 对所有深度 node trajectory 的共享 policy 更新；
- 实现 CISPO-style token-level objective；
- 支持 grouped recursive rollout、异步训练和最多 3 batch staleness；
- 优先完成 DeepDive Existing Harness profile；
- 后续再增加 Faithful CodeAct profile。

本计划不假设一次完成完整分布式训练。所有 Phase 必须按依赖顺序推进，每一阶段都要形成可测试、可回放、可审计的中间结果。

## 0.1 核心实施原则

1. Tree 是 rollout 结构，node 是训练单元。
2. Root 和所有 sub-agent 使用同一 policy snapshot。
3. Success、reward、baseline、advantage、depth weight 必须分开存储和计算。
4. Ground truth 只进入 verifier，不进入 agent prompt。
5. `completed` 只表示执行结束，不表示任务成功。
6. 有效失败 trajectory 应参与 RL；synthetic fallback 不参与 faithful training。
7. 缺少 behavior token/logprob 时禁止进入 policy optimization。
8. Rollout group 不完整时禁止计算 Eq.3。
9. Depth weight 必须在最终 trainable node 集合上计算。
10. 论文未披露参数必须显式配置并记录为 implementation choice。

## 0.2 主实现路径

主路径采用 Existing Harness profile：

- 保留当前 strict JSON `AgentAction`；
- 复用 `RecursiveAgent`、`Runner` 和 `ExecutionTree`；
- 增加训练专用 trace，不破坏现有调试 trace；
- 先用 fake/tiny policy 完成端到端算法验证；
- 再接 Hugging Face/local model；
- 最后接 AReaL 和真实 DeepDive。

Faithful CodeAct profile 不与主路径同时开发，安排在 Phase 15。

## 0.3 Phase 总览

| Phase | 名称 | 主要产物 | 进入条件 |
|---:|---|---|---|
| 0 | 基线冻结与实现决策 | 基线报告、配置决策表 | 当前测试全绿 |
| 1 | Training package 与配置骨架 | 包结构、训练配置 schema | Phase 0 完成 |
| 2 | 版本化训练数据模型 | task/group/tree/node/batch schema | Phase 1 完成 |
| 3 | Dataset 与固定 split | DeepDive loader、split manifest | Phase 2 完成 |
| 4 | Credit assignment 数学核心 | reward、LOO、depth weight | Phase 2 完成 |
| 5 | Trainability 与 trace 校验 | terminal reason、资格过滤、validator | Phase 2/4 完成 |
| 6 | Policy snapshot 与 token rollout | token ids、mask、behavior logprobs | Phase 5 完成 |
| 7 | Grouped recursive collector | 1 task × G trees | Phase 6 完成 |
| 8 | Verifier 与 reward pipeline | root/subtask signal、Eq.1 reward | Phase 7 完成 |
| 9 | Advantage 与 batch compiler | Eq.3、Eq.4、node samples | Phase 8 完成 |
| 10 | CISPO optimizer | ratio、clip、loss、checkpoint | Phase 9 完成 |
| 11 | 同步端到端 mini training | rollout 到参数更新闭环 | Phase 10 完成 |
| 12 | 异步 coordinator 与 staleness | queue、policy publication、resume | Phase 11 完成 |
| 13 | DeepDive tools 与真实 smoke test | search/web tools、真实 judge | Phase 12 完成 |
| 14 | AReaL/Qwen DeepDive 训练 | faithful DeepDive run | Phase 13 完成 |
| 15 | 评估、基线与消融 | 四基线、四消融、指标报告 | Phase 14 完成 |
| 16 | Faithful CodeAct profile | Python REPL action space | 主路径稳定 |
| 17 | TextCraft/Oolong 扩展 | domain adapters | Phase 15 后 |

# 1. 目标目录结构

后续实现建议增加独立训练包，避免把训练逻辑堆入 inference 模块：

```text
recursive_agent_training/
  __init__.py
  config.py
  schemas.py
  datasets.py
  split.py
  validation.py
  rewards.py
  advantages.py
  weighting.py
  snapshots.py
  rollout.py
  grouping.py
  verifiers/
    __init__.py
    base.py
    exact.py
    llm.py
    proxy.py
  compilation.py
  objectives/
    __init__.py
    cispo.py
  optimizers/
    __init__.py
    base.py
    torch_optimizer.py
    areal_optimizer.py
  coordinator.py
  checkpoints.py
  metrics.py
  evaluation.py
  domains/
    __init__.py
    deepdive.py
    textcraft.py
    oolong.py

examples/
  prepare_deepdive_split.py
  collect_training_rollouts.py
  build_training_batch.py
  run_mini_rao_training.py
  run_deepdive_rao_training.py
  evaluate_rao_checkpoint.py

tests/
  training/
    test_config.py
    test_schemas.py
    test_datasets.py
    test_validation.py
    test_rewards.py
    test_advantages.py
    test_weighting.py
    test_snapshots.py
    test_rollout_capture.py
    test_grouping.py
    test_verifiers.py
    test_compilation.py
    test_cispo.py
    test_checkpoints.py
    test_coordinator.py
    test_mini_training.py
    test_deepdive_domain.py

configs/
  train/
    mini_rao.yaml
    deepdive_existing_harness.yaml
    deepdive_areal.yaml
    deepdive_single_agent.yaml
    ablations/
      dense_weighted.yaml
      dense_unweighted.yaml
      sparse_weighted.yaml
      sparse_unweighted.yaml
```

可能需要小范围修改的现有文件：

```text
recursive_agent_harness/policy.py
recursive_agent_harness/agent.py
recursive_agent_harness/runner.py
recursive_agent_harness/state.py
recursive_agent_harness/tree.py
recursive_agent_harness/tools.py
recursive_agent_harness/evaluator.py
pyproject.toml
requirements.txt
tests/requirements.txt
README.md
```

原则：

- 新训练行为优先通过 adapter/hook 注入；
- 现有 inference API 保持向后兼容；
- 不把 PyTorch、Transformers、AReaL 强制加入最小 inference 安装；
- 训练依赖使用 optional dependency group 或独立 requirements。

# 2. Phase 0：冻结基线与实现决策

## 2.1 目标

在改代码前记录当前仓库状态，明确论文事实与本地选择，防止后续无法判断回归来源。

## 2.2 工作项

### 2.2.1 运行并记录现有测试

执行：

```bash
python3 -m pytest -q
```

记录：

- Python 版本；
- 测试数量；
- 当前 commit/hash；
- dirty worktree；
- 现有 inference trace schema version；
- 当前 DeepDive CSV 行数和文件 hash。

### 2.2.2 建立实现决策表

需要在训练配置或开发记录中明确：

- 主 profile：Existing Harness；
- 模型 backend 的第一阶段选择；
- tokenizer/chat template 来源；
- CISPO `epsilon_low`；
- CISPO `epsilon_high`；
- optimizer betas；
- weight decay；
- grad clip；
- scheduler/warmup；
- judge model；
- judge retry；
- DeepDive held-out split seed；
- checkpoint frequency；
- failed trajectory 是否训练；
- timeout/fallback 的 success 规则。

论文未披露项必须标为：

```text
source: implementation_choice
```

### 2.2.3 固定术语

全项目统一：

- `task`：root dataset item；
- `group`：同一 task 的 G 个 rollouts；
- `rollout`：一棵 recursive tree；
- `node`：一个 agent trajectory；
- `behavior policy`：生成 sample 的 snapshot；
- `current policy`：optimizer 正在更新的 policy；
- `success_signal`：verifier 输出；
- `reward`：Eq.1；
- `baseline`：Eq.3 的 \(b_{-g}\)；
- `advantage`：node reward - root-group baseline；
- `depth_weight`：Eq.4。

## 2.3 文件范围

Phase 0 不要求修改运行代码。可新增：

- `docs/train_baseline.md`，若后续需要保存基线；
- 或将决策写入 `configs/train/*.yaml` 的 metadata。

## 2.4 验收

- 现有测试全部通过；
- 没有修改 inference 行为；
- 所有未披露超参数都有待决项；
- 主 profile 已明确；
- 后续 Phase 不再混用 `run_id`、`rollout_id`、`group_id`。

## 2.5 Exit Gate

只有当前测试全绿且实现决策表已建立，才进入 Phase 1。

# 3. Phase 1：Training package 与配置骨架

## 3.1 目标

建立独立训练包和强校验配置，暂不实现训练算法。

## 3.2 新增文件

```text
recursive_agent_training/__init__.py
recursive_agent_training/config.py
configs/train/mini_rao.yaml
configs/train/deepdive_existing_harness.yaml
tests/training/test_config.py
```

按需要修改：

```text
pyproject.toml
requirements.txt
tests/requirements.txt
```

## 3.3 配置 schema

定义以下配置组：

- `ExperimentConfig`；
- `DatasetConfig`；
- `ModelConfig`；
- `RolloutConfig`；
- `RewardConfig`；
- `AdvantageConfig`；
- `WeightingConfig`；
- `ObjectiveConfig`；
- `OptimizerConfig`；
- `CheckpointConfig`；
- `EvaluationConfig`；
- `AsyncConfig`。

## 3.4 强校验规则

- `group_size >= 2`；
- faithful DeepDive `group_size == 8`；
- `root_batch_size > 0`；
- `max_depth >= 0`；
- `max_steps_per_node > 0`；
- `lambda >= 0`；
- `max_staleness_batches >= 0`；
- CISPO bounds 不允许静默为默认值；
- faithful DeepDive learning rate 默认为论文值 `3e-6`；
- faithful DeepDive context 为 40K train / 256K eval；
- faithful DeepDive max depth 为 4；
- faithful DeepDive每 node 25 steps；
- DeepDive `lambda=0`；
- objective 必须标明 `cispo`；
- 配置导出时包含 `source`：paper 或 implementation_choice。

## 3.5 依赖策略

建议：

```toml
[project.optional-dependencies]
train = [
  "torch",
  "transformers",
  "datasets",
  "accelerate",
]
```

AReaL/Tinker 不放入默认依赖，使用独立 backend extra。

## 3.6 测试

- 最小配置可加载；
- DeepDive 配置可加载；
- `group_size=1` 失败；
- 缺少 CISPO clip 配置时给明确错误或 unresolved 状态；
- negative lambda 失败；
- 配置 round trip；
- config hash 对相同内容稳定；
- secret 字段不进入 config dump。

## 3.7 验收

```bash
python3 -m pytest tests/training/test_config.py -q
python3 -m pytest -q
```

## 3.8 Exit Gate

- 配置 schema 完整；
- inference 最小依赖不被训练依赖污染；
- 所有未披露参数可追踪。

# 4. Phase 2：版本化训练数据模型

## 4.1 目标

实现可序列化、可校验、可迁移的训练数据契约。

## 4.2 新增文件

```text
recursive_agent_training/schemas.py
recursive_agent_training/validation.py
tests/training/test_schemas.py
tests/training/test_validation.py
```

## 4.3 必须实现的 schema

### RootTaskRecord

字段：

- `schema_version`；
- `task_id`；
- `domain`；
- `split`；
- `prompt`；
- `ground_truth`；
- `metadata`；
- `dataset_hash`。

### RolloutGroupRecord

字段：

- `group_id`；
- `task_id`；
- `expected_group_size`；
- `completed_group_size`；
- `behavior_policy_version`；
- `sampling_config_hash`；
- `rollout_ids`；
- `status`；
- timestamps。

### RolloutTreeRecord

字段：

- `rollout_id`；
- `group_id`；
- `rollout_index`；
- `seed`；
- `root_node_id`；
- `behavior_policy_version`；
- prompt/tool/environment versions；
- nodes；
- `tree_status`；
- checksum。

### TrainingNodeRecord

字段：

- task/group/rollout/node/parent ids；
- depth；
- node task；
- children ids；
- terminal status/reason；
- fallback/trainability；
- model messages；
- tokenizer/template version；
- token ids；
- action mask；
- behavior logprobs；
- truncation；
- verifier result；
- reward；
- baseline；
- advantage；
- depth count/alpha/weight；
- staleness。

### OptimizerBatchManifest

字段：

- optimizer step；
- policy before/after；
- root task count；
- group size；
- tree count；
- node count；
- depth counts/weights；
- objective；
- config hash；
- source rollout ids。

## 4.4 ID 规则

ID 不依赖内存地址：

```text
task_id       = domain + dataset_version + source_id
group_id      = task_id + policy_version + group_sequence
rollout_id    = group_id + rollout_index
node_id       = rollout-local stable id
```

保留当前 `node_0001` 风格，但 node 唯一性由 `(rollout_id, node_id)` 保证。

## 4.5 版本与迁移

- training schema 从 `1.0.0` 开始；
- inference trace schema 保持独立；
- 增加明确 converter，而非让训练代码直接读取任意 JSON；
- unknown future field 可选择 ignore 或 forbid，行为由 config 决定；
- schema migration 必须是显式函数。

## 4.6 测试

- 每种 schema round trip；
- invalid parent/depth；
- duplicate node；
- duplicate rollout index；
- checksum 不匹配；
- token/logprob/mask 长度不一致；
- unknown schema version；
- secret redaction；
- JSONL 写入与读取；
- 大字段不被意外转为字符串。

## 4.7 验收

- 可用纯 Python 构造完整 1 task × 2 rollout fixture；
- 任意 node 可追溯到 task/group/tree；
- 不依赖真实模型；
- 不修改当前 inference trace。

## 4.8 Exit Gate

训练数据契约冻结后才进入数学核心和 rollout capture。

# 5. Phase 3：DeepDive Dataset 与固定 split

## 5.1 目标

将现有 CSV 转换为稳定的 root task records，并在训练前固定 split。

## 5.2 新增文件

```text
recursive_agent_training/datasets.py
recursive_agent_training/split.py
recursive_agent_training/domains/deepdive.py
examples/prepare_deepdive_split.py
tests/training/test_datasets.py
tests/training/test_deepdive_domain.py
```

## 5.3 输入文件

```text
data/deepdive_qa_rl.csv
data/deepdive_qa_sft.csv
data/deepdive_trajectories_sft.csv
```

## 5.4 实现步骤

1. 使用 `csv.DictReader` 读取，不手工 split CSV；
2. 校验列：`id,question,answer,conversations`；
3. 规范化空值；
4. 检测重复 id；
5. 检测重复 question；
6. 计算源文件 SHA-256；
7. 生成稳定 task id；
8. 固定 held-out 50 tasks；
9. 保存 split manifest；
10. 训练时只暴露 question；
11. verifier 单独读取 answer；
12. 记录 split seed 和 dataset hash。

## 5.5 Split manifest

建议保存：

```json
{
  "schema_version": "1.0",
  "dataset_hash": "...",
  "seed": 42,
  "train_task_ids": [],
  "eval_task_ids": [],
  "eval_count": 50
}
```

若数据规模不足 50，命令必须失败，不得静默缩小 held-out。

## 5.6 Ground-truth 防泄露测试

必须检查：

- root agent messages 中不存在 `answer`；
- child context 中不存在 `answer`；
- tool schema 中不存在 `answer`；
- trace agent-visible fields 中不存在 ground truth；
- verifier input 中允许包含 ground truth。

## 5.7 验收命令

```bash
python3 examples/prepare_deepdive_split.py \
  --input data/deepdive_qa_rl.csv \
  --output output/train/deepdive_split.json \
  --eval-count 50 \
  --seed 42

python3 -m pytest tests/training/test_datasets.py tests/training/test_deepdive_domain.py -q
```

## 5.8 Exit Gate

- split manifest 可重复生成；
- ground truth 无泄露；
- task id 稳定；
- 数据 hash 被记录。

# 6. Phase 4：Credit Assignment 数学核心

## 6.1 目标

用纯函数实现 Eq.1、Eq.3、Eq.4，不依赖模型或网络。

## 6.2 新增文件

```text
recursive_agent_training/rewards.py
recursive_agent_training/advantages.py
recursive_agent_training/weighting.py
tests/training/test_rewards.py
tests/training/test_advantages.py
tests/training/test_weighting.py
```

## 6.3 Eq.1 RewardComputer

接口建议：

```python
compute_node_reward(
    own_success: float,
    child_successes: list[float],
    delegation_lambda: float,
) -> RewardBreakdown
```

返回：

- own success；
- child count；
- child success mean；
- lambda；
- delegation bonus；
- reward；
- formula version。

规则：

- success 在 `[0,1]`；
- leaf bonus 为 0；
- 只接收直接 children；
- 空 children 不除零；
- 非有限数报错；
- child limit 未创建 node 不计 child。

## 6.4 Eq.3 AdvantageComputer

接口建议：

```python
compute_group_advantages(
    rollout_root_rewards,
    node_rewards_by_rollout,
) -> GroupAdvantageResult
```

规则：

- `G >= 2`；
- group task id 相同；
- rollout index 唯一；
- root reward 完整；
- baseline 排除本 rollout；
- 同一 tree 所有 node 使用同一 baseline；
- child advantage 使用 child reward。

固定测试：

```text
root rewards = [1.0, 0.0, 0.5, 1.0]
```

## 6.5 Eq.4 DepthWeightComputer

接口建议：

```python
compute_depth_weights(trainable_nodes) -> DepthWeightResult
```

固定测试：

```text
N0=2, N1=4, N2=8
Ntotal=14
K=3
alpha=14/3
```

验证：

- 每层总 weight 为 alpha；
- 总 weight 为 14；
- 空 depth 不参与；
- 过滤后再统计；
- deterministic ordering。

## 6.6 数值类型

- reward/advantage/weight 使用 Python float 或 torch float32；
- schema 保存普通 JSON number；
- 计算时拒绝 NaN/Inf；
- tolerance 统一在测试 helper 中定义。

## 6.7 验收

```bash
python3 -m pytest \
  tests/training/test_rewards.py \
  tests/training/test_advantages.py \
  tests/training/test_weighting.py -q
```

## 6.8 Exit Gate

- 公式单测 100% 通过；
- 模块无网络、无模型依赖；
- 任意结果可从 breakdown 重算。

# 7. Phase 5：Terminal Reason、Trainability 与 Trace 校验

## 7.1 目标

把执行结束、任务成功和训练资格分离，消除当前 best-effort placeholder 被标记 completed 的风险。

## 7.2 新增或修改文件

新增：

```text
recursive_agent_training/validation.py
tests/training/test_validation.py
```

可能修改：

```text
recursive_agent_harness/state.py
recursive_agent_harness/agent.py
recursive_agent_harness/tree.py
tests/test_invalid_action.py
tests/test_runner.py
```

## 7.3 新增语义

定义：

- `terminal_reason`；
- `is_fallback`；
- `is_trainable`；
- `trainability_reasons`；
- `verifier_status`；
- `trace_complete`；
- `token_data_complete`。

推荐 terminal reasons：

- `finish`；
- `direct_answer_finish`；
- `step_limit`；
- `depth_limit`；
- `child_limit`；
- `node_limit`；
- `model_timeout`；
- `tool_timeout`；
- `invalid_action_limit`；
- `fallback_finish`；
- `cancelled`；
- `exception`。

## 7.4 向后兼容

- 不删除现有 `NodeStatus`；
- inference 用户仍可读取 `completed`；
- training converter 依据 terminal reason 判断；
- 旧 trace 无 terminal reason 时标为 legacy/unverified；
- 不通过字符串匹配 answer 来判断 fallback，优先使用显式字段。

## 7.5 TrainabilityValidator

校验：

- tree complete；
- group complete；
- generated token 非空；
- behavior logprobs 完整；
- action mask 完整；
- tokenizer/template 匹配；
- reward/advantage/weight 完整；
- staleness 合法；
- 非 synthetic fallback；
- truncation 策略允许；
- verifier signal 完整。

返回结构化原因，不只返回布尔值。

## 7.6 测试

- 正常 finish 可训练；
- zero reward 的真实失败可训练；
- fallback 不可训练；
- judge missing 不可训练；
- timeout 默认不可训练或按配置处理；
- token 缺失不可训练；
- completed 不等于 success；
- legacy trace 不会误入训练。

## 7.7 验收

```bash
python3 -m pytest tests/training/test_validation.py -q
python3 -m pytest -q
```

## 7.8 Exit Gate

- best-effort placeholder 不会进入 faithful optimizer；
- 现有 inference 测试无回归。

# 8. Phase 6：Policy Snapshot 与 Token-level Rollout Capture

## 8.1 目标

捕获 RL 所需的精确 model input、generated tokens、action mask 和 behavior logprobs。

## 8.2 新增文件

```text
recursive_agent_training/snapshots.py
recursive_agent_training/rollout.py
tests/training/test_snapshots.py
tests/training/test_rollout_capture.py
```

可能修改：

```text
recursive_agent_harness/policy.py
recursive_agent_harness/agent.py
recursive_agent_harness/runner.py
recursive_agent_harness/state.py
```

## 8.3 Policy 接口扩展

不能只返回 `AgentAction`。训练路径需要：

```python
PolicyGeneration:
  action
  messages
  rendered_prompt
  input_ids
  generated_ids
  generated_text
  action_mask
  behavior_logprobs
  sampling_metadata
  policy_version
  tokenizer_revision
  template_version
```

向后兼容策略：

- 保留 `Policy.act(state) -> AgentAction`；
- 新增可选 `TrainingPolicy.act_with_trace(state)`；
- 或让 `act()` 返回兼容 wrapper，但不得破坏现有 fake policies；
- inference policy 不支持 token trace 时明确 `supports_training_trace=False`。

## 8.4 SnapshotProvider

能力：

- 发布 immutable version；
- 为每个 tree 提供 snapshot handle；
- tree 内所有 node 共享 version；
- 记录 model/tokenizer/template hash；
- 可从 checkpoint 恢复 version sequence。

## 8.5 Action mask

Existing Harness profile：

- system/user/tool observation token mask 为 0；
- assistant JSON action generated token mask 为 1；
- padding mask 为 0；
- repair call 是独立 policy generation；
- parse-failed raw generation是否训练需明确配置；
- fallback action由 harness合成，mask 全 0，且不可训练。

## 8.6 Behavior logprob

- 必须对应 generated ids；
- 必须来自实际采样 policy；
- 不允许事后用不同 checkpoint 伪造 behavior logprob；
- temperature/top-p 等采样配置写入 metadata；
- 若 backend 返回 logits，使用采样分布的 logprob；
- tool/child observation不计算 behavior logprob。

## 8.7 Fake TrainingPolicy

先实现 deterministic fake policy：

- 固定 vocabulary；
- 固定 generated ids；
- 固定 logprobs；
- 支持 root/child 不同 scripted action；
- 支持 policy version；
- 用于后续所有数学和 coordinator 测试。

## 8.8 测试

- tree 内 policy version 一致；
- token/mask/logprob 对齐；
- repair generation 可追踪；
- fallback 无 policy tokens；
- prompt 不含 ground truth；
- child 使用 fresh context；
- parent只获得 child summary；
- tokenizer mismatch 被拒绝；
- snapshot immutable。

## 8.9 验收

- 可用 fake policy 生成一个完整 training tree record；
- 所有生成 token 可追溯到 node step；
- 现有 inference policy 仍可运行。

## 8.10 Exit Gate

没有完整 token/logprob capture，不进入 grouped rollout 和 optimizer。

# 9. Phase 7：Grouped Recursive Rollout Collector

## 9.1 目标

对同一 root task 采样 `G` 棵独立 recursive trees，并形成完整 group。

## 9.2 新增文件

```text
recursive_agent_training/grouping.py
recursive_agent_training/rollout.py
examples/collect_training_rollouts.py
tests/training/test_grouping.py
```

可能修改：

```text
recursive_agent_harness/runner.py
```

## 9.3 Collector 接口

```python
collect_group(
    task: RootTaskRecord,
    snapshot: PolicySnapshot,
    group_size: int,
) -> RolloutGroupBundle
```

## 9.4 采样规则

- faithful `G=8`；
- rollout index `0..G-1`；
- 每棵 tree seed 不同；
- root prompt 相同；
- sampling config 相同；
- behavior policy version 相同；
- tree state 不共享；
- 允许只读 tool cache；
- tree 内 child 可并行；
- group 完成前不计算 advantage。

## 9.5 并发层次

支持：

1. root task 之间并发；
2. 同 task 的 G 棵 tree 并发；
3. tree 内独立 children 并发。

必须有总并发限制，避免乘法爆炸。

## 9.6 Group 状态

- `pending`；
- `collecting`；
- `complete`；
- `incomplete`；
- `rejected`；
- `expired`。

任一 rollout 失败时：

- 保存 partial trace；
- faithful group 默认整体 incomplete；
- 可按配置重采样缺失 rollout index；
- 不允许用另一 task 的 rollout 补位。

## 9.7 测试

- 1 task × 2 rollouts；
- 1 task × 8 rollouts；
- seeds 唯一；
- mixed policy version 拒绝；
- duplicate index 拒绝；
- missing rollout incomplete；
- retry 后 group complete；
- tree checksum；
- 并发不改变 ID 稳定性。

## 9.8 验收命令

```bash
python3 examples/collect_training_rollouts.py \
  --config configs/train/mini_rao.yaml \
  --limit 1 \
  --output output/train/mini_rollouts
```

输出至少包含：

```text
group manifest
G tree records
node records
checksums
```

## 9.9 Exit Gate

- 一个完整 group 可离线加载和校验；
- group 内所有 tree snapshot 一致。

# 10. Phase 8：Verifier 与 Reward Pipeline

## 10.1 目标

为每个 node 生成带 provenance 的 success signal，并计算 Eq.1 reward。

## 10.2 新增文件

```text
recursive_agent_training/verifiers/base.py
recursive_agent_training/verifiers/exact.py
recursive_agent_training/verifiers/llm.py
recursive_agent_training/verifiers/proxy.py
recursive_agent_training/rewards.py
tests/training/test_verifiers.py
```

可能复用/修改：

```text
recursive_agent_harness/evaluator.py
recursive_agent_harness/prompts.py
```

## 10.3 Verifier 接口

```python
evaluate(
    node,
    tree,
    root_task,
) -> SuccessSignalResult
```

结果包含：

- `value`；
- `provider`；
- `provider_version`；
- `reason`；
- `raw_output_ref`；
- `is_proxy`；
- `status`；
- `attempt_count`；
- `error`。

## 10.4 DeepDive RootVerifier

输入：

- task；
- ground truth；
- root final answer；
- 可选 tree summary。

输出 strict JSON：

```json
{"reason": "...", "success": true}
```

可选前置 matcher：

- normalized exact match；
- alias matcher；
- LLM judge fallback。

## 10.5 DeepDive SubtaskVerifier

输入：

- parent/root task；
- subtask；
- node answer；
- child task list；
- delegation trace summary；
- terminal reason。

重点识别：

- whole-goal forwarding；
- 重复/重叠 child；
- 越 depth 委派；
- 无效 child launch；
- answer 与 subtask 不相关；
- 有用且高效的 decomposition。

## 10.6 Judge 稳定性

- temperature 0；
- strict JSON；
- parse repair 最多一次；
- timeout；
- retry；
- provider/model version；
- raw response 安全留存；
- judge failure 不等于 success 0，先标 missing；
- faithful batch 必须等待完整 signals。

## 10.7 Reward 顺序

1. 验证所有 node；
2. 检查 signal 完整；
3. 对每个 node 读取 direct children signal；
4. 计算 Eq.1；
5. 写 RewardBreakdown；
6. 保存公式版本。

DeepDive faithful：

```text
lambda = 0
```

仍需计算 child success mean，但 delegation bonus 为 0，便于审计。

## 10.8 测试

- exact root success；
- root failure；
- subtask useful delegation；
- degenerate delegation；
- judge malformed JSON；
- judge timeout；
- retry success；
- missing signal 阻止 reward；
- lambda 0；
- lambda 0.4；
- grandchildren 不进入 parent reward。

## 10.9 验收

- fake judge 可为完整 group 写入 success/reward；
- reward 可从 signal 重算；
- ground truth 仅在 verifier path 出现。

## 10.10 Exit Gate

所有 trainable node 均有合法 success signal 和 RewardBreakdown。

# 11. Phase 9：Advantage、Depth Weight 与 Batch Compiler

## 11.1 目标

将完整 rollout groups 编译为 optimizer-ready node samples。

## 11.2 新增文件

```text
recursive_agent_training/compilation.py
examples/build_training_batch.py
tests/training/test_compilation.py
```

## 11.3 编译流程

1. 加载完整 groups；
2. 校验 task/group/tree/node；
3. 过滤非 trainable node；
4. 检查每个 group 仍有 G 个完整 rollouts；
5. 计算 root LOO baseline；
6. 写入每个 node advantage；
7. 汇总当前 optimizer batch 的所有 trainable nodes；
8. 按 depth 统计；
9. 计算 depth weights；
10. 构造 token samples；
11. 生成 OptimizerBatchManifest；
12. 写 checksum。

## 11.4 重要顺序

禁止：

```text
先统计 depth -> 后过滤 invalid node
```

必须：

```text
validate -> filter -> group completeness -> advantage -> depth counts -> weight
```

## 11.5 Token sample

至少包含：

- `input_ids` 或完整 sequence ids；
- `attention_mask`；
- `action_mask`；
- `behavior_logprobs`；
- `advantage`；
- `depth_weight`；
- `rollout_id`；
- `node_id`；
- `policy_version`；
- `staleness`。

## 11.6 Packing

第一版不做跨 sample packing，优先保证：

- mask 正确；
- logprob 对齐；
- sample 可审计。

后续性能优化才引入 packing，并增加 segment mask 测试。

## 11.7 测试

- child 使用 tree baseline；
- incomplete group 拒绝；
- G=1 拒绝；
- mixed tasks 拒绝；
- zero-reward failure 可保留；
- fallback 过滤；
- 过滤后 depth counts 正确；
- batch manifest 可重算；
- 顺序变化不影响数值。

## 11.8 验收命令

```bash
python3 examples/build_training_batch.py \
  --config configs/train/mini_rao.yaml \
  --rollouts output/train/mini_rollouts \
  --output output/train/mini_batch
```

## 11.9 Exit Gate

可离线生成一个无模型依赖、数值完全可审计的 optimizer batch。

# 12. Phase 10：CISPO Objective 与 Policy Optimizer

## 12.1 目标

实现 token-level CISPO loss，并完成一次真实参数更新。

## 12.2 新增文件

```text
recursive_agent_training/objectives/cispo.py
recursive_agent_training/optimizers/base.py
recursive_agent_training/optimizers/torch_optimizer.py
recursive_agent_training/checkpoints.py
tests/training/test_cispo.py
tests/training/test_checkpoints.py
```

## 12.3 CISPO 实现

输入：

- current logprobs；
- behavior logprobs；
- action mask；
- node advantage；
- depth weight；
- epsilon low/high。

步骤：

```text
log_ratio = current_logprob - behavior_logprob
ratio = exp(log_ratio)
clipped_ratio = clip(ratio, lower, upper)
detached_weight = stop_gradient(clipped_ratio)
token_objective = detached_weight * advantage * depth_weight * current_logprob
loss = -masked_mean(token_objective)
```

## 12.4 Mask 规则

- prompt 0；
- system 0；
- user 0；
- observation 0；
- policy-generated assistant/action 1；
- padding 0；
- harness-generated fallback 0；
- invalid truncated region 0。

## 12.5 配置

论文未公开 clip bounds，因此：

- config 必填；
- manifest 记录；
- checkpoint 记录；
- test 覆盖 asymmetric bounds；
- faithful result 报告中标明 implementation choice。

## 12.6 Optimizer

第一版 Torch optimizer：

- tiny causal LM；
- AdamW；
- configurable betas/weight decay；
- gradient accumulation；
- grad clipping；
- scheduler optional；
- mixed precision optional；
- 单机单卡优先。

## 12.7 Checkpoint

保存：

- model；
- optimizer；
- scheduler；
- global step；
- policy version；
- RNG；
- config；
- batch/group ids；
- schema versions；
- metrics。

使用临时文件后原子 rename。

## 12.8 测试

### CISPO 数值测试

- ratio 1；
- upper clip；
- lower clip；
- lower clip disabled；
- positive advantage；
- negative advantage；
- zero advantage；
- depth weight scaling；
- mask；
- stop-gradient；
- NaN/Inf 拒绝。

### Optimizer 测试

- one step 参数变化；
- all-zero mask 不更新；
- zero advantage 不更新；
- checkpoint round trip；
- resume 后 step/version 连续；
- duplicate batch id 拒绝。

## 12.9 验收

```bash
python3 -m pytest \
  tests/training/test_cispo.py \
  tests/training/test_checkpoints.py -q
```

## 12.10 Exit Gate

- 使用固定 batch 可确定性更新 tiny model；
- loss 中只包含 policy action tokens；
- checkpoint 可恢复。

# 13. Phase 11：同步端到端 Mini RAO Training

## 13.1 目标

在 deterministic fake environment 上打通：

```text
dataset
-> snapshot
-> grouped recursive rollout
-> verifier
-> reward
-> advantage
-> depth weight
-> CISPO
-> optimizer step
-> checkpoint
-> new policy version
```

## 13.2 新增文件

```text
examples/run_mini_rao_training.py
tests/training/test_mini_training.py
configs/train/mini_rao.yaml
```

## 13.3 Mini environment

要求：

- task 可确定性验证；
- policy 可选择 direct answer 或 delegation；
- child success 可控；
- 至少包含 depth 0、1、2；
- 同一 root task 多 rollouts 有不同 reward；
- 可在少量 step 后观察 policy 参数变化。

## 13.4 同步 trainer

第一版流程保持同步：

1. sample root batch；
2. collect all groups；
3. verify all groups；
4. compile batch；
5. optimize；
6. checkpoint；
7. evaluate。

此阶段不引入 queue/staleness。

## 13.5 必测场景

- 2 root tasks × 2 rollouts；
- 多深度；
- 一个真实失败 trajectory；
- 一个 fallback trajectory；
- fallback 被过滤但 group 处理符合规则；
- reward/advantage/weight 写入；
- optimizer step；
- resume；
- 重复 group 不被再次消费。

## 13.6 验收命令

```bash
python3 examples/run_mini_rao_training.py \
  --config configs/train/mini_rao.yaml \
  --steps 2
```

验收输出：

- 两个 checkpoints；
- 两个 batch manifests；
- rollout groups；
- metrics JSONL；
- policy version 单调递增；
- 参数 checksum 变化。

## 13.7 Exit Gate

没有同步闭环，不进入异步系统和真实 DeepDive。

# 14. Phase 12：异步 Coordinator 与 Staleness

## 14.1 目标

解耦 rollout、verification、batch compilation 和 optimization，支持论文的异步 RL 语义。

## 14.2 新增文件

```text
recursive_agent_training/coordinator.py
recursive_agent_training/metrics.py
tests/training/test_coordinator.py
```

## 14.3 Pipeline queues

建议逻辑队列：

```text
task_queue
rollout_queue
verification_queue
train_queue
checkpoint_publication
```

第一版可使用 `asyncio.Queue`，不直接引入分布式队列。

## 14.4 Policy publication

- optimizer step 成功后发布新 version；
- worker 只获取完整 checkpoint；
- in-flight tree 可使用旧 snapshot 完成；
- coordinator 计算 sample staleness；
- policy version 不回退。

## 14.5 Staleness

定义：

```text
staleness = current_policy_version - behavior_policy_version
```

faithful limit：

```text
max_staleness_batches = 3
```

规则：

- `<=3` 可训练；
- `>3` 整个 group 拒绝；
- group 不允许部分保留；
- rejected group 保留诊断 metadata；
- ratio clipping 不替代 staleness gate。

## 14.6 Backpressure

配置：

- max in-flight root tasks；
- max in-flight groups；
- max rollout concurrency；
- max child concurrency；
- max verifier concurrency；
- train queue size。

## 14.7 Shutdown 与恢复

- 停止接收新 task；
- 等待或取消 in-flight tree；
- partial group 标记 incomplete；
- 保存 coordinator state；
- resume 后不重复训练 consumed group；
- unfinished group 可重采样。

## 14.8 测试

- staleness 0/3 接收；
- staleness 4 拒绝；
- mixed staleness group 整体拒绝；
- queue backpressure；
- graceful shutdown；
- checkpoint publication；
- resume；
- duplicate group；
- worker 使用旧 snapshot 但 tree 内一致。

## 14.9 Exit Gate

- fake backend 异步跑多个 optimizer steps；
- 无重复消费；
- staleness gate 可观测。

# 15. Phase 13：DeepDive Tools、Judge 与真实 Smoke Test

## 15.1 目标

补齐 DeepDive 所需环境能力，在不做大规模训练前验证真实 recursive rollout 和训练数据生成。

## 15.2 新增或修改文件

新增：

```text
recursive_agent_training/domains/deepdive.py
recursive_agent_training/verifiers/llm.py
examples/run_deepdive_rao_training.py
tests/training/test_deepdive_domain.py
```

修改：

```text
recursive_agent_harness/tools.py
recursive_agent_harness/prompts.py
recursive_agent_harness/evaluator.py
```

## 15.3 DeepDive tools

实现：

- `search_web(query, max_results)`；
- `view_webpage_content(url)`；
- URL validation；
- timeout；
- retry；
- rate-limit handling；
- cache；
- result truncation；
- provenance；
- untrusted content 标记；
- fake deterministic versions。

论文对应 Tavily search 和 webpage content。若使用其他 provider，必须在 run manifest 标注差异。

## 15.4 Prompt

Existing Harness profile prompt 应：

- 允许 direct solve；
- 允许单 child；
- 允许 parallel children；
- 强调 narrower/concrete/non-overlapping；
- 防止 whole-goal forwarding；
- 要求 evidence；
- 明确 depth 和 step budget；
- 不强迫每个 task 委派。

Prompt version 必须进入 trace。

## 15.5 Judge

配置独立于 policy：

- judge base URL；
- judge API key；
- judge model；
- temperature 0；
- timeout；
- retry；
- concurrency；
- cache；
- prompt version。

不得复用训练 policy 参数作为 judge 更新对象。

## 15.6 Smoke test 范围

建议：

```text
2 root tasks
2 rollouts/task
max depth 2
较低 step limit
不做或只做 tiny adapter update
```

Smoke test 目标是数据完整性，不是论文指标。

## 15.7 验收

- 真实 web tool 可调用；
- root/child trace 有 evidence；
- judge signals 完整；
- reward/advantage/weight 完整；
- token/logprob 完整；
- ground truth 无泄露；
- API key 被 redaction；
- timeout/fallback 不会伪装成功。

## 15.8 Exit Gate

真实 DeepDive group 可编译为 optimizer batch，且审计通过。

# 16. Phase 14：AReaL/Qwen DeepDive Faithful Training

## 16.1 目标

将已验证的算法接入论文 DeepDive 训练配置。

## 16.2 论文配置

```text
model: Qwen-3-4B-Instruct-2507
backend: AReaL
learning_rate: 3e-6
root_batch_size: 16
group_size: 8
train_context: 40K
eval_context: 256K
max_depth: 4
max_steps_per_node: 25
lambda: 0
objective: CISPO-style
max_staleness_batches: 3
```

## 16.3 新增文件

```text
recursive_agent_training/optimizers/areal_optimizer.py
configs/train/deepdive_areal.yaml
tests/training/test_areal_adapter.py
```

## 16.4 Backend adapter

训练核心不得直接依赖 AReaL 私有对象。统一接口：

```python
class PolicyOptimizer:
    current_version()
    compute_logprobs(batch)
    step(batch)
    save_checkpoint()
    load_checkpoint()
    publish_snapshot()
```

AReaL adapter 负责：

- distributed model；
- rollout worker integration；
- optimizer；
- async version；
- checkpoint；
- metrics；
- token batch transport。

## 16.5 逐级放大

### Stage 14A：单 task

- 1 task；
- G=2；
- max depth 1；
- 1 optimizer step。

### Stage 14B：小 group

- 2 tasks；
- G=4；
- max depth 2；
- 2 steps。

### Stage 14C：faithful group

- 16 tasks；
- G=8；
- max depth 4；
- 25 steps/node；
- 1 optimizer step。

### Stage 14D：短训练

- 5-10 optimizer steps；
- checkpoint/resume；
- held-out periodic evaluation。

### Stage 14E：论文对齐训练

- 目标至少 75 training steps；
- 保存所有 faithful config；
- 记录 compute、失败率和 judge cost。

## 16.6 上下文与截断

- train max context 40K；
- 超长 node sample 必须有明确策略；
- 不允许静默截掉 action tokens；
- prompt truncation 需保留 system/task 与最近 observation；
- truncation 后行为 logprob 必须仍与实际采样序列一致；
- 超出策略的 sample 标为不可训练。

## 16.7 资源预检

训练前输出：

- GPU 数量/型号；
- 显存；
- estimated trees/batch；
- estimated nodes/tree；
- expected tokens；
- judge QPS；
- web tool QPS；
- disk requirement；
- checkpoint size。

资源不足时先降并发，不改变 group size 和算法语义。

## 16.8 验收

- faithful batch 真正包含 16 × 8 trees；
- 所有 node 同一共享 Qwen policy；
- Eq.1/3/4/5 指标可重算；
- AReaL step 成功；
- staleness 不超过 3；
- checkpoint 可恢复；
- 训练日志记录所有实现差异。

# 17. Phase 15：评估、基线与消融

## 17.1 目标

复现论文比较结构，而不仅是得到一个训练 checkpoint。

## 17.2 四个主要基线

1. Base model + single agent；
2. RL-trained model + single agent；
3. Base model + recursive agent；
4. RAO-trained model + recursive agent。

所有评估使用同一 held-out 50 tasks。

## 17.3 评估配置

- held-out split 固定；
- eval context 256K；
- recursive max depth 4；
- sampling config 固定；
- judge version 固定；
- 每 task rollout 数固定；
- 报告随机 seed；
- 记录失败和 timeout。

## 17.4 指标

- success rate；
- pass@8；
- nodes/tree；
- max depth；
- steps；
- tokens；
- wall-clock time；
- concurrency；
- delegation ratio；
- judge cost；
- tool failure；
- fallback ratio。

## 17.5 四组消融

```text
Dense + Weighted
Dense + Unweighted
Sparse + Weighted
Sparse + Unweighted
```

### Dense

每个 node 使用 task-specific success。

### Sparse

按论文消融定义，只用 root reward 作为 proxy 传播给 subagents。

### Weighted

启用 Eq.4。

### Unweighted

所有 node weight 为 1。

## 17.6 控制变量

消融之间保持：

- model init；
- dataset split；
- batch size；
- group size；
- rollout limits；
- optimizer；
- clip bounds；
- tools；
- judge；
- seeds；
- evaluation protocol。

## 17.7 论文参考值

DeepDive held-out：

```text
single-agent success: 0.24
recursive-agent success: 0.40
```

这些值是参考，不是代码单测。结果报告必须说明：

- 官方代码未公开；
- action space 差异；
- tool/provider 差异；
- judge 差异；
- compute 差异；
- 未披露超参数的选择。

## 17.8 验收产物

```text
evaluation manifest
per-task results
aggregate metrics
confidence intervals
rollout behavior analysis
ablation table
implementation differences
```

# 18. Phase 16：Faithful CodeAct Profile

## 18.1 目标

在 Existing Harness profile 稳定后，增加更接近论文的 Python REPL action space。

## 18.2 新增模块

建议独立模块，不替换现有 JSON action：

```text
recursive_agent_training/codeact/
  parser.py
  environment.py
  sandbox.py
  policy.py
  prompts.py
```

## 18.3 行为

- 模型输出 reasoning + Python code；
- environment 提供 async `launch_subagent`；
- 支持 `asyncio.gather`；
- tool 调用通过 Python；
- child 可返回结构化对象；
- sandbox 限制文件、网络、进程；
- code stdout/stderr/exception 进入 observation；
- 每个 node 有独立 REPL state。

## 18.4 安全要求

- 禁止任意 host 文件访问；
- 禁止任意 subprocess；
- 网络只通过注册工具；
- CPU/memory/time 限制；
- 输出大小限制；
- secret 不注入 interpreter；
- sandbox failure 可追踪。

## 18.5 训练兼容

- assistant code tokens mask 为 1；
- environment output mask 为 0；
- code parse/execute failure仍可成为有效负奖励 trajectory；
- harness-generated recovery 不进入 policy loss；
- Eq.1/3/4/5 与 Existing Harness 共用。

## 18.6 验收

- 同一训练核心可切换 profile；
- action space 差异只在 rollout adapter；
- CodeAct 结果单独报告；
- 不破坏 Existing Harness。

# 19. Phase 17：TextCraft-Synth 与 Oolong-Real 扩展

## 19.1 TextCraft-Synth

新增：

- deterministic environment adapter；
- exact verifier；
- difficulty split；
- context-limited/unlimited profile；
- train depth 6；
- eval depth 12；
- 25 steps/node；
- lambda 0。

优先用于：

- exact reward 验证；
- 深度泛化；
- dense/sparse 消融；
- context scaling。

## 19.2 Oolong-Real

新增：

- long context dataset adapter；
- Python interpreter context storage；
- child context slicing；
- benchmark partial-credit root verifier；
- LLM subtask judge；
- max depth 2；
- 15 steps/node；
- lambda 0.4；
- Tinker LoRA rank 32 adapter；
- learning rate `3e-5`。

## 19.3 边界

- domain-specific context logic 不进入通用 RewardComputer；
- domain-specific verifier 通过统一接口；
- optimizer backend 与 domain adapter 解耦；
- 不为 Oolong 修改 DeepDive schema 语义。

# 20. 横向工作流

以下工作不是单独 Phase，但从 Phase 2 起持续执行。

## 20.1 Metrics

记录四类：

### Rollout

- trees/task；
- nodes/tree；
- depth；
- steps；
- tool calls；
- child count；
- parallel ratio；
- timeout/fallback；
- tokens；
- latency。

### Reward

- success by depth；
- reward by depth；
- child success；
- delegation bonus；
- judge error。

### Optimization

- baseline；
- advantage；
- depth counts/weights；
- loss；
- IS ratio quantiles；
- clipped fraction；
- gradient norm；
- entropy；
- stale groups。

### System

- queue size；
- rollout throughput；
- verifier throughput；
- GPU utilization；
- disk；
- checkpoint duration。

## 20.2 安全

- API key redaction；
- raw webpage 标记不可信；
- ground truth 隔离；
- judge 与 agent credentials 分离；
- checkpoint 无 secrets；
- trace 发布前扫描；
- 日志不打印完整 key/长网页。

## 20.3 可复现性

每次 run 保存：

- config；
- git revision；
- dependency versions；
- dataset hash；
- split manifest；
- prompt/tool/verifier versions；
- model/tokenizer revisions；
- random seeds；
- hardware；
- checkpoint hashes。

## 20.4 文档

实现完成后再更新：

- README；
- training quickstart；
- schema reference；
- troubleshooting；
- reproduction report。

README 更新不应早于同步 mini training 成功。

# 21. CI 与测试分层

## 21.1 Tier 1：快速纯单元测试

每次提交运行：

```bash
python3 -m pytest \
  tests/training/test_config.py \
  tests/training/test_schemas.py \
  tests/training/test_rewards.py \
  tests/training/test_advantages.py \
  tests/training/test_weighting.py \
  tests/training/test_cispo.py -q
```

要求：

- 无网络；
- 无 GPU；
- 秒级完成。

## 21.2 Tier 2：集成测试

```bash
python3 -m pytest tests/training -q
```

要求：

- fake policy；
- fake tools；
- tiny model；
- CPU 可运行；
- 完成 rollout-to-update。

## 21.3 Tier 3：真实服务 smoke

手工或受控 CI：

- web provider；
- judge provider；
- 1-2 DeepDive tasks；
- 不要求 GPU 大训练；
- credentials 通过 secret store。

## 21.4 Tier 4：GPU/backend

- Qwen；
- AReaL；
- multi-worker；
- checkpoint/resume；
- faithful batch。

不作为普通 PR 的必跑测试。

# 22. Phase 依赖与禁止并行项

可并行：

- Phase 3 Dataset 与 Phase 4 数学核心；
- Phase 8 verifier fake implementation 与 Phase 7 collector 后半；
- metrics 与各 Phase；
- DeepDive fake tools 与 coordinator 测试。

禁止并行导致接口漂移：

- Phase 2 schema 未冻结前，不实现 optimizer；
- Phase 5 trainability 未完成前，不编译 batch；
- Phase 6 token capture 未完成前，不做真实 RL；
- Phase 9 compiler 未完成前，不接 AReaL；
- Phase 11 同步闭环未完成前，不做异步；
- Phase 13 smoke 未通过前，不启动 faithful DeepDive；
- Existing Harness 未稳定前，不启动 CodeAct 主线。

# 23. 最终 Definition of Done

完整 RAO training 复现需满足：

## 23.1 算法

- Eq.1、Eq.3、Eq.4、Eq.5 独立实现和测试；
- CISPO token objective 数值正确；
- root/child 共享 policy；
- child 使用 own reward 和 tree root baseline；
- depth weighting 守恒；
- 真实失败 trajectory 可训练；
- fallback 不伪装成功。

## 23.2 数据

- task/group/tree/node/batch 全链路可追踪；
- token/mask/logprob 完整；
- reward provenance 完整；
- advantage/weight 可重算；
- schema version 和 checksum 完整；
- ground truth 无泄露。

## 23.3 系统

- grouped recursive rollout；
- checkpoint/resume；
- async queues；
- max staleness 3；
- backpressure；
- DeepDive web tools；
- root/subtask judge；
- AReaL backend。

## 23.4 实验

- 四个主要基线；
- 四组消融；
- held-out 50 tasks；
- success、pass@8、depth、tokens、latency；
- 75-step 对齐训练或明确说明停止原因；
- 报告实现差异和 compute 差异。

## 23.5 工程质量

- 原 inference tests 全绿；
- training unit/integration tests 全绿；
- 无 secret 泄露；
- 配置无隐式论文参数；
- 所有失败可诊断；
- README 和复现说明完整。

# 24. 推荐执行顺序

实际开发按以下顺序提交，每次提交保持可运行：

1. Phase 0：基线与决策。
2. Phase 1：训练配置。
3. Phase 2：数据 schema。
4. Phase 3：DeepDive dataset/split。
5. Phase 4：reward/advantage/weight。
6. Phase 5：terminal reason/trainability。
7. Phase 6：token/logprob capture。
8. Phase 7：grouped collector。
9. Phase 8：verifier/reward pipeline。
10. Phase 9：batch compiler。
11. Phase 10：CISPO/optimizer/checkpoint。
12. Phase 11：同步 mini training。
13. Phase 12：异步 coordinator/staleness。
14. Phase 13：DeepDive tools/smoke。
15. Phase 14：AReaL faithful training。
16. Phase 15：evaluation/ablation。
17. Phase 16：CodeAct。
18. Phase 17：TextCraft/Oolong。

任何 Phase 未通过 Exit Gate，都不应通过临时 hard-code 绕过进入下一阶段。
