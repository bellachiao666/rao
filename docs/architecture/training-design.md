# RAO Training 复现设计文档

## 文档状态

- 文档用途：作为后续生成 `training-plan.md` 的唯一上游设计依据。
- 当前阶段：只定义训练目标、数据契约、算法语义、系统边界和验收标准，不实现代码。
- 目标论文：Gandhi et al., *Recursive Agent Optimization*, arXiv:2605.06639v1，提交于 2026-05-07。
- 核心复现域：优先复现 DeepDive 上的 RAO 训练，再保留 TextCraft-Synth 与 Oolong-Real 的可插拔扩展能力。
- 仓库基础：复用现有 `recursive_agent_harness` 递归推理执行树，不另写一套不兼容的 rollout runtime。
- 修改边界：本设计不要求在当前阶段修改现有 inference harness。

截至 2026-06-09，论文项目页仍将官方代码标记为 “coming soon”。因此本项目追求：

1. 忠实实现论文已经公开的公式、训练流程和实验参数；
2. 对论文未披露或存在排版缺失的部分提供显式配置，而不是伪造“原文默认值”；
3. 通过可验证的中间产物证明 reward、advantage、depth weighting 和训练样本构造正确；
4. 将训练后端与递归 rollout harness 解耦，允许先用本地小模型验证，再替换为 AReaL、Tinker 或其他分布式 RL 后端。

参考来源：

- 论文：https://arxiv.org/abs/2605.06639
- PDF：https://arxiv.org/pdf/2605.06639
- 项目页：https://apga.github.io/RAO/
- AReaL：https://arxiv.org/abs/2505.24298
- CISPO 来源：https://arxiv.org/abs/2506.13585

# 1. 项目定位

## 1.1 训练目标

本阶段要在现有 recursive agent inference harness 上增加真正的 RAO 训练闭环：

```text
root task batch
  -> 每个 root task 采样 G 棵独立递归 rollout tree
  -> 对每棵树的每个 node 计算 success signal
  -> 根据 node 自身成功和直接 children 成功计算 local reward
  -> 使用同一 root group 的 leave-one-out root baseline 计算每个 node advantage
  -> 按 depth 做 inverse-frequency weighting
  -> 将所有 node trajectory 转换为共享 policy 的训练样本
  -> 使用 CISPO-style policy objective 更新同一个模型
  -> 发布新 policy version 并继续异步采样
```

训练完成后，同一个 policy 应同时学会：

- 直接完成当前节点任务；
- 判断何时值得委派；
- 生成更窄、可执行、可验证的子任务；
- 选择串行或并行委派；
- 在子 agent 返回后继续推理和聚合；
- 随任务难度自适应分配 recursion depth 和 test-time compute。

## 1.2 与普通单 agent RL 的区别

普通 agent RL 通常把一个 root task 对应的一条 flat trajectory 作为一个训练样本。RAO 中：

- 一个 root task 产生一棵动态树；
- 树中每个 node 都有独立 task、prompt、trajectory、answer 和 reward；
- 所有 node 由同一 policy snapshot 生成；
- 所有 node trajectory 都参与同一共享 policy 的更新；
- child task 分布由当前 policy 在线生成，不是固定标注数据集；
- credit assignment 同时依赖 node-local success、直接 children success 和 root-group baseline。

因此，不能把整棵树简单拼成一条超长对话，也不能只训练 root node。

## 1.3 本阶段包含

- recursive rollout group 采样；
- policy snapshot/version 管理；
- node-local verifier 和 success signal；
- Eq.1 local node reward；
- Eq.3 leave-one-out advantage；
- Eq.4 depth-level inverse-frequency weighting；
- Eq.5 对所有深度 node trajectory 的联合优化；
- CISPO-style token-level objective；
- 异步 rollout 与最多 3 batch staleness 的控制能力；
- DeepDive root/subtask judge；
- trace 到训练样本的确定性转换；
- checkpoint、resume、评估和消融实验支持；
- single-agent baseline 与 recursive-agent 对照。

## 1.4 本阶段不等于

- 训练 reward model；
- 训练单独的 planner、worker 或 critic；
- 为不同深度使用不同 policy；
- 用固定模板预先生成递归树；
- 只做 SFT trajectory imitation；
- 把 root reward 无条件复制给所有 child 并称为完整 RAO；
- 精确复刻论文未公开的私有基础设施、私有数据清洗或未披露超参数。

SFT 可以作为可选 warm start，但 RAO 核心必须是在线递归 rollout 上的强化学习。

# 2. 论文复现事实与实现假设

## 2.1 论文明确公开的算法事实

论文明确要求：

- root 和所有 sub-agent 使用同一共享 policy；
- execution tree 由 policy 在运行时动态生成；
- 每个 node 是一条可独立训练的 agent trajectory；
- 每个 root task 采样 `G` 棵独立 rollout tree；
- node reward 使用 node 自身 success signal 和直接 children 的平均 success signal；
- 所有 node 使用同一 root rollout group 的 leave-one-out root reward baseline；
- 不同深度使用 inverse-frequency weight；
- 所有深度的 trajectory 联合更新共享 policy；
- 训练使用 asynchronous RL；
- 论文实验允许最多 3 个 batch 的 policy staleness；
- 论文实验使用 CISPO-style objective；
- 所有实验 batch size 为 16 个 root tasks，group size 为 8 rollouts。

## 2.2 论文公开的后端与学习率

| 域 | 模型 | 后端 | 学习率 | 其他 |
|---|---|---|---:|---|
| TextCraft-Synth | Qwen-3-4B-Instruct-2507 | AReaL | `3e-6` | full RL backend，论文未写 LoRA |
| DeepDive | Qwen-3-4B-Instruct-2507 | AReaL | `3e-6` | 40K train context，256K eval context |
| Oolong-Real | Qwen3-VL-30B-A3B-Instruct | Tinker | `3e-5` | LoRA rank 32 |

统一参数：

- root task batch size：16；
- rollout group size：8；
- objective：CISPO-style；
- asynchronous staleness：最多 3 batches。

## 2.3 论文公开的 rollout 限制

论文附录 A.4 给出：

- TextCraft-Synth single-agent training：每个 rollout 最多 200 steps；
- DeepDive single-agent training：每个 rollout 最多 100 steps；
- single-agent evaluation：最多 20K steps；
- recursive TextCraft-Synth 与 DeepDive：root 和 sub-agent 各最多 25 steps；
- Oolong-Real single-agent：最多 50 steps；
- Oolong-Real recursive：root 和 sub-agent 各最多 15 steps。

论文附录 A.4 存在一句明显残缺文本：

```text
For recursive agents, we allow 25 steps for both the root and sub-agents,
and for DEEPDIVE we .
```

本项目不得猜测残缺部分。已明确的 25 steps 作为论文默认值；任何额外 DeepDive 限制都必须以本地配置形式出现，并标记为“实现选择”。

## 2.4 论文没有完整披露的内容

以下内容不能宣称为论文原始默认值：

- CISPO 的具体 clipping bounds；
- optimizer beta、weight decay、warmup、scheduler；
- gradient accumulation 和 micro batch size；
- AReaL 的具体并行拓扑；
- checkpoint 保存频率；
- DeepDive 训练集精确过滤过程；
- judge 请求失败时的重试策略；
- judge 成本控制策略；
- 随机种子列表；
- 完整训练步数之外的 early stopping 规则；
- 是否在 RAO 前对当前 DeepDive trajectories 进行额外 SFT。

这些值由 `training-plan.md` 后续以配置项和实验记录方式落地。

# 3. 当前仓库基础与差距

## 3.1 可以复用的现有能力

当前仓库已经具备：

- `Runner`：执行 root task；
- `RecursiveAgent`：执行 node-local agent loop；
- `ExecutionTree`：保存 node、edge、trajectory 和 final answer；
- shared `Policy` object：root 与 child 共享 policy wrapper；
- depth、step、child、node、timeout 和 tool-call 限制；
- parallel subagents；
- `LLMJudge`：root/subtask judge 雏形；
- `success_signal`、`reward` metadata 占位；
- batch manifest；
- DeepDive SFT/RL CSV 数据；
- JSON trace 和 Mermaid tree。

这些能力构成 rollout environment，不应在训练实现中重复造轮子。

## 3.2 当前 trace 不能直接用于 RL 更新

现有 trace 主要面向调试，缺少策略梯度训练必需信息：

- root task 的稳定 `task_id`；
- `rollout_group_id` 和 `rollout_index`；
- policy checkpoint/version；
- sampling seed 和 sampling parameters；
- node 对应的完整、精确 model input messages；
- tokenizer name/revision；
- input token ids；
- generated token ids；
- assistant/action token mask；
- behavior policy 下每个 generated token 的 log probability；
- 当前 policy 重算后的 log probability；
- finish/fallback/timeout 是否可训练；
- verifier version；
- reward computation version；
- advantage 和 depth weight；
- prompt/template version；
- tool/environment observation 的稳定序列化形式；
- token usage 和 truncation 信息。

结论：现有 `TrajectoryStep.raw_response` 不能替代 token-level rollout record。训练实现必须增加独立、版本化的 training trace/schema，同时继续兼容当前调试 trace。

## 3.3 当前状态语义需要修正

当前 harness 在连续 invalid action 后可能以 `completed` 状态返回 best-effort placeholder。训练中不能把：

```text
Best effort answer for task: ...
```

视为正常完成，更不能仅根据 `NodeStatus.COMPLETED` 给正奖励。

训练资格必须由独立字段决定：

- `terminal_reason`；
- `is_fallback`；
- `is_trainable`；
- `verifier_status`；
- `success_signal`。

执行状态、答案质量和训练资格必须彼此分离。

# 4. 形式化定义

## 4.1 Root task 与 rollout group

设一个 root task 为：

\[
q = (\text{task\_id}, X_{\text{root}}, y^*, \text{metadata})
\]

其中：

- `task_id`：跨 epoch 稳定；
- \(X_{\text{root}}\)：用户问题或环境目标；
- \(y^*\)：可选 ground truth；
- `metadata`：split、domain、difficulty 等。

对同一 root task，使用同一个 behavior policy version 独立采样：

\[
\mathcal{G}_q = \{T^{(1)}, \ldots, T^{(G)}\}
\]

其中 `G >= 2`。论文实验使用 `G = 8`。

## 4.2 Execution tree

每棵树：

\[
T^{(g)} = (V^{(g)}, E^{(g)})
\]

每个 node \(v \in V^{(g)}\) 包含：

- node task \(X_v\)；
- depth \(d_v\)；
- parent node；
- direct children \(C(v)\)；
- node-local trajectory \(\tau_v\)；
- final answer；
- success signal \(\tilde{s}_v\)；
- local reward \(R_v\)；
- advantage \(A_v\)；
- depth weight \(w_{d_v}\)。

每条 edge 表示 policy 生成的一次 delegated subtask。

## 4.3 Node trajectory

一条 node trajectory 只属于一个 node：

\[
\tau_v = (h_{v,1}, a_{v,1}, o_{v,1}, \ldots, h_{v,T}, a_{v,T})
\]

其中：

- \(h_{v,t}\)：当前 node 的 model-visible history；
- \(a_{v,t}\)：模型生成的 action tokens；
- \(o_{v,t}\)：environment/tool/child observation；
- child 的完整内部 trajectory 不直接拼进 parent；
- parent 只能看到 harness 返回的 child result/summary。

训练时，一个 node trajectory 是一个独立 sequence sample。它继承树级 baseline，但不继承 parent prompt。

# 5. RAO 核心算法

## 5.1 Node success signal

对 node \(v\)：

\[
\tilde{s}_v = \tilde{s}(X_v, \tau_v) \in [0,1]
\]

success signal provider 可以是：

- exact verifier；
- environment state verifier；
- partial-credit scorer；
- LLM judge；
- root success proxy。

每个 signal 必须记录：

- `value`；
- `provider`；
- `provider_version`；
- `reason`；
- `raw_output` 或 raw output 的安全引用；
- `evaluated_at`；
- `error`；
- `is_proxy`。

不得只保存一个浮点数而丢失 provenance。

## 5.2 Eq.1 Local node reward

对 node \(v\)：

\[
R_v =
\tilde{s}_v +
\lambda
\cdot
\begin{cases}
\frac{1}{|C(v)|}\sum_{c \in C(v)} \tilde{s}_c, & |C(v)| > 0 \\
0, & |C(v)| = 0
\end{cases}
\]

实现不变量：

- delegation bonus 只使用直接 children；
- 不递归累计 grandchildren；
- 使用 children success 的平均值，不使用总和；
- failed/timeout child 仍属于已创建 child，默认 signal 为 0，除非 verifier 明确给出其他分值；
- 被 child limit 拒绝且没有实际创建 node 的请求，不进入 \(C(v)\)，但应保留行为轨迹供 policy 学习；
- `lambda >= 0`；
- reward computation 必须纯函数化、可重放。

示例：

```text
parent success = 1
children success = [0, 1]
lambda = 0.4
reward = 1 + 0.4 * mean([0, 1]) = 1.2
```

## 5.3 Eq.3 Leave-one-out root baseline

同一 root task 的 rollout group 中，令第 \(g\) 棵树 root node 的 reward 为：

\[
R_{\text{root}}^{(g)}
\]

第 \(g\) 棵树的 leave-one-out baseline：

\[
b_{-g} =
\frac{1}{G-1}
\sum_{g' \ne g}
R_{\text{root}}^{(g')}
\]

该树中任意 node \(v\) 的 advantage：

\[
A_v^{(g)} = R_v^{(g)} - b_{-g}
\]

关键语义：

- child node 用自己的 local reward 减 root-group baseline；
- baseline 来自其他树的 root reward，不是其他树同深度 reward；
- 同一棵树所有 node 共享同一个 \(b_{-g}\)；
- baseline 不允许读取本树 reward；
- `G < 2` 时无法计算 Eq.3，训练 batch 必须拒绝或明确使用非论文 fallback；
- group 内 rollout 必须针对同一个 root task；
- group 内最好由同一 behavior policy version 和同一 sampling config 采样；
- 缺少 rollout 时不能偷偷按现有数量改变实验语义，应标记 incomplete group。

## 5.4 Eq.4 Depth-level inverse-frequency weighting

在一个 optimizer batch 内，将所有可训练 node trajectory 按 depth 分组：

\[
B_d = \{\tau_v \mid d_v=d\}, \quad N_d = |B_d|
\]

对非空 depth：

\[
w_d = \alpha \cdot \frac{1}{N_d}
\]

\[
\alpha =
\frac{\sum_d N_d}
{\sum_d N_d \cdot \frac{1}{N_d}}
\]

若共有 \(K\) 个非空 depth，则：

\[
\alpha = \frac{N_{\text{total}}}{K}
\]

性质：

- 每个非空 depth 的总权重均为 \(\alpha\)；
- 所有 trajectory weight 总和仍为 \(N_{\text{total}}\)；
- 深层 node 数量多时，每个深层 trajectory 权重更小；
- 统计必须基于最终进入 loss 的 trainable trajectories；
- 空 depth 不进入分母；
- weight 应记录在每个 node sample 中，便于审计。

## 5.5 Eq.5 Joint optimization

论文给出的估计量：

\[
\widehat{\nabla}J(\theta)
=
\sum_d
\sum_{\tau_v \in B_d}
w_d A_v
\nabla_\theta \log \pi_\theta(\tau_v)
\]

工程语义：

- 所有 node 使用同一可训练参数集合；
- root、internal node、leaf 都可进入更新；
- 不对 subagent 使用冻结 worker model；
- node sample 的 sequence advantage 在其所有可训练生成 token 上共享；
- prompt token、system token、user token、tool observation token 不直接产生 policy loss；
- 只有由 policy 生成的 assistant/action token 进入 action mask；
- 不训练外部 judge；
- 不对 environment observation 反向传播。

# 6. CISPO-style objective

## 6.1 使用原因

RAO 论文明确写明使用 CISPO-style objective，但没有披露 clipping bounds。CISPO 的核心是：

- 使用 behavior policy 收集 trajectory；
- 用 importance sampling 修正异步或多次更新造成的分布偏移；
- clip importance sampling weight；
- 不像 PPO/GRPO 那样因为超出 trust region 而丢弃 token gradient；
- clipped weight 使用 stop-gradient；
- 原始 CISPO 不包含 KL penalty。

## 6.2 Token-level 定义

对 trainable token \(t\)：

\[
r_{v,t}(\theta)
=
\frac{\pi_\theta(a_{v,t}\mid h_{v,t})}
{\mu(a_{v,t}\mid h_{v,t})}
=
\exp(\log\pi_\theta - \log\mu)
\]

其中 \(\mu\) 为采样时 behavior policy。

clipped IS weight：

\[
\hat{r}_{v,t}
=
\operatorname{clip}
\left(
r_{v,t},
1-\epsilon_{\text{low}},
1+\epsilon_{\text{high}}
\right)
\]

结合 RAO node advantage 和 depth weight，本项目目标形式：

\[
J_{\text{RAO-CISPO}}(\theta)
=
\frac{1}{\sum_{v,t}m_{v,t}}
\sum_v
\sum_t
m_{v,t}
\cdot w_{d_v}
\cdot A_v
\cdot \operatorname{sg}(\hat{r}_{v,t})
\cdot \log\pi_\theta(a_{v,t}\mid h_{v,t})
\]

其中：

- \(m_{v,t}\) 是 policy-generated token mask；
- `sg` 表示 stop-gradient；
- loss 实现使用负号做梯度下降；
- padding、prompt、observation、被截断无效 token 的 mask 为 0。

## 6.3 未披露参数的处理

`epsilon_low` 和 `epsilon_high` 必须：

- 由配置显式提供；
- 写入 checkpoint 和 run manifest；
- 不在文档中伪称为 RAO 论文值；
- 支持关闭 lower clipping；
- 在训练前通过小规模数值测试验证 ratio、clip 和 stop-gradient。

是否增加 KL penalty、entropy bonus、length penalty 或 dynamic sampling：

- 默认关闭，保证先复现论文明确描述；
- 若开启，必须作为实验变体；
- metric 和 checkpoint 名称必须能区分；
- 不得把变体结果混入“RAO faithful”主结果。

# 7. End-to-end training pipeline

## 7.1 阶段 A：任务采样

从 root task dataset 采样 `batch_size` 个不同 task。论文默认：

```text
batch_size = 16 root tasks
```

DeepDive task record 至少包含：

- `id`；
- `question`；
- `answer`；
- `split`；
- `conversations`，若存在；
- dataset version/hash。

同一 optimizer batch 中，每个 root task 必须有独立 rollout group。

## 7.2 阶段 B：Policy snapshot

采样前冻结 behavior snapshot：

```text
behavior_policy_version
tokenizer_revision
prompt_template_version
tool_schema_version
environment_version
```

同一棵 tree 中所有 node 必须使用同一 behavior snapshot。

推荐同一 root group 的 8 棵 tree 也使用同一 snapshot；若异步系统允许跨 version，必须拒绝混组或在组级别明确标注并做严格实验区分。

## 7.3 阶段 C：Grouped recursive rollout

对每个 root task 并发采样 `G=8` 棵独立 tree：

- 不共享 node state；
- 不共享 conversation history；
- 可共享只读 cache；
- seed 不同；
- root prompt 相同；
- sampling config 相同；
- child task 完全由各自 policy trajectory 在线生成。

树内 subagent 可以并行，但 tree 必须能在结束后形成稳定 parent-child 关系。

## 7.4 阶段 D：Trace finalization

只有在以下信息完整后 tree 才能进入 verifier：

- 所有已创建 node 到达 terminal state；
- node trajectory 序列化完成；
- behavior logprobs 完整；
- generated token mask 完整；
- root/parent/children id 一致；
- policy version 已固定；
- fallback、timeout、truncation 已标注；
- tree checksum 生成。

不完整 tree 可以保存用于诊断，但默认不能进入训练。

## 7.5 阶段 E：Verification

先计算所有 node 的 success signal，再自底向上计算 reward。

顺序：

1. root task verifier；
2. subtask verifier；
3. verifier retry/fallback；
4. signal normalization 到 `[0,1]`；
5. reward computation。

因为 Eq.1 只依赖直接 children signal，不要求先计算 child reward，但自底向上的统一遍历更容易验证完整性。

## 7.6 阶段 F：Group advantage

对每个完整 root group：

1. 收集 8 个 root reward；
2. 为每棵 tree 计算 leave-one-out baseline；
3. 将 baseline 应用到该 tree 所有 trainable nodes；
4. 保存 `advantage = node_reward - baseline`。

## 7.7 阶段 G：Depth weighting

将当前 optimizer batch 的所有 trainable node sample 汇总，按 depth 计算：

- `depth_count`；
- `alpha`；
- `depth_weight`。

必须在过滤无效 sample 后计算，不能先统计再删除。

## 7.8 阶段 H：Optimization

训练后端接收扁平化 node samples，但保留：

- root task id；
- group id；
- tree id；
- node id；
- parent id；
- depth；
- reward provenance。

完成 forward、当前 logprob、ratio、CISPO loss、backward、gradient clipping 和 optimizer step。

## 7.9 阶段 I：Checkpoint publication

每次成功 optimizer step 后产生：

- monotonic policy version；
- model/adapter checkpoint；
- optimizer/scheduler state；
- RNG state；
- dataset sampler state；
- global step；
- consumed group ids；
- config snapshot；
- code revision；
- metrics snapshot。

新 rollout worker 只从已发布 checkpoint 拉取 policy。

# 8. DeepDive 复现设计

## 8.1 论文配置

DeepDive 原文设置：

- model：Qwen-3-4B-Instruct-2507；
- training context：40K；
- evaluation context：256K；
- max recursion depth：4；
- delegation bonus：`lambda = 0`；
- sub-agent scoring：GPT-5-mini LLM judge；
- web tools：Tavily search 和 webpage content；
- training backend：AReaL；
- learning rate：`3e-6`；
- root task batch size：16；
- rollout group size：8；
- objective：CISPO-style；
- max staleness：3 batches；
- recursive root/sub-agent：各 25 steps；
- 论文报告 75 training steps 后，recursive agent 相比 single-agent 已出现明显优势；
- held-out evaluation：50 tasks；
- single-agent held-out success：0.24；
- recursive held-out success：0.40。

## 8.2 当前数据映射

本仓库已有：

- `data/deepdive_qa_sft.csv`；
- `data/deepdive_qa_rl.csv`；
- `data/deepdive_trajectories_sft.csv`。

设计约定：

- `deepdive_qa_rl.csv` 作为 RAO root task 训练候选；
- `deepdive_qa_sft.csv` 和 trajectories 仅作为可选 warm start/格式验证来源；
- held-out split 必须从训练前固定，并保存 task id 列表；
- ground-truth answer 只给 verifier，不出现在 agent prompt；
- `conversations` 为空时不影响 root RL；
- 重复 id 或重复 question 必须在 split 前检查。

## 8.3 DeepDive root success

root answer 使用 ground truth 评估。为了兼容答案表述差异，设计支持：

1. normalized exact match；
2. configurable answer matcher；
3. LLM root judge。

论文附录提供的是 root answer 与 ground truth 的 LLM judge 结构。faithful profile 默认采用论文 judge prompt 语义：

- 输入 task；
- 输入 ground truth；
- 输入 agent final answer；
- 输出 strict JSON：`reason`、`success`；
- success 转为 0/1。

不得把 ground truth 放入 rollout prompt。

## 8.4 DeepDive subtask success

subtask 通常没有标注 ground truth，因此使用 LLM judge。judge 至少看到：

- root/parent task；
- delegated subtask；
- subagent final answer；
- child tasks；
- node status；
- 是否发生重复转发、越深度委派或无效委派；
- 可选 evidence/tool trace 摘要。

判定原则：

- answer 对 subtask 有帮助；
- delegation 是 concrete、non-overlapping、efficient；
- 不奖励仅换措辞后把完整 parent task 下传；
- 不奖励因越过 depth limit 造成的浪费；
- agent 可以不委派，只要正确完成 subtask；
- judge 输出 strict JSON；
- judge failure 不得自动当 success。

DeepDive faithful profile 使用 `lambda=0`，因此 child success 不进入 parent reward bonus，但 child 自身 success 仍用于 child node reward 和训练。

## 8.5 工具要求

当前默认工具只有 calculator、read_text 和 disabled python_exec，无法复现 DeepDive。训练前必须具备：

- `search_web(query, max_results)`；
- `view_webpage_content(url)`；
- `finish(answer)` 或等价 structured action；
- `launch_subagent(goal)`；
- async execution；
- tool timeout、retry、rate limit；
- URL/content redaction 和安全限制；
- tool result cache；
- deterministic fake tools 用于测试。

若没有真实 web search，不能宣称完成 DeepDive 训练复现。

## 8.6 结构化动作与论文 Python REPL 的差异

论文使用 CodeAct-style Python REPL；当前 harness 使用 strict JSON `AgentAction`。

这是一项重要复现差异。设计提供两个 profile：

### Faithful CodeAct profile

- 模型输出 `<thought>` + Python code；
- environment 执行 async tool/subagent calls；
- action token 全部保留；
- 最大程度接近论文 action space。

### Existing Harness profile

- 使用当前 JSON actions；
- 保留 `LAUNCH_SUBAGENT`、parallel launch、tool call、finish；
- 更容易复用现有代码和测试；
- 结果必须标记为 “RAO algorithm reproduction on structured-action harness”，不能宣称 action-space exact reproduction。

`training-plan.md` 应先选择一个主 profile。建议先完成 Existing Harness profile 的算法正确性，再增加 Faithful CodeAct profile。

# 9. 其他论文域的扩展设计

## 9.1 TextCraft-Synth

论文设置：

- Qwen-3-4B-Instruct-2507；
- train only medium difficulty；
- easy depth 2-3；
- medium depth 4-6；
- hard depth 7-9；
- constrained：8K train / 8K eval；
- unconstrained：40K train / 256K eval；
- training max recursion depth 6；
- evaluation max recursion depth 12；
- `lambda=0`；
- exact environment verifier；
- root/sub-agent 各 25 steps。

该域适合作为算法单测和可验证 reward 的首个完整 RL 环境，因为不依赖昂贵 LLM judge。

## 9.2 Oolong-Real

论文设置：

- Qwen3-VL-30B-A3B-Instruct；
- Tinker；
- LoRA rank 32；
- learning rate `3e-5`；
- train context 32K；
- eval context 256K；
- train inputs 过滤到 `<=240K characters`，约 60K tokens；
- eval 最长到约 220K tokens；
- max recursion depth 2，0-indexed，共 3 层；
- root scoring 使用 benchmark partial-credit rules；
- subtask 使用 GPT-5-mini judge；
- `lambda=0.4`；
- root/sub-agent 各 15 steps；
- context 预置在 Python interpreter，child 接收切片 context。

本项目不应在 DeepDive MVP 中耦合 Oolong 的 context-chunk 特殊逻辑，应通过 domain adapter 扩展。

# 10. 数据模型

## 10.1 RootTaskRecord

```yaml
schema_version: "1.0"
task_id: "deepdive:rl:000001"
domain: "deepdive"
split: "train"
prompt: "..."
ground_truth: "..."
metadata:
  source_row_id: "1"
  dataset_hash: "..."
```

## 10.2 RolloutGroupRecord

```yaml
schema_version: "1.0"
group_id: "group_..."
task_id: "deepdive:rl:000001"
expected_group_size: 8
completed_group_size: 8
behavior_policy_version: "policy_000075"
sampling_config_hash: "..."
rollout_ids: ["...", "..."]
status: "complete"
created_at: "..."
```

## 10.3 RolloutTreeRecord

```yaml
schema_version: "1.0"
rollout_id: "rollout_..."
group_id: "group_..."
rollout_index: 0
seed: 1234
root_node_id: "node_0001"
behavior_policy_version: "policy_000075"
prompt_template_version: "deepdive_recursive_v1"
tool_schema_version: "deepdive_tools_v1"
environment_version: "deepdive_env_v1"
nodes: {}
tree_status: "complete"
checksum: "..."
```

## 10.4 TrainingNodeRecord

```yaml
schema_version: "1.0"
task_id: "deepdive:rl:000001"
group_id: "group_..."
rollout_id: "rollout_..."
node_id: "node_0004"
parent_id: "node_0002"
depth: 2
node_task: "..."
children_ids: []
terminal_status: "completed"
terminal_reason: "finish"
is_fallback: false
is_trainable: true

model_input:
  messages: []
  template_version: "..."
  tokenizer_name: "..."
  tokenizer_revision: "..."

tokens:
  input_ids: []
  generated_ids: []
  action_mask: []
  behavior_logprobs: []
  truncated: false

evaluation:
  success_signal: 1.0
  provider: "llm_judge"
  provider_version: "..."
  reason: "..."
  is_proxy: false

credit:
  own_success: 1.0
  child_success_mean: 0.0
  lambda: 0.0
  reward: 1.0
  loo_baseline: 0.625
  advantage: 0.375
  depth_count: 64
  depth_alpha: 17.5
  depth_weight: 0.2734375
```

## 10.5 OptimizerBatchManifest

```yaml
schema_version: "1.0"
optimizer_step: 75
policy_version_before: "policy_000074"
policy_version_after: "policy_000075"
root_task_count: 16
group_size: 8
tree_count: 128
trainable_node_count: 560
depth_counts:
  "0": 128
  "1": 240
  "2": 128
  "3": 48
  "4": 16
depth_weights: {}
max_staleness_batches: 3
objective: "rao_cispo"
config_hash: "..."
```

# 11. 逻辑组件与接口边界

本节定义职责，不规定最终文件名。

## 11.1 TaskDataset

职责：

- 加载 CSV/JSONL；
- 固定 train/eval split；
- 去重；
- stable task id；
- dataset hash；
- 向 agent 隐藏 ground truth。

## 11.2 PolicySnapshotProvider

职责：

- 发布 immutable behavior snapshot；
- 提供 model/tokenizer revision；
- 生成 policy version；
- 支持 worker 拉取；
- 保证 tree 内共享 snapshot。

## 11.3 RecursiveRolloutCollector

职责：

- 对 root task 采样 `G` 棵 tree；
- 调用现有 harness；
- 捕获精确 token/logprob；
- 保存 tree；
- 校验完整性；
- 支持并发和 backpressure。

## 11.4 SuccessSignalProvider

统一接口：

```text
evaluate(node, tree, root_task) -> SuccessSignalResult
```

实现：

- exact verifier；
- partial-credit verifier；
- LLM root judge；
- LLM subtask judge；
- root-proxy verifier。

## 11.5 RewardComputer

职责：

- 实现 Eq.1；
- 纯函数；
- 写 provenance；
- 支持 replay；
- 对异常 signal 拒绝计算。

## 11.6 AdvantageComputer

职责：

- group completeness validation；
- Eq.3 leave-one-out baseline；
- 将 tree baseline 分配给所有 node；
- 处理 `G < 2`；
- 数值稳定性检查。

## 11.7 DepthWeightComputer

职责：

- 基于 trainable sample 统计 \(N_d\)；
- 计算 alpha 和 \(w_d\)；
- 验证总权重守恒；
- 写入 batch manifest。

## 11.8 TrajectoryCompiler

职责：

- 从 node rollout record 生成 token-level sample；
- 对 assistant/action tokens 建 mask；
- 验证 logprob 长度；
- 检查 tokenizer/template version；
- 不把 observation token 当 action；
- 支持 packed/unpacked batch。

## 11.9 PolicyOptimizer

职责：

- 重算 current logprobs；
- 计算 IS ratio；
- CISPO clipping；
- 应用 node advantage 和 depth weight；
- backward/update；
- 输出 optimizer metrics；
- 保存 checkpoint。

## 11.10 AsyncCoordinator

职责：

- rollout queue；
- verifier queue；
- train queue；
- policy publication；
- staleness 计算；
- 超过 3 batches 的 sample 丢弃或重新采样；
- graceful shutdown 和 resume。

# 12. 异步训练与 staleness

## 12.1 定义

设 sample 使用 policy version \(v_b\) 采样，当前 optimizer version 为 \(v_t\)：

```text
staleness = v_t - v_b
```

也可以按 batch publication sequence 定义，但全系统只能采用一种定义。

faithful profile：

```text
max_staleness_batches = 3
```

## 12.2 约束

- staleness 必须写入每个 batch/sample；
- 超限 sample 默认不训练；
- 不允许只靠 ratio clipping 无限接收陈旧数据；
- 同一 root group 不得部分超限、部分有效后继续计算错误 baseline；
- group 因 staleness 被拒绝时整体拒绝；
- resume 后 policy version 不能回退；
- worker 使用旧 policy 时允许完成当前 tree，但 coordinator 决定是否接收。

# 13. 失败、超时与训练资格

## 13.1 Node 失败分类

- `policy_parse_error`；
- `tool_error`；
- `tool_timeout`；
- `model_timeout`；
- `node_timeout`；
- `step_limit`；
- `depth_limit`；
- `child_limit`；
- `total_node_limit`；
- `cancelled`；
- `fallback_finish`；
- `judge_error`；
- `trace_incomplete`；
- `token_logprob_missing`。

## 13.2 默认 success 语义

- verifier 明确成功：按 verifier 分值；
- verifier 明确失败：0 或 partial score；
- timeout/fallback：默认 0；
- judge unavailable：signal missing，不自动设为 0，也不自动设为成功；
- signal missing 的 tree 默认不能进入 faithful training；
- tool 部分失败但最终答案正确：由 verifier 决定，不按 status 硬编码。

## 13.3 Trainability

一个 node 进入 loss 必须满足：

- tree 完整；
- group 完整；
- generated token 非空；
- behavior logprob 完整；
- tokenizer/template 匹配；
- reward 和 advantage 有限；
- depth weight 有限；
- staleness 合法；
- 非 synthetic fallback；
- 未发生不支持的截断。

是否训练失败 trajectory 是独立配置。faithful RAO 应允许 verifier 判为 0 的有效失败 trajectory 进入训练；不能只保留成功样本，否则退化为 rejection sampling/SFT。

# 14. 配置设计

建议训练配置分为以下命名空间：

```yaml
experiment:
  name: "deepdive_rao"
  seed: 42
  faithful_profile: true

dataset:
  train_path: "data/deepdive_qa_rl.csv"
  eval_path: ""
  split_manifest_path: ""

model:
  name: "Qwen-3-4B-Instruct-2507"
  tokenizer_revision: ""
  train_context_tokens: 40960
  eval_context_tokens: 262144

rollout:
  root_batch_size: 16
  group_size: 8
  max_depth: 4
  max_steps_per_node: 25
  temperature: 1.0
  max_staleness_batches: 3

reward:
  lambda: 0.0
  root_provider: "llm_judge"
  subtask_provider: "llm_judge"

advantage:
  type: "root_group_leave_one_out"

weighting:
  type: "depth_inverse_frequency"

objective:
  type: "cispo"
  epsilon_low: null
  epsilon_high: null
  kl_coefficient: 0.0

optimizer:
  learning_rate: 3.0e-6
  weight_decay: null
  grad_clip_norm: null

checkpoint:
  output_dir: "output/train/deepdive_rao"
  save_every_steps: 1

evaluation:
  heldout_task_count: 50
  every_steps: 5
```

所有 `null` 表示论文未披露，必须在实际运行配置中显式解析为实现选择，不能静默填充后仍称为 faithful default。

# 15. 指标与可观测性

## 15.1 Rollout metrics

- trees/task；
- nodes/tree；
- max depth；
- depth histogram；
- steps/node；
- tool calls/node；
- children/node；
- sequential vs concurrent delegation ratio；
- timeout/fallback/error ratio；
- input/output/cache token；
- wall-clock time；
- policy version/staleness。

## 15.2 Reward metrics

- root success rate；
- node success rate by depth；
- reward mean/std by depth；
- child success mean；
- delegation bonus mean；
- judge failure/retry rate；
- reward provider distribution。

## 15.3 Optimization metrics

- advantage mean/std/min/max；
- baseline mean/std；
- depth counts/weights；
- loss；
- current/behavior logprob；
- IS ratio quantiles；
- clipped ratio fraction；
- gradient norm；
- token count；
- policy entropy；
- update time；
- dropped stale groups。

## 15.4 Evaluation metrics

至少对比：

- base model single agent；
- RL-trained single agent；
- base model recursive agent；
- RAO-trained recursive agent。

DeepDive 报告：

- held-out success rate；
- pass@8，如可实现；
- steps；
- wall-clock time；
- token usage；
- max depth；
- concurrent delegation ratio；
- judge cost。

# 16. 正确性不变量

必须自动验证：

1. 一棵 tree 的所有 node 具有同一 behavior policy version。
2. 一个 group 的所有 tree 具有同一 root task id。
3. faithful group size 等于 8。
4. `G >= 2`。
5. 每个非 root node 的 parent 存在。
6. children edge 双向一致。
7. depth 等于 parent depth + 1。
8. reward 只使用直接 children success。
9. leaf delegation bonus 为 0。
10. LOO baseline 不包含本 rollout root reward。
11. 同一 tree 所有 node baseline 相同。
12. advantage 等于 node reward 减 tree baseline。
13. depth weight 总和等于 trainable node 数。
14. 每个非空 depth 的总 weight 相同。
15. generated ids、action mask、behavior logprobs 长度一致。
16. policy loss 不覆盖 prompt/observation tokens。
17. ground truth 不出现在 agent prompt。
18. judge output 不进入 agent context。
19. stale group 不进入 optimizer。
20. fallback placeholder 不被当成正常成功。

# 17. 测试设计

## 17.1 纯数学单元测试

### Reward

- leaf；
- 一个 child；
- 多个 child；
- lambda 0；
- lambda 0.4；
- partial success；
- failed child；
- 只用 direct children。

### Leave-one-out

使用固定 root rewards：

```text
[1.0, 0.0, 0.5, 1.0]
```

验证每个 rollout baseline；
验证 child 使用相同 tree baseline；
验证本 rollout 不进入 baseline；
验证 `G=1` 报错。

### Depth weight

使用：

```text
N0=2, N1=4, N2=8
```

验证：

- alpha；
- 每个 depth 的 per-trajectory weight；
- 每层总权重相同；
- 总权重为 14。

### CISPO

- ratio 为 1；
- ratio upper clip；
- lower clip enabled/disabled；
- clipped weight stop-gradient；
- action mask；
- negative/positive advantage；
- depth weight scaling；
- 无 NaN/Inf。

## 17.2 Schema tests

- JSON round trip；
- schema version；
- unknown/missing fields；
- tokenizer mismatch；
- logprob length mismatch；
- checksum；
- secret redaction；
- old inference trace migration。

## 17.3 Tree integration tests

- root only；
- depth 1；
- depth >1；
- sequential child；
- parallel children；
- child failure；
- root timeout；
- fallback；
- incomplete tree；
- shared policy snapshot。

## 17.4 Group integration tests

- 1 task × 8 rollouts；
- 16 tasks × 8 rollouts fake backend；
- mixed tree sizes；
- missing rollout；
- duplicated rollout index；
- mixed policy versions；
- staleness boundary 3；
- staleness 4 rejection。

## 17.5 Deterministic mini-training test

使用 tiny policy/fake environment：

- rollout；
- verifier；
- reward；
- advantage；
- weight；
- one optimizer step；
- checkpoint；
- resume；
- loss/parameter 确实变化。

测试不依赖公网和真实 judge。

## 17.6 DeepDive smoke test

只用少量 task：

- 真实 search tool；
- 真实 recursive rollout；
- root/subtask judge；
- 生成完整 training batch；
- 可选择不做大模型参数更新；
- 验证所有训练字段完整。

# 18. 消融与复现实验

论文关键消融要求：

## 18.1 Reward design

- `Dense`：root/subagent 使用各自 task-specific success；
- `Sparse`：只使用 root reward，并传播给 subagent；
- 其他配置相同。

## 18.2 Trajectory weighting

- `Weighted`：启用 depth inverse-frequency；
- `Unweighted`：所有 node weight 1。

形成四组：

```text
Dense + Weighted
Dense + Unweighted
Sparse + Weighted
Sparse + Unweighted
```

该消融必须和主训练使用相同：

- model init；
- dataset split；
- root batch；
- group size；
- rollout limits；
- optimizer；
- random seed policy；
- evaluation protocol。

# 19. Checkpoint、恢复与可复现性

checkpoint 至少保存：

- model/LoRA state；
- optimizer state；
- scheduler state；
- global optimizer step；
- policy version；
- RNG states；
- data sampler state；
- completed task/group ids；
- in-flight queue metadata；
- config；
- prompt/tool/verifier versions；
- code git revision；
- schema versions。

resume 要求：

- 不重复消费已训练 group；
- 不接受来自未来/未知 policy version 的 sample；
- unfinished tree 可丢弃并重采样；
- checkpoint 原子写入；
- manifest 和 checkpoint 一致；
- 恢复后 first step 的指标可追踪。

# 20. 安全与数据治理

- API key 不进入 trace；
- search/judge raw response 可单独加密或按策略留存；
- 网页内容可能包含 prompt injection，tool layer 应标注不可信内容；
- ground truth 必须与 agent-visible data 分离；
- judge prompt 和 agent prompt 分离；
- dataset license 和来源需记录；
- checkpoint 不包含明文 secret；
- trace 发布前必须 redaction；
- 训练日志避免完整打印长 task、网页和 key。

# 21. 性能设计

## 21.1 主要成本

RAO 的成本来自：

- 每个 task 8 棵 tree；
- 每棵 tree 多 node；
- 每个 node 多步生成；
- web/tool 调用；
- root/subtask judge；
- current policy 重算 logprob。

## 21.2 优化原则

- rollout、judge、optimizer 解耦；
- tree 内安全并发；
- root group 跨 tree 并发；
- tool cache 不改变语义；
- judge batching；
- token packing；
- backpressure；
- 限制 in-flight groups；
- 保存中间产物，避免 judge 或 reward 重算；
- 先做小规模 profile，再扩大 batch。

不得为了吞吐破坏：

- group 独立性；
- policy snapshot 一致性；
- exact prompt/token reconstruction；
- staleness 上限；
- reward provenance。

# 22. 验收标准

## 22.1 算法验收

- Eq.1、Eq.3、Eq.4、Eq.5 有独立测试；
- root-group LOO 语义正确；
- depth weighting 总权重守恒；
- 所有 node 联合更新共享 policy；
- CISPO ratio 和 clipping 可审计；
- failure trajectory 不被静默过滤成成功。

## 22.2 数据验收

- 任意 optimizer sample 可追溯到 task/group/tree/node；
- 任意 reward 可追溯到 verifier；
- 任意 advantage 可重算；
- 任意 depth weight 可重算；
- 任意 token loss 可追溯到 behavior logprob；
- ground truth 未泄露到 agent prompt。

## 22.3 系统验收

- deterministic fake backend 完成端到端训练；
- 支持 checkpoint/resume；
- 支持 max staleness 3；
- 支持并行 subagents；
- 支持 DeepDive tools；
- judge 失败不会污染训练；
- trace 不泄露 secret。

## 22.4 复现实验验收

至少完成：

1. base recursive rollout；
2. RL single-agent baseline；
3. RAO recursive training；
4. held-out evaluation；
5. dense/sparse reward 消融；
6. weighted/unweighted 消融；
7. depth 和 delegation 行为分析。

论文数值是目标参考，不作为代码正确性的唯一标准。由于官方代码尚未公开、基础设施和部分超参数未披露，必须同时报告：

- faithful settings；
- 本地实现差异；
- compute 差异；
- data/tool/judge 差异；
- 结果置信区间和随机种子。

# 23. `training-plan.md` 生成约束

后续根据本文档生成 `training-plan.md` 时必须：

- 按依赖顺序拆成可执行步骤；
- 每步列出新增/修改文件；
- 每步有测试和验收命令；
- 先 schema 与纯数学模块，再 rollout capture，再 verifier/reward，再 optimizer；
- 先 deterministic fake backend，再真实 DeepDive；
- 明确哪些步骤会修改现有 inference harness；
- 不在同一步同时重写 harness、训练后端和工具系统；
- 保留 Existing Harness profile 与 Faithful CodeAct profile 的边界；
- 所有论文未披露参数都作为显式配置和实验决策；
- 任何阶段都不得把 `completed` 等同于 success；
- 不允许在缺少 token/logprob 数据时直接开始 policy optimization；
- 不允许在 group 不完整时计算 leave-one-out advantage；
- 不允许生成 `G=1` 的所谓 RAO faithful training；
- 不允许只训练 root node；
- 不允许先按 depth 计数、后过滤 sample；
- 不允许把 judge 模型与训练 policy 混为同一个被更新对象。

# 24. 最终设计原则

1. **Tree is rollout structure, node is training unit.**
2. **One shared policy is optimized across every depth.**
3. **Success、reward、advantage、weight 是四个不同概念。**
4. **Local reward 只看自身和直接 children。**
5. **LOO baseline 来自其他 rollout 的 root reward。**
6. **Depth weighting 在最终 trainable node batch 上计算。**
7. **Behavior token/logprob 是 RL 数据，不是调试附件。**
8. **Failure is training data, fallback is not success.**
9. **Ground truth belongs to verifier, never to the agent prompt.**
10. **Paper facts and implementation choices must remain distinguishable.**
11. **先证明算法正确，再扩大模型、数据和并发规模。**
12. **所有中间量必须可重放、可审计、可追溯。**
