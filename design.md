请帮我为一个 RAO-style Recursive Agent Inference Harness 项目生成 plan.md。

注意：本阶段只生成设计文档 plan.md，不要生成任何代码文件。  
后续我会基于 plan.md 再让你逐步实现代码。

项目目标：

复现论文 Recursive Agent Optimization 中的 inference harness，而不是训练 RAO。  
原文如下,其中图片在根目录的/figure下。
```
⸻

Recursive Agent Optimization

Abstract

We introduce Recursive Agent Optimization (RAO), a reinforcement learning approach for training recursive agents: agents that can spawn and delegate sub-tasks to new instantiations of themselves recursively. Recursive agents implement an inference-time scaling algorithm that naturally allows agents to scale to longer contexts and generalize to more difficult problems via divide-and-conquer.

RAO provides a method to train models to best take advantage of such recursive inference, teaching agents when and how to delegate and communicate. We find that recursive agents trained in this way enjoy better training efficiency, can scale to tasks that go beyond the model’s context window, generalize to tasks much harder than the ones the agent was trained on, and can enjoy reduced wall-clock time compared to single-agent systems.

⸻

1 Introduction

Large language model agents are increasingly deployed on real-world tasks such as software engineering, research assistance, and computer use. As these systems improve, the tasks users expect them to solve are also becoming harder: they involve longer horizons, larger effective working memory, and substantial exploration and backtracking. Many such tasks are naturally amenable to divide-and-conquer. For instance, a large software change can be decomposed into code investigation and editing subproblems; a research task can be split into retrieval, synthesis, and verification stages; a long document or log corpus can be partitioned into manageable pieces and processed in parallel.

This has led to interest in recursive agent systems, in which an agent can spawn sub-agents to solve subtasks with fresh contexts. Recursive execution offers several advantages. First, it expands effective working memory, since each sub-agent receives a fresh context window rather than inheriting the full history of its parent. Second, it enables divide-and-conquer by breaking a large task into smaller, more tractable subproblems. Third, when subproblems can be solved independently, recursion can exploit concurrency and reduce wall-clock latency for accomplishing a task. Together, these properties make recursion a compelling inference-time scaling primitive for agents.

However, most existing recursive and multi-agent systems treat recursion purely as an inference-time scaffold wrapped around a pretrained model. The model itself is typically not trained to decide when delegation is useful, how to formulate effective subtasks, how to communicate information across levels of the execution tree, or how to combine sub-agent outputs into a final solution. If recursive execution is going to be a core test-time primitive, then the policy should be trained to use it well. Prior work shows that training models with inference-time scaffolds in the loop does make them more amenable to deployment with inference-time scaffolds. This raises a central question: how should we train a model to exploit recursive inference effectively?

In this work, we introduce Recursive Agent Optimization, a reinforcement learning approach for end-to-end training of recursive agents. RAO trains a single LLM policy that is instantiated at every node of a recursively generated execution tree. As a result, the same model must learn both how to solve the task assigned to it and how to generate useful delegated subtasks for spawned copies of itself.

Importantly, RAO is not restricted to a fixed hierarchy or hand-designed orchestration scheme: it supports dynamically generated recursive execution trees and optimizes all decisions in the tree jointly. Training recursive agents may also improve learning efficiency, not just inference-time behavior. Recursive execution naturally generates related tasks at multiple levels of difficulty, yielding structured intermediate supervision and an implicit curriculum over simpler subproblems. Thus, the recursive structure used at inference time can also provide a useful training signal for generally improving model capability. This motivates a second question: how can we leverage the recursive structure of inference to better train agents?

We evaluate RAO on three benchmarks, including deep research, long document processing, and TEXTCRAFT-SYNTH, a controlled synthetic environment inspired by Minecraft-style crafting tasks, in which task horizon, difficulty, and recursive structure can be varied. The paper finds that training for recursive execution yields several benefits. Recursive agents trained with RAO solve tasks that exceed the base model’s context window, generalize to substantially harder tasks than those seen during training by leveraging recursive delegation, and when tasks can be decomposed into parallel subproblems, reduce wall-clock execution time relative to non-recursive baselines. In addition, RAO improves training efficiency over single-agent training by exploiting recursive decomposition during learning.

⸻

2 Recursive Agents: Inference and Training

The paper studies recursive agents: agents that can delegate sub-tasks to new instances of themselves during execution. A rollout no longer resembles a single flat trajectory, but instead forms a dynamically generated tree of trajectories. While prior work and deployed systems have explored recursive inference, this work focuses on how to train a model to leverage recursion effectively. The model must learn how to solve assigned tasks directly, when delegation is beneficial, how to formulate sub-tasks for child agents, and how to combine their outputs into a final solution.

Unlike approaches based on fixed hierarchies or hand-specified orchestration, this setting allows the policy to generate execution trees dynamically, with arbitrary branching patterns up to a depth limit. RAO trains all agent instances in the tree jointly.

⸻

2.1 Recursive Agent Inference

Formulation

A task X is a problem specification given to an agent. When the policy \pi_\theta is run on X, it produces a trajectory \tau_X, consisting of the sequence of actions and observations used to solve the task.

During this rollout, the agent may decide to spawn zero or more child agents on delegated sub-tasks

X_1, \ldots, X_n,

where both the number and the content of these sub-tasks are chosen by the policy itself. This process induces a rooted execution tree T: each node corresponds to one agent instance attempting one assigned task, the root corresponds to the original task, and the children of a node correspond to the delegated sub-tasks generated by that agent. Sub-agents can also recursively spawn their own child agents.

Implementation

The paper instantiates recursive agents as an extension of an agent that interleaves natural-language reasoning with code execution in a Python REPL. This interface gives the model access to standard programming constructs such as variables, control flow, string manipulation, and asynchronous execution, allowing it to solve tasks by writing short programs and executing them.

To support recursion, the agent’s action space is extended by exposing an asynchronous function:

\texttt{async launch\_subagent(goal, ...)} \rightarrow \texttt{Any}

This function launches a new instance of the same policy on a delegated sub-task and returns the child agent’s output to the parent. Because the return type is unrestricted, parents can delegate a wide range of subproblems and request outputs in whatever format is useful for downstream computation, including structured objects rather than only strings.

The parent interacts with child agents through ordinary Python control flow. It can launch children sequentially when later sub-tasks depend on earlier results, or concurrently using standard Python libraries such as asyncio when sub-tasks are independent. Returned objects can be stored in variables, transformed, combined, or passed into later sub-agent calls.

This makes recursive decomposition flexible: the policy decides when to delegate, what sub-task specification to provide, what type of output to request, whether to run child agents serially or in parallel, and how to aggregate their results.

To ensure bounded computation, explicit limits are imposed on recursion depth and on the number of environment steps available to each agent instance. The implementation resembles Recursive Language Models, with the main differences that this work demonstrates positive results for recursion depths greater than one and supports concurrent sub-agent execution by implementing delegation as an asynchronous function.

⸻

2.2 Recursive Agent Optimization

A rollout on a root task produces an execution tree whose nodes correspond to recursively instantiated copies of the same policy. RAO trains all nodes in this tree jointly. At a high level, RAO combines a local reward defined at each node with policy optimization over recursively generated execution trees.

⸻

2.2.1 Local Node Reward

A recursive agent should learn both to solve its assigned task and, when useful, to delegate productively. Ideally, one can take advantage of node-local credit assignment: a sub-agent should receive signal from whether its own assigned task and its delegated sub-tasks were solved successfully, rather than relying only on the root agent’s outcome-level reward.

Such local signals can substantially improve credit assignment for decomposition, especially early in training when root-task success may be rare or when the initialized policy does not yet make meaningful use of delegation. At the same time, in many environments fine-grained verification is unavailable or expensive, so direct success signals for intermediate sub-agents may not always be observable.

To accommodate both cases, the paper distinguishes between the underlying notion of task success and the supervisory signal actually used during training for each sub-agent. For a node corresponding to task X, let \tau_X denote the trajectory generated while solving X, and let C(X) denote the set of its immediate children.

The paper uses

\tilde{s}(X, \tau_X) \in [0, 1]

for the success signal used in training for node X. Depending on the environment, \tilde{s}(X, \tau_X) may be instantiated in different ways: it may come from exact sub-task verification, from a learned or LLM-based judge, or, when no node-local supervision is available, from a proxy such as the success of the root task. RAO’s credit assignment becomes more faithful as \tilde{s}(X, \tau_X) more accurately reflects the success of the corresponding node-local task X.

The reward for a given node is defined as:

R(X, \tau_X)
=
\underbrace{\tilde{s}(X, \tau_X)}_{\text{success}(X)\ /\ \text{proxy}}
+
\lambda \cdot
\underbrace{
\frac{1}{|C(X)|}
\sum_{c \in C(X)}
\tilde{s}(c, \tau_c)
}_{\text{delegation bonus}}
\tag{1}

with the convention that the second term is zero when |C(X)| = 0. Here, \lambda \geq 0 controls the strength of the delegation bonus.

The first term rewards the agent for solving its own assigned task according to the available supervisory signal. The second term rewards the agent when the sub-tasks it creates are successfully completed by its children. Using the success rate of immediate children, rather than the raw number of successful children, avoids directly rewarding the policy for spawning more children merely to collect additional bonus. This makes the delegation bonus better aligned with the quality of delegation rather than the quantity of delegation.

The reward is local: every node, whether root, internal, or leaf, is scored using the same rule based only on its own task signal and the signals of its immediate children. Setting \lambda = 0 recovers a purely local-success-based reward. In practice, the delegation bonus is most useful in regimes where the initial policy under-utilizes delegation and needs additional signal to learn when and how to decompose. When the policy already delegates sufficiently at initialization, one can set \lambda = 0 and train with the node-level success signal alone.

⸻

2.2.2 Policy Optimization Objective

Recursive execution induces a family of related task distributions across depths. Let D_0 denote the root task distribution, and let D_d(\theta) for d \geq 1 denote the distribution of depth-d sub-tasks generated by recursively applying the current policy.

Training therefore optimizes a shared policy over both root tasks and policy-generated descendants:

J(\theta)
=
\sum_{d=0}^{D}
\mathbb{E}_{X \sim D_d(\theta)}
\left[
\mathbb{E}_{\tau_X \sim \pi_\theta(\cdot \mid X)}
\left[
R(X, \tau_X)
\right]
\right].
\tag{2}

This view helps explain why recursive training can be effective: the same parameters are trained across a hierarchy of related tasks, and the induced sub-tasks are often simpler or more structured than the original root task, generating a natural curriculum.

For each root task, the method samples G independent recursive rollout trees:

\{T^{(g)}\}_{g=1}^{G}.

Let R^{(g)}_{\text{root}} denote the reward of the root node in rollout g. For any trajectory \tau belonging to rollout g, its advantage is defined using a leave-one-out baseline over root rewards:

A(\tau^{(g)})
=
R(\tau^{(g)}) - b_{-g},
\qquad
b_{-g}
=
\frac{1}{G-1}
\sum_{g' \neq g}
R^{(g')}_{\text{root}}.
\tag{3}

The same root-group baseline is used for all trajectories within a rollout tree, including child trajectories. Although this may not be the lowest-variance baseline, especially when each sub-task is different, it places all nodes generated under the same root task on a common reference scale and is practical because it avoids the need for a critic or for constructing separate comparison groups over policy-generated sub-tasks, which generally differ across rollouts. A standard leave-one-out argument shows that this baseline is unbiased.

A practical issue is that recursive rollouts may contain very different numbers of trajectories at different depths. In some domains, sub-agent trajectories can greatly outnumber root trajectories, causing optimization to become dominated by deeper parts of the execution tree if all trajectories are simply pooled together. To mitigate this effect, RAO uses depth-level inverse-frequency weighting, which downweights trajectories from depths that are overrepresented in the batch while preserving the overall scale of the update.

Let B_d denote the set of all depth-d trajectories in the batch, and let

N_d = |B_d|.

Each trajectory at depth d is assigned a weight:

w_d
=
\alpha \cdot \frac{1}{N_d},
\qquad
\alpha
=
\frac{\sum_{d=0}^{D} N_d}
{\sum_{d=0}^{D} N_d \cdot \frac{1}{N_d}}.
\tag{4}

Here, \alpha is a normalization constant chosen so that the total weight over the batch is preserved. The corresponding estimator is:

\widehat{\nabla} J(\theta)
=
\sum_{d=0}^{D}
\sum_{\tau \in B_d}
w_d A(\tau)
\nabla_\theta \log \pi_\theta(\tau).
\tag{5}

Intuitively, this assigns smaller weight to trajectories from depths that appear more frequently in the batch, reducing the tendency of heavily populated levels of the tree to dominate learning, while the normalization keeps the overall update magnitude approximately unchanged.

⸻

Summary: RAO

RAO trains a single, shared policy to act across recursive rollouts with dynamically generated execution trees.

It provides dense credit assignment through a local reward for each node in the recursive execution tree, as defined in Equation 1.

It computes advantages by comparing each node’s local reward to a leave-one-out baseline computed from root rollout rewards, as defined in Equation 3.

It optimizes a weighted, multi-task objective over tasks sampled from different depths of recursive rollouts, yielding a self-induced curriculum, as described in Equations 2 and 5.
```


核心是实现一个 recursive agent execution harness：一个 agent 在执行过程中可以动态 spawn 子 agent，子 agent 是同一个 policy 的新实例，形成动态递归执行树。每个节点对应一个 agent instance，负责完成自己被分配的 task，并可以继续递归委派子任务。

需要体现论文中的几个关键点：

- shared policy：root agent 和所有 sub-agent 使用同一个 policy wrapper；
- dynamic recursive execution tree：递归树不是预先写死的，而是由 policy 在运行时决定；
- node = agent instance：每个节点都是一个完整 agent；
- edge = delegated sub-task：父节点通过 launch_subagent 委派子任务；
- asynchronous delegation primitive：支持 async launch_subagent；
- parallel subagents：支持 asyncio.gather 并行执行独立子任务；
- bounded computation：支持 max_depth、max_steps、max_children、total_node_limit、timeout；
- traceable rollout：完整记录 recursive rollout tree。

请生成一个完整、工程化、可执行的 plan.md，内容需要包括以下部分。

# 1. 项目定位

说明本项目要复现的是 RAO inference harness，不包含 RAO training。  
解释 inference harness 与 training objective 的边界：

- 本项目实现 recursive rollout tree 的生成与执行；
- 不实现 Eq.3 leave-one-out advantage；
- 不实现 Eq.5 policy optimization；
- 但 trace 结构需要为未来接入 reward / training 保留扩展点。

# 2. 技术选型建议

请给出推荐技术栈，并说明理由。

需要覆盖：

## 2.1 Python 版本

建议 Python 3.10+ 或 3.11。  
说明原因：asyncio 支持成熟、类型系统较好、生态兼容性强。

## 2.2 数据结构建模

请比较 dataclass 和 Pydantic。

需要给出最终建议：

- 内部 runtime 结构可以用 dataclass；
- LLM action / state / config / trace schema 建议用 Pydantic；
- 原因是 LLM 输出需要 JSON schema 校验、字段默认值、错误恢复。

## 2.3 异步框架

建议使用原生 asyncio，而不是 Celery / Ray / multiprocessing。

说明原因：

- RAO harness 的核心是 I/O-bound LLM/tool call；
- asyncio.gather 足够复现 asynchronous subagent execution；
- 简化本地 demo 与测试。

同时说明未来如果要扩展到大规模分布式，可以抽象 Runner / Executor 接口，后续替换为 Ray。

## 2.4 LLM 接入

建议实现 OpenAI-compatible client wrapper，而不是绑定某个 provider。

要求支持：

- OpenAI API；

需要说明：

- harness 不应该依赖具体模型 provider。

## 2.5 配置管理

建议使用 YAML + Pydantic Settings 或简单 dataclass config。

需要包含：

- max_depth
- max_steps_per_node
- max_children_per_node
- total_node_limit
- timeout_seconds
- model_name
- base_url
- api_key
- temperature
- max_tokens
- trace_output_path

## 2.6 日志与 trace

建议：

- runtime logging 使用 Python logging；
- structured trace 使用 JSON；
- tree visualization 支持 pretty text 和 Mermaid；
- 每个 node 记录 trajectory、children、status、error、start_time、end_time。

## 2.7 测试框架

建议使用 pytest + pytest-asyncio。

需要测试：

- depth limit；
- total node limit；
- parallel subagent execution；
- invalid action fallback；
- trace export；
- shared policy identity；
- RuleBasedPolicy demo 行为。

# 3. 总体架构

请设计项目结构：

recursive_agent_harness/
    init.py
    agent.py
    policy.py
    actions.py
    state.py
    tools.py
    tree.py
    prompts.py
    config.py
    runner.py
    errors.py

examples/
    run_recursive_agent.py

tests/
    test_tree.py
    test_rule_based_policy.py
    test_depth_limit.py
    test_parallel_subagents.py
    test_invalid_action.py
    test_shared_policy.py

README.md
plan.md

请解释每个文件职责。

重点要求：

- agent.py 只负责执行 action，不负责决定如何拆任务；
- policy.py 负责根据 state 生成 action；
- actions.py 定义结构化 action schema；
- state.py 定义 AgentState / NodeResult / TrajectoryStep；
- tree.py 负责 ExecutionTree / TraceLogger；
- tools.py 负责工具注册与执行；
- prompts.py 保存 LLMPolicy 的 system prompt 和 state rendering；
- runner.py 负责 root agent 启动、timeout、全局预算管理；
- config.py 管理参数；
- errors.py 定义 HarnessError / DepthLimitError / NodeLimitError / InvalidActionError 等。

# 4. 核心抽象设计

请详细设计以下类和数据结构。

## 4.1 RecursiveAgent

字段至少包括：

- node_id
- parent_id
- task
- depth
- max_depth
- policy
- tools
- tree
- budget
- parent_context
- expected_output
- constraints
- trajectory
- children
- final_answer
- status

方法至少包括：

- async run() -> NodeResult
- build_state() -> AgentState
- async launch_subagent(...)
- async launch_subagents_parallel(...)
- async execute_tool_call(...)
- append_trajectory(...)
- finish(...)

要求：

RecursiveAgent.run() 只解释并执行 policy 输出的 AgentAction。  
不能把任务拆分逻辑硬编码进 run()。

## 4.2 Policy

定义抽象接口：

class Policy:
    async def act(self, state: AgentState) -> AgentAction:
        ...

实现两个 policy：

### RuleBasedPolicy

用于本地 demo，不依赖真实 LLM。  
可以把 Kyoto travel task 拆成：

- cherry blossom timing and viewing spots；
- quiet temple；
- kid-friendly activity；
- dinner near Gion；
- final synthesis。

但需要明确说明：RuleBasedPolicy 只是 demo policy，硬编码逻辑只能放在 RuleBasedPolicy 内部，不能放在 harness 内部。

### LLMPolicy

用于真实模型调用。

要求：

- 渲染 AgentState 为 prompt；
- 调用 OpenAI-compatible chat completion；
- 要求模型输出严格 JSON；
- 使用 Pydantic 解析；
- JSON 解析失败时进行一次 repair；
- repair 失败时 fallback 到 FINISH 或 DIRECT_ANSWER；
- 记录 raw_response 和 parse_error 到 trajectory 或 trace。

## 4.3 AgentAction

设计 action 类型：

- THINK
- TOOL_CALL
- DIRECT_ANSWER
- LAUNCH_SUBAGENT
- LAUNCH_SUBAGENTS_PARALLEL
- AGGREGATE
- FINISH

每种 action 的字段需要设计清楚。

示例：

LAUNCH_SUBAGENT:

{
  "type": "LAUNCH_SUBAGENT",
  "reason": "...",
  "subtask": {
    "goal": "...",
    "context": {...},
    "expected_output": "...",
    "constraints": {...}
  }
}

LAUNCH_SUBAGENTS_PARALLEL:

{
  "type": "LAUNCH_SUBAGENTS_PARALLEL",
  "reason": "...",
  "subtasks": [...]
}

FINISH:

{
  "type": "FINISH",
  "answer": "...",
  "evidence": [...],
  "limitations": [...]
}

TOOL_CALL:

{
  "type": "TOOL_CALL",
  "tool_name": "...",
  "arguments": {...}
}

## 4.4 AgentState

AgentState 至少包含：

- node_id
- parent_id
- task
- depth
- max_depth
- remaining_steps
- remaining_children
- parent_context
- expected_output
- constraints
- trajectory
- child_summaries
- available_tools
- global_tree_summary

说明 AgentState 不应该包含完整全局历史，避免上下文膨胀。  
应该只传当前节点必要信息和压缩后的 child_summaries。

## 4.5 ExecutionTree / TraceLogger

需要支持：

- register_node
- update_node_status
- append_action
- append_observation
- attach_child
- export_json
- pretty_print
- export_mermaid

每个 node trace 包含：

- node_id
- parent_id
- depth
- task
- status
- children_ids
- trajectory
- final_answer
- error
- start_time
- end_time
- metadata

# 5. Harness 运行流程

请给出清晰伪代码。

RecursiveAgent.run() 伪代码需要包括：

while not done and step < max_steps:
    state = build_state()
    action = await policy.act(state)
    append action to trajectory

    if action.type == THINK:
        append thought
    elif action.type == TOOL_CALL:
        observation = await tool_registry.call(...)
        append observation
    elif action.type == LAUNCH_SUBAGENT:
        result = await launch_subagent(...)
        append child result
    elif action.type == LAUNCH_SUBAGENTS_PARALLEL:
        results = await asyncio.gather(...)
        append child results
    elif action.type == DIRECT_ANSWER:
        store draft answer
    elif action.type == AGGREGATE:
        synthesize from child_summaries
    elif action.type == FINISH:
        final_answer = action.answer
        done = True
    else:
        handle invalid action

if not done:
    fallback finish with best available draft or child summary

return NodeResult

# 6. 递归边界与安全机制

请详细设计：

- max_depth；
- max_steps_per_node；
- max_children_per_node；
- total_node_limit；
- timeout_seconds；
- invalid_action_limit；
- max_tool_calls_per_node；
- max_parallel_children；
- cancellation handling；
- child failure handling；
- partial result fallback。

需要说明：

如果 child agent 失败，父节点不应该整体崩溃，而应该收到结构化错误结果，由父节点决定是否继续、重试、忽略或降级回答。

# 7. Prompt 设计

请在 plan.md 中写出 LLMPolicy 使用的 system prompt 草案。

要求 prompt 明确说明：

- 当前模型是递归执行树中的一个节点；
- 可以直接回答，也可以 launch subagent；
- 子 agent 使用同一个 policy；
- 递归树必须动态生成；
- 不允许固定拆分模板；
- 不要为了递归而递归；
- depth >= max_depth 时不能继续委派；
- 子任务必须更窄、更明确、可验证；
- 独立任务可以并行，依赖任务需要顺序；
- 父节点负责聚合与最终判断；
- 输出必须是严格 JSON；
- 只能输出一个 action；
- 不要输出 markdown；
- 不要输出解释性自然语言。

还需要给出 user/state prompt 模板，说明如何把 AgentState 渲染给模型。

# 8. Demo 设计

设计一个 demo：

任务：

Plan a 3-day Kyoto trip in early April for a family. We want cherry blossoms, one quiet temple, one kid-friendly activity, and dinner near Gion. Avoid overly crowded spots.

要求：

- 使用 RuleBasedPolicy；
- max_depth=2 或 3；
- 展示 root 动态拆分；
- quiet temple 子任务可以继续 spawn crowd verification；
- 最终 root 聚合结果；
- 打印 final answer；
- 打印 pretty execution tree；
- 保存 trace.json；
- 导出 Mermaid graph。

需要说明预期 execution tree 示例，但强调这只是 RuleBasedPolicy demo，不是 harness 写死结构。

# 9. 未来扩展

请给出未来扩展点：

- 接入真实搜索工具；
- 接入浏览器工具；
- 接入代码执行沙箱；
- 接入 reward evaluator；
- 记录 node-level success signal；
- 为 RAO training 生成 trajectories；
- 支持 local reward；
- 支持 root-group baseline；
- 支持 depth-level inverse-frequency weighting；
- 支持多模型 policy；
- 支持 verifier subagent；
- 支持 caching；
- 支持 distributed execution；
- 支持 human-in-the-loop。

# 10. 开发路线

请给出分阶段开发计划。

阶段一：最小可运行版本

- dataclass / Pydantic schema；
- RecursiveAgent.run；
- RuleBasedPolicy；
- ExecutionTree；
- demo script；
- trace export。

阶段二：真实 LLM 接入

- LLMPolicy；
- OpenAI-compatible client；
- JSON schema parsing；
- repair；
- fallback。

阶段三：工具系统

- ToolRegistry；
- mock search；
- calculator；
- read_text；
- optional python_exec。

阶段四：测试与鲁棒性

- pytest-asyncio；
- depth limit；
- node limit；
- timeout；
- child failure；
- invalid action。

阶段五：训练扩展准备

- node-local reward fields；
- success signal placeholder；
- trajectory export format；
- batch rollout runner。

# 11. 设计原则

最后总结设计原则：

- harness executes, policy decides；
- recursion is dynamic, not hard-coded；
- shared policy across all nodes；
- bounded recursive execution；
- structured actions over free-form text；
- trace everything；
- failure is data, not crash；
- local demo first, real LLM later；
- training hooks should be reserved but not implemented now。

请输出完整的 plan.md 内容，使用 Markdown 格式。
不要生成代码。