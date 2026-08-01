# EVE 当前架构

## 1. 边界

正式运行时保持六个边界：

```text
main + input + output + memory + core + dock
```

- Main 只负责装配、GUI 和生命周期。
- Input 的 Capture 只捕获，Buffer 是唯一启停和读取 Capture 的边界。
- Core 维护 world、self、Blackboard、TNN 生命周期和动作边界。
- Output 只执行结构化动作，不选择目标或任务。
- Memory 保存不可变 payload、Event、层级状态和正式 TNN artifact，不运行
  TNN。
- Dock 只按 TrainingOrder 在临时 workspace 中训练，不执行真实动作。

正式感知不得使用前台窗口标题或进程名作为语义捷径。

## 2. 输入与状态

屏幕、光标、键盘活动、文本输入和本地 LLM 小/慢循环在冷启动后自动运行。
LLM-based self update loop 首轮立即调度，用户提问与自主更新共用 protocol
v2；默认约每 2 秒触发一轮，模型推理期间不会重入。用户触发时必须返回
可见 reply；可选状态增量不合法时不得丢失有效 reply。
快速状态进入 Buffer；Core 只从 Buffer 读取。Blackboard 是带时间与有效期的
运行交换区，不是长期 Memory。

`world.perception` 只由代码写入，保存当前屏幕帧的视觉类别、置信度、边界框、
中心点和状态；LLM 只能更新 `interpretation`、`uncertainty` 与 `task_state`。
`self.goodness` 是范围为 [-1, 1] 的评价标量及其证据说明；倾向只是数值，
不代替权限。用户表扬/批评作为显式反馈记忆保存，不直接改写 goodness。

用户界面使用 `self`。内部 `myself` 暂时只作为兼容键存在。LLM-based self
update loop 可写简短可见摘要，但不得请求、展示或持久化隐藏思维链。
调试输出写入运行目录的 `debug.jsonl`；其中记录可见/结构化结果和循环状态，
不记录隐藏思维链。GUI 资源页从 Core 的 `loop_graph` 状态显示循环连接、
实际频率、耗时、队列深度和触发条件。

## 3. TNN 运行调度

Core 保留一个约 50 Hz 主循环。每个 TNN 记录独立目标频率和下一到期时间：

```text
Core 主循环
|- 收集已完成 Future，并原子发布输出
|- 判断各 TNN 是否到期
|- 到期且上一轮完成 -> 提交到共享执行池
`- 到期但仍运行 -> 记录跳过，不重复提交
```

不为每个 TNN 创建永久线程，也不新增调度子系统。每个节点公开目标频率、
实际频率、最近耗时、平均耗时、超期、跳过、跳过次数和运行中状态。正式
TNN 最多同时加载 5 个。

## 4. 动作与急停

动作链为：

```text
TNN actor 输出候选
-> Core 检查候选格式与时效
-> Core 检查权限、急停和用户接管
-> 有界 Output 队列
-> Output worker 执行前复检
-> mouse / keyboard / speak
```

急停立即清空候选与未执行 Output 队列。长距离鼠标移动和拖拽拆成可中断
短步。Output 反馈始终带回对应 `candidate_id` 和 `action_id`。

权限和急停语义不可绕过，但不形成独立模块；必要纯函数位于 Core 动作出队
边界。

## 5. Experience 反馈

形成正向训练 Experience 的反馈必须显式携带：

```python
{
    "candidate_id": str,
    "action_id": str,
    "executed_at_ns": int,
    "environment_event_id": str,
}
```

Core 同时验证候选存在且未消费、动作 ID、动作类型、坐标、执行时间、候选
有效期和对应环境 Event。无法精确绑定的输入只能作为普通环境事件保存。

## 6. Memory 生命周期

Catalog 是完整不可变记录，STM、MTM、LTM 是彼此独立的语义视图：

```text
create -> Catalog + STM hot view
load_to_mtm / unload_from_mtm -> 显式工作集
persist_to_ltm / remove_from_ltm -> 显式持久语义集
STM overflow -> 仅从 hot view 驱逐，Catalog 记录仍保留
```

不存在后台自动晋升线程或 `force_promotion()`。LLM 可通过显式 memory_actions
请求调整视图。正常停机生成 Snapshot v2 及可读的 `world.md`、`self.md`；
运行期 perception、Blackboard、资源状态和错误不进入耐久快照。

## 7. Dock

Dock 只接受 TrainingOrder 显式指向的具体 `model.py` 或已有模型 MemoryID；
具体模型自行实现 forward、training_step 和特殊梯度语义。LLM 的
`training_proposal` 只是建议，不能直接作为可执行 TrainingOrder。

Dock 不按任务名选择模型，也不包含 JSON 算子编译器或模型源码生成器。
订单必须携带训练数据与明确验收条件；候选通过验收后才写入正式 TNNweights
和 Memory TNN 列表，并由 Core 加载。

QNN 用于扩展训练数据时，只能在 `dock/workspace/<order_id>/` 中临时
创建、估值并删除；它不写入正式 TNNweights、不注册 Memory TNN、不由
Core 加载，也不进入运行图。

## 8. 明确不采用

- 固定任务 Pipeline 或按任务名选择网络。
- 运行期 QNN 候选评分层。
- 可绕过权限或急停的 Output 路径。
- 独立大型调度器、Graph Manager 或每 TNN 永久线程。
- 麦克风或注意力/焦点的占位伪实现。
- 前台窗口标题、进程名等 OS 语义捷径。

## 9. 统一好度数据流

```text
环境/VLM冻结帧/人工事实
  -> ValueDefinition（价值版本、输入、标尺、约束）
  -> 教师直接评价或安全数值函数
  -> 独立 GoodnessRecord
  -> 可选的 Dock 临时 QNN 对已有候选排序
  -> Actor top-k 训练样本
  -> Actor evaluation.goodness / regression.goodness
  -> 硬边界通过后才保存正式 Actor artifact
```

事实、损失、命中、reward、延迟和安全事件都不是好度本身。GoodnessRecord 的分数在 `[-1,1]`，并携带 target、value version、方法、事实、原因、置信度与真实证据 MemoryID。`self.goodness` 只保存当前总体摘要及最近记录指针，不复制全部历史记录。

Experience v2 把环境信息保存为 `GoodnessFact[]`，并用 `goodness_memory_ids` 或 Event 关联后续评价。历史 v1 payload 保持只读兼容。blocked candidate 可以形成 `awaiting_goodness` Experience，但因为没有执行，不能接受正向环境执行反馈。

安全数值函数只在 Dock 内解析白名单 AST：数字、声明变量、四则运算、比较、条件表达式和 `min/max/abs/clip`。属性访问、任意函数、import、文件/网络操作、循环与推导均不允许；缺少必需事实或出现 NaN/Inf 时等待教师评价，不产生默认好度。

临时 QNN 仍是作者提供的 concrete `model.py`/TinyNN。其权重和临时数据只进入订单 workspace；报告保留结构、来源、误差、排序一致性、价值版本、教师来源与清理状态。QNN 不进入 Memory/TNNweights、Core、运行图或输出控制链。
