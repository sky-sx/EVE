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
LLM-based self update loop 首轮立即调度；一次更新完成后按激素状态保留
1–3 秒空闲，再进入下一轮。模型推理时间较长时，它因此基本保持连续工作。
快速状态进入 Buffer；Core 只从 Buffer 读取。Blackboard 是带时间与有效期的
运行交换区，不是长期 Memory。

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

每个 Catalog 中的 MemoryID 必须恰好位于 STM、MTM 或 LTM 之一：

```text
create -> STM
STM 容量溢出或显式晋升 -> MTM
再次晋升 -> LTM
```

本轮的 `force_promotion()` 只做层级晋升，不冒充摘要、合并、索引更新或
遗忘。真正的睡眠整理后续独立设计。正常停机生成可读的 `world.md` 与
`self.md`。

## 7. Dock

Dock 只接受两种通用 TNN 来源：

1. `TrainingOrder.definition` 中的 JSON 结构，使用允许的通用算子生成模型。
2. TrainingOrder 显式指向具体 `model.py`；特殊模型自行实现 forward、
   training_step 和特殊梯度语义。

Dock 不按任务名选择模型。`teacher_mode` 与 `teacher_prompt` 只有在真正连接
到标签生成流程时才可声明；当前不能实现的模式必须明确失败。候选通过验收
后才写入正式 TNNweights 和 Memory TNN 列表，并由 Core 加载。

QNN 若未来用于扩展训练数据，只能在 `dock/workspace/<order_id>/` 中临时
创建、估值并删除；它不写入正式 TNNweights、不注册 Memory TNN、不由
Core 加载，也不进入运行图。

## 8. 明确不采用

- 固定任务 Pipeline 或按任务名选择网络。
- 运行期 QNN 候选评分层。
- 可绕过权限或急停的 Output 路径。
- 独立大型调度器、Graph Manager 或每 TNN 永久线程。
- 麦克风或注意力/焦点的占位伪实现。
- 前台窗口标题、进程名等 OS 语义捷径。
