# EVE 当前架构理解

EVE 是个人数字生命研究装置，不是通用 Agent 框架。当前代码只实现一条安全、
可观察、可停止的最小运行骨架。

## 当前因果链

```text
Capture / synthetic reader
→ monotonic InputBuffer
→ loaded TNN reads descriptor SourceRef
→ named output
→ optional ActionCandidate consume-once queue
→ Safegate
→ disabled/mock OutputResult
→ JSONL + immutable Memory payload
```

TNN 关系不是独立对象。`tnn:<id>.<field>` 已经表达下游读取哪个上游命名输出；
当前只保存 loaded 节点、active ID、最近运行时间和命名输出。

`RuntimeState` 是 world、myself、blackboard、动作队列、安全控制和 TNN 运行数据的
唯一内存权威。Blackboard 状态是 latest-value；动作是独立 queue/consume-once。

Memory 只保留一个 Catalog 和一份 payload。STM/MTM 只保存 MemoryID；Event 只组织
MemoryID；当前检索只包含时间、类型和关键词。

六激素只是 `MyselfState` 内的六个连续实验值，不直接产生动作。Sleep、Dock、GUI、
本地模型适配与复杂 Memory Graph 当前暂停。

正式 CLI 禁止 real 模式。本轮 smoke 节点是明确标记的规则占位，不是训练所得 TNN；
输入是合成 reader，输出是 Mock，因此不构成真实任务或成长证明。
