# EVE 当前实现状态

更新时间：2026-07-31。

本轮只收拢架构与错误语义，不扩展新能力。正式运行时保持
`main + input + output + memory + core + dock` 六个边界。

## 已实现边界

- Capture 仅由 Buffer 启停和读取；Core 不直接访问 Capture。
- 控制模式冷启动时 YOLO 默认启用并优先取得模型加载顺序；其当前识别质量
  仍是已记录的后续问题。
- LLM-based self update loop 冷启动后进入自主更新，用户提问与自主触发共用
  protocol v2；尝试数、成功数、失败数、schema 失败、修复次数、延迟和下一
  次触发时间均可见。
- 当前 YOLO/TNN 视觉事实会确定性投影到 code-owned `world.perception.visual`，
  并随用户对话上下文送入 LLM；LLM 不能覆盖 perception。
- GUI 高频文本窗口只在内容变化时更新并保留滚动位置；所有运行事件、
  LLM/self、YOLO、TNN、Output 结果及每秒循环快照写入 `debug.jsonl`。
- 资源页显示带状态、实际频率、耗时、队列深度和触发条件的运行循环图。
- Core 仅接收带具体 `model.py`/模型 MemoryID、数据和验收条件的可执行
  `TrainingOrder`；Dock 不包含 JSON 模型编译器。LLM training proposal 只记录，
  不自动执行。
- 正式 TNN 只有通过验收后才保存到 Memory 并由 Core 加载，最多同时加载
  5 个。
- 各 TNN 由共享执行池按各自到期时间提交，同一节点不会重入；状态记录目标
  频率、实际频率、最近与平均耗时、超期、跳过和运行中标志。
- 动作候选通过 Core 的权限、急停、有效期和接管检查后进入有界 Output
  队列；Output worker 执行前再次检查。急停清空未执行动作。
- 鼠标长距离移动与拖拽拆成可中断短步。
- Output 反馈同时返回 `candidate_id` 与 `action_id`；正向 Experience 只接受
  与候选、动作、执行时间和环境事件精确绑定的反馈。
- Memory 使用完整 Catalog 与独立 STM/MTM/LTM 视图；STM 溢出只驱逐热视图，
  MTM/LTM 只通过显式操作变化，不存在自动晋升线程。
- 正常停机生成 Snapshot v2、`world.md` 与 `self.md`；只持久化耐久语义。

## 明确未实现

- 麦克风听觉与注意力/焦点机制。
- 真正的睡眠摘要、合并、索引更新与遗忘。
- VLM 视觉解释质量与 YOLO 类别覆盖仍需后续专项改进。
- 任意连续动作空间的候选生成与通用 QNN 扩展。

以上未实现项不得在界面、文档或验收结果中表述为已完成。

## 2026-08-01：统一好度与训练闭环第一版

- `GoodnessRecord` 与 `ValueDefinition` 已作为独立、不可变 MemoryUnit 保存；重估会新增记录，并通过 Event 关联 target、证据和价值定义。
- protocol v2 保持版本号不变，新增可选 `goodness_records`；旧 backend 不返回该字段仍兼容。`goodness_update` 只表示当前 `self.goodness` 总体摘要。
- 新 Experience 写入使用 v2：环境的 `hit`、外部 `reward`、分数、延迟等只作为事实，不再自动把命中换算为 `reward = ±1`；读取仍兼容 v1。
- 冻结帧 VLM 结果只产生带原始帧 ID、时间和证据 MemoryID 的事实，不独立决定好度、动作或训练验收。
- Dock 支持 ValueDefinition 的最小安全数值表达式，以及只在 `dock/workspace/<order_id>/` 内存在的临时 QNN。QNN 仅评价已有候选、产生 Actor top-k 样本，随后删除；不会注册为正式 TNN，也不会被 Core 加载。
- 新式验收以 `evaluation.goodness` 和 `regression.goodness` 为正式比较值，支持 `mean`/`minimum` 聚合；旧 `max_loss` 等技术验收仍兼容。权限、急停、接口和 artifact 完整性仍是不可由高好度绕过的硬边界。
- 本轮验证是合成数据和单元/集成测试证据，不代表真实桌面成长实验、任意连续动作搜索、通用视觉理解或长期价值观已经完成。
