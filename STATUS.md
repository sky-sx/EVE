# EVE 当前实现状态

更新时间：2026-08-02。

本轮只收拢架构与错误语义，不扩展新能力。正式运行时保持
`main + input + output + memory + core + dock` 六个边界。

## 已实现边界

- Capture 仅由 Buffer 启停和读取；Core 不直接访问 Capture。
- 控制模式冷启动时 YOLO 默认启用并优先取得模型加载顺序；其当前识别质量
  仍是已记录的后续问题。
- LLM-based self update loop 冷启动后进入自主更新，用户提问与自主触发共用
  EVE 第一版固定提示词和唯一 JSON 输出格式；尝试数、成功数、失败数、schema 失败、修复次数、延迟和下一
  次触发时间均可见。
- 当前 YOLO/TNN 视觉事实会确定性投影到 code-owned `world.perception.visual`，
  并随用户对话上下文送入 LLM；LLM 不能覆盖 perception。
- GUI 高频文本窗口只在内容变化时更新并保留滚动位置；所有运行事件、
  LLM/self、YOLO、TNN、Output 结果及每秒循环快照写入 `debug.jsonl`。
- 资源页显示带状态、实际频率、耗时、队列深度和触发条件的运行循环图。
- `training_proposal` 与 `TrainingOrder` 仍严格分离；第一版正常运行入口只保存
  proposal，不自动进入 materialization。未来教师流程需要显式启用，才可生成
  具体 Actor 源码、可选 QNN 源码和完整订单。
  Core/Dock 只接受通过 workspace、AST、依赖、危险调用和 TinyNN 接口检查的
  材料；Dock 不包含 JSON 模型编译器，也不替 LLM 设计网络。
- 正式 TNN 只有通过验收后才保存到 Memory 并由 Core 加载，最多同时加载
  5 个。
- 各 TNN 由共享执行池按各自到期时间提交，同一节点不会重入；状态记录目标
  频率、实际频率、最近与平均耗时、超期、跳过和运行中标志。
- 动作候选通过 Core 的权限、急停、有效期和接管检查后进入有界 Output
  队列；Output worker 执行前再次检查。急停清空未执行动作。
- 鼠标长距离移动与拖拽拆成可中断短步。
- 已执行 Output 反馈同时返回 `candidate_id` 与 `action_id`；被阻断候选不再
  伪造 `action_id`。可信外部事件仍要求与候选、动作和时间精确绑定；无外部
  接口的桌面任务使用独立 `self_observed_environment` 证据路径。
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
- EVE 第一版认知输出只有一套完整字段；`goodness_records` 必须存在，无具体评价时使用空数组。旧版本号字段、缺字段 backend 和迁移别名不接受。`goodness_update` 只表示当前 `self.goodness` 总体摘要。
- Experience 第一版只接受 v2：环境的 `hit`、外部 `reward`、分数、延迟等只作为事实，不再自动把命中换算为 `reward = ±1`；v1 不读取、不迁移。
- 冻结帧 VLM 结果只产生带原始帧 ID、时间和证据 MemoryID 的事实，不独立决定好度、动作或训练验收。
- Dock 支持 ValueDefinition 的最小安全数值表达式，以及只在 `dock/workspace/<order_id>/` 内存在的临时 QNN。QNN 仅评价已有候选、产生 Actor top-k 样本，随后删除；不会注册为正式 TNN，也不会被 Core 加载。
- 训练验收只接受 `min_goodness` 和 `min_regression_goodness`，支持 `mean`/`minimum` 聚合；loss 等指标仅作诊断事实。权限、急停、接口和 artifact 完整性仍是不可由高好度绕过的硬边界。
- 本轮验证是合成数据和单元/集成测试证据，不代表真实桌面成长实验、任意连续动作搜索、通用视觉理解或长期价值观已经完成。

## 2026-08-02：自主行为与自主成长纵向闭环

### 代码已存在

- 第一版固定输出包含 `prompt_request` 与 `action_candidates`，不再包含版本号、
  `tool_requests`、`training_materialization` 或 `observation_completion`。
- Mouse / Keyboard / Speak 候选使用统一外层，具体字段由按需器官提示词说明；经 Core 原权限、暂停、
  急停、接管、时效与队列检查后，仍由 Output 执行前复检。
- LLM 可按需请求单个冻结帧 VLM；请求绑定 request/frame/time/screen/result，
  完成或失败均为结构化结果，LLM 工具结果回到同一 self-update 队列。VLM
  没有独立自治循环。
- 已执行或 mock 动作由一个共享异步 observation worker 形成 before/after、
  after-VLM 与 `ObservationBundle v1`；随后同一 LLM 可提交可见事实、
  GoodnessRecord 和 Experience v2。该路径不伪造外部 environment event。
- 第一版固定输出只保存 `training_proposal` 建议；旧的认知输出 materialization
  阶段已从正常运行入口停用，Dock 教师接口仍保留供未来按需调用。
- 现有 GUI 的 self 页面只显示纵向闭环摘要；完整结构化输出继续进入
  `debug.jsonl`。循环图保留 Qwen Text-only→Vision→Text-only、Qwen→Output、
  Output→Observation→Qwen 连接。

### 测试证据

- 本次修改前本地工作树干净，HEAD 为 `cf45437f085f60aa92fe394e92e76f8fb5fcbf4d`。
- 本次修改后：不触碰受沙箱保护的默认持久化目录时，
  `73 passed, 4 deselected in 7.66s`；Memory 与第一版运行核心定向测试
  `15 passed in 1.48s`。此前获批写入默认目录的全套运行达到
  `76 passed, 1 failed`，唯一失败是新增权限拒绝测试未清理独立
  `eve-output` 夹具线程，现已补齐清理。
- 新增第一版固定输出、器官提示词、单一 Qwen、纯文本无 `pixel_values`、
  Memory 视图持久化及旧目录哈希迁移测试。
- 1 秒 smoke 使用合成 Capture 与 mock Output；`real_output_calls=0`、
  `threads_stopped=true`、`capture_process_stopped=true`。

### 尚无证据

- 尚未进行真实桌面动作、真实模型自主代码生成、真实冻结帧 VLM 质量验收、
  真实黑箱小游戏成长、真实 TNN 性能改善或长时间稳定性验收。
- 因此当前只能称为“代码闭环 + 单元/集成 + 合成纵向闭环”，不能称为真实
  桌面自主成长完成。
