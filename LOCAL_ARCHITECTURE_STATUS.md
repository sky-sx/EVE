# EVE 本地架构状态

生成时间：2026-07-25

## 模块完成状态

| 模块 | 状态 | 说明 |
|------|------|------|
| main 入口 | 已完整实现 | 12 步初始化 + GUI 绑定 + 冷启动/急停/保存 |
| input (capture) | 已完整实现 | mss 屏幕 + pyautogui 光标 + 人类活动检测 |
| input (buffer) | 已完整实现 | 线程安全/多类型/latest/range/snapshot/淘汰 |
| output (mouse) | 已完整实现 | disabled/mock/real 三模式 |
| output (keyboard) | 已完整实现 | disabled/mock/real 三模式 |
| output (speak) | 部分实现 | disabled/mock 可用，real 接口就绪但 TTS 后端未接入 |
| Safegate | 已完整实现 | 9 条规则全覆盖 + 人类活动检测 |
| world | 已完整实现 | 场景/对象/文本/不确定性/快照 |
| myself | 已完整实现 | 任务/思维/激素/TNN/倾向/快照 |
| blackboard | 已完整实现 | 写入/读取/TTL/过期/快照 |
| Memory (memorizer) | 已完整实现 | 创建/读取/删除/STM/MTM/LTM |
| Memory (catalog) | 已完整实现 | 注册/查找/按类列出/持久化 |
| Memory (indexes) | 已完整实现 | 多图边/时间链/稠密折叠/lazy redirect |
| Memory (event) | 已完整实现 | 事件创建/分组/持久化 |
| Memory (retrieval) | 已完整实现 | 关键词/时间范围/类型过滤 |
| TNN 基类 | 已完整实现 | descriptor/forward/save/load/DummyTNN/ConvTNN |
| TNN Store | 已完整实现 | 注册/加载/卸载/版本/回滚/扫描 |
| TNNGraph | 已完整实现 | 动态图/节点/边/异频调度/缓存/轨迹/集合差分 |
| 模型适配 | 已实现但未在真实硬件验证 | LLM/VLM/YOLO 适配器结构完整，本地无模型文件 |
| Prompts | 已完整实现 | 8 个中文 Prompt + 解析/校验/错误处理 |
| 激素系统 | 已完整实现 | 6 激素/12 事件类型/节律计算/倾向推导 |
| 节律控制 | 已完整实现 | LLM 间隔 10-20s 动态映射 |
| Dock 训练 | 已完整实现 | 订单/队列/teacher/训练/评估/保存 |
| 睡眠复盘 | 已完整实现 | 收集/去重/LLM 复盘/索引整理/技能发现/训练 |
| GUI | 已完整实现 | tkinter 5 页签控制面板 |
| 日志 | 已完整实现 | JSONL 结构化日志 |
| 状态持久化 | 已完整实现 | Markdown 快照保存/恢复 |

## 关键架构验证

- [x] 没有 Router/Planner/Orchestrator/Agent 命名
- [x] MemoryUnit = MemoryID + Payload（无额外字段）
- [x] STM/MTM 是 ID 列表，不复制 Payload
- [x] TNN 使用独立 TNN ID 空间
- [x] Dock 只负责训练，不负责运行时调度
- [x] Core 只负责运行，不负责训练
- [x] TNNGraph 是动态图，非固定 Pipeline
- [x] 所有输出路径经过 Safegate
- [x] 激素不直接产生动作
- [x] 没有预定义具体任务 TNN（如 RedCircleTNN）
- [x] 没有无消费者的空字段
- [x] 没有空壳类或 pass-only 实现

## 已删除的旧代码

| 旧代码 | 状态 |
|--------|------|
| _smoke_test_new_modules.py（src/ 引用） | 已删除 |
| tests/ 旧 .pyc 缓存 | 已在 .gitignore 中 |
| main_legacy_demo.py（YOLO 演示） | 保留但不再作为入口 |
