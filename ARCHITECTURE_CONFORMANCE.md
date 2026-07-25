# Architecture Conformance Checklist

## 持续检查项

每完成一个大模块后更新以下检查。

### 当前状态（Phase 14 完成）

| 模块 | 状态 |
|------|------|
| main | 已完整实现 |
| input (capture) | 已完整实现 |
| input (buffer) | 已完整实现 |
| output (mouse) | 已完整实现 |
| output (keyboard) | 已完整实现 |
| output (speak) | 部分实现（TTS 后端未接入） |
| memory (memorizer/catalog/indexes/event/retrieval) | 已完整实现 |
| core (loop) | 已完整实现（8 循环并发架构） |
| core (safegate) | 已完整实现（9 规则） |
| core (TNN base/store) | 已完整实现 |
| core (graph) | 已完整实现（动态有向图/异频/缓存/集合差分） |
| core (hormones) | 已完整实现（6 激素/12 事件/节律） |
| core (sleep) | 已完整实现 |
| core (model_adapter) | 已实现但未在真实模型验证 |
| core (prompts) | 已完整实现（8 中文 Prompt） |
| dock (trainer/order/tiny_nn) | 已完整实现 |
| GUI | 已完整实现（tkinter 5 页签） |
| 测试 | 已完整实现（173 tests, 14 文件） |

## 架构防漂移检查

- [x] 我是否把 EVE 写成了传统 Agent？→ 否，无 Router/Planner/Orchestrator
- [x] 我是否添加了固定 Router 或 Planner？→ 否
- [x] 我是否预定义了本应训练产生的能力？→ 否，无具体任务 TNN 预定义
- [x] 我是否把 MemoryUnit 塞入过多解释字段？→ 否，只有 memory_id + payload + payload_type + timestamp
- [x] 我是否把 STM/MTM/LTM 做成三份数据？→ 否，STM/MTM 是 ID 列表，LTM 目录管理，Payload 不复制
- [x] 我是否把 Markdown 当运行时数据库？→ 否，Markdown 仅用于快照保存/恢复
- [x] 我是否让 Dock 负责运行时调度？→ 否，Dock 只负责训练，Core 负责运行时
- [x] 我是否让 Core 负责训练？→ 否，训练经 Dock 完成
- [x] 我是否把 TNNGraph 写成固定 Pipeline？→ 否，动态图支持加载/卸载/边变更/异频
- [x] 我是否用规则类冒充 TNN？→ 否，TNN 有独立参数/forward/训练/保存/加载
- [x] 我是否建立了没有真实需求的 Manager/Registry？→ 否
- [x] 我是否保留了无价值兼容层？→ 否，删除了 src/ 旧引用
- [x] 我是否为了测试建立旁路？→ 否，所有输出经 Safegate
- [x] 我是否创建了没有消费者的字段？→ 否
- [x] 我是否留下了空壳代码？→ 否

## 模块审计模板

### main
- 新增了什么？→ main.py 12 步初始化 + GUI 绑定 + 冷启动/急停/保存
- 为什么必要？→ 系统生命周期入口
- 属于哪个正式模块？→ main
- 谁写入？→ 用户启动
- 谁读取？→ 所有模块通过 state/runtime_mgr 访问
- 何时创建？→ 程序启动时；何时更新？→ 每循环；何时销毁？→ 程序退出时
- 未预设本应由训练产生的能力
- 未引入传统 Router/Planner/多 Agent
- 无重复职责
- 无无消费者的字段
- 无空壳类
- 未偏离最新设计资料

### input (capture + buffer)
- 新增了什么？→ screen_capture + cursor_capture + capture.py + buffer.py + schemas.py
- 为什么必要？→ 系统感知外部世界的基础
- 属于哪个正式模块？→ input
- 谁写入？→ 捕获线程；谁读取？→ TNNGraph/LLM/Memory
- 生命周期与进程绑定
- 未预设本应由训练产生的能力
- 未引入传统 Agent 结构
- 无重复职责

### output (mouse/keyboard/speak)
- 新增了什么？→ disabled/mock/real 三模式输出
- 为什么必要？→ 系统作用于外部世界的基础
- 属于哪个正式模块？→ output
- 谁写入？→ Safegate 批准后；谁读取？→ 外部世界（用户桌面）
- 生命周期与进程绑定
- 未预设本应由训练产生的能力

### memory (memorizer/catalog/indexes/event/retrieval)
- 新增了什么？→ MemoryUnit 最小定义、Catalog、多图索引、Event、Retrieval、lazy redirect、图折叠
- 为什么必要？→ 系统经验积累的核心
- 属于哪个正式模块？→ memory
- 谁写入？→ memorizer；谁读取？→ Dock/LLM/TNNGraph
- 持久化：关闭时保存，启动时恢复
- MemoryUnit = MemoryID + Payload，无额外解释字段
- STM/MTM 是 ID 集合，不复制 Payload
- 未预设本应由训练产生的能力

### core (loop/graph/hormones/safegate/sleep/tnn_base/tnn_store/model_adapter/prompts/runtime_state)
- 新增了什么？→ 8 循环并发架构、动态 TNNGraph、6 激素系统、9 规则 Safegate、睡眠复盘、TNN 基类/Store、模型适配器、8 中文 Prompt、运行时状态
- 为什么必要？→ 系统运行核心
- 属于哪个正式模块？→ core
- Core 不负责训练（训练归 Dock）
- TNNGraph 是动态图，非固定 Pipeline
- 激素不直接产生动作
- Safegate 不可绕过
- 所有输出路径必经 Safegate

### dock (trainer/order/tiny_nn)
- 新增了什么？→ 训练订单队列、数据组装、多类型 teacher、训练评估、TNN 保存
- 为什么必要？→ 能力生长的机床
- 属于哪个正式模块？→ dock
- Dock 不负责运行时调度
- Dock 不直接执行鼠标键盘
- QNN 仅训练期存在，不进入运行图

### GUI
- 新增了什么？→ tkinter 5 页签控制面板（显示/控制/记忆/训练/日志）
- 为什么必要？→ 观察和控制实验
- 界面朴素但功能完整，与运行时状态真实连接
