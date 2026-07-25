# EVE 本地修改报告

生成时间：2026-07-25

## 用户原有修改

| 文件 | 状态 |
|------|------|
| .gitignore | 已修改（用户原有） |

## AI 本次修改

### 新建文件（核心模块）

| 文件 | 职责 | 对应架构 |
|------|------|----------|
| eve/config.py | 配置管理（路径、频率、激素基线、循环参数） | main |
| eve/core/graph.py | 动态 TNNGraph（节点/边/调度/缓存/轨迹/集合差分） | core |
| eve/core/hormones.py | 六大激素系统（更新/节律/倾向推导） | core |
| eve/core/model_adapter.py | LLM/VLM/YOLO 模型适配器（检测/加载/卸载/推理） | core |
| eve/core/prompts.py | 中文 LLM 多角色 Prompt（world/myself/active_tnn/训练/复盘等） | core |
| eve/core/runtime_state.py | 运行时状态管理器（world/myself/blackboard 持有与快照） | core |
| eve/core/sleep.py | 睡眠复盘管理器（收集/去重/整理/训练/恢复） | core |
| eve/core/tnn_base.py | TNN 基类（TNNDescriptor/SourceRef/DummyTNN/ConvTNN） | core |
| eve/core/tnn_store.py | TNN 存储管理器（注册/加载/卸载/版本/扫描） | core |
| eve/memory/__init__.py | Memory 包初始化 | memory |
| eve/memory/catalog.py | Catalog（MemoryID→LTM 对象映射） | memory |
| eve/memory/memorizer.py | Memorizer（创建/读取/删除/STM/MTM/LTM 管理） | memory |
| eve/memory/indexes.py | 多图索引管理（边/稠密折叠/lazy redirect） | memory |
| eve/memory/event.py | Event 管理器（事件分组） | memory |
| eve/memory/retrieval.py | 记忆检索（关键词/时间范围/类型过滤） | memory |
| eve/dock/__init__.py | Dock 包初始化 | dock |
| eve/dock/order.py | 训练订单与结果数据结构 | dock |
| eve/dock/trainer.py | Dock 训练机床（订单队列/teacher/训练/评估/保存） | dock |
| eve/dock/tiny_nn.py | TNN 构建工具（从订单创建网络） | dock |
| eve/gui/__init__.py | GUI 包初始化 | main |
| eve/gui/control_panel.py | tkinter 控制面板（状态显示/控制按钮/激素/TNN/日志） | main |

### 修改的文件

| 文件 | 修改内容 |
|------|----------|
| eve/main.py | **重写** — 从 Phase 1 Demo 改为正式入口（12 步初始化 + GUI） |
| eve/state.py | **扩展** — 新增 OutputMode.REAL、WorldState、MyselfState、Blackboard、TimedEntry；RuntimeState 新增 human_activity_detected_at_ns |
| eve/core/loop.py | **重写** — 新增 EVELoops 多循环架构（8 条循环），保留 run_once/log_event |
| eve/core/safegate.py | **增强** — 新增 detect_human_cursor_activity()、detect_human_keyboard_activity() |
| eve/core/__init__.py | 更新描述 |
| eve/input/capture.py | **增强** — 新增 human_activity_callback、was_human_cursor_recent() |
| eve/output/mouse.py | **增强** — 新增 REAL 模式（pyautogui 真实鼠标控制） |
| eve/output/keyboard.py | **增强** — 新增 REAL 模式（pyautogui 真实键盘控制） |
| eve/output/speak.py | **增强** — 新增 REAL 模式接口（标记未实现） |

### 新建文件（架构文档）

| 文件 | 内容 |
|------|------|
| EVE_ARCHITECTURE_UNDERSTANDING.md | 架构理解文档（15 个关键问题回答） |
| ARCHITECTURE_TO_CODE_MAP.md | 架构到代码映射表 |
| ARCHITECTURE_CONFORMANCE.md | 架构一致性检查表 |

### 新建文件（测试）

| 文件 | 测试数 | 覆盖范围 |
|------|--------|----------|
| tests/__init__.py | - | 包初始化 |
| tests/test_safegate.py | 23 | Safegate 9 条规则全覆盖 |
| tests/test_runtime_loop.py | 15 | run_once/EVELoops/日志 |
| tests/test_input_buffer.py | 16 | 存储/范围/快照/淘汰/线程安全 |
| tests/test_capture_timing.py | 14 | 捕获启停/统计/时间戳 |
| tests/test_forbidden_architecture.py | 10 | 架构禁止项扫描 |
| tests/test_runtime_state.py | 18 | Blackboard/快照/LLM 更新 |
| tests/test_memory.py | 16 | Memorizer/Catalog/索引/Event/检索 |
| tests/test_tnn_store.py | 11 | TNN 保存加载/Store 管理 |
| tests/test_graph.py | 11 | 缓存/节点/边/调度/轨迹 |
| tests/test_hormones.py | 12 | 6 激素更新/节律/倾向 |
| tests/test_prompts.py | 12 | Prompt 结构/校验/JSON 解析 |
| tests/test_dock.py | 8 | 训练订单/队列/训练流程 |
| tests/test_integration.py | 7 | 端到端集成链路 |

### 删除的文件

| 文件 | 原因 |
|------|------|
| _smoke_test_new_modules.py | 引用已废弃的 src/ 代码，无法运行 |

## 总计

- 新建核心模块文件：21 个
- 修改现有文件：8 个
- 新建架构文档：3 个
- 新建测试文件：14 个（173 个测试）
- 删除文件：13 个（临时文件 + 废弃代码）
