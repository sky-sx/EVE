# Architecture to Code Map

## main
| 职责 | 现有文件 | 计划 | 生命周期 |
|------|----------|------|----------|
| 主入口/启动/停止 | eve/main.py | 重写为正式入口 | 进程级 |
| 配置管理 | (新) eve/config.py | 新建 | 进程级 |
| 结构化日志 | (使用) eve/core/loop.py log_event | 抽离为独立模块 | 进程级 |
| GUI 控制面板 | (新) eve/gui/control_panel.py | 新建 | 进程级 |

## input
| 职责 | 现有文件 | 计划 | 生命周周期 |
|------|----------|------|----------|
| 屏幕捕获 | eve/input/capture.py | 保留+增强(人类活动检测) | 线程级 |
| 光标捕获 | eve/input/capture.py | 保留 | 线程级 |
| 输入缓冲 | eve/input/buffer.py | 保留 | 进程级 |
| 数据结构 | eve/input/schemas.py | 保留+扩展 | - |
| 音频捕获 | (新) eve/input/audio_capture.py | 新建 | 线程级 |

## output
| 职责 | 现有文件 | 计划 | 生命周期 |
|------|----------|------|----------|
| 鼠标输出 | eve/output/mouse.py | 保留+加real模式接口 | 进程级 |
| 键盘输出 | eve/output/keyboard.py | 保留+加real模式接口 | 进程级 |
| 语音输出 | eve/output/speak.py | 保留+加real模式接口 | 进程级 |

## core
| 职责 | 现有文件 | 计划 | 生命周期 |
|------|----------|------|----------|
| Safegate | eve/core/safegate.py | 保留增强 | 进程级 |
| 主循环 | eve/core/loop.py | 重写为多循环架构 | 进程级 |
| 运行时状态 | (更新) eve/state.py | 扩展world/myself/blackboard | 进程级 |
| TNNGraph | (新) eve/core/graph.py | 新建 | 进程级 |
| TNN Store | (新) eve/core/tnn_store.py | 新建 | 进程级 |
| TNN 基类 | (新) eve/core/tnn_base.py | 新建 | 进程级 |
| 激素系统 | (新) eve/core/hormones.py | 新建 | 进程级 |
| 节律控制 | (新) eve/core/rhythm.py | 新建 | 进程级 |
| 睡眠复盘 | (新) eve/core/sleep.py | 新建 | 进程级 |
| 模型适配 | (新) eve/core/model_adapter.py | 新建 | 进程级 |
| Prompt 管理 | (新) eve/core/prompts.py | 新建 | 进程级 |

## memory
| 职责 | 现有文件 | 计划 | 生命周期 |
|------|----------|------|----------|
| Memorizer | (新) eve/memory/memorizer.py | 新建 | 进程级 |
| Catalog | (新) eve/memory/catalog.py | 新建 | 持久+内存 |
| 索引管理 | (新) eve/memory/indexes.py | 新建 | 持久+内存 |
| Event | (新) eve/memory/event.py | 新建 | 持久 |
| 检索 | (新) eve/memory/retrieval.py | 新建 | 进程级 |

## dock
| 职责 | 现有文件 | 计划 | 生命周期 |
|------|----------|------|----------|
| 训练管理 | (新) eve/dock/trainer.py | 新建 | 任务级 |
| TNN 构建 | (新) eve/dock/tiny_nn.py | 新建 | 任务级 |
| 训练订单 | (新) eve/dock/order.py | 新建 | 任务级 |

## 测试
| 覆盖范围 | 计划文件 |
|----------|----------|
| Safegate | tests/test_safegate.py |
| 运行时循环 | tests/test_runtime_loop.py |
| Input Buffer | tests/test_input_buffer.py |
| 捕获 Timing | tests/test_capture_timing.py |
| 架构禁止项 | tests/test_forbidden_architecture.py |
| 运行时状态 | tests/test_runtime_state.py |
| Memory | tests/test_memory.py |
| TNN Store | tests/test_tnn_store.py |
| TNNGraph | tests/test_graph.py |
| 激素 | tests/test_hormones.py |
| LLM Prompt | tests/test_prompts.py |
| Dock | tests/test_dock.py |
| 集成测试 | tests/test_integration.py |
