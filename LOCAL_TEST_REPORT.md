# EVE 本地测试报告

生成时间：2026-07-25

## 测试执行

```
命令: python -m pytest tests/ -v --tb=short
收集: 173 个测试
结果: 173 passed, 0 failed, 0 skipped
时间: ~15s
```

## 测试统计

| 测试文件 | 测试数 | 状态 | 覆盖范围 |
|----------|--------|------|----------|
| test_safegate.py | 23 | PASSED | 9 条安全规则全覆盖 |
| test_runtime_loop.py | 15 | PASSED | run_once/EVELoops/日志 |
| test_input_buffer.py | 16 | PASSED | 存储/范围/快照/淘汰/线程安全 |
| test_capture_timing.py | 14 | PASSED | 捕获启停/统计/时间戳 |
| test_forbidden_architecture.py | 10 | PASSED | 架构禁止词/字段最小性 |
| test_runtime_state.py | 18 | PASSED | Blackboard/快照/LLM 更新 |
| test_memory.py | 16 | PASSED | Memorizer/Catalog/索引/Event/检索 |
| test_tnn_store.py | 11 | PASSED | TNN 保存加载/Store/版本管理 |
| test_graph.py | 11 | PASSED | 缓存/节点/边/调度/轨迹 |
| test_hormones.py | 12 | PASSED | 6 激素更新/节律/倾向 |
| test_prompts.py | 12 | PASSED | Prompt 结构/校验/JSON 解析 |
| test_dock.py | 8 | PASSED | 训练订单/队列/训练流程 |
| test_integration.py | 7 | PASSED | 端到端集成链路 |

## 失败原因

无失败。

## 跳过原因

无跳过（所有 173 个测试在本地环境直接通过）。

## 未实测部分

以下模块因缺少对应硬件/模型/依赖，测试覆盖使用模拟数据，但代码逻辑已完整实现：

| 模块 | 未实测原因 | 测试覆盖 |
|------|-----------|----------|
| LLMAdapter.infer() | 本地无 transformers 模型文件 | 仅测试结构，未实际加载推理 |
| VLMAdapter.infer() | 本地无视觉语言模型 | 仅测试 detect/load 逻辑 |
| YOLOAdapter.detect_objects() | 本地 yolo26 模型未验证加载 | 仅测试适配器结构 |
| ControlPanel (GUI) | tkinter 需要 display | 未包含在自动化测试中 |
| CaptureManager 真实捕获 | 可能无 display server | 有关测试在 display 可用时通过 |
| output REAL 模式 | 需人工审核后才开启 | 接口实现完成，测试覆盖 disabled/mock 模式 |

## 运行环境

- OS: Windows 11
- Python: 3.11.7
- PyTorch: 2.10.0+cu130
- GPU: RTX 5080 16GB
- CUDA: 13.1
