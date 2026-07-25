# EVE 测试与日志事实

## 当前行为测试

| 文件 | 保护的行为 |
|---|---|
| `tests/test_minimal_runtime.py` | SourceRef 上下游、ActionCandidate、consume-once、Memory、结构化 TNN 异常、入口启停 |
| `tests/test_input_minimal.py` | 单调 Buffer、范围窗口、捕获异常与停止 |
| `tests/test_memory_minimal.py` | CRUD、Event、STM/MTM ID、类型/关键词检索、重载 |
| `tests/test_safegate.py` | denied/simulated/executed 语义、急停、cold start、权限、过期、人类接管 |
| `tests/test_forbidden_architecture.py` | 已删除架构不再定义/导入、无静默核心异常 |

## 日志

当前运行写入 `<run-dir>/eve.jsonl`。核心错误至少包含：

```text
timestamp_ns
loop_node
exception_type
message
traceback
relevant_source
recovery_action
```

本轮 smoke 原始日志：`runs/smoke/eve.jsonl`。Memory Catalog：
`runs/smoke/memory/catalog.json`。

## 证据等级

- 测试与 CLI smoke：模拟验证。
- Core/Capture 线程启停、文件持久化：真实代码路径验证。
- 输入内容：合成 reader，不是本轮真实屏幕证据。
- 输出内容：Mock，不是系统鼠标动作。
- TNN：规则占位，不是成长证明。
