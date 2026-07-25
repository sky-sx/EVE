# EVE 当前状态

更新时间：2026-07-25 CST。

当前正式代码是一条可限时运行的最小纵向闭环：

```text
synthetic/mock capture → InputBuffer
→ loaded TNN descriptor SourceRef
→ named TNN output → ActionCandidate consume-once queue
→ Safegate → disabled/mock OutputResult
→ JSONL log + immutable Memory payload
```

## 已真实运行

- `eve.main` 启动并停止 capture 线程与 core 线程。
- `state:`、`world:`、`myself:`、`blackboard:`、`tnn:<id>.<field>` SourceRef 解析。
- 动作队列只消费一次；结果进入日志和三个 MemoryUnit。
- 停机验证线程退出并保存状态快照。
- Memory 支持创建、ID 读取、删除、类型/时间/关键词检索和最小 Event。

## Mock / 结构占位

- 正式 CLI 只接受 `disabled` 或 `mock`，不会开启真实输出。
- CLI smoke 使用 `SmokeActionNode`；它明确标记为
  `rule_placeholder_not_trained`，不是已成长出的 TNN。
- 鼠标、键盘、语音 real 后端仍在各自 output 文件中，但不由正式 CLI 开启；
  语音 real 后端明确返回未实现。

## 尚未实现 / 已暂停

- 真实训练所得 TNN artifact 加载实验。
- 真实屏幕/光标 smoke（捕获实现存在，本轮验证使用合成 reader）。
- 本地 LLM、VLM、YOLO 运行接入。
- Dock 训练、Sleep 复盘、复杂 Memory Graph 与 GUI。
- 键盘人类活动 hook。

## 本次验证

- 第一方 Python：19 文件、1,643 行（收缩前 38 文件、7,533 行）。
- 测试：5 文件、11 个行为测试；`11 passed in 0.34s`。
- smoke：`executed=false`、`simulated=true`、`memory_units=3`、
  `threads_stopped=true`。
