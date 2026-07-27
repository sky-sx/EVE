# EVE 第一轮运行时整合报告

日期：2026-07-27 CST

## 1. 修改文件

| 文件 | 本轮解决的问题 |
|---|---|
| `eve/main.py` | 增加 `smoke` / `observe` Profile，明确拒绝 `control`；统一启动、全局 Esc Watch、错误传播、关闭和运行摘要 |
| `eve/input/capture.py` | 屏幕与光标独立计时采集；真实 Windows 桌面接入；frame ID、单调时间戳、光标速度和采集统计 |
| `eve/input/buffer.py` | 有容量的最近一秒窗口；`get_state()`、`get_latest_screen()`、`get_latest_cursor()` 和关闭语义 |
| `eve/state.py` | 补全运行状态、active/loaded TNN 区别、Blackboard 五类最新结果、资源状态和运行统计 |
| `eve/core/tnn.py` | 保留现有节点关系；记录加载状态、调用统计和动作观察时间；规则节点路径延迟导入 PyTorch |
| `eve/core/loop.py` | 接通 Input 摘要、节点、Blackboard、Safegate、反馈；Memory 改为非阻塞入队；关键错误停止生命周期 |
| `eve/memory/memorizer.py` | 有界异步 Writer、优先级溢出、flush、增量 Catalog、JSON/NPY 可恢复 Payload 和统计 |
| `pyproject.toml` | 声明真实 observe 与已有 TNN 路径所需的 `mss`、`numpy`、`torch` |
| `tests/test_input_minimal.py` | 强化 Capture 初始化失败可见性 |
| `tests/test_minimal_runtime.py` | 验证异步 Memory flush 以及动作/Safegate/反馈记录 |
| `tests/test_runtime_integration.py` | 新增 Profile、Buffer、Memory、Blackboard、Esc、错误传播和线程生命周期测试 |
| `STATUS.md` | 只记录本轮实际运行和验证过的能力 |

没有增加 Scheduler、Router、Event Bus、Actor Framework 或其他新的运行时模块。

## 2. 最终运行数据流

```text
真实 screen / cursor（observe）或合成 reader（smoke）
→ 两个独立 Capture 线程
→ 有界一秒 InputBuffer
→ Core Runtime
→ SmokeActionNode 或指定 TNN artifact
→ Blackboard latest_tnn_output / latest_action_candidate
→ Safegate
→ Mock 或明确阻断 OutputResult
→ Blackboard latest_safegate_result / latest_output_feedback
→ 有界 MemoryWriteRequest 队列
→ Memory Writer
→ JSON / NPY Payload + append-only catalog.jsonl
```

正式 Profile 的 `OutputMode` 固定为 Mock，权限默认关闭；真实鼠标、键盘和语音后端不会被调用。

## 3. 自动化测试

改造前基线：

```text
16 passed in 2.54s
```

改造后：

```text
python -m pytest -q
.......................
23 passed in 2.79s
```

新增验证包括：

- Profile 选择和 `control` 非零拒绝；
- Buffer 一秒淘汰、容量、并发读取和关闭；
- 异步入队、flush、NPY 恢复和增量 Catalog；
- 队列满时优先丢弃低优先级请求；
- Blackboard 五类结果；
- 全局 Esc Watch 的急停和三秒内退出；
- Core 节点异常传播；
- Memory Writer 无法写入时形成关键运行失败；
- 退出后不保留 `eve-*` 项目线程。

## 4. Smoke 集成结果

实际命令：

```text
python -m eve.main --profile smoke --duration 3
```

实际结果：

| 指标 | 值 |
|---|---:|
| 运行时长 | 3.000 s |
| 屏幕合成采集 | 30.00 FPS |
| 光标合成采集 | 60.00 Hz |
| Core 循环 | 32.33 Hz |
| TNN 调用 | 49 |
| Safegate allow / block | 0 / 1 |
| Mock backend 调用 | 0 |
| 真实 Output 调用 | 0 |
| Memory written / dropped / failed | 6 / 0 / 0 |
| 关键错误 | 无 |
| 退出原因 | duration elapsed |
| 线程清理 | 完成 |

权限默认关闭，因此动作候选进入 Safegate 后产生明确阻断反馈；没有为了制造 allow 数据而放宽正式入口权限。

## 5. Observe 实机结果

### 5.1 十秒预检

```text
python -m eve.main --profile observe --duration 10
```

结果：

- 屏幕 28.76 FPS；
- 光标 59.32 Hz；
- 屏幕平均采集延迟 18.79 ms；
- 真实 Output 0；
- Memory written 12 / dropped 0 / failed 0；
- 正常退出且线程全部清理。

### 5.2 三十分钟实机运行

实际运行：

```text
python -m eve.main --profile observe --duration 1800
```

结果文件位于 `runs/integration_observe_30m/`。

| 指标 | 实测值 |
|---|---:|
| 运行时长 | 1800.032 s |
| 屏幕平均 FPS | 28.8206 |
| 光标平均 Hz | 59.6856 |
| 屏幕平均采集延迟 | 19.0264 ms |
| 光标平均采集延迟 | 0.0061 ms |
| Core 循环 | 32.335 Hz |
| TNN 调用 | 28,206 |
| Safegate allow / block | 0 / 1 |
| Mock backend 调用 | 0 |
| 真实 Output 调用 | 0 |
| Memory written / dropped / failed | 1,795 / 0 / 0 |
| Catalog 行数 | 1,795 |
| runtime error | 0 |
| shutdown 记录 | 1 |
| 后台进程残留 | 无 |
| 线程清理 | 完成 |

屏幕频率超过任务书人工验收的 25 FPS 下限。光标配置目标为 60 Hz，实际平均 59.69 Hz，属于 Windows 调度波动。

## 6. 资源观察

30 分钟运行暖机后的 Windows 工作集采样稳定在约 744—755 MB，私有内存约 1.86—1.87 GB，没有随运行时间线性增长。该次长跑仍包含规则入口对 PyTorch 的提前导入；长跑之后已将规则节点路径调整为仅在加载训练 TNN 时才导入 PyTorch。该调整另行通过了完整 23 项回归和真实 observe 短测。

屏幕 Buffer 保存最近一秒的真实帧引用，内存基线受桌面分辨率影响；它有时间窗口和 256 条每类硬容量，不会无界保留历史帧。

## 7. Safegate 和 Output

- 正式 `smoke` / `observe` 权限均默认关闭；
- emergency stop、权限、接管冻结和过期动作检查保留；
- Blackboard 保存最新动作、Safegate 判定和反馈；
- 30 分钟实测 Safegate block 1 次；
- Mock backend 调用 0 次，因为候选在权限检查处即形成阻断反馈；
- 真实 mouse / keyboard / speak 调用均为 0。

允许路径仍由原有 Safegate 单元测试在 Mock 模式临时授权验证，没有开启真实 Output。

## 8. Memory 写入与溢出

- Core 热路径只调用 `enqueue()`；
- Writer 使用独立 `eve-memory-writer` 线程；
- 队列默认容量 256；
- 队列满时优先移除最旧低优先级输入快照；
- 无低优先级项时，关键请求拒绝会触发错误回调并停止运行，不会静默丢弃；
- 低频输入摘要最高约 1 Hz；
- NumPy/Tensor 顶层 Payload 保存为 `.npy`，读取时可恢复；
- JSON 编码不再使用 `repr()` 兜底；
- Catalog 使用 append-only `catalog.jsonl`，不会每个 MemoryUnit 全量重写。

30 分钟实测没有出现队列丢弃或写入失败。

## 9. 停止和错误传播

- Windows Esc Watch 使用 `GetAsyncKeyState(VK_ESCAPE)`，不依赖控制台焦点；
- 自动化测试模拟全局 Esc，验证设置 emergency stop 并在三秒内清理所有线程；
- 30 分钟人工运行使用 `--duration` 自然退出，没有人为按下实体 Esc；
- Capture 初始化错误会中止启动；
- 关键节点错误会使 Core 失败并通知主线程；
- Memory Writer 错误会使主生命周期失败；
- `control` 明确输出 `control profile is not enabled in this integration stage` 并返回 2；
- 常规 smoke/observe 直接运行返回 0。

## 10. 尚未完成

本轮按任务书明确没有实现：

- 真实鼠标、键盘或语音控制；
- 麦克风、键盘状态和活动窗口采集；
- 本地/云端 LLM；
- 完整持久 TNN 能力关系；
- 自动 TNN 替换、影子部署或 Dock 自动训练；
- 完整多模态关联、Event 持久化、睡眠复盘和图压缩；
- GUI 和长期资源策略。

实体键盘 Esc 尚未由 Codex 代替用户真实按下；全局 Windows API 路径已通过可重复自动化测试。

## 11. 下一阶段建议

下一步只建议实现 active TNN 目标集合与 loaded TNN 实际集合之间的最小差分生命周期，并继续保持真实 Output 关闭；不要同时引入 LLM 大循环或完整持久图。
