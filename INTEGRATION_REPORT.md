# EVE 文件结构与数据关系收拢报告

日期：2026-07-27 CST

## 1. 收拢结果

本轮只调整结构、控制关系和数据流，没有新增 EVE 能力。

删除：

- `eve/state.py`
- `eve/core/tnn.py`
- `eve/config.py`（未使用且不属于最终正式结构）
- Memory 中未使用的 Event 与 `related_ids` 预留接口

职责合并：

- `eve/state.py` 的必要运行状态并入 `eve/core/loop.py`；
- `eve/core/tnn.py` 的 artifact 加载、卸载、调度、输入解析、推理、输出、动作候选与异常处理并入 `eve/core/loop.py`；
- 未复制 SourceRef、Descriptor、Protocol、Smoke 正式类和重复 load/attach 层；
- Main 原有的 Capture 生命周期和 TNN 装配职责分别下沉到 Buffer 与 Core。

## 2. 最终数据流

```text
main.py
  ↓
InputBuffer
  ↔ capture.py 独立进程
  ↓
core/loop.py
  ↔ memorizer.py
  ↓
safegate.py
  ↓
output
  ↓ feedback
core/loop.py
```

不存在 Main/Core/Safegate/TNN/Output 越过 Buffer 读取或控制 Capture 的路径，Main 也不装配 TNN 内部运行节点。

## 3. Capture 进程结构

`InputBuffer.start_capture()` 使用 `multiprocessing.get_context("spawn")` 创建 `eve-capture` 子进程。

子进程内部使用屏幕和光标两个线程。屏幕帧写入共享内存 Ring Buffer，Pipe 只发送 frame ID、时间戳、slot、shape、dtype；光标、health、error 和 stop 控制也走 Pipe。任一采集线程异常都会发送结构化 error 并停止子进程。

父进程 Buffer 附加共享内存、维护最近一秒窗口、形成 human activity/takeover 状态并监控进程。停止时 join Capture、关闭 Pipe、释放并 unlink 共享内存。

Main 的关闭顺序为 Core Loop → Memory Writer → InputBuffer/Capture。

## 4. Core、TNN 与状态

`core/loop.py` 用简单结构保存：

```text
world / myself / blackboard
active_tnn / loaded_tnn / tnn_status
loop_status / permissions / resource_status
emergency_stop / latest_error
```

`active_tnn` 与 `loaded_tnn` 没有合并。具体 TNN 仍继承 `dock/tinynn.py` 的 `TinyNN`；Core 从 Memorizer 解析 artifact，导入具体 `model.py`、加载权重、调用 `infer()` 并负责卸载。

Safegate 仅做权限、急停、冷启动、接管截止时间、有效期、动作类型与最低范围检查。人类活动判断在 Buffer 完成。

Memory 仍异步写入并增量追加 Catalog；数组使用 NPY 而非 repr；支持 ID、类型、关键词和时间条件的最小检索；TNN artifact 仍在同一文件保存与读取。

## 5. 测试与运行证据

收拢前基线：

```text
23 passed
```

收拢后最终回归：

```text
python -m pytest -q
..........................
26 passed in 4.12s
```

验收覆盖文件集合和导入边界、Capture 独立 PID、Buffer 生命周期/健康/窗口、Capture 异常到 Main 非零退出、active/loaded 分离、TinyNN artifact 重载、Mock-only observe、Esc、Core/Memory 错误和资源清理。

Smoke：

| 指标 | 结果 |
|---|---:|
| 退出码 / 时长 | 0 / 3.187 s |
| 屏幕 / 光标 | 29.58 FPS / 60.13 Hz |
| Core | 32.17 Hz |
| Safegate allow / block | 0 / 1 |
| 真实 Output | 0 |
| Memory written / dropped / failed | 5 / 0 / 0 |
| Capture/线程残留 | 无 |

十分钟真实 observe：

| 指标 | 结果 |
|---|---:|
| 退出码 / 时长 | 0 / 600.313 s |
| 屏幕平均 FPS | 28.7817 |
| 光标平均 Hz | 59.7612 |
| 屏幕平均采集延迟 | 20.0439 ms |
| 光标平均采集延迟 | 0.0017 ms |
| Core | 32.2633 Hz |
| TNN 调用 | 9,446 |
| Safegate allow / block | 0 / 1 |
| Mock / 真实 Output | 0 / 0 |
| Memory written / dropped / failed | 600 / 0 / 0 |
| runtime error / shutdown | 0 / 1 |
| EVE/Capture/resource tracker 残留 | 无 |

产物位于 `runs/structure_observe_10m_final/`。

此前 30 分钟数据来自收拢前的线程式 Capture 版本，不作为本轮独立进程架构的验收证据。

## 6. 未实现内容

- real Output 与真实受控闭环；
- 音频、键盘状态和活动窗口采集；
- 本地/云端 LLM；
- 六大激素；
- 复杂 Memory Graph、睡眠整理、图压缩；
- 自动训练、自动替换和影子部署；
- GUI 与长期资源管理。

物理 Esc 没有由 Codex 代替用户按下；Windows 全局 Esc 检测与清理路径由自动化测试覆盖。
